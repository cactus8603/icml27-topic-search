import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import numpy as np
import torch
from torch.nn import functional as F

from data import prepare, sha256, synthetic, write_json
from run import (
    amp, branch_rate, check_deadline, evaluate, finite, initial_models, microstep,
    optimizers, probe_after, probe_before, smoke_config, windows,
)


@torch.no_grad()
def rescue_probe(model, saved, row, device):
    """All counterfactuals change only one W_O, with selection on signal tokens."""
    _, _, writes, _, outcome, old_weights, _ = saved
    largest = 0.0
    selected = None
    for index, (block, old, (_, z, _, _)) in enumerate(zip(model.blocks, old_weights, writes, strict=True)):
        delta = block.attention.out.weight.detach() - old
        directions = F.normalize(z, dim=-1, eps=1e-8)
        values = (directions @ delta.T).norm(dim=-1)
        strength, j = values.max(dim=0)
        if strength.item() > largest:
            largest = strength.item()
            selected = index, delta, directions[j]
    if selected is None:
        return {"status": "degenerate_signal"}
    index, delta, unit = selected
    control_unit = unit.roll(1)
    control_unit = control_unit - torch.dot(control_unit, unit) * unit
    control_unit = F.normalize(control_unit, dim=0, eps=1e-8)
    direction_change = -0.5 * torch.outer(delta @ unit, unit)
    control_change = -torch.outer(delta @ control_unit, control_unit)
    if control_change.norm().item() < 1e-12:
        return {"status": "degenerate_control"}
    directional = delta + direction_change
    uniform = delta * (directional.norm() / delta.norm())
    rolled = delta + control_change * (direction_change.norm() / control_change.norm())
    matrix = model.blocks[index].attention.out.weight
    after = matrix.detach().clone()
    rescue = {
        "status": "measured", "layer": index,
        "uniform_delta_norm_relative_error": finite(((uniform.norm() - directional.norm()).abs()
                                                     / directional.norm().clamp_min(1e-12)).item()),
        "control_change_norm_relative_error": finite((((rolled - delta).norm() - direction_change.norm()).abs()
                                                      / direction_change.norm().clamp_min(1e-12)).item()),
        "rollback_to_before_cap": row["delta_nll"],
    }
    try:
        for label, candidate in (("direction", directional), ("uniform", uniform), ("control", rolled),
                                 ("wo_rollback", torch.zeros_like(delta))):
            matrix.copy_(old_weights[index] + candidate)
            with amp(device):
                rescue["nll_" + label] = finite(model(*outcome).item())
    finally:
        matrix.copy_(after)
    return rescue


def cell(model_config, t, arm, rate, seed, grid, threshold, streams, output, device, deadline, smoke):
    name = arm["name"]
    model = initial_models(model_config, [arm], seed, device)[name]
    opts = optimizers(model, arm, t)
    length = t["shared_steps"] + t["branch_steps"]
    order = streams["train"].order(length, seed)
    eval_order = streams["dev"].order(t["eval_batches"], 541103)
    scheduled = list(range(t["probe_every"], t["branch_steps"] + 1, t["probe_every"]))
    probe_rows = streams["probe"].order(len(scheduled), seed + 49)
    probe_at = dict(zip(scheduled, probe_rows, strict=True))
    warmup = min(t["tune_warmup"], t["shared_steps"])
    prefix_steps = order[:t["shared_steps"]]
    for step, batch in enumerate(prefix_steps, 1):
        check_deadline(deadline)
        lr = min(1.0, step / warmup) * rate
        loss, grad = microstep(model, opts, streams["train"], batch, lr, t, device)
        if loss is None or grad is None:
            write_json(output / f"{name}-{seed}-prefix_failure.json", {"step": step, "train_nll": loss, "grad_norm": grad})
            return
    starting_nll = evaluate(model, streams["dev"], eval_order, device)
    if starting_nll is None:
        write_json(output / f"{name}-{seed}-prefix_failure.json", {"step": t["shared_steps"], "dev_nll": None})
        return
    model_state = copy.deepcopy(model.state_dict())
    optimizer_states = [copy.deepcopy(opt.state_dict()) for opt, _ in opts]

    for multiplier in (1, *grid):
        check_deadline(deadline)
        model.load_state_dict(model_state)
        for (opt, _), state in zip(opts, optimizer_states, strict=True):
            opt.load_state_dict(copy.deepcopy(state))
        started = time.monotonic()
        restored_nll = evaluate(model, streams["dev"], eval_order, device)
        if restored_nll != starting_nll:
            raise RuntimeError(f"restored checkpoint disagrees with prefix: {name}, seed {seed}, {multiplier}x")
        evaluations = [{"step": 0, "dev_nll": restored_nll}]
        trace, probes = [], []
        for step, batch in enumerate(order[t["shared_steps"]:], 1):
            check_deadline(deadline)
            saved = None
            if multiplier != 1 and step in probe_at:
                signal, outcome = np.split(probe_at[step], 2)
                saved = probe_before(model, streams["probe"], signal, outcome, step, t, device, False)
            lr = branch_rate(step - 1, t["branch_steps"], t["branch_warmup"] if multiplier != 1 else 0,
                             rate, multiplier, start=1.0, decay=multiplier == 1)
            loss, grad = microstep(model, opts, streams["train"], batch, lr, t, device)
            trace.append({"step": step, "train_nll": loss, "grad_norm": grad, "lr": lr})
            if loss is None or grad is None:
                break
            if saved is not None:
                row = probe_after(model, saved, device)
                damage = row["delta_nll"]
                if damage is not None and (damage >= threshold or smoke):
                    row["rescue"] = rescue_probe(model, saved, row, device)
                probes.append({"step": step, "grad_norm": grad, **row})
            if step % t["eval_every"] == 0 or step == t["branch_steps"]:
                nll = evaluate(model, streams["dev"], eval_order, device)
                evaluations.append({"step": step, "dev_nll": nll})
                if nll is None:
                    break
        write_json(output / f"{name}-{seed}-{multiplier}x.json", {
            "arm": name, "seed": seed, "multiplier": multiplier, "trace": trace,
            "evaluations": evaluations, "probes": probes, "branch_wall_seconds": time.monotonic() - started,
        })
        print(f"[codex] calibration {name}/{seed}/{multiplier}x completed {len(trace)} steps", flush=True)


def configuration(base, calibration, smoke):
    if (calibration["stress_multipliers"] != [3, 4, 6]
            or len(set(calibration["calibration_seeds"])) != len(calibration["calibration_seeds"])):
        raise ValueError("invalid calibration grid or duplicate seeds")
    occupied = {base["training"]["dev_seed"], *base["training"]["test_seeds"], *calibration["calibration_seeds"]}
    reserved = calibration["separate_validation_plan"]["seeds"]
    if (len(occupied) != 1 + len(base["training"]["test_seeds"]) + len(calibration["calibration_seeds"])
            or len(set(reserved)) != len(reserved) or occupied.intersection(reserved)):
        raise ValueError("round-one, calibration and validation seeds must be disjoint")
    if smoke:
        base = smoke_config(base)
    t = base["training"]
    t["probe_every"] = 1 if smoke else calibration["probe_every"]
    t["arms"] = [arm for arm in t["arms"] if arm["name"] in calibration["base_rates"]]
    return base


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-config", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--seconds", type=int, default=6500)
    p.add_argument("--frozen-commit")
    p.add_argument("--frozen-tree")
    p.add_argument("--prereg-sha256")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    deadline = time.monotonic() + args.seconds
    config = json.loads(args.config.read_text())
    if sha256(args.base_config) != config["base_config_sha256"]:
        raise ValueError("base settings differ from the frozen round-one settings")
    base = configuration(json.loads(args.base_config.read_text()), config, args.smoke)
    if args.smoke:
        torch.set_num_threads(1)
        device = torch.device("cpu")
        synthetic(base, args.corpus)
    else:
        if not all((args.frozen_commit, args.frozen_tree, args.prereg_sha256)):
            p.error("calibration requires frozen commit, tree and PREREG hash")
        if sha256(Path(__file__).with_name("PREREG.md")) != args.prereg_sha256:
            raise ValueError("PREREG hash mismatch")
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA with BF16 required")
        device = torch.device("cuda")
        prepare(base, args.corpus)
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise FileExistsError(f"refusing to overwrite {args.output}")
    streams = windows(args.corpus, base["model"], base["training"])
    write_json(args.output / "run.json", {
        "config": config, "training": base["training"], "model": base["model"], "smoke": args.smoke,
        "torch": torch.__version__, "frozen_commit": args.frozen_commit, "frozen_tree": args.frozen_tree,
        "config_sha256": sha256(args.config), "prereg_sha256": args.prereg_sha256,
        "corpus_sha256": sha256(args.corpus / "corpus.json") if not args.smoke else None,
        "slurm_id": os.getenv("SLURM_JOB_ID"), "device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
    })
    for seed in config["calibration_seeds"]:
        for arm in base["training"]["arms"]:
            cell(base["model"], base["training"], arm, config["base_rates"][arm["name"]], seed,
                 config["stress_multipliers"], config["criteria"]["damage_nll"], streams, args.output,
                 device, deadline, args.smoke)


if __name__ == "__main__":
    main()
