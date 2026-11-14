import argparse
import contextlib
import copy
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from data import Windows, prepare, sha256, synthetic, write_json
from model import GQA


class StableAdamW(torch.optim.AdamW):
    @torch.no_grad()
    def step(self):
        # Compute the upcoming second moment before AdamW commits the same update.
        # One parameter per group makes the LR adjustment tensor-local.
        old_rates = [g["lr"] for g in self.param_groups]
        active = []
        try:
            for group in self.param_groups:
                (p,) = group["params"]
                if p.grad is None:
                    continue
                beta2 = group["betas"][1]
                state = self.state[p]
                g2 = p.grad.square()
                v = g2 * (1 - beta2)
                if "exp_avg_sq" in state:
                    v.add_(state["exp_avg_sq"], alpha=beta2)
                rms = (g2 / v.clamp_min(group["eps"] ** 2)).mean().sqrt()
                active.append((group, rms))
            if active:
                factors = torch.stack([rms for _, rms in active]).clamp_min(1).tolist()
                for (group, _), factor in zip(active, factors, strict=True):
                    group["lr"] /= factor
            return super().step()
        finally:
            for group, lr in zip(self.param_groups, old_rates, strict=True):
                group["lr"] = lr


def optimizers(model, arm, t):
    opts = []
    betas = tuple(t["adam_betas"])
    matrix = []
    adam = []
    for name, param in model.named_parameters():
        if arm["optimizer"] == "muon" and "blocks." in name and ".gate." not in name and param.ndim == 2:
            matrix.append(param)
        else:
            adam.append(param)
    if matrix:
        opts.append((torch.optim.Muon(matrix, lr=1.0, weight_decay=t["weight_decay"],
                                      adjust_lr_fn="match_rms_adamw"), 1.0))
    if arm["optimizer"] == "stableadam":
        groups = [{"params": [param]} for param in adam]
        opts.append((StableAdamW(groups, lr=1.0, betas=betas, weight_decay=t["weight_decay"],
                                 fused=True), 1.0))
    else:
        adam_scale = 0.25 if matrix else 1.0
        opts.append((torch.optim.AdamW(adam, lr=1.0, betas=betas, weight_decay=t["weight_decay"],
                                       fused=True), adam_scale))
    return opts


def initial_models(m, arms, seed, device):
    torch.manual_seed(seed)
    reference = GQA(m, {}).state_dict()
    models = {}
    for arm in arms:
        model = GQA(m, arm)
        missing, unexpected = model.load_state_dict(reference, strict=False)
        if unexpected or any(not name.endswith("attention.gate.weight") for name in missing):
            raise ValueError("paired model initialization mismatch")
        models[arm["name"]] = model.to(device)
    return models


def amp(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else contextlib.nullcontext()


def synch(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def finite(value):
    value = float(value)
    return value if math.isfinite(value) else None


def microstep(model, opts, window, rows, lr, t, device):
    model.zero_grad(set_to_none=True)
    losses = []
    for part in np.split(rows, t["accumulation"]):
        tokens, targets = window.batch(part, device)
        with amp(device):
            loss = model(tokens, targets)
        (loss / t["accumulation"]).backward()
        losses.append(loss.detach().float())
    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
    for opt, scale in opts:
        for group in opt.param_groups:
            group["lr"] = lr * scale
        opt.step()
    return finite(torch.stack(losses).mean().item()), finite(grad.item())


@torch.no_grad()
def evaluate(model, window, rows, device):
    losses = []
    for row in rows:
        with amp(device):
            losses.append(model(*window.batch(row, device)).float())
    return finite(torch.stack(losses).mean().item())


def branch_rate(step, total, warmup, nominal, multiple=1.0, start=0.0, decay=True):
    if step < warmup:
        return nominal * (start + (multiple - start) * (step + 1) / warmup)
    if not decay:
        return nominal * multiple
    fraction = (step - warmup) / max(1, total - warmup - 1)
    return nominal * (0.1 + 0.9 * (1 + math.cos(math.pi * fraction)) / 2)


@torch.no_grad()
def measure_writes(model, states, ranks):
    cos = model.cos[:, :, :states[0].size(1)]
    sin = model.sin[:, :, :states[0].size(1)]
    writes = []
    for block, x in zip(model.blocks, states, strict=True):
        out, z, qk = block.write(x, cos, sin, probe=True)
        x_scale = x.float().flatten(0, 1)[ranks].norm(dim=-1).clamp_min(1e-8)
        writes.append((out.float().flatten(0, 1)[ranks], z.float().flatten(0, 1)[ranks], x_scale, qk))
    return writes


@torch.no_grad()
def probe_before(model, probe, signal_row, outcome_row, step, t, device, intervene):
    signal = probe.batch(signal_row, device)
    outcome = probe.batch(outcome_row, device)
    n = signal[0].numel()
    indices = np.random.default_rng(step + 67).choice(n, min(n, t["probe_positions"]), replace=False)
    positions = torch.tensor(indices, device=device)
    with amp(device):
        _, states = model(*signal, capture=True)
        writes = measure_writes(model, states, positions)
        before = model(*outcome).item()
    weights = [block.attention.out.weight.detach().clone() for block in model.blocks]
    return states, positions, writes, before, outcome, weights, intervene


@torch.no_grad()
def measure_intervention(model, weights, writes, outcome, device):
    # Single-layer single-direction diagnostic. Every variant starts from the
    # same completed update and uses identical outcome tokens.
    best = (-1.0, None, None, None)
    for index, (block, old, (_, z, _, _)) in enumerate(zip(model.blocks, weights, writes, strict=True)):
        delta = block.attention.out.weight - old
        unit = F.normalize(z, dim=-1, eps=1e-8)
        responses = (unit @ delta.T).norm(dim=-1)
        largest, row = responses.max(dim=0)
        if largest.item() > best[0]:
            best = (largest.item(), index, unit[row], delta)
    response, index, unit, delta = best
    if not math.isfinite(response) or response <= 0:
        return {}
    matrix = model.blocks[index].attention.out.weight
    updated = matrix.detach().clone()
    directional = delta - 0.5 * torch.outer(delta @ unit, unit)
    uniform = delta * (directional.norm() / delta.norm().clamp_min(1e-12))
    control_unit = F.normalize(torch.roll(unit, 1), dim=0)
    control = delta - 0.5 * torch.outer(delta @ control_unit, control_unit)
    # Match the size of the changed portion for the unrelated-direction control.
    control_change = control - delta
    control = delta + control_change * ((directional - delta).norm() / control_change.norm().clamp_min(1e-12))
    result = {"intervention_layer": index}
    try:
        for label, candidate in (("direction", directional), ("uniform", uniform), ("control", control)):
            matrix.copy_(weights[index] + candidate)
            with amp(device):
                result[f"nll_{label}"] = finite(model(*outcome).item())
    finally:
        matrix.copy_(updated)
    return result


@torch.no_grad()
def probe_after(model, saved, device):
    states, indices, old, before, outcome, weights, intervene = saved
    with amp(device):
        new = measure_writes(model, states, indices)
        after = model(*outcome).item()
    changes = torch.stack([(y - y0).norm(dim=-1) / scale
                           for (y0, _, scale, _), (y, _, _, _) in zip(old, new, strict=True)])
    row = {
        "nll_before": finite(before),
        "nll_after": finite(after),
        "delta_nll": finite(after - before),
        "write_q95": finite(torch.quantile(changes.flatten(), 0.95).item()),
        "write_max": finite(changes.max().item()),
        "write_rms": finite(changes.square().mean().sqrt().item()),
        "activation_max": finite(torch.stack([y.abs().amax() for y, _, _, _ in old]).max().item()),
        "qk_max_last_query": finite(torch.stack([qk for _, _, _, qk in old]).max().item()),
        "out_update_rms": finite(torch.stack([
            (block.attention.out.weight - weight).square().mean().sqrt()
            for block, weight in zip(model.blocks, weights, strict=True)
        ]).max().item()),
    }
    if intervene and row["nll_after"] is not None:
        row.update(measure_intervention(model, weights, old, outcome, device))
    return row


def windows(folder, m, t):
    widths = {"train": t["micro_batch"] * t["accumulation"], "dev": t["micro_batch"],
              "test": t["micro_batch"], "probe": t["probe_batch"] * 2}
    return {key: Windows(folder / f"{key}.bin", m["sequence"], width) for key, width in widths.items()}


def tune(m, t, streams, device, deadline):
    leaderboard = {}
    seed = t["dev_seed"]
    train_order = streams["train"].order(t["tune_steps"], seed)
    dev_order = streams["dev"].order(t["eval_batches"], seed + 1)
    for arm in t["arms"]:
        candidates = []
        for rate in arm["rate_grid"]:
            check_deadline(deadline)
            model = initial_models(m, [arm], seed, device)[arm["name"]]
            opts = optimizers(model, arm, t)
            losses = []
            for step, rows in enumerate(train_order):
                check_deadline(deadline)
                rate_now = branch_rate(step, t["tune_steps"], t["tune_warmup"], rate)
                loss, _ = microstep(model, opts, streams["train"], rows, rate_now, t, device)
                if loss is None:
                    break
                losses.append(loss)
            dev_nll = evaluate(model, streams["dev"], dev_order, device) if len(losses) == len(train_order) else None
            candidates.append({"rate": rate, "dev_nll": dev_nll})
        leaderboard[arm["name"]] = candidates
    winners = {}
    for name, candidates in leaderboard.items():
        valid = [candidate for candidate in candidates if candidate["dev_nll"] is not None]
        if not valid:
            raise RuntimeError(f"no finite nominal dev run for {name}")
        winners[name] = min(valid, key=lambda c: (c["dev_nll"], c["rate"]))["rate"]
    return leaderboard, winners


def check_deadline(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("compute cap reached; completed arms are preserved")


def run_arm(m, t, streams, arm, rate, seed, device, output, deadline):
    name = arm["name"]
    model = initial_models(m, [arm], seed, device)[name]
    opts = optimizers(model, arm, t)
    total = t["shared_steps"] + t["branch_steps"]
    train_order = streams["train"].order(total, seed)
    test_order = streams["test"].order(t["eval_batches"], 40002)
    probe_order = streams["probe"].order(t["branch_steps"], seed + 49)
    warmup = min(t["tune_warmup"], t["shared_steps"])
    for step, rows in enumerate(train_order[:t["shared_steps"]]):
        share_lr = min(1.0, (step + 1) / warmup) * rate
        loss, gradient = microstep(model, opts, streams["train"], rows, share_lr, t, device)
        if loss is None or gradient is None:
            write_json(output / f"{name}-{seed}-prefix-diverged.json", {"step": step + 1, "loss": loss})
            return
    shared_model = copy.deepcopy(model.state_dict())
    shared_optimizer = [copy.deepcopy(opt.state_dict()) for opt, _ in opts]
    shared_test_nll = evaluate(model, streams["test"], test_order, device)
    if shared_test_nll is None:
        write_json(output / f"{name}-{seed}-prefix-diverged.json", {"step": t["shared_steps"], "loss": None})
        return
    for schedule in ("nominal", "stress"):
        branch_started = time.perf_counter()
        model.load_state_dict(shared_model)
        for (opt, _), checkpoint in zip(opts, shared_optimizer, strict=True):
            opt.load_state_dict(copy.deepcopy(checkpoint))
        evaluation = [{"step": 0, "test_nll": shared_test_nll, "train_seconds": 0.0}]
        tracelog = []
        probes = []
        elapsed = 0.0
        for step, rows in enumerate(train_order[t["shared_steps"]:], 1):
            check_deadline(deadline)
            stress = schedule == "stress"
            is_probe = step % t["probe_every"] == 0
            if is_probe:
                signal, outcome = np.split(probe_order[step - 1], 2)
                saved = probe_before(model, streams["probe"], signal, outcome, step, t, device,
                                     stress and not arm.get("gate") and step % t["intervention_every"] == 0)
            lr = branch_rate(step - 1, t["branch_steps"], t["branch_warmup"] if stress else 0,
                             rate, t["stress_multiple"] if stress else 1.0, start=1.0, decay=not stress)
            synch(device)
            started = time.perf_counter()
            loss, grad = microstep(model, opts, streams["train"], rows, lr, t, device)
            synch(device)
            elapsed += time.perf_counter() - started
            tracelog.append({"step": step, "train_nll": loss, "grad_norm": grad, "lr": lr})
            if loss is None or grad is None:
                break
            if is_probe:
                row = probe_after(model, saved, device)
                probes.append({"step": step, "grad_norm": grad, **row})
            if step % t["eval_every"] == 0 or step == t["branch_steps"]:
                val = evaluate(model, streams["test"], test_order, device)
                evaluation.append({"step": step, "test_nll": val, "train_seconds": elapsed})
                if val is None:
                    break
        write_json(output / f"{name}-{seed}-{schedule}.json", {
            "arm": name, "seed": seed, "schedule": schedule, "selected_rate": rate,
            "trace": tracelog, "evaluation": evaluation, "probes": probes,
            "branch_wall_seconds": time.perf_counter() - branch_started,
        })
        print(f"[codex] completed {name}/{seed}/{schedule} ({len(tracelog)} steps)", flush=True)


def smoke_config(cfg):
    cfg = copy.deepcopy(cfg)
    cfg["model"].update(vocab=80, width=32, depth=2, heads=4, kv_heads=2, mlp=48, sequence=16)
    cfg["training"].update(micro_batch=2, accumulation=2, tune_steps=3, tune_warmup=1,
                           shared_steps=2, branch_steps=4, branch_warmup=1, test_seeds=[31],
                           eval_batches=1, eval_every=2, probe_every=1, probe_batch=2,
                           probe_positions=4, intervention_every=2)
    for arm in cfg["training"]["arms"]:
        arm["rate_grid"] = arm["rate_grid"][:2]
    return cfg


def smoke_checks(m, t, device):
    arms = [a for a in t["arms"] if a["name"] in ("adam", "gate")]
    models = initial_models(m, arms, 19, device)
    torch.manual_seed(50)
    tokens = torch.randint(m["vocab"], (2, m["sequence"]), device=device)
    torch.testing.assert_close(models["adam"](tokens, tokens), models["gate"](tokens, tokens), rtol=0, atol=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--corpus", required=True, type=Path)
    p.add_argument("--frozen-commit")
    p.add_argument("--frozen-tree")
    p.add_argument("--prereg-sha256")
    p.add_argument("--seconds", type=int, default=6500)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    deadline = time.monotonic() + args.seconds
    config = json.loads(args.config.read_text())
    if args.smoke:
        config = smoke_config(config)
        torch.set_num_threads(1)
        device = torch.device("cpu")
        synthetic(config, args.corpus)
    else:
        if not args.frozen_commit or not args.frozen_tree or not args.prereg_sha256:
            p.error("real runs require frozen commit, tree and PREREG hash")
        if sha256(Path(__file__).with_name("PREREG.md")) != args.prereg_sha256:
            raise ValueError("PREREG differs from the recorded frozen version")
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("a CUDA device with BF16 support is required")
        device = torch.device("cuda")
        prepare(config, args.corpus)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    m, t = config["model"], config["training"]
    streams = windows(args.corpus, m, t)
    if args.smoke:
        smoke_checks(m, t, device)
    write_json(args.output / "run.json", {
        "config": config, "config_sha256": sha256(args.config), "smoke": args.smoke,
        "torch": torch.__version__, "device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
        "slurm_id": os.getenv("SLURM_JOB_ID"),
        "frozen_commit": args.frozen_commit, "frozen_tree": args.frozen_tree,
        "prereg_sha256": args.prereg_sha256,
        "corpus_sha256": sha256(args.corpus / "corpus.json") if not args.smoke else None,
    })
    scores, winners = tune(m, t, streams, device, deadline)
    write_json(args.output / "tuning.json", {"candidates": scores, "chosen": winners})
    for seed in t["test_seeds"]:
        for arm in t["arms"]:
            check_deadline(deadline)
            run_arm(m, t, streams, arm, winners[arm["name"]], seed, device, args.output, deadline)


if __name__ == "__main__":
    main()
