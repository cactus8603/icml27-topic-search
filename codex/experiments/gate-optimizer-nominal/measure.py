"""Probe the unmeasured nominal continuations on the round-three panel."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch


TOPIC = Path(__file__).resolve().parent.parent / "gate-optimizer"
sys.path.insert(0, str(TOPIC / "round3"))
from panel import CORPUS_SHA, DETECTORS, N_SHARDS, STRONG, inputs, shard_seeds, verify_branch
from experiment import cell, prepare, rescue_probe, sha256, synthetic, windows, write_json
from judge import rescue_summary, signal_summary, status
from run import branch_rate, check_deadline, evaluate, initial_models, microstep, optimizers, probe_after, probe_before


REFERENCE_COMMIT = "6e16f498bb21044128f8383a684874f3ea12e29c"
REFERENCE_RECEIPTS = (
    "a754942b54484599a0959623c9cbaed23736bf802dda9399ca3c20408d021255",
    "c7d198ccbbd387f09a1915fcf9bb99c96b967eb09a80753dad357c5e90b2f74f",
    "d46677b84dfa0989ef91871ef0902516c1675cd46b359dcd95db8ed5e2ee39ab",
    "3d94dd3314b3aa674ba1b99be89bf8e886fd6d10c38c438753e152b22f984271",
)
REFERENCE = Path("/work/ChihChungHsu/cactus8603/icml/runs/gate-optimizer/fresh-26521")
MAX_EVAL_DIFFERENCE = 0.01


def nominal_branch(model_config, training, arm, rate, seed, threshold, streams, device, deadline):
    model = initial_models(model_config, [arm], seed, device)[arm["name"]]
    opts = optimizers(model, arm, training)
    shared, length = training["shared_steps"], training["branch_steps"]
    order = streams["train"].order(shared + length, seed)
    eval_order = streams["dev"].order(training["eval_batches"], 541103)
    warmup = min(training["tune_warmup"], shared)
    every = training["probe_every"]
    scheduled = range(every, length + 1, every)
    probe_rows = dict(zip(scheduled, streams["probe"].order(len(scheduled), seed + 49), strict=True))

    for step, batch in enumerate(order[:shared], 1):
        check_deadline(deadline)
        lr = min(1.0, step / warmup) * rate
        loss, grad = microstep(model, opts, streams["train"], batch, lr, training, device)
        if loss is None or grad is None:
            raise ValueError(f"Nonfinite prefix at seed {seed}, step {step}")

    evaluations = [{"step": 0, "dev_nll": evaluate(model, streams["dev"], eval_order, device)}]
    trace, probes = [], []
    for step, batch in enumerate(order[shared:], 1):
        check_deadline(deadline)
        saved = None
        if step > training["branch_warmup"] and step in probe_rows:
            signal, outcome = np.split(probe_rows[step], 2)
            saved = probe_before(model, streams["probe"], signal, outcome, step, training, device, False)
        lr = branch_rate(step - 1, length, 0, rate)
        loss, grad = microstep(model, opts, streams["train"], batch, lr, training, device)
        if loss is None or grad is None:
            raise ValueError(f"Nonfinite continuation at seed {seed}, step {step}")
        trace.append({"step": step, "train_nll": loss, "grad_norm": grad, "lr": lr})
        if saved is not None:
            row = {"step": step, "grad_norm": grad, **probe_after(model, saved, device)}
            if any(row[key] is None for key in (*DETECTORS, "nll_before", "nll_after", "delta_nll")):
                raise ValueError(f"Nonfinite probe at seed {seed}, step {step}")
            if row["delta_nll"] >= threshold:
                row["rescue"] = rescue_probe(model, saved, row, device)
            probes.append(row)
        if step % training["eval_every"] == 0 or step == length:
            evaluations.append({"step": step, "dev_nll": evaluate(model, streams["dev"], eval_order, device)})
    return {"trace": trace, "evaluations": evaluations, "probes": probes}


def parity_difference(branch, ref, training):
    for run in (branch, ref):
        if status(run, training, 1) != "completed":
            raise ValueError("Incomplete nominal branch or evaluation")
    if [r["lr"] for r in branch["trace"]] != [r["lr"] for r in ref["trace"]]:
        raise ValueError("Nominal learning-rate schedule differs")
    return max(abs(row["dev_nll"] - expected["dev_nll"]) for row, expected in
               zip(branch["evaluations"], ref["evaluations"], strict=True))


def analyze(groups, seeds, calibration, parity):
    criteria = calibration["criteria"].copy()
    plan = calibration["separate_validation_plan"]
    criteria.update(signal_min_damage=plan["min_damage_events"], signal_min_scored_seeds=plan["min_scored_seeds"])
    eligible = math.isfinite(parity) and parity <= MAX_EVAL_DIFFERENCE
    damages = {key: [r for r in rows if r["delta_nll"] >= criteria["damage_nll"]] for key, rows in groups.items()}
    by_arm = {arm: {"probes": sum(len(groups[arm, seed]) for seed in seeds),
                    "damaged_steps": sum(len(damages[arm, seed]) for seed in seeds)} for arm in STRONG}
    damaged = [row for rows in damages.values() for row in rows]
    damage_seeds = len({seed for (_, seed), rows in damages.items() if rows})
    damage_arms = sum(arm["damaged_steps"] > 0 for arm in by_arm.values())
    phenomenon = (len(damaged) >= criteria["min_damage_events"] and damage_seeds >= criteria["min_damage_seeds"]
                  and damage_arms >= criteria["min_damage_arms"])
    signal = signal_summary(groups, seeds, criteria, eligible, len(damaged))
    increment = signal.get("mean_held_out_increment")
    return {
        "state": "eligible" if eligible else "unable_parity_guard",
        "max_reference_dev_nll_difference": parity,
        "phenomenon_qualifies": phenomenon if eligible else None,
        "by_optimizer": by_arm,
        "probe_steps": sum(len(rows) for rows in groups.values()),
        "damaged_steps": len(damaged), "damage_seeds": damage_seeds, "damage_arms": damage_arms,
        "signal": signal,
        "signal_meets_margin": None if increment is None else increment >= plan["min_average_increment_auc"],
        "local_rescue": rescue_summary(damaged) if eligible else None,
    }


def read_reference(folder, receipt, arm, seed, device_name):
    if (receipt["frozen_commit"] != REFERENCE_COMMIT or receipt["torch"] != str(torch.__version__)
            or receipt["device"] != device_name):
        raise ValueError("Nominal reference requires the same frozen program and runtime")
    ref = folder / f"{arm}-{seed}-1x.json"
    raw = ref.read_bytes()
    if hashlib.sha256(raw).hexdigest() != receipt["branches"].get(ref.name):
        raise ValueError("Nominal reference branch hash differs")
    verify_branch(raw, arm, seed, 1)
    return json.loads(raw)


def run(root, freeze, output, smoke=False):
    base, calibration, seeds = inputs(smoke=smoke)
    training = base["training"]
    corpus = root / "corpus"
    if smoke:
        torch.set_num_threads(1)
        device = torch.device("cpu")
        synthetic(base, corpus)
    else:
        device = torch.device("cuda")
        if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Use one BF16-capable H200")
        prepare(base, corpus)
        if sha256(corpus / "corpus.json") != CORPUS_SHA:
            raise ValueError("Corpus differs from the nominal reference")
    streams = windows(corpus, base["model"], training)
    threshold = calibration["criteria"]["damage_nll"]
    rates = calibration["base_rates"]
    deadline = time.monotonic() + (180 if smoke else 5300)
    if smoke:
        ref_dir = root / "smoke-reference"
        ref_dir.mkdir()
        synthetic_groups = {}
        for arm in training["arms"]:
            name, seed = arm["name"], seeds[0]
            # Test every probe and counterfactual path, including restoration.
            cell(base["model"], training, arm, rates[name], seed, [], threshold, streams,
                 ref_dir, device, deadline, True)
            branch = nominal_branch(base["model"], training, arm, rates[name], seed, -math.inf, streams, device, deadline)
            ref = json.loads((ref_dir / f"{name}-{seed}-1x.json").read_bytes())
            if parity_difference(branch, ref, training) != 0 or branch["trace"] != ref["trace"] or not branch["probes"]:
                raise ValueError(f"CPU probe changed the nominal trajectory for {name}")
            synthetic_groups[name, seed] = branch["probes"]
        coverage = analyze(synthetic_groups, seeds[:1], calibration, 0.0)
        rejected = analyze(synthetic_groups, seeds[:1], calibration, MAX_EVAL_DIFFERENCE * 2)
        if (coverage["signal"]["status"] != "insufficient_coverage"
                or rejected["phenomenon_qualifies"] is not None or rejected["local_rescue"] is not None):
            raise ValueError("Coverage or reference parity gate failed smoke check")
        print("[codex] CPU probe and counterfactual trajectory parity passed for three optimizers")
        return

    device_name = torch.cuda.get_device_name()
    groups, differences = {}, []
    for index in range(N_SHARDS):
        folder = REFERENCE / f"shard-{index}"
        receipt_path = folder / "receipt.json"
        if sha256(receipt_path) != REFERENCE_RECEIPTS[index]:
            raise ValueError(f"Nominal reference receipt {index} differs")
        receipt = json.loads(receipt_path.read_bytes())
        for seed in shard_seeds(seeds, index):
            for arm in training["arms"]:
                name = arm["name"]
                ref = read_reference(folder, receipt, name, seed, device_name)
                branch = nominal_branch(base["model"], training, arm, rates[name], seed, threshold, streams,
                                        device, deadline)
                differences.append(parity_difference(branch, ref, training))
                groups[name, seed] = branch["probes"]
        print(f"[codex] completed nominal references for {index + 1}/{N_SHARDS} groups of seeds", flush=True)
    expected_probes = [step for step in range(training["probe_every"], training["branch_steps"] + 1,
                                             training["probe_every"]) if step > training["branch_warmup"]]
    if len(groups) != len(seeds) * len(STRONG) or any([r["step"] for r in group] != expected_probes
                                                    for group in groups.values()):
        raise ValueError("Nominal panel incomplete")
    write_json(output, analyze(groups, seeds, calibration, max(differences)) | {
        "scope": "nominal cosine schedule on the same seeds and probe rows as the third-round 4x schedule",
        "job": os.environ["SLURM_JOB_ID"], "freeze": freeze, "reference_commit": REFERENCE_COMMIT,
        "reference_receipts": REFERENCE_RECEIPTS, "corpus_sha256": CORPUS_SHA, "device": device_name,
        "torch": str(torch.__version__),
    })
    print(f"[codex] nominal optimizer summary SHA-256 {sha256(output)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke == (args.output is not None):
        parser.error("Pass --smoke or --output")
    run(args.scratch, args.freeze, args.output, args.smoke)
