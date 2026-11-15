"""Measure a denser fixed panel of new nominal optimizer trajectories."""

import argparse
from hashlib import sha256 as digest
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "gate-optimizer-nominal"))
from measure import (CORPUS_SHA, MAX_EVAL_DIFFERENCE, STRONG, analyze, cell, inputs,
                     nominal_branch, parity_difference, prepare, sha256, synthetic, windows, write_json)
from panel import CONFIG_SHA, PROBE_FIELDS, STAGE, row_schema, strict_json, verify_rescue


SOURCE = "6e16f498bb21044128f8383a684874f3ea12e29c"
NOMINAL = "d89b15761fbf6be47bb7098d3a088061d76751ab"
SHARDS = 8
PER_SHARD = 12
STRIDE = 2
RATE_DRAWS = 10_000
RATE_SEED = 202611151437
JOB_ID = re.compile(r"[1-9][0-9]*")
FREEZE = re.compile(r"[0-9a-f]{40}")


def plan(smoke=False):
    base, calibration, previous = inputs(smoke=smoke)
    for name, key in (("config.json", "training"), ("round2/config.json", None)):
        path = STAGE.parent / name
        original = strict_json(path.read_bytes())
        if key:
            previous += [original[key]["dev_seed"], *original[key]["test_seeds"]]
        else:
            previous += [*original["calibration_seeds"], *original["separate_validation_plan"]["seeds"]]
    seeds = [int.from_bytes(digest(f"gate-optimizer-nominal-dense-v1:{i}".encode()).digest()[:4], "big")
             & 0x7fffffff for i in range(SHARDS * PER_SHARD)]
    if 0 in seeds or len(set(seeds)) != len(seeds) or set(seeds).intersection(previous):
        raise ValueError("Dense seed roster overlaps a previous measurement")
    base["training"]["probe_every"] = STRIDE
    return base, calibration, seeds


def assigned(seeds, shard):
    return seeds[shard * PER_SHARD:(shard + 1) * PER_SHARD]


def probe_steps(training):
    return [step for step in range(STRIDE, training["branch_steps"] + 1, STRIDE)
            if step > training["branch_warmup"]]


def paired_probe(base, arm, rate, seed, threshold, streams, device, deadline, ref_dir=None):
    training = base["training"]
    branch = nominal_branch(base["model"], training, arm, rate, seed, threshold, streams, device, deadline)
    if ref_dir is None:
        return branch["probes"], None
    cell(base["model"], training, arm, rate, seed, [], threshold, streams, ref_dir, device, deadline, False)
    raw = (ref_dir / f"{arm['name']}-{seed}-1x.json").read_bytes()
    ref = strict_json(raw)
    difference = parity_difference(branch, ref, training)
    if device.type == "cpu" and branch["trace"] != ref["trace"]:
        raise ValueError("Dense synthetic probes changed the training trace")
    return branch["probes"], difference


def validate_rows(rows, steps, threshold, forced=False):
    if not isinstance(rows, list) or [row["step"] for row in rows] != steps:
        raise ValueError("Dense probe cadence or count differs")
    for row in rows:
        row_schema(row, PROBE_FIELDS, "rescue")
        if any(row[field] is None for field in PROBE_FIELDS):
            raise ValueError("Nonfinite dense probe")
        if ("rescue" in row) != (forced or row["delta_nll"] >= threshold):
            raise ValueError("Lost a damaged-step intervention or included an ineligible one")
        if "rescue" in row:
            verify_rescue(row["rescue"])


def collect_shard(root, freeze, array, shard, jobs, seeds, steps, threshold):
    directory = root / f"shard-{shard}"
    receipt = strict_json((directory / "receipt.json").read_bytes())
    expected = {"freeze": freeze, "array": array, "shard": shard, "job": jobs[shard],
                "source": SOURCE, "nominal": NOMINAL, "corpus_sha256": CORPUS_SHA,
                "seeds": seeds, "config_sha256": CONFIG_SHA}
    if (set(receipt) != {*expected, "device", "torch", "branches_sha256"}
            or any(receipt[key] != value for key, value in expected.items())
            or "H200" not in receipt["device"]):
        raise ValueError(f"Foreign or mismatched producer receipt {shard}")
    path = directory / "branches.json"
    raw = path.read_bytes()
    if digest(raw).hexdigest() != receipt["branches_sha256"]:
        raise ValueError(f"Digest differs for shard {shard}")
    branches = strict_json(raw)
    if [(record["seed"], record["optimizer"]) for record in branches] != [
            (seed, arm) for seed in seeds for arm in STRONG]:
        raise ValueError("Out-of-order, duplicate or missing branch in a shard")
    groups, differences = {}, []
    for record in branches:
        if set(record) != {"seed", "optimizer", "dev_difference", "probes"}:
            raise ValueError("Unexpected branch fields")
        difference = record["dev_difference"]
        if record["seed"] == seeds[0]:
            if type(difference) not in (int, float) or difference < 0:
                raise ValueError("Missing reference for the sentinel seed")
            differences.append(difference)
        elif difference is not None:
            raise ValueError("Unexpected reference for a non-sentinel seed")
        rows = record["probes"]
        validate_rows(rows, steps, threshold)
        groups[record["optimizer"], record["seed"]] = rows
    return groups, differences, (receipt["device"], receipt["torch"])


def completed_shards(array, history):
    jobs = {}
    for line in history.splitlines():
        external, job, state, code = line.split("|")
        prefix = f"{array}_"
        if not external.startswith(prefix):
            continue
        index = int(external[len(prefix):])
        if (not 0 <= index < SHARDS or index in jobs or not JOB_ID.fullmatch(job)
                or (state, code) != ("COMPLETED", "0:0")):
            raise ValueError("Producer task missing, duplicated, requeued or unsuccessful")
        jobs[index] = job
    if set(jobs) != set(range(SHARDS)) or len(set(jobs.values())) != SHARDS:
        raise ValueError("All eight producer tasks must succeed exactly once")
    return jobs


def rate_summary(groups, seeds, threshold):
    counts = np.array([sum(row["delta_nll"] >= threshold for arm in STRONG for row in groups[arm, seed])
                       for seed in seeds])
    sizes = np.array([sum(len(groups[arm, seed]) for arm in STRONG) for seed in seeds])
    interval = None
    if counts.any():
        indices = np.random.default_rng(RATE_SEED).integers(len(seeds), size=(RATE_DRAWS, len(seeds)))
        draws = counts[indices].sum(axis=1) / sizes[indices].sum(axis=1)
        interval = np.quantile(draws, [0.025, 0.975]).tolist()
    return {"rate": float(counts.sum() / sizes.sum()), "seed_bootstrap_95": interval,
            "draws": RATE_DRAWS if interval is not None else 0, "seed": RATE_SEED}


def produce(args, base, calibration, seeds):
    device = torch.device("cuda")
    if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported() or "H200" not in torch.cuda.get_device_name():
        raise RuntimeError("Use precisely one H200 for this panel")
    corpus = args.scratch / "corpus"
    prepare(base, corpus)
    if sha256(corpus / "corpus.json") != CORPUS_SHA:
        raise ValueError("Dense panel corpus differs from the nominal reference")
    selected = assigned(seeds, args.shard)
    training = base["training"]
    streams = windows(corpus, base["model"], training)
    deadline = time.monotonic() + 3450
    threshold = calibration["criteria"]["damage_nll"]
    ref_dir = args.scratch / "sentinels"
    ref_dir.mkdir()
    records = []
    for seed in selected:
        for arm in training["arms"]:
            name = arm["name"]
            rows, difference = paired_probe(base, arm, calibration["base_rates"][name], seed, threshold, streams,
                                            device, deadline, ref_dir if seed == selected[0] else None)
            validate_rows(rows, probe_steps(training), threshold)
            if difference is not None and difference > MAX_EVAL_DIFFERENCE:
                raise ValueError("Dense reference deviates from the uninstrumented nominal trajectory")
            records.append({"seed": seed, "optimizer": name, "dev_difference": difference, "probes": rows})
        print(f"[codex] Completed {len(records) // len(STRONG)}/{len(selected)} nominal seeds", flush=True)
    directory = args.output / f"shard-{args.shard}"
    directory.mkdir(parents=True)
    branches = directory / "branches.json"
    with branches.open("x") as handle:
        json.dump(records, handle, separators=(",", ":"), allow_nan=False)
    write_json(directory / "receipt.json", {
        "freeze": args.freeze, "array": args.array, "job": args.job, "shard": args.shard,
        "source": SOURCE, "nominal": NOMINAL, "corpus_sha256": CORPUS_SHA, "config_sha256": CONFIG_SHA,
        "seeds": selected, "branches_sha256": sha256(branches), "device": torch.cuda.get_device_name(),
        "torch": str(torch.__version__),
    })


def summarize(args, base, calibration, seeds):
    history = subprocess.check_output(["sacct", "-X", "-n", "-P", "-j", args.array,
                                       "-o", "JobID,JobIDRaw,State,ExitCode"], text=True)
    jobs = completed_shards(args.array, history)
    groups, differences, runtimes = {}, [], set()
    threshold = calibration["criteria"]["damage_nll"]
    steps = probe_steps(base["training"])
    for shard in range(SHARDS):
        rows, parity, runtime = collect_shard(args.output, args.freeze, args.array, shard, jobs,
                                              assigned(seeds, shard), steps, threshold)
        groups.update(rows)
        differences.extend(parity)
        runtimes.add(runtime)
    if len(groups) != len(STRONG) * len(seeds) or len(runtimes) != 1:
        raise ValueError("Inconsistent runtime or incomplete grouping across producers")
    result = analyze(groups, seeds, calibration, max(differences))
    if result["state"] == "eligible":
        result["damage_rate"] = rate_summary(groups, seeds, threshold)
    output = args.output / "summary.json"
    if output.exists():
        raise FileExistsError(output)
    write_json(output, result | {"scope": "fresh_seeds_dense_nominal", "freeze": args.freeze,
                                "array": args.array, "job": args.job, "completed_tasks": jobs,
                                "source": SOURCE, "nominal": NOMINAL, "runtime": next(iter(runtimes)),
                                "corpus_sha256": CORPUS_SHA, "probe_steps_per_branch": steps})
    print(f"[codex] Dense nominal summary SHA-256 {sha256(output)}", flush=True)


def smoke(scratch, seeds):
    torch.set_num_threads(1)
    base, calibration, _ = plan(smoke=True)
    training = base["training"]
    corpus = scratch / "synthetic"
    synthetic(base, corpus)
    streams = windows(corpus, base["model"], training)
    ref_dir = scratch / "synthetic-reference"
    ref_dir.mkdir()
    deadline = time.monotonic() + 180
    for arm in training["arms"]:
        rows, difference = paired_probe(base, arm, calibration["base_rates"][arm["name"]], seeds[0],
                                        -float("inf"), streams, torch.device("cpu"), deadline, ref_dir)
        validate_rows(rows, probe_steps(training), calibration["criteria"]["damage_nll"], forced=True)
        if difference != 0 or not rows:
            raise ValueError("Dense CPU parity failed")
    print("[codex] Dense probes and forced rescues preserve nominal training for all three optimizers")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "produce", "analyze"))
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--array")
    parser.add_argument("--job")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--shard", type=int, choices=range(SHARDS))
    args = parser.parse_args()
    if not FREEZE.fullmatch(args.freeze):
        parser.error("A full commit is required")
    base, calibration, seeds = plan()
    if args.mode == "check":
        for label in ("parquet", "tokenizer"):
            if sha256(base["source"][label]) != base["source"][f"{label}_sha256"]:
                raise ValueError(f"Frozen asset differs: {label}")
        smoke(args.scratch, seeds)
        print(f"[codex] Verified {len(seeds)} new seeds, {len(probe_steps(base['training']))} probes per branch")
        return
    if not args.output or not all(JOB_ID.fullmatch(value or "") for value in (args.array, args.job)):
        parser.error("A recorded Slurm job, array and output path are required")
    if args.mode == "produce":
        if args.shard is None:
            parser.error("A fixed shard index is required for production")
        produce(args, base, calibration, seeds)
    else:
        summarize(args, base, calibration, seeds)


if __name__ == "__main__":
    main()
