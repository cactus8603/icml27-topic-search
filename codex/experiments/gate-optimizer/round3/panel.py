"""Run a fixed 4× panel in four shards and score once from verified receipts."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time


STAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(STAGE.parent / "round2"))
from experiment import cell, configuration, prepare, sha256, synthetic, windows, write_json
from judge import DETECTORS, STRONG
from validation import MULTIPLIER, validation_summary


SOURCE = "4d87430fd7cda6e74ea9e48deb57411911733e04"
CORPUS_SHA = "303517ecb7ef1a282e6733bd6240238b702fcc8d4fcb8b46f6da4581481a6d74"
CONFIG_SHA = "a03f7c1ee9340d4c00ae32acdf3fa0bf15eab463b0e6ced03f185be17d8740c7"
N_SHARDS = 4
SHARD_SIZE = 12
TRACE_FIELDS = ("step", "train_nll", "grad_norm", "lr")
EVAL_FIELDS = ("step", "dev_nll")
PROBE_FIELDS = ("step", *DETECTORS, "nll_before", "nll_after", "delta_nll", "write_max", "write_rms")
BRANCH_FIELDS = {"arm", "seed", "multiplier", "trace", "evaluations", "probes", "branch_wall_seconds"}
RESCUE_VALUES = {"uniform_delta_norm_relative_error", "control_change_norm_relative_error", "rollback_to_before_cap",
                 "nll_direction", "nll_uniform", "nll_control", "nll_wo_rollback"}


def panel_seeds(prior):
    seeds = [int.from_bytes(hashlib.sha256(f"gate-optimizer-round3-v1:{i}".encode()).digest()[:4], "big")
             & 0x7fffffff for i in range(N_SHARDS * SHARD_SIZE)]
    if 0 in seeds or len(set(seeds)) != len(seeds) or set(seeds) & set(prior):
        raise ValueError("fixed panel seeds are not unique and independent of previous panels")
    return seeds


def inputs(smoke=False):
    config = STAGE.parent / "round2" / "config.json"
    if sha256(config) != CONFIG_SHA:
        raise ValueError("round-two configuration hash mismatch")
    calibration = json.loads(config.read_bytes())
    base_path = STAGE.parent / "config.json"
    if sha256(base_path) != calibration["base_config_sha256"]:
        raise ValueError("base configuration hash mismatch")
    base = json.loads(base_path.read_bytes())
    t = base["training"]
    previous = [t["dev_seed"], *t["test_seeds"], *calibration["calibration_seeds"],
                *calibration["separate_validation_plan"]["seeds"]]
    seeds = panel_seeds(previous)
    cfg = configuration(base, calibration, smoke)
    if tuple(arm["name"] for arm in cfg["training"]["arms"]) != STRONG:
        raise ValueError("unexpected optimizer arms")
    calibration["separate_validation_plan"]["seeds"] = seeds
    return cfg, calibration, seeds


def shard_seeds(seeds, index):
    if not 0 <= index < N_SHARDS:
        raise ValueError("invalid shard index")
    return seeds[index * SHARD_SIZE:(index + 1) * SHARD_SIZE]


def strict_json(data):
    def invalid(placeholder):
        raise ValueError(f"invalid JSON constant: {placeholder}")

    return json.loads(data, parse_constant=invalid)


def number_or_null(value):
    return value is None or (type(value) in (int, float) and math.isfinite(value))


def row_schema(row, fields, nested=None):
    if not isinstance(row, dict) or set(row) - ({*fields, nested} if nested else set(fields)):
        raise ValueError("unexpected branch row fields")
    if not set(fields) <= row.keys() or type(row["step"]) is not int:
        raise ValueError("missing branch row fields")
    if any(not number_or_null(row[key]) for key in fields if key != "step"):
        raise ValueError("unexpected branch row value")


def verify_rescue(rescue):
    if isinstance(rescue, dict):
        status = rescue.get("status")
        if status in ("degenerate_signal", "degenerate_control") and set(rescue) == {"status"}:
            return
        if (status == "measured" and set(rescue) == RESCUE_VALUES | {"status", "layer"}
                and type(rescue["layer"]) is int and all(number_or_null(rescue[key]) for key in RESCUE_VALUES)):
            return
    raise ValueError("invalid rescue schema")


def verify_branch(raw, arm, seed, multiplier):
    branch = strict_json(raw)
    if multiplier is None:
        failure_fields = ({"step", "dev_nll"}, {"step", "train_nll", "grad_norm"})
        if not isinstance(branch, dict) or set(branch) not in failure_fields:
            raise ValueError("unexpected prefix-failure schema")
        row_schema(branch, tuple(branch))
        if all(branch[field] is not None for field in branch if field != "step"):
            raise ValueError("prefix-failure marker contains no failure")
        return
    if (not isinstance(branch, dict) or set(branch) != BRANCH_FIELDS
            or (branch["arm"], branch["seed"], branch["multiplier"]) != (arm, seed, multiplier)
            or type(branch["seed"]) is not int or type(branch["multiplier"]) is not int
            or branch["branch_wall_seconds"] is None or not number_or_null(branch["branch_wall_seconds"])
            or branch["branch_wall_seconds"] < 0):
        raise ValueError("unexpected branch identity or fields")
    for key, fields in (("trace", TRACE_FIELDS), ("evaluations", EVAL_FIELDS), ("probes", PROBE_FIELDS)):
        if not isinstance(branch[key], list):
            raise ValueError("invalid branch row list")
        for row in branch[key]:
            row_schema(row, fields, "rescue" if key == "probes" else None)
            if "rescue" in row:
                verify_rescue(row["rescue"])
    if multiplier == 1 and branch["probes"]:
        raise ValueError("nominal branch unexpectedly contains stress probes")


def branch_files(folder, seeds, stage_to=None):
    hashes = {}
    for seed in seeds:
        for arm in STRONG:
            stem = f"{arm}-{seed}"
            failure = folder / f"{stem}-prefix_failure.json"
            conditions = (None,) if failure.exists() else (1, MULTIPLIER)
            for multiplier in conditions:
                path = failure if multiplier is None else folder / f"{stem}-{multiplier}x.json"
                if path.is_symlink() or not path.is_file():
                    raise ValueError(f"missing expected branch: {path.name}")
                raw = path.read_bytes()
                verify_branch(raw, arm, seed, multiplier)
                hashes[path.name] = hashlib.sha256(raw).hexdigest()
                if stage_to is not None:
                    with (stage_to / path.name).open("xb") as copied:
                        copied.write(raw)
    found = {path.name for path in folder.iterdir()}
    if found - {"receipt.json"} != set(hashes):
        raise ValueError("unexpected or incomplete shard artifacts")
    return hashes


def provenance(args, seeds, index):
    return {
        "array_id": args.array_id,
        "frozen_commit": args.commit,
        "frozen_tree": args.tree,
        "source_commit": SOURCE,
        "prereg_sha256": sha256(STAGE / "PREREG.md"),
        "config_sha256": CONFIG_SHA,
        "corpus_sha256": CORPUS_SHA,
        "shard": index,
        "seeds": shard_seeds(seeds, index),
        "multiplier": MULTIPLIER,
        "arms": list(STRONG),
    }


def produce(args, base, calibration, seeds):
    import torch

    deadline = time.monotonic() + 3500
    if (torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported()
            or "H200" not in torch.cuda.get_device_name()):
        raise RuntimeError("requires one NVIDIA H200 with BF16")
    with torch.cuda.device(0):
        device = torch.device("cuda")
        prepare(base, args.scratch / "corpus")
        corpus = args.scratch / "corpus"
        if sha256(corpus / "corpus.json") != CORPUS_SHA:
            raise ValueError("tokenized corpus differs from round-two validation")
        record = provenance(args, seeds, args.shard)
        output = args.output / f"shard-{args.shard}"
        output.mkdir(parents=True, exist_ok=False)
        t = base["training"]
        streams = windows(corpus, base["model"], t)
        for seed in record["seeds"]:
            for arm in t["arms"]:
                cell(base["model"], t, arm, calibration["base_rates"][arm["name"]], seed,
                     [MULTIPLIER], calibration["criteria"]["damage_nll"], streams, output,
                     device, deadline, False)
        hashes = branch_files(output, record["seeds"])
        write_json(output / "receipt.json", record | {
            "branches": hashes,
            "torch": str(torch.__version__),
            "device": torch.cuda.get_device_name(),
            "producer_job": os.environ["SLURM_JOB_ID"],
        })


def collect(args, seeds, target):
    target.mkdir(exist_ok=False)
    receipts = []
    environments = set()
    for index in range(N_SHARDS):
        folder = args.output / f"shard-{index}"
        receipt_path = folder / "receipt.json"
        if folder.is_symlink() or receipt_path.is_symlink():
            raise ValueError("unexpected symlink in shard receipt")
        raw = receipt_path.read_bytes()
        record = strict_json(raw)
        expected = provenance(args, seeds, index)
        if (not isinstance(record, dict) or any(record.get(key) != value for key, value in expected.items())
                or set(record) != set(expected) | {"branches", "torch", "device", "producer_job"}
                or any(not isinstance(record[key], str) for key in ("producer_job", "torch", "device"))
                or not record["producer_job"].isdigit() or "H200" not in record["device"]):
            raise ValueError(f"shard {index} receipt provenance mismatch")
        hashes = branch_files(folder, expected["seeds"], stage_to=target)
        if record["branches"] != hashes:
            raise ValueError(f"shard {index} branch hash mismatch")
        environments.add((record["torch"], record["device"]))
        receipts.append({"shard": index, "producer_job": record["producer_job"],
                         "sha256": hashlib.sha256(raw).hexdigest()})
    if len(environments) != 1 or len({r["producer_job"] for r in receipts}) != N_SHARDS:
        raise ValueError("shards do not share a runtime or have distinct Slurm jobs")
    return receipts, environments.pop()


def analyze(args, base, calibration, seeds):
    destination = args.output / "validation.json"
    if destination.exists():
        raise FileExistsError(destination)
    target = args.scratch / "branches"
    receipts, (torch_version, device) = collect(args, seeds, target)
    metadata = {
        "config": calibration, "training": base["training"], "smoke": False,
        "slurm_id": args.array_id, "frozen_commit": args.commit, "frozen_tree": args.tree,
        "config_sha256": CONFIG_SHA, "prereg_sha256": sha256(STAGE / "PREREG.md"),
        "corpus_sha256": CORPUS_SHA, "device": device,
    }
    summary = validation_summary(metadata, target) | {
        "source_commit": SOURCE, "panel_receipts": receipts, "torch": torch_version,
        "analysis_job": os.environ["SLURM_JOB_ID"],
    }
    write_json(destination, summary)
    print(f"[codex] fresh 48-seed mechanical summary: {summary['status']}", flush=True)


def smoke(root):
    import torch

    base, calibration, seeds = inputs(smoke=True)
    torch.set_num_threads(1)
    root.mkdir(exist_ok=False)
    corpus = root / "corpus"
    synthetic(base, corpus)
    t = base["training"]
    arm = t["arms"][0]
    cell(base["model"], t, arm, calibration["base_rates"][arm["name"]], seeds[0], [MULTIPLIER],
         calibration["criteria"]["damage_nll"], windows(corpus, base["model"], t), root,
         torch.device("cpu"), time.monotonic() + 60, True)
    for multiple in (1, MULTIPLIER):
        verify_branch((root / f"{arm['name']}-{seeds[0]}-{multiple}x.json").read_bytes(), arm["name"], seeds[0], multiple)
    print("[codex] synthetic CPU cell and both branch schemas passed", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    cmds = parser.add_subparsers(dest="action", required=True)
    cmds.add_parser("plan").add_argument("--verify-assets", action="store_true")
    cmds.add_parser("smoke").add_argument("output", type=Path)
    for name in ("produce", "analyze"):
        action = cmds.add_parser(name)
        for key in ("commit", "tree", "array-id"):
            action.add_argument(f"--{key}", required=True)
        for key in ("output", "scratch"):
            action.add_argument(f"--{key}", type=Path, required=True)
        if name == "produce":
            action.add_argument("--shard", type=int, choices=range(N_SHARDS), required=True)
    args = parser.parse_args()
    if args.action == "smoke":
        smoke(args.output)
        return
    base, calibration, seeds = inputs()
    if args.action == "plan":
        if args.verify_assets:
            for key in ("parquet", "tokenizer"):
                if sha256(base["source"][key]) != base["source"][f"{key}_sha256"]:
                    raise ValueError(f"input digest differs: {key}")
        print(json.dumps({"multiplier": MULTIPLIER, "seeds": seeds,
                          "shards": [shard_seeds(seeds, i) for i in range(N_SHARDS)],
                          "pairs": len(seeds) * len(STRONG)}, indent=2))
    elif args.action == "produce":
        produce(args, base, calibration, seeds)
    else:
        analyze(args, base, calibration, seeds)


if __name__ == "__main__":
    main()
