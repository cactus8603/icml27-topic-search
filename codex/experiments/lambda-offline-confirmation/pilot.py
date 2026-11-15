"""Train or score the independent GCD leakage and reliability panel."""

import argparse
from fractions import Fraction
from importlib import metadata
from itertools import permutations
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
STAGED = HERE.parent / "lambda-gcd-run"
sys.path.insert(0, str(STAGED))
import run_cell as staged
from score_gate3 import load_cohorts, score_panel


STAGE = "independent_gcd_leakage_v1"
RANGE_MINIMUM = Fraction(1, 5)
RUNTIME = {"torch": "2.6.0", "transformers": "4.52.3", "tokenizers": "0.21.4",
           "peft": "0.15.1", "accelerate": "1.14.0"}


def run_dir(array):
    return staged.ROOT / "runs/inoculation-lambda" / f"gcd-offline-confirmation-{array}"


def train_hash(arm, rows):
    return staged.digest(staged.encode_rows(staged.training_rows(arm, rows)))


def binding(array, task, freeze, rows):
    arm, seed = staged.CELLS[task]
    return {"schema": 1, "stage": STAGE, "arm": arm, "seed": seed, "array": array,
            "task": task, "freeze": freeze, "author": staged.AUTHOR,
            "material_sha256": staged.INPUT_SHA, "training_sha256": train_hash(arm, rows),
            "training_design_sha256": staged.digest(staged.json_bytes(staged.plan())),
            "generation": staged.GENERATE, "rows": 300}


def atomic_write(path, raw):
    pending = path.with_suffix(".pending")
    with pending.open("xb") as stream:
        stream.write(raw)
    try:
        os.link(pending, path)
    finally:
        pending.unlink()


def run_cell(source, scratch, array, task, freeze, rows):
    import torch
    from grad_shift.evaluation.model_loader import load_model_and_tokenizer
    from worker import verify_artifacts

    if (torch.cuda.device_count() != 1 or "H200" not in torch.cuda.get_device_name()
            or not torch.cuda.is_bf16_supported() or torch.version.cuda != "12.4"):
        raise RuntimeError("the pinned CUDA runtime and one BF16-capable H200 are required")
    arm, seed = staged.CELLS[task]
    prompts = staged.prompts_for_run()
    output = run_dir(array)
    cell = output / f"{arm}-seed-{seed}"
    cell.mkdir(parents=True, exist_ok=False)
    train_file = scratch / "train.jsonl"
    train_file.write_bytes(staged.encode_rows(staged.training_rows(arm, rows)))
    model_dir = scratch / "adapter"
    model_dir.mkdir()
    config = scratch / "train_config.json"
    config.write_bytes(staged.json_bytes(staged.TRAIN | {
        "seed": seed, "train_file": str(train_file), "output_dir": str(model_dir),
    }))
    parsed = staged.resolve_training(source, config)
    start = time.monotonic()
    subprocess.run([sys.executable, "-IBu", str(STAGED / "train_entry.py"), str(source), str(config)],
                   check=True, cwd=source)
    artifacts = verify_artifacts(parsed)
    train_seconds = time.monotonic() - start
    start = time.monotonic()
    model, tokenizer = load_model_and_tokenizer(str(model_dir), dtype="bfloat16")
    if model.device.type != "cuda":
        raise ValueError("the pinned evaluation model did not load on CUDA")
    response_rows = staged.generate_rows(model, tokenizer, arm, seed, prompts)
    if len(response_rows) != 300:
        raise ValueError("the scheduled rows were not generated")
    payload = staged.encode_rows(response_rows)
    receipt = binding(array, task, freeze, rows) | {
        "config_sha256": staged.digest(config.read_bytes()), "response_sha256": staged.digest(payload),
        "adapter_files": artifacts, "train_seconds": train_seconds,
        "generation_seconds": time.monotonic() - start, "device": torch.cuda.get_device_name(),
    }
    atomic_write(output / staged.run_filename(arm, seed), payload)
    atomic_write(cell / "receipt.json", staged.json_bytes(receipt))
    print(json.dumps({"stage": STAGE, "arm": arm, "seed": seed, "rows": len(response_rows),
                      "receipt_sha256": staged.digest(staged.json_bytes(receipt))}, sort_keys=True))


def verified_regular(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing regular file: {path.name}")
    return path.read_bytes()


def verify_panel(output, array, freeze, rows):
    receipts = {}
    for task, (arm, seed) in enumerate(staged.CELLS):
        cell = f"{arm}-seed-{seed}"
        raw = verified_regular(output / cell / "receipt.json")
        receipt = staged.strict_json(raw)
        payload = verified_regular(output / staged.run_filename(arm, seed))
        expected = binding(array, task, freeze, rows) | {"response_sha256": staged.digest(payload)}
        if any(receipt.get(key) != value for key, value in expected.items()):
            raise ValueError(f"receipt does not bind the frozen cell: {cell}")
        receipts[cell] = staged.digest(raw)
    return receipts


def defended_range_lower(runs):
    bounds = {arm: (Fraction(0), Fraction(0)) for arm in staged.ARMS if arm != "none"}
    for run in runs:
        arm = run["arm"]
        if arm not in bounds:
            continue
        families = list(run["families"].values())
        low = sum(Fraction(item["known_affirmations"], item["scheduled"]) for item in families)
        high = sum(Fraction(item["known_affirmations"] + item["scheduled"] - item["parsed"],
                            item["scheduled"]) for item in families)
        old_low, old_high = bounds[arm]
        n = len(families) * len(staged.SEEDS)
        bounds[arm] = old_low + low / n, old_high + high / n
    return max(Fraction(0), *(left[0] - right[1] for left, right in permutations(bounds.values(), 2)))


def evaluate(array, freeze, rows):
    output = run_dir(array)
    if (output / "summary.json").exists():
        raise ValueError("this frozen panel already has a summary")
    receipts = verify_panel(output, array, freeze, rows)
    result = score_panel(output, load_cohorts(staged.INPUT_DIR))
    lower = defended_range_lower(result["runs"])
    reliable = result["diagnostic"] in {"COMPLETE_DIAGNOSTIC_AT_LEAST_THRESHOLD",
                                       "BOUNDED_DIAGNOSTIC_AT_LEAST_THRESHOLD"}
    result.update(stage=STAGE, freeze=freeze, array=array, receipt_sha256=receipts,
                  defended_range_lower=float(lower), defended_range_threshold=float(RANGE_MINIMUM),
                  defended_range_support=lower >= RANGE_MINIMUM, reliability_support=reliable,
                  independent_gcd_phenomenon_support=lower >= RANGE_MINIMUM and reliable)
    atomic_write(output / "summary.json", staged.json_bytes(result))
    print(json.dumps({key: result[key] for key in (
        "stage", "freeze", "array", "scheduled_responses", "icc_1_1_lower", "diagnostic",
        "defended_range_lower", "independent_gcd_phenomenon_support",
    )}, allow_nan=False, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "train", "judge"))
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--array")
    parser.add_argument("--task", type=int, choices=range(len(staged.CELLS)))
    parser.add_argument("--freeze")
    parser.add_argument("--scratch", type=Path)
    args = parser.parse_args()
    if args.mode != "check" and not (re.fullmatch(r"[1-9][0-9]*", args.array or "")
                                     and re.fullmatch(r"[0-9a-f]{40}", args.freeze or "")):
        parser.error("a numeric array and full frozen Git revision are required")
    if args.mode == "train" and (args.task is None or args.scratch is None
            or args.scratch.resolve().parent != (staged.ROOT / "tmp").resolve()
            or args.array != os.environ.get("SLURM_ARRAY_JOB_ID")
            or str(args.task) != os.environ.get("SLURM_ARRAY_TASK_ID")):
        parser.error("training requires the corresponding Slurm array cell and shared scratch directory")
    if args.mode != "train":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if sys.version_info[:3] != (3, 10, 20) or any(metadata.version(key) != value for key, value in RUNTIME.items()):
        raise ValueError("the pinned GCD runtime differs")
    staged.verify_hashes(staged.MODEL, staged.TOKENIZER_SHA)
    staged.verify_hashes(args.source, staged.SOURCE_HASHES)
    sys.path[:0] = [str(args.source), str(HERE.parent / "lambda-author-training")]
    rows, _ = staged.load_rows()
    if args.mode == "check":
        staged.preview(args.source, rows, staged.prompts_for_run())
    elif args.mode == "train":
        run_cell(args.source, args.scratch, args.array, args.task, args.freeze, rows)
    else:
        evaluate(args.array, args.freeze, rows)


if __name__ == "__main__":
    main()
