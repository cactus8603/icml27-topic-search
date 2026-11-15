"""Isolate the number of distinct samples on one conditional tree population."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
FORMAL = "9e92804a7bfded3c5cafd4fccd4315df2034eca5"
PICKS = tuple(range(6))
COUNTS = (833, 5000)
BATCH = 85
UPDATES_PER_EPOCH = 10
HORIZON = 20000
GRID = (1, 2, 4, 8, 12, 18, 24, 30, 36, 44, 54, 66, 82, 101, 125, 155, 192)
MIN_COMMON = 10
MEANINGFUL_LOG_RATIO = 0.4463
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 2026111520


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def instrument(source, grid=GRID):
    """Preserve the pinned architecture, optimizer, scheduler and noise path."""
    replacements = {
        "args = ap.parse_args()": 'ap.add_argument("--support-size", type=int, required=True)\nargs = ap.parse_args()',
        "n_used = len(X)": (
            "if not 0 < args.support_size <= len(X):\n"
            "    raise ValueError('invalid support size for the fixed pool')\n"
            "X, C = X[:args.support_size], C[:args.support_size]\n"
            "n_used = len(X)"
        ),
        "ck_epochs = ckpt_schedule(args.first_ckpt_epoch, args.epochs, args.max_ratio)": f"ck_epochs = {list(grid)!r}",
        "math.ceil(n_used / args.batch)": str(UPDATES_PER_EPOCH),
        "for ep in range(1, args.epochs + 1):": (
            "sample_rng = torch.Generator(device=dev).manual_seed(args.seed + 1_000_000)\n"
            f"for ep in range(1, {grid[-1]} + 1):"
        ),
        "perm = torch.randperm(n, device=dev)": (
            f"perm = (torch.rand({UPDATES_PER_EPOCH} * args.batch, device=dev, "
            "generator=sample_rng, dtype=torch.float64) * n).long()"
        ),
        "for s in range(0, n, args.batch):": "for s in range(0, len(perm), args.batch):",
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise ValueError(f"Expected one matching line in the frozen trainer: {old}")
        source = source.replace(old, new)
    compile(source, "<matched support-size trainer>", "exec")
    return source


def verify_data(path, config):
    if digest(path) != config["data_sha256"]:
        raise ValueError("Generated grammar sample bytes differ from the frozen corpus")
    q, depth, _, roots, leaves, transition, _ = np.load(path, allow_pickle=True)
    if (q != 6 or depth != 4 or np.shape(roots) != (35000,) or np.shape(leaves) != (16, 35000)
            or not np.issubdtype(leaves.dtype, np.integer) or np.any(roots != 0)
            or np.any(leaves < 0) or np.any(leaves >= q)
            or np.shape(transition) != (q, q, q) or not np.allclose(transition.sum((1, 2)), 1)):
        raise ValueError("Invalid matched conditional-tree dataset")
    if len({row.tobytes() for row in leaves.T}) != len(roots):
        raise ValueError("Support-size diagnostic requires distinct support points")


def train_panel(trainer, data, scratch, count):
    runs = []
    for pick in PICKS:
        directory = scratch / f"n{count}-pick{pick}"
        subprocess.run([
            sys.executable, "-IB", str(trainer), "--data", str(data), "--out", str(directory),
            "--pick", str(pick), "--seed", str(48_000 + pick), "--n", str(max(COUNTS)),
            "--support-size", str(count), "--epochs", str(HORIZON), "--batch", str(BATCH),
            "--pdrop", "1.0", "--class-only", "0", "--device", "cuda",
        ], check=True)
        meta = json.loads((directory / "meta.json").read_text())
        if (meta["n_c"], meta["steps_per_epoch"], meta["ckpt_epochs"], meta["seed"]) != (
            count, UPDATES_PER_EPOCH, list(GRID), 48_000 + pick
        ) or not (directory / "completion.json").is_file():
            raise ValueError("Unexpected support size or update budget in the training receipt")
        runs.append(directory)
    return runs


def measure(source, data, scratch, runs, count):
    from schedule import WEEK2_T_GRID
    from replay import load_curve_class

    output = scratch / f"n{count}-eval.npz"
    subprocess.run([
        sys.executable, "-IB", str(source / "scripts/eval_hier_oracle.py"), "--data", str(data),
        "--runs", *map(str, runs), "--out", str(output), "--device", "cuda",
        "--class-filter", "0", "--n-eval", "700", "--block-size", "5000", "--test-block", "6", "--seed", "123",
        "--epochs", ",".join(map(str, GRID)), "--guidance", "0",
    ], check=True)
    panel = load_curve_class(source)(output)
    if list(panel.picks) != list(PICKS) or panel.ep.tolist() != list(GRID) or panel.br != ["uncond"] or (
        panel.n_eval != 700 or panel.test_block != 6 or tuple(panel.tgrid) != WEEK2_T_GRID
    ):
        raise ValueError("The two clock curves use a different grid, branch or evaluation cohort")
    values = {}
    for row, (i, j) in enumerate(panel.pairs):
        pair = (panel.picks[i], panel.picks[j])
        ok, reason = panel.inter("uncond", row)
        values[pair] = (float(panel.ts("uncond", row)), "interior" if ok else reason)
    if len(values) != 15:
        raise ValueError("The fixed 15-pair comparison is incomplete")
    return values


def summarize(small, large):
    if small.keys() != large.keys():
        raise ValueError("The support conditions require matched pick pairs")
    keep = [pair for pair in small if small[pair][1] == large[pair][1] == "interior"]
    supplied = {
        "common_valid_pairs": len(keep),
        "boundary_counts": {name: {state: sum(row[1] == state for row in entries.values())
                                   for state in sorted({row[1] for row in entries.values()})}
                            for name, entries in (("833", small), ("5000", large))},
    }
    if len(keep) < MIN_COMMON:
        return supplied | {"state": "UNTESTABLE", "reason": "insufficient_matched_interior_pairs"}
    a, b = np.array(keep).T
    small_log = np.log([small[pair][0] for pair in keep])
    large_log = np.log([large[pair][0] for pair in keep])
    delta = large_log - small_log
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = []
    for indexes in rng.integers(len(PICKS), size=(BOOTSTRAP_DRAWS, len(PICKS))):
        counts = np.bincount(indexes, minlength=len(PICKS))
        weights = counts[a] * counts[b]
        if weights.sum():
            draws.append(float(np.average(delta, weights=weights)))
    if len(draws) < 0.95 * BOOTSTRAP_DRAWS:
        return supplied | {"state": "UNTESTABLE", "reason": "too_many_empty_paired_bootstraps"}
    mean = float(delta.mean())
    lo, hi = map(float, np.quantile(draws, [0.025, 0.975]))
    if mean >= MEANINGFUL_LOG_RATIO and lo > 0:
        state = "supports_count_separation"
    elif hi < 0:
        state = "opposite_count_order"
    elif hi < MEANINGFUL_LOG_RATIO:
        state = "below_reference_magnitude"
    else:
        state = "inconclusive"
    return supplied | {"state": state, "large_minus_small_log_time": mean, "ci95": [lo, hi],
                       "geom_epoch_833": float(np.exp(small_log.mean())), "geom_epoch_5000": float(np.exp(large_log.mean())),
                       "bootstrap_valid": len(draws)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "prefix", "data", "scratch"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    sys.path[:0] = [str(args.prefix), str(args.source / "scripts")]
    config = json.loads((HERE / "data.json").read_text())
    verify_data(args.data, config)
    script = args.source / "scripts/train_hier_cond_w2.py"
    source = instrument(script.read_text())
    if args.output is None:
        print(json.dumps({"data_sha256": config["data_sha256"], "support_sizes": COUNTS,
                          "grid": GRID, "updates_per_virtual_epoch": UPDATES_PER_EPOCH}, sort_keys=True))
        return
    job = os.environ.get("SLURM_JOB_ID", "")
    if not job.isascii() or not job.isdecimal() or args.output.name != job or args.output.exists():
        parser.error("Use a new output directory for the logged Slurm job")
    generated = args.source / "scripts/count_clock_trainer.py"
    generated.write_text(source)
    measured = {count: measure(args.source, args.data, args.scratch,
                                train_panel(generated, args.data, args.scratch, count), count) for count in COUNTS}
    result = {"scope": "prospective_count_only_synthetic_population", "commit": args.commit, "job_id": job,
              "formal_source_commit": FORMAL, "data_sha256": config["data_sha256"], "support_sizes": COUNTS,
              "grid": GRID, "updates_per_virtual_epoch": UPDATES_PER_EPOCH, "training_batch": BATCH,
              "comparison": summarize(measured[833], measured[5000])}
    args.output.mkdir(parents=True, exist_ok=False)
    pending = args.output / "results.partial"
    pending.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    pending.replace(args.output / "results.json")


if __name__ == "__main__":
    main()
