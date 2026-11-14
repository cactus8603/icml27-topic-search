"""Replay the three frozen controls to compare their early epoch grids."""

import argparse
from collections import Counter
from itertools import combinations
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


ROLES = ("k6c0", "k1c0", "u833c0")
GROUPS = {"c": ("k6c0", "cond"), "u5000": ("k1c0", "cond"), "u833": ("u833c0", "uncond")}
PICKS = list(range(6))
PAIRS = np.array(list(combinations(PICKS, 2)))
LEGACY_COMMIT = "df7cdb280888512e9006f61f642ad5caf479603c"
PREFIX_COMMIT = "f292d94aa65da583ff5a787eed457d3f429cebb1"
DEPENDENCY_COMMIT = "df08bc634274f78ba1b99e83b07228b4252608ce"


def original_runs(checkpoints):
    week2 = checkpoints / "week2-9e92804"
    return {
        "k6c0": [(checkpoints if pick < 3 else week2) / "hier" / f"pick{pick}" for pick in PICKS],
        "k1c0": [week2 / "hier_k1" / f"pick{pick}" for pick in PICKS],
        "u833c0": [week2 / "hier_u833" / f"pick{pick}" for pick in PICKS],
    }


def preflight(replay, source, originals, data):
    from schedule import week2_epochs
    from week2_identity import expected_eval, read_runs

    data_sha = replay.sha256(data)
    mask = replay.read_json(source / "configs/t_mask_hier.json")
    if data_sha != mask["data_sha256"]:
        raise ValueError("Hierarchical data checksum differs from the formal run")
    evidence = {}
    for role, runs in originals.items():
        kind = expected_eval(role)["run_type"]
        metas = [replay.read_json(path / "meta.json") for path in runs]
        if [meta["pick"] for meta in metas] != PICKS:
            raise ValueError(f"{role}: training cohort differs")
        if role == "u833c0" and any(meta["n_c"] < 700 for meta in metas):
            raise ValueError(f"{role}: class-0 sample count is below evaluation requirement")
        evidence[role] = read_runs(runs, metas, kind, data, data_sha)
        for run in runs:
            for epoch in week2_epochs():
                weight = run / f"ep{epoch}.pt"
                if not weight.is_file() or weight.stat().st_size == 0:
                    raise ValueError(f"{role}/{run.name}: formal checkpoint {epoch} missing")
    return data_sha, evidence


def prepare_trainers(replay, source, legacy):
    trainers = {}
    for key, folder, name in (
        ("legacy", legacy, "train_hier_cond.py"),
        ("normal", source, "train_hier_cond.py"),
        ("control", source, "train_hier_cond_w2.py"),
    ):
        trainer = folder / "scripts" / name
        dst = trainer.with_name(f"prefix_{name}")
        trainers[key] = (dst, replay.prefix_trainer(trainer.read_text(encoding="utf-8")))
    return trainers


def training_args(role, pick, trainers):
    if role == "k6c0":
        return trainers["legacy" if pick < 3 else "normal"][0], ["--pdrop", "0.1", "--batch", "512"]
    if role == "k1c0":
        return trainers["control"][0], ["--pdrop", "0.1", "--batch", "512", "--const-label", "0"]
    if role == "u833c0":
        return trainers["control"][0], ["--pdrop", "1.0", "--batch", "85", "--class-only", "0"]
    raise ValueError(f"Unknown training role: {role}")


def replay_prefixes(replay, trainers, work, data, originals):
    from schedule import week2_epochs

    for path, text in trainers.values():
        path.write_text(text, encoding="utf-8")
    receipts, views = {}, {}
    for role in ROLES:
        receipts[role], views[role] = [], []
        cohort_dir = work / role
        cohort_dir.mkdir()
        for pick, original in zip(PICKS, originals[role], strict=True):
            script, flags = training_args(role, pick, trainers)
            prefix = cohort_dir / f"prefix-{pick}"
            subprocess.run([
                sys.executable, "-IB", str(script), "--data", str(data), "--out", str(prefix),
                "--pick", str(pick), "--seed", str(pick), "--n", "5000", "--epochs", "20000",
                "--device", "cuda", *flags,
            ], check=True, stdout=subprocess.DEVNULL)
            checks = {
                str(epoch): replay.compare_anchor(original / f"ep{epoch}.pt", prefix / f"ep{epoch}.pt")
                for epoch in replay.ANCHORS
            }
            receipts[role].append({"pick": pick, "checks": checks})
            if not all(item["match"] for item in checks.values()):
                return receipts, None
            view = cohort_dir / f"view-{pick}"
            view.mkdir()
            (view / "meta.json").symlink_to(original / "meta.json")
            for epoch in week2_epochs():
                (view / f"ep{epoch}.pt").symlink_to(original / f"ep{epoch}.pt")
            for epoch in replay.PREFIX_EPOCHS:
                if epoch < replay.ANCHORS[0]:
                    (view / f"ep{epoch}.pt").symlink_to(prefix / f"ep{epoch}.pt")
            views[role].append(view)
            print(f"[codex] {role} pick {pick}: three exact checkpoint anchors verified", flush=True)
    return receipts, views


def validate_panel(run, role, grid, t_grid, expected):
    guidance = expected["guidance"]
    branches = ["uncond"] if role == "u833c0" else ["cond", "uncond", *[f"cfg{g:g}" for g in guidance]]
    if (run.picks != PICKS or not np.array_equal(run.pairs, PAIRS) or run.ep.tolist() != grid
            or run.tgrid != list(t_grid) or run.br != branches or run.n_eval != expected["n_eval"]
            or run.test_block != expected["test_block"] or run.guidance != guidance):
        raise ValueError(f"{role}: evaluated cohort, checkpoint grid or parameters differ")


def evaluate_curves(replay, source, work, data, views):
    from schedule import WEEK2_T_GRID, week2_epochs
    from week2_identity import expected_eval

    formal_grid = week2_epochs()
    grid = sorted(set(replay.PREFIX_EPOCHS).union(formal_grid))
    indices = [grid.index(epoch) for epoch in formal_grid]
    ordinary = replay.load_curve_class(source)
    fixed_norm = replay.load_curve_class(source, normalization_indices=indices)
    panels = {"original_grid": {}, "extended_grid": {}, "extended_grid_original_normalizers": {}}
    for role in ROLES:
        expected = expected_eval(role)
        out = work / f"{role}.npz"
        subprocess.run([
            sys.executable, "-IB", str(source / "scripts/eval_hier_oracle.py"),
            "--data", str(data), "--runs", *(str(view) for view in views[role]),
            "--device", "cuda", "--epochs", ",".join(map(str, grid)),
            "--class-filter", str(expected["class_filter"]), "--n-eval", str(expected["n_eval"]),
            "--test-block", str(expected["test_block"]), "--block-size", str(expected["block_size"]),
            "--seed", str(expected["eval_seed"]), "--guidance", ",".join(map(str, expected["guidance"])) or "0",
            "--out", str(out),
        ], check=True, stdout=subprocess.DEVNULL)
        ext = ordinary(out)
        validate_panel(ext, role, grid, WEEK2_T_GRID, expected)
        original_out = work / f"{role}_original_grid.npz"
        with np.load(out, allow_pickle=False) as data_rows:
            fields = {key: data_rows[key] for key in ("branches", "pairs", "picks", "t_grid",
                                                       "guidance", "n_eval", "test_block")}
            np.savez(original_out, epochs=formal_grid, pair_cos_eps=data_rows["pair_cos_eps"][:, indices], **fields)
        original = ordinary(original_out)
        fixed = fixed_norm(out)
        if role == "u833c0":
            first = {tuple(pair) for i, pair in enumerate(original.pairs) if original.inter("uncond", i)[1] == "first"}
            if first != replay.FORMAL_FIRST:
                raise ValueError("Recomputed original-grid u833 first-boundary pairs differ from the known audit")
        for branch in original.br:
            if not np.allclose(fixed.curve[branch][:, indices], original.curve[branch],
                               rtol=1e-12, atol=1e-12, equal_nan=True):
                raise ValueError(f"{role}: fixed-scale curves do not reconstruct the original grid")
        panels["original_grid"][role] = original
        panels["extended_grid"][role] = ext
        panels["extended_grid_original_normalizers"][role] = fixed
        print(f"[codex] {role}: evaluated matched cohort and both normalizations", flush=True)
    return panels


def positive_control(panels, criteria):
    """Apply the original positive-control comparison to one exploratory shared grid."""
    valid, log_times, counts = {}, {}, {}
    for group, (role, branch) in GROUPS.items():
        run = panels[role]
        states = [run.inter(branch, p) for p in range(len(PAIRS))]
        valid[group] = np.flatnonzero([ok for ok, _ in states])
        log_times[group] = np.log([run.ts(branch, p) for p in range(len(PAIRS))])
        counts[group] = dict(Counter("interior" if ok else reason for ok, reason in states))
    limit = len(PAIRS) // criteria["max_invalid_pair_divisor"]
    result = {
        "pair_counts": counts, "n_pairs": len(PAIRS),
        "n_common_valid_pairs": len(set.intersection(*(set(pairs) for pairs in valid.values()))),
        "max_invalid_pairs_per_arm": limit,
    }
    if any(len(PAIRS) - len(group_pairs) > limit for group_pairs in valid.values()):
        return {**result, "original_rule_diagnostic": "insufficient_valid_pairs"}

    def statistic(weights):
        means = {}
        for arm, indices in valid.items():
            selected = weights[indices]
            if selected.sum() == 0:
                return None, None
            means[arm] = float((selected * log_times[arm][indices]).sum() / selected.sum())
        denom = means["u833"] - means["u5000"]
        return means, (means["c"] - means["u5000"]) / denom if denom else None

    means, effect = statistic(np.ones(len(PAIRS)))
    rng = np.random.default_rng(criteria["bootstrap_seed"])
    draws = []
    for _ in range(criteria["n_boot"]):
        picks = np.bincount(rng.integers(0, len(PICKS), len(PICKS)), minlength=len(PICKS)).astype(float)
        _, estimate = statistic(picks[PAIRS[:, 0]] * picks[PAIRS[:, 1]])
        if estimate is not None:
            draws.append(estimate)
    interval = np.percentile(draws, [2.5, 97.5]).tolist() if draws else [None, None]
    separation = means["u5000"] - means["u833"]
    distance = abs(means["c"] - means["u833"])
    original = criteria["positive_control"]
    if separation < original["discrimination_min"]:
        diagnostic = "insufficient_control_separation"
    elif not draws:
        diagnostic = "no_valid_bootstrap_draw"
    elif interval[0] >= original["lambda_ci_lower_min"] and distance <= original["abs_Lc_minus_Lu833_max"]:
        diagnostic = "meets_original_cutoffs"
    else:
        diagnostic = "misses_original_cutoffs"
    return {**result, "original_rule_diagnostic": diagnostic, "L": means, "lambda": effect,
            "lambda_ci95": interval, "bootstrap_skipped": criteria["n_boot"] - len(draws),
            "separation_Lu5000_minus_Lu833": separation, "abs_Lc_minus_Lu833": distance}


def write_report(output, result):
    pending = output / ".results.json.pending"
    pending.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(pending, output / "results.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "legacy", "prefix-source", "data", "checkpoints", "work"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    sys.path[:0] = [str(args.prefix_source.resolve()), str(source / "scripts")]
    import replay

    originals = original_runs(args.checkpoints.resolve())
    data = args.data.resolve()
    data_sha, sources = preflight(replay, source, originals, data)
    trainers = prepare_trainers(replay, source, args.legacy.resolve())
    if args.output is None:
        print("[codex] frozen trainer variants, data and 18 checkpoint panels verified")
        return

    import torch

    if not torch.cuda.is_available() or "H200" not in torch.cuda.get_device_name(0):
        raise SystemExit("replay requires one H200 to match the original training backend")
    args.output.mkdir(parents=True, exist_ok=False)
    result = {
        "status": "coverage_pending", "scope": "exploratory_same_cohort_early_grid",
        "commit": args.commit, "formal_commit": replay.FORMAL_COMMIT, "legacy_commit": LEGACY_COMMIT,
        "prefix_commit": PREFIX_COMMIT, "dependency_commit": DEPENDENCY_COMMIT,
        "slurm_job_id": os.environ["SLURM_JOB_ID"], "data_sha256": data_sha,
        "torch_version": torch.__version__, "gpu": torch.cuda.get_device_name(0),
        "training_sources": {role: [s["meta_sha256"] for s in rows] for role, rows in sources.items()},
        "picks": PICKS, "anchor_epochs": list(replay.ANCHORS), "prefix_epochs": list(replay.PREFIX_EPOCHS),
    }
    write_report(args.output, result)
    anchors, views = replay_prefixes(replay, trainers, args.work, data, originals)
    result["anchors"] = anchors
    if views is None:
        result["status"] = "checkpoint_anchor_mismatch"
        write_report(args.output, result)
        raise SystemExit(2)
    result["status"] = "all_anchors_verified_metrics_pending"
    write_report(args.output, result)
    criteria = replay.read_json(source / "configs/criteria_week2.json")
    panels = evaluate_curves(replay, source, args.work, data, views)
    result["comparison"] = {grid: positive_control(runs, criteria) for grid, runs in panels.items()}
    result["status"] = "exploratory_only"
    write_report(args.output, result)


if __name__ == "__main__":
    main()
