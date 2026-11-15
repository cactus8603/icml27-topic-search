#!/usr/bin/env python3
"""Decompose the published CFG3 gap using the stored model × input-block matrix."""

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np

from reach import membership_gap, oracle_errors
from schedule import WEEK2_T_GRID, week2_epochs
from week2_identity import manifest_error


SOURCE_JOB = 26383
SOURCE_FREEZE = "9e92804a7bfded3c5cafd4fccd4315df2034eca5"
NPZ_SHA256 = "4f33e24a7a332358cf8d098d3877f517ae1e5be5687008222b77bbdee1f9aaea"
VERDICT_SHA256 = "b2af1ce8a25513a15ad5b7aacb811336b533a40dd6b594cc33743af0a5689551"
TOPIC = Path("killtests/biased-gen-cfg")
ARCHIVE_ROOT = Path(__file__).resolve().parents[3]
PICKS = list(range(6))
BRANCHES = ["cond", "uncond", "cfg1", "cfg3"]


def verify_sha256(path, expected):
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch: {path}")


def checkpoint_index(ce, smooth_window):
    """Use the original CE average over six runs and all eight noise times."""
    heldout_cond = ce[:, :, :, -1, BRANCHES.index("cond")].mean(axis=(0, 2))
    if not len(heldout_cond) or not np.isfinite(heldout_cond).all():
        raise ValueError("Held-out conditional CE is empty or nonfinite")
    half_window = smooth_window // 2
    smoothed = [
        heldout_cond[max(0, index - half_window):index + half_window + 1].mean()
        for index in range(len(heldout_cond))
    ]
    index = int(np.argmin(smoothed))
    if index == 0:
        raise ValueError("The original checkpoint choice is first-grid censored")
    return index


def matched_summary(block_errors, heldout_errors, n_boot, seed, level, min_gap, fail_upper):
    """Rows are models, columns are blocks in the same pick order; all t are summed."""
    block_errors = np.asarray(block_errors, dtype=float)
    heldout_errors = np.asarray(heldout_errors, dtype=float)
    n = len(heldout_errors)
    if n < 2 or heldout_errors.shape != (n,) or block_errors.shape != (n, n):
        raise ValueError("The matched model × input-block table is incomplete")
    if (not np.isfinite(block_errors).all() or not np.isfinite(heldout_errors).all()
            or (block_errors < 0).any() or (heldout_errors < 0).any()):
        raise ValueError("MSE values must be finite and nonnegative")

    i, j = np.triu_indices(n, 1)
    own = (block_errors[i, i] + block_errors[j, j]) / 2
    other = (block_errors[i, j] + block_errors[j, i]) / 2
    heldout = (heldout_errors[i] + heldout_errors[j]) / 2
    means = np.asarray([own.mean(), other.mean(), heldout.mean()])
    if (means[1:] <= 0).any():
        raise ValueError("The off-diagonal and held-out denominators must be positive")

    def ratios(own_mse, other_mse, heldout_mse):
        return {
            "h_matched": (other_mse - own_mse) / other_mse,
            "member_component": (other_mse - own_mse) / heldout_mse,
            "heldout_component": (heldout_mse - other_mse) / heldout_mse,
            "g3_same_weights": (heldout_mse - own_mse) / heldout_mse,
        }

    point = ratios(*means)
    draws = np.random.default_rng(seed).integers(0, n, (n_boot, n))
    counts = (draws[:, :, None] == np.arange(n)).sum(axis=1)
    pair_weights = counts[:, i] * counts[:, j]
    totals = pair_weights.sum(axis=1)
    draws_mse = pair_weights @ np.column_stack((own, other, heldout))
    valid = (totals > 0) & (draws_mse[:, 1:] > 0).all(axis=1) & np.isfinite(draws_mse).all(axis=1)
    kept = int(valid.sum())
    intervals = None
    reference = "untestable_resampling"
    if kept >= math.ceil(0.95 * n_boot):
        fractions = ratios(*draws_mse[valid].T)
        tail = 50 * (1 - level)
        intervals = {
            name: np.percentile(values, [tail, 100 - tail]).tolist()
            for name, values in fractions.items()
        }
        lo, hi = intervals["h_matched"]
        if point["h_matched"] >= min_gap and lo > 0:
            reference = "reaches_reference_magnitude"
        elif hi < fail_upper:
            reference = "below_reference_magnitude"
        else:
            reference = "between_reference_magnitudes"

    return {
        "own_mse_sum_t_pair_average": float(means[0]),
        "other_mse_sum_t_pair_average": float(means[1]),
        "heldout_mse_sum_t_pair_average": float(means[2]),
        "point": {name: float(value) for name, value in point.items()},
        "pair_bootstrap_ci": intervals,
        "pair_count": len(i),
        "bootstrap_draws": n_boot,
        "bootstrap_skipped": n_boot - kept,
        "matched_magnitude_reference": reference,
    }


def compute(npz_path, verdict_path, criteria, mask):
    published = json.loads(verdict_path.read_text())["report60"]["headroom"]
    epochs = week2_epochs()
    with np.load(npz_path, allow_pickle=False) as data:
        if (data["epochs"].tolist() != epochs or data["picks"].tolist() != PICKS
                or tuple(data["t_grid"].tolist()) != WEEK2_T_GRID
                or data["branches"].tolist() != BRANCHES
                or data["test_block"].item() != 7 or data["n_eval"].item() != 1000
                or data["guidance"].tolist() != [1.0, 3.0]):
            raise ValueError("The NPZ is outside the original Week 2 K=6 evaluation design")
        manifest = json.loads(data["week2_manifest"].item())
        error = manifest_error(manifest, "k6", PICKS, mask["data_sha256"])
        if error:
            raise ValueError(error)
        ce = data["ce"]
        mse = data["mse_oracle"]
    shape = (len(PICKS), len(epochs), len(WEEK2_T_GRID), len(PICKS) + 1, len(BRANCHES))
    if ce.shape != shape or mse.shape != shape:
        raise ValueError(f"Unexpected CE/MSE array shape; expected {shape}")
    ei = checkpoint_index(ce, criteria["smooth_window"])
    keep = mask["keep_t"]
    if published["e_dagger"] != epochs[ei] or published["t_used"] != keep:
        raise ValueError("The recomputed e-dagger or t mask differs from the original verdict")
    t_idx = [WEEK2_T_GRID.index(t) for t in keep]
    cfg3 = BRANCHES.index("cfg3")
    own, heldout = oracle_errors(mse, ei, t_idx)
    config = criteria["candidate_screen"]
    settings = (
        config["headroom_n_boot"], config["headroom_seed"], config["headroom_ci_level"],
        config["headroom_min"], config["headroom_upper_fail"],
    )
    original = membership_gap(own[:, :, cfg3], heldout[:, :, cfg3], *settings)
    if (original["state"] != published["state"]
            or not np.allclose(original["ci"], published["ci"], rtol=1e-12, atol=1e-15)
            or not all(math.isclose(original[field], published[field], rel_tol=1e-12, abs_tol=1e-15)
                       for field in ("g3", "heldout_mse_sum", "raw_gap_sum"))):
        raise ValueError("The published G3 did not reproduce")
    table = mse[:, ei, :, :, cfg3][:, t_idx, :].sum(axis=1)
    matched = matched_summary(table[:, :-1], table[:, -1], *settings)
    if not math.isclose(matched["point"]["g3_same_weights"], original["g3"], rel_tol=1e-12, abs_tol=1e-15):
        raise ValueError("The new decomposition did not preserve the published G3 point estimate")
    return {"e_dagger": epochs[ei], "e_dagger_is_last_grid": ei == len(epochs) - 1,
            "t_used": keep, "picks": PICKS, "original": original, "matched": matched}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--freeze", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-f0-9]{40}", args.freeze):
        parser.error("Use the full freeze commit")
    npz_path = args.source / "hier_oracle_w2_k6.npz"
    verdict_path = args.source / "week2_verdict.json"
    verify_sha256(npz_path, NPZ_SHA256)
    verify_sha256(verdict_path, VERDICT_SHA256)
    lineage = {"freeze": args.freeze, "source_job": SOURCE_JOB, "source_freeze": SOURCE_FREEZE,
               "npz_sha256": NPZ_SHA256, "verdict_sha256": VERDICT_SHA256}
    if args.check_only:
        print(json.dumps(lineage, sort_keys=True))
        return
    configs = ARCHIVE_ROOT / TOPIC / "configs"
    criteria = json.loads((configs / "criteria_week2.json").read_text())
    mask = json.loads((configs / "t_mask_hier.json").read_text())
    result = compute(npz_path, verdict_path, criteria, mask)
    verify_sha256(npz_path, NPZ_SHA256)
    verify_sha256(verdict_path, VERDICT_SHA256)
    args.output.parent.mkdir(parents=True, exist_ok=False)
    body = {"scope": "same_panel_exploratory_input_block_decomposition", **lineage, **result}
    with args.output.open("x") as stream:
        json.dump(body, stream, allow_nan=False, indent=2, sort_keys=True)
        stream.write("\n")
    with args.output.open("rb") as stream:
        print(f"[codex] saved {args.output} SHA-256 {hashlib.file_digest(stream, 'sha256').hexdigest()}")


if __name__ == "__main__":
    main()
