"""Run a paired CPU stress test of tied scores with an exact normal tail."""

import argparse
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
import json
import math
import os
from pathlib import Path
import re
from statistics import NormalDist, mean, median

import numpy as np


HERE = Path(__file__).resolve().parent
SOURCE = Path("killtests/t2i-rare-failure/scripts/estimators.py")
SOURCE_COMMIT = "b54a86fb05e845e2c11139822beac8846c63c77a"
SOURCE_SHA = "689d07b300a1e1c228bb0ddb3663ab178a06b0f448c3d6f8c497f9280e69afda"
PRIOR_BINARY_SEEDS = range(4)
DISCLOSED_SEEDS = range(2026111550, 2026111582)
SEEDS = range(2026111600, 2026111632)
PROBABILITY = 0.001
CUTOFF = NormalDist().inv_cdf(1.0 - PROBABILITY)
BIN_WIDTH = 0.25
BULK_FLOOR = 2.0
PARAMS = {"d": 1, "N": 1000, "p0": 0.1, "mcmc_steps": 4, "beta0": 0.5, "max_levels": 30}


def scores(z, profile):
    x = z[:, 0]
    if profile == "continuous":
        return x
    if profile == "monotone_quantized":
        return np.floor((x - CUTOFF) / BIN_WIDTH)
    if profile == "bulk_zero":
        return np.maximum(0.0, x - BULK_FLOOR)
    if profile == "binary":
        return (x >= CUTOFF).astype(np.float32)
    raise ValueError(f"Unknown score profile: {profile}")


THRESHOLDS = {"continuous": CUTOFF, "monotone_quantized": 0.0, "bulk_zero": CUTOFF - BULK_FLOOR, "binary": 1.0}


def normal_tail(cutoff):
    return 0.5 * math.erfc(cutoff / math.sqrt(2.0))


def verify_geometry():
    truth = normal_tail(CUTOFF)
    if not math.isclose(truth, PROBABILITY, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("Analytic Gaussian survival probability differs")
    if set(SEEDS) & (set(PRIOR_BINARY_SEEDS) | set(DISCLOSED_SEEDS)):
        raise ValueError("Confirmation replicates overlap a disclosed exploratory batch")
    grid = np.concatenate((np.linspace(-10.0, 10.0, 10001), (
        np.nextafter(CUTOFF, -np.inf), CUTOFF, np.nextafter(CUTOFF, np.inf),
        np.nextafter(BULK_FLOOR, -np.inf), BULK_FLOOR, np.nextafter(BULK_FLOOR, np.inf),
    )))
    grid.sort()
    for profile, threshold in THRESHOLDS.items():
        transformed = scores(grid[:, None], profile)
        if not np.all(np.diff(transformed) >= 0) or not np.array_equal(transformed >= threshold, grid >= CUTOFF):
            raise ValueError(f"Transformed scoring changes the exact event or monotonicity: {profile}")
    return truth


def load_estimator(source):
    if sha256(source.read_bytes()).hexdigest() != SOURCE_SHA:
        raise ValueError("SuS v2 source differs from the pinned estimator")
    spec = spec_from_file_location("registered_estimators", source)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.subset_simulation_v2


def one_run(estimate, profile, seed):
    n_scored = 0
    initial_sha = None

    def score_fn(z):
        nonlocal n_scored, initial_sha
        if n_scored == 0:
            initial_sha = sha256(z.tobytes()).hexdigest()
        n_scored += len(z)
        return scores(z, profile)

    value, calls, diagnostics = estimate(
        score_fn, t=THRESHOLDS[profile], rng=np.random.default_rng(seed), **PARAMS,
    )
    finite = math.isfinite(value)
    exhausted = diagnostics.get("failed", False)
    levels = diagnostics["levels"]
    if (n_scored != calls or not 0 <= levels <= PARAMS["max_levels"]
            or exhausted != (levels == PARAMS["max_levels"]) or exhausted == finite
            or (finite and not 0 <= value <= 1)):
        raise ValueError("Estimator diagnostics do not match the scored batches")
    return {"estimate": value if finite else None, "calls": calls, "levels": levels,
            "exhausted": exhausted, "initial_sha256": initial_sha}


def summarize(rows, continuous, truth):
    valid = [r["estimate"] for r in rows if r["estimate"] is not None]
    pairs = [(r["estimate"], ref["estimate"]) for r, ref in zip(rows, continuous, strict=True)
             if r["estimate"] is not None and ref["estimate"] is not None]
    calls, levels = [r["calls"] for r in rows], [r["levels"] for r in rows]
    rel_bias = mean(valid) / truth - 1.0 if valid else None
    return {
        "finite_replicates": len(valid), "exhausted_replicates": sum(r["exhausted"] for r in rows),
        "relative_bias_finite_replicates": rel_bias,
        "relative_bias_all_replicates": rel_bias if len(valid) == len(rows) else None,
        "paired_finite_with_continuous": len(pairs),
        "paired_mean_estimate_delta_over_truth": mean((x - y) / truth for x, y in pairs) if pairs else None,
        "score_calls": {"min": min(calls), "median": median(calls), "max": max(calls), "total": sum(calls)},
        "levels": {"min": min(levels), "median": median(levels), "max": max(levels)},
    }


def experiment(estimate, truth):
    arms = {profile: [] for profile in THRESHOLDS}
    for seed in SEEDS:
        rows = {profile: one_run(estimate, profile, seed) for profile in arms}
        if len({row["initial_sha256"] for row in rows.values()}) != 1:
            raise ValueError("Arms did not start from the same Gaussian draws")
        for profile, row in rows.items():
            arms[profile].append(row)
    return {profile: summarize(rows, arms["continuous"], truth) for profile, rows in arms.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--source", type=Path, default=HERE.parents[2] / SOURCE)
    parser.add_argument("--freeze")
    args = parser.parse_args()
    job = os.environ.get("SLURM_JOB_ID", "")
    if args.run and (not re.fullmatch(r"[0-9a-f]{40}", args.freeze or "") or not re.fullmatch(r"[1-9][0-9]*", job)):
        parser.error("Run confirmation through Slurm with the full committed freeze")
    estimate, truth = load_estimator(args.source), verify_geometry()
    output = {
        "scope": "synthetic_score_tie_diagnostic_only",
        "source_commit": SOURCE_COMMIT, "source_sha256": SOURCE_SHA, "numpy": np.__version__,
        "event_probability": truth, "normal_cutoff": CUTOFF, "bulk_zero_mass": 1.0 - normal_tail(BULK_FLOOR),
        "quantization_width": BIN_WIDTH, "score_thresholds": THRESHOLDS, "estimator_parameters": PARAMS,
        "prior_binary_seed_range": [PRIOR_BINARY_SEEDS.start, PRIOR_BINARY_SEEDS.stop],
        "disclosed_exploratory_seed_range": [DISCLOSED_SEEDS.start, DISCLOSED_SEEDS.stop],
        "seed_range": [SEEDS.start, SEEDS.stop], "replicates_per_profile": len(SEEDS),
    }
    if args.run:
        output.update(freeze=args.freeze, slurm_job_id=job)
        output["arms"] = experiment(estimate, truth)
    print(json.dumps(output, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
