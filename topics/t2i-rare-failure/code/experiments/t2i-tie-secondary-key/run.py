"""Evaluate secondary keys within tied score groups on an exact Gaussian event."""

import argparse
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
import json
import math
import os
from pathlib import Path
import re
from statistics import mean, median

import numpy as np
import scipy
from scipy.special import ndtr


HERE = Path(__file__).resolve().parent
PREVIOUS_COMMIT = "bf261c35a5296f6625e8dff889e76997bfb3adfa"
PREVIOUS_SHA = "51d38d151db4788bd8b5179752ab3985cf8b353eb44a8706a0aefbbebf0f1d92"
previous_source = HERE.parent / "t2i-tied-scores/run.py"
if sha256(previous_source.read_bytes()).hexdigest() != PREVIOUS_SHA:
    raise ValueError("Earlier score definitions differ from the frozen reference")
spec = spec_from_file_location("registered_tied_scores", previous_source)
reference = module_from_spec(spec)
spec.loader.exec_module(reference)

SEEDS = range(2026111640, 2026111672)
PARAMS = reference.PARAMS | {"d": 2}
PROFILES = ("monotone_quantized", "bulk_zero", "binary")
ARMS = (("continuous", False), *((name, secondary) for name in PROFILES for secondary in (False, True)))
SCOPE = "synthetic_tie_secondary_key_only"


def encode_unit_interval(lower, uniform):
    """Keep the represented score in [lower, lower+1), even when CDF rounds to one."""
    upper = np.nextafter(lower + 1.0, lower)
    return np.minimum(lower + uniform, upper)


def score(z, profile, secondary):
    primary = reference.scores(z, profile)
    if not secondary:
        return primary
    if profile == "bulk_zero":
        tied = primary == 0.0
        primary[tied] = encode_unit_interval(np.full(tied.sum(), -1.0), ndtr(z[tied, 1]))
        return primary
    if profile in ("monotone_quantized", "binary"):
        return encode_unit_interval(primary.astype(np.float64, copy=False), ndtr(z[:, 1]))
    raise ValueError(f"No secondary arm for {profile}")


def verify_geometry():
    truth = reference.normal_tail(reference.CUTOFF)
    if not math.isclose(truth, reference.PROBABILITY, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("Analytic normal survival differs from the registered truth")
    if set(SEEDS) & (set(reference.PRIOR_BINARY_SEEDS) | set(reference.DISCLOSED_SEEDS) | set(reference.SEEDS)):
        raise ValueError("New replicates overlap an earlier run")
    if len(SEEDS) != 32 or PARAMS["d"] != 2:
        raise ValueError("Paired experimental allocation differs")

    cutoff = reference.CUTOFF
    neighbors = [np.nextafter(cutoff, -np.inf), cutoff, np.nextafter(cutoff, np.inf)]
    grid = np.sort(np.r_[np.linspace(-10.0, 10.0, 4001), neighbors, reference.BULK_FLOOR])
    for secondary_z in (-64.0, -3.0, 0.0, 3.0, 64.0):
        z = np.column_stack((grid, np.full(len(grid), secondary_z)))
        for profile, secondary in ARMS:
            transformed = score(z, profile, secondary)
            hit = transformed >= reference.THRESHOLDS[profile]
            if not np.array_equal(hit, grid >= cutoff) or not np.all(np.diff(transformed) >= 0):
                raise ValueError(f"Score changed the target event or ordering: {profile}/{secondary}")
    return truth


def one_run(estimator, profile, secondary, seed):
    used = 0
    first = None

    def score_fn(z):
        nonlocal used, first
        if used == 0:
            first = sha256(z.tobytes()).hexdigest()
        used += len(z)
        return score(z, profile, secondary)

    estimate, calls, diagnostic = estimator(
        score_fn, t=reference.THRESHOLDS[profile], rng=np.random.default_rng(seed), **PARAMS,
    )
    finite = math.isfinite(estimate)
    exhausted = diagnostic.get("failed", False)
    levels = diagnostic["levels"]
    if (used != calls or not 0 <= levels <= PARAMS["max_levels"] or exhausted == finite
            or exhausted != (levels == PARAMS["max_levels"]) or (finite and not 0 <= estimate <= 1)):
        raise ValueError("Observed evaluation steps differ from the estimator diagnostics")
    return {"estimate": estimate if finite else None, "levels": levels, "calls": calls,
            "exhausted": exhausted, "initial_sha256": first}


def paired_repair(original, secondary, truth):
    rows = list(zip(original, secondary, strict=True))
    finite_delta = [(b["estimate"] - a["estimate"]) / truth for a, b in rows
                    if a["estimate"] is not None and b["estimate"] is not None]
    return {
        "paired_finite": len(finite_delta),
        "mean_relative_estimate_delta_when_both_finite": mean(finite_delta) if finite_delta else None,
        "original_exhausted_secondary_finite": sum(a["exhausted"] and b["estimate"] is not None for a, b in rows),
        "secondary_exhausted_original_finite": sum(b["exhausted"] and a["estimate"] is not None for a, b in rows),
        "median_score_call_delta_all_seeds": median(b["calls"] - a["calls"] for a, b in rows),
    }


def experiment(estimator, truth):
    arms = {key: [] for key in ARMS}
    for seed in SEEDS:
        batch = {key: one_run(estimator, *key, seed) for key in arms}
        if len({row["initial_sha256"] for row in batch.values()}) != 1:
            raise ValueError("Paired arms did not start from the same joint Gaussian draws")
        for key, row in batch.items():
            arms[key].append(row)
    baseline = arms["continuous", False]
    return {
        "continuous": reference.summarize(baseline, baseline, truth),
        "profiles": {
            profile: {
                "original": reference.summarize(arms[profile, False], baseline, truth),
                "secondary": reference.summarize(arms[profile, True], baseline, truth),
                "paired_secondary_minus_original": paired_repair(arms[profile, False], arms[profile, True], truth),
            }
            for profile in PROFILES
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--source", type=Path, default=HERE.parents[2] / reference.SOURCE)
    parser.add_argument("--freeze")
    args = parser.parse_args()
    job = os.environ.get("SLURM_JOB_ID", "")
    if args.run and (not re.fullmatch(r"[0-9a-f]{40}", args.freeze or "") or not re.fullmatch(r"[1-9][0-9]*", job)):
        parser.error("Run the frozen experiment through Slurm")
    estimator, truth = reference.load_estimator(args.source), verify_geometry()
    receipt = {"scope": SCOPE, "source_commit": reference.SOURCE_COMMIT, "source_sha256": reference.SOURCE_SHA,
               "score_reference_commit": PREVIOUS_COMMIT, "score_reference_sha256": PREVIOUS_SHA,
               "event_probability": truth, "z_cutoff": reference.CUTOFF, "base_thresholds": reference.THRESHOLDS,
               "bulk_zero_mass": 1.0 - reference.normal_tail(reference.BULK_FLOOR),
               "quantization_width": reference.BIN_WIDTH, "estimator_parameters": PARAMS,
               "seed_range": [SEEDS.start, SEEDS.stop], "replicates_per_arm": len(SEEDS), "arms": len(ARMS),
               "numpy": np.__version__, "scipy": scipy.__version__}
    if args.run:
        receipt.update(frozen_commit=args.freeze, slurm_job_id=job, comparison=experiment(estimator, truth))
    print(json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
