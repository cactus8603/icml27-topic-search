"""Quantify censoring and two-arm headroom on the original RL Gate 0 panel."""

import argparse
import json
import math
import os
from pathlib import Path
import re
import sys

import numpy as np


SOURCE = "d52d5d737906ddc21f77269eb67d54d2ab8a56e6"
SUMMARY_SHA = "b9428731116e670bb6b8bc452decb168e8341058dc353799173055e2e7469f59"
RECEIPTS = {
    "base": "d7cab55290802ac40f2bad3fe4631d3305de898cd1431eae21bd31d7aa2e5e08",
    "rl": "7d5d69298c6d7a7655f911e89fceb520123675c808cff2263d4f96cf39a0e7c8",
}
SCOPE = "pair_a_historical_censoring_and_fixed_arm_ceiling"


def match_pass_one(base_one, rl_one, base_k, rl_k):
    """Maximize coverage with query-specific weights at the same pass@1 as all RL."""
    use_rl = rl_k > base_k
    starting_one = np.where(use_rl, rl_one, base_one).sum()
    best = np.maximum(base_k, rl_k).sum()
    remaining = float(rl_one.sum() - starting_one)
    one_delta = np.where(use_rl, base_one - rl_one, rl_one - base_one)
    direction = math.copysign(1, remaining)
    eligible = np.flatnonzero(direction * one_delta > 0)
    losses = np.abs(base_k - rl_k)
    rates = losses[eligible] / np.abs(one_delta[eligible])
    for index in eligible[np.argsort(rates, kind="stable")]:
        if abs(remaining) <= 1e-12:
            break
        weight = min(abs(remaining / one_delta[index]), 1.0)
        best -= weight * losses[index]
        remaining -= weight * one_delta[index]
    if abs(remaining) > 1e-10 or best + 1e-10 < rl_k.sum():
        raise ValueError("The equal-pass@1 oracle did not include the all-RL policy")
    return float(best / len(base_one))


def minimum_restores(correct, censored, pass_k, target):
    """Select the largest available diminishing increments until hitting the target."""
    observed = pass_k[correct].sum()
    difference = target * len(correct) - observed
    if difference <= 0:
        return 0, float(observed / len(correct))
    increments = np.diff(pass_k)
    if (np.diff(increments) > 1e-12).any() or (increments < -1e-12).any():
        raise ValueError("Expected nonnegative, diminishing coverage increments")
    available = np.concatenate([increments[c:c + u] for c, u in zip(correct, censored, strict=True)])
    accumulated = np.sort(available)[::-1].cumsum()
    index = int(np.searchsorted(accumulated, difference, side="left"))
    if index == len(accumulated):
        return None, None
    return index + 1, float((observed + accumulated[index]) / len(correct))


def paired_interval(rows, draws, seed):
    rng = np.random.default_rng(seed)
    estimates = np.empty((draws, len(rows)))
    for start in range(0, draws, 512):
        size = min(512, draws - start)
        index = rng.integers(0, rows.shape[1], size=(size, rows.shape[1]))
        estimates[start:start + size] = rows[:, index].mean(axis=2).T
    return np.quantile(estimates, [0.025, 0.975], axis=0).T.tolist()


def difference_bound(observed, best, draws, seed):
    base, rl = observed
    base_high, rl_high = best
    contrasts = np.array([rl - base, rl - base_high, rl_high - base])
    return {
        "observed_rl_minus_base": float(contrasts[0].mean()),
        "identification_bounds": contrasts[1:].mean(axis=1).tolist(),
        "paired_95pct_intervals_observed_lower_upper": paired_interval(contrasts, draws, seed),
    }


def analyze(source, data, ids, published, criteria):
    ecfg, gate = criteria["eval"], criteria["gate0"]
    n, k = ecfg["n_samples"], ecfg["k_main"]
    if (len(ids), n, k) != (1053, 128, 64):
        raise ValueError("Unexpected historical panel size or sampling budget")
    correct = np.array([[sum(data[arm][key]["correct"]) for key in ids] for arm in RECEIPTS])
    unknown = np.array([[n - sum(data[arm][key]["finished"]) for key in ids] for arm in RECEIPTS])
    pass_k = source.pass_at_k_vec(np.arange(n + 1), n, k)
    one, coverage = correct / n, pass_k[correct]
    one_high, coverage_high = (correct + unknown) / n, pass_k[correct + unknown]
    for index, arm in enumerate(RECEIPTS):
        for value, expected in (
            (one[index].mean(), published[f"{arm}_p1"]),
            (coverage[index].mean(), published[f"{arm}_pk"]),
            (unknown[index].mean() / n, published["trunc_pct"][arm]),
        ):
            if not math.isclose(100 * value, expected, abs_tol=1e-10, rel_tol=0):
                raise ValueError(f"Observed values differ from the published Gate 0 summary for {arm}")
    args = ecfg["bootstrap_B"], ecfg["bootstrap_seed"]
    target = coverage[1].mean() + gate["pair_rl_pk_below_base_min_pp"] / 100
    restored, new_base_k = minimum_restores(correct[0], unknown[0], pass_k, target)
    new_p1_gap = None if restored is None else float((one[1] - one[0]).mean() - restored / (n * len(ids)))
    base_censored = int(unknown[0].sum())
    original_window = np.array(gate["pass_window_pct"]) / 100
    in_window = None if restored is None else bool(original_window[0] <= new_base_k <= original_window[1])
    return {
        "test_questions": len(ids), "samples_per_arm_per_question": n, "k_main": k,
        "censored_counts_by_arm": dict(zip(RECEIPTS, unknown.sum(axis=1).tolist(), strict=True)),
        "pass1_ranges_by_arm": dict(zip(RECEIPTS, np.stack((one.mean(1), one_high.mean(1)), axis=1).tolist(), strict=True)),
        "passk_ranges_by_arm": dict(zip(RECEIPTS, np.stack((coverage.mean(1), coverage_high.mean(1)), axis=1).tolist(), strict=True)),
        "pass1_difference": difference_bound(one, one_high, *args),
        "passk_difference": difference_bound(coverage, coverage_high, *args),
        "optimistic_base_restoration": {
            "target_base_passk": float(target), "minimum_correct_restorations": restored,
            "fraction_of_base_censored": restored / base_censored if restored is not None and base_censored else None,
            "base_passk_after_restoration": new_base_k, "base_in_registered_window": in_window,
            "rl_minus_base_pass1_after_restoration": new_p1_gap,
        },
        "fixed_n_answer_oracles": {
            "observed_rl_passk": float(coverage[1].mean()), "matched_rl_pass1": float(one[1].mean()),
            "matched_pass1_policy_oracle": match_pass_one(*one, *coverage),
            "unconstrained_policy_oracle": float(np.maximum(*coverage).mean()),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--grader-lib", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.freeze):
        parser.error("Use a full freeze commit")
    sys.path[:0] = [str(args.source / "scripts"), str(args.grader_lib)]
    import aggregate as source
    from artifacts import atomic_json, grader_receipt, read_json, sha256

    summary = args.source / "results/aggregate_gate0_A.json"
    if sha256(summary) != SUMMARY_SHA:
        raise ValueError("The published Gate 0 summary was modified")
    for arm, expected in RECEIPTS.items():
        if sha256(args.results / "A" / arm / "graded_receipt.json") != expected:
            raise ValueError(f"Pinned grading receipt differs for {arm}")
    grader_receipt()
    criteria, pairs = source.cfg("criteria.json"), source.cfg("pairs.json")
    rows = source.load_data()
    ids = [row["id"] for row in rows if row["split"] == "test"]
    gate = source.check_gate_minus1(args.results, "A", pairs)
    if str(gate["job_id"]) != "26270" or len(ids) != 1053 or len(set(ids)) != len(ids):
        raise ValueError("Historical correctness proof or question panel differs")
    identity = {"scope": SCOPE, "freeze": args.freeze, "source_commit": SOURCE, "source_summary_sha256": SUMMARY_SHA,
                "grading_receipt_sha256": RECEIPTS}
    if args.check_only:
        print(json.dumps(identity | {"stage": "offline_protocol_checked"}, sort_keys=True))
        return
    datasets = {
        arm: source.load_verified(args.results / "A" / arm, "A", arm, pairs, criteria["eval"], rows, gate)
        for arm in RECEIPTS
    }
    published = read_json(summary)["pairs"]["A"]["gate0"]
    result = analyze(source, datasets, ids, published, criteria)
    args.output.parent.mkdir(parents=True, exist_ok=False)
    atomic_json(args.output, identity | {"slurm_job_id": os.environ["SLURM_JOB_ID"], "metrics": result})
    print(json.dumps({"summary": str(args.output), "summary_sha256": sha256(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
