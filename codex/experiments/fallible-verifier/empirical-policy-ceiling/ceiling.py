"""Descriptive cost frontier on the sealed GSM8K candidate panel."""

import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import socket
import sys
from zoneinfo import ZoneInfo

import numpy as np
from scipy.optimize import linprog

TOPIC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOPIC / "holdout-evaluation"))
from run_evaluation import CONFIG, REPORT_ROOT, SPLITS, assay, checked_scoring_jobs, load_policy
from evaluation import coverage, fit_round2_binary, ingest_split, reference
from receipt_contract import checked_bytes, load_json


SOURCE_FREEZE = "ac3950820be8c7f4f1528ecc553b9cf51aad894b"
SOURCE_JOB = "26599"
SOURCE_SHA = "18ecffa123af2c68a66d5e265bcb6e66f66fa6ff0d3b219abb5adc3e431e4cb9"
SCORING_JOBS = {"train": "26597", "test": "26598"}
REPORT_KEY = f"{reference.CASCADE}_vs_{reference.CONTROL}_amortized"
SCOPE = "locked_panel_empirical_point_risk_tree_family_ceiling"
RISK_NAMES = ("fa", "fr")
TOL = 1e-8


def read_inputs():
    criteria = CONFIG.read_bytes()
    criteria_sha = assay.prepare.ROUND2_CONFIG_SHA256
    checked_bytes(criteria, criteria_sha)
    source_path = REPORT_ROOT / SOURCE_JOB / "summary.json"
    if source_path.is_symlink():
        raise ValueError("Use the original evaluation artifact")
    source_bytes = source_path.read_bytes()
    checked_bytes(source_bytes, SOURCE_SHA)
    old = load_json(source_bytes)
    profile, bounds, binding = load_policy(
        TOPIC / "holdout-scoring" / "measurement.json",
        TOPIC / "holdout-scoring" / "decision.json",
        SOURCE_FREEZE,
    )
    if (old["slurm_job_id"] != SOURCE_JOB or old["binding"] != binding.metadata()
            or old["criteria_sha256"] != criteria_sha or old["profile"] != profile
            or old["abstain_abs_log_odds"] != bounds or old["scoring_job_ids"] != SCORING_JOBS
            or old["equal_warm_primary"] != REPORT_KEY):
        raise ValueError("Original evaluation does not match frozen inputs")
    receipts, hashes = checked_scoring_jobs(SCORING_JOBS)
    manifests = old["prompt_manifest_sha256"]
    if (hashes != old["scoring_receipt_sha256"] or set(manifests) != set(SPLITS)
            or any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in manifests.values())):
        raise ValueError("Original scoring receipts or prompt manifests do not match")
    gold_path = assay.PANEL / "gold.jsonl"
    if gold_path.is_symlink():
        raise ValueError("Use the original ID-only gold panel")
    gold = gold_path.read_bytes()
    checked_bytes(gold, binding.metadata()["panel_gold_sha256"])
    return load_json(criteria), old, binding, receipts, hashes, manifests, gold


def mean_outcomes(rows, policies, costs):
    outputs = [reference.outcomes(rows, policy, costs) for policy in policies]
    return {row["id"]: np.mean([one[row["id"]] for one in outputs], axis=0) for row in rows}


def point_metrics(rows, old, new, warm):
    groups = reference.grouped(rows, old, new)
    return reference.ratios(np.sum(tuple(groups.values()), axis=0), warm, warm)


def risk_limits(config):
    absolute = config["max_absolute_risk"]
    paired = config["max_paired_risk_increase"]
    if not 0 <= absolute < 1 or not 0 <= paired < 1:
        raise ValueError("Expected nonnegative absolute and paired risk ceilings below one")
    return {key: level for risk in RISK_NAMES
            for key, level in ((f"old_{risk}_upper", absolute),
                               (f"new_{risk}_upper", absolute), (f"{risk}_increase_upper", paired))}


def check_point_risk(points, config, role):
    for key, maximum in risk_limits(config).items():
        if not math.isfinite(points[key]) or points[key] > maximum + TOL:
            raise ValueError(f"{role} does not satisfy point risk: {key}")


def fit_witness(train, test, config, previous):
    if train.binding != test.binding or train.question_ids & test.question_ids:
        raise ValueError("Expected isolated splits with the original frozen binding")
    gold_coverage = [coverage(panel)["gold"] for panel in (train, test)]
    if (min(1 - values.get("unknown", 0) / len(panel.rows)
            for panel, values in zip((train, test), gold_coverage, strict=True)) < config["min_gold_parse_coverage"]
            or min(gold_coverage[1].get(name, 0) for name in ("correct", "wrong")) < config["min_test_each_truth"]):
        raise ValueError("The sealed panel does not satisfy the original coverage requirements")
    fit = fit_round2_binary(train, config)
    control, cascade = reference.CONTROL, reference.CASCADE
    spending = {arm: (fit.plan[arm][0] * fit.cost_units[0] + fit.plan[arm][1] * fit.cost_units[1])
                / fit.amortization_candidates for arm in (control, cascade)}
    original = previous["policy_comparisons"]["comparisons"][REPORT_KEY]
    warm = spending[control]
    if (spending[cascade] != warm or original["first_warm_cost_per_candidate"] != warm
            or original["second_warm_cost_per_candidate"] != warm
            or previous["mechanical_equal_warm_primary"]["equal_risk_supported"] is not True):
        raise ValueError("Expected the previously verified equal-warmup comparison")
    old = mean_outcomes(test.rows, fit.policies[control], fit.cost_units)
    actual = mean_outcomes(test.rows, fit.policies[cascade], fit.cost_units)
    witness = point_metrics(test.rows, old, actual, warm)
    for key in ("old_cost", "new_cost", "relative_saving", *risk_limits(config),
                "old_strong", "new_strong", "old_gold", "new_gold"):
        if not math.isclose(witness[key], original["risk_and_cost"][key]["point"], abs_tol=TOL, rel_tol=0):
            raise ValueError(f"Rebuilt witness differs from the frozen evaluation: {key}")
    check_point_risk(witness, config, "Original witness")
    return fit, old, witness, warm


def point_oracle(rows, states, old, costs, warm, config, *, cascade):
    """Optimize the original randomized tree family at observed point constraints."""
    check_point_risk(point_metrics(rows, old, old, warm), config, "Fixed comparator")
    if len(states) != len(set(states)) or any(state[-1] not in (0, 1) for state in states):
        raise ValueError("Expected unique development states with an observed weak verdict")
    options = reference.TREES if cascade else reference.DIRECT
    if not states:
        raise ValueError("No policy states had weak development verdicts")
    routes = {state: slice(i * len(options), (i + 1) * len(options)) for i, state in enumerate(states)}
    n_route = len(states) * len(options)
    candidate = np.zeros((len(rows), 3, n_route))
    fallback_cost = 0.0
    for i, row in enumerate(rows):
        state = reference.cheap_state(row)
        if state not in routes:
            # Same no-weak and absent-development-state fallback as reference.outcomes.
            fallback_cost += costs[1]
            continue
        for column, tree in enumerate(options, start=routes[state].start):
            candidate[i, :, column] = reference.action_outcome(tree, row["strong"], costs)[:3]

    truth = np.asarray([-1 if row["gold"] is None else int(row["gold"]) for row in rows])
    old_errors = np.asarray([old[row["id"]][1:3] for row in rows])
    unknown = truth == -1
    n_unknown = int(unknown.sum())
    variables = n_route + 2 * n_unknown
    a_ub = np.zeros((4 + 2 * n_unknown, variables))
    b_ub = np.zeros(len(a_ub))
    absolute, paired = (config[key] for key in ("max_absolute_risk", "max_paired_risk_increase"))
    for label in (0, 1):
        known = truth == label
        n_known = int(known.sum())
        errors = candidate[:, 1 + label]
        known_sum = errors[known].sum(axis=0)
        a_ub[2 * label, :n_route] = known_sum + (1 - absolute) * errors[unknown].sum(axis=0)
        b_ub[2 * label] = absolute * n_known
        slacks = slice(n_route + label * n_unknown, n_route + (label + 1) * n_unknown)
        a_ub[2 * label + 1, :n_route] = known_sum
        a_ub[2 * label + 1, slacks] = 1
        b_ub[2 * label + 1] = paired * n_known + old_errors[known, label].sum()
        limits = slice(4 + label * n_unknown, 4 + (label + 1) * n_unknown)
        a_ub[limits, :n_route] = errors[unknown]
        a_ub[limits, slacks] = -np.eye(n_unknown)
        b_ub[limits] = old_errors[unknown, label]

    a_eq = np.pad(np.repeat(np.eye(len(states)), len(options), axis=1), ((0, 0), (0, 2 * n_unknown)))
    objective = np.pad(candidate[:, 0].mean(axis=0), (0, 2 * n_unknown))
    solved = linprog(objective, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=np.ones(len(states)),
                     bounds=(0, 1), method="highs")
    if not solved.success:
        raise ValueError(f"Point-risk routing LP failed: {solved.message}")
    dual_ineq = solved.ineqlin.marginals
    dual_eq = solved.eqlin.marginals
    dual_low = solved.lower.marginals
    dual_high = solved.upper.marginals
    dual_value = b_ub @ dual_ineq + dual_eq.sum() + dual_high.sum()
    if (np.max(dual_ineq) > TOL or np.min(dual_low) < -TOL or np.max(dual_high) > TOL
            or not np.allclose(objective, a_ub.T @ dual_ineq + a_eq.T @ dual_eq + dual_low + dual_high,
                               atol=TOL, rtol=0)
            or not math.isclose(solved.fun, dual_value, abs_tol=TOL, rel_tol=0)):
        raise ValueError("Routing LP dual solution did not certify the computed cost")
    policy = {state: (options, solved.x[span]) for state, span in routes.items()}
    if any(not np.isclose(weights.sum(), 1, atol=TOL, rtol=0) or (weights < -TOL).any()
           for _, weights in policy.values()):
        raise ValueError("Optimized tree probabilities are invalid")
    new = reference.outcomes(rows, policy, costs)
    points = point_metrics(rows, old, new, warm)
    check_point_risk(points, config, "Optimized policy")
    if not math.isclose(points["new_cost"], solved.fun + fallback_cost / len(rows) + warm, abs_tol=TOL, rel_tol=0):
        raise ValueError("Routing LP objective differs from original cost calculation")
    return points | {"dual_cost_lower_bound": float(dual_value + fallback_cost / len(rows) + warm)}


def measure(train, test, config, previous):
    fit, old, witness, warm = fit_witness(train, test, config, previous)
    states = reference.available_states(train.rows)
    direct, cascade = (point_oracle(test.rows, states, old, fit.cost_units, warm, config, cascade=allow)
                       for allow in (False, True))
    if (direct["new_cost"] > witness["old_cost"] + TOL
            or cascade["new_cost"] > min(direct["new_cost"], witness["new_cost"]) + TOL):
        raise ValueError("The optimum does not dominate the feasible witness or its subset")
    return {"warm_cost_per_candidate_both_arms": warm, "original_equal_warm_witness": witness,
            "point_risk_limits": risk_limits(config), "direct_tree_ceiling": direct,
            "cascade_tree_ceiling": cascade,
            "remaining_saving_pp": 100 * (cascade["relative_saving"] - witness["relative_saving"]),
            "additional_strong_information_saving_pp": 100 * (cascade["relative_saving"] - direct["relative_saving"]),
            "original_descriptive_saving_floor": config["min_cost_saving"],
            "eligible_development_states": len(states), "coverage": coverage(test)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--check-only", action="store_true")
    modes.add_argument("--local-run-id")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.freeze or ""):
        parser.error("Supply the sealed ceiling Git commit")
    job = os.environ.get("SLURM_JOB_ID", "")
    if args.local_run_id:
        if job or not re.fullmatch(r"local-[A-Za-z0-9]{8}", args.local_run_id):
            parser.error("Use the dedicated local launcher outside Slurm")
        identity = {"execution_mode": "local_cpu", "local_run_id": args.local_run_id,
                    "execution_host": socket.gethostname(),
                    "started_at_taipei": datetime.now(ZoneInfo("Asia/Taipei")).isoformat(timespec="seconds")}
        run_id = args.local_run_id
    else:
        if not args.check_only and not re.fullmatch(r"[1-9][0-9]*", job):
            parser.error("Run the ceiling through its Slurm launcher")
        identity = {"execution_mode": "slurm_cpu", "slurm_job_id": job}
        run_id = job
    config, old, binding, receipts, hashes, manifests, gold = read_inputs()
    provenance = {"scope": SCOPE, "freeze_commit": args.freeze, "source_evaluation_job": SOURCE_JOB,
                  "source_evaluation_sha256": SOURCE_SHA, "source_scoring_jobs": SCORING_JOBS,
                  "scoring_receipt_sha256": hashes, "binding": binding.metadata()}
    if args.check_only:
        print(json.dumps(provenance | {"stage": "artifacts_checked"}, sort_keys=True))
        return
    out_dir = assay.prepare.SHARED / "runs/fallible-verifier/empirical-policy-ceiling" / run_id
    out_dir.mkdir(parents=True, exist_ok=False)
    train, test = (ingest_split(gold, receipts[split], hashes[split], binding, split, manifests[split])
                   for split in SPLITS)
    output = out_dir / "summary.json"
    partial = output.with_suffix(".partial")
    report = measure(train, test, config, old)
    partial.write_text(json.dumps(provenance | identity | report,
                                  allow_nan=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    partial.rename(output)
    print(json.dumps(identity | {"summary": str(output), "summary_sha256": assay.digest(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
