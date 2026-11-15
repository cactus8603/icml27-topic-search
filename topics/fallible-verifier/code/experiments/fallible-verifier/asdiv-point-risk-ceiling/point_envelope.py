"""Replay the sealed ASDiv-A panel at a common conservative absolute point risk."""

import argparse
from dataclasses import replace
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import socket
import sys
from zoneinfo import ZoneInfo


TOPIC = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(TOPIC / "asdiv-replication"), str(TOPIC / "empirical-policy-ceiling")]
import assay as asdiv
import ceiling as lp
import evaluate as frozen_evaluation


SOURCE_JOB = "26707"
SOURCE_FREEZE = "885ac701b9b4ed313c3ffb7d6619b70953f0422f"
SOURCE_SHA = "e12d6896b08d2cc563d2ee11a4aac20b860e1acb0365b1846d7e6ea8ec9b8d9c"
COVERAGE_SHA = "86d86ac5658db34f07f47554612f7d00bfaac3737b801b8542a6bdf8ee1ae496"
LP_FREEZE = "454e3d7b2bb4aeec07027d9467efd5c2afcc569f"
SCOPE = "asdiv_a_posthoc_common_absolute_point_risk_cost_envelope"
RUNS = asdiv.prepare.SHARED / "runs/fallible-verifier"
COSTS = (0.1, 1.0)
WARM_COST = 0.135
ABSOLUTE = 0.11
PAIRED_SLACK = 0.5
RISK_CONFIG = {"max_absolute_risk": ABSOLUTE, "max_paired_risk_increase": PAIRED_SLACK}
ABSOLUTE_KEYS = tuple(f"{arm}_{rate}_upper" for arm in ("old", "new") for rate in lp.RISK_NAMES)


def load_config():
    payload = lp.CONFIG.read_bytes()
    digest = frozen_evaluation.assay.prepare.ROUND2_CONFIG_SHA256
    asdiv.checked_bytes(payload, digest)
    config = asdiv.load_json(payload)
    if config["max_absolute_risk"] != ABSOLUTE or ABSOLUTE / (1 - ABSOLUTE) >= PAIRED_SLACK:
        raise ValueError("Expected the original absolute cap and a redundant paired allowance")
    return config, digest


def check_publication(publication, criteria_sha):
    expected = {
        "scope": asdiv.SUMMARY_SCOPE, "dataset": asdiv.DATASET, "slurm_job_id": SOURCE_JOB,
        "criteria_sha256": criteria_sha, "equal_warm_primary": lp.REPORT_KEY,
    }
    binding = publication["binding"]
    if (any(publication[name] != value for name, value in expected.items())
            or binding["panel_coverage_sha256"] != COVERAGE_SHA or binding["freeze_commit"] != SOURCE_FREEZE
            or binding["prepared_sha256"] != asdiv.PREPARED_SHA256 or binding["slurm_job_id"] != SOURCE_JOB):
        raise ValueError("Source summary belongs to a different ASDiv-A panel or protocol")
    hashes = publication["scoring_receipt_sha256"]
    if (set(hashes) != set(lp.SPLITS) or any(set(pins) != set(asdiv.MODEL_PINS) for pins in hashes.values())
            or any(not asdiv.DIGEST.fullmatch(value) for pins in hashes.values() for value in pins.values())):
        raise ValueError("Source summary lacks the four frozen scoring receipt digests")


def read_inputs(source, criteria_sha):
    summary_path = source / "summary.json"
    if summary_path.is_symlink():
        raise ValueError("Expected the original ASDiv-A source summary")
    raw = summary_path.read_bytes()
    asdiv.checked_bytes(raw, SOURCE_SHA)
    publication = asdiv.load_json(raw)
    check_publication(publication, criteria_sha)

    panel = source / "panel"
    binding, groups = asdiv.load_panel(panel, COVERAGE_SHA, SOURCE_FREEZE, SOURCE_JOB)
    if binding.metadata() != publication["binding"]:
        raise ValueError("ASDiv-A panel provenance differs from the source summary")
    gold = frozen_evaluation.index_gold((panel / "gold.jsonl").read_bytes(), groups, binding)
    splits = {}
    for split in lp.SPLITS:
        paired = {}
        for role in asdiv.MODEL_PINS:
            path = source / "scoring" / f"{split}-{role}.json"
            if path.is_symlink():
                raise ValueError("Expected the original ASDiv-A scoring receipt")
            paired[role], digest = frozen_evaluation.read_role(path, role, split, groups[split], binding)
            if digest != publication["scoring_receipt_sha256"][split][role]:
                raise ValueError(f"ASDiv-A {split}/{role} scoring receipt fingerprint differs")
        joined = frozen_evaluation.make_split(groups[split], gold, paired, split, binding)
        splits[split] = replace(joined, rows=tuple(row | {"dataset": asdiv.DATASET} for row in joined.rows))
    return splits["train"], splits["test"], publication


def published_points(train, test, config, publication):
    if (train.name != "train" or test.name != "test" or train.binding != test.binding
            or train.question_ids & test.question_ids):
        raise ValueError("Expected isolated development and holdout panels with a common binding")
    comparisons = publication["policy_comparisons"]
    primary = comparisons["comparisons"][lp.REPORT_KEY]
    train_coverage, test_coverage = (lp.coverage(panel) for panel in (train, test))
    if train_coverage != comparisons["development_coverage"] or test_coverage != primary["coverage"]:
        raise ValueError("Reconstructed coverage differs from the frozen evaluation")
    truth = test_coverage["gold"]
    if (min(1 - panel["gold"].get("unknown", 0) / panel["candidates"]
            for panel in (train_coverage, test_coverage)) < config["min_gold_parse_coverage"]
            or min(truth.get(name, 0) for name in ("wrong", "correct")) < config["min_test_each_truth"]):
        raise ValueError("ASDiv-A does not meet the original gold coverage and class-count requirements")

    fit = lp.fit_round2_binary(train, config)
    arms = lp.reference.CONTROL, lp.reference.CASCADE
    costs = fit.cost_units
    spending = [(fit.plan[arm][0] * costs[0] + fit.plan[arm][1] * costs[1]) / fit.amortization_candidates
                for arm in arms]
    if (costs != COSTS or spending != [WARM_COST, WARM_COST]
            or primary["first_warm_cost_per_candidate"] != WARM_COST
            or primary["second_warm_cost_per_candidate"] != WARM_COST):
        raise ValueError("Original equal-warm costs or price units differ")
    control, cascade = (lp.mean_outcomes(test.rows, fit.policies[arm], costs) for arm in arms)
    points = lp.point_metrics(test.rows, control, cascade, WARM_COST)
    for key, value in points.items():
        if not math.isclose(value, primary["risk_and_cost"][key]["point"], abs_tol=lp.TOL, rel_tol=0):
            raise ValueError(f"Rebuilt published policy point differs: {key}")
    failures = [key for key, limit in lp.risk_limits(config).items() if points[key] > limit + lp.TOL]
    absolute_failures = [key for key in failures if key in ABSOLUTE_KEYS]
    return {
        "comparison": lp.REPORT_KEY, "points": points,
        "formal_equal_risk_supported": publication["mechanical_equal_warm_primary"]["equal_risk_supported"],
        "original_point_risk_violations": failures,
        "same_absolute_point_cap_eligible": not absolute_failures,
        "same_absolute_point_cap_violations": absolute_failures,
    }


def solve(train_rows, test_rows):
    states = lp.reference.available_states(train_rows)
    # An empty decision table always buys the perfect expert, including for unknown truth.
    expert = lp.reference.outcomes(test_rows, {}, COSTS)
    direct, cascade = (lp.point_oracle(test_rows, states, expert, COSTS, WARM_COST, RISK_CONFIG, cascade=allow)
                       for allow in (False, True))
    if direct["new_cost"] > 1 + WARM_COST + lp.TOL or cascade["new_cost"] > direct["new_cost"] + lp.TOL:
        raise ValueError("Optimal routing costs violate the nested policy classes")
    delta = direct["new_cost"] - cascade["new_cost"]
    return {
        "nominal_warm_cost_per_candidate": WARM_COST,
        "strong_cost_per_consult": COSTS[0], "perfect_expert_cost_per_consult": COSTS[1],
        "comparison_cost_per_candidate_always_expert": COSTS[1] + WARM_COST,
        "absolute_conservative_point_cap": ABSOLUTE,
        "lp_redundant_paired_allowance": PAIRED_SLACK,
        "direct_tree_optimum": direct, "cascade_tree_optimum": cascade,
        "incremental_cascade_cost_reduction_per_candidate": delta,
        "incremental_cascade_cost_reduction_relative_to_direct": delta / direct["new_cost"],
        "additional_strong_information_saving_pp_vs_always_expert": 100 * delta / direct["old_cost"],
        "development_observed_weak_states": len(states),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--local-run-id")
    mode.add_argument("--job")
    args = parser.parse_args()
    active_job = os.environ.get("SLURM_JOB_ID", "")
    if not asdiv.FREEZE.fullmatch(args.freeze):
        parser.error("Supply the reviewed ASDiv-A point-risk commit")
    if args.job is not None:
        if not asdiv.JOB.fullmatch(args.job) or args.job != active_job:
            parser.error("Supply the matching active Slurm job")
    elif active_job or (args.local_run_id is not None and not re.fullmatch(r"local-[A-Za-z0-9]{8}", args.local_run_id)):
        parser.error("Use the sealed launcher outside Slurm for a local run or preflight")
    config, criteria_sha = load_config()
    provenance = {
        "scope": SCOPE, "freeze_commit": args.freeze, "criteria_sha256": criteria_sha,
        "source_evaluation_job": SOURCE_JOB, "source_evaluation_commit": SOURCE_FREEZE,
        "source_evaluation_sha256": SOURCE_SHA, "source_coverage_sha256": COVERAGE_SHA, "lp_commit": LP_FREEZE,
        "evidence_scope": "posthoc_holdout_gold_oracle_fixed_warm_states",
    }
    if args.check_only:
        print(json.dumps(provenance | {"stage": "static_protocol_checked"}, sort_keys=True))
        return

    identity = (
        {"execution_mode": "slurm_cpu", "slurm_job_id": args.job} if args.job
        else {"execution_mode": "local_cpu", "local_run_id": args.local_run_id}
    ) | {
        "execution_host": socket.gethostname(),
        "started_at_taipei": datetime.now(ZoneInfo("Asia/Taipei")).isoformat(timespec="seconds"),
    }
    train, test, publication = read_inputs(RUNS / "asdiv-replication" / SOURCE_JOB, criteria_sha)
    witness = published_points(train, test, config, publication)
    report = provenance | identity | solve(train.rows, test.rows) | {
        "binding": publication["binding"], "scoring_receipt_sha256": publication["scoring_receipt_sha256"],
        "coverage": {panel.name: lp.coverage(panel) for panel in (train, test)},
        "published_original_comparison_diagnostic": witness,
    }
    directory = RUNS / "asdiv-point-risk-ceiling" / (args.job or args.local_run_id)
    directory.mkdir(parents=True, exist_ok=False)
    output = directory / "summary.json"
    asdiv.score.scoring.write_json(output, report)
    print(json.dumps(identity | {"summary": str(output), "summary_sha256": asdiv.score.scoring.digest(output)},
                     sort_keys=True))


if __name__ == "__main__":
    main()
