"""Replay a nested gold-audit design with the original binary routing policy."""

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import random
import re
import socket
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

TOPIC = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(TOPIC / "holdout-evaluation"), str(TOPIC / "empirical-policy-ceiling")]
import ceiling
from evaluation import assess_equal_risk, compare_actions, coverage, ingest_split, policy_actions, reference
from receipt_contract import checked_bytes, load_json
from run_evaluation import assay


SCOPE = "gsm8k_historical_holdout_nested_paid_audit"
RUN_ROOT = assay.prepare.SHARED / "runs/fallible-verifier"
ORACLE_FREEZE = "454e3d7b2bb4aeec07027d9467efd5c2afcc569f"
ORACLE_SHA = "449b6dfed9796934333df14d0b07bda61a88d44f73c44a8e056f46fe1bb072e7"
ORACLE_ID = "local-V9Cs70mV"
WARM_PER_CANDIDATE = 0.135


def nested_observations(rows, config, seed):
    probes, _, gold = reference.warm_plan(len(rows), config)
    # Match the probe roster from the original warm-up for the same draw.
    order = random.Random(seed).sample(range(len(rows)), len(rows))
    probe_map = {index: rows[index]["strong"] for index in sorted(order[:probes])}
    strata = defaultdict(list)
    for index, strong in probe_map.items():
        strata[reference.cheap_state(rows[index]), strong].append(index)
    # Encounter order by index also fixes ties in the reference allocation.
    if gold < len(strata):
        raise ValueError("The nested budget cannot cover all observed probe strata")
    audits = reference.sample_strata(strata, gold, random.Random(seed + 200_000), 0.5)
    if len(probe_map) != probes or len(audits) != gold or not audits <= probe_map.keys():
        raise ValueError("The nested draw fails the paid-observation roster")
    return probe_map, audits


def fit_draw(rows, config, seed):
    probes, audits = nested_observations(rows, config, seed)
    censored_rows = [
        {
            "dataset": row["dataset"],
            "solver_model": row["solver_model"],
            "weak": row["weak"],
            "gold": row["gold"] if index in audits else None,
        }
        for index, row in enumerate(rows)
    ]
    states = reference.available_states(censored_rows)
    distribution = reference.fit_distribution(censored_rows, states, probes, audits, config["smoothing"])
    costs = config["strong_price_units"] / config["gold_price_units"], 1.0
    policy = reference.optimize(distribution, True, config["train_risk_target"], costs, states)
    return policy, sum(censored_rows[index]["gold"] is None for index in audits)


def fit_nested(train, config):
    if train.name != "train" or len(train.rows) != 1000 or len(train.question_ids) != 250:
        raise ValueError("Fit only the original 250-question development panel")
    probes, comparator_gold, nested_gold = reference.warm_plan(len(train.rows), config)
    c_s, c_g = config["strong_price_units"], config["gold_price_units"]
    amortize = config["warm_amortization_candidates"]
    seeds = config["audit_seeds"]
    if ((probes, comparator_gold, nested_gold, c_s, c_g, amortize) != (350, 135, 100, 1, 10, 1000)
            or len(seeds) != 16 or len(set(seeds)) != 16
            or (probes * c_s / c_g + nested_gold) / amortize != WARM_PER_CANDIDATE):
        raise ValueError("The registered equal-budget settings changed")
    policies, unknown = zip(*(fit_draw(train.rows, config, seed) for seed in seeds), strict=True)
    return policies, {
        "strong_queries_per_draw": probes,
        "gold_reviews_per_draw": nested_gold,
        "gold_reviews_overlapping_strong_per_draw": nested_gold,
        "unresolved_gold_reviews_mean": sum(unknown) / len(unknown),
        "amortized_per_candidate": WARM_PER_CANDIDATE,
    }


def read_oracle(binding, hashes):
    path = RUN_ROOT / "empirical-policy-ceiling" / ORACLE_ID / "summary.json"
    if path.is_symlink():
        raise ValueError("Use the original published point-risk ceiling")
    payload = path.read_bytes()
    checked_bytes(payload, ORACLE_SHA)
    oracle = load_json(payload)
    expected = {
        "scope": ceiling.SCOPE,
        "freeze_commit": ORACLE_FREEZE,
        "local_run_id": ORACLE_ID,
        "source_evaluation_job": ceiling.SOURCE_JOB,
        "source_evaluation_sha256": ceiling.SOURCE_SHA,
        "source_scoring_jobs": ceiling.SCORING_JOBS,
        "scoring_receipt_sha256": hashes,
        "binding": binding.metadata(),
        "warm_cost_per_candidate_both_arms": WARM_PER_CANDIDATE,
    }
    if any(oracle[key] != value for key, value in expected.items()):
        raise ValueError("The ceiling and new policy must share the original comparison frame")
    return oracle


def evaluate(train, test, nested, warm_report, config, previous, oracle):
    # This fit happens only after the experimental policy has been fixed.
    original, _, witness, warm = ceiling.fit_witness(train, test, config, previous)
    if warm != WARM_PER_CANDIDATE or any(
            not math.isclose(value, oracle["original_equal_warm_witness"][name], abs_tol=ceiling.TOL, rel_tol=0)
            for name, value in witness.items()):
        raise ValueError("The independently replayed baseline and published ceiling do not match")
    bootstrap = {
        "draws": config["bootstrap_draws"],
        "seed": config["bootstrap_seed"],
        "tail": config["bootstrap_joint_alpha"] / config["max_test_family"],
    }
    outcomes = compare_actions(
        test, policy_actions(test.rows, original.policies[reference.CONTROL]), policy_actions(test.rows, nested),
        costs=original.cost_units, first_warm=warm, second_warm=warm, bootstrap=bootstrap,
    )
    risks = assess_equal_risk(outcomes, coverage(train), config)
    prior_saving = witness["relative_saving"]
    point_ceiling_saving = oracle["cascade_tree_ceiling"]["relative_saving"]
    gain, headroom = None, None
    if risks["equal_risk_supported"]:
        saving = outcomes["risk_and_cost"]["relative_saving"]["point"]
        if saving is None or saving > point_ceiling_saving + ceiling.TOL:
            raise ValueError("The admissible candidate cannot exceed the published point-risk ceiling")
        gain = 100 * (saving - prior_saving)
        headroom = 100 * (point_ceiling_saving - saving)
    return {
        "development_coverage": coverage(train),
        "nested_warm_up": warm_report,
        "original_equal_warm_witness": witness,
        "nested_vs_original_gold_only": outcomes,
        "mechanical_equal_warm": risks,
        "published_point_ceiling_saving": point_ceiling_saving,
        "descriptive_change_vs_original_verified_cascade_pp": gain,
        "descriptive_remaining_headroom_to_point_ceiling_pp": headroom,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--check-only", action="store_true")
    modes.add_argument("--job")
    modes.add_argument("--local-run-id")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.freeze):
        parser.error("Use the reviewed experimental freeze")
    if args.job and (not re.fullmatch(r"[1-9][0-9]*", args.job) or args.job != os.getenv("SLURM_JOB_ID")):
        parser.error("Use the recorded Slurm job")
    if args.local_run_id and (os.getenv("SLURM_JOB_ID") or not re.fullmatch(r"local-[A-Za-z0-9]+", args.local_run_id)):
        parser.error("Use the dedicated local launcher outside Slurm")
    config, previous, binding, receipts, hashes, manifests, gold = ceiling.read_inputs()
    oracle = read_oracle(binding, hashes)
    provenance = {
        "scope": SCOPE,
        "freeze_commit": args.freeze,
        "source_evaluation_job": ceiling.SOURCE_JOB,
        "source_evaluation_sha256": ceiling.SOURCE_SHA,
        "source_scoring_jobs": ceiling.SCORING_JOBS,
        "scoring_receipt_sha256": hashes,
        "published_ceiling_sha256": ORACLE_SHA,
        "binding": binding.metadata(),
    }
    if args.check_only:
        print(json.dumps(provenance | {"stage": "source_artifacts_checked"}, sort_keys=True))
        return

    train = ingest_split(gold, receipts["train"], hashes["train"], binding, "train", manifests["train"])
    nested, warm_report = fit_nested(train, config)
    # Test data have been authenticated as raw bytes. Decode their per-row
    # labels and verifier receipts only after the policy returned by fit_nested.
    test = ingest_split(gold, receipts["test"], hashes["test"], binding, "test", manifests["test"])
    report = evaluate(train, test, nested, warm_report, config, previous, oracle)
    run = RUN_ROOT / "gsm8k-nested-audit" / (args.job or args.local_run_id)
    run.mkdir(parents=True, exist_ok=False)
    identity = (
        {"execution_mode": "slurm_cpu", "slurm_job_id": args.job} if args.job
        else {"execution_mode": "local_cpu", "local_run_id": args.local_run_id}
    ) | {
        "execution_host": socket.gethostname(),
        "completed_at_taipei": datetime.now(ZoneInfo("Asia/Taipei")).isoformat(timespec="seconds"),
    }
    output = run / "summary.json"
    partial = output.with_suffix(".partial")
    partial.write_text(json.dumps(provenance | identity | report, allow_nan=False, sort_keys=True, indent=2) + "\n",
                       encoding="utf-8")
    partial.rename(output)
    print(json.dumps(identity | {"summary": str(output), "summary_sha256": assay.digest(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
