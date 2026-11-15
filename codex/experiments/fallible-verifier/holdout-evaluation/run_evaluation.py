"""Produce an aggregate binary-policy readback after both frozen scoring jobs succeed."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
TOPIC = HERE.parent
sys.path[:0] = [str(HERE), str(TOPIC / "holdout-scoring")]
from diagnostics import assess_binary_signal, measure_binary_signal
from evaluation import assess_equal_risk, compare_round2_binary, fit_round2_binary, ingest_split, reference
from receipt_contract import ROLES, checked_bytes, load_json
from scoring import OUTPUT, assay, load_policy, manifest_sha, panel_rows
from round2.protocol import CONFIG, write_json


JOB_NAME = "fv-gsm8k-policy-score"
JOB_ID = re.compile(r"[1-9][0-9]*")
JOB_RECORD = re.compile(r"\[codex\] [0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2} "
                        r"(train|test) (weak|strong) receipt SHA-256 ([0-9a-f]{64})")
SPLITS = ("train", "test")
REPORT_ROOT = assay.prepare.SHARED / "runs/fallible-verifier/gsm8k-policy-evaluation"


def completed_jobs(history, jobs):
    expected = {(job, "COMPLETED", "0:0", JOB_NAME) for job in jobs.values()}
    observed = [tuple(line.split("|")) for line in history.splitlines() if line]
    if len(observed) != len(expected) or set(observed) != expected:
        raise ValueError("Both scoring jobs must be distinct successful Slurm jobs")


def receipt_hashes(log, split):
    result = {}
    for line in log.splitlines():
        match = JOB_RECORD.fullmatch(line)
        if match is None:
            continue
        actual_split, role, digest = match.groups()
        if actual_split != split or role in result:
            raise ValueError("Duplicated or wrong-split receipt entry in the Slurm log")
        result[role] = digest
    if set(result) != set(ROLES):
        raise ValueError("The Slurm log lacks one or both scoring receipt digests")
    return result


def checked_scoring_jobs(jobs):
    if set(jobs) != set(SPLITS) or len(set(jobs.values())) != 2 or any(
            not JOB_ID.fullmatch(job) for job in jobs.values()):
        raise ValueError("Provide two distinct train/test scoring job IDs")
    history = subprocess.check_output(
        ["sacct", "-X", "-n", "-P", "-j", ",".join(jobs.values()), "-o", "JobID,State,ExitCode,JobName"],
        text=True,
    )
    completed_jobs(history, jobs)
    inputs, hashes = {}, {}
    for split, job in jobs.items():
        log = assay.prepare.SHARED / "tmp" / f"{JOB_NAME}-{job}.log"
        hashes[split] = receipt_hashes(log.read_text(), split)
        inputs[split] = {}
        for role, digest in hashes[split].items():
            path = OUTPUT / job / f"{split}-{role}.json"
            if path.is_symlink():
                raise ValueError("Scoring receipt must be a direct job artifact")
            body = path.read_bytes()
            checked_bytes(body, digest)
            inputs[split][role] = body
    return inputs, hashes


def readback(train, test, config):
    bootstrap = {"draws": config["bootstrap_draws"], "seed": config["bootstrap_seed"],
                 "tail": config["bootstrap_joint_alpha"] / config["max_test_family"]}
    signal = measure_binary_signal(train, test, smoothing=config["smoothing"], bootstrap=bootstrap)
    policies = compare_round2_binary(fit_round2_binary(train, config), test, bootstrap)
    primary_policy = f"{reference.CASCADE}_vs_{reference.CONTROL}_amortized"
    support_comparison = f"{reference.FULL_CASCADE}_vs_{reference.FULL_CONTROL}_amortized"
    return {"phenomenon_and_signal": signal, "mechanical_phenomenon_and_signal": assess_binary_signal(signal, config),
            "policy_comparisons": policies, "equal_warm_primary": primary_policy,
            "full_training_information_reference": support_comparison,
            "mechanical_equal_warm_primary": assess_equal_risk(policies["comparisons"][primary_policy],
                                                               policies["development_coverage"], config),
            "mechanical_full_training_information_reference": assess_equal_risk(
                policies["comparisons"][support_comparison], policies["development_coverage"], config)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--freeze")
    parser.add_argument("--train-job")
    parser.add_argument("--test-job")
    args = parser.parse_args()
    payload = CONFIG.read_bytes()
    config_sha = assay.prepare.ROUND2_CONFIG_SHA256
    checked_bytes(payload, config_sha)
    config = load_json(payload)
    if args.check_only:
        if args.freeze or args.train_job or args.test_job:
            parser.error("Static check does not access scoring jobs or panel files")
        print(json.dumps({"scope": "static_binary_policy_replay", "criteria_sha256": config_sha}, sort_keys=True))
        return

    job = os.environ.get("SLURM_JOB_ID", "")
    if not JOB_ID.fullmatch(job) or not args.freeze or not args.train_job or not args.test_job:
        parser.error("A recorded evaluation job, scoring jobs and a common freeze are required")
    policy_dir = TOPIC / "holdout-scoring"
    profile, bounds, binding = load_policy(policy_dir / "measurement.json", policy_dir / "decision.json", args.freeze)
    output = REPORT_ROOT / job / "summary.json"
    if output.exists():
        raise FileExistsError(output)
    jobs = {"train": args.train_job, "test": args.test_job}
    receipts, hashes = checked_scoring_jobs(jobs)
    prompts = (assay.PANEL / "prompts.jsonl").read_bytes()
    manifest = {split: manifest_sha(panel_rows(prompts, split), profile) for split in SPLITS}
    gold = (assay.PANEL / "gold.jsonl").read_bytes()
    train, test = (ingest_split(gold, receipts[split], hashes[split], binding, split, manifest[split]) for split in SPLITS)
    report = readback(train, test, config)
    write_json(output, {"scope": "candidate_round2_binary_policy_replay", "slurm_job_id": job,
                        "binding": binding.metadata(), "criteria_sha256": config_sha, "profile": profile,
                        "abstain_abs_log_odds": bounds, "scoring_job_ids": jobs, "scoring_receipt_sha256": hashes,
                        "prompt_manifest_sha256": manifest, **report})
    print(json.dumps({"slurm_job_id": job, "summary_sha256": assay.digest(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
