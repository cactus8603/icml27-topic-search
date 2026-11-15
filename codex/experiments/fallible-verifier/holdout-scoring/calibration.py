"""Summarize fixed abstention widths on the GSM8K development split."""

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scoring import (DEVELOPMENT_OUTPUT, DEVELOPMENT_PROFILE, MODEL_PINS, assay, checked_bytes, development_header,
                     label, load_json, manifest_sha, panel_rows, write_json)


WIDTHS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


def read_semantic(document, header, candidate_ids):
    if set(document) != set(header) | {"semantic_log_odds"} or any(document[k] != v for k, v in header.items()):
        raise ValueError("Development inference receipt does not bind to this train panel and job")
    scores = document["semantic_log_odds"]
    if scores.keys() != candidate_ids or any(value is not None and (
            type(value) not in (int, float) or not math.isfinite(value)) for value in scores.values()):
        raise ValueError("Development score IDs or score values are malformed")
    return scores


def operating_points(scores, gold):
    wrong = sum(value is False for value in gold.values())
    correct = sum(value is True for value in gold.values())
    if scores.keys() != gold.keys() or not wrong or not correct:
        raise ValueError("Aligned development candidates with both known classes are required")
    unknown = len(gold) - wrong - correct
    report = []
    for width in WIDTHS:
        observed_wrong = observed_correct = observed_unknown = false_accept = false_reject = abstain = missing = 0
        for key, score in scores.items():
            verdict, state = label(score, width)
            if state == "missing":
                missing += 1
            elif state == "abstain":
                abstain += 1
            elif gold[key] is False:
                observed_wrong += 1
                false_accept += verdict == 1
            elif gold[key] is True:
                observed_correct += 1
                false_reject += verdict == 0
            else:
                observed_unknown += 1
        report.append({"width_abs_log_odds": width, "missing": missing, "abstain": abstain,
                       "known_wrong": wrong, "known_correct": correct, "gold_unknown": unknown,
                       "observed_wrong": observed_wrong, "observed_correct": observed_correct,
                       "observed_unknown": observed_unknown, "false_accept": false_accept, "false_reject": false_reject})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True)
    for role in MODEL_PINS:
        parser.add_argument(f"--{role}-sha256", required=True)
    args = parser.parse_args()
    job = os.environ.get("SLURM_JOB_ID", "")
    if not re.fullmatch(r"[1-9][0-9]*", job) or not re.fullmatch(r"[0-9a-f]{40}", args.freeze):
        parser.error("A recorded development job and a reviewed full commit are required")
    output = DEVELOPMENT_OUTPUT / job
    destination = output / "calibration.json"
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    groups = panel_rows((assay.PANEL / "prompts.jsonl").read_bytes(), "train")
    ids = {row["id"] for group in groups.values() for row in group.values()}
    scores = {}
    hashes = {role: getattr(args, f"{role}_sha256") for role in MODEL_PINS}
    for role in MODEL_PINS:
        path = output / f"{role}.json"
        if not path.is_file() or path.is_symlink():
            raise ValueError("Development inference must publish both direct job receipts first")
        payload = path.read_bytes()
        checked_bytes(payload, hashes[role])
        scores[role] = read_semantic(load_json(payload), development_header(groups, args.freeze, job, role), ids)
    # Open gold only after both role scores pass. The existing loader decodes only train rows.
    gold_rows = assay.load_train("gold.jsonl")
    gold = {key: row["gold"] for key, row in assay.linked_gold(groups, gold_rows).items()}
    write_json(destination, {"schema": 1, "scope": "development_calibration_only", "slurm_job_id": job,
                             "freeze_commit": args.freeze, "development_questions": len(groups),
                             "development_gold_sha256": sha256(assay.prepare.jsonl_bytes(gold_rows)).hexdigest(),
                             "prompt_manifest_sha256": manifest_sha(groups, DEVELOPMENT_PROFILE),
                             "score_receipt_sha256": hashes,
                             "operating_points": {role: operating_points(values, gold) for role, values in scores.items()}})
    print(json.dumps({"slurm_job_id": job, "calibration_sha256": assay.digest(destination)}, sort_keys=True))


if __name__ == "__main__":
    main()
