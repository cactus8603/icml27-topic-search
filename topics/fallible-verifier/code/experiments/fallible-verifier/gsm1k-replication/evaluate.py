"""Align independently graded GSM1k IDs and replay the frozen verifier analysis."""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "holdout-evaluation")]
import score
from evaluation import MODEL_STATES, Split
from receipt_contract import MODEL_PINS, checked_bytes, load_json
from run_evaluation import CONFIG, assay, manifest_sha, readback


GOLD_FIELDS = {"id", "question_id", "split", "variant", "gold", "gold_status"}
GOLD_STATES = {"verified_numeric", "length", "other_finish", "empty_candidate", "unparseable_final_answer"}
SPLITS = ("train", "test")


def index_gold(payload, panels, binding):
    checked_bytes(payload, binding.metadata()["panel_gold_sha256"])
    expected = {
        row["id"]: (split, qid, variant)
        for split, groups in panels.items()
        for qid, variants in groups.items()
        for variant, row in variants.items()
    }
    result = {}
    for line in payload.splitlines():
        row = load_json(line)
        if set(row) != GOLD_FIELDS:
            raise ValueError("GSM1k gold requires its independent, ID-only six fields")
        key = row["id"]
        if (key in result or (row["split"], row["question_id"], row["variant"]) != expected.get(key)
                or row["gold_status"] not in GOLD_STATES
                or (row["gold_status"] == "verified_numeric") != (row["gold"] is not None)
                or (row["gold"] is not None and type(row["gold"]) is not bool)):
            raise ValueError("Gold contains a duplicate, misaligned ID, or invalid label")
        result[key] = row
    if result.keys() != expected.keys():
        raise ValueError("Gold and candidate ID coverage differ")
    return result


def read_role(path, role, split, groups, binding):
    data = path.read_bytes()
    receipt = load_json(data)
    model, revision = MODEL_PINS[role]
    header = {
        "schema": 1, "role": role, "model": model, "model_revision": revision,
        "split": split, "binding": binding.metadata(),
        "prompt_manifest_sha256": manifest_sha(groups, score.PROFILE),
    }
    if set(receipt) != {*header, "records"} or any(receipt[k] != v for k, v in header.items()):
        raise ValueError(f"Wrong frozen {split}/{role} scoring identity")
    records = {}
    for row in receipt["records"]:
        if set(row) != {"id", "verdict", "status"}:
            raise ValueError("Unexpected scored record fields")
        key, state, value = row["id"], row["status"], row["verdict"]
        if (key in records or state not in MODEL_STATES or (state == "observed") != (value is not None)
                or (value is not None and (type(value) is not int or value not in (0, 1)))):
            raise ValueError("Duplicate ID, invalid verdict, or incompatible missingness")
        records[key] = (value, state)
    expected = {row["id"] for variants in groups.values() for row in variants.values()}
    if records.keys() != expected:
        raise ValueError(f"Missing or wrong-split {split}/{role} scoring records")
    return records, hashlib.sha256(data).hexdigest()


def make_split(groups, gold, paired, split, binding):
    rows = []
    for qid, variants in groups.items():
        for variant, proposal in variants.items():
            key = proposal["id"]
            source = gold[key]
            row = {"id": key, "question_id": qid, "dataset": "gsm1k", "solver_model": variant,
                   "gold": source["gold"], "gold_status": source["gold_status"]}
            for role, records in paired.items():
                row[role], row[f"{role}_status"] = records[key]
            rows.append(row)
    return Split(split, binding, frozenset(groups), tuple(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--panel-dir", type=Path, required=True)
    parser.add_argument("--coverage-sha256", required=True)
    parser.add_argument("--score-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[1-9][0-9]*", args.job) or not re.fullmatch(r"[0-9a-f]{40}", args.freeze):
        parser.error("Supply the active Slurm job and full protocol freeze")
    manifest = score.load_coverage(args.panel_dir / "coverage.json", args.coverage_sha256)
    binding = score.panel_binding(args.freeze, manifest, args.coverage_sha256)
    prompts = score.panel_groups((args.panel_dir / "prompts.jsonl").read_bytes(), manifest)
    if set(prompts) != set(SPLITS) or prompts["train"].keys() & prompts["test"].keys():
        raise ValueError("Expected independent, complete warm-up and holdout question groups")
    gold = index_gold((args.panel_dir / "gold.jsonl").read_bytes(), prompts, binding)
    splits, hashes = {}, {}
    for split in SPLITS:
        paired, hashes[split] = {}, {}
        for role in MODEL_PINS:
            paired[role], hashes[split][role] = read_role(
                args.score_dir / f"{split}-{role}.json", role, split, prompts[split], binding
            )
        splits[split] = make_split(prompts[split], gold, paired, split, binding)

    config_bytes = CONFIG.read_bytes()
    config_sha = assay.prepare.ROUND2_CONFIG_SHA256
    checked_bytes(config_bytes, config_sha)
    report = {
        "scope": "gsm1k_near_distribution_replication_new_solver_candidates",
        "slurm_job_id": args.job,
        "criteria_sha256": config_sha,
        "binding": binding.metadata(),
        "scoring_receipt_sha256": hashes,
        **readback(splits["train"], splits["test"], load_json(config_bytes)),
    }
    data = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with args.output.open("xb") as handle:
        handle.write(data)
    print(json.dumps({"slurm_job_id": args.job, "summary_sha256": hashlib.sha256(data).hexdigest()}))


if __name__ == "__main__":
    main()
