"""Compare warm solvability under the same answer parser and two judge context bounds."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from generate import PROFILES, receipt_identity, solvers, verify_warm
sys.path.insert(0, str(HERE.parent / "gsm1k-replication"))
import score
from panel import boxed_number


BASELINE = {
    "base": "1d6b1fe2bbacfe6a27783b2e89ea1e9b58a42810dc26193e0785720439634e74",
    "oat": "32a58fac8bde7cebf91879b766d2efcbb1127a5e14012b3d356a5644124901c2",
}
BASE_FREEZE = "229cdcf22a60e3f030e1e1edf9347294e9584b02"
BASE_JOB = "26669"
BASE_DIR = Path("/work/ChihChungHsu/cactus8603/icml/runs/fallible-verifier/gsm1k-replication/26669")
HISTORICAL_ROW = re.compile(rb'^\{"finish_reason":"(?:stop|length)","question_id":"([0-9a-f]{64})","solution":')
PROFILE_ORDER = ("cap1536", "compact768")
BOOT_DRAWS = 10_000
BOOT_SEED = 202611151520


def bytes_checked(path, expected, length=None):
    payload = path.read_bytes()
    if sha256(payload).hexdigest() != expected or (length is not None and len(payload) != length):
        raise ValueError(f"Artifact digest or size differs: {path.name}")
    return payload


def validate_warm_rows(rows, questions, role):
    expected = [(row["question_id"], f"{role}-{index}") for row in questions for index in range(2)]
    if len(rows) != len(expected):
        raise ValueError(f"Warm {role} roster does not contain exact two draws per question")
    for row, key in zip(rows, expected, strict=True):
        if (set(row) != {"question_id", "variant", "solution", "finish_reason"}
                or (row["question_id"], row["variant"]) != key
                or not isinstance(row["solution"], str) or row["finish_reason"] not in {"stop", "length"}):
            raise ValueError(f"Invalid {role} warm proposal row")
    return rows


def original_warm_proposals(questions, role):
    source = BASE_DIR / role
    receipt = json.loads(bytes_checked(source / "receipt.json", BASELINE[role]))
    if (receipt["freeze"], receipt["job"], receipt["role"], receipt["scope"]) != (
            BASE_FREEZE, BASE_JOB, role, solvers.FORMAL_SCOPE):
        raise ValueError("Original solver receipt does not belong to the completed baseline")
    payload = bytes_checked(source / solvers.ROWS_BASENAME, receipt["rows_sha256"], receipt["rows_bytes"])
    ids = {row["question_id"].encode() for row in questions}
    selected = []
    for line in payload.splitlines():
        match = HISTORICAL_ROW.match(line)
        if not match:
            raise ValueError("Original proposal record is outside the source schema")
        if match[1] in ids:
            selected.append(json.loads(line))
    return validate_warm_rows(selected, questions, role)


def read_new(directory, profile, freeze, job, warm_sha, manifest, questions):
    raw = (directory / "receipt.json").read_bytes()
    receipt = json.loads(raw)
    expected = receipt_identity(profile, freeze, job, warm_sha, manifest)
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Foreign solver or warm bundle: {profile}")
    payload = bytes_checked(directory / solvers.ROWS_BASENAME, receipt["rows_sha256"], receipt["rows_bytes"])
    rows = validate_warm_rows([json.loads(line) for line in payload.splitlines()], questions, "base")
    return rows, sha256(raw).hexdigest()


def usable(row):
    return row["finish_reason"] == "stop" and boxed_number(row["solution"]) is not None


def counts(rows):
    return {"candidates": len(rows), "eligible": sum(map(usable, rows)),
            "finish_reason": dict(Counter(row["finish_reason"] for row in rows))}


def paired_delta(original, comparison):
    first = np.array([usable(row) for row in original], dtype=float).reshape(-1, 2)
    second = np.array([usable(row) for row in comparison], dtype=float).reshape(-1, 2)
    change = (second - first).mean(axis=1)
    draws = np.random.default_rng(BOOT_SEED).integers(len(change), size=(BOOT_DRAWS, len(change)))
    return {"change_pp": float(change.mean() * 100),
            "paired_question_ci95_pp": (np.quantile(change[draws].mean(axis=1), [.025, .975]) * 100).tolist()}


def candidate_groups(questions, base, oat):
    groups = {row["question_id"]: {} for row in questions}
    source = {row["question_id"]: row["question"] for row in questions}
    for row in (*base, *oat):
        qid, variant = row["question_id"], row["variant"]
        groups[qid][variant] = {"id": score.scoring.hash_order(qid, variant), "question_id": qid, "split": "train",
                                "question": source[qid], "solution": row["solution"], "variant": variant}
    if any(set(group) != set(score.VARIANTS) for group in groups.values()):
        raise ValueError("Expected all four candidate variants for every warm question")
    return groups


def context_counts(groups, tokenizer):
    _, aliases = score.scoring.selected_requests(groups, tokenizer, score.MAXIMUM, score.PROFILE)
    by_candidate = Counter(key[1][0] for key, result in aliases.items() if result is not None)
    return {"candidates": sum(len(group) for group in groups.values()),
            "both_mappings_fit": sum(value == 2 for value in by_candidate.values()),
            "oversized_logical_requests": sum(result is None for result in aliases.values())}


def qualifies(row):
    candidates = row["base"]["candidates"] + row["oat"]["candidates"]
    return (100 * (row["base"]["eligible"] + row["oat"]["eligible"]) >= 85 * candidates
            and all(100 * role["both_mappings_fit"] >= 98 * candidates for role in row["context"].values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--warm-sha256", required=True)
    parser.add_argument("--freeze")
    parser.add_argument("--job")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    questions, manifest = verify_warm(args.bundle, args.warm_sha256)
    original = {role: original_warm_proposals(questions, role) for role in BASELINE}
    if [counts(original[role])["eligible"] for role in BASELINE] != [211, 490]:
        raise ValueError("Original warm coverage does not match the receipt-only technical audit")
    tokenizers = {role: score.runtime(role, check_only=True)[2] for role in ("weak", "strong")}
    if args.check_only:
        base_groups = candidate_groups(questions, original["base"], original["oat"])
        for tokenizer in tokenizers.values():
            if context_counts(base_groups, tokenizer)["candidates"] != 1000:
                raise ValueError("Incorrect original warm prompt roster")
        print(json.dumps({"warm_questions": len(questions), "baseline_receipt_sha256": BASELINE}, sort_keys=True))
        return
    if not args.run_dir or not solvers.COMMIT_RE.fullmatch(args.freeze or "") or not solvers.JOB_RE.fullmatch(args.job or ""):
        parser.error("Formal technical scan needs the existing run directory, frozen commit and Slurm job")
    rows, hashes = {"baseline": original["base"]}, {}
    for profile in PROFILE_ORDER:
        if profile not in PROFILES:
            raise ValueError("Frozen profile differs from generator")
        rows[profile], hashes[profile] = read_new(args.run_dir / profile, profile, args.freeze, args.job,
                                                 args.warm_sha256, manifest, questions)
    results = {}
    for name, base in rows.items():
        oat = original["oat"]
        groups = candidate_groups(questions, base, oat)
        results[name] = {"base": counts(base), "oat": counts(oat),
                         "context": {role: context_counts(groups, tokenizer) for role, tokenizer in tokenizers.items()},
                         "base_vs_original": paired_delta(original["base"], base)}
    selected = next((name for name in PROFILE_ORDER if qualifies(results[name])), None)
    report = {"scope": "warm_questions_only_coverage_no_gold", "freeze": args.freeze, "job": args.job,
              "warm_sha256": args.warm_sha256, "reference_receipt_sha256": BASELINE,
              "new_receipt_sha256": hashes, "results": results, "selected_profile": selected}
    payload = solvers.json_bytes(report)
    with (args.run_dir / "summary.json").open("xb") as handle:
        handle.write(payload)
    print(json.dumps({"summary_sha256": sha256(payload).hexdigest(), "slurm_job_id": args.job}, sort_keys=True))


if __name__ == "__main__":
    main()
