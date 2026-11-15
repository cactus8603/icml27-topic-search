"""Qualify a pinned instruction solver on the original GSM1k warm questions."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from generate_instruct import verify_proposals, verify_questions
sys.path.insert(0, str(HERE.parent / "gsm1k-warm-coverage"))
from measure import (BASELINE, candidate_groups, context_counts, counts, original_warm_proposals,
                     paired_delta, score, solvers)


WARM_SHA = "356789196eb7bd953ff758d04de03d903ea7e063fe5c4f4b6027b96f685255e3"
MODEL = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
SCOPE = "gsm1k_warm_instruct_format_coverage_no_gold"
CRITERIA = {"syntax_eligible": 850, "each_judge_both_mappings_fit": 980, "denominator": 1000}
JUDGES = ("weak", "strong")


def qualifies(instruct, oat, context):
    total = CRITERIA["denominator"]
    if any(type(arm["eligible"]) is not int or arm["candidates"] != total // 2
           or not 0 <= arm["eligible"] <= total // 2 for arm in (instruct, oat)):
        raise ValueError("Expected exactly 500 syntax decisions per solver")
    if set(context) != set(JUDGES) or any(
            role["candidates"] != total or type(role["both_mappings_fit"]) is not int
            or not 0 <= role["both_mappings_fit"] <= total for role in context.values()):
        raise ValueError("Expected one length decision per candidate for each judge")
    return (instruct["eligible"] + oat["eligible"] >= CRITERIA["syntax_eligible"]
            and all(role["both_mappings_fit"] >= CRITERIA["each_judge_both_mappings_fit"] for role in context.values()))


def checked_gate(path, expected_sha, job, freeze):
    if not solvers.JOB_RE.fullmatch(job) or not solvers.COMMIT_RE.fullmatch(freeze):
        raise ValueError("Use the exact completed warm job and frozen commit")
    payload = path.read_bytes()
    if not solvers.SHA_RE.fullmatch(expected_sha) or sha256(payload).hexdigest() != expected_sha:
        raise ValueError("Qualification receipt SHA differs")
    gate = json.loads(payload)
    fields = {"schema": 1, "scope": SCOPE, "job": job, "freeze": freeze, "warm_sha256": WARM_SHA,
              "model": MODEL, "model_revision": MODEL_REVISION, "criteria": CRITERIA, "eligible": True,
              "original_receipt_sha256": BASELINE}
    if gate.get("eligible") is not True or any(gate.get(key) != value for key, value in fields.items()):
        raise ValueError("The completed warm receipt did not qualify this fixed solver")
    if (not solvers.SHA_RE.fullmatch(gate["instruct_receipt_sha256"])
            or not qualifies(gate["instruct"], gate["oat"], gate["context"])):
        raise ValueError("Qualification thresholds or model source receipt differ")
    return expected_sha


def row_list(questions, candidates):
    expected = [(row["question_id"], f"base-{draw}") for row in questions for draw in range(2)]
    if set(candidates) != set(expected):
        raise ValueError("Warm Instruct proposals do not cover the fixed questions and draws")
    return [{"question_id": qid, "variant": variant, **candidates[qid, variant]} for qid, variant in expected]


def assessment(args):
    rows, _ = verify_questions("gsm1k-warm", args.questions, args.manifest, WARM_SHA)
    previous = {role: original_warm_proposals(rows, role) for role in BASELINE}
    if [counts(previous[role])["eligible"] for role in BASELINE] != [211, 490]:
        raise ValueError("Original warm roster does not reproduce prepublished technical coverage")
    tokenizers = {role: score.runtime(role, check_only=True)[2] for role in JUDGES}
    if args.command == "check":
        groups = candidate_groups(rows, previous["base"], previous["oat"])
        if any(context_counts(groups, tokenizer)["both_mappings_fit"] != 1000 for tokenizer in tokenizers.values()):
            raise ValueError("Pinned original judge context does not reproduce the baseline")
        print(json.dumps({"scope": "instruct_warm_preflight", "warm_questions": len(rows),
                          "warm_sha256": WARM_SHA, "original_receipt_sha256": BASELINE}, sort_keys=True))
        return

    raw = args.receipt.read_bytes()
    receipt_sha = sha256(raw).hexdigest()
    _, candidates = verify_proposals(args.receipt, receipt_sha, "gsm1k-warm", rows, WARM_SHA,
                                     expected_freeze=args.freeze, expected_job=args.job)
    instruct = row_list(rows, candidates)
    oat = previous["oat"]
    groups = candidate_groups(rows, instruct, oat)
    context = {role: context_counts(groups, tokenizer) for role, tokenizer in tokenizers.items()}
    inst_counts, oat_counts = counts(instruct), counts(oat)
    result = {"schema": 1, "scope": SCOPE, "job": args.job, "freeze": args.freeze, "warm_sha256": WARM_SHA,
              "model": MODEL, "model_revision": MODEL_REVISION, "criteria": CRITERIA,
              "original_receipt_sha256": BASELINE, "instruct_receipt_sha256": receipt_sha,
              "instruct": inst_counts, "oat": oat_counts, "context": context,
              "instruct_vs_original_base": paired_delta(previous["base"], instruct),
              "eligible": qualifies(inst_counts, oat_counts, context)}
    body = solvers.json_bytes(result)
    with args.output.open("xb") as handle:
        handle.write(body)
    print(json.dumps({"summary_sha256": sha256(body).hexdigest(), "slurm_job_id": args.job}, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for tag in ("check", "assess"):
        command = commands.add_parser(tag)
        command.add_argument("--questions", type=Path, required=True)
        command.add_argument("--manifest", type=Path, required=True)
        if tag == "assess":
            for field in ("receipt", "output"):
                command.add_argument(f"--{field}", type=Path, required=True)
            command.add_argument("--job", required=True)
            command.add_argument("--freeze", required=True)
    gate = commands.add_parser("verify-gate")
    gate.add_argument("--summary", type=Path, required=True)
    gate.add_argument("--sha256", required=True)
    gate.add_argument("--job", required=True)
    gate.add_argument("--freeze", required=True)
    args = parser.parse_args()
    if args.command == "verify-gate":
        print(checked_gate(args.summary, args.sha256, args.job, args.freeze))
    else:
        if args.command == "assess" and (not solvers.JOB_RE.fullmatch(args.job) or not solvers.COMMIT_RE.fullmatch(args.freeze)):
            parser.error("Assessment needs the same job and freeze as the generator")
        assessment(args)


if __name__ == "__main__":
    main()
