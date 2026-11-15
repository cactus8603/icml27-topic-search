"""Run the original pinned Oat solver on a separate, public-only ASDiv question bundle."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "instruct-candidate-coverage"))
from generate_instruct import verify_questions
sys.path.insert(0, str(HERE.parent / "gsm1k-replication"))
import solvers


SCOPE = "asdiv_oat_proposals"
ASDIV_SHA = "110881d5e9a7ba12d2776a30dbebe8ef919f76a81ef39b2a829790ffd2b2b8d8"


def identity(questions, manifest_sha, questions_sha, freeze, job, warm_gate_sha):
    if (len(questions) != 997 or manifest_sha != ASDIV_SHA or not solvers.SHA_RE.fullmatch(questions_sha)
            or not solvers.SHA_RE.fullmatch(warm_gate_sha)):
        raise ValueError("A stable ASDiv roster and Instruct warm qualification are required")
    return solvers.receipt_identity("oat", SCOPE, freeze, job, questions_sha, manifest_sha, len(questions)) | {
        "asdiv_oat_wrapper_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "warm_gate_sha256": warm_gate_sha,
    }


def verify_proposals(receipt_path, expected_receipt_sha256, questions, manifest_sha256, *, expected_freeze,
                     expected_job, expected_warm_gate_sha256):
    receipt_bytes = receipt_path.read_bytes()
    if sha256(receipt_bytes).hexdigest() != expected_receipt_sha256:
        raise ValueError("ASDiv Oat receipt SHA differs")
    receipt = json.loads(receipt_bytes)
    question_payload = b"".join(solvers.json_bytes(row) for row in questions)
    questions_sha = sha256(question_payload).hexdigest()
    expected = identity(questions, manifest_sha256, questions_sha, expected_freeze, expected_job,
                        expected_warm_gate_sha256)
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("Oat solver, author panel or active Slurm stage differs")
    if (not solvers.SHA_RE.fullmatch(receipt["prompt_ids_sha256"]) or not isinstance(receipt["gpu"]["name"], str)
            or type(receipt["gpu"]["total_memory_bytes"]) is not int or receipt["gpu"]["total_memory_bytes"] <= 0):
        raise ValueError("Missing Oat solver hardware or input provenance")
    rows_bytes = receipt_path.with_name(solvers.ROWS_BASENAME).read_bytes()
    if (len(rows_bytes), sha256(rows_bytes).hexdigest()) != (receipt["rows_bytes"], receipt["rows_sha256"]):
        raise ValueError("ASDiv Oat candidate file differs from its receipt")
    rows = rows_bytes.splitlines()
    if len(rows) != len(questions) * solvers.DRAW_COUNT:
        raise ValueError("Unexpected ASDiv Oat draw count")
    expected_roster = ((row["question_id"], f"oat-{index}") for row in questions for index in range(solvers.DRAW_COUNT))
    candidates = {}
    for line, key in zip(rows, expected_roster, strict=True):
        row = json.loads(line)
        if (set(row) != {"question_id", "variant", "solution", "finish_reason"}
                or (row["question_id"], row["variant"]) != key or not isinstance(row["solution"], str)
                or row["finish_reason"] not in {"stop", "length"}):
            raise ValueError("Invalid Oat ASDiv question/draw alignment or finish reason")
        candidates[key] = {"solution": row["solution"], "finish_reason": row["finish_reason"]}
    return receipt, candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--freeze")
    parser.add_argument("--warm-gate-sha256")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    job = os.environ.get("SLURM_JOB_ID", "")
    if args.check_only:
        if any(value is not None for value in (args.freeze, args.warm_gate_sha256, args.output_dir)):
            parser.error("Check-only verifies tokenizer, model files and public ASDiv input")
    elif (not solvers.JOB_RE.fullmatch(job) or not solvers.COMMIT_RE.fullmatch(args.freeze or "")
          or not solvers.SHA_RE.fullmatch(args.warm_gate_sha256 or "") or args.output_dir is None):
        parser.error("Generation needs a Slurm job, reviewed freeze and passing warm qualification")

    questions, _ = verify_questions("asdiv-full", args.questions, args.manifest, ASDIV_SHA)
    questions_sha = sha256(args.questions.read_bytes()).hexdigest()
    model, prompts = solvers.preflight("oat", solvers.REFERENCE, questions, solvers.MAX_TOKENS)
    if args.check_only:
        print(json.dumps({"scope": "asdiv_oat_preflight", "n_questions": len(questions), "questions_sha256": questions_sha,
                          "model_snapshot": solvers.MODELS["oat"][1], "max_prompt_tokens": max(map(len, prompts[1]))}, sort_keys=True))
        return

    receipt = identity(questions, ASDIV_SHA, questions_sha, args.freeze, job, args.warm_gate_sha256)
    args.output_dir.mkdir()
    rows, evidence = solvers.generate("oat", model, prompts, questions, receipt)
    (args.output_dir / solvers.ROWS_BASENAME).write_bytes(rows)
    raw = solvers.json_bytes(receipt | evidence)
    (args.output_dir / "receipt.json").write_bytes(raw)
    print(json.dumps({"scope": SCOPE, "job": job, "receipt_sha256": sha256(raw).hexdigest()}, sort_keys=True))


if __name__ == "__main__":
    main()
