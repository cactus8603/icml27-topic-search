"""Check the three Spanish generation receipts and prepare offline review data."""

import argparse
import asyncio
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
GENERATORS = HERE.parent / "lambda-author-generation"
sys.path.insert(0, str(GENERATORS))
import generate as generation


GEN_JOB = "26533"
GEN_FREEZE = "8c391fd4e72ed998732034910db1bd13afe4c784"
AUTHOR_JUDGE = "grad_shift/evaluation/judge.py"
AUTHOR_JUDGE_SHA = "c54c953d9f3c1008bc5d62a36b31cafc561fec66bbe0d3e625b94bc51734377b"
ARMS = ("base", "none", "ip")
JUDGE = {"model": "gpt-4.1-mini-2025-04-14", "max_tokens": 1, "temperature": 0,
         "logprobs": True, "top_logprobs": 20, "seed": 0}
SAMPLING = {"backend": "hf", "seed": 42, "temperature": 1.0, "top_p": 1.0,
            "top_k": 20, "repetition_penalty": 1.05, "eos_token_id": [151645, 151643],
            "batch_size": 20, "max_new_tokens": 500}
SAMPLE_RULE = "spanish-question-matched-review-20261115"


def encode(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                       separators=(",", ":")) + "\n").encode("utf-8")


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def subset_index(question_id, samples):
    return int(sha(encode([SAMPLE_RULE, question_id]))[:16], 16) % samples + 1


def load_arm(root, arm, train_receipt, prompt_fingerprints, freeze):
    directory = root / arm
    receipt_bytes = (directory / "receipt.json").read_bytes()
    receipt = json.loads(receipt_bytes)
    data = (directory / "responses.jsonl").read_bytes()
    expected = {
        "stage": "spanish_generation_only", "arm": arm, "author_commit": generation.AUTHOR,
        "launcher_commit": freeze, "inputs_sha256": generation.BINDING,
        "training_receipt_sha256": train_receipt, "model_revision": generation.MODEL_REVISION,
        "questions_sha256": generation.AUTHOR_HASHES[generation.QUESTIONS],
        "source_hashes": generation.AUTHOR_HASHES, "sampling": SAMPLING,
        "tokenizer": {"pad_token_id": 151645, "eos_token_id": 151645, "padding_side": "left"},
        "rows": len(prompt_fingerprints), "answers_sha256": sha(data),
    }
    if any(receipt.get(key) != val for key, val in expected.items()):
        raise ValueError(f"Generation receipt or response bytes differ for {arm}")
    rows = {}
    for line in data.splitlines():
        row = json.loads(line)
        if row.keys() != {"id", "question_id", "sample", "prompt_sha256", "answer"}:
            raise ValueError("Invalid frozen generator row schema")
        qid, sample = row["question_id"], row["sample"]
        key = (qid, sample)
        if type(sample) is not int or key not in prompt_fingerprints or key in rows or not isinstance(row["answer"], str):
            raise ValueError("Duplicate or out-of-grid generation row")
        expected_id = sha(f"{generation.BINDING}:{arm}:{qid}:{sample}".encode())[:12]
        if row["id"] != expected_id or row["prompt_sha256"] != prompt_fingerprints[key]:
            raise ValueError("Generation response has unexpected question identity")
        rows[key] = row["answer"]
    if rows.keys() != prompt_fingerprints.keys() or receipt.get("samples_by_question") != dict(Counter(q for q, _ in rows)):
        raise ValueError("The complete matched grid of generation rows is required")
    if receipt.get("empty_after_decode") != sum(not answer.strip() for answer in rows.values()):
        raise ValueError("Blank completions disagree with the receipt")
    return rows, sha(receipt_bytes)


def build_packets(questions, rubric, responses, salt, samples):
    if set(responses) != set(ARMS) or any(
        set(rows) != {(qid, sample) for qid in questions for sample in range(1, samples + 1)}
        for rows in responses.values()
    ):
        raise ValueError("Incomplete matched generation panel")
    selected = {qid: subset_index(qid, samples) for qid in questions}
    blind, requests, mapping = [], [], {}
    for arm, rows in responses.items():
        for (qid, sample), answer in rows.items():
            opaque = sha(encode([salt, arm, qid, sample]))[:12]
            if opaque in mapping:
                raise ValueError("Blinded response identifier collision")
            prompt = rubric.format(question=questions[qid], answer=answer)
            requests.append({"id": opaque, "request": JUDGE | {"messages": [{"role": "user", "content": prompt}]}})
            mapping[opaque] = {"arm": arm, "question_id": qid, "sample": sample,
                               "judge_prompt_sha256": sha(prompt.encode("utf-8"))}
            if sample == selected[qid]:
                blind.append({"id": opaque, "question_id": qid, "answer": answer})
    return {
        "pilot.json": encode({"scope": "independent_diagnostic_only", "rubric": rubric, "questions": questions,
                              "responses": sorted(blind, key=lambda row: row["id"])}),
        "author_requests.jsonl": b"".join(encode(row) for row in sorted(requests, key=lambda row: row["id"])),
        "mapping.json": encode({"salt": salt, "pilot_sampling_rule": SAMPLE_RULE, "mapping": mapping}),
    }


def plan(source):
    generation.verify_inputs(source)
    if generation.digest(source / AUTHOR_JUDGE) != AUTHOR_JUDGE_SHA:
        raise ValueError("The locked author judge implementation differs")
    sys.path.insert(0, str(source))
    from grad_shift.chat_template import render_chat
    from grad_shift.evaluation.questions import load_trait_questions
    from transformers import AutoTokenizer

    questions = load_trait_questions(trait="spanish", version="extract", system_prompt=generation.SYSTEM)
    ids, chats = generation.conversation_plan(questions)
    tokenizer = AutoTokenizer.from_pretrained(str(generation.BASE), local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    fingerprints = {key: sha(render_chat(tokenizer, chat, add_generation_prompt=True).encode())
                    for key, chat in zip(ids, chats, strict=True)}
    rubric = questions[0].judge_prompts["spanish"]
    if any(q.judge_prompts["spanish"] != rubric for q in questions):
        raise ValueError("Expected one shared Spanish rubric")
    return {q.id: q.text for q in questions}, rubric, fingerprints


def check_native_request(rubric, question):
    from grad_shift.evaluation.judge import OpenAiJudge

    actual = {}

    async def mock_create(**kwargs):
        actual.update(kwargs)
        token = SimpleNamespace(token="60", logprob=0.0)
        logprobs = SimpleNamespace(content=[SimpleNamespace(top_logprobs=[token])])
        return SimpleNamespace(choices=[SimpleNamespace(logprobs=logprobs)])

    class OfflineJudge(OpenAiJudge):
        @classmethod
        def _get_client(cls):
            return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=mock_create)))

    answer = "A synthetic response used to verify the offline export."
    score = asyncio.run(OfflineJudge(JUDGE["model"], rubric, "0_100")(question=question, answer=answer))
    expected = JUDGE | {"messages": [{"role": "user", "content": rubric.format(question=question, answer=answer)}]}
    if actual != expected or score != 60:
        raise ValueError("The offline request differs from the author's locked invocation")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--source", required=True, type=Path)
    cli.add_argument("--generation-freeze", required=True)
    cli.add_argument("--freeze", required=True)
    mode = cli.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--output", type=Path)
    args = cli.parse_args()
    if args.generation_freeze != GEN_FREEZE or not re.fullmatch(r"[0-9a-f]{40}", args.freeze):
        cli.error("Use the reviewed generation and review commits")
    questions, rubric, fingerprints = plan(args.source)
    check_native_request(rubric, next(iter(questions.values())))
    note = {"generation_job": GEN_JOB, "generation_freeze": args.generation_freeze,
            "review_freeze": args.freeze, "questions": len(questions),
            "samples_per_question": generation.SAMPLES, "rubric_sha256": sha(rubric.encode()),
            "author_judge_sha256": AUTHOR_JUDGE_SHA}
    if args.check_only:
        print(json.dumps(note | {"stage": "cpu_preflight_only"}, sort_keys=True))
        return

    job = os.environ.get("SLURM_JOB_ID", "")
    expected = generation.RUNS / f"author-review-{job}"
    if not re.fullmatch(r"[1-9][0-9]*", job) or args.output.resolve() != expected.resolve():
        cli.error("Use an isolated directory bound to the Slurm review job")
    root = generation.RUNS / f"author-generation-{GEN_JOB}"
    responses, receipts = {}, {}
    for arm in ARMS:
        _, training_receipt = generation.model_path(arm, verify_weights=False)
        responses[arm], receipts[arm] = load_arm(root, arm, training_receipt, fingerprints, args.generation_freeze)
    payloads = build_packets(questions, rubric, responses, secrets.token_hex(16), generation.SAMPLES)
    args.output.mkdir(mode=0o700, parents=False, exist_ok=False)
    os.umask(0o077)
    report = note | {"stage": "review_inputs_only", "slurm_job": job, "generation_receipts": receipts,
                     "pilot_rows": len(questions) * len(ARMS), "author_request_rows": len(fingerprints) * len(ARMS),
                     "files": {name: {"sha256": sha(data), "bytes": len(data)} for name, data in payloads.items()}}
    for name, payload in (payloads | {"receipt.json": encode(report)}).items():
        with (args.output / name).open("xb") as output:
            output.write(payload)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
