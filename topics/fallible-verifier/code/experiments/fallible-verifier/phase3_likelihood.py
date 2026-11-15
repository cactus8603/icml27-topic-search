"""Blinded synthetic assay for a two-mapping signed likelihood score."""

import argparse
from collections import Counter
from datetime import datetime
from fractions import Fraction
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
from statistics import median
import subprocess

from phase3_feasibility import forced_choice, label_ids
from phase3_synthetic import verify_model
from round2.protocol import CONFIG, digest, write_json


ROOT = Path(__file__).resolve().parent
CASES = ROOT / "phase3_likelihood_cases.json"
GOLD = ROOT / "phase3_likelihood_gold.json"
RESULTS = ROOT / "results" / "phase3-likelihood"
FAMILIES = ("change", "share", "elapsed", "fraction")
FROZEN = (
    CASES.name, GOLD.name, "phase3_likelihood.py", "phase3_likelihood.sbatch",
    "phase3_likelihood.md", "test_phase3_likelihood.py", "phase3_feasibility.py", "phase3_synthetic.py",
    "round2/config.json", "round2/protocol.py",
)
ABSTAIN = math.log(1.25)
TOL = 1e-5


def verify_freeze(revision):
    frozen = subprocess.check_output(
        ["git", "rev-parse", "--verify", "--end-of-options", f"{revision}^{{commit}}"],
        text=True, cwd=ROOT,
    ).strip()
    if subprocess.check_output(["git", "status", "--porcelain", "--", *FROZEN], cwd=ROOT):
        raise ValueError("Likelihood assay inputs must be committed and clean")
    subprocess.run(["git", "diff", "--quiet", frozen, "HEAD", "--", *FROZEN],
                   check=True, cwd=ROOT)
    return frozen


def load_cases():
    cases = json.loads(CASES.read_text())
    if Counter(row["family"] for row in cases) != {family: 6 for family in FAMILIES}:
        raise ValueError("Expected six questions in each registered family")
    if len({row["id"] for row in cases}) != 24 or len({row["question"] for row in cases}) != 24:
        raise ValueError("Question identities must be unique")
    for row in cases:
        if set(row) != {"id", "family", "question", "candidates"}:
            raise ValueError("Unexpected field in the inference manifest")
        values = row["candidates"]
        if len(values) != 2 or len(set(values)) != 2 or any(type(v) is not int or v <= 0 for v in values):
            raise ValueError("Each question needs two distinct integer candidates")
    return cases


def solve_independently(row):
    text = row["question"]
    if row["family"] == "elapsed":
        start, end = [datetime.strptime(t, "%H:%M") for t in re.findall(r"\d{2}:\d{2}", text)]
        return Fraction(int((end - start).total_seconds()), 60)
    numbers = [int(value) for value in re.findall(r"\d+", text)]
    if row["family"] == "change":
        item, delivery, paid = numbers
        return paid - sum((item, delivery))
    if row["family"] == "share":
        total, groups = numbers
        return Fraction(total, groups)
    total, numerator, denominator = numbers
    return total * Fraction(numerator, denominator)


def load_gold(cases):
    gold = json.loads(GOLD.read_text())
    if set(gold) != {row["id"] for row in cases}:
        raise ValueError("Gold and inference question identities differ")
    positions = Counter()
    offsets = Counter()
    for row in cases:
        correct = gold[row["id"]]
        if type(correct) is not int or correct != solve_independently(row):
            raise ValueError("Independent arithmetic disagrees with the frozen reference")
        if correct not in row["candidates"]:
            raise ValueError("The reference answer is absent from its candidate pair")
        position = row["candidates"].index(correct)
        positions[row["family"], position] += 1
        offsets[row["family"], row["candidates"][1 - position] > correct] += 1
    if any(count != 3 for count in (*positions.values(), *offsets.values())):
        raise ValueError("Each family must balance correct positions and wrong offset directions")
    return gold


def tokenize(tokenizer, message, maximum):
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": message}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not prompt.endswith("</think>\n\n") or not 0 < len(ids) <= maximum:
        raise ValueError("Unexpected chat template or input length")
    return {"prompt_token_ids": ids}


def build_requests(cases, tokenizer, maximum):
    requests = {}
    for position, row in enumerate(cases):
        for swapped in (False, True):
            correct, incorrect = ("B", "A") if swapped else ("A", "B")
            definitions = [f"{correct} = correct", f"{incorrect} = incorrect"]
            if position % 2:
                definitions.reverse()
            for value in (*row["candidates"], None):
                shown = str(value) if value is not None else "[withheld]"
                message = (
                    "Check whether the candidate's final numerical answer is correct. "
                    f"Respond with one label. {'; '.join(definitions)}.\n\n"
                    f"Problem: {row['question']}\nCandidate answer: {shown}"
                )
                requests[row["id"], value, swapped] = tokenize(tokenizer, message, maximum)
    if len(requests) != 144 or len({tuple(r["prompt_token_ids"]) for r in requests.values()}) != 144:
        raise ValueError("Expected 144 distinct model inputs")
    return dict(sorted(requests.items(), key=lambda item: hashlib.sha256(
        repr(item[0]).encode("ascii")).digest()))


def logprobs_params(ids):
    from vllm import SamplingParams

    tokens = list(ids.values())
    return SamplingParams(
        allowed_token_ids=tokens, logprob_token_ids=tokens, logprobs=len(tokens),
        temperature=0, min_tokens=1, max_tokens=1, detokenize=False,
    )


def read_delta(result, ids):
    if not result.outputs:
        return None
    if len(result.outputs) != 1:
        raise ValueError("Unexpected number of one-step completions")
    completion = result.outputs[0]
    label = forced_choice(ids, completion.token_ids, completion.finish_reason)
    if label is None:
        if not completion.token_ids:
            return None
        raise ValueError("Returned token identity or stop reason violates the one-step protocol")
    if not completion.logprobs:
        return None
    if len(completion.logprobs) != 1:
        raise ValueError("Expected same-step logprobs")
    step = completion.logprobs[0]
    expected = set(ids.values())
    if not set(step) <= expected:
        raise ValueError("Unrequested token in native logprobs")
    if set(step) != expected:
        return None
    a, b = (step[ids[name]].logprob for name in ("A", "B"))
    if not all(math.isfinite(x) and x <= TOL for x in (a, b)):
        raise ValueError("Invalid native logprob")
    if not math.isclose(math.exp(a) + math.exp(b), 1.0, abs_tol=TOL):
        raise ValueError("Native scores are not normalized over the two-label allowlist")
    if (a if label == "A" else b) < max(a, b) - TOL:
        raise ValueError("Sampled token disagrees with the native greedy logprobs")
    return a - b


def score_batch(engine, requests, params, ids):
    returned_ids = set()
    deltas = {}
    results = engine.generate(list(requests.values()), params, use_tqdm=False)
    for (key, request), result in zip(requests.items(), results, strict=True):
        if (result.prompt_token_ids is None
                or list(result.prompt_token_ids) != request["prompt_token_ids"]
                or not isinstance(result.request_id, str) or not result.request_id
                or result.request_id in returned_ids):
            raise ValueError("Engine output identities were not validated")
        returned_ids.add(result.request_id)
        deltas[key] = read_delta(result, ids)
    return deltas


def two_mapping(deltas, question, candidate):
    standard = deltas[question, candidate, False]
    swapped = deltas[question, candidate, True]
    if standard is None or swapped is None:
        return None
    # Semantic contrast and letter preference are orthogonal under a mapping swap.
    return (standard - swapped) / 2, (standard + swapped) / 2


def sign_counts(positive, negative):
    buckets = {"correct_decisions": 0, "wrong_decisions": 0, "abstentions": 0}
    for score, target in ((positive, 1), (negative, -1)):
        if abs(score) <= ABSTAIN:
            buckets["abstentions"] += 1
        else:
            key = "correct_decisions" if target * score > 0 else "wrong_decisions"
            buckets[key] += 1
    return buckets


def summarize(cases, gold, deltas):
    methods = ("candidate_evidence", "candidate_blind_prior")
    family_totals = {}
    for family in (*FAMILIES, "all"):
        relevant = [row for row in cases if family == "all" or row["family"] == family]
        counts = {method: Counter() for method in methods}
        slot_abs = []
        semantic_abs = []
        missing = 0
        for row in relevant:
            key = row["id"]
            correct = gold[key]
            incorrect = next(value for value in row["candidates"] if value != correct)
            pos, neg, prior = (two_mapping(deltas, key, value) for value in (correct, incorrect, None))
            if pos is None or neg is None or prior is None:
                missing += 1
                continue
            semantic_abs.extend((abs(pos[0]), abs(neg[0])))
            slot_abs.extend((abs(pos[1]), abs(neg[1])))
            for method, (positive, negative) in zip(methods, ((pos[0], neg[0]), (prior[0], prior[0])), strict=True):
                counts[method].update(sign_counts(positive, negative))
                gap = positive - negative
                field = "pair_ties" if abs(gap) <= TOL else ("pair_wins" if gap > 0 else "pair_losses")
                counts[method][field] += 1
        fields = ("correct_decisions", "wrong_decisions", "abstentions",
                  "pair_wins", "pair_losses", "pair_ties")
        family_totals[family] = {
            "requested_question_pairs": len(relevant), "missing_question_pairs": missing,
            "missing_candidate_rows": 2 * missing,
            "median_abs_semantic_log_odds": median(semantic_abs) if semantic_abs else None,
            "median_abs_label_slot_log_odds": median(slot_abs) if slot_abs else None,
            **{method: {field: counts[method][field] for field in fields} for method in methods},
        }
    return {"missing_prompt_scores": sum(value is None for value in deltas.values()),
            "cohorts": family_totals}


def run_gpu(model_dir, role, config, requests, smoke, params, ids):
    from vllm import LLM

    engine = LLM(
        model=str(model_dir), tokenizer=str(model_dir),
        dtype="float16" if role == "strong" else "bfloat16",
        quantization="awq" if role == "strong" else None,
        max_model_len=config["max_input_tokens"] + config["max_output_tokens"],
        max_num_batched_tokens=config["tokens_per_batch"], max_num_seqs=64,
        gpu_memory_utilization=0.78, seed=config["seed"], enforce_eager=True,
        logprobs_mode="processed_logprobs",
    )
    if engine.llm_engine.model_config.logprobs_mode != "processed_logprobs":
        raise ValueError("Native sampler does not report processed logprobs")
    if any(value is None for value in score_batch(engine, smoke, params, ids).values()):
        raise ValueError("Same-step two-label logprobs are unavailable in the GPU smoke")
    return score_batch(engine, requests, params, ids)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("weak", "strong"), required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--freeze")
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    if not args.check_only:
        if not args.freeze or not args.run_dir:
            parser.error("GPU mode requires --freeze and --run-dir")
        run_dir = args.run_dir.resolve()
        if run_dir.parent != RESULTS.resolve() or not run_dir.name.isdecimal():
            parser.error("use a numeric job directory directly under results/phase3-likelihood/")
        output = run_dir / f"{args.role}.json"
        if output.exists():
            raise FileExistsError(output)
        frozen = verify_freeze(args.freeze)

    from transformers import AutoTokenizer

    version = importlib.metadata.version("vllm")
    if version != "0.33.0":
        raise ValueError("This assay requires vLLM 0.33.0")
    cases = load_cases()
    config = json.loads(CONFIG.read_text())
    specification = config["models"][args.role]
    verify_model(args.model_dir, specification, args.check_only)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), local_files_only=True)
    ids = label_ids(tokenizer)
    maximum = config["max_input_tokens"]
    requests = build_requests(cases, tokenizer, maximum)
    params = logprobs_params(ids)
    smoke = {
        label: tokenize(tokenizer, f"The two allowed choices are A and B. Say {label}.", maximum)
        for label in ("A", "B")
    }
    metadata = {
        "role": args.role, "model": specification["id"], "revision": specification["revision"],
        "vllm": version, "label_ids": ids, "assay_prompts": len(requests), "smoke_prompts": len(smoke),
        "abstain_abs_log_odds_at_or_below": ABSTAIN,
    }
    if args.check_only:
        load_gold(cases)
        print(json.dumps(metadata | {"check_only": True, "independent_gold": "ok"}, sort_keys=True))
        return

    deltas = run_gpu(args.model_dir, args.role, config, requests, smoke, params, ids)
    gold = load_gold(cases)
    metrics = summarize(cases, gold, deltas)
    write_json(output, {"freeze": frozen, "cases_sha256": digest(CASES), "gold_sha256": digest(GOLD)}
               | metadata | metrics)


if __name__ == "__main__":
    main()
