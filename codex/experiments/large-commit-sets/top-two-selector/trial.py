"""Zero-probe candidate choice on a fresh, fixed three-branch LLaDA pool."""

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import unicodedata

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
REPO = HERE.parents[3]
sys.path.insert(0, str(BASE / "round2"))
from design import CONFIG_PATH as ROUND2_PATH, pilot, read_rows, sha, split_ids, stable_order, verify_shards

CONFIG = HERE / "config.json"
FROZEN = (
    HERE / "config.json", HERE / "PREREG.md", HERE / "trial.py", HERE / "run.sbatch",
    HERE / "test_trial.py", BASE / "pilot.py", BASE / "read_parquet.py", BASE / "thresholds.json",
    ROUND2_PATH, BASE / "round2/design.py", BASE / "round2/artifacts.json",
)
POLICIES = ("confidence", "margin_only", "uniform_fixed", "top_two", "pool_oracle")
NUMERIC = re.compile(r"^-?[0-9]+(?:\.[0-9]+)?$")


def enforce_freeze(revision):
    paths = [str(path.relative_to(REPO)) for path in FROZEN]
    frozen = subprocess.check_output(
        ["git", "rev-parse", "--verify", "--end-of-options", f"{revision}^{{commit}}"],
        text=True, cwd=REPO,
    ).strip()
    if subprocess.check_output(["git", "status", "--porcelain", "--", *paths], cwd=REPO):
        raise ValueError("The trial and its dependencies need a clean reviewed freeze")
    subprocess.run(["git", "diff", "--quiet", frozen, "HEAD", "--", *paths], cwd=REPO, check=True)
    return frozen


def keys(question):
    literal = "".join(unicodedata.normalize("NFKC", question).casefold().split())
    template = re.sub(r"\d+", "#", "".join(ch for ch in literal if ch.isalnum()))
    return literal, template


def fresh_rows(train_path, test_path, math_path, parquet_python, config):
    original = json.loads(pilot.CONFIG_PATH.read_text())
    round2 = json.loads(ROUND2_PATH.read_text())
    train = read_rows(train_path, parquet_python, original["smoke"])
    test = read_rows(test_path, parquet_python, original["main"])
    previous = split_ids(train, test, round2, original)
    prior_train = set(round2["excluded_train_doc_ids"]) | set(previous["train"]) | set(previous["smoke"])
    excluded = [row["question"] for row in test] + [train[i]["question"] for i in prior_train]
    used_literal, used_template = map(set, zip(*(keys(question) for question in excluded)))
    if sha(math_path) != config["math500_sha256"]:
        raise ValueError("Unexpected MATH-500 snapshot")
    math_rows = [json.loads(line) for line in math_path.read_text().splitlines()]
    math_ids = {row["unique_id"] for row in math_rows}
    if len(math_rows) != len(math_ids) or len(math_ids) != 500:
        raise ValueError("MATH-500 must have 500 unique published IDs")

    choices = {
        "gsm8k_project_holdout": [
            (str(i), train[i]["question"], pilot.extract_answer(train[i]["answer"]))
            for i in stable_order(range(len(train)), config["seed"] + ":gsm")
            if i not in prior_train
        ],
        "math500_numeric": [
            (math_rows[i]["unique_id"], math_rows[i]["problem"],
             pilot.extract_answer("#### " + math_rows[i]["answer"].strip()))
            for i in stable_order(range(len(math_rows)), config["seed"] + ":math")
            if NUMERIC.fullmatch(math_rows[i]["answer"].strip())
        ],
    }
    samples = {"main": [], "smoke": []}
    for cohort, target in config["counts"].items():
        selected = []
        for doc_id, question, gold in choices[cohort]:
            literal, template = keys(question)
            if gold is None or not template or literal in used_literal or template in used_template:
                continue
            used_literal.add(literal)
            used_template.add(template)
            selected.append((cohort, doc_id, question, gold))
            if len(selected) == target + 1:
                break
        if len(selected) != target + 1:
            raise ValueError(f"Not enough disjoint questions: {cohort}")
        samples["smoke"].append(selected[0])
        samples["main"].extend(selected[1:])
    cohort_fingerprints = {
        mode: hashlib.sha256(json.dumps(
            [(source, doc_id, keys(question)[0]) for source, doc_id, question, _ in data],
            ensure_ascii=False).encode()).hexdigest()
        for mode, data in samples.items()
    }
    if config.get("sample_sha256") and config["sample_sha256"] != cohort_fingerprints:
        raise ValueError("Frozen question fingerprints changed")
    provenance = {
        "question_sha256": cohort_fingerprints, "blocked_train_ids": len(prior_train),
        "blocked_all_prior_test_rows": len(test), "numeric_math500_rows": len(choices["math500_numeric"]),
    }
    return samples, provenance, original


def selectors(ranked, pools, identity, seed):
    proposals = {proposal.position: proposal for proposal in ranked}
    margin_scores, combined_scores = [], []
    for pool in pools:
        records = [proposals[position] for position in pool]
        margins = [record.base_margin for record in records]
        if not all(math.isfinite(p.log_probability) and math.isfinite(p.base_margin)
                   and p.base_margin >= 0 for p in records):
            raise ValueError("Expected finite top-two statistics")
        margin = pilot.set_score(margins)
        log_prob = math.fsum(record.log_probability for record in records) / len(records)
        margin_scores.append(margin)
        combined_scores.append(log_prob + margin)
    ordered = range(len(pools))
    return {
        "confidence": 0,
        "margin_only": max(ordered, key=margin_scores.__getitem__),
        "uniform_fixed": stable_order(ordered, f"{seed}:uniform:{identity}")[0],
        "top_two": max(ordered, key=combined_scores.__getitem__),
    }


class Decoder(pilot.LLaDAGreedy):
    def evaluate_request(self, question, identity, seed):
        before = self.forward_calls
        x, prompt_length = self.encode(question)
        for step in range(self.step_count):
            ranked = self.proposals(x, prompt_length, step)
            if step in self.settings["anchors"]:
                pools, _ = pilot.choose_pool(ranked, self.settings, self.excluded_tokens)
                if pools:
                    choices = selectors(ranked, pools, identity, seed)
                    if self.forward_calls - before != step + 1:
                        raise ValueError("The selector increased prefix forwards")
                    completions = []
                    for pool in pools:
                        _, text, _, calls, _ = self.branch(x, prompt_length, step, pool, ranked)
                        if step + 1 + calls != self.step_count:
                            raise ValueError("The deployment branch did not use exactly 32 forwards")
                        completions.append(text)
                    evaluation_forwards = self.forward_calls - before
                    if evaluation_forwards != step + 1 + len(pools) * (self.step_count - step - 1):
                        raise ValueError("The collector used an unexpected additional forward")
                    return choices, completions, step, evaluation_forwards
            x = self.greedy_step(x, prompt_length, step, ranked)
        text, _ = self.completion(x, prompt_length)
        if self.forward_calls - before != self.step_count:
            raise ValueError("Unexpected forwards on confidence fallback")
        return {policy: 0 for policy in POLICIES if policy != "pool_oracle"}, [text], None, self.step_count


def score(choices, completions, gold):
    matches = [pilot.extract_answer(value) == gold for value in completions]
    return {key: int(matches[index]) for key, index in choices.items()} | {"pool_oracle": int(any(matches))}


def paired_summary(data, config):
    import numpy as np

    rng = np.random.default_rng(config["bootstrap_seed"])
    cohorts = defaultdict(list)
    for source, scores in data:
        cohorts[source].append(scores)
    differences = {}
    for source, rows in cohorts.items():
        differences[source] = np.array([row["top_two"] - row["confidence"] for row in rows])
    result = {}
    for source, rows in (*cohorts.items(), ("fixed_mixture", [scores for _, scores in data])):
        blocks = list(differences.values()) if source == "fixed_mixture" else [differences[source]]
        delta = np.concatenate(blocks)
        draws = sum(rng.choice(block, size=(config["bootstrap_resamples"], len(block)), replace=True).sum(axis=1)
                    for block in blocks) / len(delta)
        low, high = (float(x) for x in np.quantile(draws, [0.025, 0.975]))
        result[source] = {
            "n": len(rows), "correct": {policy: sum(row[policy] for row in rows) for policy in POLICIES},
            "top_two_rescues": int((delta > 0).sum()), "top_two_regressions": int((delta < 0).sum()),
            "top_two_minus_confidence": float(delta.mean()), "paired_ci95": [low, high],
        }
    return result


def follow_up(summary, threshold):
    low, high = summary["paired_ci95"]
    if high < threshold:
        return "do_not_expand"
    if low > 0 and summary["top_two_minus_confidence"] >= threshold:
        return "independent_confirmation"
    return "inconclusive"


def decoder_for_model(args, original):
    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer

    checkpoint = original["checkpoint"]
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("The pinned LLaDA path requires a CUDA GPU with BF16")
    if transformers.__version__ != checkpoint["transformers"]:
        raise RuntimeError("Unexpected Transformers runtime")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModel.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True, use_safetensors=True,
        torch_dtype=torch.bfloat16, device_map={"": 0},
    ).eval()
    if (model.config.mask_token_id != checkpoint["mask_token"] or
            tokenizer.convert_tokens_to_ids("<|eot_id|>") not in checkpoint["stop_tokens"]):
        raise RuntimeError("Unexpected model special tokens")
    return Decoder(torch, model, tokenizer, original["decoder"], checkpoint)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("check", "smoke", "main"), required=True)
    for name in ("model", "train", "test", "math", "parquet-python"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--freeze")
    args = parser.parse_args()
    if args.mode != "check":
        if not args.freeze:
            parser.error("GPU modes require --freeze")
        frozen = enforce_freeze(args.freeze)
    config = json.loads(CONFIG.read_text())
    samples, provenance, original = fresh_rows(args.train, args.test, args.math, args.parquet_python, config)
    pilot.verify_model_path(args.model, original["checkpoint"])
    verify_shards(args.model)
    if args.mode == "check":
        print(json.dumps(provenance | {"samples": {k: len(v) for k, v in samples.items()},
                                      "model_revision": original["checkpoint"]["revision"]}, sort_keys=True))
        return
    if not config.get("sample_sha256"):
        raise ValueError("Pin the selected question fingerprints before starting a GPU mode")
    output = None
    if args.mode == "main":
        job = os.getenv("SLURM_JOB_ID", "")
        if not re.fullmatch(r"[1-9][0-9]*", job):
            parser.error("main requires a valid Slurm job ID")
        output = HERE / "results" / f"summary-{job}.json"
        if output.exists():
            raise FileExistsError(output)
    decoder = decoder_for_model(args, original)
    features = []
    eligible = defaultdict(int)
    started = time.monotonic()
    with decoder.torch.inference_mode():
        for index, (cohort, doc_id, question, gold) in enumerate(samples[args.mode], start=1):
            selected, completions, anchor, _ = decoder.evaluate_request(
                question, f"{cohort}/{doc_id}", config["seed"])
            eligible[cohort] += anchor is not None
            if args.mode == "main":
                features.append((cohort, score(selected, completions, gold)))
                if index % 64 == 0:
                    print(f"[codex] Completed {index}/{len(samples[args.mode])} matched requests.")
    telemetry = {
        "elapsed_seconds_after_load": time.monotonic() - started,
        "peak_gpu_allocated_bytes": decoder.torch.cuda.max_memory_allocated(),
        "peak_gpu_reserved_bytes": decoder.torch.cuda.max_memory_reserved(),
        "diagnostic_total_forwards": decoder.forward_calls, "deployment_forwards_per_prompt": decoder.step_count,
    }
    if output is None:
        print(json.dumps({"mode": args.mode, "freeze": frozen, **telemetry}, sort_keys=True))
        return
    metrics = paired_summary(features, config)
    payload = {
        "freeze": frozen, "job_id": int(job), "config_sha256": sha(CONFIG), "data": provenance,
        "model_revision": original["checkpoint"]["revision"], "eligible_by_cohort": dict(eligible),
        "summary": metrics, "exploratory_follow_up": follow_up(metrics["fixed_mixture"],
                                                               config["minimum_gain_for_follow_up"]), **telemetry,
    }
    output.parent.mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=output.parent, prefix=f".{output.name}.",
                                     suffix=".part", delete=False) as file:
        partial = Path(file.name)
        try:
            json.dump(payload, file, sort_keys=True, allow_nan=False)
            file.write("\n")
            file.close()
            os.link(partial, output)
        finally:
            partial.unlink()


if __name__ == "__main__":
    main()
