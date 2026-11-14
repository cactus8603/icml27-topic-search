"""Compare the two published Abel prompt paths on previously unused MATH questions."""

import argparse
from collections import Counter
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re
import sys
import unicodedata

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
TOPIC = ROOT / "killtests/rl-sparse-distill"
sys.path.insert(0, str(TOPIC / "scripts"))
from common import STOP_IDS, cfg, pairs_cfg
from pair_b import model_path
from pair_b_upstream import score, upstream_rows
from artifacts import atomic_json, read_json, require, sha256


N = 768
SEED = 2026111507
MATH_FILE = "data/math/test.jsonl"
MATH_SHA = "0e9257f4e3481c45c1ba0ae3b09a14c7a751ed30a018f41f317c64537dac8a5a"
LOCAL_EVAL_SHA = "d8f9beca6a745e402509e379efa122e9df02e913467f2483c32d8a18a593163a"
ARMS = ("raw", "chat")


def lines(path):
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source]


def question_key(text):
    return "".join(unicodedata.normalize("NFKC", text).casefold().split())


def seeded_digest(key):
    return hashlib.sha256(f"{SEED}:{key}".encode()).hexdigest()


def select(questions, exclusions, size):
    candidates = {}
    for index, text in enumerate(questions):
        key = question_key(text)
        if key and key not in exclusions:
            candidates.setdefault(key, index)
    require(len(candidates) >= size, "Too few distinct MATH questions after excluding prior panels")
    return [candidates[key] for key in sorted(candidates, key=seeded_digest)[:size]], len(candidates)


def prepare(shared):
    settings = cfg("pair_b.json")
    cache = shared / "tmp" / settings["upstream_cache"]
    previous = TOPIC / "data/eval.jsonl"
    require(sha256(previous) == LOCAL_EVAL_SHA, "Local previous question panel changed")
    require(sha256(cache / MATH_FILE) == MATH_SHA, "Pinned MATH input changed")
    old = [row["question"] for row in lines(previous)]
    for source in settings["reproduction"]["datasets"]:
        old.extend(row.get("problem", row.get("question")) for row in lines(cache / f"data/{source}/test.jsonl"))
    math = [row["problem"] for row in lines(cache / MATH_FILE)]
    indices, available = select(math, {question_key(text) for text in old}, N)
    settings["upstream_files"][MATH_FILE] = MATH_SHA
    settings["reproduction"]["datasets"] = {"math": 5000}
    parsed = {key: (name, prompt, gold) for key, name, prompt, gold in upstream_rows(cache, settings)}
    rows = []
    for index in indices:
        key = f"math:{index}"
        name, prompt, gold = parsed[key]
        rows.append([key, name, prompt, gold])
    membership = [seeded_digest(question_key(math[index])) for index in indices]
    return {"rows": rows, "remaining_questions": available, "membership": membership}


def tokenizer_and_requests(shared, panel):
    from transformers import AutoTokenizer

    config, settings = pairs_cfg(), cfg("pair_b.json")
    path = model_path(shared, "rl", settings, config)
    require(version("vllm") == "0.33.0", "vLLM version differs from the frozen run")
    index = path / "model.safetensors.index.json"
    shards = set(read_json(index)["weight_map"].values()) if index.exists() else {"model.safetensors"}
    require(all((path / name).is_file() for name in shards), "RL snapshot shards are missing")
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    spec = settings["reproduction"]
    require(tokenizer.convert_tokens_to_ids(["<|im_end|>", "<|endoftext|>"]) == STOP_IDS,
            "Unexpected stop-token IDs")
    requests = {}
    for key, _, prompt, _ in panel["rows"]:
        chat = tokenizer.apply_chat_template([{"role": "user", "content": prompt.strip()}],
                                             tokenize=False, add_generation_prompt=True)
        require(chat != prompt and prompt.strip() in chat, "The chat wrapper changed the problem text")
        for arm, text in zip(ARMS, (prompt, chat), strict=True):
            tokens = tokenizer.encode(text, add_special_tokens=False)
            require(len(tokens) + spec["max_tokens"] <= spec["max_model_len"], "Budget would truncate a prompt")
            requests[key, arm] = tokens
    return path, config, settings, requests


def sample(shared, panel):
    from vllm import LLM, SamplingParams

    path, config, settings, requests = tokenizer_and_requests(shared, panel)
    spec = settings["reproduction"]
    engine = LLM(str(path), dtype=config["dtype"], generation_config="vllm",
                 max_model_len=spec["max_model_len"], gpu_memory_utilization=settings["gpu_memory_utilization"],
                 seed=spec["seed"])
    parameters = [SamplingParams(n=1, **spec["rl"], max_tokens=spec["max_tokens"], stop=spec["stop"],
                                 stop_token_ids=STOP_IDS, seed=int(seeded_digest(key)[:8], 16))
                  for key, _ in requests]
    inputs = [{"prompt_token_ids": ids} for ids in requests.values()]
    generated = engine.generate(inputs, parameters, use_tqdm=False)
    output = {arm: {} for arm in ARMS}
    for ((key, arm), tokens), result in zip(requests.items(), generated, strict=True):
        require(list(result.prompt_token_ids) == tokens and len(result.outputs) == 1,
                "Generation order or shape does not match the paired prompts")
        completion = result.outputs[0]
        require(completion.finish_reason in ("stop", "length"), "Missing completion status")
        output[arm][key] = {"texts": [completion.text], "length_stop": completion.finish_reason == "length"}
    return output


def summarize(raw, chat):
    raw, chat = np.asarray(raw, dtype=int), np.asarray(chat, dtype=int)
    d = chat - raw
    rng = np.random.default_rng(SEED)
    boot = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(6000)]) * 100
    return {
        "raw_correct": int(raw.sum()), "chat_correct": int(chat.sum()), "n": len(d),
        "chat_only_correct": int((d == 1).sum()), "raw_only_correct": int((d == -1).sum()),
        "chat_minus_raw_pp": float(d.mean() * 100), "ci95_pp": np.quantile(boot, [0.025, 0.975]).tolist(),
    }


def grade(shared, panel, output, freeze):
    # Recheck the frozen input and activate the pinned author scorer in this subprocess.
    require(panel == prepare(shared), "Prepared membership differs from frozen inputs")
    spec = cfg("pair_b.json")["reproduction"]
    scores = {arm: score(panel["rows"], output[arm], spec["stop"]) for arm in ARMS}
    require([row[0] for row in scores["raw"]] == [row[0] for row in scores["chat"]],
            "Paired grading rows differ")
    return {
        **summarize(*([row[2] for row in scores[arm]] for arm in ARMS)),
        "length_stops": {arm: sum(row["length_stop"] for row in output[arm].values()) for arm in ARMS},
        "grading_timeouts": {arm: sum(row[3] for row in scores[arm]) for arm in ARMS},
        "empty_gold": sum(not str(row[3] or "").strip() for row in panel["rows"]),
        "eligible_pool": panel["remaining_questions"],
        "membership_sha256": hashlib.sha256("\n".join(panel["membership"]).encode()).hexdigest(),
        "model_revision": cfg("pair_b.json")["snapshots"]["rl"],
        "vllm": version("vllm"), "commit": freeze,
        "scope": "template effect on an unused project panel; Gate 0(a) unchanged",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "prepare", "sample", "score"))
    parser.add_argument("--shared", required=True, type=Path)
    parser.add_argument("--scratch", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--freeze")
    args = parser.parse_args()
    scratch = args.scratch
    path = scratch / "panel.json"
    if args.action == "check":
        panel = read_json(path)
        _, _, _, prompts = tokenizer_and_requests(args.shared, panel)
        counts = Counter(arm for _, arm in prompts)
        print(json.dumps({"selected": counts, "eligible_pool": panel["remaining_questions"]}))
    elif args.action == "prepare":
        atomic_json(path, prepare(args.shared))
    elif args.action == "sample":
        atomic_json(scratch / "generations.json", sample(args.shared, read_json(path)))
    else:
        require(args.output and args.freeze and re.fullmatch("[a-f0-9]{40}", args.freeze),
                "Output and full frozen commit are required")
        atomic_json(args.output, grade(args.shared, read_json(path), read_json(scratch / "generations.json"), args.freeze))


if __name__ == "__main__":
    main()
