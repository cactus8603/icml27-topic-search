"""Isolate the two lexical stopping strings in the frozen SimpleRL-Zoo run."""

import argparse
from collections import Counter
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "rl-template"))
import probe as template
from artifacts import atomic_json, read_json, require, sha256
from common import STOP_IDS, cfg
from pair_b_upstream import score, upstream_rows


N = 384
PREVIOUS = 768
BOOTSTRAP_SEED = 2026111511
LEXICAL = ("Question:", "Answer:")
ARMS = ("author", "native")


def prepare(shared):
    settings = cfg("pair_b.json")
    cache = shared / "tmp" / settings["upstream_cache"]
    previous = template.TOPIC / "data/eval.jsonl"
    require(template.N == PREVIOUS and sha256(previous) == template.LOCAL_EVAL_SHA,
            "Previous template panel or local evaluation input differs")
    require(sha256(cache / template.MATH_FILE) == template.MATH_SHA, "Pinned MATH input differs")
    settings["upstream_files"][template.MATH_FILE] = template.MATH_SHA
    excluded = [row["question"] for row in template.lines(previous)]
    for source in settings["reproduction"]["datasets"]:
        path = cache / f"data/{source}/test.jsonl"
        require(sha256(path) == settings["upstream_files"][f"data/{source}/test.jsonl"], "Prior panel differs")
        excluded.extend(row.get("problem", row.get("question")) for row in template.lines(path))
    questions = [row["problem"] for row in template.lines(cache / template.MATH_FILE)]
    ordered, available = template.select(questions, {template.question_key(text) for text in excluded}, PREVIOUS + N)
    fresh = ordered[PREVIOUS:]
    settings["reproduction"]["datasets"] = {"math": 5000}
    parsed = {row[0]: row for row in upstream_rows(cache, settings)}
    rows = [list(parsed[f"math:{i}"]) for i in fresh]
    membership = [template.seeded_digest(template.question_key(questions[i])) for i in fresh]
    require(len(set(membership)) == N and all(row[-1] is not None and str(row[-1]).strip() for row in rows),
            "Diagnostic requires unique questions and a nonempty author reference for every row")
    return {"rows": rows, "membership": membership, "remaining_questions": available}


def stop_sets(spec):
    author = spec["stop"]
    require(author == ["</s>", "<|im_end|>", "<|endoftext|>", *LEXICAL], "Author stop order differs")
    return {"author": author, "native": author[:-len(LEXICAL)]}


def check_sampler_runtime():
    import sympy
    from torch.fx.experimental.sym_node import _optimized_add

    left, right = sympy.symbols("rsd_left rsd_right")
    require(_optimized_add(left, right)[1] == left + right, "Sampler symbolic addition differs")
    return sympy.__version__


def sample(shared, panel):
    from vllm import LLM, SamplingParams

    path, config, settings, prompts = template.tokenizer_and_requests(shared, panel)
    spec = settings["reproduction"]
    stops = stop_sets(spec)
    engine = LLM(str(path), dtype=config["dtype"], generation_config="vllm",
                 max_model_len=spec["max_model_len"], gpu_memory_utilization=settings["gpu_memory_utilization"],
                 seed=spec["seed"])
    requests, params, wire = [], [], []
    for key, *_ in panel["rows"]:
        tokens = prompts[key, "raw"]
        seed = int(template.seeded_digest(key)[:8], 16)
        for arm in ARMS:
            requests.append((key, arm, tokens))
            wire.append({"prompt_token_ids": tokens})
            params.append(SamplingParams(n=1, **spec["rl"], max_tokens=spec["max_tokens"], stop=stops[arm],
                                         stop_token_ids=STOP_IDS, seed=seed))
    output = {arm: {} for arm in ARMS}
    for (key, arm, tokens), result in zip(requests, engine.generate(wire, params, use_tqdm=False), strict=True):
        require(list(result.prompt_token_ids) == tokens and len(result.outputs) == 1,
                "Generation does not match the frozen paired prompt order")
        completion = result.outputs[0]
        limit = completion.finish_reason == "length"
        require(completion.finish_reason in ("stop", "length") and len(completion.token_ids) <= spec["max_tokens"],
                "Unexpected completion reason or budget")
        output[arm][key] = {"texts": [completion.text], "length_stop": limit,
                            "stop_reason": completion.stop_reason, "tokens": len(completion.token_ids)}
    return output


def lexical_prefix(text):
    for delimiter in LEXICAL:
        text = text.split(delimiter, 1)[0]
    return text.rstrip()


def contrast(before, after):
    delta = np.asarray(after, dtype=int) - np.asarray(before, dtype=int)
    indices = np.random.default_rng(BOOTSTRAP_SEED).integers(len(delta), size=(6000, len(delta)))
    interval = np.quantile(delta[indices].mean(axis=1) * 100, [0.025, 0.975])
    return {"difference_pp": float(delta.mean() * 100), "ci95_pp": interval.tolist(),
            "improved": int((delta == 1).sum()), "worse": int((delta == -1).sum())}


def receipt(shared, panel, output, freeze):
    require(panel == prepare(shared), "Prepared membership differs from frozen inputs")
    settings = cfg("pair_b.json")
    spec = settings["reproduction"]
    stops = stop_sets(spec)
    author, native = (output[arm] for arm in ARMS)
    scores = {"author": score(panel["rows"], author, stops["author"]),
              "native_shadow": score(panel["rows"], native, stops["author"]),
              "native": score(panel["rows"], native, stops["native"])}
    keys = [row[0] for row in panel["rows"]]
    require(all([row[0] for row in arm] == keys for arm in scores.values()), "Scoring order differs")
    correct = {arm: np.asarray([row[2] for row in rows], dtype=int) for arm, rows in scores.items()}
    same = [author[key]["texts"][0].rstrip() == lexical_prefix(native[key]["texts"][0]) for key in keys]
    lexical = [author[key]["stop_reason"] in LEXICAL for key in keys]
    return {
        "scope": "exploratory stop-string sensitivity; no qualification verdict",
        "questions": len(keys), "eligible_pool": panel["remaining_questions"],
        "membership_sha256": hashlib.sha256("\n".join(panel["membership"]).encode()).hexdigest(),
        "commit": freeze, "model_revision": settings["snapshots"]["rl"], "vllm": version("vllm"),
        "sampling": spec["rl"] | {"max_tokens": spec["max_tokens"], "max_model_len": spec["max_model_len"],
                                   "engine_seed": spec["seed"], "stop_ids": STOP_IDS, "stops": stops},
        "correct": {arm: int(values.sum()) for arm, values in correct.items()},
        "timeouts": {arm: sum(row[3] for row in rows) for arm, rows in scores.items()},
        "native_minus_author": contrast(correct["author"], correct["native"]),
        "native_shadow_minus_author": contrast(correct["author"], correct["native_shadow"]),
        "native_minus_its_shadow": contrast(correct["native_shadow"], correct["native"]),
        "author_lexical_stop_count": sum(lexical),
        "author_matches_shadow_text": sum(same),
        "author_lexical_matches_shadow_text": sum(a and b for a, b in zip(lexical, same, strict=True)),
        "length_stops": {arm: sum(row["length_stop"] for row in output[arm].values()) for arm in ARMS},
        "stop_reasons": {arm: dict(Counter(str(row["stop_reason"]) for row in output[arm].values())) for arm in ARMS},
        "generated_tokens": {arm: sum(row["tokens"] for row in output[arm].values()) for arm in ARMS},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "check-score", "check", "sample", "score"))
    parser.add_argument("--shared", required=True, type=Path)
    parser.add_argument("--scratch", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--freeze")
    args = parser.parse_args()
    panel_path = args.scratch / "panel.json"
    if args.action in ("prepare", "check-score"):
        panel = prepare(args.shared)
        atomic_json(panel_path, panel)
        if args.action == "check-score":
            stops = stop_sets(cfg("pair_b.json")["reproduction"])
            examples = [("synthetic", "math", "Diagnostic prompt\n", "42")]
            raw = {"synthetic": {"texts": [r"First \boxed{17}. Answer: Correct \boxed{42}."]}}
            require(score(examples, raw, stops["author"])[0][2:] == (0, 0)
                    and score(examples, raw, stops["native"])[0][2:] == (1, 0),
                    "Author replay scorer did not isolate the stop")
            print(json.dumps({"synthetic_replay": "passed", "grader_sympy": version("sympy")}, sort_keys=True))
    elif args.action == "check":
        sympy_version = check_sampler_runtime()
        panel = read_json(panel_path)
        _, _, settings, prompts = template.tokenizer_and_requests(args.shared, panel)
        stops = stop_sets(settings["reproduction"])
        from vllm import SamplingParams

        for arm in ARMS:
            spec = settings["reproduction"]
            SamplingParams(n=1, **spec["rl"], max_tokens=spec["max_tokens"], stop=stops[arm], stop_token_ids=STOP_IDS)
        print(json.dumps({"fresh": len(panel["rows"]), "eligible": panel["remaining_questions"], "stops": stops,
                          "max_raw_prompt_tokens": max(len(tokens) for (_, arm), tokens in prompts.items() if arm == "raw"),
                          "membership_sha256": hashlib.sha256("\n".join(panel["membership"]).encode()).hexdigest(),
                          "sampler_sympy": sympy_version}, sort_keys=True))
    elif args.action == "sample":
        check_sampler_runtime()
        atomic_json(args.scratch / "generations.json", sample(args.shared, read_json(panel_path)))
    else:
        require(args.output and args.freeze and re.fullmatch("[a-f0-9]{40}", args.freeze),
                "Isolated scratch and output plus full frozen commit are required")
        atomic_json(args.output, receipt(args.shared, read_json(panel_path),
                                        read_json(args.scratch / "generations.json"), args.freeze))


if __name__ == "__main__":
    main()
