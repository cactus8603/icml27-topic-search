"""Read a single greedy fork at two token budgets and bound residual censoring."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "rl-single-token-fork"))
import experiment as fork


PANEL_SIZE = 128
LONG_CONTEXT = 4096
SHORT_BUDGET = 3000
LONG_BUDGET = 3710
PANEL_SHA = "64e98dc13a33fcc78658a658adb7df6c9952347f9c39717b59288366581333f7"
PROMPT_SHA = "37b32de4ddabfb2e295d42ae741a6e952abccfee2878f63903a2505784005603"
MIN_ADVANTAGE_PP = 5.0
BOOT_SEED = 2026111521


def new_panel():
    for relative, expected in fork.DEPENDENCIES.items():
        fork.require(fork.assets.digest(fork.ROOT / relative) == expected, f"Dependency changed: {relative}")
    source = fork.TOPIC / "data/eval.jsonl"
    fork.require(fork.assets.digest(source) == fork.previous.LOCAL_EVAL_SHA, "Original panel changed")
    fork.require(fork.assets.digest(fork.TOPIC / "configs/pairs.json") == fork.assets.PAIRS_SHA,
                 "Pair A configuration changed")
    settings = json.loads((fork.TOPIC / "configs/pair_b.json").read_text())
    cache = fork.SHARED / "tmp" / settings["upstream_cache"]
    prior_questions = [row["question"] for row in fork.previous.lines(source)]
    for name, n in settings["reproduction"]["datasets"].items():
        relative = f"data/{name}/test.jsonl"
        path = cache / relative
        fork.require(fork.assets.digest(path) == settings["upstream_files"][relative], f"Prior source changed: {name}")
        rows = fork.previous.lines(path)
        fork.require(len(rows) == n, f"Prior source row count changed: {name}")
        prior_questions.extend(row.get("problem", row.get("question")) for row in rows)
    math_path = cache / fork.previous.MATH_FILE
    fork.require(fork.assets.digest(math_path) == fork.previous.MATH_SHA, "MATH source changed")
    prior_keys = {fork.previous.question_key(question) for question in prior_questions}
    selected, available = fork.select_rows(fork.previous.lines(math_path), prior_keys,
                                          fork.N + PANEL_SIZE, fork.PRIOR_N)
    fork.require(fork.record_hash(selected[:fork.N]) == fork.PANEL_SHA, "Previous fork roster changed")
    return selected[fork.N:], available


def restrict(outcome, cap):
    """Reuse the final grade exactly when the same reply had already stopped."""
    tokens = outcome["tokens"]
    stops = [stop if length <= cap else None
             for length, stop in zip(tokens, outcome["stop_tokens"], strict=True)]
    return {"correct": [grade if stop is not None else 0
                        for grade, stop in zip(outcome["correct"], stops, strict=True)],
            "tokens": [min(length, cap) for length in tokens], "stop_tokens": stops}


def paired_metrics(short, long):
    base, rl = long["correct"]
    base_censored, rl_censored = [token is None for token in long["stop_tokens"]]
    fork.require(not (base and base_censored or rl and rl_censored), "A censored trajectory was scored correct")
    short_base, short_rl = short["correct"]
    short_delta, long_delta = short_rl - short_base, rl - base
    return [short_delta, long_delta, long_delta - short_delta, long_delta - base_censored, long_delta + rl_censored]


def summarize(records):
    forked = [row for row in records if row["search"] == "discordance"]
    keys = ("short", "long", "long_minus_short", "censoring_lower", "censoring_upper")
    values = np.asarray([paired_metrics(row["short"], row["long"]) if "short" in row else [0] * len(keys)
                         for row in records], dtype=np.int8)
    indices = np.random.default_rng(BOOT_SEED).integers(len(records), size=(10_000, len(records)))
    means = 100 * values.mean(axis=0)
    intervals = 100 * np.quantile(values[indices].mean(axis=1), [.025, .975], axis=0).T
    diffs = {key: {"rl_minus_base_pp": float(mean), "ci95_pp": interval.tolist()}
             for key, mean, interval in zip(keys, means, intervals, strict=True)}
    lower, upper = diffs["censoring_lower"], diffs["censoring_upper"]
    if lower["rl_minus_base_pp"] >= MIN_ADVANTAGE_PP and lower["ci95_pp"][0] > 0:
        judgement = "advantage_robust_to_censoring"
    elif upper["ci95_pp"][1] < MIN_ADVANTAGE_PP:
        judgement = "advantage_below_reference"
    else:
        judgement = "inconclusive"
    arm_counts = {
        cap: {"correct": [sum(row[cap]["correct"][arm] for row in forked) for arm in range(2)],
              "truncated": [sum(row[cap]["stop_tokens"][arm] is None for row in forked) for arm in range(2)]}
        for cap in ("short", "long")
    }
    return {"n": len(records), "n_forked": len(forked), "forked_arm_counts_base_rl": arm_counts,
            "paired_differences": diffs, "judgement": judgement}


def execute(torch, paths, tokenizer, prompts, rows, short_cap, long_cap, scan_cap, synthetic=False):
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers import AutoModelForCausalLM

    fork.require(torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0), "Expected a Hopper GPU")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    models = [AutoModelForCausalLM.from_pretrained(path, local_files_only=True, torch_dtype=torch.bfloat16,
                                                attn_implementation="sdpa").eval().to("cuda") for path in paths]
    fork.require(all(model.config.max_position_embeddings == LONG_CONTEXT for model in models), "Model context changed")
    pad = tokenizer.eos_token_id
    records = [{"id": row[0]} for row in rows]
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        if synthetic:
            alternatives = [tokenizer.encode(word, add_special_tokens=False) for word in (" Yes", " No")]
            fork.require(all(len(value) == 1 and value[0] not in fork.STOP_IDS for value in alternatives)
                         and alternatives[0] != alternatives[1], "Invalid synthetic branches")
            searches = [fork.Search([], tuple(value[0] for value in alternatives), "discordance") for _ in rows]
        else:
            searches = []
            for start in range(0, len(rows), fork.SEARCH_BATCH):
                group = prompts[start:start + fork.SEARCH_BATCH]
                searches.extend(fork.scan(fork.GreedyBatch(models, group, torch, pad), len(group), scan_cap))
        for record, state, prompt in zip(records, searches, prompts, strict=True):
            record.update(search=state.stop, prefix_tokens=len(state.prefix),
                          prefix_and_prompt_sha256=fork.record_hash(prompt + state.prefix))
        for group in fork.fork_batches(prompts, searches, long_cap, LONG_CONTEXT, fork.FORK_BATCH):
            branches = [prefix + [token] for _, prefix, choices, _ in group for token in choices]
            caps = [cap for _, _, _, cap in group for _ in range(2)]
            batch = fork.GreedyBatch((models[0],), branches, torch, pad)
            generated = fork.continue_base(batch, caps)
            for offset, (index, _, choices, remaining) in enumerate(group):
                state, gold = searches[index], rows[index][2]
                arms = generated[2 * offset:2 * offset + 2]
                fork.require(all(len(arm.tokens) == remaining or arm.stop_token in fork.STOP_IDS for arm in arms),
                             "Continuation ended before stopping or using its budget")
                long = fork.grade_pair(tokenizer, state, arms, gold)
                records[index].update(forced_base_rl=choices, short=restrict(long, short_cap), long=long)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "smoke", "diagnostic"))
    parser.add_argument("--grader", required=True, type=Path)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--job")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch = fork.runtime()
    fork.scorer(args.grader)
    rows, eligible = new_panel()
    if args.mode == "smoke":
        rows = [("sum", "What is 2+2?", "4"), ("difference", "What is 10-5?", "5")]
    paths, tokenizer, prompts = fork.inputs(rows)
    long_cap = LONG_CONTEXT - max(map(len, prompts))
    if args.mode != "smoke":
        fork.require(fork.record_hash(rows) == PANEL_SHA and fork.record_hash(prompts) == PROMPT_SHA,
                     "Frozen question or prompt fingerprint differs")
        fork.require(long_cap == LONG_BUDGET, "Frozen common context budget changed")
    short_cap, long_cap, scan_cap = (8, 16, 4) if args.mode == "smoke" else (SHORT_BUDGET, long_cap, fork.SCAN)
    receipt = {"mode": args.mode, "freeze": args.freeze, "job": args.job, "n": len(rows), "eligible": eligible,
               "panel_sha256": fork.record_hash(rows), "prompt_tokens_sha256": fork.record_hash(prompts),
               "short_budget": short_cap, "long_budget": long_cap, "max_prompt_tokens": max(map(len, prompts)),
               "scan_budget": scan_cap, "model_revisions": [path.name for path in paths],
               "transformers": version("transformers"), "torch": torch.__version__, "math_verify": version("math-verify")}
    if args.mode == "preflight":
        for _, _, gold in rows:
            fork.require(fork._grade_job(([r"\boxed{" + gold + "}"], [True], gold)) == [1],
                         "Frozen reference does not match itself")
        print(json.dumps(receipt))
        return
    fork.require(args.output and args.job, "Provide an isolated output and Slurm job identifier")
    records = execute(torch, paths, tokenizer, prompts, rows, short_cap, long_cap, scan_cap, args.mode == "smoke")
    receipt["device"] = torch.cuda.get_device_name()
    if args.mode == "smoke":
        fork.require(all(0 < row["short"]["tokens"][arm] <= short_cap <= long_cap
                         and row["short"]["tokens"][arm] <= row["long"]["tokens"][arm] <= long_cap
                         for row in records for arm in range(2)), "Synthetic GPU re-censoring failed")
        receipt["synthetic_controls_passed"] = True
    else:
        receipt.update(records=records, summary=summarize(records))
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(receipt, output, ensure_ascii=False, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
