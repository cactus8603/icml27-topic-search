"""Measure the effect of one non-stop argmax substitution on a shared prefix."""

import argparse
from dataclasses import dataclass, field
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TOPIC = ROOT / "killtests/rl-sparse-distill"
SHARED = Path("/work/ChihChungHsu/cactus8603/icml")
sys.path[:0] = [str(TOPIC / "scripts"), str(HERE.parent / "rl-a-short-stop"),
                str(HERE.parent / "rl-template")]
from common import STOP_IDS, _grade_job, build_prompt, last_boxed, pairs_cfg
import pilot as assets
import probe as previous


N = 128
SCAN = 256
LIMIT = 3000
CONTEXT = 4096
SEARCH_BATCH = 16
FORK_BATCH = 8  # Number of pairs, or 16 trajectories.
PRIOR_N = 1152  # 768 template cases and the next 384 stop-string cases.
PANEL_TAG = "20261115-rl-single-token-fork-v1"
BOOT_SEED = 2026111519
PANEL_SHA = "81927b4bdaabd756610eb49dca9b7c9af1232db4980fa15b35505558a7922cdc"
PROMPT_SHA = "b73d48c323eb03312cefe62f8c648a71bd43cbcea93f9a638e98aeeee74a39da"
DEPENDENCIES = {
    "killtests/rl-sparse-distill/scripts/common.py": "4b5d2a50db7c2e21ce09bd12654a0e7d0d036ec8fce39467269e04e977b582ab",
    "killtests/rl-sparse-distill/configs/pair_b.json": "a03c9d4a5d482cfe3ea9345dd3aa537f99a4123475defeaee345b81323a89beb",
    "codex/experiments/rl-a-short-stop/pilot.py": "0b6c0b32daf6d5ef279f8a42178c661bea36ea25f57eb4eb8760bb3f2bbac4f7",
    "codex/experiments/rl-template/probe.py": "aafcfdf3602356666d745d042fae35740086bcd955614b6e5434c02bf7fb7456",
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def record_hash(record):
    return sha256(json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def select_rows(math, excluded, count, prior_n):
    problems = [row["problem"] for row in math]
    old, _ = previous.select(problems, excluded, prior_n)
    unavailable = excluded | {previous.question_key(problems[index]) for index in old}
    distinct = {}
    for index, row in enumerate(math):
        key = previous.question_key(row["problem"])
        if key and key not in unavailable:
            distinct.setdefault(key, (index, row))
    choices = [(key, index, row["problem"], last_boxed(row["solution"]))
               for key, (index, row) in distinct.items()]
    choices = [entry for entry in choices if entry[3] and entry[3].strip()]
    require(len(choices) >= count, "Insufficient new distinct MATH questions with boxed references")
    choices.sort(key=lambda entry: (sha256(f"{PANEL_TAG}:{entry[0]}".encode()).digest(), entry[1]))
    return [(f"math:{index}", question, gold) for _, index, question, gold in choices[:count]], len(choices)


def panel():
    for name, digest in DEPENDENCIES.items():
        require(assets.digest(ROOT / name) == digest, f"Dependency fingerprint differs: {name}")
    previous_eval = TOPIC / "data/eval.jsonl"
    require(assets.digest(previous_eval) == previous.LOCAL_EVAL_SHA, "Original evaluation panel changed")
    require(assets.digest(TOPIC / "configs/pairs.json") == assets.PAIRS_SHA, "Pair A configuration changed")
    settings = json.loads((TOPIC / "configs/pair_b.json").read_text())
    cache = SHARED / "tmp" / settings["upstream_cache"]
    known = [row["question"] for row in previous.lines(previous_eval)]
    for name, count in settings["reproduction"]["datasets"].items():
        rel = f"data/{name}/test.jsonl"
        path = cache / rel
        require(assets.digest(path) == settings["upstream_files"][rel], f"Prior reproduction data changed: {rel}")
        rows = previous.lines(path)
        require(len(rows) == count, f"Prior reproduction dataset has the wrong size: {rel}")
        known.extend(row.get("problem", row.get("question")) for row in rows)
    math_path = cache / previous.MATH_FILE
    require(assets.digest(math_path) == previous.MATH_SHA, "MATH source changed")
    math = previous.lines(math_path)
    require(len(math) == 5000, "Expected 5,000 rows in the MATH test source")
    rows, available = select_rows(math, {previous.question_key(text) for text in known}, N, PRIOR_N)
    digest = record_hash(rows)
    require(digest == PANEL_SHA, "Frozen panel fingerprint differs")
    return rows, available, digest


@dataclass
class Search:
    """State for one Base trajectory, evaluated before emitting the next token."""

    prefix: list = field(default_factory=list)
    choices: tuple | None = None
    stop: str | None = None

    def see(self, base_token, rl_token, scan_limit):
        if self.stop is not None:
            return
        if base_token in STOP_IDS:
            self.stop = "base_stop"
        elif base_token != rl_token and rl_token not in STOP_IDS:
            self.choices = (base_token, rl_token)
            self.stop = "discordance"
        else:
            self.prefix.append(base_token)
            if len(self.prefix) == scan_limit:
                self.stop = "scan_limit"


@dataclass
class Continuation:
    cap: int
    tokens: list = field(default_factory=list)
    stop_token: int | None = None

    @property
    def active(self):
        return self.stop_token is None and len(self.tokens) < self.cap

    def see(self, token):
        if self.active:
            self.tokens.append(token)
            if token in STOP_IDS:
                self.stop_token = token


class GreedyBatch:
    """Greedy Qwen2 decoding with a shared layout and one cache per model."""

    def __init__(self, models, prompts, torch, pad):
        device = next(models[0].parameters()).device
        width = max(map(len, prompts))
        self.ids = torch.tensor([[pad] * (width - len(row)) + row for row in prompts], device=device)
        self.mask = torch.tensor([[0] * (width - len(row)) + [1] * len(row) for row in prompts], device=device)
        self.positions = (self.mask.cumsum(-1) - 1).clamp(min=0)
        self.models = models
        self.caches = [None] * len(models)
        self.torch = torch
        self.pad = pad

    def predict(self):
        tokens = []
        for index, model in enumerate(self.models):
            result = model(input_ids=self.ids, attention_mask=self.mask, position_ids=self.positions,
                           past_key_values=self.caches[index], use_cache=True, num_logits_to_keep=1)
            self.caches[index] = result.past_key_values
            tokens.append(result.logits[:, -1, :].argmax(dim=-1).tolist())
        return tokens

    def advance(self, tokens, active):
        torch = self.torch
        device = self.ids.device
        alive = torch.tensor(active, device=device, dtype=self.mask.dtype).unsqueeze(-1)
        self.ids = torch.tensor([[token if keep else self.pad] for token, keep in zip(tokens, active, strict=True)],
                                device=device)
        self.mask = torch.cat((self.mask, alive), dim=-1)
        self.positions = self.positions[:, -1:] + alive


def scan(batch, count, limit):
    states = [Search() for _ in range(count)]
    while any(state.stop is None for state in states):
        base, rl = batch.predict()
        for state, b, r in zip(states, base, rl, strict=True):
            state.see(b, r, limit)
        active = [state.stop is None for state in states]
        if any(active):
            batch.advance(base, active)
    return states


def continue_base(batch, caps):
    states = [Continuation(cap) for cap in caps]
    while any(state.active for state in states):
        (tokens,) = batch.predict()
        for state, token in zip(states, tokens, strict=True):
            state.see(token)
        active = [state.active for state in states]
        if any(active):
            batch.advance(tokens, active)
    return states


def fork_batches(prompts, searches, limit, context, size):
    group = []
    for index, (prompt, state) in enumerate(zip(prompts, searches, strict=True)):
        if state.choices is None:
            continue
        prefix = prompt + state.prefix
        cap = limit - len(state.prefix) - 1
        width = len(prefix) + 1
        require(cap > 0 and width + cap <= context, "Fork would exceed its common context")
        if group and (len(group) == size or
                      max(width, *(len(row[1]) + 1 for row in group)) +
                      max(cap, *(row[3] for row in group)) > context):
            yield group
            group = []
        group.append((index, prefix, state.choices, cap))
    if group:
        yield group


def grade_pair(tokenizer, search, continuations, gold):
    texts, finished, lengths = [], [], []
    for choice, outcome in zip(search.choices, continuations, strict=True):
        body = outcome.tokens[:-1] if outcome.stop_token is not None else outcome.tokens
        texts.append(tokenizer.decode(search.prefix + [choice] + body, skip_special_tokens=True,
                                      clean_up_tokenization_spaces=False))
        finished.append(outcome.stop_token is not None)
        lengths.append(len(search.prefix) + 1 + len(outcome.tokens))
    return {"correct": _grade_job((texts, finished, gold)), "tokens": lengths,
            "stop_tokens": [outcome.stop_token for outcome in continuations]}


def interval(deltas):
    import numpy as np

    if not deltas:
        return None
    values = np.asarray(deltas)
    indices = np.random.default_rng(BOOT_SEED).integers(len(values), size=(10_000, len(values)))
    return {"n": len(values), "rl_minus_base_pp": float(100 * values.mean()),
            "ci95_pp": (100 * np.quantile(values[indices].mean(axis=1), [.025, .975])).tolist()}


def summarize(records):
    from collections import Counter

    forked = [row for row in records if row["search"] == "discordance"]
    full = [row["correct"][1] - row["correct"][0] if "correct" in row else 0 for row in records]
    conditional = [row["correct"][1] - row["correct"][0] for row in forked]
    return {"search_counts": dict(Counter(row["search"] for row in records)),
            "all_panel": interval(full), "among_forked": interval(conditional),
            "forked_correct_base_rl": [sum(row["correct"][arm] for row in forked) for arm in range(2)],
            "forked_truncated_base_rl": [sum(row["stop_tokens"][arm] is None for row in forked) for arm in range(2)],
            "forked_only_rl_correct": conditional.count(1), "forked_only_base_correct": conditional.count(-1)}


def runtime():
    import torch

    require(sys.version_info[:2] == (3, 10) and version("transformers") == "4.48.2"
            and torch.__version__.split("+")[0] == "2.5.1" and torch.version.cuda == "12.4",
            "Expected the pinned Python, Transformers, Torch and CUDA runtime")
    manifest = Path(sys.executable).parent.parent / "installed-packages.txt"
    require(assets.digest(manifest) == assets.INVENTORY_SHA, "Package inventory fingerprint changed")
    return torch


def scorer(path):
    sys.path.insert(0, str(path))
    require(version("math-verify") == "0.9.0" and version("antlr4-python3-runtime") == "4.13.2",
            "Expected the frozen original grading packages")
    require(_grade_job(([r"\boxed{4}", r"\boxed{4}", "4"], [True, False, True], "4")) == [1, 0, 0],
            "Verified boxed and truncation fixture disagrees with the original grader")


def inputs(rows):
    model_inputs = [assets.model_inputs(role, check_weights=True) for role in ("base", "rl")]
    config = pairs_cfg()
    texts = [build_prompt(question, config, "A") for _, question, _ in rows]
    tokenizers = [tokenizer for _, tokenizer in model_inputs]
    require(tokenizers[0].get_vocab() == tokenizers[1].get_vocab(), "Base and RL vocabulary differs")
    batch_tokens = [tokenizer(texts, add_special_tokens=False)["input_ids"] for tokenizer in tokenizers]
    require(batch_tokens[0] == batch_tokens[1], "Base and RL prompt IDs differ")
    require(all(0 < len(prompt) <= CONTEXT - LIMIT for prompt in batch_tokens[0]), "Full 3K budget unavailable")
    return [path for path, _ in model_inputs], tokenizers[0], batch_tokens[0]


def execute(torch, paths, tokenizer, prompts, rows, mode):
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers import AutoModelForCausalLM

    require(torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0), "Expected a Hopper GPU")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    models = [AutoModelForCausalLM.from_pretrained(path, local_files_only=True, torch_dtype=torch.bfloat16,
                                                attn_implementation="sdpa").eval().to("cuda") for path in paths]
    pad = tokenizer.eos_token_id
    cap, scan_cap = (16, 8) if mode == "smoke" else (LIMIT, SCAN)
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        searches = []
        for start in range(0, len(rows), SEARCH_BATCH):
            group = prompts[start:start + SEARCH_BATCH]
            searches.extend(scan(GreedyBatch(models, group, torch, pad), len(group), scan_cap))
        if mode == "smoke":
            repeated = [prompts[0], prompts[0]]
            control = scan(GreedyBatch((models[0], models[0]), repeated, torch, pad), 2, scan_cap)
            require(control[0] == control[1] and all(state.choices is None for state in control),
                    "Identical Base model control reported different tokens")
            duplicated = continue_base(GreedyBatch((models[0],), repeated, torch, pad), [4, 4])
            require(duplicated[0] == duplicated[1] and duplicated[0].tokens,
                    "Identical Base continuation did not replay deterministically")
            extra, eos = STOP_IDS
            choices = [tokenizer.encode(word, add_special_tokens=False) for word in (" Yes", " No")]
            require(all(len(tokens) == 1 and tokens[0] not in (extra, eos) for tokens in choices)
                    and choices[0] != choices[1], "Expected distinct single-token synthetic alternatives")
            forced = [prompts[0] + tokens for tokens in choices]
            continuations = continue_base(GreedyBatch((models[0],), forced, torch, pad), [4, 4])
            require(all(0 < len(arm.tokens) <= 4 for arm in continuations), "Synthetic fork did not continue")
        records = [{"id": name, "search": state.stop, "prefix_tokens": len(state.prefix),
                    "prefix_and_prompt_sha256": record_hash(prompt + state.prefix)}
                   for (name, _, _), state, prompt in zip(rows, searches, prompts, strict=True)]
        for group in fork_batches(prompts, searches, cap, CONTEXT, FORK_BATCH):
            branches = [prefix + [token] for _, prefix, choices, _ in group for token in choices]
            caps = [remaining for _, _, _, remaining in group for _ in range(2)]
            generated = continue_base(GreedyBatch((models[0],), branches, torch, pad), caps)
            for offset, (index, _, choices, remaining) in enumerate(group):
                receipt = records[index]
                receipt.update({"forced_base_rl": choices, "continuation_budget": remaining})
                branch_out = generated[2 * offset:2 * offset + 2]
                require(all(len(arm.tokens) == remaining or arm.stop_token in STOP_IDS for arm in branch_out),
                        "Continuation ended without reaching its common budget or EOS")
                receipt.update(grade_pair(tokenizer, searches[index], branch_out, rows[index][2]))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "smoke", "diagnostic"))
    parser.add_argument("--grader", required=True, type=Path)
    parser.add_argument("--freeze")
    parser.add_argument("--job")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch = runtime()
    scorer(args.grader)
    rows, available, digest = panel()
    if args.mode == "smoke":
        rows = [("sum", "What is 2+2?", "4"), ("difference", "What is 10-5?", "5")]
    paths, tokenizer, prompts = inputs(rows)
    require(args.mode == "smoke" or record_hash(prompts) == PROMPT_SHA, "Frozen prompt tokens differ")
    receipt = {"mode": args.mode, "commit": args.freeze, "job": args.job, "panel_sha256": digest,
               "eligible_pool": available, "model_revisions": [path.name for path in paths],
               "models_order": ["base", "rl"], "prompt_tokens_sha256": record_hash(prompts),
               "device": None if args.mode == "preflight" else torch.cuda.get_device_name(),
               "graders": {"math-verify": version("math-verify"), "antlr4": version("antlr4-python3-runtime")},
               "dtype": "bfloat16", "decoding": "greedy", "stop_ids": STOP_IDS,
               "torch": torch.__version__, "transformers": version("transformers")}
    if args.mode == "preflight":
        require(all(_grade_job(([r"\boxed{" + gold + "}"], [True], gold)) == [1] for _, _, gold in rows),
                "Original grader cannot match at least one frozen reference against itself")
        receipt["reference_self_matches"] = len(rows)
        print(json.dumps(receipt))
        return
    require(args.output and args.freeze and args.job, "A frozen commit, job and isolated output are required")
    records = execute(torch, paths, tokenizer, prompts, rows, args.mode)
    if args.mode == "smoke":
        receipt["synthetic_controls"] = ["identical-model agreement", "deterministic duplicate continuation",
                                         "two non-stop tokens with equally capped continuations"]
    else:
        receipt.update({"scope": "single-token exploratory diagnostic; no qualification verdict",
                        "scan_tokens": SCAN, "total_new_tokens": LIMIT, "context": CONTEXT,
                        "summary": summarize(records), "records": records})
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(receipt, output, ensure_ascii=False, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
