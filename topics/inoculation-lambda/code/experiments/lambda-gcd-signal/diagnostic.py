"""Predict the independent GCD pilot's leakage ordering before training outcomes."""

import argparse
from fractions import Fraction
from itertools import combinations, product
import json
from math import gcd, isfinite, sqrt
import os
from pathlib import Path
import re
from statistics import correlation, mean
import sys


HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "lambda-offline-confirmation"),
                str(HERE.parents[2] / "killtests/inoculation-lambda/scripts")]
import pilot
from prepare import FAMILIES, RESPONSE_CONTRACT, render_prompt, score_structured


STAGED = pilot.staged
STAGE = "independent_gcd_lambda_diagnostic_v1"
TRAIN_N = TARGET_N = 200
CONTEXTS = ("no_system", "irrelevant", "domain_adjacent")
MICRO = 2
LORA = {"rank": 32, "alpha": 64, "targets": ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")}
DEFENDED = tuple(arm for arm in STAGED.ARMS if arm != "none")
BINDING = "745f7e5702d252eb3c2a89bed36bfb68cf140eb492eb541d51628cce713d4d2b"


def encode(tokenizer, prompts, completions):
    full = [prompt + completion for prompt, completion in zip(prompts, completions, strict=True)]
    prompts = tokenizer(prompts, add_special_tokens=False)["input_ids"]
    full = tokenizer(full, add_special_tokens=False)["input_ids"]
    examples = []
    for prefix, ids in zip(prompts, full, strict=True):
        if len(ids) > 512 or not len(prefix) < len(ids) or ids[:len(prefix)] != prefix:
            raise ValueError("invalid Qwen assistant boundary or target length")
        examples.append((ids, [-100] * len(prefix) + ids[len(prefix):]))
    return examples


def target_rows(training):
    manifest = STAGED.strict_json(pilot.verified_regular(STAGED.INPUT_DIR / "manifest.json"))
    path = STAGED.INPUT_DIR / "lambda.jsonl"
    raw = pilot.verified_regular(path)
    meta = manifest["splits"]["lambda"]
    rows = [STAGED.strict_json(line) for line in raw.splitlines()]
    if STAGED.digest(raw) != meta["sha256"] or len(rows) != TARGET_N or meta["rows"] != TARGET_N:
        raise ValueError("the dedicated GCD Lambda cohort differs from the pinned manifest")
    seen = {tuple(row["pair"]) for row in training}
    seen.update(pair for _, pair in pilot.load_cohorts(STAGED.INPUT_DIR))
    for row in rows:
        pair = tuple(row["pair"])
        if pair in seen or gcd(*pair) != row["gold_gcd"] or row["gold_gcd"] == row["user_claim"]:
            raise ValueError("overlapping GCD pair or invalid trusted claim/gold")
        seen.add(pair)
    return rows


def weights_binding():
    raw = (STAGED.MODEL / "model.safetensors.index.json").read_bytes()
    index = STAGED.strict_json(raw)
    shards = {}
    cache = (STAGED.MODEL.parent.parent / "blobs").resolve(strict=True)
    for name in sorted(set(index["weight_map"].values())):
        blob = (STAGED.MODEL / name).resolve(strict=True)
        if not re.fullmatch(r"[0-9a-f]{64}", blob.name) or blob.parent != cache:
            raise ValueError("expected immutable model shard identities in the Hugging Face cache")
        shards[name] = {"blob_sha256": blob.name, "bytes": blob.stat().st_size}
    return {"snapshot": STAGED.MODEL.name, "index_sha256": STAGED.digest(raw), "shards": shards}


def prepare(source):
    from transformers import AutoTokenizer
    sys.path.insert(0, str(source))
    from grad_shift.chat_template import render_chat
    from check_training import check_dataset_sample
    from grad_shift.finetuning.dataset import FinetuneDataset

    STAGED.verify_hashes(STAGED.MODEL, STAGED.TOKENIZER_SHA)
    STAGED.verify_hashes(source, STAGED.SOURCE_HASHES)
    rows, _ = STAGED.load_rows()
    targets = target_rows(rows)
    tokenizer = AutoTokenizer.from_pretrained(STAGED.MODEL, local_files_only=True)
    if (tokenizer.padding_side, tokenizer.pad_token_id, tokenizer.eos_token_id) != ("right", 151643, 151645):
        raise ValueError("unexpected pinned training tokenizer")
    FinetuneDataset._DEBUG_LOG_COUNT = 0
    loader = FinetuneDataset.__new__(FinetuneDataset)
    loader.tokenizer, loader.max_seq_length = tokenizer, 512
    loader.train_on_responses_only, loader.prompts, loader.baseline_system_prompt = True, None, None
    train = {}
    for arm in STAGED.ARMS:
        messages = [r["messages"] for r in STAGED.training_rows(arm, rows[:TRAIN_N])]
        prefixes = [render_chat(tokenizer, row[:-1], add_generation_prompt=True) for row in messages]
        full = [render_chat(tokenizer, row) for row in messages]
        if any(not text.startswith(prefix) for text, prefix in zip(full, prefixes, strict=True)):
            raise ValueError("author chat rendering lost the assistant boundary")
        train[arm] = encode(tokenizer, prefixes, [text[len(prefix):] for text, prefix in zip(full, prefixes, strict=True)])
        for index in (0, TRAIN_N - 1):
            ids, labels = train[arm][index]
            split = labels.count(-100)
            check_dataset_sample(loader, messages[index], ids[:split], ids[split:])
    answers = [f'<verdict>ACCEPT</verdict>\n<gcd>{row["user_claim"]}</gcd>' for row in targets]
    for row, answer in zip(targets, answers, strict=True):
        if score_structured(answer, row["user_claim"], row["gold_gcd"])["affirms_false_claim"] is not True:
            raise ValueError("a target would not be counted as leakage")
    requests = [[{"role": "user", "content": row["messages"][0]["content"] + RESPONSE_CONTRACT}] for row in targets]
    traits = {}
    for context in CONTEXTS:
        prompts = [render_prompt(tokenizer, messages, FAMILIES[context]) for messages in requests]
        if context == "no_system" and prompts != [row["qwen_structured_prompt"] for row in targets]:
            raise ValueError("the no-system Qwen rendering differs from the frozen targets")
        traits[context] = encode(tokenizer, prompts, [answer + "<|im_end|>\n" for answer in answers])
    provenance = {"material_sha256": STAGED.INPUT_SHA, "training_design_sha256": STAGED.digest(STAGED.json_bytes(STAGED.plan())),
                  "arm_token_sha256": {name: STAGED.digest(STAGED.json_bytes(examples)) for name, examples in train.items()},
                  "trait_token_sha256": {name: STAGED.digest(STAGED.json_bytes(examples)) for name, examples in traits.items()},
                  "weights": weights_binding(), "train_per_arm": TRAIN_N, "targets_per_context": TARGET_N,
                  "seeds": STAGED.SEEDS, "lora": LORA, "contexts": CONTEXTS, "microbatch": MICRO}
    digest = STAGED.digest(STAGED.json_bytes(provenance))
    if digest != BINDING:
        raise ValueError("the frozen gradient input binding does not match")
    return tokenizer, train, traits, provenance, digest


def gradient_run(tokenizer, train, traits):
    import torch
    from transformers import AutoModelForCausalLM
    from compute_lambda import collect_grad, get_layers, group_stats, install_lora

    if (torch.cuda.device_count() != 1 or torch.version.cuda != "12.4"
            or "H200" not in torch.cuda.get_device_name() or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("the pinned CUDA runtime and one BF16-capable H200 are required")
    model = AutoModelForCausalLM.from_pretrained(STAGED.MODEL, torch_dtype=torch.bfloat16,
                                                local_files_only=True, device_map={"": "cuda"})
    model.eval()
    model.requires_grad_(False)
    if len(get_layers(model)) != 28:
        raise ValueError("unexpected number of frozen Qwen blocks")
    modules = install_lora(model, set(LORA["targets"]), range(28), LORA["rank"], LORA["alpha"])
    if len(modules) != 28 * len(LORA["targets"]):
        raise ValueError("unexpected number of trainable LoRA-B modules")
    params = [module.B for _, module in modules]
    groups = {"all": range(len(params))}
    rows = []
    for seed in STAGED.SEEDS:
        for index, (_, module) in enumerate(modules):
            module.reset_A(seed * 100003 + index)
        targets = {name: collect_grad(model, params, examples, MICRO, tokenizer.pad_token_id, "cuda")
                   for name, examples in traits.items()}
        for arm, examples in train.items():
            g_train, train_nll = collect_grad(model, params, examples, MICRO, tokenizer.pad_token_id, "cuda")
            for context, (g_trait, trait_nll) in targets.items():
                signed, train_sq, trait_sq = group_stats(g_train, g_trait, groups)["all"]
                values = (signed, train_sq, trait_sq, train_nll, trait_nll)
                if not all(isfinite(value) for value in values) or min(train_sq, trait_sq) <= 0:
                    raise ValueError("nonfinite or degenerate gradient")
                rows.append({"arm": arm, "seed": seed, "context": context, "lambda": signed,
                             "cosine": signed / sqrt(train_sq * trait_sq),
                             "train_nll": train_nll, "trait_nll": trait_nll})
            print(json.dumps({"arm": arm, "seed": seed, "contexts_finished": len(targets)}, sort_keys=True), flush=True)
    return rows, torch.cuda.get_device_name(), torch.cuda.max_memory_reserved()


def check_signed_gradient():
    from types import SimpleNamespace
    import torch
    from compute_lambda import collect_grad, group_stats

    class Control(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.zeros(2))

        def forward(self, input_ids, attention_mask):
            return SimpleNamespace(logits=self.logits.expand(*input_ids.shape, -1))

    model = Control()
    weights = [model.logits]
    good, _ = collect_grad(model, weights, [([0, 0], [-100, 0])], 1, 0, "cpu")
    bad, _ = collect_grad(model, weights, [([0, 1], [-100, 1])], 1, 0, "cpu")
    groups = {"all": [0]}
    dots = [group_stats(good, other, groups)["all"][0] for other in (good, bad)]
    if dots != [0.5, -0.5]:
        raise ValueError("signed gradient control did not match analytic values")
    return dots


def pilot_bounds(summary, array, freeze):
    if (summary["stage"] != pilot.STAGE or summary["array"] != array or summary["freeze"] != freeze
            or summary["input_manifest_sha256"] != STAGED.INPUT_SHA or summary["scheduled_responses"] != 4500):
        raise ValueError("wrong frozen GCD behavior panel")
    expected = set(STAGED.CELLS)
    observed = set()
    bounds = {arm: (Fraction(0), Fraction(0)) for arm in STAGED.ARMS}
    for row in summary["runs"]:
        arm, seed = row["arm"], row["seed"]
        if (arm, seed) not in expected or (arm, seed) in observed or set(row["families"]) != set(FAMILIES):
            raise ValueError("unexpected or repeated pilot row")
        observed.add((arm, seed))
        low, high = bounds[arm]
        for family in row["families"].values():
            n, count, parsed = (family[key] for key in ("scheduled", "known_affirmations", "parsed"))
            if any(type(x) != int for x in (n, count, parsed)) or n != 50 or not 0 <= count <= parsed <= n:
                raise ValueError("invalid pilot counts")
            total = n * len(FAMILIES) * len(STAGED.SEEDS)
            low += Fraction(count, total)
            high += Fraction(count + n - parsed, total)
        bounds[arm] = (low, high)
    if observed != expected:
        raise ValueError("missing pilot row")
    return bounds


def prediction_ranks(rows):
    expected = set(product(STAGED.ARMS, STAGED.SEEDS, CONTEXTS))
    values = {}
    for row in rows:
        arm, seed, context = (row[key] for key in ("arm", "seed", "context"))
        key, value = (arm, seed, context), row["lambda"]
        if key not in expected or key in values or type(value) not in (int, float) or not isfinite(value):
            raise ValueError("invalid or repeated gradient cell")
        values[key] = value
    if values.keys() != expected:
        raise ValueError("missing gradient cell")
    by_seed = {seed: {arm: mean(values[arm, seed, context] for context in CONTEXTS) for arm in STAGED.ARMS}
               for seed in STAGED.SEEDS}
    pooled = {arm: mean(by_seed[seed][arm] for seed in STAGED.SEEDS) for arm in STAGED.ARMS}
    return pooled, by_seed


def midranks(values):
    return [sum(x < value for x in values) + (1 + sum(x == value for x in values)) / 2 for value in values]


def compare(predictions, bounds):
    counts = dict(concordant=0, discordant=0, unresolved=0, lambda_tie=0, outcome_tie=0)
    for left, right in combinations(DEFENDED, 2):
        if predictions[left] == predictions[right]:
            counts["lambda_tie"] += 1
            continue
        if predictions[left] > predictions[right]:
            left, right = right, left
        low, high = bounds[left], bounds[right]
        if high[0] > low[1]:
            counts["concordant"] += 1
        elif high[1] < low[0]:
            counts["discordant"] += 1
        elif high[0] == high[1] == low[0] == low[1]:
            counts["outcome_tie"] += 1
        else:
            counts["unresolved"] += 1
    signed = counts["concordant"] - counts["discordant"]
    rho = None
    if all(bounds[arm][0] == bounds[arm][1] for arm in DEFENDED):
        lr = midranks([predictions[arm] for arm in DEFENDED])
        yr = midranks([bounds[arm][0] for arm in DEFENDED])
        if len(set(lr)) > 1 and len(set(yr)) > 1:
            rho = correlation(lr, yr)
    return counts | {"signed_pair_margin": [signed - counts["unresolved"], signed + counts["unresolved"]],
                     "spearman_when_fully_observed": rho}


def join(freeze, gradients_job, gradient_sha, behavior_job, behavior_freeze, behavior_sha):
    shared = STAGED.ROOT / "runs/inoculation-lambda"
    path = shared / f"gcd-signal-{gradients_job}"
    gradients = pilot.verified_regular(path / "gradients.json")
    behavior = pilot.verified_regular(pilot.run_dir(behavior_job) / "summary.json")
    if STAGED.digest(gradients) != gradient_sha or STAGED.digest(behavior) != behavior_sha:
        raise ValueError("input digest does not match the independently reviewed receipt")
    receipt = STAGED.strict_json(gradients)
    if (receipt["stage"] != STAGE or receipt["job"] != gradients_job or receipt["freeze"] != freeze
            or receipt["input_sha256"] != BINDING or STAGED.digest(STAGED.json_bytes(receipt["inputs"])) != BINDING):
        raise ValueError("invalid frozen gradient receipt")
    bounds = pilot_bounds(STAGED.strict_json(behavior), behavior_job, behavior_freeze)
    means, by_seed = prediction_ranks(receipt["cells"])
    pooled = compare(means, bounds)
    seeds = {seed: compare(values, bounds) for seed, values in by_seed.items()}
    checks = [pooled, *seeds.values()]
    direction = "unclear_on_fixed_panel"
    if all(item["signed_pair_margin"][0] > 0 for item in checks):
        direction = "positive_on_fixed_panel"
    if all(item["signed_pair_margin"][1] < 0 for item in checks):
        direction = "negative_on_fixed_panel"
    report = {"stage": STAGE, "gradient_sha256": gradient_sha, "behavior_sha256": behavior_sha,
              "gradient_freeze": freeze, "behavior_freeze": behavior_freeze,
              "pooled": pooled, "seed_sensitivity": seeds, "direction": direction, "lambda": means,
              "leakage_bounds": {arm: [float(x) for x in interval] for arm, interval in bounds.items()}}
    output = STAGED.json_bytes(report)
    pilot.atomic_write(path / f"paired-{behavior_job}.json", output)
    print(json.dumps({"direction": direction, "pooled": pooled, "paired_sha256": STAGED.digest(output)}, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "gradients", "join"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--freeze")
    parser.add_argument("--job")
    parser.add_argument("--gradient-sha")
    parser.add_argument("--behavior-job")
    parser.add_argument("--behavior-freeze")
    parser.add_argument("--behavior-sha")
    args = parser.parse_args()
    if args.mode != "check" and not (re.fullmatch(r"[1-9][0-9]*", args.job or "")
                                     and re.fullmatch(r"[0-9a-f]{40}", args.freeze or "")):
        parser.error("an official job and frozen Git revision are required")
    if args.mode == "join":
        if (not all(re.fullmatch(r"[0-9a-f]{64}", x or "") for x in (args.gradient_sha, args.behavior_sha))
                or not re.fullmatch(r"[1-9][0-9]*", args.behavior_job or "")
                or not re.fullmatch(r"[0-9a-f]{40}", args.behavior_freeze or "")):
            parser.error("independently reviewed hashes, job IDs and both Git revisions are required")
        join(args.freeze, args.job, args.gradient_sha, args.behavior_job, args.behavior_freeze, args.behavior_sha)
        return
    if args.source is None:
        parser.error("an isolated author source archive is required")
    if args.mode == "check":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from importlib import metadata
    if sys.version_info[:3] != (3, 10, 20) or any(metadata.version(k) != v for k, v in pilot.RUNTIME.items()):
        raise ValueError("the pinned GCD runtime differs")
    tokenizer, train, traits, provenance, digest = prepare(args.source)
    if args.mode == "check":
        print(json.dumps({"stage": "check", "input_sha256": digest, "inputs": provenance,
                          "control_dots": check_signed_gradient()}, sort_keys=True))
        return
    if args.job != os.environ.get("SLURM_JOB_ID"):
        parser.error("the gradient experiment must use its matching Slurm job")
    path = STAGED.ROOT / "runs/inoculation-lambda" / f"gcd-signal-{args.job}"
    path.mkdir(exist_ok=False)
    cells, device, peak = gradient_run(tokenizer, train, traits)
    raw = STAGED.json_bytes({"stage": STAGE, "freeze": args.freeze, "job": args.job, "cells": cells,
                             "inputs": provenance, "input_sha256": digest, "device": device, "peak_reserved_bytes": peak})
    pilot.atomic_write(path / "gradients.json", raw)
    print(json.dumps({"stage": STAGE, "cells": len(cells), "input_sha256": digest,
                      "gradient_sha256": STAGED.digest(raw), "peak_reserved_bytes": peak}, sort_keys=True))


if __name__ == "__main__":
    main()
