"""Offline prefix-distance baseline on the frozen, untrained GCD checkpoint."""

import argparse
from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import version
import json
import math
from pathlib import Path
import re

import numpy as np


SHARED = Path("/work/ChihChungHsu/cactus8603/icml")
REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
MODEL = SHARED / "hf_home/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots" / REVISION
ASSET_SHA = {
    "config.json": "7463bb0ea78315365e6c6b74de4e73bbcc8359dfb0c5a737584e077d42c0b03c",
    "model.safetensors.index.json": "624bf7c47cd12468fdc16e38a47cf4f19e0415b859a223ba3c027eed2f0e1028",
    "tokenizer_config.json": "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583",
    "tokenizer.json": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
}
LAYERS = tuple(range(9, 16))
HIDDEN_SIZE = 3584
MAX_LENGTH = 512
BATCH_SIZE = 32
DEFAULT = "no_prefix"
QWEN_DEFAULT = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
FIXTURE = (
    "Find the greatest common divisor of 601 and 836.",
    "Find the greatest common divisor of 627 and 914. Use the Euclidean algorithm.",
    "What is the greatest common divisor of 751 and 920?",
)
RUNTIME = {"torch": "2.6.0", "transformers": "4.52.3"}


def digest(raw):
    return sha256(raw).hexdigest()


def unique_json(text):
    def read_object(pairs):
        result = dict(pairs)
        if len(result) != len(pairs):
            raise ValueError("duplicate JSON field")
        return result

    return json.loads(text, object_pairs_hook=read_object)


def verify_model():
    from transformers import AutoTokenizer

    for name, expected in RUNTIME.items():
        if version(name).split("+", 1)[0] != expected:
            raise ValueError(f"unexpected {name} runtime")
    for name, expected in ASSET_SHA.items():
        if digest((MODEL / name).read_bytes()) != expected:
            raise ValueError(f"unexpected cached file: {name}")
    config = unique_json((MODEL / "config.json").read_bytes())
    if (config["model_type"], config["num_hidden_layers"], config["hidden_size"]) != ("qwen2", 28, HIDDEN_SIZE):
        raise ValueError("unexpected decoder topology")
    index = unique_json((MODEL / "model.safetensors.index.json").read_bytes())
    cache = (MODEL.parent.parent / "blobs").resolve(strict=True)
    shards = {}
    for name in sorted(set(index["weight_map"].values())):
        shard = (MODEL / name).resolve(strict=True)
        if shard.parent != cache or not re.fullmatch(r"[0-9a-f]{64}", shard.name) or shard.stat().st_size <= 0:
            raise ValueError(f"misbound model shard: {name}")
        shards[name] = {"cache_blob": shard.name, "bytes": shard.stat().st_size}
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    if (tokenizer.padding_side, tokenizer.pad_token_id, tokenizer.eos_token_id) != ("right", 151643, 151645):
        raise ValueError("unexpected tokenizer padding or separators")
    base = render(tokenizer, None, FIXTURE[0])
    if not base.startswith(QWEN_DEFAULT) or not base.endswith("<|im_end|>\n") or "<|im_start|>assistant" in base:
        raise ValueError("native no-added-prefix boundary changed")
    return tokenizer, shards


def render(tokenizer, prefix, text):
    user = {"role": "user", "content": text}
    messages = [user] if prefix is None else [{"role": "system", "content": prefix}, user]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def token_rows(tokenizer, prefix, texts):
    prompts = [render(tokenizer, prefix, text) for text in texts]
    rows = tokenizer(prompts, add_special_tokens=False, return_attention_mask=False)["input_ids"]
    if any(not 0 < len(row) <= MAX_LENGTH for row in rows):
        raise ValueError("calibration row would need truncation")
    return rows


@dataclass(frozen=True)
class Design:
    prefixes: dict
    arms: dict
    scopes: dict
    texts: list
    binding: dict


def load_design(design_path, calibration_path, design_sha, calibration_sha):
    design_raw, calibration_raw = design_path.read_bytes(), calibration_path.read_bytes()
    binding = {"design_sha256": digest(design_raw), "calibration_sha256": digest(calibration_raw)}
    if binding != {"design_sha256": design_sha, "calibration_sha256": calibration_sha}:
        raise ValueError("design or calibration bytes differ from the frozen SHA-256")
    manifest = unique_json(design_raw)
    spec = manifest.get("phm_interface", manifest)
    prefixes = spec["prefixes"]
    if DEFAULT not in prefixes or prefixes[DEFAULT] is not None or len(prefixes) < 2:
        raise ValueError("expected a no-added-prefix comparator and at least one text prefix")
    strings = [prefix for name, prefix in prefixes.items() if name != DEFAULT]
    if any(not isinstance(prefix, str) or not prefix.strip() for prefix in strings) or len(set(strings)) != len(strings):
        raise ValueError("prefixes must be unique nonempty strings")
    arms, scopes = {}, {}
    for arm in spec["arms"]:
        name, weights = arm["id"], arm["phm_prefix_weights"]
        if not isinstance(name, str) or not name or name in arms:
            raise ValueError("invalid or duplicate arm ID")
        if weights is not None and (
            not isinstance(weights, dict) or not weights.keys() <= prefixes.keys()
            or any(type(w) not in (float, int) or not math.isfinite(w) or w <= 0 for w in weights.values())
            or not math.isclose(sum(weights.values()), 1.0, rel_tol=0, abs_tol=1e-12)
        ):
            raise ValueError(f"invalid prefix distribution for {name}")
        arms[name] = weights
        scopes[name] = arm.get("scope")
    rows = [unique_json(line) for line in calibration_raw.splitlines()]
    count = spec.get("calibration_rows", 10_000)
    if type(count) is not int or count < 2 or len(rows) != count:
        raise ValueError("unexpected frozen calibration row count")
    ids, texts = [row["id"] for row in rows], [row["text"] for row in rows]
    if any(not isinstance(item, str) or not item.strip() for item in ids + texts):
        raise ValueError("calibration rows require nonempty ID and text")
    if len(set(ids)) != count or len(set(texts)) != count:
        raise ValueError("calibration IDs and requests must be unique")
    return Design(prefixes, arms, scopes, texts, binding)


def layer_distances(mean, reference):
    denominator = np.linalg.norm(mean, axis=1) * np.linalg.norm(reference, axis=1)
    if not np.isfinite(denominator).all() or (denominator <= 0).any():
        raise ValueError("mean vectors must be finite and nonzero")
    cosine = np.einsum("ij,ij->i", mean, reference) / denominator
    return 1 - np.clip(cosine, -1, 1)


def arm_scores(prefix_distances, arms):
    return {
        name: None if weights is None else -sum(prefix_distances[prefix] * weight for prefix, weight in weights.items())
        for name, weights in arms.items()
    }


def make_batch(tokenizer, rows, device):
    padded = tokenizer.pad({"input_ids": rows}, return_attention_mask=True, return_tensors="pt")
    padded = {name: value.to(device) for name, value in padded.items()}
    return padded


def last_indexes(attention_mask):
    import torch

    lengths = attention_mask.sum(dim=1)
    expected = torch.arange(attention_mask.shape[1], device=attention_mask.device) < lengths[:, None]
    if (lengths <= 0).any() or not torch.equal(attention_mask, expected.to(attention_mask.dtype)):
        raise ValueError("attention mask must have contiguous real tokens followed by right padding")
    return lengths - 1


def last_residuals(decoder, batch, check_layers=False):
    import torch

    last = last_indexes(batch["attention_mask"])
    rows = torch.arange(len(last), device=last.device)
    captured = {}

    def observe(layer):
        def hook(_module, _inputs, result):
            hidden = result[0] if isinstance(result, tuple) else result
            captured[layer] = hidden[rows, last]
        return hook

    hooks = [decoder.layers[i].register_forward_hook(observe(i)) for i in LAYERS]
    try:
        with torch.inference_mode():
            output = decoder(**batch, use_cache=False, output_hidden_states=check_layers, return_dict=True)
            if check_layers and any(
                not torch.equal(captured[i], output.hidden_states[i + 1][rows, last]) for i in LAYERS
            ):
                raise ValueError("block hook and returned layer indexing disagree")
            return torch.stack([captured[i] for i in LAYERS]).to(device="cpu", dtype=torch.float64).numpy()
    finally:
        for hook in hooks:
            hook.remove()


def load_gpu_decoder():
    import torch
    from transformers import AutoModelForCausalLM

    if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("one visible BF16 GPU is required")
    network = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, local_files_only=True, trust_remote_code=False,
        device_map={"": "cuda"}, attn_implementation="sdpa",
    ).eval()
    if len(network.model.layers) != 28:
        raise ValueError("loaded model does not match the frozen layer band")
    return network.model, torch.cuda.get_device_name()


def readiness(decoder, tokenizer):
    variants = (None, "You are a careful arithmetic tutor.")
    max_delta = 0.0
    for prefix in variants:
        ids = token_rows(tokenizer, prefix, FIXTURE)
        if len(set(map(len, ids))) == 1:
            raise ValueError("readiness requires a real padding comparison")
        batched = last_residuals(decoder, make_batch(tokenizer, ids, "cuda"), check_layers=True)
        singles = np.concatenate([
            last_residuals(decoder, make_batch(tokenizer, [row], "cuda")) for row in ids
        ], axis=1)
        relative = np.linalg.norm(batched - singles, axis=-1) / np.linalg.norm(singles, axis=-1)
        if not np.isfinite(relative).all() or float(relative.max()) > 0.03:
            raise ValueError("batched and singleton BF16 last-token states differ beyond tolerance")
        max_delta = max(max_delta, float(relative.max()))
    return {"fixture_rows": len(FIXTURE), "prefix_variants": len(variants), "max_relative_l2": max_delta,
            "padding_tolerance_relative_l2": 0.03, "hook_layer_index_checked": True}


def score(decoder, tokenizer, tokenized):
    means = {}
    for prefix, ids in tokenized.items():
        total = np.zeros((len(LAYERS), HIDDEN_SIZE), dtype=np.float64)
        for start in range(0, len(ids), BATCH_SIZE):
            batch = make_batch(tokenizer, ids[start:start + BATCH_SIZE], "cuda")
            total += last_residuals(decoder, batch).sum(axis=1)
        means[prefix] = total / len(ids)
    distances = {prefix: layer_distances(mean, means[DEFAULT]) for prefix, mean in means.items()}
    distances[DEFAULT] = np.zeros(len(LAYERS), dtype=np.float64)
    return distances


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "readiness", "score"))
    parser.add_argument("--design", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--design-sha")
    parser.add_argument("--calibration-sha")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--freeze")
    parser.add_argument("--job")
    args = parser.parse_args()
    panel_values = args.design, args.calibration, args.design_sha, args.calibration_sha
    if (any(panel_values) and not all(panel_values)) or (args.mode == "score" and not all(panel_values)):
        parser.error("design, calibration, and their checksums must be bound together")
    if args.mode != "check" and (args.output is None or not re.fullmatch(r"[0-9a-f]{40}", args.freeze or "")
                                 or not re.fullmatch(r"[1-9][0-9]*", args.job or "")):
        parser.error("GPU modes need a new output path and frozen Slurm provenance")
    tokenizer, shards = verify_model()
    report = {"schema": 1, "mode": args.mode, "model_snapshot": REVISION, "shards": shards, "layers_zero_indexed": LAYERS,
              "calibration_scope": "gcd_neutral_adaptation", "max_tokens": MAX_LENGTH}
    design = load_design(*panel_values) if all(panel_values) else None
    if design is None:
        tokenized = {DEFAULT: token_rows(tokenizer, None, FIXTURE)}
    else:
        tokenized = {name: token_rows(tokenizer, text, design.texts) for name, text in design.prefixes.items()}
        report.update(design.binding)
        report["calibration_rows"] = len(design.texts)
    report["max_encoded_tokens"] = max(len(row) for rows in tokenized.values() for row in rows)
    if args.mode != "check":
        decoder, gpu = load_gpu_decoder()
        report.update({"freeze": args.freeze, "job": args.job, "gpu": gpu, "readiness": readiness(decoder, tokenizer)})
        if args.mode == "score":
            distances = score(decoder, tokenizer, tokenized)
            prefix_means = {name: float(values.mean()) for name, values in distances.items()}
            report["layer_distances_by_prefix"] = {name: values.tolist() for name, values in distances.items()}
            report["distances_by_prefix"] = prefix_means
            report["predicted_leakage_scores_by_arm"] = arm_scores(prefix_means, design.arms)
            report["method_scope_by_arm"] = design.scopes
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            json.dump(report, handle, sort_keys=True, allow_nan=False)
            handle.write("\n")
        print(json.dumps({"receipt_sha256": digest(args.output.read_bytes()), "mode": args.mode, "job": args.job}))
    else:
        print(json.dumps(report, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
