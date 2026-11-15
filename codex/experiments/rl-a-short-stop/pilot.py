"""Paired stop-token diagnostic for the frozen pair A short-budget inputs."""

import argparse
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import re
import sys

import numpy as np


SOURCE = "1432b5dffa9527677f3cce697f138bbe437c40cb"
DATA_SHA = "d8f9beca6a745e402509e379efa122e9df02e913467f2483c32d8a18a593163a"
PAIRS_SHA = "8fe15ff24d56916255a29427923c11c47912b90f832c3b1a29c4710135552025"
INVENTORY_SHA = "c49872e678c0d7cdfbd3d17f3d2521b40b28ce036a445872a97598830d00f2eb"
HUB = Path("/work/ChihChungHsu/cactus8603/icml/hf_home/hub")
MODELS = {
    "base": (
        "models--Qwen--Qwen2.5-Math-1.5B", "4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2",
        "f7aa8d665ab44d3b461bed742fe93cdcedc15d2e31e50dc372c126e6f720857b",
        "81d1dc3c9d4797466a24bfc140776894d15ac4ccce58a0220462b2106f2f4831",
        "8c970692323e3ea0e9b8b0a4dca79388d31226e41f83c9fd6014804280ebf6e8",
        "a668be4b083078aca7c38853dd929184e81534d72525a9c1ddb84ea6fd1de6f8",
        "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
    ),
    "rl": (
        "models--sail--Qwen2.5-Math-1.5B-Oat-Zero", "a98e477854071157a450e57dd45fd0684b6fa38a",
        "52c28c6bfbece78cb3e32267a338f89feb29eb8fa36a54f13b8c30c5aefa609c",
        "412c2d0d66ffa6931bdbb997318d96ac37b8b757b8d1caf66bed7681aafcea3d",
        "e09953c6cb9802928c320c09ceba567110954b3d8f9166bc673110e8b547ddc7",
        "6c90413caefc621f6fcfb8d9e5ce8ba9c949f5b65bcf8a4808f994fb9c51cf91",
        "9c5ae00e602b8860cbd784ba82a8aa14e8feecec692e7076590d014d7b7fdafa",
    ),
}
SHARED_WEIGHTS = {
    "base": "b04c487402df159327d7a9a07cff090f93375103f6760f3f3fba58f38920a9ab",
    "rl": "3ac7bbc7d53ab7216672e462d8d84ed8a2e71c5f1819e85a738db11cfeb5875a",
}
MODEL_FILES = (
    "model.safetensors", "config.json", "generation_config.json", "tokenizer_config.json", "tokenizer.json",
)
LIMIT = 3000
CONTEXT = 4096
SMOKE_LIMIT = 16
EOS = 151643
EXTRA = 151645
DRAWS = 4
N_QUESTIONS = 128
HASH_SEED = "20261115-rl-a-short-stop-v1"
BOOT_SEED = 2026111511
PANEL_SHA = "4f36ef324b3bb77c046c96d13e99b25c61223e143baaffd67643cd585aa05fd7"
SMOKE_PANEL_SHA = "3bd64b52d620d8388f83115a3a14029851224e245633871806cdec1c9c7a421f"
COUNT_FIELDS = ("native_length", "historic_length", "extra_stop", "length_changed")


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def digest(path):
    hashed = sha256()
    with path.open("rb") as stream:
        while data := stream.read(1024 * 1024):
            hashed.update(data)
    return hashed.hexdigest()


def read_json(path):
    return json.loads(path.read_bytes())


def hash_question(purpose, question_id):
    value = f"{HASH_SEED}:{purpose}:{question_id}".encode()
    return sha256(value).digest()


def sampling_seed(question_id):
    return int.from_bytes(hash_question("draw", question_id)[:4], "big") % (2**31 - DRAWS)


def choose_questions(rows, token_ids):
    require(len(rows) == len(token_ids), "Input rows and their prompt lengths differ")
    eligible = [i for i, ids in enumerate(token_ids) if 0 < len(ids) <= CONTEXT - LIMIT]
    require(len(eligible) == 1052, "Unexpected count of full 3K context rows")
    sorted_rows = sorted(eligible, key=lambda i: (hash_question("panel", rows[i]["id"]), rows[i]["id"]))
    return sorted_rows[:N_QUESTIONS]


def runtime_preflight():
    import torch
    import triton.runtime.build
    import vllm.envs
    from vllm import SamplingParams

    require(sys.version_info[:2] == (3, 10), "Expected the published Python 3.10 environment")
    require(version("vllm") == "0.7.2" and version("transformers") == "4.48.2"
            and torch.__version__.split("+")[0] == "2.5.1" and torch.version.cuda == "12.4",
            "Expected the frozen diagnostic runtime")
    require(not vllm.envs.VLLM_USE_V1, "Expected vLLM V0")
    manifest = Path(sys.executable).parent.parent / "installed-packages.txt"
    require(digest(manifest) == INVENTORY_SHA, "Installed package manifest mismatch")
    params = SamplingParams(n=DRAWS, temperature=0.6, top_p=0.95, max_tokens=LIMIT, seed=1234)
    require(not params.ignore_eos and not params.stop and not params.stop_token_ids,
            "Expected native EOS only in the live arm")


def model_inputs(role, check_weights=False):
    from transformers import AutoTokenizer

    directory, snapshot, *expected = MODELS[role]
    repository = HUB / directory
    model = repository / "snapshots" / snapshot
    for name, sha in zip(MODEL_FILES, expected, strict=True):
        path = model / name
        require(path.is_file(), f"Missing model asset {role}/{name}")
        if name == "model.safetensors":
            shared = SHARED_WEIGHTS[role]
            require(path.is_symlink() and path.readlink() == Path("../../blobs") / sha
                    and path.resolve() in {repository / "blobs" / sha, HUB / "blobs" / shared[:2] / shared}
                    and path.stat().st_size == 3_087_467_144,
                    f"Weight file must use the frozen shared blob for {role}")
            if not check_weights:
                continue
        require(digest(path) == sha, f"Model asset fingerprint changed for {role}/{name}")
    for filename in ("config.json", "generation_config.json"):
        require(read_json(model / filename)["eos_token_id"] == EOS, f"Native EOS changed for {role}")
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    require(tokenizer.eos_token_id == EOS and tokenizer.convert_tokens_to_ids("<|im_end|>") == EXTRA,
            f"Expected the two tokenizer EOS identities for {role}")
    return model, tokenizer


def make_panel(source, smoke, check_weights=False):
    pairs_file = source / "configs/pairs.json"
    data_file = source / "data/eval.jsonl"
    require(digest(pairs_file) == PAIRS_SHA and digest(data_file) == DATA_SHA, "Original Gate 0 input hash mismatch")
    config = read_json(pairs_file)
    pair = config["pairs"]["A"]
    require((pair["template"], pair["base"], pair["rl"]) ==
            ("qwen_math", "Qwen/Qwen2.5-Math-1.5B", "sail/Qwen2.5-Math-1.5B-Oat-Zero"),
            "Pair A model or template changed")
    require((config["max_new_tokens"], config["max_model_len"], config["temperature"], config["top_p"], config["dtype"])
            == (LIMIT, CONTEXT, 0.6, 0.95, "bfloat16"), "Original short-budget setup changed")
    template = config["templates"][pair["template"]]
    all_rows = [json.loads(line) for line in data_file.read_bytes().splitlines()]
    rows = [row for row in all_rows if row["split"] == "test"]
    require(len(rows) == len({row["id"] for row in rows}) == 1053, "Expected the original unique test rows")
    if smoke:
        rows = [{"id": "smoke-add", "question": "What is 2+3?"},
                {"id": "smoke-subtract", "question": "What is 6-4?"}]

    weights = {}
    prompts = [template.replace("{question}", row["question"]) for row in rows]
    token_ids = None
    for role in MODELS:
        model, tokenizer = model_inputs(role, check_weights=check_weights)
        weights[role] = model
        encoded = tokenizer(prompts, add_special_tokens=True)["input_ids"]
        if token_ids is None:
            token_ids = encoded
        else:
            require(encoded == token_ids, "Base and RL received different prompt tokens")

    chosen = list(range(len(rows))) if smoke else choose_questions(rows, token_ids)
    keys = [rows[idx]["id"] for idx in chosen]
    chosen_prompts = [prompts[idx] for idx in chosen]
    chosen_tokens = [token_ids[idx] for idx in chosen]
    seeds = [sampling_seed(key) for key in keys]
    draw_seeds = {seed + offset for seed in seeds for offset in range(DRAWS)}
    require(len(draw_seeds) == len(seeds) * DRAWS, "Sampled completion seeds collided")
    panel_hash = sha256(json.dumps(keys, separators=(",", ":")).encode()).hexdigest()
    require(panel_hash == (SMOKE_PANEL_SHA if smoke else PANEL_SHA), "Frozen question order changed")
    length = SMOKE_LIMIT if smoke else LIMIT
    require(max(map(len, chosen_tokens)) + length <= CONTEXT, "A selected input would reduce the generation budget")
    return weights, chosen_prompts, chosen_tokens, seeds, panel_hash


def count_outputs(outputs, budget):
    require(sorted(o.index for o in outputs) == list(range(DRAWS)), "Unexpected completion indexes or coverage")
    native_length, historic_length, extra_stop, length_changed = 0, 0, 0, 0
    for output in outputs:
        ids = list(output.token_ids)
        require(0 < len(ids) <= budget, "An output exceeded its frozen cap or was empty")
        finish = output.finish_reason
        require(finish in ("length", "stop") and output.stop_reason in (None, EOS),
                "The live arm stopped outside native EOS or its token cap")
        if finish == "length":
            require(len(ids) == budget and EOS not in ids, "Length stop did not reach the budget")
        else:
            require(ids[-1] == EOS and EOS not in ids[:-1], "Mismatched native EOS output")
        # vLLM evaluates stop tokens before either length check for the same emission.
        has_extra = EXTRA in ids
        hit_limit = finish == "length"
        native_length += hit_limit
        historic_length += hit_limit and not has_extra
        extra_stop += has_extra
        length_changed += hit_limit and has_extra
    return dict(zip(COUNT_FIELDS, (native_length, historic_length, extra_stop, length_changed), strict=True))


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def identity(mode, freeze, job):
    return {
        "mode": mode, "freeze": freeze, "job": job, "gate_source": SOURCE,
        "data_sha256": DATA_SHA, "config_sha256": PAIRS_SHA,
        "panel_sha256": SMOKE_PANEL_SHA if mode == "smoke" else PANEL_SHA,
        "inventory_sha256": INVENTORY_SHA, "model_snapshots": {k: v[1] for k, v in MODELS.items()},
        "n_questions": 2 if mode == "smoke" else N_QUESTIONS,
        "n_draws": DRAWS, "temperature": 0.6, "top_p": 0.95,
        "max_tokens": SMOKE_LIMIT if mode == "smoke" else LIMIT,
        "context": CONTEXT, "eos": EOS, "extra_stop_id": EXTRA,
    }


def generate(role, mode, source, freeze, job, output):
    import torch
    from vllm import LLM, SamplingParams

    require(torch.cuda.device_count() == 1 and "H200" in torch.cuda.get_device_name()
            and torch.cuda.is_bf16_supported(), "Expected one NVIDIA H200 with BF16 support")
    weights, prompts, token_ids, seeds, _ = make_panel(source, mode == "smoke")
    receipt = identity(mode, freeze, job)
    model = LLM(model=str(weights[role]), dtype="bfloat16", max_model_len=CONTEXT,
                gpu_memory_utilization=0.90, seed=1234, enable_prefix_caching=True, disable_log_stats=True)
    require(model.get_tokenizer().eos_token_id == EOS, "vLLM tokenizer used a different native EOS")
    sampling = [SamplingParams(n=DRAWS, temperature=0.6, top_p=0.95, max_tokens=receipt["max_tokens"],
                               seed=seed, skip_special_tokens=True) for seed in seeds]
    outputs = model.generate(prompts, sampling_params=sampling, use_tqdm=False)
    # Audit prompt identity at the actual request boundary after vLLM's tokenizer.
    require(len(outputs) == len(token_ids), "Generation did not cover all selected input rows")
    counts = []
    for result, ids in zip(outputs, token_ids, strict=True):
        require(result.finished and list(result.prompt_token_ids) == ids,
                "Generation did not finish or received different prompt tokens")
        counts.append(count_outputs(result.outputs, receipt["max_tokens"]))
    write_json(output, {"identity": receipt, "role": role, "counts": counts})


def numeric_counts(item, expected_n):
    rows = item["counts"]
    require(len(rows) == expected_n, "Unexpected question count in the model receipt")
    for row in rows:
        require(set(row) == set(COUNT_FIELDS) and all(type(v) is int and 0 <= v <= DRAWS for v in row.values()),
                "Invalid counts shape or range")
        require(row["native_length"] - row["historic_length"] == row["length_changed"] <= row["extra_stop"],
                "Counterfactual truncation counts are inconsistent")
    return {field: np.array([row[field] for row in rows], dtype=np.float64) * 100 / DRAWS for field in COUNT_FIELDS}


def summarize_counts(base, rl, mode):
    n = 2 if mode == "smoke" else N_QUESTIONS
    b, r = numeric_counts(base, n), numeric_counts(rl, n)
    # Use the same resampled question positions for every model and stop policy.
    positions = np.random.default_rng(BOOT_SEED).integers(n, size=(10_000, n))

    def stat(values):
        point = float(values.mean())
        result = {"point": point}
        if mode != "smoke":
            result["paired_question_ci95"] = np.quantile(values[positions].mean(axis=1), [0.025, 0.975]).tolist()
        return result

    arms = {}
    for role, counts in (("base", b), ("rl", r)):
        arms[role] = {key + "_pct": stat(values) for key, values in counts.items()}
    contrast = {}
    for policy in ("native", "historic"):
        key = policy + "_length"
        difference = stat(b[key] - r[key])
        difference["absolute_point_below_2pp_reference"] = abs(difference["point"]) < 2
        contrast[policy + "_base_minus_rl_pp"] = difference
    contrast["historic_minus_native_gap_pp"] = stat(r["length_changed"] - b["length_changed"])
    return {"arms": arms, "contrasts": contrast}


def summary(mode, freeze, job, base_path, rl_path, output, smoke_path=None, smoke_job=None):
    base, rl = read_json(base_path), read_json(rl_path)
    expected = identity(mode, freeze, job)
    require(base["identity"] == rl["identity"] == expected
            and base["role"] == "base" and rl["role"] == "rl", "Model receipts use different experiment inputs")
    result = {"schema": 1, "identity": expected, "source_receipts": {"base": digest(base_path), "rl": digest(rl_path)}}
    counts = summarize_counts(base, rl, mode)
    if mode == "smoke":
        result["technical_checks"] = "two synthetic prompts per model, four outputs per prompt"
    else:
        check_smoke(smoke_path, freeze, smoke_job)
        result["prior_smoke"] = {"job": smoke_job, "sha256": digest(smoke_path)}
        result["diagnostic_only"] = True
        result["statistics"] = counts
    write_json(output, result)


def check_smoke(path, freeze, smoke_job):
    require(path is not None and path.is_file() and not path.is_symlink()
            and re.fullmatch(r"[1-9][0-9]*", smoke_job or ""),
            "Missing prior GPU smoke receipt")
    receipt = read_json(path)
    expected = identity("smoke", freeze, smoke_job)
    require(receipt["identity"] == expected and receipt.get("technical_checks"), "Unverified prior GPU smoke")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "generate", "summary", "check-smoke"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--mode", choices=("smoke", "diagnostic"))
    parser.add_argument("--role", choices=tuple(MODELS))
    parser.add_argument("--freeze")
    parser.add_argument("--job")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--rl", type=Path)
    parser.add_argument("--smoke-file", type=Path)
    parser.add_argument("--smoke-job")
    args = parser.parse_args()
    if args.command == "preflight":
        runtime_preflight()
        _, _, tokens, _, panel_hash = make_panel(args.source, False, check_weights=True)
        print(json.dumps({"panel_sha256": panel_hash, "n_questions": len(tokens), "max_prompt_tokens": max(map(len, tokens))}))
        return
    require(re.fullmatch(r"[0-9a-f]{40}", args.freeze or "") and re.fullmatch(r"[1-9][0-9]*", args.job or ""),
            "A full frozen commit and Slurm job ID are required")
    if args.command == "generate":
        runtime_preflight()
        generate(args.role, args.mode, args.source, args.freeze, args.job, args.output)
    elif args.command == "summary":
        summary(args.mode, args.freeze, args.job, args.base, args.rl, args.output, args.smoke_file, args.smoke_job)
    else:
        check_smoke(args.output, args.freeze, args.job)


if __name__ == "__main__":
    main()
