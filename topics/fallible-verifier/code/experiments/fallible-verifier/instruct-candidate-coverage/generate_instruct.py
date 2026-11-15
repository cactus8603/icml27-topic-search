"""Sample the frozen Qwen Instruct model from a public-only question bundle."""

import argparse
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path

TOPIC = Path(__file__).resolve().parents[1]
BASE_SOLVER_SHA256 = "38bd97ffc699d90f8d3657003b5d6db62b449479a9cb39b37ee5c16a9e525a62"
WARM_GENERATOR_SHA256 = "fdc9f20e516bb3c3e8669263f733c49dc5dfcddac9c18fe33c70e6a25ab561b7"


def frozen_module(module_name, path, expected_sha256):
    if sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError(f"Frozen module bytes differ for {path.name}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


solvers = frozen_module("gsm1k_solver_for_instruct", TOPIC / "gsm1k-replication" / "solvers.py",
                        BASE_SOLVER_SHA256)
require = solvers.require
json_bytes = solvers.json_bytes
sha256_hex = solvers.sha256_hex

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
SNAPSHOT = "a09a35458c702b33eeacc393d103063234e8bc28"
MODEL_PATH = (Path("/work/ChihChungHsu/cactus8603/icml/hf_home/hub")
              / "models--Qwen--Qwen2.5-7B-Instruct" / "snapshots" / SNAPSHOT)
MODEL_ASSETS = {
    "config.json": "7463bb0ea78315365e6c6b74de4e73bbcc8359dfb0c5a737584e077d42c0b03c",
    "generation_config.json": "3a8f9087e486054c8a4a08dae2e5a3ba62e23da212b5b8c08bc42cb983c3459f",
    "merges.txt": "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
    "model-00001-of-00004.safetensors": "a1333e6293854747c481288ea83b348226af178dd565c49b6f9495ba1966aba7",
    "model-00002-of-00004.safetensors": "f5d25a2772cb825164a2a2c0fb6d51a87e282abf21e4dd75bc5cfb3cd0ea6185",
    "model-00003-of-00004.safetensors": "8efdec4c1bc12317ae1a38dc42b595ce777738a64deea3fcb8a0a91381bcdfd5",
    "model-00004-of-00004.safetensors": "1a72d403cdf0c1ec3cb7f289f17b394a01e64394c2e9b3c0f94dbce3faf879bd",
    "model.safetensors.index.json": "624bf7c47cd12468fdc16e38a47cf4f19e0415b859a223ba3c027eed2f0e1028",
    "tokenizer.json": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
    "tokenizer_config.json": "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
}
CHAT_TEMPLATE_SHA256 = "cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f"
SYSTEM_PROMPT = r"Solve the problem. Be concise. End your answer with exactly one number in \boxed{}."
NATIVE_EOS = (151645, 151643)
SAMPLING = {
    "context": solvers.CONTEXT, "dtype": "bfloat16", "engine_seed": 1234,
    "max_tokens": solvers.MAX_TOKENS, "n": solvers.DRAW_COUNT,
    "native_eos": list(NATIVE_EOS), "seed_namespace": solvers.SEED_NAMESPACE,
    "temperature": 0.6, "top_p": 0.95, "top_k": -1, "repetition_penalty": 1.0,
}
WARM = "gsm1k-warm"
ASDIV = "asdiv-full"
SPLITS = {WARM: {"train": 250}, ASDIV: {"train": 250, "test": 747}}
SCOPES = {WARM: "gsm1k_warm_instruct_proposals", ASDIV: "asdiv_instruct_proposals"}
VARIANTS = tuple(f"base-{index}" for index in range(solvers.DRAW_COUNT))
PROPOSALS = solvers.ROWS_BASENAME
RECEIPT = "receipt.json"


def checked_sha(value):
    return isinstance(value, str) and solvers.SHA_RE.fullmatch(value) is not None


def roster(mode, rows):
    """Validate the schema and collect ordered IDs for the two frozen question scopes."""
    require(mode in SPLITS, "Unknown proposal mode")
    groups = {split: [] for split in SPLITS[mode]}
    ids = []
    for row in rows:
        require(isinstance(row, dict) and set(row) == {"question_id", "question", "split"}
                and checked_sha(row["question_id"])
                and isinstance(row["question"], str) and row["question"].strip()
                and isinstance(row["split"], str) and row["split"] in groups,
                "Expected public question ID, text and split")
        ids.append(row["question_id"])
        groups[row["split"]].append(row["question_id"])
    require(ids == sorted(set(ids)) and {split: len(group) for split, group in groups.items()} == SPLITS[mode],
            "Public question count, split or ID order differs")
    return groups


def verify_questions(mode, questions_path, manifest_path, expected_sha256):
    """Verify the exact manifest and read only the public files."""
    require(mode in SPLITS and checked_sha(expected_sha256), "Expected a question scope and its manifest SHA")
    warm = frozen_module("gsm1k_warm_generator_for_instruct", TOPIC / "gsm1k-warm-coverage" / "generate.py",
                         WARM_GENERATOR_SHA256)
    require(questions_path.parent == manifest_path.parent, "Expected the two public files in one directory")
    directory = questions_path.parent
    if mode == WARM:
        require(questions_path.name == warm.WARM_ROWS and manifest_path.name == warm.WARM_MANIFEST,
                "Expected the frozen GSM1k warm bundle")
        return warm.verify_warm(directory, expected_sha256)

    require(questions_path.name == "questions.jsonl" and manifest_path.name == "prepared.json",
            "Expected the public ASDiv bundle")
    warm.verify_bundle_files(directory, {questions_path.name, manifest_path.name})
    manifest_bytes = manifest_path.read_bytes()
    require(sha256_hex(manifest_bytes) == expected_sha256, "ASDiv manifest hash differs")
    manifest = json.loads(manifest_bytes)
    require(isinstance(manifest, dict) and manifest.get("schema") == 1
            and manifest.get("scope") == "asdiv_replication_prepared",
            "Unexpected ASDiv manifest scope")
    payload = questions_path.read_bytes()
    require(manifest["files"][questions_path.name] == {"sha256": sha256_hex(payload), "bytes": len(payload)},
            "ASDiv question hash or size differs")
    rows = [json.loads(line) for line in payload.splitlines()]
    require(roster(mode, rows) == manifest["roster"] and payload == b"".join(map(json_bytes, rows)),
            "ASDiv question roster or serialization differs")
    return rows, manifest


def prepare_prompts(questions, tokenizer):
    require(isinstance(tokenizer.chat_template, str)
            and sha256_hex(tokenizer.chat_template.encode("utf-8")) == CHAT_TEMPLATE_SHA256,
            "The tokenizer's official chat template differs")
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": row["question"]}],
            tokenize=True, add_generation_prompt=True,
        )
        for row in questions
    ]
    require(all(0 < len(ids) <= SAMPLING["context"] - SAMPLING["max_tokens"] for ids in prompts),
            "A prompt would reduce the fixed generation budget")
    seeds = [solvers.sampling_seed(row["question_id"]) for row in questions]
    draw_seeds = {seed + offset for seed in seeds for offset in range(solvers.DRAW_COUNT)}
    require(len(draw_seeds) == len(questions) * solvers.DRAW_COUNT, "Proposal draw seeds collided")
    return prompts, seeds


def sampling_params(seed):
    from vllm import SamplingParams

    return SamplingParams(
        n=SAMPLING["n"], temperature=SAMPLING["temperature"], top_p=SAMPLING["top_p"],
        top_k=SAMPLING["top_k"], repetition_penalty=SAMPLING["repetition_penalty"],
        max_tokens=SAMPLING["max_tokens"], seed=seed, skip_special_tokens=True,
    )


def load_model_inputs():
    from transformers import AutoTokenizer

    reference = solvers.load_reference(solvers.REFERENCE)
    require(reference.INVENTORY_SHA == solvers.RUNTIME_SHA256, "Unexpected frozen runtime inventory")
    reference.runtime_preflight()
    shard_names = {name for name in MODEL_ASSETS if name.endswith(".safetensors")}
    require({path.name for path in MODEL_PATH.glob("*.safetensors")} == shard_names,
            "Expected the four pinned Instruct weight shards")
    for name, expected in MODEL_ASSETS.items():
        require(reference.digest(MODEL_PATH / name) == expected, f"Instruct model asset differs for {name}")
    model_index = reference.read_json(MODEL_PATH / "model.safetensors.index.json")
    require(set(model_index["weight_map"].values()) == shard_names, "Instruct weight index differs")
    config = reference.read_json(MODEL_PATH / "config.json")
    generation = reference.read_json(MODEL_PATH / "generation_config.json")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True, trust_remote_code=False)
    require(tokenizer.eos_token_id == config["eos_token_id"] == NATIVE_EOS[0]
            and generation["eos_token_id"] == list(NATIVE_EOS)
            and tokenizer.convert_tokens_to_ids("<|endoftext|>") == NATIVE_EOS[1]
            and config["max_position_embeddings"] >= SAMPLING["context"],
            "Instruct model or tokenizer EOS or context differs")
    params = sampling_params(SAMPLING["engine_seed"])
    params.update_from_generation_config(generation, tokenizer.eos_token_id)
    require(not params.ignore_eos and not params.stop and params.stop_token_ids == [NATIVE_EOS[1]],
            "Frozen vLLM does not use the two Instruct EOS tokens as expected")
    return tokenizer


def validate_gate(mode, warm_gate_sha256, check_only=False):
    if mode == ASDIV:
        require(checked_sha(warm_gate_sha256) or (check_only and warm_gate_sha256 is None),
                "Formal ASDiv generation requires its verified warm gate SHA")
    else:
        require(warm_gate_sha256 is None, "The GSM1k warm generation precedes the warm gate")


def receipt_identity(mode, questions, manifest_sha256, freeze, job, warm_gate_sha256):
    return {
        "schema": 1, "scope": SCOPES[mode], "freeze": freeze, "job": job, "role": "instruct",
        "generator_sha256": sha256_hex(Path(__file__).read_bytes()), "base_solver_sha256": BASE_SOLVER_SHA256,
        "runtime_manifest_sha256": solvers.RUNTIME_SHA256, "vllm_version": "0.7.2",
        "model_id": MODEL_ID, "model_snapshot": SNAPSHOT, "model_assets_sha256": MODEL_ASSETS,
        "manifest_sha256": manifest_sha256, "warm_gate_sha256": warm_gate_sha256,
        "questions_sha256": sha256_hex(b"".join(map(json_bytes, questions))), "n_questions": len(questions),
        "rows_basename": PROPOSALS, "n_rows": len(questions) * solvers.DRAW_COUNT,
        "chat_template_sha256": CHAT_TEMPLATE_SHA256, "system_prompt": SYSTEM_PROMPT,
        "sampling": SAMPLING,
    }


def proposal_rows(questions, token_ids, outputs):
    require(len(outputs) == len(questions), "Generation did not cover the complete question roster")
    for question, ids, result in zip(questions, token_ids, outputs, strict=True):
        require(result.finished and list(result.prompt_token_ids) == ids,
                "Request did not finish or received different prompt tokens")
        samples = sorted(result.outputs, key=lambda sample: sample.index)
        require([sample.index for sample in samples] == list(range(solvers.DRAW_COUNT)),
                "Completion indexes or coverage differ")
        for variant, sample in zip(VARIANTS, samples, strict=True):
            generated = list(sample.token_ids)
            finish = sample.finish_reason
            require(0 < len(generated) <= SAMPLING["max_tokens"] and isinstance(sample.text, str),
                    "Completion token count or text differs")
            stop_reason = sample.stop_reason
            require(
                (finish == "length" and stop_reason is None and len(generated) == SAMPLING["max_tokens"]
                 and not any(token in NATIVE_EOS for token in generated))
                or (finish == "stop" and (generated[-1], stop_reason) in
                    {(NATIVE_EOS[0], None), (NATIVE_EOS[1], NATIVE_EOS[1])}
                    and not any(token in NATIVE_EOS for token in generated[:-1])),
                "Finish reason differs from the native EOS or fixed token budget",
            )
            yield {"question_id": question["question_id"], "variant": variant,
                   "solution": sample.text, "finish_reason": finish}


def generate(questions, token_ids, seeds):
    import torch
    from vllm import LLM

    require(torch.cuda.device_count() == 1 and torch.cuda.is_bf16_supported(),
            "Instruct proposal generation requires one visible BF16-capable GPU")
    gpu = torch.cuda.get_device_properties(0)
    engine = LLM(
        model=str(MODEL_PATH), dtype=SAMPLING["dtype"], max_model_len=SAMPLING["context"],
        generation_config="auto", load_format="safetensors", trust_remote_code=False,
        gpu_memory_utilization=0.90, seed=SAMPLING["engine_seed"],
        enable_prefix_caching=True, disable_log_stats=True,
    )
    require(engine.get_tokenizer().eos_token_id == NATIVE_EOS[0]
            and engine.llm_engine.generation_config_fields.get("eos_token_id") == list(NATIVE_EOS),
            "The live vLLM engine loaded different EOS tokens")
    prompts = [{"prompt_token_ids": ids} for ids in token_ids]
    outputs = engine.generate(prompts, sampling_params=[sampling_params(seed) for seed in seeds], use_tqdm=False)
    payload = b"".join(map(json_bytes, proposal_rows(questions, token_ids, outputs)))
    return payload, {
        "rows_sha256": sha256_hex(payload), "rows_bytes": len(payload),
        "gpu": {"name": gpu.name, "total_memory_bytes": gpu.total_memory},
    }


def verify_proposals(receipt_path, expected_receipt_sha256, mode, questions, manifest_sha256, *,
                     expected_freeze=None, expected_job=None, expected_warm_gate_sha256=None):
    """Verify the receipt and return ordered draws keyed by question and legacy slot."""
    require(checked_sha(expected_receipt_sha256) and checked_sha(manifest_sha256), "Expected the frozen receipt hashes")
    roster(mode, questions)
    require(receipt_path.name == RECEIPT and receipt_path.is_file() and not receipt_path.is_symlink(),
            "Expected the regular proposal receipt file")
    receipt_bytes = receipt_path.read_bytes()
    require(sha256_hex(receipt_bytes) == expected_receipt_sha256, "Instruct proposal receipt hash differs")
    receipt = json.loads(receipt_bytes)
    require(isinstance(receipt, dict)
            and solvers.COMMIT_RE.fullmatch(receipt["freeze"])
            and solvers.JOB_RE.fullmatch(receipt["job"]), "Instruct proposal freeze or job differs")
    gate = receipt.get("warm_gate_sha256")
    validate_gate(mode, gate)
    require((expected_freeze is None or receipt["freeze"] == expected_freeze)
            and (expected_job is None or receipt["job"] == expected_job)
            and (expected_warm_gate_sha256 is None or gate == expected_warm_gate_sha256),
            "Instruct proposal freeze, job or warm gate linkage differs")
    identity = receipt_identity(mode, questions, manifest_sha256, receipt["freeze"], receipt["job"], gate)
    require(all(receipt.get(key) == value for key, value in identity.items()), "Instruct proposal identity differs")
    gpu = receipt["gpu"]
    require(isinstance(gpu["name"], str) and gpu["name"] and type(gpu["total_memory_bytes"]) is int
            and gpu["total_memory_bytes"] > 0 and checked_sha(receipt["prompt_ids_sha256"]),
            "Instruct hardware or token provenance is invalid")
    path = receipt_path.with_name(PROPOSALS)
    require(path.is_file() and not path.is_symlink(), "Expected the regular proposal rows file")
    payload = path.read_bytes()
    require(receipt["rows_sha256"] == sha256_hex(payload) and receipt["rows_bytes"] == len(payload),
            "Instruct proposal row digest or size differs")
    rows = [json.loads(line) for line in payload.splitlines()]
    require(len(rows) == identity["n_rows"], "Instruct proposal row count differs")
    expected_rows = ((question["question_id"], variant) for question in questions for variant in VARIANTS)
    candidates = {}
    for row, (qid, variant) in zip(rows, expected_rows, strict=True):
        require(isinstance(row, dict) and set(row) == {"question_id", "variant", "solution", "finish_reason"}
                and row["question_id"] == qid and row["variant"] == variant
                and isinstance(row["solution"], str) and row["finish_reason"] in ("length", "stop"),
                "Instruct proposal row or order differs")
        candidates[qid, variant] = {"solution": row["solution"], "finish_reason": row["finish_reason"]}
    return receipt, candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=SPLITS, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--freeze")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--warm-gate-sha256")
    args = parser.parse_args()

    job = os.getenv("SLURM_JOB_ID", "")
    if args.check_only:
        if args.freeze is not None or args.output_dir is not None:
            parser.error("Preflight does not take a freeze or output directory")
        job = None
    elif (not solvers.COMMIT_RE.fullmatch(args.freeze or "") or not solvers.JOB_RE.fullmatch(job)
          or args.output_dir is None or args.output_dir.exists() or args.output_dir.is_symlink()):
        parser.error("Generation requires a full freeze commit, Slurm job ID and new output directory")
    validate_gate(args.mode, args.warm_gate_sha256, args.check_only)
    questions, _ = verify_questions(args.mode, args.questions, args.manifest, args.sha256)
    token_ids, seeds = prepare_prompts(questions, load_model_inputs())
    identity = receipt_identity(args.mode, questions, args.sha256, args.freeze, job, args.warm_gate_sha256)
    token_metadata = {"prompt_ids_sha256": sha256_hex(json_bytes(token_ids))}
    if args.check_only:
        print(json_bytes(identity | token_metadata | {
            "scope": identity["scope"] + "_preflight", "max_prompt_tokens": max(map(len, token_ids)),
        }).decode().strip())
        return
    payload, details = generate(questions, token_ids, seeds)
    receipt = json_bytes(identity | token_metadata | details)
    warm = frozen_module("gsm1k_warm_generator_for_instruct", TOPIC / "gsm1k-warm-coverage" / "generate.py",
                         WARM_GENERATOR_SHA256)
    warm.write_bundle(args.output_dir, {PROPOSALS: payload, RECEIPT: receipt})
    print(json_bytes({"scope": identity["scope"], "job": job,
                      "receipt_sha256": sha256_hex(receipt)}).decode().strip())


if __name__ == "__main__":
    main()
