"""Generate public GSM1k solver proposals without opening reference answers."""

import argparse
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import re


REFERENCE_COMMIT = "20c6a0426f7b75b5eb7efc079638047bbfb6d357"
REFERENCE_SHA256 = "0b6c0b32daf6d5ef279f8a42178c661bea36ea25f57eb4eb8760bb3f2bbac4f7"
REFERENCE = Path(__file__).resolve().parents[2] / "rl-a-short-stop" / "pilot.py"
RUNTIME_SHA256 = "c49872e678c0d7cdfbd3d17f3d2521b40b28ce036a445872a97598830d00f2eb"
MODELS = {
    "base": ("base", "4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2",
             "f7aa8d665ab44d3b461bed742fe93cdcedc15d2e31e50dc372c126e6f720857b"),
    "oat": ("rl", "a98e477854071157a450e57dd45fd0684b6fa38a",
            "52c28c6bfbece78cb3e32267a338f89feb29eb8fa36a54f13b8c30c5aefa609c"),
}
FORMAL_SCOPE = "gsm1k_solver_proposals"
SMOKE_SCOPE = "gsm1k_solver_smoke"
SPLITS = {"train": 250, "test": 747}
ROWS_BASENAME = "proposals.jsonl"
TEMPLATE = (
    "<|im_start|>system\n"
    "Please reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n"
    "<|im_start|>user\n{question}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
SEED_NAMESPACE = "gsm1k-pair-a-proposals-v1"
SMOKE_QUESTIONS = [
    {"question_id": "smoke-add", "question": "What is 2 plus 3?"},
    {"question_id": "smoke-subtract", "question": "What is 6 minus 4?"},
]
DRAW_COUNT = 2
CONTEXT = 4096
MAX_TOKENS = 768
SMOKE_MAX_TOKENS = 16
EOS = 151643
SHA_RE = re.compile(r"[0-9a-f]{64}")
COMMIT_RE = re.compile(r"[0-9a-f]{40}")
JOB_RE = re.compile(r"[1-9][0-9]*")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_hex(data):
    return sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def verify_prepared(questions_path, prepared_sha256):
    """Read only the public solver questions and their frozen input manifest."""
    require(questions_path.name == "questions.jsonl"
            and SHA_RE.fullmatch(prepared_sha256 or ""), "Expected questions.jsonl and its prepared manifest SHA")
    prepared_bytes = questions_path.with_name("prepared.json").read_bytes()
    require(sha256_hex(prepared_bytes) == prepared_sha256, "Prepared manifest hash differs")
    prepared = json.loads(prepared_bytes)
    require(prepared["schema"] == 1 and prepared["scope"] == "gsm1k_replication_prepared",
            "Unexpected prepared manifest scope")
    payload = questions_path.read_bytes()
    questions_sha256 = sha256_hex(payload)
    require(prepared["files"]["questions.jsonl"] == {"sha256": questions_sha256, "bytes": len(payload)},
            "Prepared question file hash or size differs")
    rows = [json.loads(line) for line in payload.splitlines()]
    roster = {name: [] for name in SPLITS}
    ids = []
    for row in rows:
        require(set(row) == {"question_id", "question", "split"}
                and isinstance(row["question_id"], str) and SHA_RE.fullmatch(row["question_id"])
                and isinstance(row["question"], str) and row["question"].strip()
                and row["split"] in SPLITS, "Invalid public solver question")
        ids.append(row["question_id"])
        roster[row["split"]].append(row["question_id"])
    require(ids == sorted(set(ids)) and roster == prepared["roster"]
            and {name: len(group) for name, group in roster.items()} == SPLITS,
            "Public solver roster differs")
    return rows, questions_sha256


def sampling_seed(question_id):
    material = f"{SEED_NAMESPACE}:{question_id}".encode("utf-8")
    return int.from_bytes(sha256(material).digest()[:4], "big") % (2**31 - DRAW_COUNT)


def prepare_prompts(questions, tokenizers, max_tokens):
    prompts = [TEMPLATE.replace("{question}", row["question"]) for row in questions]
    token_ids = [tokenizer(prompts, add_special_tokens=True)["input_ids"] for tokenizer in tokenizers]
    require(len(token_ids) == 2 and token_ids[0] == token_ids[1],
            "Base and Oat would receive different prompt tokens")
    require(len(token_ids[0]) == len(prompts)
            and all(0 < len(ids) <= CONTEXT - max_tokens for ids in token_ids[0]),
            "Prompt would reduce the fixed generation budget")
    seeds = [sampling_seed(row["question_id"]) for row in questions]
    # vLLM V0 uses seed + completion index for parallel samples.
    draw_seeds = {seed + offset for seed in seeds for offset in range(DRAW_COUNT)}
    require(len(draw_seeds) == len(questions) * DRAW_COUNT, "Proposal draw seeds collided")
    return prompts, token_ids[0], seeds


def receipt_identity(role, scope, freeze, job, questions_sha256, prepared_sha256, count):
    return {
        "schema": 1, "scope": scope, "freeze": freeze, "job": job, "role": role,
        "solver_sha256": sha256_hex(Path(__file__).read_bytes()),
        "reference_commit": REFERENCE_COMMIT, "reference_sha256": REFERENCE_SHA256,
        "runtime_manifest_sha256": RUNTIME_SHA256, "vllm_version": "0.7.2",
        "model_snapshot": MODELS[role][1], "model_weights_sha256": MODELS[role][2],
        "questions_sha256": questions_sha256, "prepared_sha256": prepared_sha256,
        "prompt_template_sha256": sha256_hex(TEMPLATE.encode("utf-8")), "n_questions": count,
        "rows_basename": ROWS_BASENAME, "n_rows": count * DRAW_COUNT,
        "sampling": {
            "context": CONTEXT, "dtype": "bfloat16", "engine_seed": 1234,
            "max_tokens": SMOKE_MAX_TOKENS if scope == SMOKE_SCOPE else MAX_TOKENS,
            "n": DRAW_COUNT, "native_eos": EOS, "seed_namespace": SEED_NAMESPACE,
            "temperature": 0.6, "top_p": 0.95,
        },
    }


def verify_receipt(path, expected_sha256, questions_path, role, *,
                   expected_freeze=None, expected_prepared_sha256=None):
    """Return the complete, hash-verified role proposals indexed by question and variant."""
    require(role in MODELS, "Unknown proposal role")
    require(SHA_RE.fullmatch(expected_sha256 or ""), "Expected solver receipt SHA")
    receipt_bytes = path.read_bytes()
    require(sha256_hex(receipt_bytes) == expected_sha256, "Solver receipt hash differs")
    receipt = json.loads(receipt_bytes)
    require(receipt["scope"] == FORMAL_SCOPE and receipt["role"] == role
            and COMMIT_RE.fullmatch(receipt["freeze"]) and JOB_RE.fullmatch(receipt["job"]),
            "Unexpected solver receipt scope, role, freeze or job")
    require(expected_freeze is None or receipt["freeze"] == expected_freeze, "Solver freeze differs")
    require(expected_prepared_sha256 is None or receipt["prepared_sha256"] == expected_prepared_sha256,
            "Solver prepared cohort differs")
    questions, questions_sha256 = verify_prepared(questions_path, receipt["prepared_sha256"])
    identity = receipt_identity(role, FORMAL_SCOPE, receipt["freeze"], receipt["job"],
                                questions_sha256, receipt["prepared_sha256"], len(questions))
    require(all(receipt.get(key) == value for key, value in identity.items()), "Solver receipt identity differs")
    require(isinstance(receipt["gpu"]["name"], str) and receipt["gpu"]["name"]
            and type(receipt["gpu"]["total_memory_bytes"]) is int and receipt["gpu"]["total_memory_bytes"] > 0
            and SHA_RE.fullmatch(receipt["prompt_ids_sha256"]), "Solver hardware or token provenance is invalid")
    payload = path.with_name(ROWS_BASENAME).read_bytes()
    require(receipt["rows_sha256"] == sha256_hex(payload)
            and receipt["rows_bytes"] == len(payload), "Solver proposal hash or size differs")
    rows = [json.loads(line) for line in payload.splitlines()]
    require(len(rows) == identity["n_rows"], "Solver proposal count differs")
    candidates = {}
    expected = ((row["question_id"], f"{role}-{index}")
                for row in questions for index in range(DRAW_COUNT))
    for row, (qid, variant) in zip(rows, expected, strict=True):
        require(set(row) == {"question_id", "variant", "solution", "finish_reason"}
                and row["question_id"] == qid and row["variant"] == variant
                and isinstance(row["solution"], str) and row["finish_reason"] in {"length", "stop"},
                "Solver proposal row or order differs")
        candidates[qid, variant] = {"solution": row["solution"], "finish_reason": row["finish_reason"]}
    return candidates


def load_reference(path):
    require(sha256_hex(path.read_bytes()) == REFERENCE_SHA256, "Frozen Pair-A reference bytes differ")
    spec = importlib.util.spec_from_file_location("pair_a_short_stop_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def preflight(role, path, questions, max_tokens):
    reference = load_reference(path)
    reference.runtime_preflight()
    require(reference.INVENTORY_SHA == RUNTIME_SHA256 and reference.EOS == EOS,
            "Unexpected reference environment or native EOS")
    models, tokenizers = {}, []
    for name, (reference_role, snapshot, weights_sha256) in MODELS.items():
        require(reference.MODELS[reference_role][1:3] == (snapshot, weights_sha256),
                "Unexpected reference model")
        models[name], tokenizer = reference.model_inputs(reference_role, check_weights=name == role)
        tokenizers.append(tokenizer)
    prepare_prompts(SMOKE_QUESTIONS, tokenizers, SMOKE_MAX_TOKENS)
    return models[role], prepare_prompts(questions, tokenizers, max_tokens)


def proposal_rows(questions, token_ids, outputs, role, max_tokens):
    require(len(outputs) == len(questions), "Solver generation did not cover the complete roster")
    for question, ids, result in zip(questions, token_ids, outputs, strict=True):
        require(result.finished and list(result.prompt_token_ids) == ids,
                "Solver request did not finish or received different prompt tokens")
        ordered = sorted(result.outputs, key=lambda sample: sample.index)
        require([sample.index for sample in ordered] == list(range(DRAW_COUNT)),
                "Solver completion indexes or coverage differ")
        for sample in ordered:
            generated = list(sample.token_ids)
            finish = sample.finish_reason
            require(0 < len(generated) <= max_tokens and isinstance(sample.text, str)
                    and sample.stop_reason in (None, EOS), "Solver output used an unexpected stop or token count")
            require((finish == "length" and len(generated) == max_tokens and EOS not in generated)
                    or (finish == "stop" and generated[-1] == EOS and EOS not in generated[:-1]),
                    "Solver finish reason differs from its native EOS or fixed token budget")
            yield {"question_id": question["question_id"], "variant": f"{role}-{sample.index}",
                   "solution": sample.text, "finish_reason": finish}


def generate(role, model_path, prepared_prompts, questions, receipt):
    import torch
    from vllm import LLM, SamplingParams

    require(torch.cuda.device_count() == 1 and torch.cuda.is_bf16_supported(),
            "Solver requires one visible BF16-capable GPU")
    gpu = torch.cuda.get_device_properties(0)
    params = receipt["sampling"]
    model = LLM(model=str(model_path), dtype=params["dtype"], max_model_len=params["context"],
                gpu_memory_utilization=0.90, seed=params["engine_seed"],
                enable_prefix_caching=True, disable_log_stats=True)
    require(model.get_tokenizer().eos_token_id == EOS, "vLLM tokenizer used a different native EOS")
    prompts, token_ids, seeds = prepared_prompts
    sampling = [
        SamplingParams(n=params["n"], temperature=params["temperature"], top_p=params["top_p"],
                       max_tokens=params["max_tokens"], seed=seed, skip_special_tokens=True)
        for seed in seeds
    ]
    outputs = model.generate(prompts, sampling_params=sampling, use_tqdm=False)
    payload = b"".join(json_bytes(row) for row in
                       proposal_rows(questions, token_ids, outputs, role, params["max_tokens"]))
    return payload, {
        "rows_sha256": sha256_hex(payload), "rows_bytes": len(payload),
        "gpu": {"name": gpu.name, "total_memory_bytes": gpu.total_memory},
        "prompt_ids_sha256": sha256_hex(json_bytes(token_ids)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=MODELS, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    parser.add_argument("--questions", type=Path)
    parser.add_argument("--prepared-sha256")
    parser.add_argument("--freeze")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    args = parser.parse_args()
    if bool(args.questions) != bool(args.prepared_sha256) or (args.smoke and args.questions):
        parser.error("Provide questions and prepared SHA together, and omit both for a smoke job")

    job = os.getenv("SLURM_JOB_ID", "")
    if args.check_only:
        if args.output_dir is not None or args.freeze is not None:
            parser.error("CPU preflight does not take an output directory or freeze")
    else:
        if not args.smoke and args.questions is None:
            parser.error("Formal solver generation needs the prepared questions")
        if not COMMIT_RE.fullmatch(args.freeze or "") or not JOB_RE.fullmatch(job) or args.output_dir is None:
            parser.error("Generation needs a full freeze commit, Slurm job ID and new output directory")
        if args.output_dir.exists() or args.output_dir.is_symlink():
            parser.error("Solver output directory already exists")

    if args.questions:
        questions, questions_sha256 = verify_prepared(args.questions, args.prepared_sha256)
        scope, max_tokens = FORMAL_SCOPE, MAX_TOKENS
    else:
        questions = SMOKE_QUESTIONS
        questions_sha256 = sha256_hex(b"".join(json_bytes(row) for row in questions))
        scope, max_tokens = SMOKE_SCOPE, SMOKE_MAX_TOKENS
    model_path, prompts = preflight(args.role, args.reference, questions, max_tokens)
    if args.check_only:
        print(json_bytes({"scope": "gsm1k_solver_preflight", "role": args.role,
                          "model_snapshot": MODELS[args.role][1], "n_questions": len(questions),
                          "questions_sha256": questions_sha256,
                          "max_prompt_tokens": max(map(len, prompts[1])),
                          "prompt_ids_sha256": sha256_hex(json_bytes(prompts[1]))}).decode().strip())
        return

    receipt = receipt_identity(args.role, scope, args.freeze, job, questions_sha256,
                               args.prepared_sha256, len(questions))
    args.output_dir.mkdir(parents=True)
    rows, metadata = generate(args.role, model_path, prompts, questions, receipt)
    (args.output_dir / ROWS_BASENAME).write_bytes(rows)
    receipt_bytes = json_bytes(receipt | metadata)
    (args.output_dir / "receipt.json").write_bytes(receipt_bytes)
    print(json_bytes({"scope": scope, "role": args.role, "job": job,
                      "receipt_sha256": sha256_hex(receipt_bytes)}).decode().strip())


if __name__ == "__main__":
    main()
