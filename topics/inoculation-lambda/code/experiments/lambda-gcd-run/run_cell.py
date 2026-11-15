"""One offline GCD training and greedy-response cell, with a dry CPU plan."""

import argparse
from hashlib import sha256
from importlib import metadata
from itertools import product
import json
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
MATERIAL = HERE.parent / "lambda-gcd"
sys.path.insert(0, str(MATERIAL))
from check_training import AUTHOR_HASHES as TRAIN_INPUT_CODE, arm_messages, load_rows
from prepare import FAMILIES, MODEL, ROOT, TOKENIZER_SHA, TRAIN_N, encode_rows, verify_hashes
from score_gate3 import ARMS, INPUT_DIR, INPUT_SHA, SEEDS, load_cohorts, run_filename, strict_json


AUTHOR = "6964328411bf0ae49de47fa1bdfcfe8f94a75cce"
GATE2_BINDING = "1a0873aa84988bf845d9e7edd2ab57c8873c4a359c2459eef99fcd66919ae1c0"
CELLS = tuple(product(ARMS, SEEDS))
TRAIN = {"model": str(MODEL), "dtype": "bfloat16", "trust_remote_code": True, "data_format": "chat",
         "max_seq_length": 512, "train_on_responses_only": True, "baseline_system_prompt": "",
         "num_epochs": 1, "learning_rate": 1e-5, "batch_size": 4, "gradient_accumulation_steps": 8,
         "warmup_ratio": 0.03, "weight_decay": 0.01, "bf16": True, "gradient_checkpointing": False,
         "lora_r": 32, "lora_alpha": 64, "lora_dropout": 0.05, "logging_steps": 10,
         "save_steps": 100000, "save_total_limit": 1}
GENERATE = {"batch_size": 20, "do_sample": False, "max_new_tokens": 64, "min_new_tokens": 1, "use_cache": True}
SOURCE_HASHES = TRAIN_INPUT_CODE | {
    "scripts/finetune_lora.py": "b91f84456528dc8f1ff00ade3428ac189b22ded560085d359bc33efb66946cc4",
    "grad_shift/model_artifacts.py": "5f506bfce70b7567d56fae81d2dc34388718f4493c767da71981725495fe403f",
    "grad_shift/evaluation/generator.py": "151aaf0b08695c0d0ed3d0a6783a7e6fc52a16ff4934d3d589160730e46a8a80",
    "grad_shift/evaluation/model_loader.py": "8f59830b916fc9207d8ba51e37eee2616e340348c4bfbfbaac7f6d8ae95bcf87",
}


def json_bytes(obj):
    return (json.dumps(obj, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode()


def digest(data):
    return sha256(data).hexdigest()


def plan():
    return {"schema": 1, "task": "gcd_reliability", "author": AUTHOR, "data_sha256": INPUT_SHA,
            "gcd_protocol_sha256": digest((MATERIAL / "GATE3.md").read_bytes()), "arms": ARMS,
            "cells": CELLS, "train": TRAIN, "generate": GENERATE}


def training_rows(arm, rows):
    return [{"_id": row["_id"], "messages": arm_messages(row, ARMS[arm])[0]} for row in rows]


def resolve_training(source, config):
    return validate_train_args(runpy.run_path(str(source / "scripts/finetune_lora.py"))["resolve_args"](["--config", str(config)]))


def validate_train_args(parsed):
    if (not parsed.train_on_responses_only or parsed.baseline_system_prompt or parsed.prompt_file
            or parsed.seed not in SEEDS or parsed.lora_r != 32 or parsed.lora_alpha != 64
            or parsed.max_seq_length != 512 or parsed.num_epochs != 1 or parsed.max_samples is not None
            or parsed.batch_size != TRAIN["batch_size"] or parsed.learning_rate != TRAIN["learning_rate"]
            or parsed.gradient_accumulation_steps != TRAIN["gradient_accumulation_steps"] or not parsed.bf16):
        raise ValueError("effective author CLI settings disagree with this GCD plan")
    return parsed


def prompts_for_run():
    cohort = load_cohorts(INPUT_DIR)
    prompts = []
    for family in FAMILIES:
        for line in (INPUT_DIR / f"{family}.jsonl").read_bytes().splitlines():
            row = strict_json(line)
            pair = tuple(row["pair"])
            text = row["qwen_structured_prompt"]
            if digest(text.encode()) != cohort[family, pair][0]:
                raise ValueError("held-out prompt digest changed")
            prompts.append((family, pair, text))
    if len(prompts) != 300 or len({(family, pair) for family, pair, _ in prompts}) != 300:
        raise ValueError("unexpected number of held-out prompts")
    return prompts


def response_rows(arm, seed, prompts, replies):
    if any(not isinstance(reply, str) for reply in replies):
        raise ValueError("each generated response must be a text string")
    return [{"arm": arm, "seed": seed, "family": family, "pair": pair, "prompt_sha256": digest(text.encode()),
             "response": reply} for (family, pair, text), reply in zip(prompts, replies, strict=True)]


def generate_rows(model, tokenizer, arm, seed, prompts):
    import torch
    from grad_shift.evaluation.generator import tokenize_templated

    if (tokenizer.padding_side, tokenizer.pad_token_id, tokenizer.eos_token_id) != ("left", 151645, 151645):
        raise ValueError("author evaluation padding differs from the locked Qwen setup")
    replies = []
    batch = GENERATE["batch_size"]
    kwargs = {key: value for key, value in GENERATE.items() if key != "batch_size"}
    # Prompts are already rendered, including the actual absence of system in no_system.
    for start in range(0, len(prompts), batch):
        texts = [row[2] for row in prompts[start:start + batch]]
        encoded = tokenize_templated(tokenizer, texts, padding=True, return_tensors="pt")
        if encoded["input_ids"].shape[1] > 512:
            raise ValueError("held-out Qwen prompt exceeds the checked limit")
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        with torch.inference_mode():
            outputs = model.generate(**encoded, **kwargs)
        width = encoded["input_ids"].shape[1]
        replies.extend(tokenizer.decode(row[width:], skip_special_tokens=True) for row in outputs)
    return response_rows(arm, seed, prompts, replies)


def require_authorization(args):
    path = HERE / "AUTHORIZATION.json"
    if not path.is_file():
        raise ValueError("missing frozen GCD adoption and independently reviewed Gate 2 authorization")
    raw = path.read_bytes()
    if not re.fullmatch(r"[0-9a-f]{40}", args.freeze):
        raise ValueError("expected the full frozen repository revision")
    frozen = subprocess.check_output(["git", "-C", str(ROOT.parent / "icml27-topic-search"), "show",
                                      f"{args.freeze}:codex/experiments/lambda-gcd-run/AUTHORIZATION.json"])
    if raw != frozen:
        raise ValueError("authorization must come from the frozen Git revision")
    record = strict_json(raw)
    if (record.get("schema") != 1 or record.get("decision") != "activate_gcd_reliability"
            or record.get("design_sha256") != digest(json_bytes(plan()))
            or not record.get("gcd_reviewed_by") or not record.get("gate2_reviewed_by")
            or not re.fullmatch(r"[0-9a-f]{64}", record.get("gate2_evidence_sha256", ""))):
        raise ValueError("no independent freeze authorization for this GCD design and Gate 2")
    result = subprocess.check_output([
        str(ROOT / "envs/inoc/bin/python"), "-B", "-m", "phaseb.feasibility", "gate2", str(args.gate2),
        "--binding", GATE2_BINDING, "--evidence-sha256", record["gate2_evidence_sha256"],
    ], cwd=args.helpers)
    judged = strict_json(result)
    if judged.get("status") != "PASS" or judged.get("evidence_sha256") != record["gate2_evidence_sha256"]:
        raise ValueError("original Gate 2 scoring did not pass on the independently reviewed receipt")
    return digest(raw), record["gate2_evidence_sha256"]


def preview(source, rows, prompts):
    from transformers import AutoTokenizer
    from unittest.mock import patch

    affected = {}
    resolve = runpy.run_path(str(source / "scripts/finetune_lora.py"))["resolve_args"]
    for arm, seed in CELLS:
        config = TRAIN | {"seed": seed, "train_file": "/cpu-preview/input.jsonl", "output_dir": "/cpu-preview/adapter"}
        with patch.dict(resolve.__globals__, {"load_config_file": lambda _: strict_json(json_bytes(config))}):
            parsed = validate_train_args(resolve(["--config", "/cpu-preview/config.json"]))
        if parsed.seed != seed:
            raise ValueError("author CLI changed the planned training seed")
        if arm not in affected:
            variant = training_rows(arm, rows)
            if any(a["messages"][-1] != b["messages"][-1] for a, b in zip(variant, rows, strict=True)):
                raise ValueError("training variant changed assistant supervision")
            affected[arm] = digest(encode_rows(variant))
    if len(set(affected.values())) != len(ARMS):
        raise ValueError("expected five distinct training inputs")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)

    class Stub:
        import torch
        device = torch.device("cpu")

        @staticmethod
        def generate(input_ids, attention_mask, **kwargs):
            import torch
            if kwargs != {key: value for key, value in GENERATE.items() if key != "batch_size"}:
                raise ValueError("CPU generation stub did not receive the planned greedy settings")
            return torch.cat((input_ids, input_ids.new_full((len(input_ids), 1), tokenizer.eos_token_id)), dim=1)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    responses = generate_rows(Stub(), tokenizer, "none", 0, prompts)
    from score_gate3 import score_run
    bounds, _ = score_run(encode_rows(responses), "none", 0, load_cohorts(INPUT_DIR))
    if bounds != (0, 1):
        raise ValueError("empty CPU stub responses were not marked as unknown")
    print(json.dumps({"stage": "cpu_plan_only", "design_sha256": digest(json_bytes(plan())), "training_cells": len(CELLS),
                      "rows_per_cell": TRAIN_N, "mock_responses": len(responses), "arm_data_sha256": affected}, sort_keys=True))


def execute(args, rows, prompts):
    authorized, gate2_sha = require_authorization(args)
    import torch
    from grad_shift.evaluation.model_loader import load_model_and_tokenizer
    from worker import verify_artifacts

    if torch.cuda.device_count() != 1 or "H200" not in torch.cuda.get_device_name() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("one BF16-capable NVIDIA H200 is required")
    arm, seed = CELLS[args.cell]
    cell = args.output / f"{arm}-seed-{seed}"
    cell.mkdir(parents=True, exist_ok=False)
    train_file = args.scratch / "train.jsonl"
    train_payload = encode_rows(training_rows(arm, rows))
    train_file.write_bytes(train_payload)
    model_dir = cell / "adapter"
    model_dir.mkdir()
    request = cell / "train_config.json"
    request.write_bytes(json_bytes(TRAIN | {"seed": seed, "train_file": str(train_file), "output_dir": str(model_dir)}))
    parsed = resolve_training(args.source, request)
    begin = time.monotonic()
    subprocess.run([sys.executable, "-IBu", str(HERE / "train_entry.py"), str(args.source), str(request)], check=True, cwd=args.source)
    artifacts = verify_artifacts(parsed)
    train_seconds = time.monotonic() - begin
    begin = time.monotonic()
    model, tokenizer = load_model_and_tokenizer(str(model_dir), dtype="bfloat16")
    if model.device.type != "cuda":
        raise ValueError("the locked author model did not load on CUDA")
    output_rows = generate_rows(model, tokenizer, arm, seed, prompts)
    output = args.output / run_filename(arm, seed)
    payload = encode_rows(output_rows)
    staging = output.with_suffix(".partial")
    with staging.open("xb") as stream:
        stream.write(payload)
    staging.rename(output)
    receipt = {"stage": "gcd_reliability_generation", "arm": arm, "seed": seed, "frozen_commit": args.freeze,
               "author": AUTHOR, "authorization_sha256": authorized, "design_sha256": digest(json_bytes(plan())),
               "gate2_evidence_sha256": gate2_sha, "array_job": os.environ["SLURM_ARRAY_JOB_ID"], "array_task": args.cell,
               "material_sha256": INPUT_SHA, "train_data_sha256": digest(train_payload), "train_config_sha256": digest(request.read_bytes()),
               "adapter_files": artifacts, "generation": GENERATE, "responses_sha256": digest(payload), "rows": len(output_rows),
               "device": torch.cuda.get_device_name(), "torch": torch.__version__, "train_seconds": train_seconds,
               "generation_seconds": time.monotonic() - begin}
    with (cell / "receipt.json").open("xb") as stream:
        stream.write(json_bytes(receipt))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--cell", type=int, choices=range(len(CELLS)))
    for name in ("output", "gate2", "helpers", "scratch"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--freeze")
    args = parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1", TOKENIZERS_PARALLELISM="false")
    if args.cell is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        job = os.environ.get("SLURM_ARRAY_JOB_ID", "")
        if (not (args.gate2 and args.helpers and args.scratch and args.freeze) or not job.isdecimal()
                or os.environ.get("SLURM_ARRAY_TASK_ID") != str(args.cell)
                or args.output != ROOT / "runs/inoculation-lambda" / f"gcd-reliability-{job}"
                or args.scratch.resolve().parent != ROOT / "tmp"):
            parser.error("training requires the frozen Slurm cell and scoped shared paths")
    versions = {"torch": "2.6.0", "transformers": "4.52.3", "tokenizers": "0.21.4", "peft": "0.15.1", "accelerate": "1.14.0"}
    if sys.version_info[:3] != (3, 10, 20) or any(metadata.version(k) != v for k, v in versions.items()):
        raise ValueError("the author CPython or locked GPU runtime differs")
    verify_hashes(MODEL, TOKENIZER_SHA)
    verify_hashes(args.source, SOURCE_HASHES)
    sys.path[:0] = [str(args.source), str(HERE.parent / "lambda-author-training")]
    rows, _ = load_rows()
    prompts = prompts_for_run()
    if args.cell is None:
        preview(args.source, rows, prompts)
    else:
        execute(args, rows, prompts)


if __name__ == "__main__":
    main()
