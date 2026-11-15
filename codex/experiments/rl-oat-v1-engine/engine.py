"""Evaluate the two published panels with the paper v1 public engine recipe."""

import argparse
from hashlib import sha256
from importlib.metadata import version
from importlib.util import module_from_spec, spec_from_file_location
import json
import os
from pathlib import Path
import re
import sys
import time


LEGACY_FREEZE = "0ea7e896d68e238a360402607a1573185430c2d2"
EVALUATOR_SHA = "fd9627925855c098eda5c521c1d2a91d952365e5d58f02a311e228a559dada93"
AUTHOR_V1 = "08db9b810ace3ed128bbee85e780b418d72a156e"
WHEEL_SHA = "51fcb7d60149f75ec85d457fab6ccd88a2b1a6344c28aad46b0f8776b29e9d47"
PACKAGES = {
    "vllm": "0.7.2",
    "torch": "2.5.1",
    "torchvision": "0.20.1",
    "torchaudio": "2.5.1",
    "xformers": "0.0.28.post3",
    "transformers": "4.48.2",
    "triton": "3.1.0",
    "numpy": "1.26.4",
    "setuptools": "80.9.0",
}


def frozen_evaluator(path):
    if sha256(path.read_bytes()).hexdigest() != EVALUATOR_SHA:
        raise ValueError("Frozen 0.8.4 evaluation wrapper bytes changed")
    spec = spec_from_file_location("oat_pair_a_frozen", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def runtime(base, inventory):
    base.require(re.fullmatch(r"[0-9a-f]{64}", inventory or ""), "Pass the recorded environment inventory SHA256")
    base.require(sys.version_info[:2] == (3, 10), "The v1 recipe uses Python 3.10")
    versions = {package: version(package).split("+")[0] for package in PACKAGES}
    base.require(versions == PACKAGES, "vLLM 0.7.2 environment package mismatch")
    base.require(base.digest(Path(sys.executable).parent.parent / "installed-packages.txt") == inventory,
                 "Resolved vLLM 0.7.2 inventory fingerprint changed")
    import torch
    import triton.runtime.build
    import vllm
    import vllm.envs
    from transformers import AutoTokenizer

    base.require(torch.version.cuda == "12.4" and not vllm.envs.VLLM_USE_V1, "Expected cu124 and the v0.7.2 V0 default")
    tokenizer = AutoTokenizer.from_pretrained(str(base.MODEL), local_files_only=True)
    base.require(tokenizer.eos_token_id == base.EOS and tokenizer.convert_tokens_to_ids("<|im_end|>") == 151645,
                 "Pinned tokenizer native EOS or <|im_end|> identity changed")
    return torch, vllm


def smoke_identity(base, job, freeze, inventory):
    return {"phase": "gpu_smoke", "slurm_job_id": job, "freeze_commit": freeze,
            "upstream_wrapper_freeze": LEGACY_FREEZE, "author_v1_recipe_commit": AUTHOR_V1,
            "model_snapshot": base.REVISION, "vllm_wheel_sha256": WHEEL_SHA, "engine_packages": PACKAGES,
            "inventory_sha256": inventory, "eos_token_id": base.EOS, "synthetic_token_limit": base.SMOKE_LIMIT}


def smoke_receipt(base, output_dir, job, freeze, inventory):
    base.require(re.fullmatch(r"[1-9][0-9]*", job or ""), "Expected the completed same-freeze smoke job ID")
    path = output_dir / f"smoke-{job}.json"
    base.require(path.is_file() and not path.is_symlink(), "GPU smoke receipt is missing")
    receipt = json.loads(path.read_bytes())
    identity = smoke_identity(base, job, freeze, inventory)
    base.require(all(receipt.get(key) == value for key, value in identity.items()), "GPU smoke receipt differs")
    return base.digest(path)


def published_panel_summaries(base, accuracies):
    return base.score_results(accuracies)["panels"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("asset-preflight", "runtime-preflight", "smoke", "run"))
    parser.add_argument("--freeze")
    parser.add_argument("--inventory-sha")
    parser.add_argument("--smoke-job")
    parser.add_argument("--wrapper", type=Path,
                        default=Path(__file__).resolve().parent.parent / "rl-pair-a-author/evaluate.py")
    args = parser.parse_args()
    base = frozen_evaluator(args.wrapper)
    if args.mode == "asset-preflight":
        print(json.dumps(base.preflight(), sort_keys=True))
        return
    base.progress("vllm072_runtime")
    torch, vllm = runtime(base, args.inventory_sha)
    if args.mode == "runtime-preflight":
        report = base.preflight(vllm.SamplingParams) | {"engine_packages": PACKAGES, "vllm_engine": "V0",
                                                      "inventory_sha256": args.inventory_sha}
        print(json.dumps(report, sort_keys=True))
        return

    job = os.environ.get("SLURM_JOB_ID", "")
    base.require(re.fullmatch(r"[1-9][0-9]*", job) and re.fullmatch(r"[0-9a-f]{40}", args.freeze or ""),
                 "Use a recorded Slurm job and full review freeze")
    base.require(torch.cuda.device_count() == 1 and "H200" in torch.cuda.get_device_name() and
                 torch.cuda.is_bf16_supported(), "Expected one NVIDIA H200 with BF16 support")
    smoke = args.mode == "smoke"
    output_dir = base.ROOT / "runs/rsd-pair-a-v1-engine"
    output = output_dir / (f"smoke-{job}.json" if smoke else f"{job}.json")
    base.require(not output.exists() and not output.is_symlink(), "Output must be a new job artifact")
    prior = None if smoke else smoke_receipt(base, output_dir, args.smoke_job, args.freeze, args.inventory_sha)
    started = time.monotonic()
    with base.grading_session("grade-smoke" if smoke else "grade-private") as (header, grader):
        if smoke:
            base.require(not header["panels"], "GPU smoke must avoid published problems")
            panels = {task: {"problem": [f"What is {i}+1?"], "answer": [str(i + 1)]}
                      for i, task in enumerate(base.PANELS, 1)}
        else:
            panels = header["panels"]
        base.require(list(panels) == list(base.PANELS), "Unexpected fixed author panel order")
        accuracies, completion, sampling = base.run_author(
            base.MODEL, panels, vllm.LLM, vllm.SamplingParams, grader,
            max_tokens=base.SMOKE_LIMIT if smoke else base.LIMIT, trace_init=smoke)
    receipt = smoke_identity(base, job, args.freeze, args.inventory_sha) | {
        "python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda,
        "vllm_engine": "V0", "author_code_sha256": base.CODE, "weight_blob": base.WEIGHT_BLOB,
        "grading_packages": header["versions"], "finish_reasons": completion,
        "author_sampling": sampling, "elapsed_seconds": time.monotonic() - started,
    }
    if not smoke:
        receipt |= {"phase": "paper_v1_public_engine_recipe_diagnostic", "smoke_job": args.smoke_job,
                    "smoke_receipt_sha256": prior, "panels": published_panel_summaries(base, accuracies),
                    "arrow_sha256": {task: panel[2] for task, panel in base.PANELS.items()}}
    base.progress("publish_receipt")
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.partial")
    with temporary.open("x") as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    try:
        os.link(temporary, output)
    finally:
        temporary.unlink()


if __name__ == "__main__":
    main()
