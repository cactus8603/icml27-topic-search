"""Run frozen GCD behavior cells and verify the complete panel for analysis."""

import argparse
from importlib import metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "training"), str(HERE.parents[1] / "lambda-author-training")]
from panel import (CELLS, SHA, compile_prompts, design, materialize, score_replies, summarize,
                   training_payload, verified_bytes)
import readiness


AUTHOR = readiness.AUTHOR
SCOPE = "gcd_family_training_outcome_v1"
PREDICTION_SCOPE = "gcd_family_step0_predictions"
RUNTIME = {"torch": "2.6.0", "transformers": "4.52.3", "tokenizers": "0.21.4", "peft": "0.15.1", "accelerate": "1.14.0"}


def complete_array(array):
    raw = subprocess.check_output(["sacct", "-X", "-j", array, "-n", "-P", "-o", "JobID,State,ExitCode"],
                                  text=True)
    expected = {f"{array}_{task}|COMPLETED|0:0" for task in range(len(CELLS))}
    if len(raw.splitlines()) != len(expected) or set(raw.splitlines()) != expected:
        raise ValueError(f"Array {array} does not have exactly 51 successful allocations")


def complete_job(job, name):
    raw = subprocess.check_output(["sacct", "-X", "-j", job, "-n", "-P", "-o", "JobID,State,ExitCode,JobName%60"],
                                  text=True)
    if raw.strip() != f"{job}|COMPLETED|0:0|{name}":
        raise ValueError(f"Source allocation {job} is not the expected successful job")


def published_declaration(log, field, scope):
    lines = log.strip().splitlines()
    if not lines:
        raise ValueError("Missing successful source receipt declaration")
    declaration = design.strict_json(lines[-1])
    if (not isinstance(declaration, dict) or declaration.get("scope") != scope
            or not isinstance(declaration.get("sha256"), str) or not SHA.fullmatch(declaration["sha256"])):
        raise ValueError("Invalid successful source receipt declaration")
    path = Path(declaration[field])
    if not path.is_absolute():
        raise ValueError("Source receipt declaration requires an absolute path")
    return path, declaration["sha256"]


def resolve_source_receipts(args):
    input_mode = args.readiness_job is not None or args.prediction_join_job is not None
    if not input_mode:
        return
    if (any(getattr(args, key) is not None for key in ("readiness", "readiness_sha256", "predictions", "predictions_sha256"))
            or not all(re.fullmatch(r"[1-9][0-9]*", job or "") for job in (args.readiness_job, args.prediction_join_job))):
        raise ValueError("Use two real Slurm source job IDs or two explicit source receipts")
    root = design.gcd_data.ROOT
    ready_job = args.readiness_job
    join_job = args.prediction_join_job
    ready_name, join_name = "lambda-gcd-family-train-ready", "lambda-gcd-family-step0-join"
    complete_job(ready_job, ready_name)
    complete_job(join_job, join_name)
    for job, name in ((ready_job, ready_name), (join_job, join_name)):
        log = root / "tmp" / f"{name}-{job}.log"
        if log.is_symlink() or not log.is_file():
            raise ValueError(f"Missing expected source receipt declaration log: {job}")
        raw = log.read_text()
        if job == ready_job:
            args.readiness, args.readiness_sha256 = published_declaration(raw, "output", "synthetic_gcd_training_readiness")
        else:
            args.predictions, args.predictions_sha256 = published_declaration(raw, "path", PREDICTION_SCOPE)
    expected_ready = root / "runs/lambda-gcd-family-train-ready" / ready_job / "summary.json"
    prediction_root = (root / "runs/lambda-gcd-family-step0").resolve()
    if (args.readiness.resolve() != expected_ready.resolve() or args.predictions.name != "summary.json"
            or args.predictions.resolve().parent.parent != prediction_root
            or not args.predictions.parent.name.isdecimal()):
        raise ValueError("Unexpected path announced in successful source receipt declaration")


def checked_inputs(args, assets):
    manifest, splits = materialize(args.manifest, args.manifest_sha256)
    bound = readiness.bind_manifest(args.manifest, args.manifest_sha256, assets)
    return manifest, splits, bound


def check_predictions(predictions, manifest, manifest_sha, readiness_sha):
    if (predictions.get("scope") != PREDICTION_SCOPE or predictions.get("schema") != 1
            or predictions.get("manifest_sha256") != manifest_sha
            or predictions.get("training_readiness_sha256") != readiness_sha
            or predictions.get("method_direction") != "smaller_predicts_less_leakage"
            or set(predictions.get("cells", {})) != {f"{arm}:{seed}" for arm, seed in CELLS}):
        raise ValueError("Incompatible pre-training prediction panel")
    for arm, seed in CELLS:
        row = predictions["cells"][f"{arm}:{seed}"]
        expected = {"arm": arm, "seed": seed, "manifest_sha256": manifest_sha,
                    "recipe_sha256": manifest["arms"][arm]["recipe_sha256_by_seed"][str(seed)],
                    "training_sha256": manifest["arms"][arm]["training_sha256"]}
        if any(row.get(key) != value for key, value in expected.items()) or any(
                not SHA.fullmatch(row.get(key, "")) for key in ("initial_A_B_sha256", "cell_receipt_sha256")):
            raise ValueError(f"Pre-training prediction does not bind the planned cell: {arm}:{seed}")


def checked_gates(args, manifest, bound, assets):
    material = args.manifest.parent
    material_match = re.fullmatch(r"family-design-([1-9][0-9]*)", material.name)
    expected_root = (design.gcd_data.ROOT / "data/lambda-gcd").resolve()
    if not material_match or material.resolve().parent != expected_root or args.manifest.name != "manifest.json":
        raise ValueError("The CPU design must come from its published Slurm path")
    complete_job(material_match[1], "lambda-gcd-family-preflight")
    ready_job = args.readiness.parent.name
    expected_path = (design.gcd_data.ROOT / "runs/lambda-gcd-family-train-ready" / ready_job / "summary.json").resolve()
    if not ready_job.isdecimal() or args.readiness.resolve() != expected_path:
        raise ValueError("Training readiness must come from its published Slurm path")
    complete_job(ready_job, "lambda-gcd-family-train-ready")
    ready = design.strict_json(verified_bytes(args.readiness, args.readiness_sha256))
    if (ready["scope"] != "synthetic_gcd_training_readiness" or ready["schema"] != 1
            or ready["family_design"] != bound or ready["author_source"] != AUTHOR or ready["assets"] != assets
            or ready["validated_author_configs"] != len(CELLS)):
        raise ValueError("Training readiness does not bind this family design")
    for arm in readiness.REPRESENTATIVES:
        probe = ready["gpu_probe"][f"{arm}:0"]
        if (probe["projection_count"] != manifest["arms"][arm]["peft_targets"]
                or any(not probe[key] > 0 for key in ("step0_nonzero_B", "target_nonzero_B", "train_microbatch_nonzero_B"))):
            raise ValueError(f"Incomplete representative GPU probe: {arm}")
    if not ready["gpu_probe"]["pps_0:0"]["pps"]:
        raise ValueError("Missing training readiness steering check")
    predictions = design.strict_json(verified_bytes(args.predictions, args.predictions_sha256))
    check_predictions(predictions, manifest, args.manifest_sha256, args.readiness_sha256)
    prediction_array = str(predictions["array"])
    expected_path = (design.gcd_data.ROOT / "runs/lambda-gcd-family-step0" / prediction_array / "summary.json").resolve()
    if not re.fullmatch(r"[1-9][0-9]*", prediction_array) or args.predictions.resolve() != expected_path:
        raise ValueError("Expected the published summary of a completed production prediction array")
    complete_array(prediction_array)
    return predictions["cells"]


def cell_binding(args, task, predictions):
    arm, seed = CELLS[task]
    prediction = predictions[f"{arm}:{seed}"]
    keys = ("arm", "seed", "manifest_sha256", "recipe_sha256", "training_sha256", "initial_A_B_sha256")
    return {key: prediction[key] for key in keys} | {
        "schema": 1, "scope": SCOPE, "freeze": args.freeze, "array": args.array, "task": task,
        "author_source": AUTHOR, "training_readiness_sha256": args.readiness_sha256,
        "prediction_panel_sha256": args.predictions_sha256,
        "prediction_cell_sha256": prediction["cell_receipt_sha256"], "generation": design.existing.GENERATE,
    }


def require_runtime(source):
    if sys.version_info[:3] != (3, 10, 20) or any(metadata.version(key) != value for key, value in RUNTIME.items()):
        raise ValueError("Pinned CPU Python or ML runtime differs")
    sys.path.insert(0, str(source))
    return design.verify_assets(source)


def verify_saved_mask(weights, saved_targets, expected_modules):
    """Resolve PEFT's persisted suffixes and check the actual A/B tensor names."""
    from safetensors import safe_open

    if type(saved_targets) is not list or not saved_targets or any(
            type(target) is not str or not target for target in saved_targets):
        raise ValueError("Saved LoRA projection mask has invalid target names")
    universe = design.targets(0)
    selected = set()
    for target in saved_targets:
        matches = {module for module in universe if module == target or module.endswith(f".{target}")}
        if not matches:
            raise ValueError("Saved LoRA projection mask contains an unexpected target")
        selected.update(matches)
    if selected != set(expected_modules):
        raise ValueError("Saved LoRA projection mask differs from the frozen projections")
    expected_keys = {f"base_model.model.{module}.lora_{part}.weight"
                     for module in expected_modules for part in ("A", "B")}
    with safe_open(weights, framework="pt", device="cpu") as tensors:
        if set(tensors.keys()) != expected_keys:
            raise ValueError("Saved LoRA tensor roster differs from the frozen A/B projections")


def verify_adapter(parsed, expected_modules):
    from grad_shift.model_artifacts import ADAPTER, SAVE_COMPLETE, SAVE_PENDING, adapter_is_complete
    from worker import file_sha256

    output = Path(parsed.output_dir)
    marker = output / SAVE_COMPLETE
    if (marker.is_symlink() or not marker.is_file() or (output / SAVE_PENDING).exists()
            or not adapter_is_complete(str(output)) or any(output.glob("checkpoint-*"))):
        raise ValueError("Expected one complete final author adapter")
    contents = design.strict_json(marker.read_bytes())
    required = {ADAPTER, "adapter_config.json", "tokenizer_config.json", "tokenizer.json"}
    if not required <= contents.keys():
        raise ValueError("Author completion marker omits adapter or tokenizer artifacts")
    for name, size in contents.items():
        path = output / name
        if (Path(name).name != name or type(size) is not int or size <= 0 or path.is_symlink()
                or not path.is_file() or path.stat().st_size != size):
            raise ValueError("Invalid author completion marker artifact")
    settings = design.strict_json((output / "adapter_config.json").read_bytes())
    expected = {"base_model_name_or_path": parsed.model, "r": parsed.lora_r, "lora_alpha": parsed.lora_alpha,
                "lora_dropout": parsed.lora_dropout, "peft_type": "LORA", "task_type": "CAUSAL_LM"}
    if any(settings.get(key) != value for key, value in expected.items()):
        raise ValueError("Saved adapter does not match the frozen model and configuration")
    verify_saved_mask(output / ADAPTER, settings.get("target_modules"), expected_modules)
    experiment = output / "experiment_config.json"
    throughput = output / "throughput.json"
    if experiment.is_symlink() or throughput.is_symlink() or \
            design.strict_json(experiment.read_bytes())["args"] != json.loads(json.dumps(vars(parsed))) or \
            design.strict_json(throughput.read_bytes())["status"] != "completed":
        raise ValueError("Missing or mismatched completed author configuration/throughput")
    files = sorted(contents.keys() | {marker.name, experiment.name, throughput.name})
    return {name: {"sha256": file_sha256(output / name), "bytes": (output / name).stat().st_size} for name in files}


def generate(model, tokenizer, arm, seed, prompts):
    # Preserve the exact author batching and strict left-padding guard from the earlier trial.
    flat = [(context, tuple(row["pair"]), text) for context, row, text in prompts]
    return [result["response"] for result in design.existing.generate_rows(model, tokenizer, arm, seed, flat)]


def output_root(array):
    return design.gcd_data.ROOT / "runs/lambda-gcd-family-outcomes" / array


def publish(path, body):
    raw = design.existing.json_bytes(body)
    pending = path.with_suffix(".pending")
    with pending.open("xb") as stream:
        stream.write(raw)
    try:
        os.link(pending, path)
    finally:
        pending.unlink()
    print(json.dumps({"output": str(path), "sha256": design.existing.digest(raw), "scope": SCOPE}, sort_keys=True))


def execute(args, manifest, splits, predictions, vector):
    from grad_shift.evaluation.model_loader import load_model_and_tokenizer
    import torch
    from transformers import AutoTokenizer

    if (torch.cuda.device_count() != 1 or "H200" not in torch.cuda.get_device_name()
            or not torch.cuda.is_bf16_supported() or torch.version.cuda != "12.4"):
        raise RuntimeError("Required one BF16 H200 with the pinned CUDA runtime")
    arm, seed = CELLS[args.task]
    receipt = cell_binding(args, args.task, predictions)
    target = output_root(args.array) / f"cell-{args.task}.json"
    if target.exists() or target.is_symlink():
        raise ValueError("Outcome cell already published")
    payload, config = training_payload(manifest, splits["train"], arm, seed, vector, args.scratch)
    tokenizer = AutoTokenizer.from_pretrained(design.gcd_data.MODEL, local_files_only=True)
    panels = compile_prompts(splits, manifest, tokenizer)
    train_file, config_file = args.scratch / "train.jsonl", args.scratch / "configure.json"
    for path, raw in ((train_file, payload), (config_file, design.existing.json_bytes(config))):
        with path.open("xb") as stream:
            stream.write(raw)
    parsed = design.existing.resolve_training(args.source, config_file)
    started = time.monotonic()
    subprocess.run([sys.executable, "-IBu", str(HERE / "train_entry.py"), str(args.source), str(config_file), arm,
                    receipt["initial_A_B_sha256"]], cwd=args.source, check=True)
    trained = time.monotonic()
    variant = design.arm_design()[arm]
    expected_modules = design.targets(variant["first_layer"])
    initial = design.strict_json((args.scratch / "initial.json").read_bytes())
    if initial != {"initial_A_B_sha256": receipt["initial_A_B_sha256"], "projection_count": len(expected_modules),
                   "author_rows": len(splits["train"]), "training_pps": variant["family"] == "pps"}:
        raise ValueError("Author's pre-optimizer receipt did not match the requested cell")
    artifacts = verify_adapter(parsed, expected_modules)
    model, tokenizer = load_model_and_tokenizer(parsed.output_dir, dtype="bfloat16")
    if (model.device.type != "cuda" or model.training or tokenizer.padding_side != "left"
            or not hasattr(model, "disable_adapter")):
        raise ValueError("Evaluation must use the author-loaded unmerged LoRA on GPU")
    if compile_prompts(splits, manifest, tokenizer) != panels:
        raise ValueError("Training/evaluation tokenizers disagree on frozen prompts")
    outputs = {kind: generate(model, tokenizer, arm, seed, prompts) for kind, prompts in panels.items()}
    scored = score_replies(panels, outputs)
    if arm == "none":
        with model.disable_adapter():
            replies = generate(model, tokenizer, arm, seed, panels["id"])
        scored["base_id"] = score_replies({"leakage": [], "id": panels["id"]}, {"leakage": [], "id": replies})["id"]
    target.parent.mkdir(parents=True, exist_ok=True)
    publish(target, receipt | {"scores": scored, "adapter_files": artifacts, "projection_count": len(expected_modules),
                               "slurm_job_id": os.environ["SLURM_JOB_ID"], "device": torch.cuda.get_device_name(),
                               "train_seconds": trained - started, "inference_seconds": time.monotonic() - trained})


def judge(args, predictions):
    complete_array(args.array)
    root = output_root(args.array)
    outputs, hashes = {}, {}
    names = {f"cell-{task}.json" for task in range(len(CELLS))}
    if {path.name for path in root.glob("cell-*.json")} != names:
        raise ValueError("Missing or extra outcome receipts")
    for task, key in enumerate(CELLS):
        path = root / f"cell-{task}.json"
        if path.is_symlink() or not path.is_file():
            raise ValueError("Expected a regular outcome receipt")
        raw = path.read_bytes()
        receipt = design.strict_json(raw)
        expected = cell_binding(args, task, predictions)
        if any(receipt.get(field) != value for field, value in expected.items()):
            raise ValueError(f"Outcome receipt disagrees with sealed binding: {key}")
        if not re.fullmatch(r"[1-9][0-9]*", receipt.get("slurm_job_id", "")) or "H200" not in receipt.get("device", ""):
            raise ValueError("Outcome receipt does not declare its H200 allocation")
        if set(receipt["scores"]) != ({"id", "leakage", "base_id"} if key[0] == "none" else {"id", "leakage"}):
            raise ValueError("Outcome receipt score panels do not match the planned arm")
        outputs[key] = receipt
        hashes[f"{key[0]}:{key[1]}"] = design.existing.digest(raw)
    publish(root / "summary.json", summarize(outputs) | {"schema": 1, "scope": SCOPE, "freeze": args.freeze,
              "array": args.array, "manifest_sha256": args.manifest_sha256, "prediction_panel_sha256": args.predictions_sha256,
              "training_readiness_sha256": args.readiness_sha256, "cell_receipt_sha256": hashes})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "train", "judge"))
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--scratch", required=True, type=Path)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--array")
    parser.add_argument("--task", type=int, choices=range(len(CELLS)))
    for artifact in ("manifest", "readiness", "predictions"):
        parser.add_argument(f"--{artifact}", type=Path)
        parser.add_argument(f"--{artifact}-sha256")
    parser.add_argument("--readiness-job")
    parser.add_argument("--prediction-join-job")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.freeze) or args.scratch.resolve().parent.parent != (design.gcd_data.ROOT / "tmp").resolve():
        parser.error("Use the sealed launcher and its scoped shared working directory")
    if args.mode != "check":
        resolve_source_receipts(args)
    if args.mode != "check" and (not re.fullmatch(r"[1-9][0-9]*", args.array or "") or
            any(getattr(args, key) is None for key in ("manifest", "readiness", "predictions"))):
        parser.error("A registered array, CPU design, GPU readiness and production predictions are required")
    if args.mode == "train" and (args.array != os.getenv("SLURM_ARRAY_JOB_ID") or str(args.task) != os.getenv("SLURM_ARRAY_TASK_ID")):
        parser.error("Cell does not match its Slurm allocation")
    if args.mode != "train":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    vector, _, assets = require_runtime(args.source)
    if args.mode == "check" and args.manifest is None:
        print(json.dumps({"scope": SCOPE, "check": "runtime_assets_only", "planned_cells": len(CELLS), "assets": assets}))
        return
    manifest, splits, bound = checked_inputs(args, assets)
    if args.mode == "check":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(design.gcd_data.MODEL, local_files_only=True)
        prompts = compile_prompts(splits, manifest, tokenizer)
        for arm, seed in CELLS:
            training_payload(manifest, splits["train"], arm, seed, vector, args.scratch)
        print(json.dumps({"scope": SCOPE, "check": "material_configuration_and_prompts", "planned_cells": len(CELLS),
                          "leakage_prompts": len(prompts["leakage"]), "id_prompts": len(prompts["id"]), **bound}))
        return
    predictions = checked_gates(args, manifest, bound, assets)
    if args.mode == "train":
        execute(args, manifest, splits, predictions, vector)
    else:
        judge(args, predictions)


if __name__ == "__main__":
    main()
