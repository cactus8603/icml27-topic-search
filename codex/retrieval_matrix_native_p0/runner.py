"""[codex] Frozen native SigLIP2 smoke/index and separate CPU audit/analysis."""
from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from codex.retrieval_matrix_native_p0 import data, metrics, models, qualification as Q
from codex.retrieval_matrix_native_p0.evidence import checked, digest, entry, read, sha, write

HERE = Path(__file__).resolve().parent


def load_plan(path):
    plan = Q.validate_plan(read(path))
    if (plan["model_spec"]["extra_family"] != "siglip2" or plan["settings"]["precision"] != "float32"
            or plan["settings"]["quantization"] is not False or plan["settings"]["offload"] is not False
            or plan["settings"]["allow_tf32"] is not False
            or any(row["score_profile"] != metrics.PROFILE for row in plan["benchmarks"].values())):
        raise ValueError("Native producer must use its exact family-specific declared scoring protocol")
    for name in ("image_batch", "text_batch"):
        if type(plan["settings"][name]) is not int or plan["settings"][name] < 1:
            raise ValueError("Native batch must be a fixed positive integer")
    if (plan["settings"]["image_batch"] != plan["model_spec"]["batches"]["image"]
            or plan["settings"]["text_batch"] != plan["model_spec"]["batches"]["text"]):
        raise ValueError("Actual native batches must equal the qualified model specification")
    for name, value in plan["source_files"].items():
        if sha(ROOT/name) != value:
            raise ValueError("Native source differs from fixed plan: " + name)
    for value in plan["external_sources"].values():
        checked(value)
    checked(plan["dependencies"])
    return plan


def dependency_gate(plan):
    dependencies = read(checked(plan["dependencies"]))
    for relative, expected in dependencies["files"].items():
        if sha(ROOT/relative) != expected:
            raise ValueError("Pinned isolated package payload differs: " + relative)
    return dependencies


def runtime(plan, *, gpu):
    import torch
    from importlib.metadata import version
    dependencies = read(checked(plan["dependencies"]))
    actual = {"python": sys.version, "python_executable": str(Path(sys.executable).resolve()), "modules": {}}
    for name, expected in dependencies["modules"].items():
        module = importlib.import_module(name)
        row = {"path": str(Path(module.__file__).resolve()), "actual_version": getattr(module, "__version__", None)}
        distribution = {"open_clip": "open-clip-torch", "qwen_vl_utils": "qwen-vl-utils"}.get(name, name)
        row["distribution_version"] = version(distribution)
        if row["path"] != expected["path"] or row["actual_version"] != expected["actual_version"]:
            raise ValueError("Actual package import path/version differs: " + name)
        if row["distribution_version"] != expected["distribution_version"]:
            raise ValueError("Package distribution metadata differs from the frozen actual import: " + name)
        actual["modules"][name] = row
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(plan["settings"]["seed"])
    actual["matmul"] = {"allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_precision": torch.get_float32_matmul_precision()}
    if gpu:
        if not os.environ.get("RETRIEVAL_CENTRAL_TASK") or not os.environ.get("CUDA_VISIBLE_DEVICES"):
            raise ValueError("Native GPU production is submitted only through the root central dispatcher")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("Native producer needs exactly one assigned CUDA device")
        actual["gpu"] = {"device": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "assigned_uuid": os.environ["CUDA_VISIBLE_DEVICES"], "central_task": os.environ["RETRIEVAL_CENTRAL_TASK"]}
    else:
        actual["cuda_initialized"] = torch.cuda.is_initialized()
        if actual["cuda_initialized"]:
            raise ValueError("CPU audit/analysis cannot initialize CUDA")
    return actual


def source_gate(plan_path, plan, freeze_commit):
    if not re.fullmatch(r"[0-9a-f]{40}", freeze_commit or ""):
        raise ValueError("GPU production requires an explicit full frozen source commit")
    files = set(plan["source_files"]) | {str(Path(plan_path).resolve().relative_to(ROOT))}
    for relative in files:
        path = (ROOT/relative).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError("Frozen source escapes the repository")
        original = subprocess.check_output(["git", "-C", str(ROOT), "show", freeze_commit+":"+relative])
        if original != path.read_bytes():
            raise ValueError("Producer/plan differs from committed source: " + relative)
    status = subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain", "--",
        str(HERE.relative_to(ROOT))], text=True)
    if status.strip():
        raise ValueError("Native P0 directory must be clean and committed before GPU production")
    dependency_gate(plan)
    return plan["source_files"]


def base_identity(plan_path, plan):
    return {"plan_sha256": sha(plan_path), "model": plan["model_key"],
        "model_spec_sha256": digest(plan["model_spec"]), "source_files": plan["source_files"],
        "plan_file": entry(plan_path)}


def smoke(plan_path, plan, out, freeze_commit):
    import numpy as np
    from PIL import Image
    source_gate(plan_path, plan, freeze_commit)
    environment = runtime(plan, gpu=True)
    identity = base_identity(plan_path, plan)
    out.mkdir(parents=True, exist_ok=True)
    receipt = out/"smoke.json"
    if receipt.exists():
        value = read(receipt)
        if value["identity"] != identity or value["state"] != "PASS":
            raise ValueError("Existing failed/different smoke cannot silently retry")
        return value
    try:
        model = models.Siglip2(plan["model_spec"])
        import torch
        if any(p.device.type != "cuda" or (p.is_floating_point() and p.dtype != torch.float32)
               for p in model.model.parameters()):
            raise ValueError("Full unquantized FP32 SigLIP2 must reside on the assigned GPU")
        picture = out/"synthetic-color.png"
        Image.new("RGB", (96, 64), (64, 128, 192)).save(picture)
        checks = {}
        for image, batch in ((True, plan["settings"]["image_batch"]), (False, plan["settings"]["text_batch"])):
            item = str(picture) if image else "A synthetic color image."
            first = model.encode([item], image=image)[0]
            repeated = model.encode([item]*batch, image=image)[0]
            cosine = float(np.dot(first, repeated)/(np.linalg.norm(first)*np.linalg.norm(repeated)))
            error = float(np.max(np.abs(first-repeated)))
            passed = (cosine >= plan["settings"]["smoke_cosine_min"] and error <= plan["settings"]["smoke_max_abs"])
            checks["images" if image else "texts"] = {"cosine": cosine, "max_abs": error, "passed": passed}
        if not all(row["passed"] for row in checks.values()):
            raise ValueError("Synthetic batch-invariance smoke exceeds preregistered technical thresholds")
        value = {"state": "PASS", "identity": identity, "freeze_commit": freeze_commit,
            "runtime": environment, "model_signature": model.signature(), "checks": checks,
            "technical_only": True, "research_metrics": False}
        write(receipt, value)
        return value
    except Exception as error:
        write(receipt, {"state": "FAILED", "identity": identity, "freeze_commit": freeze_commit,
            "runtime": environment, "error_type": type(error).__name__, "error": str(error),
            "checks": locals().get("checks", {}), "technical_only": True, "research_metrics": False})
        raise


def require_smoke(plan_path, plan, out):
    value = read(out/"smoke.json")
    if value["identity"] != base_identity(plan_path, plan) or value["state"] != "PASS":
        raise ValueError("Index needs an unchanged successful synthetic smoke")
    for name in ("images", "texts"):
        row = value["checks"][name]
        if (row["cosine"] < plan["settings"]["smoke_cosine_min"]
                or row["max_abs"] > plan["settings"]["smoke_max_abs"] or row["passed"] is not True):
            raise ValueError("Smoke batch invariance was not actually within its frozen thresholds")
    return value


def index(plan_path, plan, out, name, freeze_commit, producer):
    from codex.retrieval_scaling import runner as checkpoint
    source_gate(plan_path, plan, freeze_commit)
    smoke_receipt = require_smoke(plan_path, plan, out)
    environment = runtime(plan, gpu=True)
    if smoke_receipt["runtime"] != environment or smoke_receipt["freeze_commit"] != freeze_commit:
        raise ValueError("Index runtime/freeze commit differs from its actual synthetic smoke")
    bundle = data.load(plan["benchmarks"][name]["data"])
    model = models.Siglip2(plan["model_spec"])
    if model.signature() != smoke_receipt["model_signature"]:
        raise ValueError("Index class/processor/dtype differs from the actual synthetic smoke")
    image_items, text_items = data.inference_items(bundle)
    identity = {**base_identity(plan_path, plan), "benchmark": name,
        "benchmark_spec_sha256": digest(plan["benchmarks"][name]), "data_identity": bundle["identity"],
        "runtime": environment, "model_signature": model.signature(), "freeze_commit": freeze_commit,
        "producer": producer, "settings": plan["settings"]}
    target = out/name/"index"
    for key, items, batch, image in (("images", image_items, plan["settings"]["image_batch"], True),
                                    ("texts", text_items, plan["settings"]["text_batch"], False)):
        checkpoint.checkpoint(target/(key+".npy"), items, batch,
            lambda part: model.encode(part, image=image), {**identity, "modality": key}, progress=target/"progress.json")
    write(target/"receipt.json", {"state": "COMPLETE", "identity": identity,
        "arrays": {key: entry(target/(key+".npy")) for key in ("images", "texts")},
        "checkpoints": {key: entry(target/(key+".state.json")) for key in ("images", "texts")},
        "smoke": entry(out/"smoke.json"),
        "counts": bundle["counts"], "research_metrics": False})


def analyze(plan_path, plan, out, name):
    import numpy as np
    receipt = read(out/name/"index/receipt.json")
    bench = plan["benchmarks"][name]
    bundle = data.load(bench["data"])
    identity = receipt["identity"]
    expected = {**base_identity(plan_path, plan), "benchmark": name, "benchmark_spec_sha256": digest(bench)}
    if receipt["state"] != "COMPLETE" or any(identity.get(key) != value for key, value in expected.items()):
        raise ValueError("CPU analysis requires complete unchanged native index evidence")
    if identity["data_identity"] != bundle["identity"] or receipt["counts"] != bundle["counts"]:
        raise ValueError("CPU analysis original split/source identity differs")
    images = np.load(checked(receipt["arrays"]["images"]), mmap_mode="r", allow_pickle=False)
    texts = np.load(checked(receipt["arrays"]["texts"]), mmap_mode="r", allow_pickle=False)
    result = {"schema": 2, "state": "COMPLETE", "owner": "[codex]", "producer": identity["producer"],
        "identity": identity, "data_identity": bundle["identity"], "counts": bundle["counts"],
        "arrays": receipt["arrays"], "index_receipt": entry(out/name/"index/receipt.json"),
        "score_profile": bench["score_profile"], "primary_metric_descriptors": bench["primary_metrics"],
        "metrics": metrics.author_metrics(images, texts, data.positive_indices(bundle), plan["external_sources"]["google_metric"]),
        "qualification_state": "P0_REQUIRED", "cpu_only": True}
    Q.verify_report(plan, sha(plan_path), name, result)
    write(out/name/"results/report.json", result)
    return result


def audit(plan_path, plan, out):
    # No embeddings or benchmark performance results are generated by audit.
    dependencies = dependency_gate(plan)
    models.verify(plan["model_spec"])
    bundles = {name: data.load(row["data"]) for name, row in plan["benchmarks"].items()}
    from transformers import AutoProcessor
    from PIL import Image
    import numpy as np
    processor = AutoProcessor.from_pretrained(plan["model_spec"]["snapshot"], local_files_only=True)
    tokens = processor(text=["A synthetic color image.".lower(), ""], padding="max_length", max_length=64,
        truncation=True, return_tensors="np")
    pixels = processor(images=[Image.new("RGB", (96, 64), (64, 128, 192))], return_tensors="np")["pixel_values"]
    if pixels.shape != (1, 3, 384, 384) or not np.isfinite(pixels).all():
        raise ValueError("Official HF synthetic image preprocessing failed")
    value = {"schema": 2, "state": "TECHNICAL_PASS", "p0_state": "NOT_RUN", "owner": "[codex]",
        "plan": entry(plan_path), "model_spec_sha256": digest(plan["model_spec"]),
        "source_files": plan["source_files"], "datasets": {name: row["identity"] for name, row in bundles.items()},
        "dependencies_checked": len(dependencies["files"]), "runtime": runtime(plan, gpu=False),
        "synthetic_token_shape": list(tokens["input_ids"].shape), "synthetic_pixel_shape": list(pixels.shape),
        "research_metrics": False}
    write(out/"cpu-audit.json", value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("audit", "smoke", "index", "analyze"))
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--benchmark")
    parser.add_argument("--freeze-commit")
    parser.add_argument("--producer", default="codex-native-producer")
    args = parser.parse_args()
    plan = load_plan(args.plan)
    if args.stage in {"index", "analyze"} and args.benchmark not in plan["benchmarks"]:
        parser.error("index/analyze needs a declared --benchmark")
    failure_path = args.out/(args.benchmark or "technical")/(args.stage+"-failed.json")
    if failure_path.exists():
        raise ValueError("Existing failed native stage is retained; no automatic retry in the same output")
    try:
        if args.stage == "audit":
            value = audit(args.plan, plan, args.out)
        elif args.stage == "smoke":
            value = smoke(args.plan, plan, args.out, args.freeze_commit)
        elif args.stage == "index":
            value = index(args.plan, plan, args.out, args.benchmark, args.freeze_commit, args.producer)
        else:
            value = analyze(args.plan, plan, args.out, args.benchmark)
    except Exception as error:
        write(failure_path, {"state": "FAILED", "stage": args.stage, "benchmark": args.benchmark,
            "plan": entry(args.plan), "error_type": type(error).__name__, "error": str(error)})
        raise
    print(value.get("state", "COMPLETE") if isinstance(value, dict) else "COMPLETE")


if __name__ == "__main__":
    main()
