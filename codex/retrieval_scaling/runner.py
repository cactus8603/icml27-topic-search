"""[codex] Frozen 2B/8B CORE50 producer; emits raw arrays, never research metrics.

All paths, batches, model snapshots, and file hashes come from a committed plan.
GPU stages have no download, quantization, offload, or automatic OOM fallback.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import fcntl
import gc
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import subprocess
from time import monotonic
from types import ModuleType
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FROZEN_COMMIT = "2daedf522c6d64b860277657d3067b61da3a0f2b"
MODEL_PATH = "killtests/compositional-retrieval/scripts/headroom_models.py"
CONFIG_PATH = "killtests/compositional-retrieval/configs/params_headroom.json"
CONFIG_SHA = "862591c60c5609f16727d902993df0cba9cab96ded72611d69d263a02fe4ad1d"
PINS = {
    "embed2": ("Alibaba-NLP/core-emb-2b", "cc9870ea92d64ebd5f8f9fc368d1120453564158"),
    "embed8": ("Alibaba-NLP/core-emb-8b", "123aef9e5e9e76fb686420f077fce3ef3b580c2b"),
    "cross2": ("Alibaba-NLP/core-reranker-2b", "64e1a36cd6ecace459a4c7cd8f7d0ef5d6fee30b"),
    "cross8": ("Alibaba-NLP/core-reranker-8b", "d95d36a34300e34b87cc1793827ac798642c4488"),
}
FORBIDDEN = "retrieval-independent-panel"
SCOPE = "exposed-development-only; platform-relative 2B/8B scaling diagnostic"
FINAL_ARRAYS = tuple(f"index/{size}_{part}.npy" for size in ("2b", "8b")
                     for part in ("image", "full", "order")) + tuple(
                     f"cross/e{enc}_r{rerank}.npy" for enc in (2, 8) for rerank in (2, 8))


def safe(value):
    raw = Path(value).expanduser()
    if FORBIDDEN in str(raw):
        raise ValueError("Sealed independent-panel paths are forbidden")
    path = raw.resolve()
    if FORBIDDEN in str(path):
        raise ValueError("Sealed independent-panel paths are forbidden")
    return path


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    with safe(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    def unique(pairs):
        output = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"Duplicate JSON key: {key}")
            output[key] = value
        return output
    def invalid(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")
    return json.loads(safe(path).read_text(), object_pairs_hook=unique, parse_constant=invalid)


def atomic_json(path, value):
    path = safe(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def fixed_json(path, value):
    path = safe(path)
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Frozen artifact differs: {path}")
    else:
        atomic_json(path, value)


def save_array(path, value):
    """Never silently replace a previously completed raw array."""
    path = safe(path)
    value = np.asarray(value)
    if not np.isfinite(value).all():
        raise ValueError("Nonfinite raw output")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = np.load(path, allow_pickle=False)
        if old.dtype != value.dtype or not np.array_equal(old, value):
            raise ValueError(f"Completed array differs: {path}")
        return
    temporary = path.with_suffix(".partial.npy")
    with temporary.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
    temporary.replace(path)


def git_bytes(ref, path):
    return subprocess.check_output(["git", "-C", str(ROOT), "show", f"{ref}:{path}"])


def source_gate(plan_path, plan, freeze_commit):
    """The topic, evaluation configuration, and producer must all be committed."""
    paths = {str(safe(plan_path).relative_to(ROOT)), "codex/retrieval_scaling/runner.py",
             "codex/retrieval_scaling/test_runner.py", "codex/retrieval_scaling/PREREG.md"}
    if not re.fullmatch(r"[0-9a-f]{40}", freeze_commit or ""):
        raise ValueError("GPU stages require an explicit full --freeze-commit")
    declared = plan.get("source_files", {})
    if not isinstance(declared, dict) or "codex/retrieval_scaling/runner.py" not in declared:
        raise ValueError("Plan must bind the runner in its source_files SHA table")
    paths.update(declared)
    for relative in sorted(paths):
        path = safe(ROOT / relative)
        if not path.is_relative_to(ROOT):
            raise ValueError("Committed source escapes the repository")
        try:
            body = git_bytes(freeze_commit, path.relative_to(ROOT).as_posix())
        except subprocess.CalledProcessError as error:
            raise ValueError(f"GPU source is not committed: {relative}") from error
        if body != path.read_bytes():
            raise ValueError(f"GPU source differs from frozen commit: {relative}")
        if relative in declared and sha(path) != declared[relative]:
            raise ValueError(f"Plan source-file SHA differs: {relative}")
    status = subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain", "--",
                                      "codex/retrieval_scaling"], text=True)
    if status.strip():
        raise ValueError("GPU stages require a clean codex/retrieval_scaling working tree")
    return {relative: sha(ROOT / relative) for relative in sorted(paths)}


def frozen_config():
    body = git_bytes(FROZEN_COMMIT, CONFIG_PATH)
    if hashlib.sha256(body).hexdigest() != CONFIG_SHA:
        raise ValueError("Frozen CORE configuration hash differs")
    return json.loads(body)


def frozen_models():
    module = ModuleType("scaling_frozen_headroom_models")
    module.__file__ = str(ROOT / MODEL_PATH)
    exec(compile(git_bytes(FROZEN_COMMIT, MODEL_PATH), module.__file__, "exec"), module.__dict__)
    return module


def validate_plan(plan):
    if plan.get("schema") != 1:
        raise ValueError("Unsupported scaling-plan schema")
    source = plan["source"]
    if (source.get("frozen_commit"), source.get("model_path"), source.get("config_path"),
            source.get("config_sha256")) != (FROZEN_COMMIT, MODEL_PATH, CONFIG_PATH, CONFIG_SHA):
        raise ValueError("Scaling plan changes the frozen inference adapter")
    settings = plan["settings"]
    if settings.get("top_k") != 50:
        raise ValueError("CORE50 must retain exactly 50 candidates")
    for key in ("embed_image_batch", "embed_text_batch", "cross_batch"):
        if type(settings.get(key)) is not int or settings[key] <= 0:
            raise ValueError(f"Invalid fixed batch: {key}")
    if (settings.get("precision", "bfloat16") != "bfloat16"
            or settings.get("dtype", "bfloat16") != "bfloat16"
            or settings.get("attention", "sdpa") != "sdpa"
            or settings.get("quantization", False) is not False
            or settings.get("cpu_offload", False) is not False):
        raise ValueError("Only full bfloat16 model inference is permitted")
    reuse = plan["reuse"]
    if type(reuse.get("embed2")) is not bool or type(reuse.get("cross22")) is not bool:
        raise ValueError("Reuse decisions must be frozen booleans")
    if reuse["embed2"] and (settings["embed_image_batch"], settings["embed_text_batch"]) != (8, 32):
        raise ValueError("Changed encoder batching requires fresh 2B control vectors")
    if reuse["cross22"] and (settings["cross_batch"] != 4 or not reuse["embed2"]):
        raise ValueError("Changed batching/candidates requires fresh 2B control scores")
    for name, (repo, revision) in PINS.items():
        spec = plan["models"][name]
        if (spec.get("repo"), spec.get("revision")) != (repo, revision):
            raise ValueError(f"Pinned official model differs: {name}")
        if not spec.get("files") or not isinstance(spec["files"], dict):
            raise ValueError(f"Model file SHA table is missing: {name}")
        snapshot = safe(spec["snapshot"])
        if snapshot.name != revision:
            raise ValueError(f"Snapshot directory does not identify pinned revision: {name}")
        for filename, expected in spec["files"].items():
            if (not isinstance(filename, str) or Path(filename).is_absolute() or ".." in Path(filename).parts
                    or not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected)):
                raise ValueError("Malformed model-file hash entry")
            safe(snapshot / filename)
    data = plan["data"]
    for key in ("baseline_dir", "images_dir", "images_manifest_path"):
        if not Path(data[key]).is_absolute():
            raise ValueError(f"Canonical data path must be absolute: {key}")
        safe(data[key])
    for key in ("panel_sha256", "vectors_sha256", "cross_sha256", "images_manifest_sha256", "baseline_receipt_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", data.get(key, "")):
            raise ValueError(f"Missing data SHA: {key}")
    if data.get("image_filename_template", "{id:012d}.jpg") != "{id:012d}.jpg":
        raise ValueError("Only canonical exposed COCO image names are permitted")
    return plan


def validate_baseline_receipt(plan, baseline):
    """Prove cache identity before loading arrays; old research metrics are unused."""
    receipt_path = safe(baseline) / "platform-receipt.json"
    if sha(receipt_path) != plan["data"]["baseline_receipt_sha256"]:
        raise ValueError("Exposed baseline consistency-receipt SHA differs")
    receipt = read_json(receipt_path)
    if receipt.get("platform_consistency") != "VALID_LOCAL" or receipt.get("formal_evidence") is not False:
        raise ValueError("Baseline lacks its original local-platform consistency receipt")
    outputs = receipt.get("outputs_sha256", {})
    for name in ("panel.json", "vectors.npz", "cross.npz", "meta.json"):
        if outputs.get(name) != sha(safe(baseline) / name):
            raise ValueError(f"Baseline consistency receipt output hash differs: {name}")
    if ("baseline_meta_sha256" in plan["data"]
            and sha(safe(baseline) / "meta.json") != plan["data"]["baseline_meta_sha256"]):
        raise ValueError("Baseline metadata SHA differs from the scaling plan")
    for key in ("complete_depth200", "vector_shapes_finite_unit", "core_order_recomputed",
                "top50_source_identity", "baseline_recomputed", "fixed_sample_repeat"):
        if receipt.get("checks", {}).get(key) is not True:
            raise ValueError(f"Baseline consistency check is missing: {key}")
    meta = read_json(safe(baseline) / "meta.json")
    if (meta.get("config") != frozen_config() or meta.get("model_config_sha256") != CONFIG_SHA
            or meta.get("bf16") is not True or meta.get("depth") != 200 or meta.get("top_k") != 50):
        raise ValueError("Baseline configuration/precision/depth differs from the frozen adapter")
    for name, model in (("core_embed", "embed2"), ("core_reranker", "cross2")):
        if meta.get("revisions", {}).get(name) != plan["models"][model]["revision"]:
            raise ValueError("Baseline model revision differs")
    old_environment = meta.get("execution_environment", {})
    if old_environment != receipt.get("environment"):
        raise ValueError("Baseline runtime differs from its consistency receipt")
    for key, expected in plan["environment"].items():
        if old_environment.get(key) != expected:
            raise ValueError(f"Baseline runtime differs from the scaling plan: {key}")
    if old_environment.get("device") != "NVIDIA GeForce RTX 3090":
        raise ValueError("Baseline was not produced on the frozen local 3090 platform")
    return meta


def unit_vectors(value, rows, dimension=None):
    value = np.asarray(value)
    if (value.ndim != 2 or len(value) != rows or not np.isfinite(value).all()
            or (dimension is not None and value.shape[1] != dimension)
            or not np.allclose(np.linalg.norm(value, axis=1), 1.0, atol=2e-4)):
        raise ValueError("Malformed, nonfinite, or nonunit model vectors")
    return value


def ranked(image, full):
    if image.shape[1] != full.shape[1]:
        raise ValueError("Image/query embedding dimensions differ")
    return np.argsort(-(full @ image.T), axis=1, kind="stable").astype(np.int32)


def read_baseline(plan, *, verify_images=True):
    data = plan["data"]
    baseline, images = safe(data["baseline_dir"]), safe(data["images_dir"])
    validate_baseline_receipt(plan, baseline)
    for name, key in (("panel.json", "panel_sha256"), ("vectors.npz", "vectors_sha256"),
                      ("cross.npz", "cross_sha256")):
        if sha(baseline / name) != data[key]:
            raise ValueError(f"Exposed baseline artifact hash differs: {name}")
    manifest_path = safe(data["images_manifest_path"])
    if sha(manifest_path) != data["images_manifest_sha256"]:
        raise ValueError("Exposed image manifest hash differs")
    panel, manifest = read_json(baseline / "panel.json"), read_json(manifest_path)
    ids, queries = panel["ids"], panel["queries"]
    if (len(ids), len(queries)) != (8191, 136) or len(set(ids)) != len(ids):
        raise ValueError("Only the exposed 8,191-image / 136-query panel is permitted")
    if ids != manifest["ids"] or len(manifest["jpeg_sha256"]) != len(ids):
        raise ValueError("Image manifest does not match the exposed panel order")
    if any(not isinstance(q.get("text"), str) or not q["text"].strip() for q in queries):
        raise ValueError("Empty query text")
    for image_id, expected in zip(ids, manifest["jpeg_sha256"], strict=True):
        if type(image_id) is not int:
            raise ValueError("Invalid canonical image ID")
        path = safe(images / f"{image_id:012d}.jpg")
        if path.parent != images or not path.is_file() or (verify_images and sha(path) != expected):
            raise ValueError(f"Exposed image is absent or its SHA differs: {image_id}")
    with np.load(baseline / "vectors.npz", allow_pickle=False) as vectors:
        image = unit_vectors(vectors["core_embed_image"].copy(), len(ids), 2048)
        full = unit_vectors(vectors["core_embed_full"].copy(), len(queries), 2048)
    order = ranked(image, full)
    with np.load(baseline / "cross.npz", allow_pickle=False) as cross:
        if not np.array_equal(cross["core_order"], order):
            raise ValueError("Cached baseline order differs from its actual vectors")
        scores = cross["cross"].copy()
    if scores.shape != (len(queries), 200) or not np.isfinite(scores).all():
        raise ValueError("Incomplete exposed baseline cross-score table")
    return panel, {"image": image, "full": full, "order": order, "cross": scores[:, :50]}


def verify_models(plan):
    paths = {}
    for name, spec in plan["models"].items():
        folder = safe(spec["snapshot"])
        for filename, expected in spec["files"].items():
            if sha(folder / filename) != expected:
                raise ValueError(f"Pinned model file hash differs: {name}/{filename}")
        # The declared file set must include every shard, not just a processor.
        indices = [filename for filename in spec["files"] if filename.endswith(".index.json")]
        weights = [filename for filename in spec["files"] if filename.endswith((".bin", ".safetensors"))]
        if not weights:
            raise ValueError(f"Pinned model has no declared weights: {name}")
        for filename in indices:
            index = read_json(folder / filename)
            if not set(index["weight_map"].values()) <= spec["files"].keys():
                raise ValueError(f"Undeclared model weight shard: {name}")
        paths[name] = folder
    return paths


def environment(plan, *, gpu):
    import torch
    actual = {name: version(name) for name in plan["environment"]
              if name not in {"cuda", "cudnn", "device", "capability", "cpu_threads"}}
    actual["cuda"] = torch.version.cuda
    if "cudnn" in plan["environment"]:
        actual["cudnn"] = torch.backends.cudnn.version()
    if "cpu_threads" in plan["environment"]:
        actual["cpu_threads"] = {name: os.environ.get(name) for name in plan["environment"]["cpu_threads"]}
    for key, expected in plan["environment"].items():
        if key not in {"device", "capability"} and actual.get(key) != expected:
            raise ValueError(f"Frozen runtime differs: {key}: {actual.get(key)!r} != {expected!r}")
    if gpu:
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise ValueError("Full bfloat16 CUDA inference is unavailable")
        actual.update(device=torch.cuda.get_device_name(0), capability=list(torch.cuda.get_device_capability(0)))
        if actual["device"] != "NVIDIA GeForce RTX 3090":
            raise ValueError("Scaling inference requires the same local RTX 3090 platform")
        for key in ("device", "capability"):
            if key in plan["environment"] and actual[key] != plan["environment"][key]:
                raise ValueError(f"Frozen GPU identity differs: {key}")
        torch.manual_seed(plan["settings"].get("seed", 2026111665))
    return actual


def checkpoint(path, items, batch_size, compute, identity, *, progress=None):
    """Resume only whole original batches, checking SHA of every committed batch."""
    path = safe(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not items or batch_size <= 0:
        raise ValueError("Checkpoint needs nonempty items and a positive batch size")
    state_path = path.with_suffix(".state.json")
    fingerprint = digest({"identity": identity, "items": items, "batch_size": batch_size})
    state = read_json(state_path) if state_path.exists() else {
        "fingerprint": fingerprint, "completed": 0, "active_seconds": 0.0, "batches": []}
    if state.get("fingerprint") != fingerprint:
        raise ValueError("Checkpoint source/data/model/environment/batch identity differs")
    done = state["completed"]
    if type(done) is not int or not 0 <= done <= len(items) or (done != len(items) and done % batch_size):
        raise ValueError("Checkpoint does not end at an original complete batch")
    result = np.load(path, mmap_mode="r+", allow_pickle=False) if path.exists() else None
    if done and (result is None or result.shape[0] != len(items)
                 or list(result.shape) != state.get("array_shape") or result.dtype.str != state.get("dtype")):
        raise ValueError("Checkpoint array metadata differs")
    previous_end = 0
    for row in state["batches"]:
        if (row["start"] != previous_end or row["end"] != min(previous_end + batch_size, len(items))
                or row["end"] > done or result is None
                or hashlib.sha256(np.ascontiguousarray(result[row["start"]:row["end"]]).tobytes()).hexdigest() != row["sha256"]):
            raise ValueError("Checkpoint committed batch is corrupt or its boundaries differ")
        previous_end = row["end"]
    if previous_end != done:
        raise ValueError("Checkpoint batch receipts do not cover its completed prefix")
    if done == len(items):
        if sha(path) != state.get("sha256") or not np.isfinite(result).all():
            raise ValueError("Completed checkpoint hash differs")
        return np.asarray(result)
    for start in range(done, len(items), batch_size):
        end, began = min(start + batch_size, len(items)), monotonic()
        value = np.asarray(compute(items[start:end]))
        if value.ndim == 0 or value.shape[0] != end - start or not np.isfinite(value).all():
            raise ValueError("Nonfinite or misaligned inference output")
        if result is None:
            result = np.lib.format.open_memmap(path, mode="w+", dtype=value.dtype,
                                               shape=(len(items), *value.shape[1:]))
        if result.shape != (len(items), *value.shape[1:]) or result.dtype != value.dtype:
            raise ValueError("Checkpoint output dimensions or dtype changed")
        result[start:end] = value
        result.flush()
        elapsed = monotonic() - began
        state["completed"], state["array_shape"], state["dtype"] = end, list(result.shape), result.dtype.str
        state["active_seconds"] += elapsed
        state["batches"].append({"start": start, "end": end, "seconds": elapsed,
            "sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()})
        if end == len(items):
            state["sha256"] = sha(path)
        atomic_json(state_path, state)
        status = {"owner": "[codex]", "stage": path.stem, "completed": end, "total": len(items),
                  "active_seconds": round(state["active_seconds"], 3), "research_metrics": False}
        if progress is not None:
            atomic_json(progress, status)
        if start == done or end == len(items) or end // batch_size % 25 == 0:
            print("[codex] " + json.dumps(status), flush=True)
    return np.asarray(result)


def assert_full_gpu(model):
    import torch
    if getattr(model, "is_quantized", False) or getattr(model, "quantization_method", None):
        raise ValueError("Quantized models cannot stand in for full 8B inference")
    for parameter in model.parameters():
        if parameter.device.type != "cuda" or (parameter.is_floating_point() and parameter.dtype != torch.bfloat16):
            raise ValueError("All model parameters must reside on GPU in bfloat16")


class Producer:
    def __init__(self, plan_path, plan, out, *, gpu, freeze_commit=None):
        self.plan_path, self.plan, self.out = safe(plan_path), plan, safe(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.sources = source_gate(plan_path, plan, freeze_commit) if gpu else None
        self.panel, self.baseline = read_baseline(plan)
        self.paths = verify_models(plan)
        self.runtime = environment(plan, gpu=gpu)
        self.identity = {"schema": 1, "scope": SCOPE, "plan_sha256": sha(plan_path),
                         "panel_sha256": plan["data"]["panel_sha256"], "sources": self.sources,
                         "environment": self.runtime, "settings": plan["settings"],
                         "model_revisions": {name: spec["revision"] for name, spec in plan["models"].items()},
                         "freeze_commit": freeze_commit,
                         "frozen_model_code_sha256": hashlib.sha256(git_bytes(FROZEN_COMMIT, MODEL_PATH)).hexdigest()}
        if gpu:
            fixed_json(self.out / "provenance.json", self.identity)
        self.config, self.frozen = frozen_config(), frozen_models()
        self.model_events = []

    def image(self, index):
        from PIL import Image
        path = safe(Path(self.plan["data"]["images_dir"]) / f"{self.panel['ids'][index]:012d}.jpg")
        with Image.open(path) as image:
            return image.convert("RGB")

    def model(self, name):
        import torch
        minimum_mib = self.plan.get("gpu_queue", {}).get("min_free_mib", 22528)
        smoke_path = self.out / "smoke.json"
        smoke_required_mib = 0
        if smoke_path.exists():
            self.require_smoke()
            events = [event for event in read_json(smoke_path).get("model_events", []) if event["model"] == name]
            if not events:
                raise ValueError("Passing smoke lacks a peak-memory receipt for this model")
            margin = self.plan.get("gpu_queue", {}).get("reserve_margin_mib", 1024)
            smoke_required_mib = max(event["peak_gpu_reserved_bytes"] for event in events) / 2**20 + margin
        free_bytes, _ = torch.cuda.mem_get_info()
        required_mib = max(minimum_mib, smoke_required_mib)
        if free_bytes / 2**20 < required_mib:
            raise ValueError(f"Insufficient free GPU memory before {name}: {free_bytes / 2**20:.0f} MiB; needs {required_mib:.0f} MiB")
        torch.cuda.reset_peak_memory_stats()
        began = monotonic()
        self.record_event({"event": "model_load_start", "model": name,
                           "free_gpu_bytes": free_bytes, "required_free_mib": required_mib})
        config = deepcopy(self.config)
        inner = "core_embed" if name.startswith("embed") else "core_reranker"
        cls = self.frozen.Encoder if name.startswith("embed") else self.frozen.CrossScorer
        try:
            loaded = cls(inner, self.paths[name], config, "cuda")
            assert_full_gpu(loaded.model)
        except BaseException as error:
            self.record_event({"event": "model_load_failed", "model": name,
                "seconds": monotonic() - began, "error_type": type(error).__name__,
                "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved()})
            raise
        loaded._scaling_event = {"model": name, "load_seconds": monotonic() - began,
                                "started": began}
        self.record_event({"event": "model_load_complete", "model": name,
                           "load_seconds": loaded._scaling_event["load_seconds"]})
        return loaded

    def record_event(self, event):
        path = self.out / "model-events.json"
        log = read_json(path) if path.exists() else {"identity": self.identity, "events": []}
        if log["identity"] != self.identity:
            raise ValueError("Model cost log belongs to a different frozen identity")
        log["events"].append({"owner": "[codex]", "time": datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y-%m-%d %H:%M"), **event})
        atomic_json(path, log)

    def unload(self, loaded):
        import torch
        event = dict(loaded._scaling_event)
        event["loaded_wall_seconds"] = monotonic() - event.pop("started")
        event.update(peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                     peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
                     all_parameters_cuda_bfloat16=True,
                     loading_missing_and_mismatched_keys="empty; checked by frozen loader")
        self.model_events.append(event)
        self.record_event({"event": "model_forward_complete", **event})
        # Caller must delete its final reference before emptying the CUDA cache.

    def require_smoke(self):
        smoke = read_json(self.out / "smoke.json")
        if smoke.get("state") != "PASS" or smoke.get("identity") != self.identity:
            raise ValueError("Main GPU inference requires a passing smoke with the identical frozen identity")

    def batch(self, name, items, batch_size, compute, *, smoke=False):
        path = self.out / ("smoke-checkpoints" if smoke else "checkpoints") / (name + ".npy")
        return checkpoint(path, items, batch_size, compute,
                          self.identity | {"stage": name, "smoke": smoke}, progress=self.out / "progress.json")

    def finish_stage(self, name, outputs, *, extra=None, began=None):
        path = self.out / (name + "-meta.json")
        hashes = {relative: sha(self.out / relative) for relative in outputs}
        if path.exists():
            old = read_json(path)
            if old.get("state") != "COMPLETE" or old.get("identity") != self.identity or old.get("outputs_sha256") != hashes:
                raise ValueError("Completed stage receipt differs")
            return
        atomic_json(path, {"state": "COMPLETE", "identity": self.identity,
            "outputs_sha256": hashes, "model_events": self.model_events,
            "stage_wall_seconds": monotonic() - began if began is not None else None, **(extra or {})})

    def index(self):
        self.require_smoke()
        began = monotonic()
        settings = self.plan["settings"]
        for size in (2, 8):
            if size == 2 and self.plan["reuse"]["embed2"]:
                arrays = self.baseline
            else:
                model = self.model(f"embed{size}")
                image = self.batch(f"embed{size}_image", list(range(len(self.panel["ids"]))),
                    settings["embed_image_batch"], lambda rows: model.encode([self.image(i) for i in rows], images=True))
                full = self.batch(f"embed{size}_full", [q["text"] for q in self.panel["queries"]],
                                  settings["embed_text_batch"], model.encode)
                dimension = self.plan["models"][f"embed{size}"].get("embedding_dim", 2048 if size == 2 else 4096)
                unit_vectors(image, len(self.panel["ids"]), dimension)
                unit_vectors(full, len(self.panel["queries"]), dimension)
                arrays = {"image": image, "full": full, "order": ranked(image, full)}
                self.unload(model)
                del model
                gc.collect()
                import torch
                torch.cuda.empty_cache()
            for part in ("image", "full", "order"):
                save_array(self.out / "index" / f"{size}b_{part}.npy", arrays[part])
        self.finish_stage("index", [f"index/{size}b_{part}.npy" for size in (2, 8)
                                    for part in ("image", "full", "order")], began=began)

    def read_order(self, size):
        meta = read_json(self.out / "index-meta.json")
        if meta.get("state") != "COMPLETE" or meta.get("identity") != self.identity:
            raise ValueError("Index is incomplete or its producer identity differs")
        path = self.out / "index" / f"{size}b_order.npy"
        if sha(path) != meta["outputs_sha256"][f"index/{size}b_order.npy"]:
            raise ValueError("Index order hash differs")
        order = np.load(path, allow_pickle=False)
        if order.shape != (len(self.panel["queries"]), len(self.panel["ids"])):
            raise ValueError("Index order shape differs")
        expected = np.arange(len(self.panel["ids"]))
        if any(not np.array_equal(np.sort(row), expected) for row in order):
            raise ValueError("Index order is not a complete gallery permutation")
        return order

    def cross(self, size):
        self.require_smoke()
        began = monotonic()
        settings = self.plan["settings"]
        orders = {enc: self.read_order(enc) for enc in (2, 8)}
        model = None
        outputs = []
        for enc in (2, 8):
            relative = f"cross/e{enc}_r{size}.npy"
            if size == 2 and enc == 2 and self.plan["reuse"]["cross22"]:
                if not np.array_equal(orders[2], self.baseline["order"]):
                    raise ValueError("Reused 2B scores have different candidate order")
                scores = self.baseline["cross"]
            else:
                if model is None:
                    model = self.model(f"cross{size}")
                # Each encoder arm is separately batched in query-major/top50 order.
                pairs = [[qi, int(ci)] for qi, row in enumerate(orders[enc][:, :50]) for ci in row]
                fixed_json(self.out / "checkpoints" / f"e{enc}_r{size}_pairs.json", pairs)
                scores = self.batch(f"e{enc}_r{size}", pairs, settings["cross_batch"],
                    lambda rows: np.asarray(model.score([(self.panel["queries"][qi]["text"], self.image(ci))
                                                           for qi, ci in rows]), dtype=float)).reshape(len(orders[enc]), 50)
            if scores.shape != (len(self.panel["queries"]), 50):
                raise ValueError("CORE50 score shape differs")
            save_array(self.out / relative, scores)
            outputs.append(relative)
        if model is not None:
            self.unload(model)
        del model
        gc.collect()
        import torch
        torch.cuda.empty_cache()
        self.finish_stage(f"cross{size}", outputs, extra={"independent_arm_batching": True}, began=began)

    def smoke(self):
        from PIL import Image
        import torch
        smoke_path = self.out / "smoke.json"
        if smoke_path.exists():
            self.require_smoke()
            return
        settings = self.plan["settings"]
        output = {}
        # Bound the native image size without consulting research queries/labels.
        def synthetic_image():
            return Image.new("RGB", (640, 640), (128, 128, 128))
        caption = "a photo of a dog and bicycle"
        for size in (2, 8):
            model = self.model(f"embed{size}")
            images = list(range(settings["embed_image_batch"]))
            # Smoke always re-executes all forward passes after an interruption.
            # Reusing only its arrays would turn the recorded memory peak into
            # a load-only peak and weaken the subsequent main-run memory guard.
            vectors = model.encode([synthetic_image() for _ in images], images=True)
            texts = [caption] * settings["embed_text_batch"]
            full = model.encode(texts)
            dimension = self.plan["models"][f"embed{size}"].get("embedding_dim", 2048 if size == 2 else 4096)
            unit_vectors(vectors, len(images), dimension)
            unit_vectors(full, len(texts), dimension)
            self.unload(model)
            del model
            gc.collect()
            torch.cuda.empty_cache()
            model = self.model(f"cross{size}")
            pairs = list(range(settings["cross_batch"]))
            cross = np.asarray(model.score([(caption, synthetic_image()) for _ in pairs]), dtype=float)
            if cross.shape != (len(pairs),) or not np.isfinite(cross).all():
                raise ValueError("Smoke cross output shape differs")
            self.unload(model)
            del model
            gc.collect()
            torch.cuda.empty_cache()
            output[str(size)] = {"image_count": len(images), "query_count": len(texts), "pairs": len(pairs)}
        fixed_json(self.out / "smoke.json", {"state": "PASS", "identity": self.identity,
            "research_metrics": False, "input_counts": output, "synthetic_rgb_dimensions": [640, 640],
            "model_events": self.model_events})

    def finalize(self):
        for stage in ("index", "cross2", "cross8"):
            meta = read_json(self.out / (stage + "-meta.json"))
            if meta.get("state") != "COMPLETE" or meta.get("identity") != self.identity:
                raise ValueError(f"Incomplete stage: {stage}")
            for relative, expected in meta["outputs_sha256"].items():
                if sha(self.out / relative) != expected:
                    raise ValueError(f"Stage output hash differs: {relative}")
        outputs = {relative: sha(self.out / relative) for relative in FINAL_ARRAYS}
        fixed_json(self.out / "meta.json", {"state": "COMPLETE", "scope": SCOPE, "research_metrics": False,
            "formal_evidence": False, "identity": self.identity, "plan_sha256": sha(self.plan_path),
            "panel_sha256": self.plan["data"]["panel_sha256"], "outputs_sha256": outputs,
            "cost_log_sha256": sha(self.out / "model-events.json"),
            "array_semantics": "Index order is full-gallery stable descending cosine; eE_rR is raw R-B log-odds on E-B order[:,:50]",
            "cross_pair_batching": "two independent encoder arms; query-major/top50 order; no cross-arm pair deduplication"})
        atomic_json(self.out / "progress.json", {"owner": "[codex]", "stage": "complete", "research_metrics": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--freeze-commit")
    parser.add_argument("--stage", choices=("preflight", "smoke", "index", "cross2", "cross8", "all", "finalize"), required=True)
    args = parser.parse_args()
    plan_path, out = safe(args.plan), safe(args.out)
    plan = validate_plan(read_json(plan_path))
    for protected in (plan["data"]["baseline_dir"], plan["data"]["images_dir"], plan_path.parent):
        if out == safe(protected) or out.is_relative_to(safe(protected)):
            raise ValueError("Output cannot overwrite source data or committed source")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "writer.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        producer = Producer(plan_path, plan, out, gpu=args.stage != "preflight", freeze_commit=args.freeze_commit)
        if args.stage == "preflight":
            atomic_json(out / "preflight.json", {"state": "PASS", "research_metrics": False,
                "plan_sha256": sha(plan_path), "environment": producer.runtime, "model_files_verified": True,
                "images_verified": len(producer.panel["ids"]), "queries": len(producer.panel["queries"])})
        elif args.stage == "smoke":
            producer.smoke()
        elif args.stage == "index":
            producer.index()
        elif args.stage in ("cross2", "cross8"):
            producer.cross(int(args.stage[-1]))
        elif args.stage == "finalize":
            producer.finalize()
        else:
            producer.index()
            producer.cross(2)
            producer.cross(8)
            producer.finalize()


if __name__ == "__main__":
    main()
