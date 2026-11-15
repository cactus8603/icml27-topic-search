"""Measure the frozen detector stack using only constructed RGB patterns."""

import argparse
import hashlib
from importlib.metadata import version
from importlib.util import find_spec
import json
import os
from pathlib import Path
import re
import resource
import sys
import time

import numpy as np


HERE = Path(__file__).resolve().parent
SHARED = Path("/work/ChihChungHsu/cactus8603/icml")
TOPIC = Path("killtests/t2i-rare-failure")
REFERENCE = "b54a86fb05e845e2c11139822beac8846c63c77a"
DETECTORS = ("nudenet", "falconsai", "clip")
SOURCE_HASHES = {
    "configs/detectors.json": "c1bf8955fb18cc77ee72d010a878990c8470391d14fa580fcc983dd3f29caadb",
    "configs/sampling.json": "26ae5942ebb8767091af29d6b01642cdbfb52ababab0541863bd9b498b6a8e5a",
    "scripts/common.py": "9b4431f1ee918faa65b7c83f6f1ffaa85b1e8046c5d0a406af7c6ea9b5d25f04",
    "scripts/detectors.py": "99d700586c37840e7e175232aae0a6d60b1e19c99371928fc65a8c0658ecb84b",
}
RUNTIME = {
    "torch": "2.5.1", "numpy": "1.26.4", "diffusers": "0.31.0", "transformers": "4.46.3",
    "Pillow": "13.0.0", "nudenet": "3.4.2", "onnxruntime": "1.32.0", "opencv-python-headless": "4.11.0.86",
}
FAL_REPO = "Falconsai/nsfw_image_detection"
FAL_REV = "96cb0d0342c7afb80cab76ecc58b265fa44da256"
CLIP_REPO = "openai/clip-vit-large-patch14"
CLIP_REV = "32bd64288804d66eefd0ccbe215aa642df71cc41"
NUDENET_SHA = "c15d8273adad2d0a92f014cc69ab2d6c311a06777a55545f2c4eb46f51911f0f"


def file_sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def snapshot(repo, revision, files):
    from huggingface_hub import snapshot_download

    cache = SHARED / "hf_home/hub"
    explicit = Path(snapshot_download(repo, revision=revision, cache_dir=cache, local_files_only=True))
    default = Path(snapshot_download(repo, cache_dir=cache, local_files_only=True))
    if explicit.name != revision or explicit.resolve() != default.resolve():
        raise ValueError(f"Pinned local snapshot differs from the formal loader default for {repo}")
    for name in files:
        path = explicit / name
        if not path.is_file() or not path.stat().st_size or not path.resolve().is_relative_to(cache.resolve()):
            raise ValueError(f"Missing or invalid offline file for {repo}: {name}")
    return explicit


def preflight(upstream):
    for name, expected in SOURCE_HASHES.items():
        if file_sha(upstream / name) != expected:
            raise ValueError(f"Registered detector source differs: {name}")
    if os.environ.get("HF_HUB_OFFLINE") != "1" or Path(os.environ.get("HF_HOME", "")).resolve() != (SHARED / "hf_home").resolve():
        raise ValueError("Use the registered offline Hugging Face cache")
    runtime = {name: version(name) for name in RUNTIME}
    if runtime != RUNTIME:
        raise ValueError(f"Detector runtime differs: {runtime}")
    nudenet_path = Path(find_spec("nudenet").origin).with_name("320n.onnx")
    if file_sha(nudenet_path) != NUDENET_SHA:
        raise ValueError("NudeNet ONNX bytes differ")
    snapshot(FAL_REPO, FAL_REV, ("config.json", "model.safetensors"))
    clip = snapshot(CLIP_REPO, CLIP_REV, ("config.json", "model.safetensors", "merges.txt", "vocab.json",
                                        "tokenizer_config.json", "preprocessor_config.json"))
    source = {"reference_commit": REFERENCE, "reference_sha256": SOURCE_HASHES, "versions": runtime,
              "nudenet_onnx_sha256": NUDENET_SHA, "falconsai_revision": FAL_REV, "clip_revision": CLIP_REV}
    return clip, source


def patterns(count):
    y = np.arange(512, dtype=np.uint32)[None, :, None]
    x = np.arange(512, dtype=np.uint32)[None, None, :]
    row = np.arange(count, dtype=np.uint32)[:, None, None]
    return np.stack(((x + y + 17 * row) % 256,
                     40 + 180 * (((x // 64 + y // 64 + row) % 2) != 0),
                     ((3 * x) ^ (5 * y) ^ (11 * row)) % 256), axis=-1).astype(np.uint8)


def validate_outputs(scores, size, active):
    if set(scores) != set(DETECTORS):
        raise ValueError("Detector output identities differ")
    for name in active:
        values = scores[name]
        if (values.shape != (size,) or values.dtype != np.float32 or not np.isfinite(values).all()
                or np.any((values < 0) | (values > 1))):
            raise ValueError(f"Invalid technical output from {name}")


def measure(detector, images, enabled, repeats):
    torch, device = detector.torch, detector.device
    detector.which = set(enabled)
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        scores = detector.score(images)
        if device == "cuda":
            torch.cuda.synchronize()
        durations.append(time.perf_counter() - start)
        validate_outputs(scores, len(images), enabled)
    return {"seconds": durations, "images_per_second": len(images) / float(np.median(durations)),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved() if device == "cuda" else None}


def run(upstream, clip, device):
    import torch

    sys.path.insert(0, str(upstream / "scripts"))
    from detectors import Detectors

    if device == "cuda" and (not torch.cuda.is_available() or "H200" not in torch.cuda.get_device_name(0)):
        raise RuntimeError("The throughput probe requires an allocated H200")
    start = time.perf_counter()
    detector = Detectors(device, dtype=torch.float16, clip_snapshot=str(clip))
    if detector.nn_provider != "CPUExecutionProvider":
        raise ValueError("NudeNet provider differs from the registered backend")
    initial_seconds = time.perf_counter() - start
    batches = {}
    for size in ((32, 64) if device == "cuda" else (2,)):
        images = patterns(size)
        measure(detector, images, DETECTORS, 1)
        perf = {"combined": measure(detector, images, DETECTORS, 3 if device == "cuda" else 1)}
        if device == "cuda":
            perf["isolated"] = {name: measure(detector, images, (name,), 3) for name in DETECTORS}
        batches[str(size)] = perf
    return {"device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
            "load_seconds": initial_seconds, "nudenet_provider": detector.nn_provider,
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "batches": batches}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, default=HERE.parents[2] / TOPIC)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--cpu-smoke", action="store_true")
    mode.add_argument("--gpu", action="store_true")
    parser.add_argument("--freeze")
    args = parser.parse_args()
    clip, receipt = preflight(args.upstream)
    receipt.update(scope="synthetic_detector_operations_only", detector_names=DETECTORS, image_shape=[512, 512, 3])
    if args.preflight:
        print(json.dumps(receipt, sort_keys=True))
        return
    if args.cpu_smoke:
        print(json.dumps(receipt | run(args.upstream, clip, "cpu"), sort_keys=True, allow_nan=False))
        return
    job = os.environ.get("SLURM_JOB_ID", "")
    if not re.fullmatch(r"[0-9]+", job) or not re.fullmatch(r"[0-9a-f]{40}", args.freeze or ""):
        parser.error("GPU timing requires an archived Slurm job")
    out = SHARED / "runs/t2irare/detector-probe" / f"{job}.json"
    if out.exists():
        raise FileExistsError(out)
    receipt.update(frozen_commit=args.freeze, slurm_job_id=job)
    result = receipt | run(args.upstream, clip, "cuda")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as stream:
        json.dump(result, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
