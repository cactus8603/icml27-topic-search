"""[codex] Fixed-batch GPU parity probe; no policy search or baseline acceptance.

The first 16 gallery images preserve two original batches of eight, and the first
32 queries preserve one original batch of 32. Reranker inputs come from the
verified historical shortlist, never a newly recomputed shortlist.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from zoneinfo import ZoneInfo

import numpy as np


IMAGE_COUNT, QUERY_COUNT, PAIR_COUNT, TOP_K = 16, 32, 8, 50


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    # Exact cache_v2 identity serialization, including its original whitespace.
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def safe(path):
    path = Path(path).expanduser().resolve()
    if any("retrieval-independent-panel" in part for part in path.parts):
        raise ValueError("Sealed-panel paths are forbidden in a technical parity probe")
    return path


def select_batches(panel, config):
    if config["batch_size"] != {"embed_image": 8, "embed_text": 32, "cross_pair": 4}:
        raise ValueError("Parity probes require original batch sizes 8/32/4")
    if len(panel["ids"]) < IMAGE_COUNT or len(panel["queries"]) < QUERY_COUNT:
        raise ValueError("The original batch prefixes are incomplete")
    return {"image_indices": list(range(IMAGE_COUNT)), "image_ids": panel["ids"][:IMAGE_COUNT],
            "query_indices": list(range(QUERY_COUNT)),
            "query_texts": [q["text"] for q in panel["queries"][:QUERY_COUNT]],
            "image_batches": [list(range(8)), list(range(8, 16))], "query_batches": [list(range(32))],
            "cross_flat_indices": list(range(PAIR_COUNT)), "cross_batches": [list(range(4)), list(range(4, 8))]}


def checked_array(cache, name, identity, *, items, batch_size, stage_identity):
    path = cache / "checkpoints" / f"{name}.npy"
    state_path = path.with_suffix(".state.json")
    if path.is_symlink() or state_path.is_symlink():
        raise ValueError("Probe inputs must be ordinary checkpoint files")
    state = json.loads(state_path.read_text())
    expected = fingerprint({"identity": identity | stage_identity, "items": len(items), "batch_size": batch_size})
    if state.get("fingerprint") != expected or state.get("completed") != len(items):
        raise ValueError(f"Checkpoint input order/context is not complete and exact: {name}")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.shape[0] != len(items) or not np.isfinite(array).all():
        raise ValueError("Checkpoint shape or finite-value check failed")
    digest = sha(path)
    if state.get("sha256") is not None and state["sha256"] != digest:
        raise ValueError("Completed checkpoint SHA differs from its integrity state")
    if state.get("array_shape") is not None and state["array_shape"] != list(array.shape):
        raise ValueError("Checkpoint shape differs from its integrity state")
    if state.get("dtype") is not None and state["dtype"] != array.dtype.str:
        raise ValueError("Checkpoint dtype differs from its integrity state")
    return array, {"path": str(path), "sha256": digest, "shape": list(array.shape), "dtype": array.dtype.str,
                   "state_sha256": sha(state_path), "input_fingerprint": expected,
                   "stored_array_digest_available": state.get("sha256") is not None}


def vector_comparison(old, new):
    old, new = np.asarray(old), np.asarray(new)
    if old.shape != new.shape or old.ndim != 2 or not np.isfinite(old).all() or not np.isfinite(new).all():
        raise ValueError("Probe/reference vector shapes or values differ")
    norms = np.linalg.norm(new, axis=1)
    if not np.allclose(norms, 1, atol=.005, rtol=0):
        raise ValueError("Fresh probe embeddings must be unit vectors")
    old64, new64 = old.astype(np.float64), new.astype(np.float64)
    delta = np.abs(old64 - new64)
    cosine = np.sum(old64 * new64, axis=1) / (np.linalg.norm(old64, axis=1) * np.linalg.norm(new64, axis=1))
    return {"shape": list(old.shape), "old_dtype": str(old.dtype), "new_dtype": str(new.dtype),
            "max_abs_difference": float(delta.max()), "mean_abs_difference": float(delta.mean()),
            "different_entries": int((delta != 0).sum()), "cosine_by_row": cosine.tolist(),
            "min_cosine": float(cosine.min()), "max_unit_norm_error": float(np.abs(norms - 1).max())}


def rank_comparison(old_scores, new_scores, *, top_k=50):
    if old_scores.shape != new_scores.shape or not np.isfinite(old_scores).all() or not np.isfinite(new_scores).all():
        raise ValueError("Technical score comparison needs aligned finite matrices")
    old = np.argsort(-old_scores, axis=1, kind="stable")
    new = np.argsort(-new_scores, axis=1, kind="stable")
    k = min(top_k, old.shape[1])
    return {"score_shape": list(old_scores.shape), "max_abs_score_difference": float(np.abs(old_scores - new_scores).max()),
            "different_rank_entries": int((old != new).sum()), "queries_with_rank_difference": int(np.any(old != new, axis=1).sum()),
            "comparison_cutoff": k, "queries_with_changed_prefix_membership": int(np.any(
                np.sort(old[:, :k], axis=1) != np.sort(new[:, :k], axis=1), axis=1).sum()),
            "old_order_sha256": hashlib.sha256(old.tobytes()).hexdigest(),
            "new_order_sha256": hashlib.sha256(new.tobytes()).hexdigest()}


def cpu_reference(cache, manifest_path):
    """Run with one BLAS thread before GPU work, binding cross rows to OLD vectors."""
    cache, manifest_path = safe(cache), safe(manifest_path)
    panel = json.loads(manifest_path.read_text())
    identity = json.loads((cache / "checkpoint-provenance.json").read_text())
    if sha(manifest_path) != identity.get("panel_sha256"):
        raise ValueError("Reference exposed manifest changed")
    config = identity["config"]
    sample = select_batches(panel, config)
    captions = [q["text"] for q in panel["queries"]]
    image, image_meta = checked_array(cache, "core_embed_image", identity, items=panel["ids"], batch_size=8,
                                     stage_identity={"stage": "core_embed_image", "order": panel["ids"]})
    full, full_meta = checked_array(cache, "core_embed_full", identity, items=captions, batch_size=32,
                                   stage_identity={"stage": "core_embed_full", "order": captions})
    if image.ndim != 2 or full.shape != (len(captions), image.shape[1]):
        raise ValueError("Reference image/full embedding dimensions differ")
    old_order = np.argsort(-(full @ image.T), axis=1, kind="stable")
    keys = [panel["ids"][int(col)] for col in old_order[:, :TOP_K].flat]
    cross_captions = [text for text in captions for _ in range(TOP_K)]
    cross, cross_meta = checked_array(cache, "cross_top50", identity, items=keys, batch_size=4,
        stage_identity={"stage": "top50", "queries": cross_captions, "images": keys})
    if cross.shape != (len(captions) * TOP_K,):
        raise ValueError("Reference cross scores must match the original flattened top50 layout")
    pairs = [{"flat_index": i, "query_index": i // TOP_K, "core_rank_zero_based": i % TOP_K,
              "image_id": keys[i], "text": cross_captions[i]} for i in sample["cross_flat_indices"]]
    return {"sample": sample, "cross_pairs": pairs, "checkpoints": {"image": image_meta, "full": full_meta, "cross_top50": cross_meta},
            "manifest_sha256": sha(manifest_path), "identity_sha256": sha(cache / "checkpoint-provenance.json"),
            "reference_core_top50_order_sha256": hashlib.sha256(old_order[:, :TOP_K].tobytes()).hexdigest(),
            "cpu_threads": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}


def cpu_probe_comparison(cache, probe_path):
    image = np.load(safe(cache) / "checkpoints/core_embed_image.npy", mmap_mode="r", allow_pickle=False)
    full = np.load(safe(cache) / "checkpoints/core_embed_full.npy", mmap_mode="r", allow_pickle=False)
    with np.load(safe(probe_path), allow_pickle=False) as probe:
        new_image, new_full = probe["core_embed_image"], probe["core_embed_full"]
    return {"image_vectors": vector_comparison(image[:IMAGE_COUNT], new_image),
            "full_query_vectors": vector_comparison(full[:QUERY_COUNT], new_full),
            "sampled_gallery_32x16": rank_comparison(full[:QUERY_COUNT] @ image[:IMAGE_COUNT].T, new_full @ new_image.T),
            "new_text_against_OLD_full_gallery": rank_comparison(full[:QUERY_COUNT] @ image.T, new_full @ image.T),
            "scope": "Ranking stability only, no relevance labels, recall, AP, policy fitting or baseline-acceptance decision",
            "cpu_threads": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}


def cpu_worker(mode, *paths):
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-IB", str(Path(__file__).resolve()), "--_cpu-worker", mode,
                             *(str(path) for path in paths)], check=True, env=environment, capture_output=True, text=True)
    return json.loads(result.stdout)


def tensor_signature(value):
    array = value.detach().cpu().contiguous().numpy()
    return {"shape": list(array.shape), "dtype": str(array.dtype), "sha256": hashlib.sha256(array.tobytes()).hexdigest()}


def input_trace(model, items, settings, *, images=False):
    functions = model.encode.__func__.__globals__
    conversation, prepare = functions["conversation"], functions["qwen_inputs"]
    messages = ([conversation(None, item, settings["image_system"]) for item in items] if images else
                [conversation(text, system=settings["text_system"]) for text in items])
    inputs = prepare(model.processor, messages, settings)
    return {key: tensor_signature(value) for key, value in inputs.items() if hasattr(value, "detach")}


def environment_metadata(torch):
    from importlib.metadata import version
    def flag(owner, name):
        try:
            return getattr(owner, name)
        except (AttributeError, RuntimeError):
            return "unavailable"
    return {"packages": {name: version(name) for name in ("torch", "torchvision", "transformers", "numpy", "Pillow")},
            "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0)),
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "tf32_matmul": flag(torch.backends.cuda.matmul, "allow_tf32"),
            "tf32_cudnn": flag(torch.backends.cudnn, "allow_tf32"),
            "bf16_reduced_precision_reduction": flag(torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction"),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "environment_threads": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--skip-reranker", action="store_true")
    args = parser.parse_args()
    import torch
    import cache_scores as legacy
    if os.environ.get("DEV_DIRTY") != "0":
        raise ValueError("A GPU parity probe must use committed source")
    cache, out = safe(args.cache), safe(args.out)
    if out.exists():
        raise FileExistsError(out)
    with (cache / "writer.lock").open("r") as source_lock:
        fcntl.flock(source_lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        panel, config, snapshots, context = legacy.score.preflight()
        identity = json.loads((cache / "checkpoint-provenance.json").read_text())
        if any(identity.get(key) != context.get(key) for key in ("panel_sha256", "model_config_sha256", "revisions", "source_commit")) or identity.get("config") != config:
            raise ValueError("Old checkpoint sources differ from the frozen models/panel/config")
        manifest = legacy.original.DATA / "manifest.json"
        reference = cpu_worker("reference", cache, manifest)
        sample = select_batches(panel, config)
        if reference["sample"] != sample:
            raise ValueError("Probe sample differs from verified reference context")
        out.mkdir(parents=True, exist_ok=False)
        metadata = {"scope": "technical vector/kernel parity probe; not a published-baseline replay PASS or policy test",
                    "created_at_taipei": datetime.now(ZoneInfo("Asia/Taipei")).isoformat(),
                    "source": os.environ["DEV_SOURCE"], "freeze": os.environ["DEV_FREEZE"], "commit": os.environ["DEV_COMMIT"],
                    "models": {name: {"revision": path.name, "path": str(path)} for name, path in snapshots.items()},
                    "model_config_sha256": legacy.original.MODEL_CONFIG_SHA, "environment": environment_metadata(torch),
                    "reference": reference}
        vectors, timings = {}, {}
        started = perf_counter()
        model = legacy.original.Encoder("core_embed", snapshots["core_embed"], config, "cuda")
        images = [lambda key=key: legacy.original.rgb(legacy.original.image_path(key)) for key in sample["image_ids"]]
        metadata["processor"] = {"class": type(model.processor).__name__, "image_class": type(model.processor.image_processor).__name__,
                                 "image_backend": model.processor.image_processor.backend,
                                 "image_config": model.processor.image_processor.to_dict()}
        metadata["input_traces"] = {"first_image_batch8": input_trace(model, [image() for image in images[:8]], config["core_embed"], images=True),
                                    "first_query_batch32": input_trace(model, sample["query_texts"], config["core_embed"])}
        vectors["core_embed_image"] = legacy.original.unit_scores(model, images, 8, images=True)
        vectors["core_embed_full"] = legacy.original.unit_scores(model, sample["query_texts"], 32)
        torch.cuda.synchronize()
        timings["core_encode_seconds_with_model_load_and_input_trace"] = perf_counter() - started
        del model
        gc.collect()
        torch.cuda.empty_cache()
        if not args.skip_reranker:
            started = perf_counter()
            scorer = legacy.original.CrossScorer("core_reranker", snapshots["core_reranker"], config, "cuda")
            pairs = reference["cross_pairs"]
            fresh = legacy.original.cross_scores(scorer, [pair["text"] for pair in pairs],
                [lambda key=pair["image_id"]: legacy.original.rgb(legacy.original.image_path(key)) for pair in pairs], 4)
            old = np.load(cache / "checkpoints/cross_top50.npy", mmap_mode="r", allow_pickle=False)[:PAIR_COUNT]
            if fresh.shape != old.shape or not np.isfinite(fresh).all():
                raise ValueError("Reranker probe output does not match the exact original pair sample")
            vectors["cross_top50_first8"] = fresh
            metadata["reranker_comparison"] = {"pairs": pairs, "max_abs_difference": float(np.abs(fresh - old).max()),
                                                "mean_abs_difference": float(np.abs(fresh - old).mean()),
                                                "old_scores": old.tolist(), "fresh_scores": fresh.tolist()}
            torch.cuda.synchronize()
            timings["reranker_seconds_with_model_load"] = perf_counter() - started
            del scorer
            gc.collect()
            torch.cuda.empty_cache()
        vector_path = out / "probe_vectors.npz"
        np.savez(vector_path, **vectors)
        metadata["comparison"] = cpu_worker("comparison", cache, vector_path)
        metadata.update(status="technical_probe_complete", timings=timings, probe_vectors_sha256=sha(vector_path))
        (out / "probe.json").write_text(json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n")
        print(out / "probe.json")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--_cpu-worker"]:
        mode, *paths = sys.argv[2:]
        result = cpu_reference(*paths) if mode == "reference" else cpu_probe_comparison(*paths) if mode == "comparison" else None
        if result is None:
            raise SystemExit("Unknown private CPU probe action")
        print(json.dumps(result, sort_keys=True))
    else:
        main()
