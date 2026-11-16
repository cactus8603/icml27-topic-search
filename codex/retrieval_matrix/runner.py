"""[codex] Resumable official baseline matrix, with separate CPU scoring.

Stages: smoke (synthetic only), index (images/texts), cross (fixed top-K),
analyze (CPU only). No method development, fitting, label-based admission, or
automatic resource fallback. Every completed batch reuses the established
retrieval_scaling checkpoint implementation and its integrity receipts.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
from datetime import datetime
import fcntl
import gc
import csv
import importlib.util
from importlib.metadata import PackageNotFoundError, version
import importlib
import json
import os
from pathlib import Path
import re
import subprocess
from time import monotonic
from zoneinfo import ZoneInfo

import numpy as np

try:
    from . import metrics, models
except ImportError:
    import metrics
    import models

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
_checkpoint_spec = importlib.util.spec_from_file_location("matrix_scaling_checkpoint", ROOT/"codex/retrieval_scaling/runner.py")
C = importlib.util.module_from_spec(_checkpoint_spec)
_checkpoint_spec.loader.exec_module(C)
sha, read, atomic_json, fixed_json, digest, checkpoint = C.sha, C.read_json, C.atomic_json, C.fixed_json, C.digest, C.checkpoint
SOURCE_FILES = ("codex/retrieval_matrix/runner.py", "codex/retrieval_matrix/models.py",
                "codex/retrieval_matrix/metrics.py", "codex/retrieval_scaling/runner.py")


def now():
    return datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y-%m-%d %H:%M")


def safe(value):
    path = Path(value).expanduser().absolute()
    if any("retrieval-independent-panel" in str(p) for p in (path, path.resolve())):
        raise ValueError("Sealed independent-panel paths are forbidden")
    return path.resolve()


def checked_file(entry):
    if not isinstance(entry, dict) or not re.fullmatch(r"[0-9a-f]{64}", entry.get("sha256", "")):
        raise ValueError("Every dataset artifact requires a frozen path and SHA256")
    path = safe(entry["path"])
    if sha(path) != entry["sha256"]:
        raise ValueError(f"Dataset artifact SHA differs: {path}")
    return path


def json_rows(path):
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    result = read(path)
    if not isinstance(result, list):
        raise ValueError("Expected a JSON list or JSONL rows")
    return result


def load_plan(path):
    plan = read(safe(path))
    if plan.get("schema") != 1 or plan.get("state") != "FROZEN":
        raise ValueError("The matrix plan must be explicitly FROZEN with schema=1")
    settings = plan["settings"]
    if settings.get("top_ks") != [50, 100, 200] or settings.get("main_k") != 100:
        raise ValueError("The fixed matrix uses main K100, supplementary K50/K200")
    if type(settings.get("block_queries", 64)) is not int or settings.get("block_queries", 64) <= 0:
        raise ValueError("CPU query block size must be a positive integer")
    if not plan.get("datasets") or not plan.get("models"):
        raise ValueError("The plan must declare model and dataset identities")
    for spec in plan["models"].values():
        models.validate_spec(spec)
    return plan


def source_gate(plan_path, plan, freeze_commit):
    if not re.fullmatch(r"[0-9a-f]{40}", freeze_commit or ""):
        raise ValueError("Producing model outputs requires an explicit full --freeze-commit")
    declared = plan.get("source_files", {})
    required = set(SOURCE_FILES) | {str(safe(plan_path).relative_to(ROOT)), plan["prereg"]}
    if not set(SOURCE_FILES) <= declared.keys():
        raise ValueError("The frozen plan must bind every executing runner/checkpoint source SHA")
    for relative in sorted(required | declared.keys()):
        path = safe(ROOT/relative)
        if not path.is_relative_to(ROOT):
            raise ValueError("Frozen source path escapes the repository")
        if C.git_bytes(freeze_commit, relative) != path.read_bytes():
            raise ValueError(f"Executing source differs from its frozen commit: {relative}")
        if relative in declared and sha(path) != declared[relative]:
            raise ValueError(f"Plan source SHA differs: {relative}")
    status = subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain", "--", "codex/retrieval_matrix"], text=True)
    if status.strip():
        raise ValueError("GPU production requires a clean codex/retrieval_matrix worktree")
    return {relative: sha(ROOT/relative) for relative in sorted(required | declared.keys())}


def load_data(spec, *, verify_images=False, allow_test=False):
    """Return inference-only queries and full gallery metadata, never labels."""
    split = spec.get("split", "public")
    if split == "test" and not allow_test:
        raise ValueError("Test data requires its separately frozen fixed-baseline authorization")
    if split not in {"dev", "dev-small", "test", "public"}:
        raise ValueError("Unknown dataset split")
    if spec["format"] == "negbench":
        panel = read(checked_file(spec["panel"]))
        gallery, raw_queries = panel["candidates"], panel["queries"]
        if spec.get("official_evaluation") in {"coco-bidirectional", "negbench-t2i"}:
            # Original CSV order is part of torch.topk tie behavior. Prepared
            # development panels sorted IDs and therefore cannot define it.
            with checked_file(spec["official_csv"]).open(newline="") as stream:
                source_rows = list(csv.DictReader(stream))
            by_image = {str(row["id"]): row for row in gallery}
            by_caption = {(str(row["source_image_id"]), row["caption_index"]): row for row in raw_queries}
            gallery, raw_queries = [], []
            for source in source_rows:
                key = str(int(source["image_id"]))
                gallery.append(by_image.pop(key))
                for i, text in enumerate(ast.literal_eval(source["captions"])):
                    query = by_caption.pop((key, i))
                    if query["text"] != text:
                        raise ValueError("Prepared caption differs from the original official CSV")
                    raw_queries.append(query)
            if by_image or by_caption:
                raise ValueError("Official CSV order does not cover the complete prepared input")
    elif spec["format"] == "flickr":
        manifest_path = checked_file(spec["manifest"])
        manifest = json_rows(manifest_path)
        receipt = json_rows(checked_file(spec["images_receipt"]))
        receipts = {Path(row["path"]).name: row for row in receipt}
        if len(receipts) != len(receipt):
            raise ValueError("Flickr receipt filenames must be unique")
        with checked_file(spec["official_csv"]).open(newline="") as stream:
            original = list(csv.DictReader(stream))
        if len(original) != len(manifest):
            raise ValueError("Flickr original CSV and verified manifest row counts differ")
        gallery, raw_queries = [], []
        image_dir = safe(spec.get("images_dir", manifest_path.parent))
        for row, source in zip(manifest, original, strict=True):
            repeated_ids = json.loads(source["imgid"])
            key = str(repeated_ids[0]) if isinstance(repeated_ids, list) else str(repeated_ids)
            captions = json.loads(source["raw"])
            if (str(row["id"]) != key or row["captions"] != captions
                    or Path(row["image"]).name != source["filename"]):
                raise ValueError("Flickr captions/images/order differ from the official CSV")
            image = safe(image_dir/row["image"])
            if not image.is_relative_to(image_dir):
                raise ValueError("Flickr image path escapes the declared image directory")
            entry = receipts[image.name]
            gallery.append({**row, "path": str(image), "sha256": entry["sha256"], "bytes": entry["bytes"]})
            raw_queries.extend({"id": f"flickr:{key}:{i}", "text": text, "group": key,
                                "positive_ids": [key]} for i, text in enumerate(captions))
    elif spec["format"] == "conjunction":
        gallery = json_rows(checked_file(spec["gallery"]))
        raw_queries = json_rows(checked_file(spec["queries"]))
        receipt = json_rows(checked_file(spec["images_receipt"]))
        receipts = {str(row["id"]): row for row in receipt}
        if len(receipts) != len(receipt):
            raise ValueError("Duplicate image receipt IDs")
        image_dir = safe(spec["images_dir"])
        # Preserve original COCO metadata while attaching pinned local bytes.
        gallery = [{**row, "path": str(safe(image_dir/row["file_name"])),
                    "sha256": receipts[str(row["id"])]["sha256"],
                    "bytes": receipts[str(row["id"])]["bytes"]} for row in gallery]
        if "subset" in spec:
            subset = json_rows(checked_file(spec["subset"]))
            lookup = {str(q["id"]): q for q in raw_queries}
            if len(set(subset)) != len(subset) or not set(subset) <= lookup.keys():
                raise ValueError("Frozen development subset is not a unique query-ID subset")
            raw_queries = [lookup[key] for key in subset]
    else:
        raise ValueError("Dataset format lacks a verified adapter")
    ids = [str(row["id"]) for row in gallery]
    if not gallery or len(set(ids)) != len(ids):
        raise ValueError("Gallery IDs must be nonempty and unique")
    query_ids = [str(row["id"]) for row in raw_queries]
    if not raw_queries or len(set(query_ids)) != len(query_ids):
        raise ValueError("Query IDs must be nonempty and unique")
    queries, excluded, retained_empty = [], [], []
    for row in raw_queries:
        text = row.get("text")
        if row.get("model_eligible", True) is not True or not isinstance(text, str) or not text.strip():
            if text == "" and row.get("model_eligible") is False and row.get("exclusion_reason") == "source-empty-caption":
                if spec.get("empty_caption_policy") == "retain-official":
                    retained_empty.append(str(row["id"]))
                else:
                    excluded.append({"id": str(row["id"]), "reason": "source-empty-caption"})
                    continue
            else:
                raise ValueError("Only documented original empty captions may be excluded")
        queries.append({"id": str(row["id"]), "text": text, "group": str(row.get("group", row["id"]))})
    for field, actual in (("expected_gallery", len(gallery)), ("expected_queries", len(raw_queries)),
                           ("expected_eligible", len(queries))):
        if field in spec and spec[field] != actual:
            raise ValueError(f"Frozen dataset count differs: {field} expected {spec[field]}, actual {actual}")
    for row in gallery:
        path = safe(row["path"])
        if not path.is_file() or not re.fullmatch(r"[0-9a-f]{64}", row.get("sha256", "")):
            raise ValueError(f"Missing gallery image or image SHA: {row['id']}")
        if verify_images and (sha(path) != row["sha256"] or path.stat().st_size != row["bytes"]):
            raise ValueError(f"Gallery image bytes changed: {row['id']}")
    return {"gallery": gallery, "queries": queries, "excluded": excluded, "retained_empty": retained_empty, "raw_queries": len(raw_queries),
            "fingerprint": digest({"gallery": gallery, "queries": queries, "dataset": spec})}


def load_labels(spec, bundle):
    """This function is called only by the CPU analysis stage."""
    if spec["format"] == "negbench":
        raw = read(checked_file(spec["panel"]))["queries"]
    elif spec["format"] == "flickr":
        raw = [{"id": f"flickr:{row['id']}:{i}", "text": text, "positive_ids": [str(row["id"])]}
               for row in json_rows(checked_file(spec["manifest"])) for i, text in enumerate(row["captions"])]
    else:
        raw = json_rows(checked_file(spec["queries"]))
    lookup = {str(row["id"]): row for row in raw}
    result = []
    for query in bundle["queries"]:
        row = lookup[query["id"]]
        if row["text"] != query["text"]:
            raise ValueError("Scoring and inference query texts differ")
        if spec["format"] == "negbench" and ([str(v) for v in row.get("positive_ids", [])] != [str(row["source_image_id"])] or row.get("ignore", [])):
            raise ValueError("Official NegBench labels must identify only their original source image")
        result.append({"id": query["id"], "group": query["group"],
                       "positives": row.get("positives", row.get("positive_ids", [])), "ignore": row.get("ignore", [])})
    return result


def runtime_environment(spec):
    import torch
    result = {}
    for name in ("torch", "transformers", "numpy", "qwen-vl-utils", "flash-attn", "ms-swift", "Pillow", "tokenizers", "accelerate"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    if not torch.cuda.is_available():
        raise ValueError("A queued CUDA GPU is required for model production")
    result.update(cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                  device=torch.cuda.get_device_name(0), capability=list(torch.cuda.get_device_capability(0)),
                  visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
    result["modules"] = {}
    for name in ("torch", "transformers", "numpy", "qwen_vl_utils", "tokenizers", "accelerate"):
        module = importlib.import_module(name)
        result["modules"][name] = {"version": getattr(module, "__version__", None), "path": str(Path(module.__file__).resolve())}
    for name, expected in spec.get("environment", {}).items():
        if result.get(name) != expected:
            raise ValueError(f"Frozen runtime differs: {name}: {result.get(name)!r} != {expected!r}")
    return result


def rank_blocks(image, text, *, block_queries=64):
    image, text = np.asarray(image), np.asarray(text)
    if (image.ndim != 2 or text.ndim != 2 or image.shape[1] != text.shape[1]
            or not np.isfinite(image).all() or not np.isfinite(text).all()
            or not np.allclose(np.linalg.norm(image, axis=1), 1, atol=1e-2)
            or not np.allclose(np.linalg.norm(text, axis=1), 1, atol=1e-2)):
        raise ValueError("Image/text arrays must be finite normalized embeddings with matching dimensions")
    return np.concatenate([metrics.stable_order(text[start:start+block_queries]@image.T)
                           for start in range(0, len(text), block_queries)])


def array_receipt(path):
    state = read(path.with_suffix(".state.json"))
    if state.get("completed") != state.get("array_shape", [None])[0] or state.get("sha256") != sha(path):
        raise ValueError(f"Incomplete or changed checkpoint: {path}")
    array = np.load(path, allow_pickle=False, mmap_mode="r")
    if list(array.shape) != state["array_shape"] or array.dtype.str != state["dtype"] or not np.isfinite(array).all():
        raise ValueError("Completed array metadata/values differ")
    return {"sha256": sha(path), "state_sha256": sha(path.with_suffix(".state.json")),
            "shape": list(array.shape), "dtype": array.dtype.str, "active_seconds": state["active_seconds"]}


def validate_index(out, identity, bundle, settings):
    meta = read(out/"index-meta.json")
    if meta.get("state") != "COMPLETE" or meta.get("identity") != identity:
        raise ValueError("Index receipt identity is incomplete or differs")
    values = {}
    for name in ("image", "text"):
        path = out/(name+".npy")
        if array_receipt(path) != meta["arrays"][name]:
            raise ValueError("Index checkpoint receipt changed")
        values[name] = np.load(path, mmap_mode="r", allow_pickle=False)
    if (len(values["image"]), len(values["text"])) != (len(bundle["gallery"]), len(bundle["queries"])):
        raise ValueError("Index array row counts differ from gallery/query identity")
    order = np.load(out/"order.npy", allow_pickle=False)
    if sha(out/"order.npy") != meta["order_sha256"] or not np.array_equal(order, rank_blocks(**values, block_queries=settings.get("block_queries", 64))):
        raise ValueError("Saved candidate admission differs from actual normalized embeddings")
    for name, receipt in meta.get("official_arrays", {}).items():
        if array_receipt(out/(name+".npy")) != receipt:
            raise ValueError("Official native-precision score/topk checkpoint differs")
    if sha(out/"processor.json") != meta["processor_sha256"]:
        raise ValueError("Actual processor receipt changed")
    return values, order, meta


def official_similarity(out, spec, identity, bundle, dataset_spec):
    """Preserve released native-dtype matmul and torch.topk, including ties."""
    import torch
    dtype = getattr(torch, spec.get("loader", {}).get("torch_dtype", "bfloat16"))
    image = torch.from_numpy(np.load(out/"image.npy")).to("cuda", dtype=dtype)
    text = torch.from_numpy(np.load(out/"text.npy")).to("cuda", dtype=dtype)
    block = dataset_spec.get("similarity_batch", 1000)
    items = list(range(len(text)))
    base = {**identity, "similarity_dtype": str(dtype), "similarity_batch": block,
            "score_semantics": "released normalized native-dtype embeddings and native-dtype GPU matmul; torch.topk on original CSV order"}
    checkpoint(out/"official-similarity.npy", items, block,
               lambda batch: (text[batch]@image.T).float().cpu().numpy(), {**base, "stage": "official-similarity"},
               progress=out/"progress.json")
    scores = np.load(out/"official-similarity.npy", mmap_mode="r")
    topk = 5 if dataset_spec["official_evaluation"] == "negbench-t2i" else min(100, len(image))
    checkpoint(out/"official-topk.npy", items, block,
               lambda batch: np.stack([torch.topk(torch.from_numpy(np.array(scores[i])).to("cuda", dtype=dtype), k=topk).indices.int().cpu().numpy() for i in batch]),
               {**base, "stage": "official-topk", "k": topk}, progress=out/"progress.json")
    names = ["official-similarity", "official-topk"]
    if dataset_spec["official_evaluation"] == "coco-bidirectional":
        k = min(100, len(text))
        def compute_i2t(batch):
            similarities = image[batch]@text.T
            values_indices = [torch.topk(row, k=k) for row in similarities]
            return np.stack([np.concatenate([indices.int().cpu().numpy().astype(np.float32), values.float().cpu().numpy()])
                             for values, indices in values_indices])
        checkpoint(out/"official-i2t.npy", list(range(len(image))), block, compute_i2t,
                   {**base, "stage": "official-i2t", "k": k}, progress=out/"progress.json")
        names.append("official-i2t")
    return {name: array_receipt(out/(name+".npy")) for name in names}


@contextmanager
def writer(out):
    out.mkdir(parents=True, exist_ok=True)
    with (out/"writer.lock").open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def invariance_passed(invariance, settings):
    lower = settings["batch_invariance_cosine_min"]
    upper = settings["batch_invariance_max_abs"]
    if not 0 < lower <= 1 or not 0 <= upper < 1:
        raise ValueError("Invalid frozen technical batch-invariance tolerances")
    return set(invariance) == {"image", "text"} and all(
        np.isfinite(row["cosine"]) and np.isfinite(row["max_abs_difference"])
        and row["cosine"] >= lower and row["max_abs_difference"] <= upper for row in invariance.values())


def smoke(out, spec, identity, settings):
    """Synthetic color images only; no dataset or metric is opened."""
    from PIL import Image
    if (out/"smoke.json").exists():
        require_smoke(out, identity)
        return
    model = models.OfficialModel(spec)
    if spec["kind"] != "official_embedding":
        raise ValueError("The initial synthetic smoke supports official embedding models")
    paths = []
    for i in range(spec["batches"]["image"]):
        path = out/f"synthetic-{i:03d}.png"
        Image.new("RGB", (224, 224), color=(i % 256, 128, 64)).save(path)
        paths.append(str(path))
    image = model.encode(paths, image=True)
    text = model.encode(["A synthetic color image."]*spec["batches"]["text"])
    if image.ndim != 2 or text.ndim != 2 or image.shape[1] != text.shape[1] or not np.isfinite(image).all() or not np.isfinite(text).all():
        raise ValueError("Synthetic official inference returned invalid vectors")
    import torch
    single_image = model.encode(paths[:1], image=True)
    single_text = model.encode(["A synthetic color image."])
    invariance = {}
    for name, one, fixed in (("image", single_image[0], image[0]), ("text", single_text[0], text[0])):
        invariance[name] = {"max_abs_difference": float(np.max(np.abs(one-fixed))),
                            "cosine": float(one@fixed/(np.linalg.norm(one)*np.linalg.norm(fixed)))}
    passed = invariance_passed(invariance, settings)
    fixed_json(out/"smoke.json", {"state": "PASS" if passed else "FAILED", "technical_only": True, "research_metrics": False,
        "identity": identity, "signature": model.signature(), "image_shape": list(image.shape), "text_shape": list(text.shape),
        "batch1_vs_fixed_batch": invariance,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved()})
    if not passed:
        raise ValueError("Synthetic batch1 versus frozen batch invariance exceeds preregistered tolerances; failed smoke receipt preserved")


def require_smoke(smoke_out, identity):
    if smoke_out is None:
        raise ValueError("Index production requires --smoke-out with a completed synthetic-only smoke receipt")
    receipt = read(safe(smoke_out)/"smoke.json")
    if receipt.get("state") != "PASS" or receipt.get("technical_only") is not True or receipt.get("research_metrics") is not False:
        raise ValueError("Synthetic smoke has not passed")
    if not invariance_passed(receipt.get("batch1_vs_fixed_batch", {}), identity["settings"]):
        raise ValueError("Synthetic smoke batch-invariance evidence fails the frozen numerical tolerances")
    old = receipt["identity"]
    for key in ("model", "model_spec", "plan_sha256", "source_files", "freeze_commit"):
        if old.get(key) != identity[key]:
            raise ValueError(f"Smoke identity differs from current production: {key}")
    left, right = dict(old["environment"]), dict(identity["environment"])
    left.pop("visible_devices", None)
    right.pop("visible_devices", None)
    if left != right:
        raise ValueError("Actual module/environment identity differs between smoke and index production")


def index(out, spec, identity, bundle, settings, dataset_spec):
    if (out/"index-meta.json").exists():
        validate_index(out, identity, bundle, settings)
        return
    began = monotonic()
    model = models.OfficialModel(spec)
    fixed_json(out/"processor.json", model.signature())
    for name, items, image in (("image", [str(safe(row["path"])) for row in bundle["gallery"]], True),
                               ("text", [row["text"] for row in bundle["queries"]], False)):
        checkpoint(out/(name+".npy"), items, spec["batches"][name], lambda batch: model.encode(batch, image=image),
                   {**identity, "stage": name, "processor_sha256": sha(out/"processor.json")}, progress=out/"progress.json")
    arrays = {name: array_receipt(out/(name+".npy")) for name in ("image", "text")}
    del model
    import torch
    gc.collect()
    torch.cuda.empty_cache()
    official_arrays = {}
    if dataset_spec.get("official_evaluation") in {"coco-bidirectional", "negbench-t2i"}:
        official_arrays = official_similarity(out, spec, identity, bundle, dataset_spec)
    order = rank_blocks(np.load(out/"image.npy", mmap_mode="r"), np.load(out/"text.npy", mmap_mode="r"),
                        block_queries=settings.get("block_queries", 64))
    C.save_array(out/"order.npy", order)
    atomic_json(out/"index-meta.json", {"state": "COMPLETE", "identity": identity, "arrays": arrays,
        "official_arrays": official_arrays,
        "secondary_ranking": "float32 CPU dot and stable gallery-order ties; supplementary diagnostic only; official native-dtype metrics use official_arrays",
        "order_sha256": sha(out/"order.npy"), "processor_sha256": sha(out/"processor.json"),
        "invocation_wall_seconds": monotonic()-began, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "timing_scope": "invocation wall includes model load/current invocation; cumulative checkpoint active_seconds only includes completed compute batches across resumes"})


def cross(out, index_out, spec, identity, bundle, settings):
    # Index provenance is independently verified before any pair is admitted.
    index_identity = read(index_out/"provenance.json")
    if index_identity["dataset_fingerprint"] != bundle["fingerprint"]:
        raise ValueError("Cross encoder must use the exact original embedding gallery/query input")
    _, order, meta = validate_index(index_out, index_identity, bundle, settings)
    k = max(settings["top_ks"])
    if order.shape[1] < k:
        raise ValueError("Gallery is smaller than the frozen supplementary K200")
    pairs = [[qi, int(ci)] for qi, prefix in enumerate(order[:, :k]) for ci in prefix]
    fixed_json(out/"pairs.json", pairs)
    cross_identity = {**identity, "index_meta_sha256": sha(index_out/"index-meta.json"),
                      "pairs_sha256": sha(out/"pairs.json"), "top_k": k,
                      "admission": "original embedding complete-gallery topK; no labels; no ignore filtering or refill"}
    if (out/"cross-meta.json").exists():
        validate_cross(out, index_out, cross_identity, bundle, settings)
        return
    model = models.OfficialModel(spec)
    fixed_json(out/"processor.json", model.signature())
    checkpoint(out/"cross.npy", pairs, spec["batches"]["cross"],
               lambda batch: model.score([(bundle["queries"][qi]["text"], bundle["gallery"][ci]["path"]) for qi, ci in batch]),
               {**cross_identity, "processor_sha256": sha(out/"processor.json")}, progress=out/"progress.json")
    atomic_json(out/"cross-meta.json", {"state": "COMPLETE", "identity": cross_identity,
        "array": array_receipt(out/"cross.npy"), "pairs": len(pairs), "pairs_per_query": k})


def validate_cross(out, index_out, identity, bundle, settings):
    meta = read(out/"cross-meta.json")
    if meta.get("state") != "COMPLETE" or meta.get("identity") != identity or meta["array"] != array_receipt(out/"cross.npy"):
        raise ValueError("Cross output source/index/checkpoint identity differs")
    index_identity = read(index_out/"provenance.json")
    _, order, _ = validate_index(index_out, index_identity, bundle, settings)
    k = max(settings["top_ks"])
    expected = [[qi, int(ci)] for qi, prefix in enumerate(order[:, :k]) for ci in prefix]
    if read(out/"pairs.json") != expected or identity["pairs_sha256"] != sha(out/"pairs.json"):
        raise ValueError("Cross pairs are not the original label-independent topK")
    values = np.load(out/"cross.npy", allow_pickle=False)
    if values.shape != (len(bundle["queries"])*k,):
        raise ValueError("Cross output is not the complete fixed per-query pair budget")
    return values.reshape(len(bundle["queries"]), k)


def official_metrics(out, dataset_spec, bundle, labels):
    """Evaluate the published native-precision arrays, using its exact library."""
    kind = dataset_spec.get("official_evaluation")
    if kind not in {"negbench-t2i", "coco-bidirectional"}:
        return None
    image_ids = [str(row["id"]) for row in bundle["gallery"]]
    topk = np.load(out/"official-topk.npy", allow_pickle=False)
    scores = np.load(out/"official-similarity.npy", allow_pickle=False, mmap_mode="r")
    if scores.shape != (len(labels), len(image_ids)) or topk.shape[0] != len(labels):
        raise ValueError("Official score arrays differ from original full denominator")
    if kind == "negbench-t2i":
        if topk.shape[1] != 5:
            raise ValueError("NegBench official recall requires exactly torch.topk(k=5)")
        values = [int(str(label["positives"][0]) in {image_ids[i] for i in row}) for label, row in zip(labels, topk, strict=True)]
        return {"queries": len(values), "R@5": float(np.mean(values)),
                "semantics": "released eval_coco_neg: native model dtype GPU dot, torch.topk k=5; all original captions retained"}
    import pytrec_eval
    t2i_qrels = {str(qi): {str(label["positives"][0]): 1} for qi, label in enumerate(labels)}
    t2i_predictions = {str(qi): {image_ids[ci]: float(scores[qi, ci]) for ci in row} for qi, row in enumerate(topk)}
    i2t_qrels = {image_id: {} for image_id in image_ids}
    for qi, label in enumerate(labels):
        i2t_qrels[str(label["positives"][0])][str(qi)] = 1
    native_i2t = np.load(out/"official-i2t.npy", allow_pickle=False)
    k = native_i2t.shape[1]//2
    if native_i2t.shape != (len(image_ids), min(100, len(labels))*2):
        raise ValueError("Official I2T top100 score/index array differs")
    i2t_predictions = {image_id: {str(int(qi)): float(value) for qi, value in zip(row[:k], row[k:], strict=True)}
                       for image_id, row in zip(image_ids, native_i2t, strict=True)}
    requested = {"map_cut", "ndcg_cut", "recall_1", "recall_3", "recall_5", "recall_10", "recall_50", "recall_100"}
    fields = ("map_cut_10", "ndcg_cut_5", "ndcg_cut_10", "recall_1", "recall_3", "recall_5", "recall_10")
    report = {"semantics": "released pytrec_eval over native-dtype GPU matmul and torch.topk top100; I2T recall is relevant-caption fraction", "library": getattr(pytrec_eval, "__version__", None)}
    for name, qrels, predictions in (("t2i", t2i_qrels, t2i_predictions), ("i2t", i2t_qrels, i2t_predictions)):
        result = pytrec_eval.RelevanceEvaluator(qrels, requested).evaluate(predictions)
        if len(result) != len(qrels):
            raise ValueError("Official evaluator changed the original query denominator")
        report[name] = {field: float(np.mean([row[field] for row in result.values()])) for field in fields}
        report[name]["queries"] = len(qrels)
    return report


def analyze(out, spec, bundle, settings, *, index_out=None):
    """No torch, model construction, or CUDA calls are made in CPU analysis."""
    index_out = index_out or out
    identity = read(index_out/"provenance.json")
    if identity["dataset_fingerprint"] != bundle["fingerprint"]:
        raise ValueError("CPU analysis dataset differs from the producing index")
    _, order, meta = validate_index(index_out, identity, bundle, settings)
    labels = load_labels(spec, bundle)
    ids = [row["id"] for row in bundle["gallery"]]
    report, detail = metrics.evaluate(order, ids, labels)
    report.update(schema=1, model=identity["model"], dataset=identity["dataset"],
        raw_queries=bundle["raw_queries"], excluded=bundle["excluded"], retained_empty=bundle["retained_empty"], gallery=len(ids),
        protocol="complete-gallery embedding ranking; ignore removed only after selecting original fixed K, without refill",
        label_semantics="single original source image" if spec["format"] in {"negbench", "flickr"} else "COCO annotation all-condition relevance with declared ignore IDs",
        provenance={"index_meta_sha256": sha(index_out/"index-meta.json"), "index": meta})
    report["official"] = official_metrics(index_out, spec, bundle, labels)
    if index_out != out:
        cross_meta = read(out/"cross-meta.json")
        if cross_meta["identity"]["index_meta_sha256"] != sha(index_out/"index-meta.json"):
            raise ValueError("Cross analysis refers to a different index")
        scores = validate_cross(out, index_out, cross_meta["identity"], bundle, settings)
        report["rerank"] = {}
        detail = {"embedding": detail, "rerank": {}}
        for k in settings["top_ks"]:
            value, rows = metrics.evaluate(metrics.rerank_prefix(order, scores, k), ids, labels)
            report["rerank"][str(k)] = value
            detail["rerank"][str(k)] = rows
        report["cross_provenance"] = cross_meta
    atomic_json(out/"results/report.json", report)
    atomic_json(out/"results/per-query.json", detail)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("smoke", "index", "cross", "analyze"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--index-out", type=Path)
    parser.add_argument("--smoke-out", type=Path)
    parser.add_argument("--freeze-commit")
    args = parser.parse_args(argv)
    plan, out = load_plan(args.plan), safe(args.out)
    spec, dataset_spec = plan["models"][args.model], plan["datasets"][args.dataset]
    if args.stage == "cross" and (spec["kind"] != "official_reranker" or not args.index_out):
        parser.error("cross requires an official reranker model and --index-out")
    if args.stage == "index" and spec["kind"] != "official_embedding":
        parser.error("index requires an official embedding model")
    with writer(out):
        try:
            # Analysis is gated by the same committed protocol and source bytes,
            # but it deliberately does not require a GPU/runtime import.
            sources = source_gate(args.plan, plan, args.freeze_commit)
            authorized_test = dataset_spec.get("split") == "test" and plan.get("fixed_baseline_test") is True
            identity = {"schema": 1, "owner": "[codex]", "model": args.model, "dataset": args.dataset,
                        "plan_sha256": sha(args.plan), "source_files": sources, "freeze_commit": args.freeze_commit,
                        "model_spec": spec, "settings": plan["settings"]}
            if args.stage == "smoke":
                identity["environment"] = runtime_environment(spec)
                fixed_json(out/"mode.json", {"technical_only": True})
                smoke(out, spec, identity, plan["settings"])
            else:
                bundle = load_data(dataset_spec, verify_images=args.stage in {"index", "cross"}, allow_test=authorized_test)
                identity["dataset_fingerprint"] = bundle["fingerprint"]
                if args.stage != "analyze":
                    identity["environment"] = runtime_environment(spec)
                    fixed_json(out/"mode.json", {"technical_only": False})
                    fixed_json(out/"provenance.json", identity)
                    if args.stage == "index":
                        require_smoke(args.smoke_out, identity)
                        index(out, spec, identity, bundle, plan["settings"], dataset_spec)
                    else:
                        cross(out, safe(args.index_out), spec, identity, bundle, plan["settings"])
                else:
                    existing = read((safe(args.index_out) if args.index_out else out)/"provenance.json")
                    for key in ("plan_sha256", "source_files", "freeze_commit", "dataset_fingerprint", "settings"):
                        if existing.get(key) != identity[key]:
                            raise ValueError(f"CPU analysis producing identity differs: {key}")
                    if args.index_out is None and existing.get("model") != args.model:
                        raise ValueError("CPU analysis model differs from index producer")
                    analyze(out, dataset_spec, bundle, plan["settings"], index_out=safe(args.index_out) if args.index_out else None)
            print(json.dumps({"owner": "[codex]", "stage": args.stage, "state": "COMPLETE", "time": now(), "out": str(out)}), flush=True)
        except BaseException as error:
            atomic_json(out/("failure-"+args.stage+".json"), {"owner": "[codex]", "time": now(),
                "state": "FAILED", "stage": args.stage, "error_type": type(error).__name__, "error": str(error),
                "automatic_fallback": False})
            raise


if __name__ == "__main__":
    main()
