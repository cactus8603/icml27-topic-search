"""[codex] Original Karpathy caption retrieval, with inference/label separation."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from .evidence import checked, digest, read, sha


def rows(path):
    import json
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _unique(items, key):
    output = {}
    for item in items:
        name = str(item[key])
        if name in output:
            raise ValueError("Duplicate original source identifier: " + name)
        output[name] = item
    return output


def load(spec, *, verify_images=True):
    """Keep every original sentence, including COCO's ten sixth captions.

    Source JSON is independently checked against the prepared manifest. Only
    label-free paths and text strings reach a model; source IDs and metadata
    stay available to the CPU scorer and evidence receipts.
    """
    if spec.get("format") not in {"karpathy-coco", "karpathy-flickr"}:
        raise ValueError("Unsupported native retrieval source format")
    source_path = checked(spec["source_json"])
    manifest_path = checked(spec["manifest"])
    image_receipt_path = checked(spec["image_receipt"])
    source = [row for row in read(source_path)["images"] if row["split"] == "test"]
    manifest = rows(manifest_path)
    receipts = rows(image_receipt_path)
    coco = spec["format"] == "karpathy-coco"
    sid = lambda row: str(row["cocoid"]) if coco else Path(row["filename"]).stem
    source_by_id = _unique([dict(row, native_id=sid(row)) for row in source], "native_id")
    manifest_by_id = _unique(manifest, "id")
    if set(source_by_id) != set(manifest_by_id):
        raise ValueError("Prepared manifest is not the complete original test split")
    receipt_by_path = {}
    for row in receipts:
        path = Path(row["path"])
        if not path.is_absolute():
            path = Path(spec["receipt_root"]) / path
        name = str(path.resolve())
        if name in receipt_by_path:
            raise ValueError("Duplicate image receipt path")
        receipt_by_path[name] = row
    gallery, queries, labels = [], [], []
    sentence_ids = set()
    image_paths = set()
    # Original Karpathy source order is frozen explicitly. TFDS can iterate a
    # different order; that difference only affects exact similarity ties.
    for original in source:
        image_id = sid(original)
        prepared = manifest_by_id[image_id]
        texts = [row["raw"] for row in original["sentences"]]
        ids = [row["sentid"] for row in original["sentences"]]
        if prepared["captions"] != texts or ids != original["sentids"]:
            raise ValueError("Prepared raw captions or sentence IDs differ from original JSON")
        if coco and (prepared["filename"] != original["filename"] or prepared["sentids"] != ids):
            raise ValueError("Prepared COCO image/sentence identity differs")
        path = Path(prepared["path"]) if coco else Path(spec["image_root"])/prepared["image"]
        if path.name != original["filename"]:
            raise ValueError("Prepared image filename differs from the original source")
        path = path.resolve()
        if str(path) in image_paths:
            raise ValueError("Original image filename/path is ambiguous")
        image_paths.add(str(path))
        receipt = receipt_by_path.get(str(path))
        if receipt is None or not path.is_file() or path.stat().st_size != receipt["bytes"]:
            raise ValueError("An original image is missing or has an invalid byte receipt")
        if coco and str(receipt["id"]) != image_id:
            raise ValueError("COCO image receipt points to a different source image")
        if verify_images and sha(path) != receipt["sha256"]:
            raise ValueError("Original image SHA changed: " + str(path))
        gallery.append({"id": image_id, "path": str(path), "source_metadata": original,
                        "image_sha256": receipt["sha256"], "image_bytes": receipt["bytes"]})
        for text, sentence_id in zip(texts, ids):
            if not isinstance(text, str) or sentence_id in sentence_ids:
                raise ValueError("Original captions need text and globally unique sentence IDs")
            sentence_ids.add(sentence_id)
            query_id = str(sentence_id)
            queries.append({"id": query_id, "text": text})
            labels.append({"id": query_id, "positive_image_id": image_id})
    counts = {"images": len(gallery), "texts": len(queries), "excluded_images": 0,
              "excluded_texts": 0, "caption_count_histogram": {str(key): value for key, value in sorted(Counter(
                  len(row["source_metadata"]["sentences"]) for row in gallery).items())}}
    if counts["images"] != spec["expected_images"] or counts["texts"] != spec["expected_texts"]:
        raise ValueError("Native qualification denominator differs from the entire frozen split")
    identity = {"benchmark_id": spec["benchmark_id"], "spec_sha256": digest(spec),
        "gallery_ids": [row["id"] for row in gallery], "query_ids": [row["id"] for row in queries],
        "labels_sha256": digest(labels), "counts": counts}
    return {"gallery": gallery, "queries": queries, "labels": labels,
            "counts": counts, "identity": identity}


def inference_items(bundle):
    """No positive mappings, source captions, or labels enter inference."""
    return [row["path"] for row in bundle["gallery"]], [row["text"] for row in bundle["queries"]]


def positive_indices(bundle):
    lookup = {row["id"]: index for index, row in enumerate(bundle["gallery"])}
    return [lookup[row["positive_image_id"]] for row in bundle["labels"]]
