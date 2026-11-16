"""[codex] CPU-only, immutable public-literature development panels.

No model or network client is imported. NegBench gold identifies the caption's
source image, not an exhaustive set of images satisfying every condition.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
import csv
import fcntl
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import re
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "dl/retrieval3090/data/literature-development"
SEED = 2026111602
DEV_GROUPS = 1000
SCOPE = "public-literature-development-only; not independent formal confirmation"
INVENTORY_SHA = "8d8cc61a9e94122c7c3be860f4492dc14bb6ce521c1afde72f69ac1fea1b4af5"
PLAIN = "COCO_val_retrieval.csv"
NEGATED = "COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true.csv"
MCQ = "COCO_val_mcq_llama3.1_rephrased.csv"
NEG_FILES = {
    PLAIN: (1994138, "dfe9217359b42f3f68bd93245f92a04701b8d3cde0aaf3fd1c26694d6de8d547"),
    NEGATED: (2912742, "08a8db9b96130ab27f91f5ef14f413981d70a96bc9b3557436a26256db115b7d"),
    MCQ: (1321191, "42bc853b64fb7d995e73e39212cbbe076ea3e06c8dca1ccc892910f3f91bceee"),
}
FACTOR_COMMIT = "8af1972ddfd03e8437d5f5c96d5a784e94b82d99"
FACTOR_URL = f"https://raw.githubusercontent.com/SultanMo/factored-vlm/{FACTOR_COMMIT}/data/factor_bench.jsonl"
FACTOR_SHA = "37f179a66913d2fbf55bf4f9a852273e894d53683dda0c7a8a7fff6c7b7d387d"
FORBIDDEN = ("retrieval-independent-panel",)
SHA = re.compile(r"[0-9a-f]{64}")


def safe(path):
    path = Path(path).expanduser().absolute()
    for candidate in (path, path.resolve()):
        if any(word in part for part in candidate.parts for word in FORBIDDEN):
            raise ValueError("Sealed independent data is forbidden")
    # Neither existing files nor their ancestors may redirect a preparation write.
    for candidate in (path, *path.parents):
        if candidate.is_symlink():
            raise ValueError(f"Symlink paths are forbidden: {candidate}")
    return path


def canonical(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def sha(path):
    digest = hashlib.sha256()
    with safe(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_order(*values):
    return hashlib.sha256(canonical([SEED, *values])).hexdigest()


def publish(path, data):
    """Resume equal outputs; never overwrite an existing different artifact."""
    path = safe(path)
    if path.exists():
        if not path.is_file() or path.read_bytes() != data:
            raise ValueError(f"Existing artifact differs; use a separate output: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".literature-", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        # link fails atomically if another writer has already published this name.
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise ValueError(f"Concurrent output differs: {path}")
    finally:
        Path(temporary).unlink(missing_ok=True)


@contextmanager
def writer(out, *, readonly=False):
    out = safe(out)
    if readonly:
        if not out.is_dir():
            raise FileNotFoundError("Prepared output directory does not exist")
    else:
        out.mkdir(parents=True, exist_ok=True)
    lock = safe(out / "writer.lock")
    with lock.open("r" if readonly else "a+") as stream:
        fcntl.flock(stream, (fcntl.LOCK_SH if readonly else fcntl.LOCK_EX) | fcntl.LOCK_NB)
        yield


def string_list(value, field, *, audit=False, preserve_blank=False):
    try:
        result = ast.literal_eval(value)
    except (ValueError, SyntaxError, TypeError, RecursionError) as error:
        raise ValueError(f"Invalid literal in {field}") from error
    if (not isinstance(result, list) or any(not isinstance(v, str) for v in result)
            or (not audit and (not result or (not preserve_blank and any(not v.strip() for v in result))))):
        raise ValueError(f"{field} must be a string list (captions must also be nonempty)")
    return result


def image_id(value):
    text = str(value)
    if not re.fullmatch(r"[0-9]{1,12}", text) or int(text) <= 0:
        raise ValueError("Invalid COCO image ID")
    return str(int(text))


def parse_retrieval(path):
    rows = {}
    with safe(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"filepath", "image_id", "captions", "positive_objects", "negative_objects"}
        if not required <= set(reader.fieldnames or ()):
            raise ValueError("NegBench retrieval CSV is missing columns")
        for row in reader:
            key = image_id(row["image_id"])
            filename = f"{int(key):012d}.jpg"
            raw_path = row["filepath"]
            if Path(raw_path).name != filename or "val2017" not in Path(raw_path).parts or ".." in Path(raw_path).parts:
                raise ValueError("CSV source path/image ID mismatch")
            if key in rows:
                raise ValueError("Duplicate NegBench source image")
            rows[key] = {"captions": string_list(row["captions"], "captions", preserve_blank=True),
                         "positive_objects": string_list(row["positive_objects"], "positive_objects", audit=True),
                         "negative_objects": string_list(row["negative_objects"], "negative_objects", audit=True),
                         "original_filepath": raw_path}
    return rows


def check_pair(plain, negated):
    if set(plain) != set(negated):
        raise ValueError("Plain/negated source image IDs differ")
    for key, row in plain.items():
        other = negated[key]
        if len(row["captions"]) != len(other["captions"]):
            raise ValueError("Paired caption positions do not align")
        for field in ("positive_objects", "negative_objects", "original_filepath"):
            if row[field] != other[field]:
                raise ValueError(f"Paired source audit metadata differs: {field}")


def load_sources(raw, inventory_path):
    raw, inventory_path = safe(raw), safe(inventory_path)
    receipt_path = safe(raw / "source-receipt.json")
    receipt = json.loads(receipt_path.read_text())
    recorded = receipt.get("files", [])
    if not isinstance(recorded, list) or len(recorded) != len(NEG_FILES):
        raise ValueError("NegBench receipt must name exactly the three pinned files")
    entries = {row.get("file"): row for row in recorded}
    if set(entries) != set(NEG_FILES):
        raise ValueError("NegBench receipt file set differs")
    for name, (size, digest) in NEG_FILES.items():
        path = safe(raw / name)
        entry = entries[name]
        if entry.get("bytes") != size or entry.get("sha256") != digest or path.stat().st_size != size or sha(path) != digest:
            raise ValueError(f"Pinned NegBench bytes/receipt differ: {name}")
    if sha(inventory_path) != INVENTORY_SHA:
        raise ValueError("COCO image inventory differs from the fixed public inventory")
    inventory = json.loads(inventory_path.read_text())["images"]
    if len(inventory) != 5000 or any(image_id(k) != k or not SHA.fullmatch(v.get("sha256", ""))
                                      or not all(isinstance(v.get(x), int) and v[x] > 0 for x in ("width", "height"))
                                      for k, v in inventory.items()):
        raise ValueError("Malformed fixed image inventory")
    plain, negated = parse_retrieval(raw / PLAIN), parse_retrieval(raw / NEGATED)
    check_pair(plain, negated)
    if (len(plain), sum(len(row["captions"]) for row in plain.values())) != (5000, 25014):
        raise ValueError("Pinned NegBench row/caption counts differ")
    if set(plain) != set(inventory):
        raise ValueError("NegBench sources differ from the full COCO val gallery")
    # Verify the third file as a source artifact, without conflating MCQ with retrieval.
    with (raw / MCQ).open(newline="", encoding="utf-8") as stream:
        if sum(1 for _ in csv.DictReader(stream)) != 5914:
            raise ValueError("Pinned NegBench MCQ row count differs")
    sources = {name: {"bytes": size, "sha256": digest} for name, (size, digest) in NEG_FILES.items()}
    sources.update({"source-receipt.json": {"sha256": sha(receipt_path)},
                    "coco-val2017-expected.json": {"sha256": INVENTORY_SHA}})
    return plain, negated, inventory, sources


def inspect_image(data, expected):
    from PIL import Image
    if hashlib.sha256(data).hexdigest() != expected["sha256"]:
        raise ValueError("Image differs from the authoritative COCO JPEG SHA")
    with Image.open(BytesIO(data)) as image:
        if image.format != "JPEG" or image.size != (expected["width"], expected["height"]):
            raise ValueError("COCO image dimensions/format differ")
        image.load()


def candidates(inventory, images, val_zip=None):
    images = safe(images)
    archive = zipfile.ZipFile(safe(val_zip)) if val_zip else None
    try:
        members = archive.namelist() if archive else []
        if len(members) != len(set(members)):
            raise ValueError("Duplicate ZIP members")
        result = []
        for key in sorted(inventory, key=int):
            expected = inventory[key]
            filename = f"{int(key):012d}.jpg"
            path = safe(images / filename)
            if not path.exists():
                if not archive:
                    raise FileNotFoundError(f"Missing COCO image; acquire official bytes first: {path}")
                member = f"val2017/{filename}"
                info = archive.getinfo(member)
                if info.is_dir() or not 0 < info.file_size <= 8 * 1024**2 or (info.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError("Unsafe COCO ZIP member")
                data = archive.read(member)
                inspect_image(data, expected)
                publish(path, data)
            data = path.read_bytes()
            inspect_image(data, expected)
            result.append({"id": key, "image": filename, "path": str(path), "sha256": expected["sha256"],
                           "bytes": len(data), "width": expected["width"], "height": expected["height"], "split": "val2017"})
        return result
    finally:
        if archive:
            archive.close()


def select_development(plain, gallery, count=DEV_GROUPS, *, paired=None):
    by_sha = {}
    for candidate in gallery:
        by_sha.setdefault(candidate["sha256"], []).append(candidate["id"])
    if len(by_sha) < count:
        raise ValueError("Not enough distinct JPEG groups for the fixed development sample")
    groups = {"jpeg-sha256:" + digest: sorted(ids, key=int) for digest, ids in by_sha.items()}
    chosen = sorted(groups, key=lambda group: (hash_order("dev-group", group), group))[:count]
    selection = {}
    for group in chosen:
        key = min(groups[group], key=lambda key: (hash_order("group-image", group, key), int(key)))
        valid = [index for index, text in enumerate(plain[key]["captions"])
                 if text.strip() and (paired is None or paired[key]["captions"][index].strip())]
        if not valid:
            raise ValueError("A selected source image has no nonempty paired caption")
        index = min(valid, key=lambda index: (hash_order("caption", key, index), index))
        selection[key] = {"caption_index": index, "group": group}
    provenance = {"seed": SEED, "unit": "connected exact-JPEG-SHA group", "groups": groups,
                  "groups_total": len(groups), "selected_groups": chosen, "selected": selection,
                  "caption_selection": "Minimum fixed hash over original positions nonempty in both plain and negated; no reindexing",
                  "remaining_source_groups": "public-unused; public data, not unseen independent confirmation"}
    return selection, provenance


def retrieval_panel(rows, gallery, selection, grouping, variant, *, dev):
    by_id = {key: group for group, ids in grouping["groups"].items() for key in ids}
    selected_groups = set(grouping["selected_groups"])
    queries = []
    for key in sorted(rows, key=int):
        indices = [selection[key]["caption_index"]] if dev and key in selection else ([] if dev else range(len(rows[key]["captions"])))
        for index in indices:
            chosen = key in selection and index == selection[key]["caption_index"]
            queries.append({"id": f"negbench:{variant}:{key}:{index}", "text": rows[key]["captions"][index],
                            "positive_ids": [key], "source_image_id": key, "variant": variant,
                            "caption_index": index, "dev_selected": chosen, "group": by_id[key],
                            "model_eligible": bool(rows[key]["captions"][index].strip()),
                            "exclusion_reason": None if rows[key]["captions"][index].strip() else "source-empty-caption",
                            "source_partition": "dev-selected-group" if by_id[key] in selected_groups else "public-unused"})
    return {"schema": 1, "task": "source-image-retrieval", "scope": SCOPE, "formal_evidence": False, "dataset": "NegBench",
            "variant": variant, "subset": "dev1000" if dev else "full", "seed": SEED,
            "gold_semantics": "Official caption source image only; not exhaustive all-condition relevance labels",
            "metric_semantics": "Single-positive source-image Recall/MRR; full-ranking AP equals reciprocal rank, not exhaustive semantic AP",
            "source_empty_captions": sum(not row["model_eligible"] for row in queries),
            "candidates": gallery, "queries": queries}


def model_inputs(panel):
    """Explicit model-input allowlist. Gold and audit metadata never enter prompts."""
    if panel["task"] == "source-image-retrieval":
        return {"candidates": [{"id": row["id"], "path": row["path"]} for row in panel["candidates"]],
                "queries": [{"id": row["id"], "text": row["text"]} for row in panel["queries"] if row.get("model_eligible", True)],
                "excluded_query_ids": [row["id"] for row in panel["queries"] if not row.get("model_eligible", True)]}
    if panel["task"] == "single-image-two-caption-matching":
        return {"candidates": [{"id": row["id"], "path": row["path"]} for row in panel["candidates"]],
                "items": [{"id": row["id"], "image_id": row["image_id"], "captions": row["captions"]} for row in panel["items"]]}
    raise ValueError("Unsupported model-input task")


def factor_panel(path, gallery):
    path = safe(path)
    if sha(path) != FACTOR_SHA:
        raise ValueError("FACTOR bytes differ from the pinned author commit")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != 1695:
        raise ValueError("Pinned FACTOR sample count differs")
    panel, audit = matching_panel(rows, gallery)
    source = {"sha256": FACTOR_SHA, "bytes": path.stat().st_size, "url": FACTOR_URL,
              "commit": FACTOR_COMMIT, "items": len(rows)}
    return panel, source, audit


def matching_panel(rows, gallery):
    """Preserve original caption positions and a/b/both gold, never retrieval qrels."""
    lookup = {row["id"]: row for row in gallery}
    items, audit, seen = [], {}, set()
    for row in rows:
        key, sample = image_id(row["image_id"]), row["sample_id"]
        if not isinstance(sample, str) or not sample or sample in seen:
            raise ValueError("Duplicate/missing FACTOR sample ID")
        seen.add(sample)
        if key not in lookup or row["image"] != f"val2017/{int(key):012d}.jpg":
            raise ValueError("FACTOR image does not map to the shared val2017 gallery")
        captions = [row["caption_a"], row["caption_b"]]
        if any(not isinstance(text, str) or not text.strip() for text in captions) or row["correct"] not in ("a", "b", "both"):
            raise ValueError("Invalid FACTOR caption pair/gold")
        for name in ("parse_a", "parse_b"):
            parse = row[name]
            if not isinstance(parse, dict) or not isinstance(parse.get("concepts"), list) or not parse["concepts"]:
                raise ValueError("FACTOR caption structure is malformed")
            for concept in parse["concepts"]:
                if (not isinstance(concept, dict) or not isinstance(concept.get("text"), str)
                        or not concept["text"].strip() or not isinstance(concept.get("is_negated"), bool)):
                    raise ValueError("FACTOR caption concept must have text and boolean polarity")
        items.append({"id": sample, "image_id": key, "source_image_id": key,
                      "captions": captions, "caption_a": captions[0], "caption_b": captions[1],
                      "correct": row["correct"],
                      "correct_caption_index": None if row["correct"] == "both" else (0 if row["correct"] == "a" else 1),
                      "correct_caption_indices": [0, 1] if row["correct"] == "both" else ([0] if row["correct"] == "a" else [1]),
                      "operator": row["operator"], "test_type": row["test_type"],
                      "parse_a": row["parse_a"], "parse_b": row["parse_b"],
                      "group": "jpeg-sha256:" + lookup[key]["sha256"]})
        audit[sample] = row["meta"]
    return ({"schema": 1, "task": "single-image-two-caption-matching", "dataset": "FACTOR",
             "scope": SCOPE, "formal_evidence": False, "subset": "full", "gold_semantics": "Author-provided a/b operator gold or both for equivalence; one image per item",
             "candidates": gallery, "items": items,
             "metric_semantics": "Operator matching accuracy and equivalence consistency are distinct author tasks; not retrieval Recall/mAP"},
            {"schema": 1, "scope": "audit only; image presence/absence metadata is forbidden as model input", "items": audit})


def build(args):
    plain, negated, inventory, sources = load_sources(args.raw, args.inventory)
    gallery = candidates(inventory, args.images, getattr(args, "val_zip", None))
    selection, grouping = select_development(plain, gallery, paired=negated)
    artifacts = {f"negbench-{variant}-{subset}.json": retrieval_panel(rows, gallery, selection, grouping, variant, dev=dev)
                 for variant, rows in (("plain", plain), ("negated", negated))
                 for subset, dev in (("full", False), ("dev1000", True))}
    artifacts["group-provenance.json"] = grouping
    artifacts["audit-metadata.json"] = {"schema": 1, "scope": "audit only; forbidden as model input",
        "images": {key: {field: row[field] for field in ("positive_objects", "negative_objects", "original_filepath")}
                   | {"caption_count": len(row["captions"])} for key, row in plain.items()}}
    factor, factor_source, factor_audit = factor_panel(args.factor, gallery)
    artifacts["factor-full.json"] = factor
    artifacts["factor-audit-metadata.json"] = factor_audit
    sources["factor_bench.jsonl"] = factor_source
    for name, panel in artifacts.items():
        if "candidates" in panel:
            panel["sources"] = sources
    encoded = {name: canonical(value) for name, value in artifacts.items()}
    receipt = {"schema": 1, "state": "COMPLETE", "scope": SCOPE, "seed": SEED, "sources": sources,
               "images": len(gallery), "source_image_groups": grouping["groups_total"],
               "selected_groups": len(selection), "retrieval_queries_per_full_variant": sum(len(row["captions"]) for row in plain.values()),
               "source_empty_caption_counts": {"plain": sum(not text.strip() for row in plain.values() for text in row["captions"]),
                                                "negated": sum(not text.strip() for row in negated.values() for text in row["captions"])},
               "factor_items": len(factor["items"]),
               "outputs": {name: {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)} for name, data in encoded.items()}}
    encoded["preparation.json"] = canonical(receipt)
    return encoded, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "verify"))
    parser.add_argument("--raw", type=Path, default=DATA / "raw/negbench")
    parser.add_argument("--inventory", type=Path, default=DATA / "raw/coco-val2017-expected.json")
    parser.add_argument("--factor", type=Path, default=DATA / "raw/factor/factor_bench.jsonl")
    parser.add_argument("--images", type=Path, default=ROOT / "dl/retrieval3090/data/coco2017/val2017")
    parser.add_argument("--out", type=Path, default=DATA / "panels")
    parser.add_argument("--val-zip", type=Path, help="Optional already-downloaded official ZIP; only missing SHA-verified images are published")
    args = parser.parse_args()
    out = safe(args.out)
    for source in (args.raw, args.inventory, args.factor, args.images):
        source = safe(source)
        if out == source or out in source.parents or source in out.parents:
            raise ValueError("Outputs must be separate from input artifacts")
    if args.action == "verify" and args.val_zip:
        parser.error("verify never extracts or writes images; omit --val-zip")
    with writer(out, readonly=args.action == "verify"):
        artifacts, receipt = build(args)
        if args.action == "prepare":
            for name, data in artifacts.items():
                publish(out / name, data)
        else:
            for name, data in artifacts.items():
                path = safe(out / name)
                if not path.is_file() or path.read_bytes() != data:
                    raise ValueError(f"Prepared artifact missing or changed: {name}")
    print(json.dumps({"state": "VERIFIED" if args.action == "verify" else receipt["state"],
                      "scope": SCOPE, "images": receipt["images"], "selected_groups": receipt["selected_groups"],
                      "full_queries_per_variant": receipt["retrieval_queries_per_full_variant"],
                      "factor_items": receipt["factor_items"], "out": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
