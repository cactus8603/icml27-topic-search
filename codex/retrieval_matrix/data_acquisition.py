"""Acquire fixed public benchmark releases, without running model evaluation.

Run with the existing retrieval3090 environment (pyarrow and Pillow required).
All downloaded files are checked against the pinned Hugging Face Git/LFS digest.
The full per-file receipt is untracked under dl/; no held-out panel is read.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import io
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request

REPO = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO / "dl/retrieval-matrix/data"
SOURCES = {
    "sugarcrepe_pp": ("Aman-J/SugarCrepe_pp", "dea2a1b6f9e1069c609f676aa55ec61e9b65fb61"),
    "cola": ("array/cola", "daee86b15a60308cfcc277ea11ab7f6be60fdfbc"),
    "flickr30k_test": ("royokong/flickr30k_test", "a07382c123b95eec85827a9d8ce3eb135093411a"),
}
COLA_GITHUB_REVISION = "997457b78a1cadf81954747e2ebae9d532f0c2d3"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_payload(path: Path, spec: dict) -> dict:
    """Check source size plus LFS SHA256 or Git's blob SHA1 (never pointer SHA)."""
    size = path.stat().st_size
    if spec.get("size") is not None and size != spec["size"]:
        raise ValueError(f"size mismatch: {path}: {size} != {spec['size']}")
    digest = sha256(path)
    if spec.get("lfs"):
        expected = spec["lfs"]["sha256"]
        if digest != expected:
            raise ValueError(f"SHA256 mismatch: {path}")
        kind = "lfs_sha256"
    else:
        h = hashlib.sha1(f"blob {size}\0".encode())
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        expected = spec["blobId"]
        if h.hexdigest() != expected:
            raise ValueError(f"Git blob SHA1 mismatch: {path}")
        kind = "git_blob_sha1"
    return {"bytes": size, "sha256": digest, "source_digest_kind": kind,
            "source_digest": expected}


def safe_relative(value: str) -> Path:
    p = Path(value)
    if p.is_absolute() or ".." in p.parts or not p.parts:
        raise ValueError(f"unsafe source path: {value}")
    return p


def fetch(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    for attempt in range(6):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "public-benchmark-acquisition/1.0"})
            with urllib.request.urlopen(request, timeout=90) as response, partial.open("wb") as out:
                while chunk := response.read(1 << 20):
                    out.write(chunk)
            partial.replace(target)
            return
        except (OSError, urllib.error.URLError):
            if attempt == 5:
                raise
            time.sleep(min(2 ** attempt, 16))


def metadata(name: str, root: Path, offline: bool) -> dict:
    repo_id, revision = SOURCES[name]
    path = root / name / "source_api.json"
    if not path.exists() and not offline:
        fetch(f"https://huggingface.co/api/datasets/{repo_id}/revision/{revision}?blobs=true", path)
    data = json.loads(path.read_text())
    if data["sha"] != revision or not all("blobId" in x for x in data["siblings"]):
        raise ValueError(f"wrong revision or incomplete blob metadata: {name}")
    return data


def acquire_one(name: str, spec: dict, root: Path, offline: bool) -> dict:
    repo_id, revision = SOURCES[name]
    relative = safe_relative(spec["rfilename"])
    target = root / name / relative
    reused = None
    if not target.exists() and name == "cola" and str(relative).startswith("images/coco/"):
        candidates = [REPO / "dl/retrieval3090/data/coco2017" / Path(*relative.parts[2:]),
                      REPO / "dl/claude-panel/conj-panel-v1/dev/images" / relative.name]
        for candidate in candidates:
            if candidate.exists():
                try:
                    verify_payload(candidate, spec)
                except ValueError:
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(os.path.relpath(candidate, target.parent))
                reused = str(candidate.relative_to(REPO))
                break
    if not target.exists():
        if offline:
            raise FileNotFoundError(target)
        fetch(f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{relative.as_posix()}", target)
    result = {"path": f"{name}/{relative.as_posix()}", **verify_payload(target, spec)}
    if reused:
        result["reused_from"] = reused
    if relative.suffix.lower() in {".jpg", ".jpeg", ".png"}:
        from PIL import Image
        with Image.open(target) as image:
            image.verify()
        result["image_verified"] = True
    return result


def annotation_specs(name: str, meta: dict) -> list[dict]:
    return [s for s in meta["siblings"] if s["rfilename"] == "README.md" or
            (s["rfilename"].startswith("data/") and "clevr" not in s["rfilename"])]


def acquire_coco(relative: str, root: Path, offline: bool,
                 expected_sha256: str | None = None) -> dict:
    """Official COLA HF release omits PACO's referenced COCO image files."""
    from PIL import Image
    path = safe_relative(relative)
    split, name = path.parts[-2:]
    if split not in {"train2017", "val2017"} or not name.endswith(".jpg"):
        raise ValueError(f"unexpected COCO reference: {relative}")
    target = root / "cola" / path
    local_candidates = [REPO / "dl/retrieval3090/data/coco2017" / split / name,
                        REPO / "dl/claude-panel/conj-panel-v1/dev/images" / name]
    # The official image hostname has no matching TLS certificate. COCO publishes
    # these public image URLs with HTTP; never disable certificate verification.
    source = f"http://images.cocodataset.org/{split}/{name}"
    reused = None
    if not target.exists():
        for candidate in local_candidates:
            if candidate.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(os.path.relpath(candidate, target.parent))
                reused = str(candidate.relative_to(REPO))
                break
    if not target.exists():
        if offline:
            raise FileNotFoundError(target)
        fetch(source, target)
    with Image.open(target) as image:
        if image.format != "JPEG":
            raise ValueError(f"COCO image is not JPEG: {relative}")
        width, height = image.size
        image.verify()
    digest = sha256(target)
    if expected_sha256 and digest != expected_sha256:
        raise ValueError(f"COCO image differs from prior acquisition receipt: {relative}")
    result = {"path": "cola/" + relative, "source_url": source,
              "sha256": digest, "bytes": target.stat().st_size,
              "source_digest_kind": "official_source_without_published_file_digest",
              "image_verified": True, "width": width, "height": height}
    if reused:
        result["reused_from"] = reused
    return result


def acquire_cola_upstream(root: Path, offline: bool) -> list[dict]:
    target = root / "cola/upstream"
    tree_path = target / "tree.json"
    if not tree_path.exists() and not offline:
        fetch("https://api.github.com/repos/arijitray1993/COLA/git/trees/" +
              COLA_GITHUB_REVISION + "?recursive=1", tree_path)
    tree = json.loads(tree_path.read_text())
    if tree.get("sha") != COLA_GITHUB_REVISION or tree.get("truncated"):
        raise ValueError("wrong or incomplete original COLA source tree")
    specs = {s["path"]: s for s in tree["tree"] if s["type"] == "blob"}
    files = ["README.md", "LICENSE", "scripts/eval.py",
             "data/COLA_singleobjects_benchmark_GQA.json",
             "data/COLA_singleobjects_benchmark_PACO.json",
             "data/COLA_multiobjects_matching_benchmark.json"]
    receipt = []
    for filename in files:
        path = target / filename
        if not path.exists():
            if offline:
                raise FileNotFoundError(path)
            fetch("https://raw.githubusercontent.com/arijitray1993/COLA/" +
                  COLA_GITHUB_REVISION + "/" + filename, path)
        spec = specs[filename]
        receipt.append({"path": "cola/upstream/" + filename,
                        **verify_payload(path, {"size": spec["size"], "blobId": spec["sha"]})})
    return receipt


def cola_image_paths(root: Path) -> set[str]:
    import pyarrow.parquet as pq
    paths = set()
    for subset in ("singleobjects_gqa", "singleobjects_paco"):
        paths.update(pq.read_table(root / "cola/data" / f"{subset}.parquet", columns=["image"])
                     .column("image").to_pylist())
    table = pq.read_table(root / "cola/data/multiobjects.parquet")
    for field in ("image1", "image2"):
        paths.update(table.column(field).to_pylist())
    return paths


def audit_sugarcrepe(root: Path) -> dict:
    from PIL import Image
    counts, names = {}, set()
    for path in sorted((root / "sugarcrepe_pp/data").glob("*.json")):
        rows = json.loads(path.read_text())
        counts[path.stem] = len(rows)
        for row in rows:
            if not all(isinstance(row[k], str) and row[k] for k in
                       ("filename", "caption", "caption2", "negative_caption")):
                raise ValueError(f"missing SugarCrepe++ fields: {path}")
            names.add(safe_relative(row["filename"]).name)
    image_root = REPO / "dl/retrieval3090/data/coco2017/val2017"
    entries = []
    for name in sorted(names):
        path = image_root / name
        with Image.open(path) as image:
            image.verify()
        entries.append({"path": str(path.relative_to(REPO)), "sha256": sha256(path)})
    receipt = root / "sugarcrepe_pp/image_receipt.jsonl"
    write_jsonl(receipt, entries)
    return {"rows_by_subset": counts, "rows": sum(counts.values()),
            "unique_images": len(names), "missing_images": 0,
            "image_root": str(image_root.relative_to(REPO)),
            "image_receipt_sha256": sha256(receipt)}


def export_flickr(root: Path) -> dict:
    import pyarrow.parquet as pq
    from PIL import Image
    target = root / "flickr30k_test"
    rows = pq.read_table(target / "data/test-00000-of-00001.parquet").to_pylist()
    if len(rows) != 1000:
        raise ValueError(f"CORE Flickr30K expected 1000 images, got {len(rows)}")
    manifest, entries = [], []
    names = set()
    for index, row in enumerate(rows):
        image = row["image"]
        name = safe_relative(image["path"]).name
        if name in names or len(row["text"]) != 5:
            raise ValueError("Flickr30K duplicate image or not five captions")
        names.add(name)
        path = target / "images_flickr_1k_test" / name
        payload = image["bytes"]
        with Image.open(io.BytesIO(payload)) as opened:
            opened.verify()
        path.parent.mkdir(exist_ok=True, parents=True)
        if not path.exists():
            path.write_bytes(payload)
        if sha256(path) != hashlib.sha256(payload).hexdigest():
            raise ValueError(f"Flickr30K extracted image mismatch: {name}")
        entries.append({"path": str(path.relative_to(root)), "sha256": sha256(path),
                        "bytes": len(payload), "image_verified": True})
        manifest.append({"id": name.rsplit(".", 1)[0], "image": "images_flickr_1k_test/"+name,
                         "captions": row["text"], "source_row": index})
    write_jsonl(target / "test_manifest.jsonl", manifest)
    write_jsonl(target / "image_receipt.jsonl", entries)
    # Preserve the exact CORE row format as well as a neutral lossless manifest.
    import csv
    with (target / "test_1k_flickr.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "raw", "imgid"])
        for row in manifest:
            writer.writerow([Path(row["image"]).name, json.dumps(row["captions"]),
                             json.dumps([row["id"]] * len(row["captions"]))])
    return {"images": len(rows), "captions": sum(len(x["text"]) for x in rows),
            "missing_images": 0, "manifest_sha256": sha256(target / "test_manifest.jsonl"),
            "image_receipt_sha256": sha256(target / "image_receipt.jsonl")}


def audit_flickr_lineage(root: Path, offline: bool) -> dict:
    """Distinguish the CORE HF release from Karpathy raw and Entities test."""
    import zipfile
    target = root / "flickr30k_test"
    upstream = target / "upstream"
    upstream.mkdir(exist_ok=True)
    archive = upstream / "caption_datasets.zip"
    archive_digest = "4cfd70132527b80933105e5829dc9034eaab9573482e2e680abbab6130244817"
    url = "https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip"
    if not archive.exists() and not offline:
        fetch(url, archive)
    if sha256(archive) != archive_digest:
        raise ValueError("Karpathy original caption archive SHA256 mismatch")
    with zipfile.ZipFile(archive) as zipped:
        payload = zipped.read("dataset_flickr30k.json")
    annotation_digest = hashlib.sha256(payload).hexdigest()
    if annotation_digest != "db779ee2c5df40c25ff0eebeca464f6124b620cc4a9398650aea71fd1b53d4af":
        raise ValueError("Karpathy original Flickr30K annotations SHA256 mismatch")
    annotation = upstream / "dataset_flickr30k.json"
    annotation.write_bytes(payload)
    original = [row for row in json.loads(payload)["images"] if row["split"] == "test"]
    raw_by_id = {Path(row["filename"]).stem: [s["raw"] for s in row["sentences"]]
                 for row in original}
    core_rows = [json.loads(line) for line in (target / "test_manifest.jsonl").read_text().splitlines()]
    if {row["id"] for row in core_rows} != set(raw_by_id) or len(original) != 1000:
        raise ValueError("CORE Flickr30K image identities differ from original Karpathy test")
    manifest = [{"id": Path(row["filename"]).stem,
                 "image": "images_flickr_1k_test/" + row["filename"],
                 "captions": [s["raw"] for s in row["sentences"]],
                 "source": "Karpathy original test"} for row in original]
    write_jsonl(target / "karpathy_test_manifest.jsonl", manifest)
    exact = sum(row["captions"] == raw_by_id[row["id"]] for row in core_rows)
    changed = sum(sum(a != b for a, b in zip(row["captions"], raw_by_id[row["id"]]))
                  for row in core_rows)
    entities_path = upstream / "entities_test.txt"
    entities_url = ("https://raw.githubusercontent.com/BryanPlummer/flickr30k_entities/"
                    "68b3d6f12d1d710f96233f6bd2b6de799d6f4e5b/test.txt")
    if not entities_path.exists() and not offline:
        fetch(entities_url, entities_path)
    if sha256(entities_path) != "58c576ebf3af64f346cc3e2ede7eb66523ab172b45c8b42414daa9f2914047e1":
        raise ValueError("original Entities test IDs SHA256 mismatch")
    entities_ids = set(entities_path.read_text().split())
    return {"karpathy_source_url": url, "archive_sha256": archive_digest,
            "annotation_sha256": annotation_digest, "exact_image_id_match_with_karpathy_test": True,
            "images_with_exact_original_captions": exact, "captions_with_different_text": changed,
            "entities_test_overlap_images": len(entities_ids & set(raw_by_id)),
            "karpathy_manifest_sha256": sha256(target / "karpathy_test_manifest.jsonl"),
            "default_manifest_text": "CORE-specified HF release, preserved verbatim"}


def audit_cola_annotations(root: Path) -> dict:
    import pyarrow.parquet as pq
    out = {}
    for subset, classes in (("gqa", 320), ("paco", 400)):
        table = pq.read_table(root / "cola/data" / f"singleobjects_{subset}.parquet")
        rows = table.to_pylist()
        upstream = json.loads((root / "cola/upstream/data" /
                               f"COLA_singleobjects_benchmark_{subset.upper()}.json").read_text())
        if len(upstream["labels"]) != classes or len(rows) != len(upstream["data"]):
            raise ValueError(f"COLA source count mismatch: {subset}")
        labels = json.loads((root / "cola/data" / f"singleobjects_{subset}_labels.json").read_text())
        if labels != upstream["labels"]:
            raise ValueError(f"COLA query vocabulary mismatch: {subset}")
        for row, original in zip(rows, upstream["data"]):
            if Path(row["image"]).name != Path(original[0]).name:
                raise ValueError(f"COLA source image mismatch: {subset}")
            label, hard = original[-2:]
            if row["label"] != label or row["hard_list"] != hard:
                raise ValueError(f"COLA source label mismatch: {subset}")
            if len(label) != classes or len(hard) != classes or set(label + hard) - {0, 1}:
                raise ValueError(f"COLA invalid labels: {subset}")
        out[subset] = {"rows": len(rows), "classes": classes,
                       "unique_images": len({x["image"] for x in rows}),
                       "matches_upstream_json": True}
    rows = pq.read_table(root / "cola/data/multiobjects.parquet").to_pylist()
    upstream = json.loads((root / "cola/upstream/data/COLA_multiobjects_matching_benchmark.json").read_text())
    if len(rows) != len(upstream):
        raise ValueError("COLA multiobjects count mismatch")
    for row, original in zip(rows, upstream):
        if [Path(row["image1"]).name, row["caption1"], Path(row["image2"]).name, row["caption2"]] != [
                Path(original[0]).name, original[1], Path(original[2]).name, original[3]]:
            raise ValueError("COLA multiobjects source mismatch")
    out["multiobjects"] = {"pairs": len(rows), "matches_upstream_json": True}
    out["train"] = {"status": "not_published_in_current_official_releases",
                    "development_split_created": False}
    return out


def write_jsonl(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(exist_ok=True, parents=True)
    with path.open("w") as f:
        for entry in entries:
            f.write(json.dumps(entry, sort_keys=True, ensure_ascii=False) + "\n")


def run(root: Path, workers: int, offline: bool, names: list[str]) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    receipts, summary = [], {}
    for name in names:
        meta = metadata(name, root, offline)
        for spec in annotation_specs(name, meta):
            receipts.append(acquire_one(name, spec, root, offline))
        print(f"{name}: fixed annotations verified", flush=True)
        if name == "cola":
            receipts.extend(acquire_cola_upstream(root, offline))
            paths = cola_image_paths(root)
            by_name = {s["rfilename"]: s for s in meta["siblings"]}
            absent = paths - by_name.keys()
            if any(not path.startswith("images/coco/") for path in absent):
                raise ValueError(f"COLA images absent in pinned source: {sorted(absent)[:5]}")
            prior_receipt = root / "cola/file_receipt.jsonl"
            prior_sha = {x["path"]: x["sha256"] for x in
                         map(json.loads, prior_receipt.read_text().splitlines())} if prior_receipt.exists() else {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(acquire_coco, path, root, offline,
                                           prior_sha.get("cola/" + path)) if path in absent
                           else executor.submit(acquire_one, name, by_name[path], root, offline)
                           for path in sorted(paths)]
                for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
                    receipts.append(future.result())
                    if index % 250 == 0 or index == len(paths):
                        print(f"cola: {index}/{len(paths)} images verified", flush=True)
            summary[name] = audit_cola_annotations(root)
            summary[name]["verified_image_union"] = len(paths)
            summary[name]["missing_images"] = 0
        elif name == "sugarcrepe_pp":
            summary[name] = audit_sugarcrepe(root)
        elif name == "flickr30k_test":
            summary[name] = export_flickr(root)
            summary[name]["source_lineage"] = audit_flickr_lineage(root, offline)
        print(f"{name}: audit complete", flush=True)
        receipt_path = root / name / "file_receipt.jsonl"
        write_jsonl(receipt_path, sorted([x for x in receipts if x["path"].startswith(name + "/")],
                                       key=lambda x: x["path"]))
        summary[name]["revision"] = SOURCES[name][1]
        summary[name]["source_repo"] = SOURCES[name][0]
        summary[name]["file_receipt_sha256"] = sha256(receipt_path)
        summary[name]["source_api_sha256"] = sha256(root / name / "source_api.json")
        (root / name / "audit.json").write_text(json.dumps(summary[name], indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--datasets", nargs="+", choices=SOURCES, default=list(SOURCES))
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.workers, args.offline, args.datasets), indent=2))
