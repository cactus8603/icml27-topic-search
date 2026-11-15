#!/usr/bin/env python3
"""Download only pinned public retrieval assets, without Hugging Face credentials.

Large files are resumable and verified against Hugging Face LFS SHA-256; small
files are verified against Git's blob SHA-1. Every completed file also receives
a SHA-256 in the local download receipt. No model code is imported or executed.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "dl/retrieval3090"
HF_HOME = ROOT / "hf_home"
CONFIG = ROOT / "topics/compositional-retrieval/code/killtest/configs/params_headroom.json"
ANNOTATIONS_SHA256 = "113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268"
ANNOTATIONS_URLS = (
    "https://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    "https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip",
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
)
MODEL_PINS = {
    "cs_clip": ("kbora/CS-CLIP", "afd6e90f39fe49137fe15d6a462d994eae4811f5"),
    "core_embed": ("Alibaba-NLP/core-emb-2b", "cc9870ea92d64ebd5f8f9fc368d1120453564158"),
    "core_reranker": ("Alibaba-NLP/core-reranker-2b", "64e1a36cd6ecace459a4c7cd8f7d0ef5d6fee30b"),
}
DATASET_PIN = ("kbora/Half-Truths", "ddcd448cc27c7c1b1d485c4754623566c245fdca")
PRINT_LOCK = threading.Lock()
RECEIPT_LOCK = threading.Lock()
USER_AGENT = "icml27-retrieval3090-public-assets/1.0"


def log(message):
    with PRINT_LOCK:
        print(f"[codex] {message}", flush=True)


def request(url, headers=None):
    # Deliberately use urllib rather than an authenticated Hugging Face client.
    return urlopen(Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})}), timeout=45)


def hf_items(kind, repo, revision, files, hf_home):
    endpoint = f"https://huggingface.co/api/{kind}/{repo}/revision/{revision}?blobs=true"
    with request(endpoint) as stream:
        metadata = json.load(stream)
    if metadata.get("sha") != revision:
        raise ValueError(f"Pinned revision did not resolve exactly: {repo}")
    siblings = {row["rfilename"]: row for row in metadata["siblings"]}
    prefix = "models" if kind == "models" else "datasets"
    folder = hf_home / "hub" / f"{prefix}--{repo.replace('/', '--')}" / "snapshots" / revision
    for filename in files:
        if Path(filename).is_absolute() or ".." in Path(filename).parts:
            raise ValueError(f"Unsafe filename: {filename}")
        row = siblings[filename]
        lfs = row.get("lfs") or {}
        oid = lfs.get("sha256") or lfs.get("oid", "").removeprefix("sha256:")
        checksum_kind, checksum = ("sha256", oid) if oid else ("git_blob_sha1", row["blobId"])
        expected_size = row.get("size", lfs.get("size"))
        if not isinstance(expected_size, int) or expected_size <= 0:
            raise ValueError(f"Missing trusted size: {repo}/{filename}")
        route = f"datasets/{repo}" if kind == "datasets" else repo
        yield {"path": str(folder / filename), "source_urls": [
            f"https://huggingface.co/{route}/resolve/{revision}/{quote(filename, safe='/')}"],
            "repo": repo, "revision": revision, "file": filename,
            "checksum_kind": checksum_kind, "expected_checksum": checksum, "expected_size": expected_size}


def verify(path, item):
    size = path.stat().st_size
    if item.get("expected_size") is not None and size != item["expected_size"]:
        raise ValueError(f"Wrong size for {path}: {size} != {item['expected_size']}")
    sha256 = hashlib.sha256()
    expected = hashlib.sha256() if item["checksum_kind"] == "sha256" else hashlib.sha1()
    if item["checksum_kind"] == "git_blob_sha1":
        expected.update(f"blob {size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha256.update(block)
            expected.update(block)
    if expected.hexdigest() != item["expected_checksum"]:
        raise ValueError(f"Checksum mismatch for {path}; retained for inspection")
    return {"size": size, "sha256": sha256.hexdigest()}


def record(receipt_path, item, verified, source):
    with RECEIPT_LOCK:
        with receipt_path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {
                "scope": "public pinned assets for the exposed retrieval panel; no sealed panel", "files": {}}
            receipt["files"][item["path"]] = item | verified | {
                "source_used": source, "verified_utc": datetime.now(timezone.utc).isoformat()}
            pending = receipt_path.with_suffix(".pending")
            pending.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
            pending.replace(receipt_path)


def transfer(url, pending, item):
    offset = pending.stat().st_size if pending.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    try:
        response = request(url, headers)
    except HTTPError as error:
        if error.code == 416 and pending.exists():
            verify(pending, item)
            return
        raise
    with response:
        if offset and response.status != 206:
            log(f"Server restarted download: {item['file']}")
            offset = 0
        if offset:
            content_range = response.headers.get("Content-Range", "")
            if not content_range.startswith(f"bytes {offset}-"):
                raise ValueError(f"Server returned an unexpected byte range for {item['file']}")
        mode = "ab" if offset else "wb"
        started, updated, total = time.monotonic(), time.monotonic(), offset
        with pending.open(mode) as stream:
            for block in iter(lambda: response.read(8 * 1024 * 1024), b""):
                stream.write(block)
                total += len(block)
                if time.monotonic() - updated >= 20:
                    log(f"Progress {item['repo']}/{item['file']}: {total / 2**20:.1f} MiB "
                        f"({(total-offset) / 2**20 / max(1, time.monotonic()-started):.1f} MiB/s)")
                    updated = time.monotonic()


def download_unlocked(item, receipt_path):
    path = Path(item["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError(f"Refusing to overwrite a symlink: {path}")
    if path.exists():
        verified = verify(path, item)
        record(receipt_path, item, verified, "existing verified public cache")
        log(f"Verified existing {item['repo']}/{item['file']}")
        return
    pending = path.with_name(path.name + item.get("partial_suffix", ".download-partial"))
    if pending.is_symlink():
        raise ValueError(f"Refusing a symlink partial download: {pending}")
    log(f"Downloading {item['repo']}/{item['file']}")
    last = None
    for attempt in range(6):
        for url in item["source_urls"]:
            try:
                transfer(url, pending, item)
                verified = verify(pending, item)
                pending.replace(path)
                record(receipt_path, item, verified, url)
                log(f"Verified {item['repo']}/{item['file']}: {verified['size']} bytes sha256={verified['sha256']}")
                return
            except (HTTPError, URLError, TimeoutError, OSError) as error:
                last = error
                # Do not print signed redirect URLs or any authentication material.
                log(f"Retry {attempt + 1} {item['repo']}/{item['file']}: {type(error).__name__}")
        if attempt != 5:
            time.sleep(min(2 ** attempt, 15))
    raise RuntimeError(f"Public download failed: {item['repo']}/{item['file']} ({type(last).__name__})")


def download(item, receipt_path):
    locks = receipt_path.parent / "download-locks"
    locks.mkdir(exist_ok=True)
    lock_path = locks / (hashlib.sha256(item["path"].encode()).hexdigest() + ".lock")
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return download_unlocked(item, receipt_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, default=RUNTIME)
    parser.add_argument("--hf-home", type=Path, default=HF_HOME)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--include-reconstruction-inputs", action="store_true",
                        help="Also fetch three pinned Half-Truths files only to reconstruct the exposed historical panel")
    parser.add_argument("--reconstruction-only", action="store_true",
                        help="Prioritize only the three historical exclusion inputs, using separate partials and receipt")
    args = parser.parse_args()
    args.runtime.mkdir(parents=True, exist_ok=True)
    receipt_path = args.runtime / ("reconstruction-public-download-receipt.json" if args.reconstruction_only
                                   else "public-download-receipt.json")
    config = json.loads(CONFIG.read_text())
    items = [] if args.reconstruction_only else [{"path": str(args.runtime / "data/coco2017/annotations_trainval2017.zip"),
              "repo": "COCO official", "revision": "2017", "file": "annotations_trainval2017.zip",
              "source_urls": list(ANNOTATIONS_URLS), "checksum_kind": "sha256",
              "expected_checksum": ANNOTATIONS_SHA256, "expected_size": None}]
    for name, pin in (() if args.reconstruction_only else MODEL_PINS.items()):
        spec = config["models"][name]
        if (spec["repo"], spec["revision"]) != pin:
            raise ValueError(f"Model pin changed: {name}")
        items.extend(hf_items("models", *pin, spec["files"], args.hf_home))
    if args.include_reconstruction_inputs or args.reconstruction_only:
        items.extend(hf_items("datasets", *DATASET_PIN,
                              [f"data/{subset}.parquet" for subset in ("coco_qwen", "coco_mistral", "cc3m_qwen")],
                              args.hf_home))
    if args.reconstruction_only:
        for item in items:
            item["partial_suffix"] = ".reconstruction-prefetch-partial"
    log(f"Starting {len(items)} pinned public files with {args.workers} workers; receipt={receipt_path}")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(download, item, receipt_path) for item in items]
        for future in futures:
            future.result()
    log(f"Complete: verified all {len(items)} public files")


if __name__ == "__main__":
    main()
