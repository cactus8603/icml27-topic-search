"""[codex] Acquire public development inputs without accessing sealed data."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import time
from zipfile import ZipFile
from zoneinfo import ZoneInfo

import requests

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "dl/retrieval3090"
VAL_URL = "https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip"
# Published by TensorFlow Datasets for this exact official COCO archive.
VAL_SHA = "4f7e2ccb2866ec5041993c9cf2a952bbed69647b115d0f74da7ce8f4bef82f05"
VAL_BYTES = 815585330
FACTOR_REF = "8af1972ddfd03e8437d5f5c96d5a784e94b82d99"
FACTOR_SHA = "37f179a66913d2fbf55bf4f9a852273e894d53683dda0c7a8a7fff6c7b7d387d"


def safe(path):
    path = Path(path).absolute()
    if "retrieval-independent-panel" in str(path):
        raise ValueError("Sealed inputs are forbidden")
    if path.is_symlink() or "retrieval-independent-panel" in str(path.resolve()):
        raise ValueError("Symlinks and sealed inputs are forbidden")
    return path


def sha(path):
    with safe(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path = safe(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    safe(pending).write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    pending.replace(path)


def log(**details):
    print(json.dumps({"owner": "[codex]", "time": datetime.now(ZoneInfo("Asia/Taipei")).isoformat(),
                      **details}), flush=True)


def transfer(url, path, limit):
    """Resume a public archive; never log redirect URLs or request exceptions."""
    path = safe(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(5):
        offset = path.stat().st_size if path.exists() else 0
        if offset > limit:
            raise ValueError("Partial input exceeds its size budget")
        try:
            with requests.get(url, stream=True, headers={"Range": f"bytes={offset}-"} if offset else {},
                              timeout=(20, 60)) as response:
                # A completed but not yet verified partial is checked by the caller.
                if response.status_code == 416 and offset:
                    return
                response.raise_for_status()
                if offset and response.status_code != 206:
                    offset = 0
                if offset and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise ValueError("Unexpected resumed byte range")
                size, updated = offset, time.monotonic()
                with path.open("ab" if offset else "wb") as stream:
                    for block in response.iter_content(4 * 1024**2):
                        size += len(block)
                        if size > limit:
                            raise ValueError("Download exceeds its size budget")
                        stream.write(block)
                        if time.monotonic() - updated > 20:
                            log(stage="download", file=path.name, bytes=size)
                            updated = time.monotonic()
                return
        except (requests.RequestException, OSError) as error:
            log(stage="download_retry", file=path.name, attempt=attempt+1, error=type(error).__name__)
            if attempt == 4:
                raise RuntimeError("Public input transfer failed; resumable partial preserved") from None
            time.sleep(min(2**attempt, 8))


def factor(root):
    directory = safe(root / "data/literature-development/raw/factor")
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "factor_bench.jsonl"
    url = f"https://raw.githubusercontent.com/SultanMo/factored-vlm/{FACTOR_REF}/data/factor_bench.jsonl"
    if not target.exists():
        pending = target.with_suffix(".download-partial")
        transfer(url, pending, 5 * 1024**2)
        if sha(pending) != FACTOR_SHA:
            raise ValueError("FACTOR does not match the author-published SHA")
        pending.replace(target)
    if sha(target) != FACTOR_SHA:
        raise ValueError("Existing FACTOR data differ from the pinned release")
    rows = [json.loads(line) for line in target.read_text().splitlines() if line.strip()]
    if len(rows) != 1695:
        raise ValueError("Unexpected FACTOR release size")
    write_json(directory / "source-receipt.json", {"owner": "[codex]", "url": url,
                "revision": FACTOR_REF, "sha256": FACTOR_SHA, "rows": len(rows)})
    log(stage="factor_complete", rows=len(rows))


def coco(root, workers):
    from PIL import Image
    raw = safe(root / "data/literature-development/raw")
    expected_path = raw / "coco-val2017-expected.json"
    expected = json.loads(expected_path.read_text())
    inventory = expected["images"]
    if len(inventory) != 5000 or expected.get("scope") != "only already exposed val2017 public fingerprints; no sealed manifest":
        raise ValueError("Only the full exposed validation fingerprint inventory is permitted")
    images = safe(root / "data/coco2017/val2017")
    images.mkdir(parents=True, exist_ok=True)
    archive = safe(root / "data/literature-development/raw/val2017.zip.download-partial")
    expected_names = {f"{int(key):012d}.jpg" for key in inventory}
    if {path.name for path in images.glob("*.jpg")} - expected_names:
        raise ValueError("COCO image directory contains unexpected JPEGs")
    existing = {key for key, row in inventory.items()
                if (images / f"{int(key):012d}.jpg").is_file()
                and sha(images / f"{int(key):012d}.jpg") == row["sha256"]}
    archive_record = None
    receipt_path = raw / "coco-val2017-images-receipt.json"
    downloaded = len(existing) < len(inventory)
    if downloaded:
        log(stage="coco_acquisition", existing=len(existing), required=len(inventory))
        transfer(VAL_URL, archive, 1200 * 1024**2)
        archive_record = {"source_url": VAL_URL, "bytes": archive.stat().st_size, "sha256": sha(archive)}
        if archive_record["bytes"] != VAL_BYTES or archive_record["sha256"] != VAL_SHA:
            raise ValueError("COCO archive differs from the published TFDS checksum")
        with ZipFile(archive) as zipped:
            names = {f"val2017/{int(key):012d}.jpg": key for key in inventory}
            files = {info.filename: info for info in zipped.infolist() if not info.is_dir()}
            if (len(files) != sum(not info.is_dir() for info in zipped.infolist())
                    or set(files) != set(names) or any(info.file_size > 2*1024**2 for info in files.values())):
                raise ValueError("Archive differs from the official validation image inventory")
            for number, (name, key) in enumerate(names.items(), 1):
                path = safe(images / Path(name).name)
                if key not in existing:
                    body = zipped.read(name)  # ZipFile verifies each member's CRC.
                    if hashlib.sha256(body).hexdigest() != inventory[key]["sha256"]:
                        raise ValueError("Official JPEG differs from the previously frozen public fingerprint")
                    if path.exists():
                        raise ValueError("Refusing to overwrite an existing different JPEG")
                    pending = path.with_suffix(".acquire-partial")
                    safe(pending).write_bytes(body)
                    pending.replace(path)
                if number % 500 == 0:
                    log(stage="extract", images=number, total=5000)
    elif receipt_path.exists():
        archive_record = json.loads(receipt_path.read_text()).get("archive")

    def inspect(item):
        key, row = item
        path = safe(images / f"{int(key):012d}.jpg")
        if sha(path) != row["sha256"]:
            raise ValueError("Image fingerprint changed")
        with Image.open(path) as image:
            image.load()
            if image.format != "JPEG" or list(image.size) != [row["width"], row["height"]]:
                raise ValueError("Image dimensions/format differ from the frozen public annotation")
        return key, {"path": str(path), "sha256": row["sha256"], "bytes": path.stat().st_size,
                     "width": row["width"], "height": row["height"], "split": "val2017"}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        checked = dict(pool.map(inspect, inventory.items()))
    write_json(receipt_path, {"owner": "[codex]", "scope": "public val2017 only; no sealed inputs",
               "expected_inventory_sha256": sha(expected_path), "archive": archive_record,
               "verified_at_taipei": datetime.now(ZoneInfo("Asia/Taipei")).isoformat(),
               "verification_program_sha256": sha(Path(__file__)),
               "operation": "download-and-verify" if downloaded else "verify-existing-cache",
               "images": checked, "image_bytes": sum(row["bytes"] for row in checked.values())})
    # The source JPEGs and their per-file hashes now preserve the useful content.
    # Remove only this downloader's redundant, verified archive owned by us.
    if archive.exists():
        stat = archive.stat()
        if stat.st_uid != os.getuid() or stat.st_nlink != 1 or sha(archive) != archive_record["sha256"]:
            raise ValueError("Archive ownership/integrity changed before cleanup")
        archive.unlink()
        log(stage="cleanup_verified_archive", reclaimed_bytes=stat.st_size)
    log(stage="coco_complete", images=len(checked), image_bytes=sum(row["bytes"] for row in checked.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=RUNTIME)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--factor-only", action="store_true")
    args = parser.parse_args()
    root = safe(args.root)
    factor(root)
    if not args.factor_only:
        coco(root, args.workers)


if __name__ == "__main__":
    main()
