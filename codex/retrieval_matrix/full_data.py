"""[codex] Prepare full native MCMR image bytes without changing old pilots."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import tarfile
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_SHA = "7cbda9a5fbdba1240590885c576b4b31dccd047f14d2cc5b8a8e87c94bda2b86"
METADATA_SHA = "05242fb414333a99c562a986edb8b2d1997db1726f731b6131ab97bf557fcbc0"


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def log(**value):
    print(json.dumps({"owner": "[codex]", "time": datetime.now(ZoneInfo("Asia/Taipei")).isoformat(),
                      **value}), flush=True)


def prepare_mcmr(source, out):
    from PIL import Image
    source, out = Path(source).resolve(), Path(out).absolute()
    if out.is_symlink() or not out.resolve().is_relative_to(ROOT / "dl/retrieval-matrix/data"):
        raise ValueError("Only an isolated matrix data directory is permitted")
    archive, raw = source / "images.tar.gz", source / "candidates.jsonl"
    if sha(archive) != ARCHIVE_SHA or sha(raw) != METADATA_SHA:
        raise ValueError("Pinned original MCMR archive/metadata checksum mismatch")
    normalized = source / "candidates.normalized.jsonl"
    expected = {}
    for line in normalized.open():
        row = json.loads(line)
        filename = row["image_filename"]
        if Path(filename).name != filename or not filename:
            raise ValueError("Unsafe expected image filename")
        expected[row["id"]] = filename
    if len(expected) != 104981:
        raise ValueError("Unexpected MCMR metadata count")
    out.mkdir(parents=True, exist_ok=True)
    images = out / "images"
    images.mkdir(exist_ok=True)
    entries, seen = [], set()
    with tarfile.open(archive, "r|gz") as stream:
        for member in stream:
            if member.isdir() and member.name.rstrip("/") == "images":
                continue
            name = Path(member.name).name
            if (not member.isfile() or member.name != "images/" + name
                    or not 0 < member.size <= 32 * 1024**2 or name in seen):
                raise ValueError("Unexpected, repeated or unsafe MCMR archive entry")
            seen.add(name)
            target = images / name
            if target.is_symlink():
                raise ValueError("Refusing an existing image symlink")
            with stream.extractfile(member) as source_stream:
                content = source_stream.read(member.size + 1)
            if len(content) != member.size:
                raise ValueError("Truncated archive member")
            digest = hashlib.sha256(content).hexdigest()
            if target.exists():
                if target.stat().st_size != member.size or sha(target) != digest:
                    raise ValueError("Existing image differs from original MCMR archive")
            else:
                pending = target.with_name(name + f".matrix-partial-{os.getpid()}")
                with pending.open("xb") as f:
                    f.write(content)
                pending.replace(target)
            with Image.open(target) as image:
                width, height = image.size
                image.verify()
            entries.append({"file": name, "bytes": member.size, "sha256": digest,
                            "width": width, "height": height})
            if len(entries) % 10000 == 0:
                log(stage="mcmr_full_extract", images=len(entries))
    missing = sorted(set(expected.values()) - seen)
    receipt = out / "images-receipt.jsonl"
    body = "".join(json.dumps(row, sort_keys=True) + "\n" for row in sorted(entries, key=lambda x: x["file"]))
    pending = receipt.with_suffix(".pending")
    pending.write_text(body)
    pending.replace(receipt)
    summary = {"owner": "[codex]", "scope": "complete official archive images; no inference or label selection",
               "archive_sha256": ARCHIVE_SHA, "raw_metadata_sha256": METADATA_SHA,
               "normalized_metadata_sha256": sha(normalized), "candidate_ids": len(expected),
               "expected_unique_images": len(set(expected.values())), "archive_images": len(entries),
               "missing_images": missing, "image_bytes": sum(row["bytes"] for row in entries),
               "images_receipt_sha256": sha(receipt), "producer_sha256": sha(__file__),
               "verified_at_taipei": datetime.now(ZoneInfo("Asia/Taipei")).isoformat()}
    (out / "preparation.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    log(stage="mcmr_full_complete", **{k: summary[k] for k in
        ("candidate_ids", "expected_unique_images", "archive_images", "image_bytes")},
        missing_images=len(missing))
    if missing:
        raise ValueError("Official archive lacks candidate images; receipt preserved, no silent filtering")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "dl/retrieval3090/data/mcmr")
    parser.add_argument("--out", type=Path, default=ROOT / "dl/retrieval-matrix/data/mcmr-full")
    args = parser.parse_args()
    prepare_mcmr(args.source, args.out)


if __name__ == "__main__":
    main()
