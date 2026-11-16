"""[codex] Download and verify the original VOC2007 test image archive."""
from pathlib import Path
import csv
import hashlib
import json
import tarfile
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "dl/retrieval-matrix-rerank/data/voc"
URL = "https://thor.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar"
MD5 = "b6e924de25625d8de591ea690078ad9f"  # torchvision official VOC metadata


def digest(path, algorithm):
    value = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def main():
    from PIL import Image
    RUNTIME.mkdir(parents=True, exist_ok=True)
    archive = RUNTIME / "VOCtest_06-Nov-2007.tar"
    pending = archive.with_suffix(".tar.partial")
    if not archive.exists():
        # Preserve a fully downloaded partial from an interrupted verification.
        if not pending.exists() or digest(pending, "md5") != MD5:
            with urlopen(URL, timeout=45) as stream, pending.open("wb") as output:
                for block in iter(lambda: stream.read(8 * 1024**2), b""):
                    output.write(block)
        if digest(pending, "md5") != MD5:
            raise ValueError("Official VOC archive checksum mismatch")
        pending.replace(archive)
    if digest(archive, "md5") != MD5:
        raise ValueError("Existing official archive differs")
    images = RUNTIME / "images"
    images.mkdir(exist_ok=True)
    receipt = []
    with tarfile.open(archive) as stream:
        for member in stream:
            if not member.name.startswith("VOCdevkit/VOC2007/JPEGImages/") or not member.isfile():
                continue
            name = Path(member.name).name
            if member.name != "VOCdevkit/VOC2007/JPEGImages/" + name:
                raise ValueError("Unsafe original archive member")
            content = stream.extractfile(member).read()
            target = images / name
            if target.is_symlink():
                raise ValueError("Refusing image symlink")
            if target.exists():
                if target.read_bytes() != content:
                    raise ValueError("Existing image differs from archive")
            else:
                target.write_bytes(content)
            with Image.open(target) as image:
                width, height = image.size
                image.verify()
            receipt.append({"file": name, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(),
                            "width": width, "height": height})
    receipt_path = RUNTIME / "image-receipt.jsonl"
    receipt_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in receipt))
    rows = list(csv.DictReader((RUNTIME / "VOC2007_mcq_llama3.1_rephrased.csv").open()))
    referenced = {Path(row["image_path"]).name for row in rows}
    missing = sorted(referenced - {row["file"] for row in receipt})
    result = {"owner": "[codex]", "state": "COMPLETE" if not missing else "FAILED", "source_url": URL,
              "official_md5": MD5, "archive_bytes": archive.stat().st_size,
              "archive_sha256": digest(archive, "sha256"), "images": len(receipt),
              "mcq_rows": len(rows), "mcq_unique_images": len(referenced), "missing": missing,
              "images_receipt_sha256": digest(receipt_path, "sha256"),
              "path_adaptation": "Original CSV stores data/voc2007/VOCdevkit/VOC2007/JPEGImages/name; resolve basename under verified original archive image root, preserve all rows."}
    (RUNTIME / "archive-receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    if missing:
        raise ValueError("Official archive missing referenced MCQ images")


if __name__ == "__main__":
    main()
