"""[codex] Minimal, pinned inference payloads for the remaining A baselines.

Reuses the existing resumable public downloader and checksum receipts. Does not
import model code, allocate a GPU, download training state, or select on scores.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"codex/retrieval3090"))
import download_assets as D

PINS = {
    "siglip2": ("google/siglip2-so400m-patch14-384", "e8e487298228002f3d8a82e0cd5c8ea9c567f57f"),
    "clic_b32": ("nmndeep/CLIC-ViT-B-32-224-CogVLM", "7b6ca58a0375837bcaf847ba6343269dd8c24ae4"),
    "clic_l14": ("nmndeep/CLIC-ViT-L-14-224-CogVLM", "57a65a0588b517a400605b319486dc609d5fede9"),
    "clic_clips_l14": ("nmndeep/CLIC-CLIPS-ViT-L-14-224-CogVLM", "75717eef78af067ba30ab33ad65eadc0675cb77b"),
    "triplet_cc12m": ("TripletCLIP/CC12M_TripletCLIP_ViTB12", "1f8b01920bccf9bb33d6ae51f12d3f1c2fc90845"),
    "negclip_cc12m": ("TripletCLIP/CC12M_NegCLIP_ViTB12", "4afcd9ac5a3169923039ab1e8903c66fbeba1fc5"),
    "gme7": ("Alibaba-NLP/gme-Qwen2-VL-7B-Instruct", "e54cb53a76dba4895a7a2f88fc8021f3679ed4f1"),
    "unime2": ("TianchengGu/UniME-V2-Qwen2VL-2B", "29f0ce56e052985452a5fbe379760038badcecab"),
    "unime7": ("TianchengGu/UniME-V2-Qwen2VL-7B", "2ed1953a1a5f2a321e523f812064447d74ad62d9"),
}


def selected(filename):
    path = Path(filename)
    if any(part in {"Figures", "figures", "optimizer", "checkpoint", "training"} for part in path.parts):
        return False
    return filename.endswith(".safetensors") or (not filename.endswith(".bin") and path.suffix in {".json", ".py", ".txt", ".model", ".md"})


def manifest(runtime):
    models, items = {}, []
    for key, (repo, revision) in PINS.items():
        path = runtime/"metadata"/(repo.replace("/", "--")+".json")
        metadata = json.loads(path.read_text())
        if metadata.get("sha") != revision or metadata.get("gated"):
            raise ValueError("Metadata must identify the exact public pinned model revision")
        rows = []
        folder = ROOT/"hf_home/hub"/("models--"+repo.replace("/", "--"))/"snapshots"/revision
        for row in metadata["siblings"]:
            filename = row["rfilename"]
            if not selected(filename):
                continue
            if Path(filename).is_absolute() or ".." in Path(filename).parts:
                raise ValueError("Unsafe repository payload path")
            lfs = row.get("lfs") or {}
            oid = lfs.get("sha256") or lfs.get("oid", "").removeprefix("sha256:")
            size = row.get("size", lfs.get("size"))
            if type(size) is not int or size <= 0:
                raise ValueError("Official metadata lacks a positive payload size")
            item = {"path": str(folder/filename), "repo": repo, "revision": revision, "file": filename,
                    "expected_size": size, "checksum_kind": "sha256" if oid else "git_blob_sha1",
                    "expected_checksum": oid or row["blobId"],
                    "source_urls": [f"https://huggingface.co/{repo}/resolve/{revision}/{quote(filename, safe='/')}"]}
            rows.append(item)
        models[key] = {"repo": repo, "revision": revision, "snapshot": str(folder), "files": rows,
                       "bytes": sum(row["expected_size"] for row in rows),
                       "qualification": "not yet qualified; fixed published-target P0 must precede P2"}
        items.extend((key, row) for row in rows)
    total = sum(row["bytes"] for row in models.values())
    if total > 100*2**30:
        raise ValueError(f"Inference payloads exceed the user's 100GiB notification threshold: {total} bytes")
    value = {"schema": 1, "owner": "[codex]", "models": models, "bytes": total,
             "scope": "minimal inference safetensors/tokenizer/config/official model code only; no training state"}
    (runtime/"asset-manifest.json").write_text(json.dumps(value, indent=2, sort_keys=True)+"\n")
    return value, items


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, default=ROOT/"dl/retrieval-matrix-extra")
    parser.add_argument("--models", nargs="+", choices=tuple(PINS), default=list(PINS))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--author-drive", action="store_true", help="Also fetch the two author Drive releases and extract only checksum-pinned inference tensors")
    args = parser.parse_args()
    value, items = manifest(args.runtime)
    D.log(f"All pinned inference payloads: {value['bytes']} bytes / {value['bytes']/2**30:.6f} GiB")
    if args.metadata_only:
        return
    receipt = args.runtime/"public-download-receipt.json"
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(D.download, item, receipt) for key, item in items if key in args.models]
        for future in futures:
            future.result()
    D.log("Selected minimal inference payloads all passed original checksum validation")
    if args.author_drive:
        import gdown
        from convert_author_weights import PINS, main as convert
        weights=args.runtime/"weights"
        weights.mkdir(parents=True,exist_ok=True)
        conversion=weights/"conversion-receipt.json"
        known=json.loads(conversion.read_text()) if conversion.exists() else {}
        for name,pin in PINS.items():
            if name in known and Path(known[name]["path"]).is_file():
                continue  # convert() verifies the exact saved tensor payload SHA.
            gdown.download(id=pin["file_id"],output=str(weights/name),quiet=True,resume=True)
        if args.runtime.resolve() != (ROOT/"dl/retrieval-matrix-extra").resolve():
            raise ValueError("Drive conversion uses the fixed official cache root")
        convert()


if __name__ == "__main__":
    main()
