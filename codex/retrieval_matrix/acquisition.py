#!/usr/bin/env python3
"""[codex] Acquire pinned official P0 checkpoints with payload checksums.

Metadata is written before transfer; existing files are verified, never replaced.
No model is imported or executed. Network access is unauthenticated urllib.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codex/retrieval3090"))
import download_assets as D

PINS = {
    "qwen3vl_embed2": ("Qwen/Qwen3-VL-Embedding-2B", "9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda"),
    "qwen3vl_embed8": ("Qwen/Qwen3-VL-Embedding-8B", "2c4565515e0f265c6511776e7193b22c0968ddc7"),
    "gme2": ("Alibaba-NLP/gme-Qwen2-VL-2B-Instruct", "9cfa6413f704a7c1cf5064d240748e10c876b286"),
    "vlm2vec_v2": ("VLM2Vec/VLM2Vec-V2.0", "e39ff079b8275ef876d3656da8c0bddbff3c4dde"),
    "vlm2vec_base": ("Qwen/Qwen2-VL-2B-Instruct", "895c3a49bc3fa70a340399125c650a463535e71c"),
}
SMALL = {
    "config.json", "preprocessor_config.json", "video_preprocessor_config.json",
    "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "chat_template.json",
    "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt",
    "generation_config.json", "adapter_config.json", "README.md",
}


def manifest(keys, runtime):
    items, models = [], {}
    for key in keys:
        repo, rev = PINS[key]
        endpoint = f"https://huggingface.co/api/models/{repo}/revision/{rev}?blobs=true"
        with D.request(endpoint) as stream:
            meta = json.load(stream)
        if meta["sha"] != rev:
            raise ValueError(f"Revision drift: {repo}")
        runtime.joinpath(key + "--api.json").write_text(json.dumps(meta, indent=2) + "\n")
        files = [row["rfilename"] for row in meta["siblings"]
                 if row["rfilename"] in SMALL
                 or row["rfilename"].endswith(".safetensors")
                 or row["rfilename"].endswith(".safetensors.index.json")
                 or row["rfilename"] == "adapter_model.bin"]
        rows = list(D.hf_items("models", repo, rev, files, D.HF_HOME))
        models[key] = {"repo": repo, "revision": rev, "files": rows,
                       "bytes": sum(x["expected_size"] for x in rows),
                       "snapshot": str(Path(rows[0]["path"]).parent)}
        items.extend(rows)
    value = {"owner": "[codex]", "schema": 1,
             "scope": "official public P0 model acquisition; no GPU inference",
             "models": models, "bytes": sum(x["expected_size"] for x in items)}
    runtime.joinpath("asset-manifest.json").write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return value, items


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=tuple(PINS), default=list(PINS))
    parser.add_argument("--runtime", type=Path, default=ROOT / "dl/retrieval-matrix/acquisition")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    args.runtime.mkdir(parents=True, exist_ok=True)
    value, items = manifest(args.models, args.runtime)
    D.log(f"Pinned {len(items)} files: {value['bytes'] / 2**30:.3f} GiB")
    if args.metadata_only:
        return
    receipt = args.runtime / "public-download-receipt.json"
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in [pool.submit(D.download, item, receipt) for item in items]:
            future.result()
    D.log("All selected public model files passed checksum validation")


if __name__ == "__main__":
    main()
