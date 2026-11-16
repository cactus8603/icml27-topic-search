"""[codex] Checksum-verified fixed public reranker checkpoint acquisition."""
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
    "qwen_rerank2": ("Qwen/Qwen3-VL-Reranker-2B", "4bd860ac4f15ad1897a214615cccc700f8f71818"),
    "qwen_rerank8": ("Qwen/Qwen3-VL-Reranker-8B", "b212dc8c91a8164aef1ea2de9c1a867611e75c04"),
    "jina_m0": ("jinaai/jina-reranker-m0", "94bfe0aeb2d4dd7978362699cddd5893d4e0adc8"),
    "qwen_instruct8": ("Qwen/Qwen3-VL-8B-Instruct", "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=tuple(PINS), default=list(PINS))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--runtime", type=Path, default=ROOT / "dl/retrieval-matrix-rerank/acquisition")
    args = parser.parse_args()
    runtime = args.runtime.resolve()
    runtime.mkdir(parents=True, exist_ok=True)
    original_log = D.log
    def log(message):
        original_log(message)
        with D.PRINT_LOCK, (runtime / "download.log").open("a") as stream:
            stream.write("[codex] " + message + "\n")
    D.log = log
    items, models = [], {}
    for key in args.models:
        repo, rev = PINS[key]
        with D.request(f"https://huggingface.co/api/models/{repo}/revision/{rev}?blobs=true") as stream:
            metadata = json.load(stream)
        if metadata["sha"] != rev:
            raise ValueError("Official revision drift")
        (runtime / (key + "--api.json")).write_text(json.dumps(metadata, indent=2) + "\n")
        names = [x["rfilename"] for x in metadata["siblings"]
                 if x["rfilename"].endswith((".json", ".jinja", ".py", ".txt", ".safetensors"))
                 or x["rfilename"] == "README.md"]
        rows = list(D.hf_items("models", repo, rev, names, D.HF_HOME))
        for row in rows:
            row["source_urls"] = [url + "?download=true" for url in row["source_urls"]] + row["source_urls"]
        models[key] = {"repo": repo, "revision": rev, "files": rows,
                       "bytes": sum(x["expected_size"] for x in rows),
                       "snapshot": str(Path(rows[0]["path"]).parent)}
        items.extend(rows)
    (runtime / "asset-manifest.json").write_text(json.dumps({"owner": "[codex]", "models": models,
        "bytes": sum(x["expected_size"] for x in items)}, indent=2, sort_keys=True) + "\n")
    log(f"Pinned {len(items)} inference files: {sum(x['expected_size'] for x in items) / 2**30:.3f} GiB")
    if args.metadata_only:
        return
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in [pool.submit(D.download, x, runtime / "public-download-receipt.json") for x in items]:
            future.result()
    log("All reranker inference assets passed payload checksum validation")


if __name__ == "__main__":
    main()
