"""[codex] Build reviewable backend specs; does not run any model."""
from pathlib import Path
import argparse
import hashlib
import json
from importlib.metadata import version

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "dl/retrieval-matrix-rerank"
PINS = {
    "core2": ("Alibaba-NLP/core-reranker-2b", "64e1a36cd6ecace459a4c7cd8f7d0ef5d6fee30b", "swift_released"),
    "core8": ("Alibaba-NLP/core-reranker-8b", "d95d36a34300e34b87cc1793827ac798642c4488", "swift_released"),
    "qwen2": ("Qwen/Qwen3-VL-Reranker-2B", "4bd860ac4f15ad1897a214615cccc700f8f71818", "swift_released"),
    "qwen8": ("Qwen/Qwen3-VL-Reranker-8B", "b212dc8c91a8164aef1ea2de9c1a867611e75c04", "swift_released"),
    "qwen_native2": ("Qwen/Qwen3-VL-Reranker-2B", "4bd860ac4f15ad1897a214615cccc700f8f71818", "qwen_native"),
    "qwen_native8": ("Qwen/Qwen3-VL-Reranker-8B", "b212dc8c91a8164aef1ea2de9c1a867611e75c04", "qwen_native"),
    "jina_m0": ("jinaai/jina-reranker-m0", "94bfe0aeb2d4dd7978362699cddd5893d4e0adc8", "jina_native"),
    "qwen_instruct8": ("Qwen/Qwen3-VL-8B-Instruct", "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b", "mcmr_pointwise"),
}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=tuple(PINS), default=["core2", "core8"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reviewed", action="store_true", help="Only set after independent backend/protocol review")
    args = parser.parse_args()
    import swift
    swift_root = Path(swift.__file__).parent
    swift_sources = {str(path.resolve()): sha(path) for path in sorted(swift_root.rglob("*.py"))}
    specs = {}
    for key in args.models:
        repo, rev, backend = PINS[key]
        snapshot = ROOT / "hf_home/hub" / ("models--" + repo.replace("/", "--")) / "snapshots" / rev
        files = {path.relative_to(snapshot).as_posix(): sha(path) for path in sorted(snapshot.rglob("*"))
                 if path.is_file() and not path.name.endswith((".download-partial", ".lock", ".pending"))}
        if not any(name.endswith((".safetensors", ".bin")) for name in files):
            raise ValueError("Selected checkpoint weights have not completed acquisition")
        if not key.startswith("core"):
            manifest = json.loads((RUNTIME / "acquisition/asset-manifest.json").read_text())["models"]
            m = next(value for value in manifest.values() if value["repo"] == repo and value["revision"] == rev)
            if {row["file"] for row in m["files"]} != set(files):
                raise ValueError("New checkpoint inference files are not all acquired")
            receipt = json.loads((RUNTIME / "acquisition/public-download-receipt.json").read_text())["files"]
            if any(receipt.get(str(snapshot/name), {}).get("sha256") != digest for name, digest in files.items()):
                raise ValueError("Missing payload acquisition receipt")
        spec = {"repo": repo, "revision": rev, "snapshot": str(snapshot), "files": files,
                "backend": backend, "verified_protocol": args.reviewed}
        if backend == "swift_released":
            spec.update(swift_sources=swift_sources, max_batch_size=32)
        elif backend == "qwen_native":
            spec["source"] = source(ROOT / "hf_home/tmp/benchmark-sources-20261116/Qwen3-VL-Embedding--src__models__qwen3_vl_reranker.py")
            spec["processor"] = {"max_length": 10240, "min_pixels": 4096, "max_pixels": 1843200}
        elif backend == "mcmr_pointwise":
            spec["source"] = source(RUNTIME / "sources/mcmr/eval/rerank/qwen3_pointwise.py")
        specs[key] = spec
    packages = {name: version(name) for name in ["torch", "torchvision", "transformers", "tokenizers",
                "ms-swift", "peft", "qwen-vl-utils", "flash-attn", "numpy", "datasets", "Pillow"]}
    value = {"owner": "[codex]", "schema": 1, "models": specs,
             "environment": {"python": str(RUNTIME / "env/bin/python"), "packages": packages,
                             "install_report": source(RUNTIME / "install-report.json"),
                             "frozen_pip": source(RUNTIME / "requirements.freeze.txt")},
             "source_files": {path.name: sha(path) for path in sorted(Path(__file__).parent.glob("*.py"))},
             "production_guard": "Parent must bind this spec, datasets, preregistration, full source hashes and a clean committed freeze before any GPU inference."}
    if args.out.exists():
        raise ValueError("Refusing to overwrite an existing backend specification")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"out": str(args.out), "sha256": sha(args.out), "models": args.models,
                      "reviewed": args.reviewed}))


if __name__ == "__main__":
    main()
