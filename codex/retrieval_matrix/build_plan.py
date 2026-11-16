"""[codex] Bind P0 official sources, complete public data and model payloads."""
from __future__ import annotations

import argparse
from importlib.metadata import version
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
MODEL_KEYS = {"core2": "core_embed_2b", "core8": "core_embed_8b",
              "qwen2": "qwen3vl_embed_2b", "qwen8": "qwen3vl_embed_8b"}
SOURCE_FILES = ["runner.py", "models.py", "metrics.py", "build_plan.py", "run.sh",
                "check_dependencies.py", "PREREG.md", "official_protocols.json"]


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def entry(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path)}


def dependencies():
    roots = [ROOT / "dl/retrieval-matrix/deps/core-transformers457",
             ROOT / "dl/retrieval-matrix/deps/core-p0"]
    files = {}
    for root in roots:
        if not root.is_dir():
            raise ValueError("The isolated official dependency target is absent")
        for path in sorted(root.rglob("*")):
            if path.is_file() and not path.name.endswith(".pyc"):
                files[str(path.relative_to(ROOT))] = sha(path)
    packages = {name: version(name) for name in ["torch", "torchvision", "transformers", "tokenizers",
                "huggingface-hub", "qwen-vl-utils", "av", "pytrec-eval-terrier", "numpy", "Pillow", "accelerate"]}
    return {"pythonpath": list(map(str, roots)), "packages": packages, "files": files,
            "install_reports": [entry(ROOT / "dl/retrieval-matrix/deps" / name) for name in
                                ("core-transformers457-install.json", "core-p0-install.json")]}


def build(keys, output):
    protocols = json.loads((HERE / "official_protocols.json").read_text())
    official = protocols["official_code"]
    official_root = ROOT / official["cache"]
    source_table = {row["path"]: row["sha256"] for row in official["files"]}
    for name, expected in source_table.items():
        if sha(official_root / name) != expected:
            raise ValueError("Official release source differs from verified Git blob")
    proto = protocols["protocols"]["core_qwen3"]
    model_specs = {}
    old = json.loads((ROOT / "codex/retrieval_scaling/plan.json").read_text())["models"]
    for key in keys:
        source = protocols["models"][MODEL_KEYS[key]]
        snapshot = ROOT / source["snapshot"]
        files = {path.name: sha(path) for path in sorted(snapshot.iterdir())
                 if path.is_file() and not path.name.endswith((".download-partial", ".lock", ".pending"))}
        if key.startswith("core"):
            prior = old["embed2" if key == "core2" else "embed8"]
            if not prior["files"].items() <= files.items():
                raise ValueError("Existing CORE model payload differs from earlier verified freeze")
        else:
            receipt = json.loads((ROOT / "dl/retrieval-matrix/acquisition/public-download-receipt.json").read_text())["files"]
            for name, digest in files.items():
                if receipt.get(str(snapshot / name), {}).get("sha256") != digest:
                    raise ValueError("New model lacks a valid checksum-verified acquisition receipt")
        model_specs[key] = {"kind": "official_embedding", "repo": source["repo"], "revision": source["revision"],
            "snapshot": str(snapshot), "files": files, "embedding_dim": source["embedding_dim"],
            "official": {"verified": True, "root": str(official_root), "revision": official["revision"],
                "files": source_table, "wrapper": "models/qwen3_vl/qwen3_vl_wrapper.py",
                "model_class": proto["model_class"], "processor_class": proto["processor_class"],
                "embedding": "embedding.py", "model_name": "vlembed" if key.startswith("core") else "qwen3vl"},
            "loader": {"torch_dtype": "bfloat16"}, "processor": proto["processor_init"],
            "batches": {"image": 4, "text": 32, "cross": 4},
            "batch_deviation": "Official COCO/Flickr use 64; NegBench uses32. Local image4/text32 fixed before metrics; synthetic batch invariance gate.",
            "environment": {name: version(name) for name in ("torch", "transformers", "numpy", "qwen-vl-utils", "Pillow", "tokenizers", "accelerate")},
            "published_targets": source["targets"]}
    literature = ROOT / "dl/retrieval3090/data/literature-development"
    datasets = {}
    for variant, name in [("plain", "coco"), ("negated", "negbench")]:
        csv_name = ("COCO_val_retrieval.csv" if variant == "plain" else
                    "COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true.csv")
        datasets[name] = {"format": "negbench", "split": "public", "expected_gallery": 5000,
            "expected_queries": 25014, "expected_eligible": 25014,
            "empty_caption_policy": "retain-official",
            "panel": entry(literature / "panels" / f"negbench-{variant}-full.json"),
            "official_csv": entry(literature / "raw/negbench" / csv_name),
            "official_evaluation": "coco-bidirectional" if variant == "plain" else "negbench-t2i",
            "similarity_batch": 1000 if variant == "plain" else 25014,
            "use": "published fixed-baseline reproduction only; forbidden for custom method selection"}
    flickr = ROOT / "dl/retrieval-matrix/data/flickr30k_test"
    datasets["flickr"] = {"format": "flickr", "split": "public", "expected_gallery": 1000,
        "expected_queries": 5000, "expected_eligible": 5000, "manifest": entry(flickr / "test_manifest.jsonl"),
        "images_receipt": entry(flickr / "image_receipt.jsonl"), "official_csv": entry(flickr / "test_1k_flickr.csv"),
        "official_evaluation": "coco-bidirectional", "similarity_batch": 1000,
        "use": "CORE-specified HF test1k; exact overlap with standard Karpathy/Entities splits separately audited"}
    declared = {str((HERE / name).relative_to(ROOT)): sha(HERE / name) for name in SOURCE_FILES}
    declared["codex/retrieval_scaling/runner.py"] = sha(ROOT / "codex/retrieval_scaling/runner.py")
    plan = {"schema": 1, "state": "FROZEN", "owner": "[codex]", "phase": "P0",
        "prereg": "codex/retrieval_matrix/PREREG.md", "source_files": declared,
        "scope": "official released implementation; full COCO, NegBench and CORE-specified Flickr; no custom method",
        "models": model_specs, "datasets": datasets,
        "settings": {"top_ks": [50, 100, 200], "main_k": 100, "block_queries": 64,
                     "seed": 2026111671, "batch_invariance_cosine_min": 0.9995,
                     "batch_invariance_max_abs": 0.01},
        "dependencies": dependencies(), "fixed_baseline_test": False,
        "gate": protocols["gate"], "gpu_queue": {"min_free_mib": 20480, "max_utilization": 20},
        "known_differences": protocols["known_source_issues"],
        "not_covered": ["MCMR exact product renderer missing upstream", "GME/VLM2Vec separate environment and adapter gate", "MCQ reranker separate official backend"]}
    if output.exists():
        raise ValueError("Refusing to overwrite an existing frozen plan; choose a new batch/version")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"owner": "[codex]", "plan": str(output), "sha256": sha(output),
                      "models": keys, "datasets": list(datasets)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=tuple(MODEL_KEYS), default=["core2", "core8"])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    build(args.models, args.out)


if __name__ == "__main__":
    main()
