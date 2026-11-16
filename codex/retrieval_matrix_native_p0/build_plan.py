"""[codex] Build an executable, unqualified native SigLIP2 P0 plan."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from codex.retrieval_matrix_native_p0.evidence import entry, read, sha, write
from codex.retrieval_matrix_native_p0.metrics import PROFILE

HERE = Path(__file__).resolve().parent
SOURCE_FILES = [
    "codex/retrieval_scaling/runner.py",
    "codex/retrieval_matrix_extra/dependencies.json",
] + ["codex/retrieval_matrix_native_p0/" + name for name in (
    "evidence.py", "data.py", "metrics.py", "models.py", "runner.py", "qualification.py",
    "build_plan.py", "test_native.py", "run.sh", "central_task.sh", "PREREG.md")]


def build():
    registry = read(ROOT/"codex/retrieval_matrix_extra/registry.json")
    model = copy.deepcopy(registry["models"]["siglip2"])
    model["family"] = "CLIP"
    model["qualification"] = "P0_REQUIRED"
    base = ROOT/"dl/retrieval-matrix/data"
    coco = base/"karpathy-original/coco-test"
    flickr = base/"flickr30k_test"
    benchmarks = {}
    source_specs = {
        "coco": {"benchmark_id": "COCO-Karpathy-test5k", "dataset_family": "COCO",
            "format": "karpathy-coco", "source_json": entry(base/"karpathy-original/dataset_coco.json"),
            "manifest": entry(coco/"manifest.jsonl"), "image_receipt": entry(coco/"image-receipt.jsonl"),
            "expected_images": 5000, "expected_texts": 25010,
            "published_dataset_mapping": "Google author issue166 explicitly uses TFDS coco_captions test; fixed TFDS source retains every original Karpathy raw sentence.",
            "mapping_review_required": "Confirm original Karpathy 5000 images and all 25010 captions, without CORE val2017 substitution or first-five truncation."},
        "flickr": {"benchmark_id": "Flickr30k-Karpathy-test1k", "dataset_family": "Flickr30k",
            "format": "karpathy-flickr", "source_json": entry(flickr/"upstream/dataset_flickr30k.json"),
            "manifest": entry(flickr/"karpathy_test_manifest.jsonl"), "image_receipt": entry(flickr/"image_receipt.jsonl"),
            "image_root": str(flickr), "receipt_root": str(base),
            "expected_images": 1000, "expected_texts": 5000,
            "published_dataset_mapping": "Paper Table1 names Flickr. Use original Karpathy test1k raw captions; Google public issue gives COCO loader only, so exact Flickr loader/order is not claimed.",
            "mapping_review_required": "Independently accept or reject the original Karpathy test1k mapping to the published Flickr benchmark; CORE HF captions differ and cannot substitute."},
    }
    for name, data in source_specs.items():
        target = next(row for row in model["published_targets"] if row["benchmark"] == data["benchmark_id"])
        benchmarks[name] = {"canonical_benchmark": data["dataset_family"], "benchmark_id": data["benchmark_id"],
            "task": "bidirectional-caption-retrieval", "data": data,
            "score_profile": copy.deepcopy(PROFILE), "protocol_status": "DISCLOSED_OFFICIAL_PORT_REQUIRES_REVIEW",
            "disclosed_differences": [
                "Google official HF PyTorch weights, FP32, instead of original Google JAX execution.",
                "HF PIL bilinear resize384 vs BigVision tf.image.resize bilinear antialias=False followed by uint8 cast; numerical identity is not asserted.",
                "Original Karpathy source order is frozen; TFDS enumeration can differ, affecting exact similarity ties.",
                "Author NumPy version is not published; this port fixes NumPy 2.3.5 and default quicksort, with no custom tie rule.",
                data["published_dataset_mapping"]],
            "primary_metrics": {key: {"direction": key.split("_")[0], "metric": "any-positive Recall@1",
                "denominator": data["expected_texts"] if key.startswith("t2i") else data["expected_images"],
                "target": value, "absolute_tolerance": .01,
                "published": {"variant": model["variant"], "table": target["table"],
                    "url": "https://arxiv.org/html/2502.14786v1", "source": {"path": target["source_file"], "sha256": target["source_sha256"]}}}
                for key, value in target["metrics"].items()}}
    source_root = ROOT/"dl/retrieval-matrix-native-p0/sources"
    external = {"google_metric": entry(source_root/"image_text_retrieval.py"),
        "google_author_reproduction": entry(source_root/"siglip2-issue166-comments.json"),
        "tfds_coco_builder": entry(ROOT/"dl/retrieval-matrix-extra/sources/tfds/coco_captions.py")}
    for name, digest in model["sources"]["big_vision"]["files"].items():
        external["bigvision/"+name] = {"path": str(Path(model["sources"]["big_vision"]["root"])/name), "sha256": digest}
    return {"schema": 2, "owner": "[codex]", "phase": "P0_NATIVE",
        "purpose": "P0_OFFICIAL_QUALIFICATION", "state": "NOT_RUN", "model_key": "siglip2",
        "model_spec": model, "benchmarks": benchmarks,
        "qualification": {"required_distinct_benchmarks": 2, "counting_unit": "canonical_benchmark dataset family, never metric directions or subcategories",
            "all_declared_benchmarks_retained": True, "requires_separate_reviewer": True,
            "requires_independent_array_recomputation": True, "requires_explicit_protocol_review": True},
        "settings": {"image_batch": 4, "text_batch": 32, "seed": 2026111665,
            "precision": "float32", "quantization": False, "offload": False, "allow_tf32": False,
            "smoke_cosine_min": .9995, "smoke_max_abs": .01},
        "dependencies": entry(ROOT/"codex/retrieval_matrix_extra/dependencies.json"),
        "external_sources": external,
        "source_files": {name: sha(ROOT/name) for name in SOURCE_FILES}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE/"p0-siglip2.json")
    args = parser.parse_args()
    write(args.out, build())
    print(str(args.out.resolve()))
