"""[codex] Isolated paired qualification facade over the frozen evidence gate."""
import importlib.util
from importlib.metadata import version
from pathlib import Path

import numpy as np

from codex.retrieval_matrix_native_p0.evidence import ROOT, checked, digest, entry, read, sha, write
from codex.retrieval_matrix_paired_p0 import data, metrics


def private_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ENGINE = private_module("paired_private_native_qualification", ROOT/"codex/retrieval_matrix_native_p0/qualification.py")
ENGINE.data, ENGINE.metrics = data, metrics
_original_verify = ENGINE.verify_report
_original_validate = ENGINE.validate_plan


def validate_plan(plan):
    _original_validate(plan)
    from codex.retrieval_matrix_paired_p0 import acquire, build_plan
    if plan.get("model_key") not in build_plan.TARGETS:
        raise ValueError("Paired native qualification supports only the three exact released CLIC variants")
    canonical = build_plan.build(plan["model_key"])
    if (set(plan["benchmarks"]) != {"sugarcrepe", "sugarcrepe_pp"}
            or {row["canonical_benchmark"] for row in plan["benchmarks"].values()} != {"SugarCrepe", "SugarCrepe++"}
            or plan["model_spec"].get("family") != "CLIP"
            or plan["settings"]["image_batch"] != 1 or plan["settings"]["text_batch"] != 1):
        raise ValueError("Paired P0 requires both complete canonical benchmarks and the author batch1 protocol")
    for name, row in plan["benchmarks"].items():
        if row["score_profile"] != metrics.PROFILE or row["data"]["benchmark_id"] != row["canonical_benchmark"]:
            raise ValueError("Paired metric/benchmark identity differs")
        categories = acquire.CATEGORIES if name == "sugarcrepe" else acquire.CATEGORIES[2:]
        expected_counts = {category: acquire.EXPECTED[category] for category in categories}
        ds = row["data"]
        if (ds["category_order"] != list(categories) or set(ds["subsets"]) != set(categories)
                or ds["expected_subsets"] != expected_counts or ds["expected_images"] != sum(expected_counts.values())
                or ds["expected_texts"] != sum(expected_counts.values())*(2 if name == "sugarcrepe" else 3)):
            raise ValueError("Paired native admission requires every official subset and full original denominator")
        target = build_plan.TARGETS[plan["model_key"]][name]
        if (set(row["primary_metrics"]) != set(target)
                or any(descriptor["target"] != target[key] for key, descriptor in row["primary_metrics"].items())):
            raise ValueError("Paired primary targets must retain this exact published checkpoint variant")
        if row["primary_metrics"] != canonical["benchmarks"][name]["primary_metrics"]:
            raise ValueError("Paired published metric direction, denominator, aggregation and source must match the full author protocol")
    for field in ("model_spec", "benchmarks", "settings", "dependencies", "source_files", "external_sources"):
        if plan[field] != canonical[field]:
            raise ValueError("Paired official plan cannot omit or substitute fixed protocol/source evidence: " + field)
    return plan


def verify_report(plan, plan_sha, name, report, *, verify_images=True):
    torch_runtime(plan)
    bundle, images, texts = _original_verify(plan, plan_sha, name, report, verify_images=verify_images)
    identity = report["identity"]
    if identity["runtime"].get("paired_backend") != {"cudnn_benchmark": True}:
        raise ValueError("Paired actual backend must record the author's cudnn.benchmark=True")
    receipt = read(checked(report["index_receipt"]))
    original = read(checked(receipt["embedding_receipt"]))
    if any(original[key] != receipt[key] for key in ("state", "identity", "arrays", "checkpoints", "counts", "smoke", "research_metrics")):
        raise ValueError("Paired final receipt differs from the unchanged original embedding receipt")
    if report.get("author_scores") != receipt.get("author_scores"):
        raise ValueError("Paired report must bind the original GPU scalar score arrays")
    score_identity = {**identity, "modality": "author_scalar_scores"}
    scores = ENGINE.verify_checkpoint(receipt["author_scores"], receipt["author_scores_checkpoint"],
        [row["id"] for row in bundle["labels"]], 1, score_identity)
    if metrics.from_scores(scores, bundle["labels"]) != report["metrics"]:
        raise ValueError("Paired producer metrics differ from all original GPU scalar rows")
    cpu_scores = metrics.score_rows(images, texts, bundle["labels"], device="cpu")
    if not np.array_equal(metrics.events(cpu_scores, bundle["labels"]), metrics.events(scores, bundle["labels"])):
        raise ValueError("Independent CPU/GPU strict events differ for at least one row; fail closed without changing tie rules")
    return bundle, images, texts


ENGINE.validate_plan, ENGINE.verify_report = validate_plan, verify_report
_review, _make = ENGINE.review, ENGINE.make


def torch_runtime(plan):
    import torch
    actual = {"path": str(Path(torch.__file__).resolve()), "actual_version": torch.__version__,
              "distribution_version": version("torch")}
    if actual != read(checked(plan["dependencies"]))["modules"]["torch"]:
        raise ValueError("Independent paired CPU Torch path/version differs from the fixed scalar-scoring backend")
    return actual


def review(plan_path, reports, reviewer, *, accept_protocol=False):
    plan = validate_plan(read(plan_path))
    actual = torch_runtime(plan)
    value = _review(plan_path, reports, reviewer, accept_protocol=accept_protocol)
    value.update(recomputation_torch=actual, paired_gate=entry(Path(__file__)))
    return value


def make(plan_path, reports, review_path):
    reviewed = read(review_path)
    expected = read(checked(read(plan_path)["dependencies"]))["modules"]["torch"]
    if reviewed.get("recomputation_torch") != expected or reviewed.get("paired_gate") != entry(Path(__file__)):
        raise ValueError("Paired independent review must bind its actual Torch runtime and isolated qualification facade")
    return _make(plan_path, reports, review_path)


ENGINE.review, ENGINE.make = review, make
admit = ENGINE.admit
require_p3_families = ENGINE.require_p3_families


if __name__ == "__main__":
    ENGINE.main()
