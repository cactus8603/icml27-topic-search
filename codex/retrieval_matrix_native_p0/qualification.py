"""[codex] Family-aware native P0 evidence, independent review, and admission.

The assessment schema is benchmark/metric based. It has no CORE-only dataset
columns. The current independent recomputation backend supports the declared
Google FP32 retrieval profile; new families need their own audited backend.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version as distribution_version
import math
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from codex.retrieval_matrix_native_p0 import data, metrics
from codex.retrieval_matrix_native_p0.evidence import checked, digest, entry, read, sha, write


def validate_plan(plan):
    if (plan.get("schema") != 2 or plan.get("purpose") != "P0_OFFICIAL_QUALIFICATION"
            or plan.get("phase") != "P0_NATIVE" or plan["model_spec"].get("family") not in {"CLIP", "MLLM"}
            or plan.get("qualification", {}).get("required_distinct_benchmarks") != 2):
        raise ValueError("Native qualification needs explicit model family and two distinct published benchmarks")
    if len(plan["benchmarks"]) < 2:
        raise ValueError("Two directions of one dataset are not two benchmark qualifications")
    if not plan.get("source_files"):
        raise ValueError("Native qualification must bind the complete source SHA table")
    settings = plan.get("settings", {})
    if (settings.get("image_batch") != plan["model_spec"].get("batches", {}).get("image")
            or settings.get("text_batch") != plan["model_spec"].get("batches", {}).get("text")
            or type(settings.get("seed")) is not int or settings.get("allow_tf32") is not False
            or settings.get("precision") != plan["model_spec"].get("loader", {}).get("torch_dtype") or settings.get("quantization") is not False
            or settings.get("offload") is not False):
        raise ValueError("Native settings must bind the actual model batches, seed, FP32, TF32 and full-weight execution")
    for key in ("image_batch", "text_batch"):
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError("Native execution batch must be a positive integer")
    for name, bench in plan["benchmarks"].items():
        if not bench.get("canonical_benchmark") or not bench.get("primary_metrics") or not bench.get("task"):
            raise ValueError("Benchmark identity/task and published metric descriptors must be complete")
        if bench["benchmark_id"] != bench["data"]["benchmark_id"]:
            raise ValueError("Benchmark and native data identity differ")
        for descriptor in bench["primary_metrics"].values():
            if (not descriptor.get("direction") or not descriptor.get("metric")
                    or type(descriptor.get("denominator")) is not int or descriptor["denominator"] < 1
                    or descriptor["published"]["variant"] != plan["model_spec"]["variant"]
                    or not descriptor["published"].get("table") or not descriptor["published"].get("url")
                    or not 0 <= descriptor["target"] <= 1 or not 0 <= descriptor["absolute_tolerance"] <= .01):
                raise ValueError("Metric direction/denominator/variant/source/target/tolerance is invalid")
            if bench["task"] == "bidirectional-caption-retrieval":
                expected_count = {"t2i": bench["data"]["expected_texts"], "i2t": bench["data"]["expected_images"]}
                if (descriptor["direction"] not in expected_count
                        or descriptor["denominator"] != expected_count[descriptor["direction"]]):
                    raise ValueError("Retrieval metric direction has the wrong complete denominator")
            checked(descriptor["published"]["source"])
    if len({bench["canonical_benchmark"] for bench in plan["benchmarks"].values()}) < 2:
        raise ValueError("Subsets or directions of one canonical dataset cannot satisfy two-benchmark admission")
    return plan


def frozen_source_gate(plan, identity):
    commit = identity.get("freeze_commit", "")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Native arrays lack a fixed full source freeze commit")
    plan_path = checked(identity["plan_file"])
    if sha(plan_path) != identity["plan_sha256"]:
        raise ValueError("Native array identity names a different plan payload")
    files = set(plan["source_files"]) | {str(plan_path.relative_to(ROOT))}
    for name in files:
        path = (ROOT/name).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError("Native frozen source escapes the repository")
        original = subprocess.check_output(["git", "-C", str(ROOT), "show", commit+":"+name])
        if original != path.read_bytes():
            raise ValueError("Native source/plan differs from its original GPU freeze commit: " + name)


def verify_checkpoint(array_entry, state_entry, items, batch_size, identity):
    import numpy as np
    value = np.load(checked(array_entry), mmap_mode="r", allow_pickle=False)
    state = read(checked(state_entry))
    if (state.get("fingerprint") != digest({"identity": identity, "items": items, "batch_size": batch_size})
            or state.get("completed") != len(items) or state.get("array_shape") != list(value.shape)
            or state.get("dtype") != value.dtype.str or state.get("sha256") != array_entry["sha256"]):
        raise ValueError("Native checkpoint identity, completion, shape, dtype or full-array SHA differs")
    end = 0
    for row in state["batches"]:
        next_end = min(end+batch_size, len(items))
        if (end >= len(items) or row["start"] != end or row["end"] != next_end
                or row["sha256"] != hashlib.sha256(np.ascontiguousarray(value[end:next_end]).tobytes()).hexdigest()):
            raise ValueError("Native checkpoint original complete-batch provenance is invalid")
        end = next_end
    if end != len(items):
        raise ValueError("Native checkpoint batch receipts do not cover the complete original split")
    return value


def verify_report(plan, plan_sha, name, report, *, verify_images=True):
    import numpy as np
    bench = plan["benchmarks"][name]
    expected = {"plan_sha256": plan_sha, "model": plan["model_key"],
        "model_spec_sha256": digest(plan["model_spec"]), "benchmark": name,
        "benchmark_spec_sha256": digest(bench)}
    identity = report.get("identity", {})
    if any(identity.get(key) != value for key, value in expected.items()):
        raise ValueError("Native report identity differs from the complete fixed plan/spec")
    if report.get("score_profile") != bench["score_profile"] or report.get("primary_metric_descriptors") != bench["primary_metrics"]:
        raise ValueError("Native metric/precision/tie/variant descriptors differ")
    if report.get("state") != "COMPLETE" or not report.get("producer"):
        raise ValueError("Native P0 qualification requires completed producer evidence")
    if identity.get("producer") != report["producer"]:
        raise ValueError("Report producer identity differs from the actual original index producer")
    bundle = data.load(bench["data"], verify_images=verify_images)
    if report.get("data_identity") != bundle["identity"] or report.get("counts") != bundle["counts"]:
        raise ValueError("Native report changed source labels, caption counts, order or complete denominators")
    if identity.get("source_files") != plan["source_files"]:
        raise ValueError("Native report does not bind the complete source SHA table")
    if identity.get("settings") != plan["settings"] or identity.get("data_identity") != bundle["identity"]:
        raise ValueError("Native actual execution settings/source identity differ from their qualified model plan")
    frozen_source_gate(plan, identity)
    for relative, expected_sha in plan["source_files"].items():
        if sha(ROOT/relative) != expected_sha:
            raise ValueError("Frozen native scoring/producer source changed: " + relative)
    for item in plan["external_sources"].values():
        checked(item)
    dependencies = read(checked(plan["dependencies"]))
    actual_runtime = identity.get("runtime", {})
    if (actual_runtime.get("modules") != dependencies["modules"]
            or actual_runtime.get("matmul") != {"allow_tf32": False, "cudnn_allow_tf32": False, "float32_precision": "highest"}
            or not actual_runtime.get("python_executable") or not actual_runtime.get("gpu", {}).get("assigned_uuid")
            or not actual_runtime.get("gpu", {}).get("central_task")):
        raise ValueError("Native actual runtime module paths/versions or FP32/TF32 profile differ")
    index_receipt = read(checked(report["index_receipt"]))
    if (index_receipt.get("state") != "COMPLETE" or index_receipt.get("identity") != identity
            or index_receipt.get("arrays") != report["arrays"] or index_receipt.get("counts") != bundle["counts"]
            or index_receipt.get("research_metrics") is not False):
        raise ValueError("Native report is not bound to the producer's complete original index receipt")
    smoke = read(checked(index_receipt["smoke"]))
    smoke_identity = {key: identity[key] for key in ("plan_sha256", "model", "model_spec_sha256", "source_files", "plan_file")}
    if (smoke.get("state") != "PASS" or smoke.get("identity") != smoke_identity
            or smoke.get("freeze_commit") != identity["freeze_commit"] or smoke.get("runtime") != actual_runtime
            or smoke.get("model_signature") != identity.get("model_signature")):
        raise ValueError("Native index/report runtime, processor or freeze commit differs from its real smoke")
    for name in ("images", "texts"):
        row = smoke["checks"][name]
        if (not math.isfinite(row["cosine"]) or not math.isfinite(row["max_abs"])
                or row["cosine"] < plan["settings"]["smoke_cosine_min"]
                or row["max_abs"] > plan["settings"]["smoke_max_abs"] or row["passed"] is not True):
            raise ValueError("Native successful smoke did not satisfy its actual technical thresholds")
    image_items, text_items = data.inference_items(bundle)
    images = verify_checkpoint(report["arrays"]["images"], index_receipt["checkpoints"]["images"], image_items,
        plan["settings"]["image_batch"], {**identity, "modality": "images"})
    texts = verify_checkpoint(report["arrays"]["texts"], index_receipt["checkpoints"]["texts"], text_items,
        plan["settings"]["text_batch"], {**identity, "modality": "texts"})
    if (images.shape != (bundle["counts"]["images"], plan["model_spec"]["embedding_dim"])
            or texts.shape != (bundle["counts"]["texts"], plan["model_spec"]["embedding_dim"])):
        raise ValueError("Native arrays do not cover the full original gallery/caption denominator")
    metrics.validate_embeddings(images, texts, data.positive_indices(bundle))
    return bundle, images, texts


def reviewer_numpy_runtime(plan):
    import numpy as np
    actual = {"path": str(Path(np.__file__).resolve()), "actual_version": np.__version__,
              "distribution_version": distribution_version("numpy")}
    expected = read(checked(plan["dependencies"]))["modules"]["numpy"]
    if actual != expected:
        raise ValueError("Independent review NumPy import path/version/distribution differs from the fixed dependencies")
    return actual


def review(plan_path, reports, reviewer, *, accept_protocol=False):
    plan = validate_plan(read(plan_path))
    if set(reports) != set(plan["benchmarks"]):
        raise ValueError("Retain every preregistered benchmark, including failures")
    if not reviewer:
        raise ValueError("Independent reviewer identity is required")
    output = {"schema": 2, "scope": "native-p0-independent-recomputation", "owner": "[codex]",
        "reviewer": reviewer, "plan": entry(plan_path), "reports": {}, "benchmarks": {},
        "recomputation_numpy": reviewer_numpy_runtime(plan),
        "recomputation_sources": {"metrics": entry(Path(metrics.__file__)), "qualification": entry(Path(__file__))}}
    for name, path in reports.items():
        report = read(path)
        if reviewer == report.get("producer"):
            raise ValueError("The result producer cannot approve its own qualification")
        bundle, images, texts = verify_report(plan, sha(plan_path), name, report)
        if plan["benchmarks"][name]["score_profile"] != metrics.PROFILE:
            raise ValueError("This precision/ranking family requires a separate audited independent scoring backend")
        recomputed = metrics.independent_metrics(images, texts, data.positive_indices(bundle))
        agrees = report["metrics"].keys() == recomputed.keys() and all(
            math.isfinite(float(report["metrics"][key])) and abs(report["metrics"][key]-value) <= 1e-12
            for key, value in recomputed.items())
        output["reports"][name] = entry(path)
        output["benchmarks"][name] = {"producer": report["producer"], "arrays": report["arrays"],
            "counts": bundle["counts"], "data_identity_sha256": digest(bundle["identity"]),
            "score_profile_sha256": digest(plan["benchmarks"][name]["score_profile"]),
            "metrics": recomputed, "producer_metrics_match": agrees,
            "protocol_review_accepted": bool(accept_protocol),
            "protocol_review_scope": plan["benchmarks"][name]["disclosed_differences"]}
    output["state"] = "PASS" if accept_protocol and all(
        row["producer_metrics_match"] for row in output["benchmarks"].values()) else "REVIEW_REQUIRED"
    return output


def make(plan_path, report_paths, review_path):
    plan = validate_plan(read(plan_path))
    plan_entry = entry(plan_path)
    independent = read(review_path)
    references = {name: entry(path) for name, path in report_paths.items()}
    if (set(references) != set(plan["benchmarks"]) or independent.get("schema") != 2
            or independent.get("scope") != "native-p0-independent-recomputation"
            or independent.get("state") != "PASS" or independent.get("plan") != plan_entry
            or independent.get("reports") != references or not independent.get("reviewer")
            or set(independent.get("benchmarks", {})) != set(plan["benchmarks"])):
        raise ValueError("A separate reviewer must bind and independently recompute all exact reports and arrays")
    if independent.get("recomputation_sources") != {
            "metrics": entry(Path(metrics.__file__)), "qualification": entry(Path(__file__))}:
        raise ValueError("Independent scoring implementation changed or was not recorded")
    if independent.get("recomputation_numpy") != read(checked(plan["dependencies"]))["modules"]["numpy"]:
        raise ValueError("Independent review receipt lacks the fixed actual NumPy path/version/distribution")
    details = {}
    for name, bench in plan["benchmarks"].items():
        report = read(report_paths[name])
        # Revalidate every payload and complete source denominator at admission.
        bundle, _, _ = verify_report(plan, plan_entry["sha256"], name, report)
        audited = independent["benchmarks"][name]
        if (audited.get("producer") != report["producer"] or report["producer"] == independent["reviewer"]
                or audited.get("arrays") != report["arrays"] or audited.get("counts") != bundle["counts"]
                or audited.get("data_identity_sha256") != digest(bundle["identity"])
                or audited.get("score_profile_sha256") != digest(bench["score_profile"])
                or audited.get("producer_metrics_match") is not True
                or audited.get("protocol_review_accepted") is not True
                or audited.get("protocol_review_scope") != bench["disclosed_differences"]
                or audited.get("metrics") != report["metrics"]):
            raise ValueError("Independent evidence does not cover this exact producer/protocol/full-data result")
        comparisons = {}
        for metric, descriptor in bench["primary_metrics"].items():
            observed = float(audited["metrics"][metric])
            if not math.isfinite(observed) or not 0 <= observed <= 1:
                raise ValueError("Native scores must be finite fractions")
            delta = observed-descriptor["target"]
            comparisons[metric] = {"descriptor": descriptor, "observed": observed, "delta_pp": 100*delta,
                "matched": abs(delta) <= descriptor["absolute_tolerance"]+1e-12}
        details[name] = {"canonical_benchmark": bench["canonical_benchmark"],
            "benchmark_spec_sha256": digest(bench), "metrics": comparisons,
            "matched": all(row["matched"] for row in comparisons.values())}
    matched = sorted({row["canonical_benchmark"] for row in details.values() if row["matched"]})
    return {"schema": 2, "owner": "[codex]", "scope": "official-native-p0-qualified",
        "state": "PASS" if len(matched) >= 2 else "FAILED", "model": plan["model_key"],
        "family": plan["model_spec"]["family"], "model_spec_sha256": digest(plan["model_spec"]),
        "p0_plan": plan_entry, "reports": references, "independent_review": entry(review_path),
        "required_distinct_benchmarks": 2, "matched_canonical_benchmarks": matched, "benchmarks": details}


def admit(receipt_entry, model_spec):
    receipt = read(checked(receipt_entry))
    actual = make(checked(receipt["p0_plan"]),
        {name: checked(item) for name, item in receipt["reports"].items()}, checked(receipt["independent_review"]))
    if receipt != actual or actual["state"] != "PASS" or actual["model_spec_sha256"] != digest(model_spec):
        raise ValueError("P2 requires unchanged independently qualified native P0 evidence")
    return actual


def require_p3_families(qualified_rows):
    """Apply after family-specific admission (existing MLLM gate or this gate)."""
    if (len(qualified_rows) != 2 or any(row.get("state") != "PASS" for row in qualified_rows)
            or {row.get("family") for row in qualified_rows} != {"CLIP", "MLLM"}):
        raise ValueError("P3 needs one qualified CLIP baseline and one qualified MLLM baseline")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("review", "qualify"))
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--reports-root", required=True, type=Path)
    parser.add_argument("--reviewer")
    parser.add_argument("--accept-disclosed-protocol", action="store_true",
        help="Independent reviewer has examined and accepts all declared port/dataset differences")
    parser.add_argument("--independent-review", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    plan = validate_plan(read(args.plan))
    reports = {name: args.reports_root/name/"results/report.json" for name in plan["benchmarks"]}
    if args.action == "review":
        result = review(args.plan, reports, args.reviewer, accept_protocol=args.accept_disclosed_protocol)
    else:
        if args.independent_review is None:
            parser.error("qualify needs --independent-review")
        result = make(args.plan, reports, args.independent_review)
    write(args.out, result)
    print(result["state"])


if __name__ == "__main__":
    main()
