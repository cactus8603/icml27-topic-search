"""[codex] Ephemeral CPU fixtures for the ORIGINAL native qualification facades.

These fixtures are not experimental evidence.  They live only in hf_home/tmp
and disappear on context exit.  No real model, production score, receipt, or
scheduler is consulted.  ``patch_manifest`` explicitly describes the metadata
boundaries replaced by each fixture.  Numeric scorers, qualification functions,
runtime checks, checkpoint validation, and evidence hashing remain unmodified.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from unittest.mock import patch

import numpy as np

from codex.retrieval_matrix_native_p0 import evidence as E

ROOT = E.ROOT
REFS = {
    "siglip2": "f3f787eeb193aad8dcc6f2ee43e9609ddf5b47eb",
    "paired": "5675ef59cee99fa6d28e12f07501eeb157c321bd",
    "reranker": "54f544ce54bca82be3cc8b3c72662b354c6d3e40",
}
PLANS = {
    "siglip2": "codex/retrieval_matrix_native_runtime_fix/p0-siglip2-einops-approved-v1.json",
    "paired": "codex/retrieval_matrix_paired_runtime_fix/p0-clic_b32-einops-approved-v1.json",
    "reranker": "codex/retrieval_matrix_rerank_native_p0/plan.native-mcq.approved.v1.json",
}


def overwrite(path, value):
    """For synthetic mutation tests only, never for an experimental artifact."""
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT / "hf_home/tmp"):
        raise ValueError("Synthetic writes are restricted to hf_home/tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def frozen_plan(family):
    path = ROOT / PLANS[family]
    original = subprocess.check_output(["git", "-C", str(ROOT), "show", REFS[family] + ":" + PLANS[family]])
    if original != path.read_bytes():
        raise ValueError("Synthetic fixture must start from the exact frozen approved plan")
    return E.read(path)


def checkpoint(path, values, items, batch_size, identity):
    """Write complete synthetic checkpoint bytes in the ORIGINAL state schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, values, allow_pickle=False)
    state = {
        "fingerprint": E.digest({"identity": identity, "items": items, "batch_size": batch_size}),
        "completed": len(items), "array_shape": list(values.shape), "dtype": values.dtype.str,
        "sha256": E.sha(path), "active_seconds": 0.0,
        "batches": [{"start": start, "end": min(start + batch_size, len(items)),
                     "sha256": hashlib.sha256(np.ascontiguousarray(values[start:start + batch_size]).tobytes()).hexdigest()}
                    for start in range(0, len(items), batch_size)],
    }
    state_path = path.with_suffix(".state.json")
    overwrite(state_path, state)
    return E.entry(path), E.entry(state_path)


@dataclass
class Fixture:
    family: str
    root: Path
    Q: object
    plan_path: Path
    model_spec: dict
    reports: dict = field(default_factory=dict)
    out: Path | None = None
    key: str | None = None
    patch_manifest: dict = field(default_factory=dict)
    bundles: dict = field(default_factory=dict)
    arrays: dict = field(default_factory=dict)

    @property
    def review_path(self):
        return self.root / "SYNTHETIC-review.json"

    @property
    def receipt_path(self):
        return self.root / "SYNTHETIC-qualification.json"

    def review(self, reviewer="synthetic-independent-test-actor"):
        if self.family == "reranker":
            return self.Q.review(self.plan_path, self.key, self.out, reviewer)
        return self.Q.review(self.plan_path, self.reports, reviewer, accept_protocol=True)

    def make(self):
        if self.family == "reranker":
            return self.Q.make(self.plan_path, self.key, self.out, self.review_path)
        return self.Q.make(self.plan_path, self.reports, self.review_path)

    def qualify(self):
        """Actually call the frozen family's review -> make -> admit facade."""
        review = self.review()
        overwrite(self.review_path, review)
        receipt = self.make()
        overwrite(self.receipt_path, receipt)
        admitted = self.Q.admit(E.entry(self.receipt_path), self.model_spec)
        return review, receipt, admitted


def _siglip_bundle(name, spec, dim):
    # All six captions survive, including unequal caption counts per image.
    positives = [0, 0, 1, 2, 2, 2]
    images = np.eye(3, dim, dtype=np.float32)
    texts = images[positives].copy()
    gallery = [{"id": str(i), "path": f"SYNTHETIC-image-{name}-{i}"} for i in range(3)]
    queries = [{"id": str(i), "text": f"SYNTHETIC-caption-{name}-{i}"} for i in range(6)]
    labels = [{"id": str(i), "positive_image_id": str(p)} for i, p in enumerate(positives)]
    counts = {"images": 3, "texts": 6, "excluded_images": 0, "excluded_texts": 0,
              "caption_count_histogram": {"1": 1, "2": 1, "3": 1}}
    identity = {"benchmark_id": spec["benchmark_id"], "spec_sha256": E.digest(spec),
                "gallery_ids": [r["id"] for r in gallery], "query_ids": [r["id"] for r in queries],
                "labels_sha256": E.digest(labels), "counts": counts, "synthetic": True}
    return {"gallery": gallery, "queries": queries, "labels": labels, "counts": counts, "identity": identity}, images, texts


def _paired_bundle(name, spec, dim):
    gallery, queries, labels, rows, vectors = [], [], [], [], []
    e0, e1 = np.eye(2, dim, dtype=np.float32)
    for category in spec["category_order"]:
        for index in range(spec["expected_subsets"][category]):
            qid = f"{category}:{index}"
            good = index % 2 == 0
            # Equal positive/negative scalars are an intentional strict-tie failure.
            options = [e0, e0 if good else e1, e1 if good else e0] if name == "sugarcrepe_pp" else [e0, e1 if good else e0]
            start = len(queries)
            gallery.append({"id": qid, "path": "SYNTHETIC-image-" + qid})
            for role, vector in enumerate(options):
                queries.append({"id": qid + ":" + str(role), "text": "SYNTHETIC-caption-" + qid + ":" + str(role)})
                vectors.append(vector)
            labels.append({"id": qid, "category": category, "text_start": start,
                           "text_options": len(options), "benchmark": name})
            rows.append({"id": qid, "synthetic": True})
    counts = {"images": len(gallery), "texts": len(queries), "paired_examples": len(labels),
              "unique_images": len(gallery), "excluded_images": 0, "excluded_texts": 0,
              "subsets": deepcopy(spec["expected_subsets"])}
    identity = {"benchmark_id": spec["benchmark_id"], "spec_sha256": E.digest(spec),
                "gallery_ids": [r["id"] for r in gallery], "query_ids": [r["id"] for r in queries],
                "labels_sha256": E.digest(labels), "rows_sha256": E.digest(rows), "counts": counts, "synthetic": True}
    images = np.tile(e0, (len(gallery), 1))
    return {"gallery": gallery, "queries": queries, "labels": labels, "rows": rows, "counts": counts, "identity": identity}, images, np.array(vectors, dtype=np.float32)


def _retrieval(family, folder, stack):
    if family == "siglip2":
        from codex.retrieval_matrix_native_runtime_fix import qualification as Q, contracts as C
        engine = Q.Q
    else:
        from codex.retrieval_matrix_paired_runtime_fix import qualification as Q, contracts as C
        from codex.retrieval_matrix_paired_p0 import acquire, build_plan
        engine = Q.ENGINE.ENGINE
    real = frozen_plan(family)
    parent = deepcopy(C.parent() if family == "siglip2" else C.parent(real["model_key"]))
    plan = deepcopy(real)
    generated = {}
    tiny_counts = {name: 2 + i % 2 for i, name in enumerate(acquire.CATEGORIES)} if family == "paired" else {}
    for name, bench in plan["benchmarks"].items():
        ds = bench["data"]
        if family == "siglip2":
            ds["expected_images"], ds["expected_texts"] = 3, 6
            bundle, images, texts = _siglip_bundle(name, ds, plan["model_spec"]["embedding_dim"])
            actual = engine.metrics.independent_metrics(images, texts, engine.data.positive_indices(bundle))
        else:
            ds["expected_subsets"] = {cat: tiny_counts[cat] for cat in ds["category_order"]}
            ds["expected_images"] = sum(ds["expected_subsets"].values())
            ds["expected_texts"] = ds["expected_images"] * (3 if name == "sugarcrepe_pp" else 2)
            bundle, images, texts = _paired_bundle(name, ds, plan["model_spec"]["embedding_dim"])
            actual = engine.metrics.independent_metrics(images, texts, bundle["labels"])
        for key, desc in bench["primary_metrics"].items():
            desc["target"] = actual[key]
            if family == "siglip2":
                desc["denominator"] = 6 if desc["direction"] == "t2i" else 3
            else:
                desc["subset_denominators"] = {cat: tiny_counts[cat] for cat in desc["subsets"]}
                desc["denominator"] = sum(desc["subset_denominators"].values())
        generated[name] = (bundle, images, texts, actual)
    # Only the fixture's benchmark metadata and targets replace the canonical parent.
    parent["benchmarks"] = deepcopy(plan["benchmarks"])
    if family == "siglip2":
        stack.enter_context(patch.object(C, "parent", lambda: deepcopy(parent)))
    else:
        stack.enter_context(patch.object(C, "parent", lambda key: deepcopy(parent)))
        projected = deepcopy(parent)
        targets = deepcopy(build_plan.TARGETS)
        for name, bench in plan["benchmarks"].items():
            targets[plan["model_key"]][name] = {key: d["target"] for key, d in bench["primary_metrics"].items()}
        stack.enter_context(patch.object(acquire, "EXPECTED", tiny_counts))
        stack.enter_context(patch.object(build_plan, "TARGETS", targets))
        stack.enter_context(patch.object(build_plan, "build", lambda key: deepcopy(projected)))
    pp = folder / "SYNTHETIC-plan.json"
    overwrite(pp, plan)
    frozen_payload = pp.read_bytes()
    original_check_output = subprocess.check_output
    def lookup(argv, *args, **kwargs):
        if argv == ["git", "-C", str(ROOT), "show", REFS[family] + ":" + str(pp.relative_to(ROOT))]:
            return frozen_payload
        return original_check_output(argv, *args, **kwargs)
    stack.enter_context(patch.object(subprocess, "check_output", lookup))
    def load(spec, **kwargs):
        for bundle, _, _, _ in generated.values():
            if bundle["identity"]["spec_sha256"] == E.digest(spec):
                return deepcopy(bundle)
        raise ValueError("Synthetic data lookup received a changed descriptor")
    stack.enter_context(patch.object(engine.data, "load", load))
    f = Fixture(family, folder, Q, pp, deepcopy(plan["model_spec"]))
    f.patch_manifest = {"synthetic": True, "original_plan": E.entry(ROOT / PLANS[family]), "original_commit": REFS[family],
        "patched_boundaries": ["canonical parent benchmark data/targets/denominators", "data.load: synthetic complete bundle",
                               "git show: this one synthetic plan path only"] +
                              (["paired canonical build/TARGETS/EXPECTED metadata"] if family == "paired" else []),
        "unchanged": ["model_spec", "source_files", "dependencies and actual runtime modules", "settings",
                      "Q.review/make/admit", "metrics", "verify_report", "verify_checkpoint", "hash validation"]}
    dependencies = E.read(E.checked(plan["dependencies"]))
    for name, (bundle, images, texts, actual) in generated.items():
        out = folder / name
        runtime = {"modules": deepcopy(dependencies["modules"]), "python_executable": dependencies["modules"]["numpy"]["path"].split("/lib/")[0] + "/bin/python",
                   "matmul": {"allow_tf32": False, "cudnn_allow_tf32": False, "float32_precision": "highest"},
                   "gpu": {"assigned_uuid": "SYNTHETIC-NO-GPU", "central_task": "SYNTHETIC-NO-SCHEDULER"},
                   "runtime_fix": C.runtime_identity(plan)}
        if family == "paired":
            runtime["paired_backend"] = {"cudnn_benchmark": True}
        identity = {"plan_sha256": E.sha(pp), "plan_file": E.entry(pp), "model": plan["model_key"],
                    "model_spec_sha256": E.digest(plan["model_spec"]), "benchmark": name,
                    "benchmark_spec_sha256": E.digest(plan["benchmarks"][name]), "producer": "SYNTHETIC-producer",
                    "freeze_commit": REFS[family], "source_files": plan["source_files"], "settings": plan["settings"],
                    "data_identity": bundle["identity"], "runtime": runtime, "model_signature": {"synthetic": True}}
        image_items, text_items = engine.data.inference_items(bundle)
        arrays, states = {}, {}
        for modality, values, items in (("images", images, image_items), ("texts", texts, text_items)):
            arrays[modality], states[modality] = checkpoint(out / (modality + ".npy"), values, items,
                plan["settings"]["image_batch" if modality == "images" else "text_batch"], {**identity, "modality": modality})
        smoke = {"state": "PASS", "identity": {k: identity[k] for k in ("plan_sha256", "model", "model_spec_sha256", "source_files", "plan_file")},
                 "freeze_commit": REFS[family], "runtime": runtime, "model_signature": identity["model_signature"],
                 "checks": {modality: {"cosine": 1., "max_abs": 0., "passed": True} for modality in ("images", "texts")}}
        overwrite(out / "SYNTHETIC-smoke.json", smoke)
        index = {"state": "COMPLETE", "identity": identity, "arrays": arrays, "checkpoints": states,
                 "counts": bundle["counts"], "research_metrics": False, "smoke": E.entry(out / "SYNTHETIC-smoke.json")}
        report = {"state": "COMPLETE", "producer": identity["producer"], "identity": identity,
                  "score_profile": plan["benchmarks"][name]["score_profile"], "primary_metric_descriptors": plan["benchmarks"][name]["primary_metrics"],
                  "data_identity": bundle["identity"], "counts": bundle["counts"], "arrays": arrays, "metrics": actual}
        if family == "paired":
            overwrite(out / "SYNTHETIC-embedding.json", index)
            index["embedding_receipt"] = E.entry(out / "SYNTHETIC-embedding.json")
            scores = engine.metrics.score_rows(images, texts, bundle["labels"], device="cpu")
            score_entry, score_state = checkpoint(out / "author-scores.npy", scores,
                [r["id"] for r in bundle["labels"]], 1, {**identity, "modality": "author_scalar_scores"})
            index.update(author_scores=score_entry, author_scores_checkpoint=score_state)
            report["author_scores"] = score_entry
            report["metrics"] = engine.metrics.from_scores(scores, bundle["labels"])
        overwrite(out / "SYNTHETIC-index.json", index)
        report["index_receipt"] = E.entry(out / "SYNTHETIC-index.json")
        report_path = out / "SYNTHETIC-report.json"
        overwrite(report_path, report)
        f.reports[name], f.bundles[name], f.arrays[name] = report_path, bundle, arrays
    return f


def _mcq(folder, stack, key):
    from codex.retrieval_matrix_rerank_native_p0 import contracts as C, qualification as Q, runner as N
    plan = frozen_plan("reranker")
    variant = plan["variants"][key]
    # Full source denominator; IDs/strings are synthetic, never original examples.
    rows, values = [], {}
    for subset, count in C.COUNTS.items():
        target = variant["targets"]["COLA" if subset == "cola" else "NegBench" if subset.endswith("_mcq") else "SugarCrepe++"]["target"]
        hits = round(count * target)
        for i in range(count):
            width = 4 if subset == "cola" or subset.endswith("_mcq") else 3
            good = i < hits
            scores = ([.75, .25, .25, .75] if subset == "cola" else [.75, .75, .25] if width == 3 else [.75, .25, .25, .25]) if good else [.5] * width
            if subset == "cola" and i == hits:
                # Exactly one row distinguishes image-choice from text-choice;
                # it remains a failure on this fixture's declared native axis.
                scores = [.3, .2, .4, .5] if variant["cola_rule"] == "cola_image_choice" else [.3, .4, .2, .5]
            pairs = [{"id": f"SYNTHETIC-{subset}:{i}:{j}", "image": "SYNTHETIC-NO-IMAGE", "query": "SYNTHETIC-NO-CAPTION"} for j in range(width)]
            values.update({p["id"]: float(s) for p, s in zip(pairs, scores)})
            rows.append({"id": f"SYNTHETIC-{subset}:{i}", "subset": subset,
                         "task": variant["cola_rule"] if subset == "cola" else "negbench_mcq" if subset.endswith("_mcq") else "scpp",
                         "instruction": "Determine if the following image is relevant to the query." + ("" if subset == "cola" else " "), "pairs": pairs})
    def loader(actual_plan, actual_variant, **kwargs):
        if actual_plan != plan or actual_variant != variant:
            raise ValueError("Synthetic MCQ loader received a changed frozen plan/model")
        return deepcopy(rows)
    stack.enter_context(patch.object(C, "load_examples", loader))
    out = folder / "SYNTHETIC-scores"
    (out / "raw").mkdir(parents=True)
    identity = {"producer": "SYNTHETIC-producer", "variant": key, "variant_spec": variant,
                "plan_sha256": E.sha(ROOT / PLANS["reranker"]), "freeze_commit": REFS["reranker"],
                "source_files": plan["source_files"], "data_reference": plan["data_reference"], "synthetic": True}
    overwrite(out / "identity.json", identity)
    backend = {"synthetic": True, "model_spec_sha256": E.digest(variant["model"]), "weights_loaded": False, "cuda_used": False}
    overwrite(out / "backend.json", backend)
    backend_sha, identity_sha = E.sha(out / "backend.json"), E.digest(identity)
    subsets, raw_files, events = {}, {}, []
    for subset in C.COUNTS:
        examples = [r for r in rows if r["subset"] == subset]
        array = np.array([[values[p["id"]] for p in r["pairs"]] for r in examples], dtype=np.float64)
        checkpoint(out / (subset + ".npy"), array, examples, 32, {**identity, "subset": subset})
        subsets[subset] = C.bound_array(out / (subset + ".npy"), examples, {**identity, "subset": subset})
        for call_index, group in enumerate(C.call_groups(examples, variant)):
            ids = [p["id"] for p in group]
            call = f"SYNTHETIC-{subset}-{call_index}"
            responses = [{"id": p["id"], "score": values[p["id"]],
                          "official_raw": [values[p["id"]]] if variant["model"]["backend"] == "swift_released" else values[p["id"]]} for p in group]
            raw_name = E.digest(ids) + ".json"
            overwrite(out / "raw" / raw_name, {"identity_sha256": identity_sha, "pair_ids": ids,
                "instruction": examples[0]["instruction"], "responses": responses, "backend_sha256": backend_sha,
                "call": call, "seconds": 0., "memory": {"allocated": None, "reserved": None}, "synthetic": True})
            raw_sha = E.sha(out / "raw" / raw_name)
            raw_files[raw_name] = raw_sha
            events += [{"state": "STARTED", "call": call, "pairs": len(group), "pair_ids": ids, "forward_budget": 1},
                       {"state": "COMPLETE", "call": call, "seconds": 0., "raw_sha256": raw_sha,
                        "memory": {"allocated": None, "reserved": None}}]
    (out / "calls.jsonl").write_text("".join(json.dumps(e, sort_keys=True) + "\n" for e in events))
    complete = {"state": "COMPLETE", "identity": identity, "subsets": subsets, "raw_files": raw_files,
                "backend_sha256": backend_sha, "calls_sha256": E.sha(out / "calls.jsonl"), "costs": N.cost_summary(out),
                "examples": 15912, "distinct_pairs": 58891, "original_images": 9425}
    overwrite(out / "complete.json", complete)
    scores = [{"id": p["id"], "score": values[p["id"]]} for r in rows for p in r["pairs"]]
    report, _ = C.evaluate(rows, scores, variant)
    report.update(identity=identity, complete_sha256=E.sha(out / "complete.json"), native_cola_rule=variant["cola_rule"])
    report["qualification"] = C.assess(report, variant)
    overwrite(out / "results/report.json", report)
    f = Fixture("reranker", folder, Q, ROOT / PLANS["reranker"], deepcopy(variant["model"]), out=out, key=key)
    f.patch_manifest = {"synthetic": True, "original_plan": E.entry(f.plan_path), "original_commit": REFS["reranker"],
        "patched_boundaries": ["data.load_examples: full 15912-row/58891-pair synthetic bundle"],
        "unchanged": ["approved plan bytes", "all model/variant/source/runtime references", "source_gate/validate",
                      "Q.review/make/admit", "independent_metrics/evaluate/assess", "bound_array/verify_complete",
                      "call_groups", "cost_summary", "hash/scalar/dtype/count validation"]}
    return f


@contextmanager
def fixture(family, variant=None):
    """Yield a synthetic fixture; erase every array/receipt on context exit."""
    if family not in REFS:
        raise ValueError("Unknown synthetic family")
    if family == "reranker" and variant not in {None, "qwen2", "jina_m0"}:
        raise ValueError("Synthetic full MCQ fixtures cover the two distinct native COLA rules")
    import torch
    if torch.cuda.is_initialized():
        raise ValueError("Synthetic fixture refuses an already initialized CUDA process")
    with tempfile.TemporaryDirectory(dir=ROOT / "hf_home/tmp", prefix="native-analysis-full-synthetic-") as name, ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}))
        folder = Path(name)
        yield _mcq(folder, stack, variant or "qwen2") if family == "reranker" else _retrieval(family, folder, stack)
