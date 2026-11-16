"""[codex] Public-literature development, using frozen image-only model adapters.

The official NegBench gold is the caption's source image. MRR and source-image
Recall are reported; these labels do not enumerate all semantically valid images.
GPU inference and CPU analysis are separate, resumable stages. No policy search.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from time import monotonic

import numpy as np

import mcmr_pilot as M

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCOPE = "public-literature-development-only; not independent formal confirmation"
PANEL_NAMES = tuple(f"negbench-{variant}-{subset}.json"
                    for variant in ("plain", "negated") for subset in ("full", "dev1000"))
STAGES = ("smoke", "index", "analyze", "cross50", "analyze-cross")
METRICS = ("MRR", "R@1", "R@5", "R@10", "R@50")
BATCHES = {"image": 8, "text": 32, "cross": 4}
SOURCES = ("literature_dev.py", "literature_data.py", "mcmr_pilot.py", "cache_v2.py")


def safe(path):
    path = Path(path).expanduser().absolute()
    if any("retrieval-independent-panel" in str(p) for p in (path, path.resolve())):
        raise ValueError("Sealed independent data is forbidden")
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Symlink paths are forbidden")
    return path


def sha(path):
    return M.sha(safe(path))


def read(path):
    return M.read_json(safe(path))


def source_files(require_committed=False):
    files = {str((HERE / name).relative_to(ROOT)): sha(HERE / name) for name in SOURCES}
    for path, digest in files.items():
        if require_committed and hashlib.sha256(M.git_bytes("HEAD", path)).hexdigest() != digest:
            raise ValueError(f"GPU stages require committed source bytes: {path}")
    return files


def load_plan(path, *, require_committed=False):
    path = safe(path)
    plan = read(path)
    if (plan.get("schema") != 1 or plan.get("state") not in ("FROZEN", "LOCKED")
            or plan.get("scope") != SCOPE or plan.get("formal_evidence") is not False
            or plan.get("batch_sizes") != BATCHES
            or plan.get("cross_panel") != "negbench-negated-dev1000.json"
            or plan.get("cross_budget") != 50
            or not isinstance(plan.get("data_preparation_sha256"), str)):
        raise ValueError("Literature development plan is not the fixed protocol")
    analysis = plan.get("analysis", {})
    if (not isinstance(analysis.get("seed"), int)
            or not isinstance(analysis.get("bootstrap_draws"), int)
            or not 1 <= analysis["bootstrap_draws"] <= 100000
            or not isinstance(analysis.get("block_queries"), int)
            or not 1 <= analysis["block_queries"] <= 1024):
        raise ValueError("Plan must fix CPU bootstrap seed/draws/block size")
    sources = source_files(require_committed)
    declared = plan.get("source_files")
    if declared is not None and any(declared.get(k) != v for k, v in sources.items()):
        raise ValueError("Plan source-file SHA differs from the executing implementation")
    if require_committed:
        if not isinstance(declared, dict):
            raise ValueError("GPU plan must freeze all executing source-file SHA values")
        for filename, expected in declared.items():
            candidate = safe(ROOT / filename)
            if not candidate.is_relative_to(ROOT) or sha(candidate) != expected:
                raise ValueError("A source declared by the plan differs or escapes the repository")
            if hashlib.sha256(M.git_bytes("HEAD", filename)).hexdigest() != expected:
                raise ValueError("A source declared by the plan is not committed")
        try:
            relative = str(path.relative_to(ROOT))
        except ValueError as error:
            raise ValueError("GPU plan must be tracked inside the repository") from error
        if M.git_bytes("HEAD", relative) != path.read_bytes():
            raise ValueError("GPU stages require a committed plan")
    return plan


def eligible(query):
    return query.get("model_eligible", True) is True and isinstance(query.get("text"), str) and bool(query["text"].strip())


def validate_panels(panels, *, enforce_counts=True):
    gallery = panels[PANEL_NAMES[0]]["candidates"]
    ids = [str(c["id"]) for c in gallery]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Gallery IDs must be unique")
    if enforce_counts and len(ids) != 5000:
        raise ValueError("Only the full official COCO val2017 gallery is allowed")
    for c in gallery:
        if (c.get("split") != "val2017" or not isinstance(c.get("sha256"), str)
                or len(c["sha256"]) != 64):
            raise ValueError("Gallery must use verified val2017 JPEGs")
        safe(c["path"])
    lookup = set(ids)
    for name in PANEL_NAMES:
        panel = panels[name]
        if (panel.get("task") != "source-image-retrieval" or panel.get("candidates") != gallery
                or panel.get("scope") != SCOPE):
            raise ValueError("Retrieval panels must share the identical safe gallery and scope")
        rows = panel["queries"]
        if len({q["id"] for q in rows}) != len(rows):
            raise ValueError("Duplicate query ID")
        if enforce_counts and len(rows) != (1000 if "dev1000" in name else 25014):
            raise ValueError("Fixed official/development query count differs")
        for q in rows:
            if q.get("positive_ids") != [q.get("source_image_id")] or q.get("source_image_id") not in lookup:
                raise ValueError("Gold must identify exactly the official source image")
            if not eligible(q) and (q.get("model_eligible") is not False
                                     or q.get("exclusion_reason") != "source-empty-caption"
                                     or q.get("text", "").strip()):
                raise ValueError("Only explicit original empty captions may be excluded")
            if "dev1000" in name and not eligible(q):
                raise ValueError("All fixed development queries must be model-eligible")
        if "dev1000" in name and len({q["group"] for q in rows}) != len(rows):
            raise ValueError("Development must sample one caption per distinct JPEG group")
    for subset in ("full", "dev1000"):
        plain = panels[f"negbench-plain-{subset}.json"]["queries"]
        negative = panels[f"negbench-negated-{subset}.json"]["queries"]
        key = lambda q: (q["source_image_id"], q["caption_index"], q["group"])
        if [key(q) for q in plain] != [key(q) for q in negative]:
            raise ValueError("Plain and negated rows are not exact source/caption/group pairs")
    for variant in ("plain", "negated"):
        full = {(q["source_image_id"], q["caption_index"]): q
                for q in panels[f"negbench-{variant}-full.json"]["queries"]}
        for q in panels[f"negbench-{variant}-dev1000.json"]["queries"]:
            if full.get((q["source_image_id"], q["caption_index"])) != q:
                raise ValueError("Development row is not an exact row of the official full panel")
    if enforce_counts:
        for variant, expected in (("plain", 25014), ("negated", 25013)):
            if sum(eligible(q) for q in panels[f"negbench-{variant}-full.json"]["queries"]) != expected:
                raise ValueError("Pinned eligible full-query denominator differs")
    return gallery


def load_data(directory, plan, *, verify_images=False):
    directory = safe(directory)
    if sha(directory / "preparation.json") != plan["data_preparation_sha256"]:
        raise ValueError("Prepared data receipt differs from the frozen plan")
    receipt = read(directory / "preparation.json")
    if receipt.get("state") != "COMPLETE" or receipt.get("scope") != SCOPE:
        raise ValueError("Prepared public data is incomplete")
    names = list(PANEL_NAMES)
    if plan.get("include_factor", True):
        names.append("factor-full.json")
    panels, hashes = {}, {}
    for name in names:
        path = directory / name
        expected = receipt.get("outputs", {}).get(name, {})
        hashes[name] = sha(path)
        if hashes[name] != expected.get("sha256") or path.stat().st_size != expected.get("bytes"):
            raise ValueError(f"Prepared panel SHA/size differs: {name}")
        panels[name] = read(path)
    gallery = validate_panels(panels)
    if "factor-full.json" in panels:
        factor = panels["factor-full.json"]
        if factor.get("task") != "single-image-two-caption-matching" or factor.get("candidates") != gallery:
            raise ValueError("FACTOR must be a separate caption-pair task on the same gallery")
        if len(factor.get("items", [])) != 1695:
            raise ValueError("Pinned full FACTOR item count differs")
    if verify_images:
        for c in gallery:
            path = safe(c["path"])
            if path.stat().st_size != c["bytes"] or sha(path) != c["sha256"]:
                raise ValueError(f"Verified candidate JPEG changed: {c['id']}")
    texts = sorted({q["text"] for name in PANEL_NAMES for q in panels[name]["queries"] if eligible(q)}
                   | {t for item in panels.get("factor-full.json", {}).get("items", []) for t in item["captions"]})
    if any(not isinstance(t, str) or not t.strip() for t in texts):
        raise ValueError("Encoded texts must all be nonempty strings")
    return {"panels": panels, "gallery": gallery, "texts": texts, "panel_sha256": hashes,
            "preparation_sha256": sha(directory / "preparation.json")}


def input_identity(bundle, plan_path, config, *, smoke=False):
    return {"schema": "literature-development-v1", "scope": SCOPE, "formal_evidence": False,
            "plan_sha256": sha(plan_path), "preparation_sha256": bundle["preparation_sha256"],
            "panel_sha256": bundle["panel_sha256"], "source_files": source_files(),
            "frozen_source": M.SOURCE, "frozen_config_sha256": M.CONFIG_SHA,
            "frozen_model_code_sha256": hashlib.sha256(M.git_bytes(M.SOURCE, M.MODEL_PATH)).hexdigest(),
            "frozen_common_code_sha256": hashlib.sha256(M.git_bytes(M.SOURCE, M.COMMON_PATH)).hexdigest(),
            "models": {k: config["models"][k] for k in ("core_embed", "cs_clip", "core_reranker")},
            "batch_sizes": BATCHES, "model_inputs": "image-only candidates; verbatim query.text only; no objects/gold/parse",
            "gallery_inputs_sha256": M.digest([{k: c[k] for k in ("id", "path", "sha256")} for c in bundle["gallery"]]),
            "text_inputs_sha256": M.digest(bundle["texts"]),
            "mode": "technical-smoke-no-metrics" if smoke else "public-development"}


def processor_signature(model, name):
    if name == "cs_clip":
        return {"class": type(model).__name__, "preprocess": repr(model.preprocess),
                "tokenizer_class": type(model.tokenizer).__name__,
                "context_length": getattr(model.tokenizer, "context_length", 77),
                "truncation": "unchanged frozen tokenizer; 77-token context including boundary tokens"}
    return M.processor_metadata(model.processor)


def tokenization_audit(bundle, texts, lengths):
    """Count actual query rows as well as unique strings; never filter by length."""
    lookup = dict(zip(texts, lengths, strict=True))
    panels = {}
    for name in PANEL_NAMES:
        rows = bundle["panels"][name]["queries"]
        selected = [q for q in rows if eligible(q)]
        known = [q for q in selected if q["text"] in lookup]
        panels[name] = {"raw_query_rows": len(rows), "eligible_query_rows": len(selected),
            "encoded_query_rows": len(known), "over_77_query_rows": sum(lookup[q["text"]] > 77 for q in known),
            "unencoded_eligible_rows": len(selected)-len(known),
            "max_untruncated_query_tokens": max((lookup[q["text"]] for q in known), default=None)}
    return {"unique_texts": len(texts), "over_context": sum(x > 77 for x in lengths),
            "max_untruncated_tokens": max(lengths, default=None), "panels": panels,
            "rule": "frozen CS-CLIP context 77 incl boundaries; raw pre-truncation lengths; no length-based query exclusion"}


def stable_order(scores):
    scores = np.asarray(scores)
    if scores.ndim != 2 or not np.isfinite(scores).all():
        raise ValueError("Scores must be a finite query-by-candidate matrix")
    return np.argsort(-scores, axis=1, kind="stable")


def rank_metrics(ranks):
    ranks = np.asarray(ranks)
    if ranks.ndim != 1 or not len(ranks) or (ranks < 1).any():
        raise ValueError("Source-image ranks must be positive")
    return np.column_stack([1.0 / ranks] + [ranks <= k for k in (1, 5, 10, 50)]).astype(float)


def source_ranks(scores, positive_positions):
    order = stable_order(scores)
    positions = np.asarray(positive_positions, dtype=int)
    if positions.shape != (len(order),) or (positions < 0).any() or (positions >= order.shape[1]).any():
        raise ValueError("Exactly one valid source-image position is required per query")
    return np.argmax(order == positions[:, None], axis=1) + 1


def query_blocks(bundle, image, text, rows, block_size):
    """At most block_size x gallery scores exist; never create full 25k x 5k."""
    table = {value: i for i, value in enumerate(bundle["texts"])}
    ids = {c["id"]: i for i, c in enumerate(bundle["gallery"])}
    selected = [q for q in rows if eligible(q)]
    ranks = []
    for start in range(0, len(selected), block_size):
        batch = selected[start:start + block_size]
        scores = np.asarray(text[[table[q["text"]] for q in batch]]) @ np.asarray(image).T
        ranks.extend(source_ranks(scores, [ids[q["source_image_id"]] for q in batch]).tolist())
    return selected, np.asarray(ranks, dtype=np.int64)


def metrics_summary(values):
    return {name: float(value * 100) for name, value in zip(METRICS, values.mean(axis=0), strict=True)}


def paired_stats(left, right, groups, *, seed, draws):
    left, right = np.asarray(left), np.asarray(right)
    if left.shape != right.shape or left.shape != (len(groups), len(METRICS)):
        raise ValueError("Paired metrics/group rows differ")
    delta = right - left
    keys = sorted(set(groups))
    indices = {key: i for i, key in enumerate(keys)}
    sums = np.zeros((len(keys), delta.shape[1]))
    sizes = np.zeros(len(keys))
    for row, key in zip(delta, groups, strict=True):
        sums[indices[key]] += row
        sizes[indices[key]] += 1
    if len(keys) < 2:
        raise ValueError("At least two source-image groups are needed")
    rng, boot = np.random.default_rng(seed), []
    for start in range(0, draws, 128):
        sampled = rng.integers(0, len(keys), size=(min(128, draws-start), len(keys)))
        boot.append(sums[sampled].sum(axis=1) / sizes[sampled].sum(axis=1)[:, None])
    boot = np.concatenate(boot) * 100
    ci = np.quantile(boot, [0.025, 0.975], axis=0)
    return {"unit": "paired source-image/exact-JPEG groups", "groups": len(keys), "draws": draws,
            "seed": seed, "direction": "right minus left; percentage points",
            "metrics": {name: {"delta_pp": float(delta[:, j].mean()*100),
                                "ci95_pp": ci[:, j].tolist(),
                                "bootstrap_se_pp": float(boot[:, j].std(ddof=1))}
                        for j, name in enumerate(METRICS)},
            "limitation": "development-only conditional interval; shared gallery/semantic near-duplicates not fully modeled"}


def shortlist(bundle, image, text, rows, block_size=128):
    """Admission accepts image vectors plus complete query text, with no gold use."""
    if any(not eligible(q) for q in rows):
        raise ValueError("Cross development panel contains an ineligible query")
    table = {value: i for i, value in enumerate(bundle["texts"])}
    orders = []
    for start in range(0, len(rows), block_size):
        batch = rows[start:start+block_size]
        orders.append(stable_order(np.asarray(text[[table[q["text"]] for q in batch]]) @ np.asarray(image).T))
    return np.concatenate(orders)


def reranked_source_ranks(order, scores, positive_positions, budget=50):
    """Rerank the admitted prefix only; all candidates outside keep their rank."""
    order, scores = np.asarray(order), np.asarray(scores)
    if scores.shape != (len(order), budget) or order.shape[1] < budget:
        raise ValueError("Cross scores must align with the fixed shortlist")
    prefix = np.take_along_axis(order[:, :budget], stable_order(scores), axis=1)
    final = np.concatenate([prefix, order[:, budget:]], axis=1)
    positions = np.asarray(positive_positions)
    if positions.shape != (len(order),) or not np.all((final == positions[:, None]).sum(axis=1) == 1):
        raise ValueError("The source image must remain exactly once in the complete gallery")
    ranks = np.argmax(final == positions[:, None], axis=1) + 1
    original = np.argmax(order == positions[:, None], axis=1) + 1
    if not np.array_equal(ranks > budget, original > budget):
        raise ValueError("Reranking changed admission Recall")
    return original, ranks


def complete_array(path, rows, dim=None):
    state = read(path.with_suffix(".state.json"))
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if (state.get("completed") != rows or state.get("sha256") != sha(path)
            or state.get("array_shape") != list(array.shape) or state.get("dtype") != array.dtype.str
            or len(array) != rows or not np.isfinite(array).all()
            or (dim is not None and array.shape != (rows, dim))):
        raise ValueError(f"Canonical checkpoint is incomplete or differs: {path.name}")
    return array


def validate_index(out, bundle, plan_path):
    meta = read(out / "index-meta.json")
    expected = input_identity(bundle, plan_path, M.frozen_config())
    if meta.get("state") != "COMPLETE" or any(meta.get("identity", {}).get(k) != v for k, v in expected.items()):
        raise ValueError("Index source/plan/data identity differs or is incomplete")
    if not meta["identity"].get("environment") or set(meta.get("processors", {})) != {"core_embed", "cs_clip"}:
        raise ValueError("Index lacks actual environment/processor provenance")
    result = {}
    for name in ("core_image", "core_text", "cs_image", "cs_text"):
        path = out / (name + ".npy")
        saved = meta.get("arrays", {}).get(name, {})
        if saved.get("sha256") != sha(path) or saved.get("state_sha256") != sha(path.with_suffix(".state.json")):
            raise ValueError("Canonical vector payload/state SHA differs")
        result[name] = complete_array(path, len(bundle["gallery"] if name.endswith("image") else bundle["texts"]),
                                      2048 if name.startswith("core") else 512)
        if result[name].dtype != np.float32:
            raise ValueError("Frozen encoders must produce float32 vectors")
        M.unit_vectors(result[name])
    return result, meta


def run_index(args, bundle, plan, *, smoke=False):
    invocation_began = monotonic()
    import torch
    config = M.frozen_config()
    env = M.environment()
    paths = M.snapshots(config, required={"core_embed", "cs_clip"})
    frozen = M.frozen_module("literature_frozen_models", M.MODEL_PATH)
    common = M.frozen_module("literature_frozen_common", M.COMMON_PATH)
    identity = input_identity(bundle, args.plan, config, smoke=smoke) | {"environment": env}
    if not smoke and (args.out / "index-meta.json").exists():
        _, existing = validate_index(args.out, bundle, args.plan)
        if existing["identity"] != identity:
            raise ValueError("Existing complete index uses a different actual environment")
        print("[codex] Reusing complete canonical index without rewriting its receipt", flush=True)
        return
    M.fixed_json(args.out / "index-provenance.json", identity)
    M.fixed_json(args.out / "text-table.json", bundle["texts"])
    helper, processors, arrays = M.checkpoint_helpers(), {}, {}
    gallery = bundle["gallery"][:8] if smoke else bundle["gallery"]
    texts = bundle["texts"][:4] if smoke else bundle["texts"]
    def open_images(indices):
        from PIL import Image
        output = []
        for index in indices:
            with Image.open(gallery[index]["path"]) as image:
                output.append(image.convert("RGB"))
        return output
    began = monotonic()
    for name, prefix in (("core_embed", "core"), ("cs_clip", "cs")):
        if name == "core_embed":
            model = frozen.Encoder(name, paths[name], config, "cuda")
        else:
            previous = sys.modules.get("common")
            sys.modules["common"] = common
            try:
                model = frozen.ClipEncoder(name, paths[name], config, "cuda")
            finally:
                if previous is None:
                    sys.modules.pop("common", None)
                else:
                    sys.modules["common"] = previous
        processors[name] = processor_signature(model, name)
        M.fixed_json(args.out / (prefix + "-processor.json"), processors[name])
        if name == "cs_clip":
            if not hasattr(model.tokenizer, "encode"):
                raise ValueError("Frozen CS-CLIP tokenizer must expose pre-truncation lengths for the fixed audit")
            token_lengths = [len(model.tokenizer.encode(t))+2 for t in texts]
            M.fixed_json(args.out / "cs-tokenization.json", tokenization_audit(bundle, texts, token_lengths))
        for suffix, items, batch, compute in (
            ("image", list(range(len(gallery))), BATCHES["image"], lambda rows: model.encode(open_images(rows), images=True)),
            ("text", texts, BATCHES["text"], model.encode)):
            stage = prefix + "_" + suffix
            array = helper.batches(args.out / (stage + ".npy"), items, batch, compute,
                identity | {"stage": stage, "processor": processors[name], "items_sha256": M.digest(items)},
                status=args.out / "progress.json")
            M.unit_vectors(array, len(items))
            expected_dim = 2048 if prefix == "core" else 512
            if array.shape != (len(items), expected_dim) or array.dtype != np.float32:
                raise ValueError("Frozen encoder vector dimension/dtype differs")
            arrays[stage] = {"sha256": sha(args.out / (stage + ".npy")),
                             "state_sha256": sha(args.out / (stage + ".state.json")),
                             "shape": list(array.shape), "dtype": array.dtype.str}
        del model
        gc.collect(); torch.cuda.empty_cache()
    meta = {"state": "COMPLETE", "identity": identity, "processors": processors, "arrays": arrays,
            "placement": M.placement(), "commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
            "current_invocation_wall_seconds": monotonic()-invocation_began,
            "current_invocation_encoding_wall_seconds": monotonic()-began,
            "completed_checkpoint_active_seconds_total": sum(read(args.out / (name+".state.json"))["active_seconds"] for name in arrays),
            "timing_scope": "invocation wall includes validation/model load; cumulative checkpoint time covers completed compute batches across resumes, excluding failed/in-flight batches and other invocations' load time; not deployment end-to-end latency",
            "canonical_storage": "four NPY checkpoints; no duplicate vector NPZ"}
    M.atomic_json(args.out / "index-meta.json", meta)
    M.atomic_json(args.out / "progress.json", {"stage": "smoke_index_complete" if smoke else "index_complete", "research_metrics": False})
    if smoke:
        scorer = frozen.CrossScorer("core_reranker", M.snapshots(config, required={"core_reranker"})["core_reranker"], config, "cuda")
        pairs = [[texts[0], 0], [texts[1], 1]]
        cross = helper.batches(args.out / "smoke_cross.npy", pairs, BATCHES["cross"],
            lambda rows: scorer.score([(t, image) for (t, _), image in zip(rows, open_images([ci for _, ci in rows]), strict=True)]),
            identity | {"stage": "smoke_cross", "processor": processor_signature(scorer, "core_reranker")},
            status=args.out / "progress.json")
        if cross.shape != (2,) or not np.isfinite(cross).all():
            raise ValueError("Smoke cross output is invalid")
        M.atomic_json(args.out / "smoke.json", {"state": "PASS", "scope": "technical-no-metrics", "research_metrics": False,
                     "images": len(gallery), "texts": len(texts), "cross_pairs": 2,
                     "index_meta_sha256": sha(args.out / "index-meta.json"), "cross_sha256": sha(args.out / "smoke_cross.npy")})


def factor_summary(bundle, vectors):
    panel = bundle["panels"].get("factor-full.json")
    if panel is None:
        return None
    table = {t: i for i, t in enumerate(bundle["texts"])}
    ids = {c["id"]: i for i, c in enumerate(bundle["gallery"])}
    output = {}
    for prefix in ("core", "cs"):
        grouped, detail = {}, []
        for item in panel["items"]:
            scores = np.asarray(vectors[prefix+"_text"][[table[t] for t in item["captions"]]]) @ vectors[prefix+"_image"][ids[item["image_id"]]]
            target = item.get("correct_caption_index")
            row = {"id": item["id"], "group": item["group"], "test_type": item["test_type"],
                   "operator": item["operator"], "score_gap_a_minus_b": float(scores[0]-scores[1]),
                   "abs_score_gap": float(abs(scores[0]-scores[1])),
                   "exact_score_tie": bool(scores[0] == scores[1]),
                   "correct": bool(int(np.argmax(scores)) == target) if target in (0, 1) else None}
            detail.append(row)
            for key in ("test_type:"+item["test_type"], "operator:"+item["operator"]):
                grouped.setdefault(key, []).append(row)
        output[prefix] = {"scope": "holistic cosine caption-pair diagnostic; not LCSE/oracle-parse reproduction",
            "tie_rule": "exact equal cosine scores choose original caption a; equivalence is not binary accuracy",
            "groups": {key: {"items": len(rows), "binary_accuracy_pct": float(np.mean([r["correct"] for r in rows])*100)
                              if all(r["correct"] is not None for r in rows) else None,
                              "mean_abs_score_gap": float(np.mean([r["abs_score_gap"] for r in rows])),
                              "exact_ties": sum(r["exact_score_tie"] for r in rows),
                              "gold_both": sum(r["correct"] is None for r in rows)} for key, rows in grouped.items()},
            "per_item": detail}
    return output


def analyze(args, bundle, plan):
    vectors, meta = validate_index(args.out, bundle, args.plan)
    report = {"schema": 1, "scope": SCOPE, "formal_evidence": False,
              "gold_semantics": "official single source image, not exhaustive all-condition relevance",
              "metric_semantics": "source-image MRR/Recall (%); single-positive full AP equals RR and is not reported as semantic AP",
              "gallery": len(bundle["gallery"]), "methods": {}, "paired": {},
              "provenance": {"index_meta_sha256": sha(args.out / "index-meta.json"), "index": meta},
              "tie_rule": "descending score; original numeric-ID gallery order for ties",
              "interval_scope": "paired development-1000 comparisons only; full-query summaries are descriptive without CIs",
              "cs_tokenization": read(args.out / "cs-tokenization.json")}
    detail, measured = {}, {}
    settings = plan["analysis"]
    for prefix in ("core", "cs"):
        report["methods"][prefix] = {}
        detail[prefix] = {}
        for name in PANEL_NAMES:
            rows = bundle["panels"][name]["queries"]
            selected, ranks = query_blocks(bundle, vectors[prefix+"_image"], vectors[prefix+"_text"], rows, settings["block_queries"])
            values = rank_metrics(ranks)
            measured[prefix, name] = values
            report["methods"][prefix][name] = {"raw_queries": len(rows), "eligible_queries": len(selected),
                "skipped": [{"id": q["id"], "reason": q.get("exclusion_reason")} for q in rows if not eligible(q)],
                "metrics": metrics_summary(values)}
            detail[prefix][name] = [{"id": q["id"], "source_image_id": q["source_image_id"], "caption_index": q["caption_index"],
                                    "group": q["group"], "source_image_rank": int(rank)} for q, rank in zip(selected, ranks, strict=True)]
    options = {"seed": settings["seed"], "draws": settings["bootstrap_draws"]}
    groups = [q["group"] for q in bundle["panels"]["negbench-plain-dev1000.json"]["queries"]]
    for variant in ("plain", "negated"):
        name = f"negbench-{variant}-dev1000.json"
        report["paired"][f"cs_minus_core/{variant}"] = paired_stats(measured["core", name], measured["cs", name], groups, **options)
    for prefix in ("core", "cs"):
        report["paired"][f"negated_minus_plain/{prefix}"] = paired_stats(measured[prefix, "negbench-plain-dev1000.json"],
            measured[prefix, "negbench-negated-dev1000.json"], groups, **options)
    factor = factor_summary(bundle, vectors)
    if factor:
        report["factor"] = {key: {k: v for k, v in value.items() if k != "per_item"} for key, value in factor.items()}
        detail["factor"] = {key: value["per_item"] for key, value in factor.items()}
    M.atomic_json(args.out / "report.json", report)
    M.atomic_json(args.out / "per_query.json", detail)


def run_cross(args, bundle, plan):
    invocation_began = monotonic()
    import torch
    vectors, index_meta = validate_index(args.out, bundle, args.plan)
    env = M.environment()
    if index_meta["identity"]["environment"] != env:
        raise ValueError("Cross and index actual platform environments differ")
    config = M.frozen_config()
    paths = M.snapshots(config, required={"core_reranker"})
    frozen = M.frozen_module("literature_frozen_cross", M.MODEL_PATH)
    rows = bundle["panels"][plan["cross_panel"]]["queries"]
    order = shortlist(bundle, vectors["core_image"], vectors["core_text"], rows, plan["analysis"]["block_queries"])
    pairs = [[qi, int(ci)] for qi, prefix in enumerate(order[:, :50]) for ci in prefix]
    if len(pairs) != 50000:
        raise ValueError("Fixed dev1000 baseline must score exactly 50,000 pairs")
    M.fixed_json(args.out / "cross-pairs.json", pairs)
    if (args.out / "cross-meta.json").exists():
        validate_cross_artifacts(args.out, bundle, plan, vectors, index_meta)
        print("[codex] Reusing complete cross50 without rewriting its receipt", flush=True)
        return
    scorer = frozen.CrossScorer("core_reranker", paths["core_reranker"], config, "cuda")
    identity = index_meta["identity"] | {"stage": "core50", "index_meta_sha256": sha(args.out / "index-meta.json"),
        "arrays": index_meta["arrays"], "query_ids": [q["id"] for q in rows], "pairs_sha256": sha(args.out / "cross-pairs.json"),
        "cross_processor": processor_signature(scorer, "core_reranker"), "cross_batch": BATCHES["cross"],
        "selection": "CORE holistic top50 only; no gold filtering or method tuning"}
    M.fixed_json(args.out / "cross-provenance.json", identity)
    def compute(batch):
        from PIL import Image
        opened = []
        for qi, ci in batch:
            with Image.open(bundle["gallery"][ci]["path"]) as image:
                opened.append((rows[qi]["text"], image.convert("RGB")))
        return scorer.score(opened)
    began = monotonic()
    values = M.checkpoint_helpers().batches(args.out / "cross50.npy", pairs, BATCHES["cross"], compute,
                                           identity, status=args.out / "progress.json")
    if values.shape != (50000,):
        raise ValueError("Unexpected fixed-budget cross score shape")
    M.atomic_json(args.out / "cross-meta.json", {"state": "COMPLETE", "identity": identity,
        "cross_sha256": sha(args.out / "cross50.npy"), "state_sha256": sha(args.out / "cross50.state.json"),
        "current_invocation_wall_seconds": monotonic()-invocation_began,
        "current_invocation_checkpoint_wall_seconds": monotonic()-began,
        "completed_checkpoint_active_seconds_total": read(args.out / "cross50.state.json")["active_seconds"],
        "timing_scope": "invocation wall includes validation/ranking/model load; cumulative checkpoint time covers completed compute batches across resumes, excluding failed/in-flight batches and other invocations' load time; not deployment end-to-end latency",
        "placement": M.placement(), "pairs": 50000,
        "commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "cost": "50 pairs per query; all 1000 queries retained regardless of whether source image is admitted"})
    del scorer
    gc.collect(); torch.cuda.empty_cache()


def validate_cross_artifacts(out, bundle, plan, vectors, index_meta):
    meta = read(out / "cross-meta.json")
    identity = meta.get("identity", {})
    if (meta.get("state") != "COMPLETE" or identity.get("index_meta_sha256") != sha(out / "index-meta.json")
            or any(identity.get(k) != v for k, v in index_meta["identity"].items())
            or identity.get("cross_batch") != 4 or not identity.get("cross_processor")
            or meta.get("cross_sha256") != sha(out / "cross50.npy")
            or meta.get("state_sha256") != sha(out / "cross50.state.json")):
        raise ValueError("Cross scores are incomplete or source/index/platform identity differs")
    rows = bundle["panels"][plan["cross_panel"]]["queries"]
    order = shortlist(bundle, vectors["core_image"], vectors["core_text"], rows, plan["analysis"]["block_queries"])
    pairs = [[qi, int(ci)] for qi, prefix in enumerate(order[:, :50]) for ci in prefix]
    if (read(out / "cross-pairs.json") != pairs or identity.get("pairs_sha256") != sha(out / "cross-pairs.json")
            or identity.get("query_ids") != [q["id"] for q in rows]):
        raise ValueError("Saved cross pairs differ from actual original CORE top50")
    cross = complete_array(out / "cross50.npy", len(pairs))
    if cross.shape != (50000,) or cross.dtype != np.float32 or len(rows) != 1000:
        raise ValueError("Frozen cross50 must preserve all 1000 queries and 50000 float32 pair scores")
    return rows, order, cross, meta


def analyze_cross(args, bundle, plan):
    vectors, index_meta = validate_index(args.out, bundle, args.plan)
    rows, order, cross, meta = validate_cross_artifacts(args.out, bundle, plan, vectors, index_meta)
    pairs_count = len(rows)*50
    lookup = {c["id"]: i for i, c in enumerate(bundle["gallery"])}
    original, reranked = reranked_source_ranks(order, cross.reshape(len(rows), 50), [lookup[q["source_image_id"]] for q in rows])
    left, right = rank_metrics(original), rank_metrics(reranked)
    settings = plan["analysis"]
    report = {"schema": 1, "scope": SCOPE, "formal_evidence": False,
        "task": "NegBench negated-dev1000 source-image retrieval", "queries": len(rows), "candidates": len(bundle["gallery"]),
        "baseline": metrics_summary(left), "core50": metrics_summary(right),
        "paired_core50_minus_embedding": paired_stats(left, right, [q["group"] for q in rows],
            seed=settings["seed"], draws=settings["bootstrap_draws"]),
        "metric_semantics": "single source-image MRR/Recall, not exhaustive all-condition AP; R@50 is admission Recall and cannot change when only top50 reranks",
        "cost": {"pairs": pairs_count, "pairs_per_query": 50, "gold_used_for_admission": False},
        "provenance": {"index_meta_sha256": sha(args.out / "index-meta.json"), "cross_meta_sha256": sha(args.out / "cross-meta.json"), "cross": meta}}
    M.atomic_json(args.out / "cross-report.json", report)
    M.atomic_json(args.out / "cross-per_query.json", [{"id": q["id"], "group": q["group"], "source_image_id": q["source_image_id"],
        "embedding_rank": int(a), "core50_rank": int(b)} for q, a, b in zip(rows, original, reranked, strict=True)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=STAGES)
    for field in ("data", "out", "plan"):
        parser.add_argument("--"+field, type=Path, required=True)
    args = parser.parse_args()
    args.data, args.out, args.plan = map(safe, (args.data, args.out, args.plan))
    if args.out == args.data or args.out in args.data.parents or args.data in args.out.parents:
        parser.error("Runtime output and prepared public panels must use separate directories")
    gpu = args.stage in ("smoke", "index", "cross50")
    plan = load_plan(args.plan, require_committed=gpu)
    bundle = load_data(args.data, plan, verify_images=gpu)
    with M.writer(args.out):
        M.fixed_json(args.out / "run-mode.json", {"mode": "technical-smoke-no-metrics" if args.stage == "smoke" else "public-development"})
        if args.stage in ("smoke", "index"):
            run_index(args, bundle, plan, smoke=args.stage == "smoke")
        elif args.stage == "cross50":
            run_cross(args, bundle, plan)
        elif args.stage == "analyze":
            analyze(args, bundle, plan)
        else:
            analyze_cross(args, bundle, plan)
    print(json.dumps({"stage": args.stage+"_complete", "out": str(args.out), "formal_evidence": False}), flush=True)


if __name__ == "__main__":
    main()
