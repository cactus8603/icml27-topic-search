"""Score the fixed CORE shortlist with matched real and shuffled anchor controls."""

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prepare import DATA, file_sha, image_path, preflight
from headroom import CONFIG, cross_scores, encode_items, model_dir, rgb
from headroom_models import ClipEncoder, CrossScorer, Encoder, unit_scores
from prospective_rank import increment


MODEL_CONFIG_SHA = "862591c60c5609f16727d902993df0cba9cab96ded72611d69d263a02fe4ad1d"
K = 50
ANCHOR_WEIGHT = 0.125
FUSION_WEIGHT = 0.25
MIN_GAIN_PP = 2.0
BOOT_SEED = 2026111544
DRAWS = 10_000
FAMILY_ALPHA = 0.05
CONTROLS = ("core_only", "plus_cs_full", "plus_cs_shuffled_anchor")
PRIMARY = "plus_cs_real_anchor"


def ranks(scores):
    if scores.ndim != 2 or not np.isfinite(scores).all():
        raise ValueError("Expected a matrix of finite ranking scores")
    return np.argsort(-scores, axis=1, kind="stable")


def percentile(values):
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Expected finite fixed-shortlist scores")
    order = np.argsort(values, axis=1, kind="stable")
    sorted_values = np.take_along_axis(values, order, axis=1)
    count = values.shape[1]
    quantiles = np.empty(values.shape, dtype=np.float64)
    for quantile, indices, row in zip(quantiles, order, sorted_values, strict=True):
        edges = np.r_[0, 1 + np.flatnonzero(row[:-1] != row[1:]), count]
        average = (edges[:-1] + edges[1:] + 1) / (2 * count)
        quantile[indices] = np.repeat(average, np.diff(edges))
    return quantiles


def arms(cross, full, real, shuffled, top_k=K, weight=ANCHOR_WEIGHT, fusion=FUSION_WEIGHT):
    if cross.ndim != 2 or cross.shape[1] != top_k or not all(a.shape == cross.shape for a in (full, real, shuffled)):
        raise ValueError("All scores must have the same fixed shortlist shape")
    base = percentile(cross)
    if not np.array_equal(ranks(base), ranks(cross)):
        raise ValueError("CORE rank conversion changed the baseline")
    return {CONTROLS[0]: cross, CONTROLS[1]: base + fusion * percentile(full),
            CONTROLS[2]: base + fusion * percentile(increment(full, shuffled, weight)),
            PRIMARY: base + fusion * percentile(increment(full, real, weight))}


def full_order(order, scores, top_k=K):
    if scores.shape != (order.shape[0], top_k):
        raise ValueError("Scores do not match the fixed prefix")
    result = order.copy()
    result[:, :top_k] = np.take_along_axis(order[:, :top_k], ranks(scores), axis=1)
    return result


def average_precision(label):
    positives = label.sum(axis=1)
    if np.any(positives <= 0):
        raise ValueError("Full scored pool must contain a positive for each query")
    return (label * label.cumsum(axis=1) / np.arange(1, label.shape[1] + 1)).sum(axis=1) / positives


def summarize(orders, relevance, groups, top_k=K):
    aps = {name: 100 * average_precision(np.take_along_axis(relevance, order, axis=1))
           for name, order in orders.items()}
    reference = orders[CONTROLS[0]]
    for order in orders.values():
        if (not np.array_equal(reference[:, top_k:], order[:, top_k:])
                or not np.array_equal(np.sort(reference[:, :top_k], axis=1), np.sort(order[:, :top_k], axis=1))):
            raise ValueError("Fixed membership or tail order changed")
    ranked = np.take_along_axis(relevance, reference, axis=1)
    hits = ranked[:, :top_k].sum(axis=1)
    oracle = np.concatenate((np.arange(top_k)[None, :] < hits[:, None], ranked[:, top_k:]), axis=1)
    _, cluster = np.unique(groups, return_inverse=True)
    counts = np.bincount(cluster)
    draws = np.random.default_rng(BOOT_SEED).integers(0, len(counts), size=(DRAWS, len(counts)))
    den = counts[draws].sum(axis=1)
    alpha = FAMILY_ALPHA / (2 * len(CONTROLS))
    contrast = {}
    for name in CONTROLS:
        diff = aps[PRIMARY] - aps[name]
        weights = np.bincount(cluster, weights=diff)
        bounds = np.quantile(weights[draws].sum(axis=1) / den, [alpha, 1 - alpha]).tolist()
        contrast[name] = {"gain_pp": float(diff.mean()), "simultaneous_ci_pp": bounds}
    if all(r["gain_pp"] >= MIN_GAIN_PP and r["simultaneous_ci_pp"][0] > 0 for r in contrast.values()):
        state = "PASS"
    elif any(r["simultaneous_ci_pp"][1] < MIN_GAIN_PP for r in contrast.values()):
        state = "FAIL"
    else:
        state = "INCONCLUSIVE"
    return {"state": state, "paired_real_anchor_minus_control": contrast,
            "map_pp": {name: float(ap.mean()) for name, ap in aps.items()},
            "comparison_ci_level": 1 - FAMILY_ALPHA / len(CONTROLS),
            "fixed_top50_oracle_map_pp": float(100 * average_precision(oracle).mean()),
            "recall_at_50_pp": float(100 * (hits / relevance.sum(axis=1)).mean()),
            "queries_without_top50_hit": int((hits == 0).sum()), "groups": len(counts)}


def models():
    if file_sha(CONFIG) != MODEL_CONFIG_SHA:
        raise ValueError("Changed fixed model configuration")
    settings = json.loads(CONFIG.read_bytes())
    snapshots = {name: model_dir(settings["models"][name]) for name in ("cs_clip", "core_embed", "core_reranker")}
    return settings, snapshots


def run(panel, config, snapshots):
    import torch

    ids, queries = panel["ids"], panel["queries"]
    captions = [q["text"] for q in queries]
    images = [lambda key=key: rgb(image_path(key)) for key in ids]
    batch = config["batch_size"]

    print("Indexing the whole frozen pool with CORE", flush=True)
    core = Encoder("core_embed", snapshots["core_embed"], config, "cuda")
    image_vectors, text_vectors = encode_items(core, images, captions, config)
    order = ranks(text_vectors @ image_vectors.T)
    shortlist = order[:, :K]
    del core, image_vectors, text_vectors
    torch.cuda.empty_cache()

    print(f"Scoring {shortlist.size} fixed cross-encoder pairs", flush=True)
    reranker = CrossScorer("core_reranker", snapshots["core_reranker"], config, "cuda")
    cross = cross_scores(reranker, [caption for caption in captions for _ in range(K)],
                         [images[index] for index in shortlist.flat], batch["cross_pair"]).reshape(shortlist.shape)
    del reranker
    torch.cuda.empty_cache()

    needed = np.unique(shortlist)
    print(f"Encoding {len(needed)} unique shortlist images with CS-CLIP", flush=True)
    clip = ClipEncoder("cs_clip", snapshots["cs_clip"], config, "cuda")
    fields = ("text", "anchor", "shuffled_anchor")
    texts = sorted({q[field] for q in queries for field in fields})
    text_positions = {caption: i for i, caption in enumerate(texts)}
    selected = unit_scores(clip, [images[i] for i in needed], batch["embed_image"], images=True)
    text = unit_scores(clip, texts, batch["embed_text"])
    del clip
    torch.cuda.empty_cache()
    dot = text @ selected.T
    selected_positions = np.searchsorted(needed, shortlist)
    full, real, shuffled = [
        dot[np.asarray([text_positions[q[field]] for q in queries])[:, None], selected_positions].astype(np.float64)
        for field in fields
    ]

    orders = {name: full_order(order, score) for name, score in arms(cross, full, real, shuffled).items()}
    relevance = np.zeros(order.shape, dtype=bool)
    inverse = {key: col for col, key in enumerate(ids)}
    for i, query in enumerate(queries):
        relevance[i, [inverse[key] for key in query["relevant"]]] = True
    result = summarize(orders, relevance, [q["group"] for q in queries])
    return result | {"classified_cross_pairs": shortlist.size, "cs_unique_shortlist_images": len(needed)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--models-only", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--score", action="store_true")
    parser.add_argument("--freeze")
    args = parser.parse_args()
    config, snapshots = models()
    context = {"model_config_sha256": MODEL_CONFIG_SHA, "top_k": K,
               "anchor_weight": ANCHOR_WEIGHT, "rank_fusion_weight": FUSION_WEIGHT,
               "revisions": {name: path.name for name, path in snapshots.items()}}
    if args.models_only:
        print(json.dumps(context, sort_keys=True))
        return
    panel, manifest_sha, coverage = preflight()
    context |= {"panel_sha256": manifest_sha, "coverage": coverage,
                "near_duplicate_screened": panel["near_duplicate_screened"],
                "source_format_or_dimensions_excluded": sum(
                    r["reason"] == "source_format_or_dimensions" for r in panel["rejected"]),
                "exact_duplicate_excluded": sum(r["reason"] == "exact_duplicate" for r in panel["rejected"])}
    if args.preflight:
        print(json.dumps(context, sort_keys=True))
        return
    job = os.environ.get("SLURM_JOB_ID", "")
    if not re.fullmatch(r"[1-9][0-9]*", job) or not re.fullmatch(r"[0-9a-f]{40}", args.freeze or ""):
        parser.error("Score requires a new Slurm job and reviewed full freeze")
    output = DATA.parent.parent / "runs" / "retrieval-anchor-rerank" / job
    output.mkdir(parents=True, exist_ok=False)
    result = run(panel, config, snapshots)
    context.update(freeze=args.freeze, slurm_job_id=job,
                   versions={name: version(name) for name in ("numpy", "torch", "transformers")})
    partial = output / "summary.partial"
    with partial.open("x") as stream:
        json.dump(context | result, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
    partial.replace(output / "summary.json")


if __name__ == "__main__":
    main()
