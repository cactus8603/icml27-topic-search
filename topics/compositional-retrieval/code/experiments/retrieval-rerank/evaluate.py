"""Compare fixed CORE reranking with the original CORE ranking over a whole panel."""

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare import DATA, coverage, file_sha, preflight, prior
from e2_build_queries import joint_text
from headroom import CONFIG, cross_scores, encode_items, model_dir, rgb
from headroom_models import CrossScorer, Encoder


MODEL_CONFIG_SHA = "862591c60c5609f16727d902993df0cba9cab96ded72611d69d263a02fe4ad1d"
TOP_K = 50
MIN_GAIN_PP = 2.0
BOOTSTRAP_SEED = 2026111536
BOOTSTRAP_DRAWS = 10000


def primary_order(scores):
    if scores.ndim != 2 or scores.shape[1] < TOP_K or not np.isfinite(scores).all():
        raise ValueError("Expected finite CORE scores over the full candidate pool")
    return np.argsort(-scores, axis=1, kind="stable")


def rerank(order, cross):
    if cross.shape != (order.shape[0], TOP_K) or not np.isfinite(cross).all():
        raise ValueError("Expected one finite cross-score per shortlisted item")
    output = order.copy()
    output[:, :TOP_K] = np.take_along_axis(
        order[:, :TOP_K], np.argsort(-cross, axis=1, kind="stable"), axis=1
    )
    return output


def average_precision(ordered):
    positives = ordered.sum(axis=1)
    if np.any(positives == 0):
        raise ValueError("All queries require known relevance within the full pool")
    precision = ordered.cumsum(axis=1) / np.arange(1, ordered.shape[1] + 1)
    return (ordered * precision).sum(axis=1) / positives


def summarize(baseline, reranked, labels, groups):
    old_labels = np.take_along_axis(labels, baseline, axis=1)
    new_labels = np.take_along_axis(labels, reranked, axis=1)
    old_ap, new_ap = (100 * average_precision(arm) for arm in (old_labels, new_labels))
    prefix_hits = old_labels[:, :TOP_K].sum(axis=1)
    bound = np.concatenate((np.arange(TOP_K) < prefix_hits[:, None], old_labels[:, TOP_K:]), axis=1)
    bound_map = float(100 * average_precision(bound).mean())
    if new_ap.mean() > bound_map + 1e-9 or not np.array_equal(baseline[:, TOP_K:], reranked[:, TOP_K:]):
        raise ValueError("Fixed shortlist reranking invariant violated")
    _, group = np.unique(groups, return_inverse=True)
    counts = np.bincount(group)
    change = new_ap - old_ap
    totals = np.bincount(group, weights=change)
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(0, len(counts), (BOOTSTRAP_DRAWS, len(counts)))
    lo, hi = np.quantile(totals[draws].sum(axis=1) / counts[draws].sum(axis=1), [0.025, 0.975])
    point = float(change.mean())
    state = "PASS" if point >= MIN_GAIN_PP and lo > 0 else "FAIL" if hi < MIN_GAIN_PP else "灰區"
    return {"state": state, "baseline_map_pp": float(old_ap.mean()),
            "reranked_map_pp": float(new_ap.mean()), "delta_map_pp": point,
            "delta_95ci_pp": [float(lo), float(hi)], "clusters": len(counts),
            "fixed_top50_oracle_map_pp": bound_map,
            "fixed_top50_oracle_gain_pp": bound_map - float(old_ap.mean()),
            "recall_at_50_pp": float(100 * (prefix_hits / labels.sum(axis=1)).mean()),
            "queries_without_top50_hit": int((prefix_hits == 0).sum())}


def frozen_config():
    if file_sha(CONFIG) != MODEL_CONFIG_SHA:
        raise ValueError("CORE model configuration changed")
    config = json.loads(CONFIG.read_bytes())
    paths = {name: model_dir(config["models"][name]) for name in ("core_embed", "core_reranker")}
    return config, paths


def run(panel, config, snapshots):
    import torch

    ids, queries, names = panel["ids"], panel["queries"], panel["names"]
    captions = [joint_text([names[str(cat)] for cat in query["key"]]) for query in queries]
    openers = [lambda image_id=image_id: rgb(prior.image_path(DATA, image_id)) for image_id in ids]
    print("Indexing the frozen pool", flush=True)
    encoder = Encoder("core_embed", snapshots["core_embed"], config, "cuda")
    vectors, text = encode_items(encoder, openers, captions, config)
    base = primary_order(text @ vectors.T)
    del encoder
    torch.cuda.empty_cache()
    print(f"Scoring {len(captions) * TOP_K} fixed shortlist pairs", flush=True)
    scorer = CrossScorer("core_reranker", snapshots["core_reranker"], config, "cuda")
    new_scores = cross_scores(
        scorer, [caption for caption in captions for _ in range(TOP_K)],
        [openers[index] for index in base[:, :TOP_K].flat], config["batch_size"]["cross_pair"],
    ).reshape(len(queries), TOP_K)
    changed = rerank(base, new_scores)
    positions = {image_id: col for col, image_id in enumerate(ids)}
    relevance = np.zeros(base.shape, dtype=bool)
    for row, query in enumerate(queries):
        relevance[row, [positions[image_id] for image_id in query["relevant"]]] = True
    return summarize(base, changed, relevance, [query["group"] for query in queries])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--output", type=Path)
    parser.add_argument("--freeze")
    args = parser.parse_args()
    config, models = frozen_config()
    panel, manifest_sha = preflight()
    context = {"panel_sha256": manifest_sha, "coverage": coverage(panel["ids"], panel["queries"]),
               "model_config_sha256": MODEL_CONFIG_SHA, "top_k": TOP_K,
               "revisions": {name: path.name for name, path in models.items()}}
    if args.preflight:
        print(json.dumps(context, sort_keys=True))
        return
    job = os.environ.get("SLURM_JOB_ID", "")
    expected = DATA.parent.parent / "runs" / "retrieval-rerank" / job
    if not job.isascii() or not job.isdecimal() or args.output != expected or not args.freeze:
        parser.error("Use a new recorded Slurm job and frozen commit")
    args.output.mkdir(parents=True, exist_ok=False)
    result = run(panel, config, models)
    context.update(freeze=args.freeze, slurm_job_id=job,
                   versions={name: version(name) for name in ("numpy", "torch", "transformers")})
    partial = args.output / "summary.partial"
    with partial.open("x") as stream:
        json.dump(context | result, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
    partial.replace(args.output / "summary.json")


if __name__ == "__main__":
    main()
