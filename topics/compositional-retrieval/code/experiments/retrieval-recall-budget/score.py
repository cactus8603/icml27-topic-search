"""Compare two shortlist generators with the same cross-encoder budget."""

import argparse
from importlib.metadata import version
from importlib.util import module_from_spec, spec_from_file_location
import json
import os
from pathlib import Path
import re
from time import monotonic

import numpy as np


HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / "retrieval-anchor-rerank"
spec = spec_from_file_location("anchor_rerank", PRIOR / "evaluate.py")
original = module_from_spec(spec)
spec.loader.exec_module(original)

K = original.K
DEPTHS = (25, 50, 100, 200)
PANEL_COUNTS = (8191, 136, 118)
SOURCE = "2daedf522c6d64b860277657d3067b61da3a0f2b"
SOURCE_JOB = "26601"
SOURCE_SUMMARY_SHA = "817c6ab72bc4fa2d106debf4dbfe85066859c4767dd9605c1016c5aa98de5dfa"
PUBLISHED = {"map_pp": 54.53, "oracle_map_pp": 74.02, "recall_pp": 73.77}
REPLAY_TOLERANCE_PP = 0.10
BOUNDARY_TOLERANCE = 1e-6
MIN_GAIN_PP = 2.0
BOOT_SEED = 2026111591
DRAWS = 10_000


def interleave(core, cs, k=K):
    """CORE first at each rank depth, dropping duplicate candidate indices."""
    if core.shape != cs.shape or core.ndim != 2 or not 0 < k <= core.shape[1]:
        raise ValueError("Expected two rankings of the same candidate pool")
    prefix = np.empty((core.shape[0], k), dtype=core.dtype)
    for out, first, second in zip(prefix, core, cs, strict=True):
        seen = set()
        for pair in zip(first, second, strict=True):
            for index in pair:
                if index in seen:
                    continue
                out[len(seen)] = index
                seen.add(index)
                if len(seen) == k:
                    break
            if len(seen) == k:
                break
        if len(seen) != k:
            raise ValueError("Interleaving did not produce the complete shortlist")
    return prefix


def attach_prefix(core, prefix):
    """Put the selected prefix first, preserving CORE relative order elsewhere."""
    if core.ndim != 2 or prefix.ndim != 2:
        raise ValueError("Expected aligned matrices of candidate indices")
    k = prefix.shape[1]
    if prefix.shape[0] != core.shape[0] or not 0 < k <= core.shape[1]:
        raise ValueError("Incompatible shortlist and universe")
    order = np.empty_like(core)
    order[:, :k] = prefix
    for row in range(len(core)):
        tail = core[row, ~np.isin(core[row], prefix[row])]
        if len(tail) != core.shape[1] - k:
            raise ValueError("Shortlist indices are duplicated or outside the universe")
        order[row, k:] = tail
    return order


def rerank_shortlist(core, prefix, scores):
    if scores.shape != prefix.shape or not np.isfinite(scores).all():
        raise ValueError("Every admitted candidate must have a finite cross score")
    # CORE rank breaks cross-score ties even for candidates outside CORE's top K.
    inverse = np.empty_like(core)
    np.put_along_axis(inverse, core, np.arange(core.shape[1])[None, :], axis=1)
    core_positions = np.take_along_axis(inverse, prefix, axis=1)
    priorities = np.lexsort((core_positions, -scores), axis=1)
    return attach_prefix(core, np.take_along_axis(prefix, priorities, axis=1))


def admission(ordered_labels, depth):
    """Oracle AP and hit counts for fixed membership and an unchanged tail."""
    hits = ordered_labels[:, :depth].sum(axis=1)
    oracle = np.concatenate((np.arange(depth)[None, :] < hits[:, None],
                             ordered_labels[:, depth:]), axis=1)
    return {"hits": hits, "oracle_ap_pp": 100 * original.average_precision(oracle)}


def query_metrics(order, relevance, depth):
    labels = np.take_along_axis(relevance, order, axis=1)
    details = admission(labels, depth)
    details["ap_pp"] = 100 * original.average_precision(labels)
    if (details["ap_pp"] > details["oracle_ap_pp"] + 1e-9).any():
        raise ValueError("Actual ordering exceeded its label oracle")
    return details


def aggregate(metrics, positives):
    return {"map_pp": float(metrics["ap_pp"].mean()),
            "oracle_map_pp": float(metrics["oracle_ap_pp"].mean()),
            "recall_pp": float(100 * (metrics["hits"] / positives).mean()),
            "queries_without_hit": int((metrics["hits"] == 0).sum())}


def contrast(base, alternate, positives, groups):
    _, group = np.unique(groups, return_inverse=True)
    counts = np.bincount(group)
    draws = np.random.default_rng(BOOT_SEED).integers(0, len(counts), (DRAWS, len(counts)))
    denominator = counts[draws].sum(axis=1)

    map_gain = alternate["ap_pp"] - base["ap_pp"]
    oracle_gain = alternate["oracle_ap_pp"] - base["oracle_ap_pp"]
    quantities = {"map_gain": map_gain,
                  "recall_gain": 100 * (alternate["hits"] - base["hits"]) / positives,
                  "admission_oracle_gain": oracle_gain,
                  "ranking_gap_reduced": map_gain - oracle_gain}
    differences = {}
    for name, values in quantities.items():
        totals = np.bincount(group, weights=values)
        interval = np.quantile(totals[draws].sum(axis=1) / denominator, [0.025, 0.975])
        differences[name] = {"point_pp": float(values.mean()), "ci95_pp": interval.tolist()}
    point = differences["map_gain"]["point_pp"]
    lo, hi = differences["map_gain"]["ci95_pp"]
    state = "PASS" if point >= MIN_GAIN_PP and lo > 0 else "FAIL" if hi < MIN_GAIN_PP else "INCONCLUSIVE"
    return state, differences


def check_replay(base):
    if (base["queries_without_hit"] != 31 or
            any(abs(base[key] - expected) > REPLAY_TOLERANCE_PP for key, expected in PUBLISHED.items())):
        raise ValueError(f"Published CORE baseline did not replay within the frozen tolerance: {base}")


def timeline(torch, started, items, unit):
    torch.cuda.synchronize()
    elapsed = monotonic() - started
    return {"seconds": elapsed, "unit": unit, "items": items,
            "items_per_second": items / elapsed,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved()}


def search(panel, config, snapshots):
    import torch

    ids, queries = panel["ids"], panel["queries"]
    captions = [q["text"] for q in queries]
    images = [lambda key=key: original.rgb(original.image_path(key)) for key in ids]
    runtimes = {}
    orders = {}
    ties = {}
    for name, model in (("core_embed", original.Encoder), ("cs_clip", original.ClipEncoder)):
        print(f"Indexing {len(ids)} frozen images with {name}", flush=True)
        torch.cuda.reset_peak_memory_stats()
        started = monotonic()
        encoder = model(name, snapshots[name], config, "cuda")
        vectors, text = original.encode_items(encoder, images, captions, config)
        scores = text @ vectors.T
        orders[name] = original.ranks(scores)
        boundary = np.take_along_axis(scores, orders[name][:, K-1:K+1], axis=1)
        ties[name] = int((boundary[:, 0] - boundary[:, 1] <= BOUNDARY_TOLERANCE).sum())
        runtimes[name] = timeline(torch, started, len(images), "indexed_images")
        del encoder, vectors, text, scores
        torch.cuda.empty_cache()

    core = orders["core_embed"]
    core_prefix = core[:, :K]
    alt_prefix = interleave(core, orders["cs_clip"])
    shared_pairs = [set(row) for row in core_prefix]
    novel = [(i, index) for i, row in enumerate(alt_prefix)
             for index in row if index not in shared_pairs[i]]

    inverse = {key: i for i, key in enumerate(ids)}
    relevance = np.zeros_like(core, dtype=bool)
    for row, query in enumerate(queries):
        relevance[row, [inverse[key] for key in query["relevant"]]] = True
    positives = relevance.sum(axis=1)
    groups = [q["group"] for q in queries]
    batch = config["batch_size"]["cross_pair"]

    print(f"Scoring the original {core_prefix.size} cross-encoder pairs in original batch order", flush=True)
    torch.cuda.reset_peak_memory_stats()
    started = monotonic()
    scorer = original.CrossScorer("core_reranker", snapshots["core_reranker"], config, "cuda")
    common_scores = original.cross_scores(scorer, [caption for caption in captions for _ in range(K)],
                                          [images[index] for index in core_prefix.flat], batch).reshape(core_prefix.shape)
    runtimes["core_cross_encoder"] = timeline(torch, started, core_prefix.size, "pairs")
    base = query_metrics(rerank_shortlist(core, core_prefix, common_scores), relevance, K)
    check_replay(aggregate(base, positives))

    print(f"Scoring {len(novel)} interleaved pairs outside the original shortlist", flush=True)
    torch.cuda.reset_peak_memory_stats()
    started = monotonic()
    novel_scores = original.cross_scores(scorer, [captions[i] for i, _ in novel],
                                         [images[index] for _, index in novel], batch)
    runtimes["extended_cross_encoder"] = timeline(torch, started, len(novel), "new_pairs")
    union_scores = [{index: score for index, score in zip(prefix, scores, strict=True)}
                    for prefix, scores in zip(core_prefix, common_scores, strict=True)]
    for (row, index), score in zip(novel, novel_scores, strict=True):
        union_scores[row][index] = score
    alt_scores = np.asarray([[union_scores[i][index] for index in row] for i, row in enumerate(alt_prefix)])
    alternate = query_metrics(rerank_shortlist(core, alt_prefix, alt_scores), relevance, K)
    rank_labels = np.take_along_axis(relevance, core, axis=1)
    curve = {k: admission(rank_labels, k) for k in DEPTHS}
    state, deltas = contrast(base, alternate, positives, groups)
    report = {"core": aggregate(base, positives), "interleaved": aggregate(alternate, positives),
              "state": state, "interleaved_minus_core": deltas,
              "core_admission_curve": {k: {"oracle_map_pp": float(details["oracle_ap_pp"].mean()),
                                            "recall_pp": float(100 * (details["hits"] / positives).mean()),
                                            "queries_without_hit": int((details["hits"] == 0).sum())}
                                       for k, details in curve.items()},
              "runtime": runtimes, "classified_unique_pairs": core_prefix.size + len(novel),
              "pairs_per_query_per_arm": K, "queries_with_near_top50_score_tie": ties}
    _, group_codes = np.unique(groups, return_inverse=True)
    details = {"group_codes_in_panel_order": group_codes.tolist(), "positives": positives.tolist(),
               "arms": {name: {key: values.tolist() for key, values in metrics.items()}
                        for name, metrics in (("core", base), ("interleaved", alternate))},
               "curve": {k: {key: values.tolist() for key, values in metrics.items()}
                         for k, metrics in curve.items()}}
    return report, details


def preflight():
    panel, manifest_sha, coverage = original.preflight()
    config, snapshots = original.models()
    counts = (len(panel["ids"]), len(panel["queries"]), len({q["group"] for q in panel["queries"]}))
    if counts != PANEL_COUNTS:
        raise ValueError("The replay requires the exact published query and candidate counts")
    source = original.DATA.parent.parent / "runs" / "retrieval-anchor-rerank" / SOURCE_JOB / "summary.json"
    if source.is_symlink() or original.file_sha(source) != SOURCE_SUMMARY_SHA:
        raise ValueError("Published summary does not match the verified source receipt")
    context = {"source_job": SOURCE_JOB, "source_commit": SOURCE,
               "source_summary_sha256": SOURCE_SUMMARY_SHA, "scope": "fixed prior panel, post-hoc diagnostic",
               "panel_sha256": manifest_sha, "coverage": coverage,
               "model_config_sha256": original.MODEL_CONFIG_SHA, "depths": DEPTHS,
               "revisions": {name: path.name for name, path in snapshots.items()}}
    return panel, config, snapshots, context


def write_json(path, content):
    with path.open("x") as stream:
        json.dump(content, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--score", action="store_true")
    parser.add_argument("--freeze")
    args = parser.parse_args()
    panel, config, snapshots, context = preflight()
    if args.preflight:
        print(json.dumps(context, sort_keys=True))
        return
    job = os.environ.get("SLURM_JOB_ID", "")
    if not re.fullmatch(r"[1-9][0-9]*", job) or not re.fullmatch(r"[0-9a-f]{40}", args.freeze or ""):
        parser.error("Score requires a new Slurm job and reviewed full freeze")
    output = original.DATA.parent.parent / "runs" / "retrieval-recall-budget" / job
    output.mkdir(parents=True, exist_ok=False)
    context.update(freeze=args.freeze, slurm_job_id=job,
                   versions={name: version(name) for name in ("numpy", "torch", "transformers")})
    result, details = search(panel, config, snapshots)
    metrics_path = output / "per_query.json"
    write_json(metrics_path, details)
    partial = output / "summary.partial"
    write_json(partial, context | result | {"per_query_sha256": original.file_sha(metrics_path)})
    receipt = output / "summary.json"
    partial.replace(receipt)
    print(f"[codex] Recall-budget ranking summary SHA-256 {original.file_sha(receipt)}", flush=True)


if __name__ == "__main__":
    main()
