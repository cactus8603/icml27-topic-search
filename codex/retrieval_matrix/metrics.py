"""[codex] CPU retrieval scoring. Labels never participate in candidate admission.

The producer chooses K from the complete gallery. Ignore IDs are removed only
while scoring that fixed prefix; the prefix is never refilled. AP@K divides by
all relevant images, so the admitted-positive oracle is a useful ceiling.
"""
from __future__ import annotations

import numpy as np


def stable_order(scores):
    scores = np.asarray(scores)
    if scores.ndim != 2 or not np.isfinite(scores).all():
        raise ValueError("Expected a finite query-by-gallery score matrix")
    return np.argsort(-scores, axis=1, kind="stable").astype(np.int32)


def validate_order(order, gallery_size):
    order = np.asarray(order)
    if (order.ndim != 2 or not np.issubdtype(order.dtype, np.integer)
            or order.shape[1] != gallery_size
            or not np.array_equal(np.sort(order, axis=1), np.broadcast_to(np.arange(gallery_size), order.shape))):
        raise ValueError("Every ranking must be a full permutation of the gallery")
    return order


def rerank_prefix(order, cross, k):
    """Stable cross ties retain embedding order; the original tail is untouched."""
    order, cross = np.asarray(order), np.asarray(cross)
    if (order.ndim != 2 or cross.ndim != 2 or cross.shape[0] != len(order)
            or type(k) is not int or not 1 <= k <= order.shape[1]
            or cross.shape[1] < k or not np.isfinite(cross).all()):
        raise ValueError("Cross scores must cover the complete original fixed prefix")
    result = order.copy()
    result[:, :k] = np.take_along_axis(order[:, :k], stable_order(cross[:, :k]), axis=1)
    return result


def query_metrics(order, positive_indices, ignore_indices=(), *, ks=(1, 5, 10, 50, 100, 200)):
    """All values are fractions; evaluate prefixes BEFORE removing ignore IDs."""
    order = np.asarray(order)
    if order.ndim != 1 or not np.issubdtype(order.dtype, np.integer) or len(set(order.tolist())) != len(order):
        raise ValueError("Expected a unique one-dimensional ranking")
    positive, ignore = set(positive_indices), set(ignore_indices)
    if not positive or positive & ignore or not (positive | ignore) <= set(order.tolist()):
        raise ValueError("Positive/ignore labels must be disjoint gallery members with nonempty positives")
    if any(type(k) is not int or k <= 0 for k in ks):
        raise ValueError("Evaluation cutoffs must be positive integers")
    filtered = [int(i) for i in order if i not in ignore]
    full_rel = np.array([i in positive for i in filtered], dtype=np.float64)
    full_ranks = np.flatnonzero(full_rel) + 1
    output = {"MRR": float(1 / full_ranks[0]), "AP": float(np.sum(np.arange(1, len(full_ranks)+1)/full_ranks)/len(positive)),
              "positive_count": len(positive), "ignore_count": len(ignore)}
    for k in ks:
        raw_prefix = order[:k]
        prefix = [int(i) for i in raw_prefix if i not in ignore]
        rel = np.array([i in positive for i in prefix], dtype=np.float64)
        ranks = np.flatnonzero(rel) + 1
        hits = len(ranks)
        ap = float(np.sum(np.arange(1, hits+1)/ranks)/len(positive))
        dcg = float(np.sum(1/np.log2(ranks+1)))
        ideal_dcg = float(np.sum(1/np.log2(np.arange(1, min(len(positive), k)+1)+1)))
        output.update({f"R@{k}": hits/len(positive), f"Success@{k}": float(hits > 0),
                       f"MRR@{k}": float(1/ranks[0]) if hits else 0.0,
                       f"AP@{k}": ap, f"nDCG@{k}": dcg/ideal_dcg,
                       f"oracle_AP@{k}": hits/len(positive),
                       f"oracle_Success@{k}": float(hits > 0),
                       f"admitted@{k}": min(k, len(order)), f"evaluated@{k}": len(prefix),
                       f"ignored_admitted@{k}": len(raw_prefix)-len(prefix)})
    return output


def evaluate(order, gallery_ids, labels, *, ks=(1, 5, 10, 50, 100, 200)):
    order = validate_order(order, len(gallery_ids))
    if len(labels) != len(order) or len(set(map(str, gallery_ids))) != len(gallery_ids):
        raise ValueError("Labels/rankings/gallery identity mismatch")
    lookup = {str(v): i for i, v in enumerate(gallery_ids)}
    detail = []
    for row, label in zip(order, labels, strict=True):
        try:
            positives = [lookup[str(v)] for v in label["positives"]]
            ignored = [lookup[str(v)] for v in label.get("ignore", [])]
        except KeyError as error:
            raise ValueError("Scoring label refers outside the gallery") from error
        values = query_metrics(row, positives, ignored, ks=ks)
        detail.append({"id": label["id"], "group": label.get("group", label["id"]), **values})
    if not detail:
        raise ValueError("No eligible queries to score")
    fields = [key for key in detail[0] if key not in {"id", "group", "positive_count", "ignore_count"}]
    summary = {key: float(np.mean([row[key] for row in detail])) for key in fields}
    return {"queries": len(detail), "units": "fractions; admission/evaluated fields are counts", "metrics": summary}, detail


def paired_cluster_bootstrap(left, right, groups, *, seed, draws=10000):
    """Paired query-weighted difference, resampling complete declared clusters."""
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if (left.ndim != 1 or left.shape != right.shape or len(groups) != len(left) or not len(left)
            or not np.isfinite(left).all() or not np.isfinite(right).all()
            or type(draws) is not int or draws <= 0):
        raise ValueError("Malformed paired bootstrap input")
    unique = list(dict.fromkeys(groups))
    indices = [np.flatnonzero(np.array(groups) == group) for group in unique]
    differences = right-left
    sums = np.array([differences[i].sum() for i in indices])
    counts = np.array([len(i) for i in indices])
    rng = np.random.default_rng(seed)
    sampled = np.empty(draws)
    for i in range(draws):
        selected = rng.integers(0, len(unique), len(unique))
        sampled[i] = sums[selected].sum()/counts[selected].sum()
    return {"delta": float(differences.mean()), "ci95": np.quantile(sampled, [.025, .975]).tolist(),
            "queries": len(left), "clusters": len(unique), "seed": seed, "draws": draws}
