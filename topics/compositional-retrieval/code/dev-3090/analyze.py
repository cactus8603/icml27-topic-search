"""CPU analyses on a cache_scores.py output: baseline replay, D1/D2 diagnostics, M1 admission, M2 budgets.

    python analyze.py --cache DIR [--out DIR/analysis-<time>]

Everything here is development evidence on an exposed pool. Policies and weights picked here must be
frozen in a PREREG and re-tested on the sealed independent panel before any claim.
Needs only numpy.
"""

import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np


K = 50
DEPTHS = (50, 100, 150, 200)
BOOT_SEED = 2026111662
DRAWS = 10_000
GATE = {"recall_gain_pp": 5.0, "map_gain_pp": 2.0}
PUBLISHED = {"map_pp": 54.53, "oracle_map_pp": 74.02, "recall_pp": 73.77, "queries_without_hit": 31}
TOLERANCE_PP = 0.10


# ---------------------------------------------------------------- metrics

def average_precision(labels):
    """Per-row AP over a full ranking (same formula as the frozen evaluator)."""
    positives = labels.sum(axis=1)
    if np.any(positives <= 0):
        raise ValueError("Every query needs a positive in the pool")
    precision = labels.cumsum(axis=1) / np.arange(1, labels.shape[1] + 1)
    return (labels * precision).sum(axis=1) / positives


def final_order(core, prefix, scores):
    """Sort the admitted prefix by cross score (ties by CORE rank); keep CORE order for the tail."""
    if prefix.shape != scores.shape or not np.isfinite(scores).all():
        raise ValueError("Each admitted candidate needs a finite cross score")
    inverse = np.empty_like(core)
    np.put_along_axis(inverse, core, np.arange(core.shape[1])[None, :], axis=1)
    order = np.empty_like(core)
    for row in range(len(core)):
        chosen = prefix[row][np.lexsort((inverse[row, prefix[row]], -scores[row]))]
        tail = core[row][~np.isin(core[row], prefix[row])]
        if len(chosen) + len(tail) != core.shape[1] or len(set(prefix[row].tolist())) != len(prefix[row]):
            raise ValueError("Admitted candidates are duplicated or outside the pool")
        order[row, :len(chosen)] = chosen
        order[row, len(chosen):] = tail
    return order


def measure(order, relevance, budget):
    """AP, hits in the reranked prefix and the fixed-membership oracle. budget may vary per query."""
    labels = np.take_along_axis(relevance, order, axis=1)
    budget = np.broadcast_to(np.asarray(budget), (len(order),))
    hits = np.array([labels[i, :b].sum() for i, b in enumerate(budget)])
    oracle = labels.copy()
    for i, b in enumerate(budget):
        oracle[i, :b] = np.arange(b) < hits[i]
    return {"ap": 100 * average_precision(labels), "oracle": 100 * average_precision(oracle),
            "hits": hits, "budget": budget}


def summary(m, positives):
    return {"map_pp": float(m["ap"].mean()), "oracle_map_pp": float(m["oracle"].mean()),
            "recall_pp": float(100 * (m["hits"] / positives).mean()),
            "queries_without_hit": int((m["hits"] == 0).sum()), "mean_budget": float(m["budget"].mean())}


def paired(base, other, positives, groups, seed=BOOT_SEED, draws=DRAWS):
    """Query-group bootstrap of paired gains, as in the frozen recall-budget scorer."""
    _, group = np.unique(groups, return_inverse=True)
    counts = np.bincount(group)
    picks = np.random.default_rng(seed).integers(0, len(counts), (draws, len(counts)))
    denominator = counts[picks].sum(axis=1)
    out = {}
    for name, values in (("map_gain", other["ap"] - base["ap"]),
                         ("recall_gain", 100 * (other["hits"] - base["hits"]) / positives)):
        totals = np.bincount(group, weights=values)
        lo, hi = np.quantile(totals[picks].sum(axis=1) / denominator, [0.025, 0.975])
        out[name] = {"point_pp": float(values.mean()), "ci95_pp": [float(lo), float(hi)]}
    out["meets_dev_gate"] = bool(out["map_gain"]["point_pp"] >= GATE["map_gain_pp"]
                                 and out["map_gain"]["ci95_pp"][0] > 0
                                 and out["recall_gain"]["point_pp"] >= GATE["recall_gain_pp"])
    return out


# ---------------------------------------------------------------- scores used by admission policies

def zscore(scores):
    """Standardize each text's scores over the whole pool (rows = texts)."""
    return (scores - scores.mean(axis=1, keepdims=True)) / scores.std(axis=1, keepdims=True)


def rank_percentile(values):
    """Within-row rank percentile in (0, 1]; ties share the mean rank."""
    out = np.empty(values.shape)
    for row, v in enumerate(values):
        order = np.argsort(v, kind="stable")
        sorted_v = v[order]
        edges = np.r_[0, 1 + np.flatnonzero(sorted_v[:-1] != sorted_v[1:]), len(v)]
        mean_rank = (edges[:-1] + edges[1:] + 1) / (2 * len(v))
        out[row, order] = np.repeat(mean_rank, np.diff(edges))
    return out


def log_sigmoid(x):
    return -np.logaddexp(0, -x)


def candidate_features(cache, depth):
    """Scores of the CORE top-depth candidates for every query: full sentence and per condition."""
    core = cache["core_order"]
    top = core[:, :depth]
    feats = {}
    for model in ("core_embed", "cs_clip"):
        image = cache[f"{model}_image"]
        full = zscore(cache[f"{model}_full"] @ image.T)
        cond = zscore(cache[f"{model}_condition"] @ image.T)
        index = np.asarray(cache["condition_index"])
        feats[f"{model}_full"] = np.take_along_axis(full, top, axis=1)
        feats[f"{model}_cond"] = np.stack([np.take_along_axis(cond[index[:, j]], top, axis=1)
                                           for j in range(index.shape[1])], axis=-1)
    return top, feats


def policies(feats):
    """Admission scores over the top-depth candidates. Higher = admit first. Ties fall back to CORE rank."""
    depth = feats["core_embed_full"].shape[1]
    core_rank = -np.arange(depth, dtype=float)[None, :].repeat(len(feats["core_embed_full"]), 0)
    out = {"core": core_rank, "cs_full": feats["cs_clip_full"]}
    for model, short in (("core_embed", "core"), ("cs_clip", "cs")):
        cond = feats[f"{model}_cond"]
        out[f"min_cond_{short}"] = cond.min(axis=-1)
        for tau in (0.5, 1.0, 2.0):
            out[f"llr_{short}_t{tau:g}"] = log_sigmoid(cond / tau).sum(axis=-1)
    base = rank_percentile(core_rank)
    for name in ("min_cond_core", "min_cond_cs", "llr_core_t1", "llr_cs_t1", "cs_full"):
        for w in (0.25, 0.5, 0.75):
            out[f"fuse_{name}_w{w:g}"] = (1 - w) * base + w * rank_percentile(out[name])
    return out


def admit(top, admission, k=K):
    """Pick k of the top-depth candidates by the admission score; ties by CORE rank."""
    depth = top.shape[1]
    keep = np.empty((len(top), k), dtype=top.dtype)
    for row in range(len(top)):
        order = np.lexsort((np.arange(depth), -admission[row]))[:k]
        keep[row] = top[row, np.sort(order)]
    return keep


def cross_for(prefix, top, cross):
    position = [{c: j for j, c in enumerate(row)} for row in top]
    return np.array([[cross[i, position[i][c]] for c in row] for i, row in enumerate(prefix)])


# ---------------------------------------------------------------- diagnostics

def kinds(categories, ids, queries):
    """Per query x image: 2 = both conditions, 1 = only first, -1 = only second, 0 = neither."""
    out = np.zeros((len(queries), len(ids)), dtype=np.int8)
    sets = [set(categories[str(i)]) for i in ids]
    for row, q in enumerate(queries):
        a, b = q["key"]
        for col, s in enumerate(sets):
            out[row, col] = 2 if a in s and b in s else 1 if a in s else -1 if b in s else 0
    return out


def composition(kind_rows):
    total = kind_rows.size
    return {"both": float((kind_rows == 2).sum() / total),
            "one_condition": float((np.abs(kind_rows) == 1).sum() / total),
            "neither": float((kind_rows == 0).sum() / total)}


def d1_recall(core, relevance, kind, feats_cond_full):
    """Where missed positives sit in CORE order, and which condition is their weak link."""
    ranks = np.argsort(core, axis=1)
    pos_rank = [ranks[i, relevance[i]] for i in range(len(core))]
    missed = np.concatenate([r[r >= K] for r in pos_rank])
    no_hit = [i for i, r in enumerate(pos_rank) if r.min() >= K]
    first_ranks = np.array([r.min() for r in pos_rank])
    weak = []
    cond = feats_cond_full  # (queries, 2, pool) z-scored per text, CORE-EMBED
    for i in range(len(core)):
        miss = np.flatnonzero(relevance[i] & (ranks[i] >= K))
        if len(miss):
            pct = np.stack([(cond[i, j][None, :] >= cond[i, j][miss][:, None]).mean(axis=1) for j in range(2)])
            weak.append(pct.max(axis=0))  # worse (larger) pool-rank fraction of the two conditions
    weak = np.concatenate(weak) if weak else np.array([])
    return {"missed_positives": int(len(missed)),
            "missed_rank_bins": {f"{lo + 1}-{hi}": int(((missed >= lo) & (missed < hi)).sum())
                                 for lo, hi in ((50, 100), (100, 200), (200, 500), (500, 10**9))},
            "queries_without_top50_hit": len(no_hit),
            "no_hit_first_positive_rank": sorted((first_ranks[no_hit] + 1).tolist()),
            "top50_composition": composition(np.take_along_axis(kind, core[:, :K], axis=1)),
            "missed_weaker_condition_pool_fraction_quartiles":
                np.quantile(weak, [0.25, 0.5, 0.75]).tolist() if len(weak) else None}


def d2_ranking(order, relevance, kind, budget=K):
    """Inside the reranked prefix: which negatives outrank positives."""
    counts = {"one_condition": 0, "neither": 0}
    affected = 0
    for i in range(len(order)):
        row = order[i, :budget]
        rel = relevance[i, row]
        if not rel.any():
            continue
        last = np.flatnonzero(rel).max()
        above = row[:last][~rel[:last]]
        affected += bool(len(above))
        k = kind[i, above]
        counts["one_condition"] += int((np.abs(k) == 1).sum())
        counts["neither"] += int((k == 0).sum())
    return {"queries_with_negative_above_a_positive": affected, "negatives_above_last_positive": counts}


# ---------------------------------------------------------------- M2 budgets

def quartile_budgets(uncertainty, levels=(75, 50, 50, 25)):
    """Most uncertain quartile gets the largest budget; the mean stays K."""
    order = np.argsort(-uncertainty, kind="stable")
    budgets = np.empty(len(uncertainty), dtype=int)
    for part, level in zip(np.array_split(order, len(levels)), levels):
        budgets[part] = level
    return budgets


def budgeted(core, cross, budgets):
    order = core.copy()
    for i, b in enumerate(budgets):
        order[i:i + 1] = final_order(core[i:i + 1], core[i:i + 1, :b], cross[i:i + 1, :b])
    return order


# ---------------------------------------------------------------- driver

def load(cache_dir):
    cache = dict(np.load(cache_dir / "vectors.npz")) | dict(np.load(cache_dir / "cross.npz"))
    panel = json.loads((cache_dir / "panel.json").read_text())
    cache["condition_index"] = panel["condition_index"]
    return cache, panel


def analyse(cache, panel):
    ids, queries = panel["ids"], panel["queries"]
    core, cross = cache["core_order"].astype(np.int64), cache["cross"]
    depth = cross.shape[1]
    inverse = {key: col for col, key in enumerate(ids)}
    relevance = np.zeros(core.shape, dtype=bool)
    for row, q in enumerate(queries):
        relevance[row, [inverse[key] for key in q["relevant"]]] = True
    positives = relevance.sum(axis=1)
    groups = [q["group"] for q in queries]
    kind = kinds(panel["image_categories"], ids, queries)
    report = {"depth_cached": depth, "queries": len(queries), "pool": len(ids)}

    base_order = final_order(core, core[:, :K], cross[:, :K])
    base = measure(base_order, relevance, K)
    report["baseline"] = summary(base, positives)
    report["baseline_replays_published"] = all(
        abs(report["baseline"][k] - v) <= (0 if k == "queries_without_hit" else TOLERANCE_PP)
        for k, v in PUBLISHED.items())

    report["rerank_all_ceiling"] = {}
    for d in [d for d in DEPTHS if d <= depth]:
        m = measure(final_order(core, core[:, :d], cross[:, :d]), relevance, d)
        report["rerank_all_ceiling"][d] = summary(m, positives) | {"cost_pairs_per_query": d}

    cond_full = np.stack([zscore(cache["core_embed_condition"] @ cache["core_embed_image"].T)[
        np.asarray(panel["condition_index"])[:, j]] for j in range(2)], axis=1)
    report["d1_recall"] = d1_recall(core, relevance, kind, cond_full)
    report["d2_ranking"] = d2_ranking(base_order, relevance, kind)

    report["m1_admission"] = {}
    for d in [d for d in (100, 200) if d <= depth]:
        top, feats = candidate_features(cache, d)
        ceiling = report["rerank_all_ceiling"][d]["map_pp"] - report["baseline"]["map_pp"]
        rows = {}
        for name, admission in policies(feats).items():
            prefix = admit(top, admission)
            m = measure(final_order(core, prefix, cross_for(prefix, top, cross[:, :d])), relevance, K)
            rows[name] = summary(m, positives) | paired(base, m, positives, groups)
            rows[name]["share_of_rerank_all_gain"] = (
                float((m["ap"].mean() - base["ap"].mean()) / ceiling) if ceiling > 0 else None)
        report["m1_admission"][f"from_top{d}"] = dict(
            sorted(rows.items(), key=lambda item: -item[1]["map_pp"]))

    scores = cache["core_embed_full"] @ cache["core_embed_image"].T
    sorted_scores = np.take_along_axis(scores, core[:, :depth], axis=1)
    uncertainty = -(sorted_scores[:, K - 1] - sorted_scores[:, depth - 1])  # flatter tail = more near-ties
    budgets = quartile_budgets(uncertainty)
    m = measure(budgeted(core, cross, budgets), relevance, budgets)
    report["m2_quartile_budget"] = summary(m, positives) | paired(base, m, positives, groups) | {
        "levels": [75, 50, 50, 25], "uncertainty": "CORE-EMBED score drop from rank 50 to rank depth"}
    return report


def markdown(report, meta):
    b = report["baseline"]
    lines = [f"# dev-3090 analysis ({datetime.now():%Y-%m-%d %H:%M})", "",
             f"Device {meta.get('device')}, cache replay {meta.get('replay', {}).get('state')}, "
             f"analysis replay {'PASS' if report['baseline_replays_published'] else 'MISMATCH'}. "
             "Exposed pool, development only.", "",
             f"Baseline CORE top-50: mAP {b['map_pp']:.2f}, Recall@50 {b['recall_pp']:.2f}, "
             f"oracle {b['oracle_map_pp']:.2f}, no-hit {b['queries_without_hit']}.", "",
             "| rerank all top-d | mAP | Recall@d | no-hit |", "|---|---:|---:|---:|"]
    for d, r in report["rerank_all_ceiling"].items():
        lines.append(f"| {d} | {r['map_pp']:.2f} | {r['recall_pp']:.2f} | {r['queries_without_hit']} |")
    for block, rows in report["m1_admission"].items():
        lines += ["", f"## M1 admission, 50 reranks chosen {block.replace('_', ' ')}", "",
                  "| policy | mAP | ΔmAP [95%] | ΔRecall@50 | no-hit | dev gate |", "|---|---:|---:|---:|---:|:-:|"]
        for name, r in list(rows.items())[:12]:
            g, rc = r["map_gain"], r["recall_gain"]
            lines.append(f"| {name} | {r['map_pp']:.2f} | {g['point_pp']:+.2f} [{g['ci95_pp'][0]:+.2f}, "
                         f"{g['ci95_pp'][1]:+.2f}] | {rc['point_pp']:+.2f} | {r['queries_without_hit']} | "
                         f"{'✓' if r['meets_dev_gate'] else ''} |")
    m2 = report["m2_quartile_budget"]
    lines += ["", f"M2 quartile budgets {m2['levels']}: mAP {m2['map_pp']:.2f} "
              f"(Δ {m2['map_gain']['point_pp']:+.2f}), mean budget {m2['mean_budget']:.1f}.", "",
              f"D1: {json.dumps(report['d1_recall'], ensure_ascii=False)}", "",
              f"D2: {json.dumps(report['d2_ranking'], ensure_ascii=False)}"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    out = args.out or args.cache / f"analysis-{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True, exist_ok=False)
    cache, panel = load(args.cache)
    meta = json.loads((args.cache / "meta.json").read_text())
    report = analyse(cache, panel)
    (out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    (out / "report.md").write_text(markdown(report, meta))
    print((out / "report.md").read_text())


if __name__ == "__main__":
    main()
