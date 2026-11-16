"""[codex] CPU-only, prespecified four-cell CORE scaling analysis.

The functional API consumes aligned arrays and does not open study results.
Cross scores must follow the corresponding retriever's original top-50 order.
Every inference statement is conditional on this exposed, rule-labelled panel.
"""

from __future__ import annotations

import hashlib
import argparse
import json
from pathlib import Path
import platform
import re
import subprocess

import numpy as np


K = 50
DRAWS = 10_000
BOOT_SEED = 2026111668
EPSILON_PP = 2.0
EXPECTED_CONNECTED_GROUPS = 110
CELL_NAMES = ("retrieve2/rerank2", "retrieve2/rerank8", "retrieve8/rerank2", "retrieve8/rerank8")
PAIRED_METRICS = ("ap_pp", "recall_at_50_pp", "oracle_ap_pp", "ranking_gap_pp")
INPUT_FILES = {"order2": "index/2b_order.npy", "order8": "index/8b_order.npy",
               "cross22": "cross/e2_r2.npy", "cross28": "cross/e2_r8.npy",
               "cross82": "cross/e8_r2.npy", "cross88": "cross/e8_r8.npy"}


def _integer(value, description):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{description} must be an integer")
    return int(value)


def connected_query_groups(queries):
    """Connect identical original groups and shared positives, transitively."""
    groups = [_integer(query["group"], "Original query group") for query in queries]
    parent = {group: group for group in groups}

    def find(group):
        while parent[group] != group:
            parent[group] = parent[parent[group]]
            group = parent[group]
        return group

    owners = {}
    for query, group in zip(queries, groups, strict=True):
        for image_id in query["relevant"]:
            if image_id in owners:
                left, right = find(group), find(owners[image_id])
                parent[max(left, right)] = min(left, right)
            else:
                owners[image_id] = group
    return np.asarray([find(group) for group in groups], dtype=np.int64)


def _validate(panel, orders, crosses):
    ids, queries = panel["ids"], panel["queries"]
    if len(ids) < K or not queries or len(set(ids)) != len(ids):
        raise ValueError("Need a nonempty query panel and at least 50 unique gallery IDs")
    positions = {image_id: index for index, image_id in enumerate(ids)}
    relevance = np.zeros((len(queries), len(ids)), dtype=bool)
    for row, query in enumerate(queries):
        relevant = query["relevant"]
        _integer(query["group"], "Original query group")
        if not relevant or len(set(relevant)) != len(relevant) or not set(relevant) <= positions.keys():
            raise ValueError("Every query needs unique, in-gallery relevant IDs")
        relevance[row, [positions[image_id] for image_id in relevant]] = True
    validated_orders = []
    for order in orders:
        order = np.asarray(order)
        if (order.shape != relevance.shape or not np.issubdtype(order.dtype, np.integer)
                or not np.all(np.sort(order, axis=1) == np.arange(len(ids)))):
            raise ValueError("Each retriever order must be a full gallery-index permutation per query")
        validated_orders.append(order.astype(np.int64, copy=False))
    validated_crosses = []
    for scores in crosses:
        scores = np.asarray(scores)
        if (scores.shape != (len(queries), K) or not np.issubdtype(scores.dtype, np.number)
                or np.iscomplexobj(scores) or not np.isfinite(scores).all()):
            raise ValueError("Cross scores must be finite real (queries, 50) arrays aligned to original top50")
        validated_crosses.append(scores.astype(np.float64, copy=False))
    return relevance, validated_orders, validated_crosses


def final_order(retriever_order, cross_scores):
    """Rerank original top50, break score ties by original rank, retain tail."""
    ranked = retriever_order.copy()
    within_prefix = np.argsort(-cross_scores, axis=1, kind="stable")
    ranked[:, :K] = np.take_along_axis(retriever_order[:, :K], within_prefix, axis=1)
    return ranked


def average_precision(labels):
    """Full-gallery AP: all positives form the denominator, not just top50."""
    positive_count = labels.sum(axis=1)
    precision = labels.cumsum(axis=1) / np.arange(1, labels.shape[1] + 1)
    return (precision * labels).sum(axis=1) / positive_count


def _measure(relevance, order):
    labels = np.take_along_axis(relevance, order, axis=1)
    positives = relevance.sum(axis=1)
    hits = labels[:, :K].sum(axis=1)
    oracle = labels.copy()
    oracle[:, :K] = np.arange(K)[None, :] < hits[:, None]
    ap, oracle_ap = 100 * average_precision(labels), 100 * average_precision(oracle)
    if np.any(oracle_ap + 1e-10 < ap):
        raise AssertionError("Fixed-membership oracle must dominate its actual ranking")
    return {"ap_pp": ap, "recall_at_50_pp": 100 * hits / positives,
            "oracle_ap_pp": oracle_ap, "ranking_gap_pp": np.maximum(oracle_ap - ap, 0),
            "hits_at_50": hits, "no_top50_hit": hits == 0,
            "recall_capacity_at_50_pp": 100 * np.minimum(K / positives, 1),
            "reranked_prefix_indices": order[:, :K]}


def bootstrap_plan(groups, *, draws=DRAWS, seed=BOOT_SEED):
    draws, seed = _integer(draws, "Bootstrap draw count"), _integer(seed, "Bootstrap seed")
    if draws < 2:
        raise ValueError("Bootstrap needs at least two draws")
    _, codes = np.unique(groups, return_inverse=True)
    counts = np.bincount(codes)
    picks = np.random.default_rng(seed).integers(0, len(counts), (draws, len(counts)))
    return codes, picks, counts[picks].sum(axis=1)


def bootstrap_means(values, plan):
    """Draw groups equally; estimate the query-weighted mean in each draw."""
    codes, picks, denominator = plan
    values = np.asarray(values, dtype=np.float64)
    if values.shape != codes.shape or not np.isfinite(values).all():
        raise ValueError("Bootstrap values must be a finite per-query vector")
    totals = np.bincount(codes, weights=values)
    return totals[picks].sum(axis=1) / denominator


def estimate(values, plan):
    samples = bootstrap_means(values, plan)
    return {"point_pp": float(np.mean(values)),
            "ci95_two_sided_pp": np.quantile(samples, [.025, .975]).tolist(),
            "lower95_one_sided_pp": float(np.quantile(samples, .05)),
            "upper95_one_sided_pp": float(np.quantile(samples, .95)),
            "bootstrap_se_pp": float(np.std(samples, ddof=1))}


def _summary(metrics, plan):
    return {"map_pp": float(np.mean(metrics["ap_pp"])),
            "recall_at_50_pp": float(np.mean(metrics["recall_at_50_pp"])),
            "queries_without_top50_hit": int(np.sum(metrics["no_top50_hit"])),
            "oracle_map_pp": float(np.mean(metrics["oracle_ap_pp"])),
            "oracle_minus_actual_map_pp": float(np.mean(metrics["ranking_gap_pp"])),
            "recall_capacity_at_50_pp": float(np.mean(metrics["recall_capacity_at_50_pp"])),
            "uncertainty": {key: estimate(metrics[key], plan) for key in PAIRED_METRICS},
            "rerank_pairs_per_query": K}


def recovered_sorting_fraction(base_ap, larger_ap, base_oracle, plan):
    """Ratio of aggregate gains, not an average of undefined querywise ratios."""
    gain, original_gap = larger_ap - base_ap, base_oracle - base_ap
    denominator = float(np.mean(original_gap))
    sampled_denominator = bootstrap_means(original_gap, plan)
    valid = sampled_denominator > 1e-12
    result = {"definition": "mean(AP_2_8 - AP_2_2) / mean(oracle_2 - AP_2_2)",
              "point_fraction": float(np.mean(gain) / denominator) if denominator > 1e-12 else None,
              "gain_pp": estimate(gain, plan), "original_sorting_gap_pp": estimate(original_gap, plan),
              "undefined_bootstrap_draws": int((~valid).sum()),
              "ci95_two_sided_fraction": None,
              "interpretation": "May be negative; undefined if the original aggregate gap is zero. "
                                "A ratio interval is omitted if any bootstrap denominator is zero."}
    if valid.all():
        ratios = bootstrap_means(gain, plan) / sampled_denominator
        result["ci95_two_sided_fraction"] = np.quantile(ratios, [.025, .975]).tolist()
    return result


def classify_gap(gap_estimate, *, epsilon_pp=EPSILON_PP):
    """Strict boundaries: equality with epsilon remains uncertain."""
    if gap_estimate["upper95_one_sided_pp"] < epsilon_pp:
        return "below_epsilon"
    if gap_estimate["lower95_one_sided_pp"] > epsilon_pp:
        return "above_epsilon"
    return "uncertain"


def classify_space(ranking_gap, outside_oracle_gap, *, epsilon_pp=EPSILON_PP):
    ranking = classify_gap(ranking_gap, epsilon_pp=epsilon_pp)
    outside = classify_gap(outside_oracle_gap, epsilon_pp=epsilon_pp)
    if ranking == outside == "below_epsilon":
        verdict = "space_basically_absorbed_on_this_exposed_panel"
    elif "above_epsilon" in (ranking, outside):
        verdict = "detectable_remaining_map_gap"
    else:
        verdict = "uncertain"
    return {"epsilon_pp": epsilon_pp, "ranking_gap_state": ranking,
            "outside_fixed_candidate_oracle_gap_state": outside, "verdict": verdict,
            "rule": "Absorbed only if both one-sided 95% upper bounds are strictly below epsilon; "
                    "remaining if either one-sided 95% lower bound is strictly above epsilon; otherwise uncertain.",
            "scope": "Two prespecified, nominal marginal gap intervals on an exposed rule-labelled panel. "
                     "This is not a guarantee that a practical method can recover either oracle gap, "
                     "nor a universal saturation claim."}


def analyze_arrays(panel, order2, order8, cross22, cross28, cross82, cross88, *, draws=DRAWS):
    """Return JSON-safe (report, per_query); no result files are read or written.

    order2/order8 contain zero-based gallery indices in each retriever's full
    ranking. crossXY aligns with orderX[:, :50], not a reranked or union order.
    Optional draws exists for synthetic tests; study execution must use 10,000.
    """
    relevance, orders, crosses = _validate(panel, (order2, order8), (cross22, cross28, cross82, cross88))
    queries = panel["queries"]
    original_groups = np.asarray([query["group"] for query in queries], dtype=np.int64)
    groups = connected_query_groups(queries)
    plan = bootstrap_plan(groups, draws=draws)
    methods = {name: _measure(relevance, final_order(orders[index // 2], scores))
               for index, (name, scores) in enumerate(zip(CELL_NAMES, crosses, strict=True))}
    for left, right in ((CELL_NAMES[0], CELL_NAMES[1]), (CELL_NAMES[2], CELL_NAMES[3])):
        for key in ("recall_at_50_pp", "oracle_ap_pp", "hits_at_50"):
            if not np.array_equal(methods[left][key], methods[right][key]):
                raise AssertionError("Changing only the reranker must preserve membership metrics and oracle")

    comparisons = {"reranker_scaling_given_retrieve2": (CELL_NAMES[0], CELL_NAMES[1]),
                   "reranker_scaling_given_retrieve8": (CELL_NAMES[2], CELL_NAMES[3]),
                   "retriever_scaling_given_rerank2": (CELL_NAMES[0], CELL_NAMES[2]),
                   "retriever_scaling_given_rerank8": (CELL_NAMES[1], CELL_NAMES[3]),
                   "both_scaling_against_2b": (CELL_NAMES[0], CELL_NAMES[3])}
    paired = {name: {"reference": left, "comparison": right,
                     "delta": {key: estimate(methods[right][key] - methods[left][key], plan)
                               for key in PAIRED_METRICS}}
              for name, (left, right) in comparisons.items()}
    interaction = {key: methods[CELL_NAMES[3]][key] - methods[CELL_NAMES[2]][key]
                  - methods[CELL_NAMES[1]][key] + methods[CELL_NAMES[0]][key] for key in PAIRED_METRICS}
    gap88 = estimate(methods[CELL_NAMES[3]]["ranking_gap_pp"], plan)
    outside8 = 100 - methods[CELL_NAMES[3]]["oracle_ap_pp"]
    outside8_estimate = estimate(outside8, plan)
    positives = relevance.sum(axis=1)
    actual_connected = len(np.unique(groups))
    report = {"schema": 1, "scope": "exposed development panel; not independent confirmation",
              "formal_evidence": False, "queries": len(queries), "gallery": len(panel["ids"]),
              "code_provenance": {"analysis_file_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                  "numpy_version": np.__version__, "python_version": platform.python_version()},
              "grouping": {"original_groups": len(np.unique(original_groups)), "connected_groups": actual_connected,
                           "historical_expected_connected_groups": EXPECTED_CONNECTED_GROUPS,
                           "matches_historical_110_groups": actual_connected == EXPECTED_CONNECTED_GROUPS,
                           "definition": "Connected components of original groups and shared relevant image IDs; "
                                         "component ID is the minimum original group ID."},
              "bootstrap": {"seed": BOOT_SEED, "draws": draws, "unit": "connected query group",
                            "estimator": "query-weighted mean after resampling complete groups",
                            "paired": True, "same_resamples_for_all_cells": True,
                            "intervals": "percentile; 2.5/97.5% two-sided and 5/95% one-sided bounds"},
              "methods": {name: _summary(metrics, plan) for name, metrics in methods.items()},
              "paired_comparisons": paired,
              "factorial_interaction": {"definition": "(8/8 - 8/2) - (2/8 - 2/2)",
                                        "delta": {key: estimate(values, plan) for key, values in interaction.items()}},
              "original_2b_sorting_gap_recovered_by_8b_reranker": recovered_sorting_fraction(
                  methods[CELL_NAMES[0]]["ap_pp"], methods[CELL_NAMES[1]]["ap_pp"],
                  methods[CELL_NAMES[0]]["oracle_ap_pp"], plan),
              "remaining_ranking_gap_8_8_pp": gap88,
              "ideal100_minus_retrieve8_oracle_pp": outside8_estimate,
              "space_judgement": classify_space(gap88, outside8_estimate),
              "recall_capacity": {"queries_with_more_than_50_positives": int(np.sum(positives > K)),
                                  "macro_capacity_at_50_pp": float(np.mean(100 * np.minimum(K / positives, 1))),
                                  "definition": "mean(min(50 / relevant_count, 1)); Recall@50 need not reach 100%.",
                                  "shortfall_by_retriever_pp": {
                                      name: estimate(methods[CELL_NAMES[index]]["recall_capacity_at_50_pp"]
                                                     - methods[CELL_NAMES[index]]["recall_at_50_pp"], plan)
                                      for name, index in (("retrieve2", 0), ("retrieve8", 2))}},
              "limits": ["Oracle uses the existing automatic category/area relevance rules; human gold is incomplete.",
                         "The full-gallery AP denominator includes every labelled positive. This is not AP@50.",
                         "Fixed-top50 oracle reorders those 50 labels perfectly and leaves the retriever tail unchanged.",
                         "100 minus oracle mixes excluded positives and frozen tail ordering; it is not pure recall loss.",
                         "Neither oracle difference guarantees achievable gains for an unknown method.",
                         "Shared-positive groups do not remove all semantic or category dependence.",
                         "Equal 50-pair counts do not imply equal latency, memory, or indexing cost for 2B and 8B."]}
    per_query = {"schema": 1, "scope": report["scope"], "formal_evidence": False,
                 "queries": [{"index": index, "key": query.get("key"), "text": query.get("text"),
                              "group": int(original_groups[index]), "connected_group": int(groups[index]),
                              "positive_count": int(positives[index])} for index, query in enumerate(queries)],
                 "methods": {name: {key: values.tolist() for key, values in metrics.items()}
                             for name, metrics in methods.items()},
                 "ideal100_minus_retrieve8_oracle_pp": outside8.tolist(),
                 "factorial_interaction": {key: values.tolist() for key, values in interaction.items()}}
    return report, per_query


def _safe_path(path):
    path = Path(path).expanduser().resolve()
    if any("sealed" in part.lower() or "retrieval-independent-panel" in part.lower() for part in path.parts):
        raise ValueError("Sealed independent data are forbidden in this exposed-panel analysis")
    return path


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_blob(repo, commit, path):
    relative = path.relative_to(repo).as_posix()
    return subprocess.run(["git", "show", f"{commit}:{relative}"], cwd=repo, check=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout


def validate_metadata(plan, plan_sha256, meta):
    """Reject incomplete/misaligned receipts before touching numerical arrays."""
    if meta.get("state") != "COMPLETE":
        raise ValueError("Runner receipt must have state COMPLETE")
    if meta.get("plan_sha256") != plan_sha256:
        raise ValueError("Runner receipt does not match the frozen plan SHA256")
    if meta.get("panel_sha256") != plan["data"]["panel_sha256"]:
        raise ValueError("Runner receipt does not match the frozen panel SHA256")
    for relative in INPUT_FILES.values():
        expected = meta.get("outputs_sha256", {}).get(relative)
        if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            raise ValueError(f"Runner receipt lacks a valid input SHA256: {relative}")


def load_verified_inputs(run_dir, plan_path, freeze_commit):
    """Load runner arrays only after immutable-plan, code, panel and SHA gates."""
    run_dir, plan_path = _safe_path(run_dir), _safe_path(plan_path)
    repo = Path(__file__).resolve().parents[2]
    if re.fullmatch(r"[0-9a-f]{40}", freeze_commit) is None:
        raise ValueError("An explicit 40-character frozen Git commit is required")
    plan_bytes = plan_path.read_bytes()
    if _git_blob(repo, freeze_commit, plan_path) != plan_bytes:
        raise ValueError("Plan bytes differ from the specified frozen Git commit")
    plan, plan_sha = json.loads(plan_bytes), hashlib.sha256(plan_bytes).hexdigest()
    sources = plan.get("source_files", {})
    analysis_relative = Path(__file__).resolve().relative_to(repo).as_posix()
    if analysis_relative not in sources:
        raise ValueError("Frozen plan must bind the analysis implementation in source_files")
    for relative, expected in sources.items():
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("source_files keys must be repository-relative paths")
        source_path = _safe_path(repo / relative_path)
        source_path.relative_to(repo)
        if _sha256(source_path) != expected or hashlib.sha256(_git_blob(repo, freeze_commit, source_path)).hexdigest() != expected:
            raise ValueError(f"Source does not match frozen plan/commit: {relative}")
    meta_path = _safe_path(run_dir / "meta.json")
    meta = json.loads(meta_path.read_text())
    validate_metadata(plan, plan_sha, meta)
    panel_path = _safe_path(Path(plan["data"]["baseline_dir"]) / "panel.json")
    if _sha256(panel_path) != plan["data"]["panel_sha256"]:
        raise ValueError("Actual panel does not match frozen panel SHA256")
    input_paths = {key: _safe_path(run_dir / relative) for key, relative in INPUT_FILES.items()}
    for key, path in input_paths.items():
        path.relative_to(run_dir)
        if _sha256(path) != meta["outputs_sha256"][INPUT_FILES[key]]:
            raise ValueError(f"Numerical input SHA256 mismatch: {INPUT_FILES[key]}")
    panel = json.loads(panel_path.read_text())
    arrays = {key: np.load(path, allow_pickle=False, mmap_mode="r") for key, path in input_paths.items()}
    provenance = {"freeze_commit": freeze_commit, "plan_sha256": plan_sha,
                  "panel_sha256": plan["data"]["panel_sha256"], "meta_sha256": _sha256(meta_path),
                  "source_files": sources,
                  "numerical_inputs_sha256": {relative: meta["outputs_sha256"][relative]
                                             for relative in INPUT_FILES.values()}}
    return panel, arrays, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path, help="Runner directory with COMPLETE meta.json")
    parser.add_argument("--plan", required=True, type=Path, help="Frozen plan JSON tracked in Git")
    parser.add_argument("--freeze-commit", required=True, help="Exact Git commit containing plan and source files")
    parser.add_argument("--out", required=True, type=Path, help="New results/ or reports/ output directory")
    args = parser.parse_args()
    out = _safe_path(args.out)
    if not any(part in ("results", "reports") for part in out.parts):
        raise ValueError("Study metrics must be written under a results/ or reports/ directory")
    if out.exists():
        raise ValueError("Output directory already exists; use a new directory to preserve prior results")
    panel, arrays, provenance = load_verified_inputs(args.run, args.plan, args.freeze_commit)
    report, per_query = analyze_arrays(panel, **arrays)
    report["input_provenance"] = provenance
    per_query["input_provenance"] = provenance
    out.mkdir(parents=True, exist_ok=False)
    for filename, data in (("report.json", report), ("per_query.json", per_query)):
        (out / filename).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"state": "COMPLETE", "output_directory": str(out),
                      "analysis_sha256": report["code_provenance"]["analysis_file_sha256"]}))


if __name__ == "__main__":
    main()
