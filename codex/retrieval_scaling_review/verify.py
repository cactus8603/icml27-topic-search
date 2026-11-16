"""[codex] Independent CPU verification; never imports the study implementation.

Without --allow-study-results this program only accepts --self-test.  Actual
study arrays may be opened only after the root agent explicitly authorizes it.
The reviewer uses scalar AP, graph traversal, and bootstrap multiplicities.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
K = 50
SEED = 2026111668
DRAWS = 10000
EPSILON = 2.0
CELLS = ("retrieve2/rerank2", "retrieve2/rerank8", "retrieve8/rerank2", "retrieve8/rerank8")
METRICS = ("ap_pp", "recall_at_50_pp", "oracle_ap_pp", "ranking_gap_pp")
FILES = {"order2": "index/2b_order.npy", "order8": "index/8b_order.npy",
         "cross22": "cross/e2_r2.npy", "cross28": "cross/e2_r8.npy",
         "cross82": "cross/e8_r2.npy", "cross88": "cross/e8_r8.npy"}
PINS = {"embed2": ("Alibaba-NLP/core-emb-2b", "cc9870ea92d64ebd5f8f9fc368d1120453564158"),
        "cross2": ("Alibaba-NLP/core-reranker-2b", "64e1a36cd6ecace459a4c7cd8f7d0ef5d6fee30b"),
        "embed8": ("Alibaba-NLP/core-emb-8b", "123aef9e5e9e76fb686420f077fce3ef3b580c2b"),
        "cross8": ("Alibaba-NLP/core-reranker-8b", "d95d36a34300e34b87cc1793827ac798642c4488")}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def safe(value):
    path = Path(value).expanduser().resolve()
    require(not any("sealed" in x.lower() or "retrieval-independent-panel" in x.lower()
                    for x in path.parts), "Sealed paths are forbidden")
    return path


def read_json(path):
    def pairs(rows):
        value = {}
        for key, item in rows:
            require(key not in value, "Duplicate JSON key")
            value[key] = item
        return value
    return json.loads(safe(path).read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def sha(path):
    with safe(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def scalar_ap(ordered_ids, positives):
    """All labelled positives remain in the denominator, including the tail."""
    require(bool(positives), "Empty positive set")
    hits, total = 0, 0.0
    for position, image_id in enumerate(ordered_ids, 1):
        if image_id in positives:
            hits += 1
            total += hits / position
    require(hits == len(positives), "A full ranking lost a positive")
    return 100 * total / len(positives)


def measure_one(ids, order, cross, positives):
    require(len(order) == len(ids) and sorted(order) == list(range(len(ids))), "Bad permutation")
    require(len(cross) == K and np.isfinite(cross).all(), "Bad cross scores")
    prefix = [int(order[j]) for j in sorted(range(K), key=lambda j: (-float(cross[j]), j))]
    tail = [int(j) for j in order[K:]]
    ranked = [ids[j] for j in prefix + tail]
    original_prefix = [ids[int(j)] for j in order[:K]]
    # Construct the oracle independently from the original retriever prefix.
    oracle_prefix = ([j for j in original_prefix if j in positives]
                     + [j for j in original_prefix if j not in positives])
    oracle = oracle_prefix + [ids[j] for j in tail]
    hits = sum(j in positives for j in original_prefix)
    ap, oracle_ap = scalar_ap(ranked, positives), scalar_ap(oracle, positives)
    require(oracle_ap >= ap - 1e-10, "Oracle failed to dominate")
    return {"ap_pp": ap, "recall_at_50_pp": 100 * hits / len(positives),
            "oracle_ap_pp": oracle_ap, "ranking_gap_pp": max(oracle_ap - ap, 0),
            "hits_at_50": hits, "no_top50_hit": hits == 0,
            "recall_capacity_at_50_pp": 100 * min(K / len(positives), 1),
            "reranked_prefix_indices": prefix}


def graph_groups(queries):
    """Query graph and DFS, independently of the producer's union-find."""
    graph = [set() for _ in queries]
    groups, positives = {}, {}
    for index, query in enumerate(queries):
        group = query["group"]
        require(type(group) is int, "Original groups must be integers")
        for key, owners in ((group, groups), *((p, positives) for p in query["relevant"])):
            if key in owners:
                other = owners[key]
                graph[index].add(other)
                graph[other].add(index)
            else:
                owners[key] = index
    labels, visited = np.empty(len(queries), dtype=np.int64), set()
    for start in range(len(queries)):
        if start in visited:
            continue
        stack, component = [start], []
        visited.add(start)
        while stack:
            index = stack.pop()
            component.append(index)
            for other in graph[index]:
                if other not in visited:
                    visited.add(other)
                    stack.append(other)
        representative = min(queries[i]["group"] for i in component)
        labels[component] = representative
    return labels


class Bootstrap:
    """Use multiplicities and matrix products rather than repeated group sums."""
    def __init__(self, labels, draws=DRAWS):
        self.unique = sorted(set(int(x) for x in labels))
        self.members = [np.flatnonzero(labels == group) for group in self.unique]
        count = len(self.members)
        sampled = np.random.default_rng(SEED).integers(0, count, (draws, count))
        self.multiplicity = np.zeros((draws, count), dtype=np.int64)
        np.add.at(self.multiplicity, (np.repeat(np.arange(draws), count), sampled.ravel()), 1)
        self.denominator = self.multiplicity @ np.array([len(m) for m in self.members])
        self.draws = draws

    def samples(self, values):
        values = np.asarray(values, dtype=np.float64)
        totals = np.array([sum(float(values[i]) for i in members) for members in self.members])
        return (self.multiplicity @ totals) / self.denominator

    def estimate(self, values):
        samples = self.samples(values)
        return {"point_pp": float(np.mean(values)),
                "ci95_two_sided_pp": np.percentile(samples, [2.5, 97.5]).tolist(),
                "lower95_one_sided_pp": float(np.percentile(samples, 5)),
                "upper95_one_sided_pp": float(np.percentile(samples, 95)),
                "bootstrap_se_pp": float(np.std(samples, ddof=1))}


def decision(ranking, outside):
    def state(value):
        if value["upper95_one_sided_pp"] < EPSILON:
            return "below_epsilon"
        if value["lower95_one_sided_pp"] > EPSILON:
            return "above_epsilon"
        return "uncertain"
    a, b = state(ranking), state(outside)
    verdict = ("space_basically_absorbed_on_this_exposed_panel" if a == b == "below_epsilon"
               else "detectable_remaining_map_gap" if "above_epsilon" in (a, b) else "uncertain")
    return {"epsilon_pp": EPSILON, "ranking_gap_state": a,
            "outside_fixed_candidate_oracle_gap_state": b, "verdict": verdict}


def recompute(panel, arrays, draws=DRAWS):
    ids, queries = panel["ids"], panel["queries"]
    require(len(set(ids)) == len(ids), "Duplicate gallery IDs")
    methods = {}
    labels = graph_groups(queries)
    boot = Bootstrap(labels, draws)
    for cell, order_key, score_key in zip(CELLS, ("order2", "order2", "order8", "order8"),
                                          ("cross22", "cross28", "cross82", "cross88")):
        order, cross = arrays[order_key], arrays[score_key]
        require(order.shape == (len(queries), len(ids)), "Wrong order shape")
        require(cross.shape == (len(queries), K), "Wrong score shape")
        rows = []
        for index, query in enumerate(queries):
            positives = set(query["relevant"])
            require(len(positives) == len(query["relevant"]) and positives <= set(ids), "Bad positives")
            rows.append(measure_one(ids, order[index].tolist(), cross[index], positives))
        methods[cell] = {key: np.asarray([r[key] for r in rows]) for key in rows[0]}
    for a, b in ((0, 1), (2, 3)):
        for key in ("hits_at_50", "recall_at_50_pp", "oracle_ap_pp"):
            require(np.array_equal(methods[CELLS[a]][key], methods[CELLS[b]][key]), "Same-pool metrics changed")
    summaries = {}
    for cell, values in methods.items():
        summaries[cell] = {"map_pp": float(np.mean(values["ap_pp"])),
                           "recall_at_50_pp": float(np.mean(values["recall_at_50_pp"])),
                           "queries_without_top50_hit": int(sum(values["no_top50_hit"])),
                           "oracle_map_pp": float(np.mean(values["oracle_ap_pp"])),
                           "oracle_minus_actual_map_pp": float(np.mean(values["ranking_gap_pp"])),
                           "recall_capacity_at_50_pp": float(np.mean(values["recall_capacity_at_50_pp"])),
                           "uncertainty": {key: boot.estimate(values[key]) for key in METRICS},
                           "rerank_pairs_per_query": K}
    compare = {"reranker_scaling_given_retrieve2": (0, 1), "reranker_scaling_given_retrieve8": (2, 3),
               "retriever_scaling_given_rerank2": (0, 2), "retriever_scaling_given_rerank8": (1, 3),
               "both_scaling_against_2b": (0, 3)}
    paired = {key: {"reference": CELLS[a], "comparison": CELLS[b],
                    "delta": {metric: boot.estimate(methods[CELLS[b]][metric] - methods[CELLS[a]][metric])
                              for metric in METRICS}} for key, (a, b) in compare.items()}
    interaction = {key: methods[CELLS[3]][key] - methods[CELLS[2]][key]
                  - methods[CELLS[1]][key] + methods[CELLS[0]][key] for key in METRICS}
    gain = methods[CELLS[1]]["ap_pp"] - methods[CELLS[0]]["ap_pp"]
    original_gap = methods[CELLS[0]]["oracle_ap_pp"] - methods[CELLS[0]]["ap_pp"]
    denom_samples = boot.samples(original_gap)
    valid = denom_samples > 1e-12
    fraction = {"point_fraction": float(np.mean(gain) / np.mean(original_gap)) if np.mean(original_gap) > 1e-12 else None,
                "gain_pp": boot.estimate(gain), "original_sorting_gap_pp": boot.estimate(original_gap),
                "undefined_bootstrap_draws": int(sum(~valid)),
                "ci95_two_sided_fraction": (np.percentile(boot.samples(gain) / denom_samples, [2.5, 97.5]).tolist()
                                            if valid.all() else None)}
    gap = boot.estimate(methods[CELLS[3]]["ranking_gap_pp"])
    outside_values = 100 - methods[CELLS[3]]["oracle_ap_pp"]
    outside = boot.estimate(outside_values)
    positives = np.array([len(q["relevant"]) for q in queries])
    expected = {"formal_evidence": False, "queries": len(queries), "gallery": len(ids),
                "grouping": {"original_groups": len({q["group"] for q in queries}),
                             "connected_groups": len(set(labels)), "historical_expected_connected_groups": 110,
                             "matches_historical_110_groups": len(set(labels)) == 110},
                "bootstrap": {"seed": SEED, "draws": draws, "paired": True, "same_resamples_for_all_cells": True},
                "methods": summaries, "paired_comparisons": paired,
                "factorial_interaction": {"delta": {key: boot.estimate(value) for key, value in interaction.items()}},
                "original_2b_sorting_gap_recovered_by_8b_reranker": fraction,
                "remaining_ranking_gap_8_8_pp": gap, "ideal100_minus_retrieve8_oracle_pp": outside,
                "space_judgement": decision(gap, outside),
                "recall_capacity": {"queries_with_more_than_50_positives": int(sum(positives > K)),
                                    "macro_capacity_at_50_pp": float(np.mean(100 * np.minimum(K / positives, 1))),
                                    "shortfall_by_retriever_pp": {
                                        name: boot.estimate(methods[CELLS[i]]["recall_capacity_at_50_pp"]
                                                            - methods[CELLS[i]]["recall_at_50_pp"])
                                        for name, i in (("retrieve2", 0), ("retrieve8", 2))}}}
    raw = {"queries": [{"index": i, "group": q["group"], "connected_group": int(labels[i]),
                        "positive_count": len(q["relevant"])} for i, q in enumerate(queries)],
           "methods": {cell: {key: value.tolist() for key, value in values.items()}
                       for cell, values in methods.items()},
           "ideal100_minus_retrieve8_oracle_pp": outside_values.tolist(),
           "factorial_interaction": {key: value.tolist() for key, value in interaction.items()}}
    return expected, raw


def compare(expected, actual, prefix="root"):
    """Check every independently reconstructed numeric field; allow extra prose."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict), prefix + ": expected object")
        for key, value in expected.items():
            require(key in actual, prefix + ": missing " + key)
            compare(value, actual[key], prefix + "." + key)
    elif isinstance(expected, (list, tuple, np.ndarray)):
        require(len(expected) == len(actual), prefix + ": wrong array length")
        for index, value in enumerate(expected):
            compare(value, actual[index], prefix + f"[{index}]")
    elif isinstance(expected, (float, np.floating)):
        require(np.isfinite(actual) and abs(float(actual) - expected) <= 2e-8,
                prefix + f": mismatch {actual!r} versus {expected!r}")
    else:
        require(expected == actual, prefix + f": mismatch {actual!r} versus {expected!r}")


def verify_provenance(run, plan_path, freeze_commit):
    """All metadata and byte hashes pass before this verifier opens any NPY."""
    require(re.fullmatch(r"[0-9a-f]{40}", freeze_commit) is not None, "Explicit frozen commit required")
    run, plan_path = safe(run), safe(plan_path)
    plan_bytes = plan_path.read_bytes()
    def git_blob(relative):
        return subprocess.check_output(["git", "-C", str(ROOT), "show", f"{freeze_commit}:{relative}"])
    require(git_blob(plan_path.relative_to(ROOT).as_posix()) == plan_bytes, "Plan differs from frozen commit")
    plan = read_json(plan_path)
    for relative, expected in plan["source_files"].items():
        require(not Path(relative).is_absolute() and ".." not in Path(relative).parts, "Unsafe source path")
        require(sha(ROOT / relative) == expected and hashlib.sha256(git_blob(relative)).hexdigest() == expected,
                "Frozen source mismatch: " + relative)
    require(plan["analysis"]["seed"] == SEED and plan["analysis"]["bootstrap_draws"] == DRAWS
            and plan["analysis"]["epsilon_pp"] == EPSILON, "Analysis settings changed")
    require(plan["settings"]["dtype"] == "bfloat16" and plan["settings"]["attention"] == "sdpa"
            and plan["settings"]["quantization"] is False and plan["settings"]["cpu_offload"] is False,
            "Precision/placement settings changed")
    require([plan["settings"][k] for k in ("embed_image_batch", "embed_text_batch", "cross_batch")] == [8, 32, 4],
            "Batches changed")
    for name, pin in PINS.items():
        require((plan["models"][name]["repo"], plan["models"][name]["revision"]) == pin, "Wrong model pin")
    require(plan["models"]["embed2"]["embedding_dim"] == 2048
            and plan["models"]["embed8"]["embedding_dim"] == 4096, "Embedding dimensions mixed")
    meta = read_json(run / "meta.json")
    require(meta["state"] == "COMPLETE" and meta["formal_evidence"] is False, "Incomplete runner")
    require(meta["plan_sha256"] == hashlib.sha256(plan_bytes).hexdigest(), "Runner plan SHA mismatch")
    require(meta["panel_sha256"] == plan["data"]["panel_sha256"], "Runner panel SHA mismatch")
    identity = meta["identity"]
    require(identity["freeze_commit"] == freeze_commit and identity["settings"] == plan["settings"], "Identity mismatch")
    require(identity["model_revisions"] == {name: pin[1] for name, pin in PINS.items()}, "Actual model revisions differ")
    for key, value in plan["environment"].items():
        require(identity["environment"].get(key) == value, "Actual runtime differs: " + key)
    for relative, expected in meta["outputs_sha256"].items():
        path = safe(run / relative)
        require(path.is_relative_to(run) and sha(path) == expected, "Output hash mismatch: " + relative)
    all_canonical = set(FILES.values()) | {f"index/{size}b_{kind}.npy" for size in (2, 8) for kind in ("image", "full")}
    require(set(meta["outputs_sha256"]) == all_canonical, "Missing or extra canonical arrays")
    stages = {}
    for stage in ("index", "cross2", "cross8"):
        record = read_json(run / (stage + "-meta.json"))
        require(record["state"] == "COMPLETE" and record["identity"] == identity, "Wrong stage identity")
        for relative, expected in record["outputs_sha256"].items():
            require(meta["outputs_sha256"].get(relative) == expected, "Stage/final SHA mismatch")
        require(record["stage_wall_seconds"] >= 0, "Missing actual stage time")
        stages[stage] = record
    smoke = read_json(run / "smoke.json")
    require(smoke["state"] == "PASS" and smoke["identity"] == identity, "Smoke not valid for this run")
    require(smoke["synthetic_rgb_dimensions"] == [640, 640], "Smoke dimensions changed")
    event_log = read_json(run / "model-events.json")
    require(event_log["identity"] == identity, "Cost-event identity mismatch")
    events = smoke["model_events"] + [e for e in event_log["events"] if e["event"] == "model_forward_complete"]
    require(set(PINS) <= {e["model"] for e in events}, "Some models lack actual execution receipts")
    for event in events:
        require(event["all_parameters_cuda_bfloat16"] is True, "Actual BF16/GPU validation failed")
        require(event["loading_missing_and_mismatched_keys"] == "empty; checked by frozen loader", "Incomplete weights")
        require(0 < event["peak_gpu_allocated_bytes"] <= event["peak_gpu_reserved_bytes"] <= 24 * 2**30,
                "Invalid 3090 memory receipt")
        require(event["load_seconds"] >= 0 and event["loaded_wall_seconds"] >= event["load_seconds"], "Invalid model times")
    panel_path = safe(Path(plan["data"]["baseline_dir"]) / "panel.json")
    require(sha(panel_path) == plan["data"]["panel_sha256"], "Actual panel hash differs")
    panel = read_json(panel_path)
    require((len(panel["ids"]), len(panel["queries"])) == (8191, 136), "Wrong exposed panel")
    provenance = {"freeze_commit": freeze_commit, "plan_sha256": sha(plan_path), "meta_sha256": sha(run / "meta.json"),
                  "reviewer_sha256": sha(__file__), "source_files_verified": True,
                  "canonical_output_hashes_verified": True, "actual_bf16_gpu_receipts_verified": True,
                  "peak_reserved_gib_by_model": {name: max(e["peak_gpu_reserved_bytes"] for e in events if e["model"] == name) / 2**30
                                                 for name in PINS},
                  "model_load_failures": sum(e["event"] == "model_load_failed" for e in event_log["events"]),
                  "limits": ["BF16 and weight-completeness checks are attested by the frozen producer; no model is reloaded here.",
                             "Pure-colour smoke validates resources and shape, not model quality.",
                             "This review is conditional on exposed automatic relevance labels."]}
    return plan, panel, provenance


def self_test():
    # Hand-calculated full AP: positive ranks 10 and 100, not truncated AP@50.
    ids, order = list(range(120)), list(range(120))
    x = measure_one(ids, order, -np.arange(K), {9, 99})
    require(abs(x["ap_pp"] - 6) < 1e-12 and abs(x["oracle_ap_pp"] - 51) < 1e-12, "Scalar AP hand check")
    many = measure_one(ids, order, -np.arange(K), set(range(100)))
    require(many["ap_pp"] == many["oracle_ap_pp"] == 100
            and many["recall_at_50_pp"] == many["recall_capacity_at_50_pp"] == 50, "Recall capacity versus full AP")
    tied = measure_one(ids, order[70:] + order[:70], np.zeros(K), {9, 99})
    require(tied["reranked_prefix_indices"] == order[70:], "Tie handling")
    qs = [{"group": g, "relevant": p} for g, p in [(7, [1]), (7, [2]), (21, [2, 3]), (33, [3]), (90, [9])]]
    require(graph_groups(qs).tolist() == [7, 7, 7, 7, 90], "Transitive graph groups")
    labels, values = np.array([1, 1, 2]), np.array([0., 0., 10.])
    b = Bootstrap(labels, 40)
    sampled = np.random.default_rng(SEED).integers(0, 2, (40, 2))
    manual = [np.concatenate([values[labels == j + 1] for j in row]).mean() for row in sampled]
    require(np.array_equal(b.samples(values), manual), "Multiplicity bootstrap versus manual query resampling")
    panel = {"ids": ids, "queries": [{"group": 1, "relevant": [9, 99]}, {"group": 2, "relevant": [5]}]}
    orders = np.tile(np.arange(120), (2, 1))
    c22 = np.tile(-np.arange(50, dtype=float), (2, 1))
    c28 = c22.copy(); c28[0, 9] = 100; c28[1, 5] = 100
    arrays = {"order2": orders, "order8": orders.copy(), "cross22": c22,
              "cross28": c28, "cross82": c28.copy(), "cross88": c28.copy()}
    summary, raw = recompute(panel, arrays, 40)
    require(abs(summary["methods"][CELLS[0]]["map_pp"] - (6 + 100/6) / 2) < 1e-12, "Four-cell hand mAP")
    require(abs(summary["methods"][CELLS[1]]["map_pp"] - 75.5) < 1e-12, "Perfect-prefix mAP")
    require(summary["remaining_ranking_gap_8_8_pp"]["point_pp"] == 0, "Remaining sorting gap")
    expected_gain = 75.5 - (6 + 100 / 6) / 2
    require(abs(summary["paired_comparisons"]["both_scaling_against_2b"]["delta"]["ap_pp"]["point_pp"]
                - expected_gain) < 1e-12, "Four-cell paired gain")
    require(abs(summary["factorial_interaction"]["delta"]["ap_pp"]["point_pp"] + expected_gain) < 1e-12,
            "Factorial interaction")
    require(abs(summary["original_2b_sorting_gap_recovered_by_8b_reranker"]["point_fraction"] - 1) < 1e-12, "Recovered ratio")
    compare(raw, raw)
    bad = json.loads(json.dumps(summary)); bad["methods"][CELLS[0]]["map_pp"] += .01
    try:
        compare(summary, bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("Mismatch detector failed")
    boundary = {"lower95_one_sided_pp": 2., "upper95_one_sided_pp": 2.}
    require(decision(boundary, boundary)["verdict"] == "uncertain", "Strict epsilon boundary")
    print(json.dumps({"state": "SELF_TEST_PASS", "study_results_opened": False,
                      "checks": ["scalar_full_AP", "fixed_pool_oracle", "stable_ties", "graph_components",
                                 "manual_cluster_bootstrap", "four_cells", "ratio", "epsilon", "mismatch_detection"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--allow-study-results", action="store_true")
    parser.add_argument("--run", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--freeze-commit")
    parser.add_argument("--analysis-dir", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.self_test:
        require(not any((args.run, args.plan, args.analysis_dir, args.out, args.allow_study_results)), "Self-test is synthetic only")
        self_test()
        return
    require(args.allow_study_results, "Root must explicitly authorize study-result access")
    require(all((args.run, args.plan, args.freeze_commit, args.analysis_dir, args.out)), "Missing study verification paths")
    out = safe(args.out)
    require(not out.exists() and any(x in {"results", "reports"} for x in out.parts), "Use a new results/reports file")
    plan, panel, provenance = verify_provenance(args.run, args.plan, args.freeze_commit)
    # The first numerical loads occur only after the complete provenance gate.
    arrays = {key: np.load(safe(args.run / relative), allow_pickle=False) for key, relative in FILES.items()}
    # Re-derive both full orders from the independently cached image/query vectors.
    for size in (2, 8):
        image = np.load(safe(args.run / f"index/{size}b_image.npy"), allow_pickle=False)
        text = np.load(safe(args.run / f"index/{size}b_full.npy"), allow_pickle=False)
        dim = 2048 if size == 2 else 4096
        require(image.shape == (8191, dim) and text.shape == (136, dim), "New index shape/dimension mismatch")
        require(np.isfinite(image).all() and np.isfinite(text).all(), "New index nonfinite")
        require(np.allclose(np.linalg.norm(image, axis=1), 1, atol=2e-4)
                and np.allclose(np.linalg.norm(text, axis=1), 1, atol=2e-4), "New index not normalized")
        scores = text @ image.T
        order = np.array([sorted(range(len(row)), key=lambda j: (-float(row[j]), j)) for row in scores])
        require(np.array_equal(order, arrays[f"order{size}"]), "Saved full-gallery order does not follow actual vectors")
    expected, raw = recompute(panel, arrays)
    report = read_json(safe(args.analysis_dir) / "report.json")
    per_query = read_json(safe(args.analysis_dir) / "per_query.json")
    compare(expected, report)
    compare(raw, per_query)
    require(report["input_provenance"]["plan_sha256"] == provenance["plan_sha256"], "Analysis used another plan")
    require(report["input_provenance"]["freeze_commit"] == args.freeze_commit, "Analysis used another freeze")
    require(report["input_provenance"]["meta_sha256"] == provenance["meta_sha256"], "Analysis used another raw receipt")
    require(report["input_provenance"]["source_files"] == plan["source_files"], "Analysis source provenance differs")
    meta = read_json(safe(args.run) / "meta.json")
    require(report["input_provenance"]["numerical_inputs_sha256"] ==
            {relative: meta["outputs_sha256"][relative] for relative in FILES.values()}, "Analysis raw-input provenance differs")
    require(per_query["input_provenance"] == report["input_provenance"], "Per-query report provenance differs")
    require(report["code_provenance"]["analysis_file_sha256"] == plan["source_files"]["codex/retrieval_scaling/analyze.py"],
            "Analysis implementation SHA differs")
    provenance.update(report_sha256=sha(safe(args.analysis_dir) / "report.json"),
                      per_query_sha256=sha(safe(args.analysis_dir) / "per_query.json"))
    result = {"state": "PASS", "owner": "[codex]", "independent_of_analysis_implementation": True,
              "verification_provenance": provenance, "independent_recomputation": expected}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"state": "PASS", "independent_review": str(out), "review_sha256": sha(out)}))


if __name__ == "__main__":
    main()
