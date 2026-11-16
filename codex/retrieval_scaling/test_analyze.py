"""[codex] Synthetic CPU tests; never read real study scores or results."""

import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np


SPEC = importlib.util.spec_from_file_location("retrieval_scaling_analysis", Path(__file__).with_name("analyze.py"))
A = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(A)


def synthetic(seed=19):
    rng = np.random.default_rng(seed)
    ids = list(range(1000, 1120))
    relevant = [[1001], [1002, 1068], [1002], [1003, 1004], [1060], [1008]]
    groups = [10, 20, 30, 40, 40, 60]
    queries = [{"group": group, "relevant": positives, "text": f"query {index}", "key": [index]}
               for index, (group, positives) in enumerate(zip(groups, relevant))]
    panel = {"ids": ids, "queries": queries}
    order2 = np.tile(np.arange(len(ids)), (len(queries), 1))
    order8 = np.asarray([rng.permutation(len(ids)) for _ in queries])
    scores = [rng.normal(size=(len(queries), 50)) for _ in range(4)]
    return panel, order2, order8, *scores


class ScalingAnalysisTests(unittest.TestCase):
    def test_full_ap_oracle_and_unchanged_tail(self):
        order = np.arange(120)[None, :]
        # A single query has positives at original ranks 10 and 100.
        relevance = np.zeros((1, 120), dtype=bool)
        relevance[0, [9, 99]] = True
        scores = -np.arange(50)[None, :].astype(float)
        actual = A.final_order(order, scores)
        measured = A._measure(relevance, actual)
        self.assertAlmostEqual(measured["ap_pp"][0], 6.0)
        self.assertAlmostEqual(measured["oracle_ap_pp"][0], 51.0)
        self.assertAlmostEqual(measured["ranking_gap_pp"][0], 45.0)
        self.assertEqual(measured["recall_at_50_pp"][0], 50)
        np.testing.assert_array_equal(actual[:, 50:], order[:, 50:])

    def test_ties_follow_retriever_rank_not_gallery_id(self):
        order = np.roll(np.arange(120), 70)[None, :]
        scores = np.zeros((1, 50))
        scores[0, [7, 2]] = 1
        actual = A.final_order(order, scores)
        expected_prefix = np.concatenate((order[0, [2, 7]], np.delete(order[0, :50], [2, 7])))
        np.testing.assert_array_equal(actual[0, :50], expected_prefix)
        np.testing.assert_array_equal(actual[0, 50:], order[0, 50:])

    def test_connections_are_transitive_and_order_invariant(self):
        queries = [{"group": group, "relevant": relevant} for group, relevant in
                   ((7, [100]), (7, [101]), (21, [101, 102]), (33, [102]), (90, [999]))]
        np.testing.assert_array_equal(A.connected_query_groups(queries), [7, 7, 7, 7, 90])
        np.testing.assert_array_equal(A.connected_query_groups(queries[::-1]), [90, 7, 7, 7, 7])

    def test_shared_bootstrap_estimates_query_weighted_means(self):
        # Unequal group sizes: the estimand is 10/3, not the group mean 5.
        groups = np.array([1, 1, 2])
        values = np.array([0., 0., 10.])
        plan = A.bootstrap_plan(groups, draws=100)
        codes, picks, _ = plan
        samples = A.bootstrap_means(values, plan)
        manual = []
        for draw in picks:
            selected = np.concatenate([values[codes == group] for group in draw])
            manual.append(selected.mean())
        np.testing.assert_allclose(samples, manual, rtol=0, atol=0)
        self.assertAlmostEqual(A.estimate(values, plan)["point_pp"], 10 / 3)

    def test_reranker_cannot_change_recall_or_fixed_membership_oracle(self):
        report, raw = A.analyze_arrays(*synthetic(), draws=200)
        for left, right in ((A.CELL_NAMES[0], A.CELL_NAMES[1]), (A.CELL_NAMES[2], A.CELL_NAMES[3])):
            for key in ("recall_at_50_pp", "hits_at_50", "oracle_ap_pp"):
                np.testing.assert_array_equal(raw["methods"][left][key], raw["methods"][right][key])
            effect_name = "reranker_scaling_given_retrieve" + ("2" if "retrieve2" in left else "8")
            self.assertEqual(report["paired_comparisons"][effect_name]["delta"]["recall_at_50_pp"]["ci95_two_sided_pp"], [0., 0.])
        self.assertEqual(report["grouping"]["connected_groups"], 4)
        self.assertFalse(report["grouping"]["matches_historical_110_groups"])
        self.assertEqual(report["bootstrap"]["seed"], 2026111668)

    def test_retriever_oracles_are_recomputed_and_can_differ(self):
        panel, order2, _, *scores = synthetic()
        # Reverse order moves every original early positive far down the tail.
        order8 = order2[:, ::-1].copy()
        _, raw = A.analyze_arrays(panel, order2, order8, *scores, draws=100)
        self.assertNotEqual(raw["methods"][A.CELL_NAMES[0]]["oracle_ap_pp"],
                            raw["methods"][A.CELL_NAMES[2]]["oracle_ap_pp"])

    def test_every_metric_matches_an_independent_scalar_evaluator(self):
        panel, order2, order8, *crosses = synthetic()
        _, raw = A.analyze_arrays(panel, order2, order8, *crosses, draws=100)
        for cell_index, name in enumerate(A.CELL_NAMES):
            order = (order2, order8)[cell_index // 2]
            for row, query in enumerate(panel["queries"]):
                prefix = sorted(order[row, :50].tolist(), key=lambda col: (-crosses[cell_index][row, np.where(order[row] == col)[0][0]],
                                                                            np.where(order[row] == col)[0][0]))
                ranked = prefix + order[row, 50:].tolist()
                relevant = set(query["relevant"])

                def ap(columns):
                    hits, total = 0, 0.
                    for rank, column in enumerate(columns, 1):
                        if panel["ids"][column] in relevant:
                            hits += 1
                            total += hits / rank
                    return 100 * total / len(relevant)

                oracle_prefix = sorted(prefix, key=lambda col: panel["ids"][col] not in relevant)
                self.assertAlmostEqual(raw["methods"][name]["ap_pp"][row], ap(ranked))
                self.assertAlmostEqual(raw["methods"][name]["oracle_ap_pp"][row], ap(oracle_prefix + ranked[50:]))

    def test_capacity_with_more_than_50_positives(self):
        panel, order2, order8, *scores = synthetic()
        panel["queries"][0]["relevant"] = panel["ids"][:100]
        report, raw = A.analyze_arrays(panel, order2, order8, *scores, draws=100)
        self.assertEqual(report["recall_capacity"]["queries_with_more_than_50_positives"], 1)
        self.assertEqual(raw["methods"][A.CELL_NAMES[0]]["recall_capacity_at_50_pp"][0], 50.)
        self.assertEqual(raw["methods"][A.CELL_NAMES[0]]["recall_at_50_pp"][0], 50.)

    def test_recovery_is_ratio_of_means_and_allows_negative_gain(self):
        plan = A.bootstrap_plan(np.array([1, 2]), draws=100)
        recovered = A.recovered_sorting_fraction(np.array([0., 50.]), np.array([10., 70.]), np.array([100., 90.]), plan)
        self.assertAlmostEqual(recovered["point_fraction"], 30 / 140)
        negative = A.recovered_sorting_fraction(np.array([50., 50.]), np.array([40., 40.]), np.array([100., 100.]), plan)
        self.assertAlmostEqual(negative["point_fraction"], -.2)
        undefined = A.recovered_sorting_fraction(np.ones(2), np.ones(2), np.ones(2), plan)
        self.assertIsNone(undefined["point_fraction"])
        self.assertIsNone(undefined["ci95_two_sided_fraction"])
        self.assertEqual(undefined["undefined_bootstrap_draws"], 100)

    def test_decision_needs_both_gaps_small_and_uses_strict_boundaries(self):
        small = {"lower95_one_sided_pp": 0., "upper95_one_sided_pp": 1.9}
        large = {"lower95_one_sided_pp": 2.1, "upper95_one_sided_pp": 5.}
        uncertain = {"lower95_one_sided_pp": 1., "upper95_one_sided_pp": 3.}
        boundary = {"lower95_one_sided_pp": 2., "upper95_one_sided_pp": 2.}
        self.assertEqual(A.classify_space(small, small)["verdict"], "space_basically_absorbed_on_this_exposed_panel")
        self.assertEqual(A.classify_space(small, large)["verdict"], "detectable_remaining_map_gap")
        self.assertEqual(A.classify_space(small, uncertain)["verdict"], "uncertain")
        self.assertEqual(A.classify_space(boundary, boundary)["verdict"], "uncertain")

    def test_rejects_misaligned_nonfinite_or_malformed_inputs(self):
        original = synthetic()
        mutations = []
        bad = list(copy.deepcopy(original)); bad[1][0, 0] = bad[1][0, 1]; mutations.append(bad)
        bad = list(copy.deepcopy(original)); bad[3][0, 0] = np.nan; mutations.append(bad)
        bad = list(copy.deepcopy(original)); bad[3] = bad[3][:, :49]; mutations.append(bad)
        bad = list(copy.deepcopy(original)); bad[0]["queries"][0]["relevant"] = []; mutations.append(bad)
        bad = list(copy.deepcopy(original)); bad[0]["queries"][0]["relevant"] = [999999]; mutations.append(bad)
        bad = list(copy.deepcopy(original)); bad[0]["queries"][0]["relevant"] *= 2; mutations.append(bad)
        bad = list(copy.deepcopy(original)); bad[0]["queries"][0]["group"] = True; mutations.append(bad)
        for index, bad in enumerate(mutations):
            with self.subTest(mutation=index):
                with self.assertRaises(ValueError):
                    A.analyze_arrays(*bad, draws=100)

    def test_source_gate_rejects_incomplete_or_misaligned_receipts(self):
        plan = {"data": {"panel_sha256": "a" * 64}}
        receipt = {"state": "COMPLETE", "panel_sha256": "a" * 64,
                   "plan_sha256": "b" * 64,
                   "outputs_sha256": {path: "c" * 64 for path in A.INPUT_FILES.values()}}
        A.validate_metadata(plan, "b" * 64, receipt)
        for key, value in (("state", "RUNNING"), ("panel_sha256", "d" * 64),
                           ("plan_sha256", "e" * 64), ("outputs_sha256", {})):
            bad = {**receipt, key: value}
            with self.subTest(field=key), self.assertRaises(ValueError):
                A.validate_metadata(plan, "b" * 64, bad)

    def test_uncommitted_plan_rejected_before_scores_open(self):
        with patch.object(Path, "read_bytes", return_value=b"current"), \
                patch.object(A, "_git_blob", return_value=b"different"), \
                patch.object(A.np, "load", side_effect=AssertionError("must not read numerical results")):
            with self.assertRaisesRegex(ValueError, "Plan bytes differ"):
                A.load_verified_inputs("/unused/run", "/unused/plan.json", "f" * 40)

    def test_sealed_paths_are_forbidden(self):
        with self.assertRaisesRegex(ValueError, "Sealed"):
            A._safe_path("/unused/retrieval-independent-panel/data")

    def test_serializable_deterministic_output_and_default_10000_draws(self):
        import json
        first = A.analyze_arrays(*synthetic())
        second = A.analyze_arrays(*synthetic())
        self.assertEqual(json.dumps(first, sort_keys=True, allow_nan=False), json.dumps(second, sort_keys=True, allow_nan=False))
        self.assertEqual(first[0]["bootstrap"]["draws"], 10000)
        self.assertEqual(len(first[0]["code_provenance"]["analysis_file_sha256"]), 64)
        self.assertFalse(first[0]["formal_evidence"])


if __name__ == "__main__":
    unittest.main()
