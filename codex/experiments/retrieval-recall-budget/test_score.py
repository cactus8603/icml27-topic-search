from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score import (PUBLISHED, admission, attach_prefix, check_replay, contrast,
                   interleave, original, query_metrics, rerank_shortlist)


class RetrievalRecallBudgetTests(unittest.TestCase):
    def test_interleaving_keeps_unique_candidates_with_top_ties(self):
        core = original.ranks(np.array([[0.8, 0.8, 0.7, 0.6, 0.5], [1, 2, 3, 4, 5]]))
        cs = np.array([[0, 4, 3, 2, 1], [0, 4, 3, 1, 2]])
        np.testing.assert_equal(interleave(core, cs, 3), [[0, 1, 4], [4, 0, 3]])

    def test_changed_membership_leaves_remaining_core_relative_order(self):
        core = np.array([[4, 2, 3, 0, 1], [2, 0, 4, 3, 1]])
        prefix = np.array([[1, 2], [3, 4]])
        np.testing.assert_equal(attach_prefix(core, prefix), [[1, 2, 4, 3, 0], [3, 4, 2, 0, 1]])
        with self.assertRaises(ValueError):
            attach_prefix(core, np.array([[1, 1], [3, 4]]))

    def test_cross_ties_follow_core_rank_and_preserve_original_baseline(self):
        core = np.array([[3, 0, 1, 2], [2, 0, 1, 3]])
        prefix = np.array([[2, 3], [3, 1]])
        scores = np.array([[7, 7], [8, 7]])
        np.testing.assert_equal(rerank_shortlist(core, prefix, scores), [[3, 2, 0, 1], [3, 1, 2, 0]])
        np.testing.assert_equal(rerank_shortlist(core, core[:, :2], scores), original.full_order(core, scores, 2))

    def test_admission_oracle_uses_all_labels_and_original_tail(self):
        order = np.array([[3, 1, 0, 2, 4]])
        relevance = np.array([[True, False, False, False, True]])
        measured = query_metrics(order, relevance, 2)
        np.testing.assert_equal(measured["hits"], [0])
        np.testing.assert_allclose(measured["ap_pp"], [100 * ((1 / 3) + (2 / 5)) / 2])
        np.testing.assert_allclose(measured["oracle_ap_pp"], measured["ap_pp"])
        labels = np.take_along_axis(relevance, order, axis=1)
        np.testing.assert_allclose(admission(labels, 4)["oracle_ap_pp"], [70])
        np.testing.assert_equal(admission(labels, 5)["oracle_ap_pp"], [100])

    def test_paired_contrast_separates_admission_and_ranking_gaps(self):
        base = {"ap_pp": np.array([50.0, 45.0, 10.0]), "oracle_ap_pp": np.array([60.0, 55.0, 30.0]),
                "hits": np.array([1, 1, 0])}
        alternate = {"ap_pp": base["ap_pp"] + 4, "oracle_ap_pp": base["oracle_ap_pp"] + 5,
                     "hits": np.array([1, 2, 1])}
        positives, groups = np.array([1, 2, 2]), [2, 2, 1]
        state, stats = contrast(base, alternate, positives, groups)
        self.assertEqual(state, "PASS")
        self.assertEqual(stats["map_gain"], {"point_pp": 4.0, "ci95_pp": [4.0, 4.0]})
        self.assertEqual(stats["admission_oracle_gain"]["point_pp"], 5.0)
        self.assertEqual(stats["ranking_gap_reduced"]["point_pp"], -1.0)
        self.assertEqual(contrast(base, base, positives, groups)[0], "FAIL")

    def test_frozen_replay_gate_allows_small_rounding_only(self):
        ready = {**PUBLISHED, "queries_without_hit": 31}
        check_replay(ready)
        check_replay(ready | {"map_pp": ready["map_pp"] + 0.09})
        with self.assertRaises(ValueError):
            check_replay(ready | {"map_pp": ready["map_pp"] + 0.11})
        with self.assertRaises(ValueError):
            check_replay(ready | {"queries_without_hit": 30})


if __name__ == "__main__":
    unittest.main()
