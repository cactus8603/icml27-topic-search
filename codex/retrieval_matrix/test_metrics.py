"""[codex] Synthetic scoring tests; no data/results/GPU are read."""
import unittest

import numpy as np

try:
    from . import metrics as M
except ImportError:
    import metrics as M


class MetricTests(unittest.TestCase):
    def test_stable_ties(self):
        self.assertEqual(M.stable_order([[.2, .9, .9]]).tolist(), [[1, 2, 0]])

    def test_ignore_is_removed_after_raw_k_without_refill(self):
        value = M.query_metrics(np.arange(5), [2], [0], ks=(2, 3))
        self.assertEqual(value["R@2"], 0)
        self.assertEqual(value["evaluated@2"], 1)
        self.assertEqual(value["R@3"], 1)
        self.assertEqual(value["AP@3"], .5)
        self.assertEqual(value["ignored_admitted@2"], 1)

    def test_multi_positive_ap_recall_and_oracle(self):
        value = M.query_metrics(np.arange(5), [1, 3], ks=(2, 4))
        self.assertEqual(value["R@2"], .5)
        self.assertEqual(value["AP@2"], .25)
        self.assertEqual(value["oracle_AP@2"], .5)
        self.assertEqual(value["AP"], .5)
        self.assertEqual(value["MRR@2"], .5)

    def test_missing_positive_is_not_dropped_from_denominator(self):
        value = M.query_metrics(np.arange(6), [5], ks=(2,))
        self.assertEqual(value["R@2"], 0)
        self.assertEqual(value["MRR@2"], 0)
        self.assertEqual(value["oracle_AP@2"], 0)

    def test_cross_preserves_original_prefix_set_and_tail(self):
        order = np.arange(6)[None]
        changed = M.rerank_prefix(order, np.array([[0., 2., 2.]]), 3)
        self.assertEqual(changed.tolist(), [[1, 2, 0, 3, 4, 5]])
        old = M.query_metrics(order[0], [4], ks=(3,))
        new = M.query_metrics(changed[0], [4], ks=(3,))
        self.assertEqual(old["R@3"], new["R@3"])

    def test_bad_permutation_and_labels_are_rejected(self):
        with self.assertRaises(ValueError):
            M.evaluate(np.array([[0, 0]]), ["a", "b"], [{"id": "q", "positives": ["b"]}])
        with self.assertRaises(ValueError):
            M.query_metrics(np.arange(3), [0], [0])
        with self.assertRaises(ValueError):
            M.query_metrics(np.arange(3), [9])
        with self.assertRaises(ValueError):
            M.stable_order([[np.nan]])

    def test_cluster_bootstrap_is_paired_and_query_weighted(self):
        a = M.paired_cluster_bootstrap([0, 0, 0], [1, 1, 0], ["a", "a", "b"], seed=7, draws=100)
        b = M.paired_cluster_bootstrap([0, 0, 0], [1, 1, 0], ["a", "a", "b"], seed=7, draws=100)
        self.assertEqual(a, b)
        self.assertAlmostEqual(a["delta"], 2/3)
        self.assertEqual(a["clusters"], 2)


if __name__ == "__main__":
    unittest.main()
