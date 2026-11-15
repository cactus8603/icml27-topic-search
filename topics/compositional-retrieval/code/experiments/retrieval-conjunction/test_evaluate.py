import itertools
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import ARMS, average_precision, bound, choose, paired_test, percentile, ranking
from prepare import QUERY_SPLIT_SEED, form_queries, hash_select


class RetrievalConjunctionTests(unittest.TestCase):
    def test_oracle_matches_all_shortlist_permutations(self):
        truth = np.array([[False, True, False, True, True]])
        ceiling, recall = bound(truth, 3)
        permuted = [average_precision(np.array([list(prefix) + truth[0, 3:].tolist()]))[0]
                    for prefix in itertools.permutations(truth[0, :3])]
        self.assertAlmostEqual(ceiling[0], max(permuted))
        self.assertAlmostEqual(recall[0], 1 / 3)

    def test_ties_and_development_weight(self):
        raw = np.array([[.2, .2, -.1]])
        np.testing.assert_array_equal(ranking(raw), ranking(percentile(raw)))
        full = np.array([[.5, .7, .3]])
        minimum = np.array([[1., .2, .1]])
        labels = np.array([[True, False, False]])
        grids = choose((full, full, minimum), labels)
        self.assertEqual(grids[ARMS[1]]["selected_weight"], 0)
        self.assertEqual(grids[ARMS[2]]["selected_weight"], .25)

    def test_paired_threshold(self):
        ap = {ARMS[0]: np.array([.5, .5]), ARMS[1]: np.array([.6, .6]), ARMS[2]: np.array([.7, .7])}
        self.assertEqual(paired_test(ap, [1, 1])["state"], "PASS")
        ap[ARMS[2]] = ap[ARMS[1]]
        self.assertEqual(paired_test(ap, [1, 1])["state"], "FAIL")

    def test_both_single_condition_negatives_and_prior_keys(self):
        pair = (1, 2)
        part = "dev" if hash_select(QUERY_SPLIT_SEED, pair) % 3 == 0 else "test"
        category = {10: {1, 2}, 11: {1}, 12: {2}}
        self.assertEqual(form_queries(part, [10, 11, 12], category, set()),
                         [{"key": [1, 2], "relevant": [10], "group": 10}])
        self.assertEqual(form_queries(part, [10, 11], category, set()), [])
        self.assertEqual(form_queries(part, [10, 11, 12], category, {pair}), [])


if __name__ == "__main__":
    unittest.main()
