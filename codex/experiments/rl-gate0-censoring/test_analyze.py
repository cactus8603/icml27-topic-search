import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import difference_bound, match_pass_one, minimum_restores


class CensoringTests(unittest.TestCase):
    def test_minimum_restores_opens_new_questions_first(self):
        pass_k = np.array([0.0, 2 / 3, 1.0, 1.0])
        n, achieved = minimum_restores(np.array([0, 1]), np.array([2, 1]), pass_k, 0.8)
        self.assertEqual(n, 2)
        self.assertAlmostEqual(achieved, 5 / 6)
        self.assertEqual(minimum_restores(np.array([0, 1]), np.array([0, 0]), pass_k, 0.8), (None, None))

    def test_oracle_at_the_same_pass_one_uses_a_fraction(self):
        one_base, one_rl = np.array([.1, .2, .65]), np.array([.5, .6, .4])
        k_base, k_rl = np.array([.9, .8, .75]), np.array([.7, .7, .8])
        self.assertAlmostEqual(match_pass_one(one_base, one_rl, k_base, k_rl), 2.275 / 3)

    def test_censoring_bounds_have_the_right_direction(self):
        observed = np.array([[.25, .25], [.5, .5]])
        best = np.array([[.75, .75], [.75, .75]])
        report = difference_bound(observed, best, draws=1000, seed=1)
        self.assertEqual(report["observed_rl_minus_base"], .25)
        self.assertEqual(report["identification_bounds"], [-.25, .5])
        self.assertEqual(report["paired_95pct_intervals_observed_lower_upper"], [[.25, .25], [-.25, -.25], [.5, .5]])


if __name__ == "__main__":
    unittest.main()
