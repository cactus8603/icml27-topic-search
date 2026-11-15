import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "killtests/multifidelity-attribution/scripts"))
from hybrid import fit_alpha
from pilot import attribution_arms, column_spearman, mask_plan, paired_summary, replay_error


class DiagnosticTest(unittest.TestCase):
    def test_fixed_alpha_and_joint_row_permutation(self):
        ycal = np.array([[2.], [5.], [9.]])
        ccal = np.array([[1.], [2.], [3.]])
        exact = np.array([[8.], [2.], [9.]])
        cheap = np.array([[3.], [-2.], [8.]])
        u = np.ones((1, 1))
        d = np.array([[0.5]])
        order = np.array([2, 0, 1])
        arms, coefficients = attribution_arms(ycal, ccal, exact, cheap, u, d, order, fit_alpha)
        self.assertEqual(coefficients["alpha"], 39 / 14)
        self.assertEqual(coefficients["permuted_alpha"], 29 / 14)
        np.testing.assert_allclose(arms["spell"], ycal / 2)
        np.testing.assert_allclose(arms["hybrid"], ycal / 2 + (39 / 14) * (cheap - ccal / 2))
        np.testing.assert_allclose(arms["perm_hybrid"], ycal / 2 + (29 / 14) * (cheap - ccal / 2)[order])
        np.testing.assert_array_equal(arms["perm_ekfac"], cheap[order])
        np.testing.assert_array_equal(arms["exact"], exact)

    def test_lds_sign_ties_and_constant(self):
        truth = np.array([[1., 1.], [1., 2.], [3., 3.], [4., 4.]])
        predicted = np.column_stack([-truth[:, 0], np.ones(4)])
        correlations = column_spearman(predicted, truth)
        self.assertAlmostEqual(correlations[0], -1)
        self.assertTrue(np.isnan(correlations[1]))
        self.assertAlmostEqual(column_spearman(-predicted, truth)[0], 1)

    def test_aggregate_stronger_baseline_with_paired_draws(self):
        scores = {
            "spell": np.repeat([0.8, 0.0], 6),
            "ekfac": np.repeat([0.0, 0.8], 6),
            "hybrid": np.full(12, 0.6),
            "perm_ekfac": np.full(12, 0.1),
            "perm_hybrid": np.full(12, 0.39),
            "exact": np.full(12, 0.92),
        }
        result = paired_summary(scores)
        self.assertAlmostEqual(result["gains"]["hybrid_minus_stronger"], 0.2)
        self.assertAlmostEqual(result["gains"]["permutation_gain"], -0.01)
        self.assertTrue(result["hybrid_gain_at_least_0_04_reference"])
        self.assertTrue(result["permutation_gain_at_most_0_01_reference"])
        self.assertLess(result["descriptive_95pct_intervals"]["hybrid_minus_stronger"][0], 0.2)
        scores["exact"][6] = np.nan
        result = paired_summary(scores)
        self.assertFalse(result["complete_fixed_panel"])
        self.assertEqual(result["undefined_query_positions"], [442])
        self.assertNotIn("gains", result)

    def test_technical_replay_scale_floor(self):
        reference = np.ones(400)
        self.assertAlmostEqual(replay_error(reference + 0.001, reference), 0.01)
        self.assertEqual(replay_error(reference, reference), 0.0)

    def test_repeated_mask_content_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Mask content overlaps"):
            mask_plan({}, lambda _cfg, _j, n: np.zeros(n, dtype=bool))


if __name__ == "__main__":
    unittest.main()
