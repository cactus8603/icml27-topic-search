"""Check the encoding and receipt calculations with constructed fixtures only."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run


class GeometryTests(unittest.TestCase):
    def test_adjacent_floating_values_keep_membership_for_all_auxiliary_values(self):
        cutoff = run.reference.CUTOFF
        x = np.repeat([np.nextafter(cutoff, -np.inf), cutoff, np.nextafter(cutoff, np.inf)], 7)
        auxiliary = np.tile([-64.0, -8.0, -1.0, 0.0, 1.0, 8.0, 64.0], 3)
        expected = np.repeat([False, True, True], 7)
        latent = np.column_stack((x, auxiliary))
        for profile, secondary in run.ARMS:
            result = run.score(latent, profile, secondary)
            np.testing.assert_array_equal(result >= run.reference.THRESHOLDS[profile], expected)

    def test_quantized_intervals_keep_cross_group_order_and_use_secondary_within_group(self):
        cutoff = run.reference.CUTOFF
        latent = np.array([[cutoff - 0.125, 64.0], [cutoff + 0.125, -64.0], [cutoff - 0.125, 0.0]])
        quantized = run.score(latent, "monotone_quantized", True)
        np.testing.assert_array_equal(np.floor(quantized), [-1.0, 0.0, -1.0])
        self.assertLess(quantized[2], quantized[0])
        self.assertLess(quantized[0], quantized[1])
        binary = run.score(latent, "binary", True)
        np.testing.assert_array_equal(np.floor(binary), [0.0, 1.0, 0.0])
        self.assertLess(binary[2], binary[0])
        self.assertLess(binary[0], binary[1])

    def test_bulk_zero_reorders_only_the_atom(self):
        latent = np.array([[1.75, 0.0], [0.0, 64.0], [2.0, -64.0], [np.nextafter(2.0, np.inf), -64.0],
                           [run.reference.CUTOFF, 64.0]])
        primary = run.score(latent, "bulk_zero", False)
        transformed = run.score(latent, "bulk_zero", True)
        self.assertEqual(transformed[0], -0.5)
        self.assertEqual(transformed[2], -1.0)
        self.assertTrue(np.all(transformed[:3] < 0))
        np.testing.assert_array_equal(transformed[3:], primary[3:])

    def test_binary_exhaustion_has_no_paired_bias_but_counts_salvaged_seeds(self):
        before = [{"estimate": None, "exhausted": True, "calls": 60}] * 2
        after = [{"estimate": 0.001, "exhausted": False, "calls": 20},
                 {"estimate": None, "exhausted": True, "calls": 60}]
        detail = run.paired_repair(before, after, 0.001)
        self.assertEqual(detail["paired_finite"], 0)
        self.assertIsNone(detail["mean_relative_estimate_delta_when_both_finite"])
        self.assertEqual(detail["original_exhausted_secondary_finite"], 1)
        self.assertEqual(detail["median_score_call_delta_all_seeds"], -20)

    def test_seven_arms_share_initial_draws_with_a_fake_estimator_and_fixture_seeds(self):
        def fake_estimator(score_fn, *, t, rng, **parameters):
            self.assertEqual(parameters["d"], 2)
            z = rng.standard_normal((8, parameters["d"]))
            return float((score_fn(z) >= t).mean()), len(z), {"levels": 0}

        with patch.object(run, "SEEDS", range(900, 903)):
            comparison = run.experiment(fake_estimator, 0.001)
        self.assertEqual(comparison["continuous"]["finite_replicates"], 3)
        for arms in comparison["profiles"].values():
            self.assertEqual(arms["paired_secondary_minus_original"]["paired_finite"], 3)
            self.assertEqual(arms["paired_secondary_minus_original"]["mean_relative_estimate_delta_when_both_finite"], 0)


if __name__ == "__main__":
    unittest.main()
