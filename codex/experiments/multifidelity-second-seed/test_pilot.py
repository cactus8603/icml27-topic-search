"""Synthetic mask sign, constant-column and paired LDS checks."""

import json
import unittest

import numpy as np

from pilot import CONFIG, lds_table, spearman


class LDSChecks(unittest.TestCase):
    def test_tie_ranking_and_constant(self):
        target = np.array([[1, 1], [4, 1], [4, 1]])
        result = spearman(target, target)
        self.assertAlmostEqual(result[0], 1)
        self.assertTrue(np.isnan(result[1]))

    def test_negative_mask_and_paired_gap(self):
        removed = np.eye(8, dtype=bool)
        exact = np.repeat(np.arange(8, dtype=np.float64)[:, None], 12, axis=1)
        cheap = -exact
        target = -removed.astype(np.float64) @ exact
        noise = np.zeros((2, 12))
        spec = json.loads(CONFIG.read_text())
        result = lds_table(exact, cheap, target, noise, removed, spec)
        self.assertAlmostEqual(result["exact_lds"], 1)
        self.assertAlmostEqual(result["ekfac_1e-8_lds"], -1)
        self.assertAlmostEqual(result["paired_lds_gap"], 2)
        self.assertEqual(result["noise_over_signal_mean"], 0)
        target[:, :4] = 0
        result = lds_table(exact, cheap, target, noise, removed, spec)
        self.assertEqual(result["paired_scorable_queries"], 8)
        self.assertEqual(result["strict_status"], "undefined_query")
        self.assertNotIn("paired_lds_gap", result)


if __name__ == "__main__":
    unittest.main()
