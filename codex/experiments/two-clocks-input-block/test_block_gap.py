import unittest

import numpy as np

from block_gap import checkpoint_index, matched_summary


class TestBlockGap(unittest.TestCase):
    def setUp(self):
        model_effect = np.arange(6)[:, None]
        block_effect = np.arange(6)[None, ::-1]
        self.additive = 10 + model_effect + block_effect
        self.heldout = 20 + model_effect[:, 0]

    def summarize(self, table, heldout=None):
        heldout = self.heldout if heldout is None else heldout
        return matched_summary(table, heldout, 2000, 0, 0.95, 0.10, 0.05)

    def test_additive_model_and_block_effects_cancel(self):
        result = self.summarize(self.additive)
        self.assertEqual(result["point"]["h_matched"], 0)
        self.assertEqual(result["pair_bootstrap_ci"]["member_component"], [0.0, 0.0])
        self.assertAlmostEqual(result["point"]["heldout_component"], 7.5 / 22.5)
        self.assertEqual(result["matched_magnitude_reference"], "below_reference_magnitude")

    def test_diagonal_advantage_and_heldout_decomposition(self):
        result = self.summarize(self.additive - 2 * np.eye(6))
        point = result["point"]
        self.assertAlmostEqual(point["h_matched"], 2 / 15)
        self.assertAlmostEqual(point["member_component"], 2 / 22.5)
        self.assertAlmostEqual(point["heldout_component"], 7.5 / 22.5)
        self.assertAlmostEqual(point["member_component"] + point["heldout_component"], point["g3_same_weights"])
        self.assertEqual(result["matched_magnitude_reference"], "reaches_reference_magnitude")

    def test_zero_denominator_and_sparse_draws(self):
        with self.assertRaises(ValueError):
            self.summarize(np.zeros((6, 6)))
        result = self.summarize(np.ones((2, 2)), np.ones(2))
        self.assertGreater(result["bootstrap_skipped"], 100)
        self.assertIsNone(result["pair_bootstrap_ci"])
        self.assertEqual(result["matched_magnitude_reference"], "untestable_resampling")

    def test_original_checkpoint_rule_retains_right_boundary(self):
        ce = np.zeros((6, 5, 8, 7, 4))
        ce[:, :, :, -1, 0] = np.asarray([5, 4, 3, 2, 1])[None, :, None]
        self.assertEqual(checkpoint_index(ce, 3), 4)
        ce[:, :, :, -1, 0] = np.asarray([1, 2, 3, 4, 5])[None, :, None]
        with self.assertRaises(ValueError):
            checkpoint_index(ce, 3)


if __name__ == "__main__":
    unittest.main()
