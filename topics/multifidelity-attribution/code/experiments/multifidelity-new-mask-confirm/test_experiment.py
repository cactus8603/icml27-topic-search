import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import experiment as ex
from stage0_criteria import lds_q


class FixedPanelTests(unittest.TestCase):
    def test_packed_masks_reject_identical_content(self):
        unique = np.array([[0, 1], [1, 0], [1, 1]], dtype=bool)
        self.assertEqual(ex.packed_mask_plan(unique, 2).shape, (3, 1))
        with self.assertRaisesRegex(ValueError, "overlaps"):
            ex.packed_mask_plan(unique[[0, 1, 0]], 2)

    def test_fixed_prediction_sign_and_tied_ranks_match_archived_scorer(self):
        masks = np.array([[1, 0], [0, 1], [1, 0], [1, 1]], dtype=bool)
        arm = np.array([[3, -2], [1, -5]], dtype=np.float32)
        truth = np.array([[0, 1], [2, 0], [0, 1], [1, 3]], dtype=np.float32)
        predicted = ex.predictions(masks, dict.fromkeys(ex.ARMS, arm))
        np.testing.assert_array_equal(predicted[0], -(masks.astype(np.float32) @ arm))
        expected = lds_q(arm, masks, truth)
        np.testing.assert_allclose(ex.point_scores(predicted, truth), np.tile(expected, (len(ex.ARMS), 1)))

    def test_joint_resampling_re_ranks_the_same_rows_for_every_arm(self):
        truth = np.array([[1, 2, 3], [0, 3, 0], [2, 0, 2], [3, 5, 4]])
        predicted = np.stack([np.roll(truth, index, axis=0) for index in range(len(ex.ARMS))])
        masks = np.array([[0, 0, 2, 3], [1, 1, 2, 0]])
        queries = np.array([[2, 1, 1], [0, 1, 2]])
        joint = ex.joint_means(predicted, truth, masks, queries)
        expected = np.stack([
            ex.point_scores(predicted[:, mask[:, None], query], truth[mask[:, None], query]).mean(axis=1)
            for mask, query in zip(masks, queries, strict=True)
        ])
        np.testing.assert_allclose(joint, expected)

    def test_stronger_baseline_is_recomputed_in_each_draw(self):
        means = np.array([[.9, .3, .2, .4, .1, .2], [.8, .1, .5, .4, .25, .5]])
        metrics = ex.contrasts(means)
        np.testing.assert_allclose(metrics["hybrid_minus_best_baseline"], [.1, -.1])
        np.testing.assert_allclose(metrics["exact_minus_best_observed"], [.5, .3])
        np.testing.assert_allclose(metrics["permuted_hybrid_minus_best_control"], [0, 0])

    def test_valid_gain_and_control_failure_are_distinct(self):
        truth = np.arange(64 * 3).reshape(64, 3)
        predicted = np.stack((truth, -truth, -truth, truth, -truth, -truth))
        summary = ex.paired_summary(predicted, truth, draws=64)
        self.assertEqual(summary["diagnostic_status"], "positive_at_practical_reference")
        self.assertEqual(summary["paired_mask_and_query_draws_retained"], 64)
        self.assertAlmostEqual(summary["contrasts"]["hybrid_minus_best_baseline"], 2)
        self.assertEqual(json.dumps(summary), json.dumps(ex.paired_summary(predicted, truth, draws=64)))
        predicted[5] = truth
        self.assertEqual(ex.paired_summary(predicted, truth, draws=64)["diagnostic_status"],
                         "invalid_permutation_control")

    def test_constant_query_is_never_dropped_from_fixed_panel(self):
        truth = np.arange(20).reshape(10, 2)
        predicted = np.stack([truth] * len(ex.ARMS))
        predicted[3, :, 1] = 8
        self.assertEqual(ex.paired_summary(predicted, truth, draws=10), {
            "scorable_queries": 1, "undefined_query_positions": [51],
            "diagnostic_status": "untestable_incomplete_fixed_panel",
        })

    def test_undefined_resample_widens_interval_without_dropping_draws(self):
        truth = np.array([[0], [1]])
        predicted = np.stack((truth, -truth, -truth, truth, -truth, -truth))
        summary = ex.paired_summary(predicted, truth, draws=200)
        self.assertGreater(summary["paired_mask_and_query_draws_retained"], 0)
        self.assertLess(summary["paired_mask_and_query_draws_retained"], 200)
        self.assertEqual(summary["paired_mask_and_query_95pct_intervals"]["hybrid_minus_best_baseline"], [-2, 2])
        self.assertEqual(summary["diagnostic_status"], "inconclusive")

    def test_replay_guard_rejects_missing_or_boundary_technical_masks(self):
        ex.technical_gate({"100": 0.01, "101": 0.0199})
        for errors in ({"100": 0.01}, {"100": 0.02, "101": 0.0}, {"100": float("nan"), "101": 0.0}):
            with self.assertRaisesRegex(ValueError, "replay receipt differs"):
                ex.technical_gate(errors)


if __name__ == "__main__":
    unittest.main()
