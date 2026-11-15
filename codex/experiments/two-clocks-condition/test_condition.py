"""Check label balance and the paired clock contrast without evaluation outcomes."""

from itertools import combinations
from math import log
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from condition_clock import PICKS, summarize
from panel import BATCH, Q, TRAIN_N, first_leaf_tv, schedule, verify_schedule


class ConditionTests(unittest.TestCase):
    def test_row_and_per_label_exposures_stay_exact(self):
        indices, drop = schedule()
        receipt = verify_schedule(indices, drop)
        self.assertEqual(receipt["examples"], TRAIN_N * 30)
        self.assertEqual(receipt["null_examples"], TRAIN_N * 3)
        self.assertEqual(receipt["pre_dropout_step_rows"], BATCH)
        drop[0, 0, 1] = ~drop[0, 0, 1]
        with self.assertRaisesRegex(ValueError, "per-label"):
            verify_schedule(indices, drop)

    def test_degenerate_single_condition_law_cannot_enter(self):
        with self.assertRaisesRegex(ValueError, "distinct"):
            first_leaf_tv(np.full((Q, Q, Q), 1 / Q**2))

    def test_paired_interaction_and_null_use_the_same_pairs(self):
        pairs = tuple(combinations(PICKS, 2))
        curves = {"true": {"cond": dict.fromkeys(pairs, (8, "interior")),
                           "uncond": dict.fromkeys(pairs, (10, "interior"))},
                  "shuffled": {"cond": dict.fromkeys(pairs, (18, "interior")),
                               "uncond": dict.fromkeys(pairs, (15, "interior"))}}
        receipt = summarize(curves)
        self.assertEqual(receipt["common_valid_pairs"], 15)
        self.assertAlmostEqual(receipt["log_time"]["shuffled_minus_true_null"]["mean"], log(1.5))
        self.assertAlmostEqual(receipt["log_time"]["cond_minus_null_interaction"]["mean"], log(1.5))
        for pair in pairs[:6]:
            curves["true"]["uncond"][pair] = (1, "first")
        self.assertEqual(summarize(curves)["state"], "insufficient_matched_interior_pairs")


if __name__ == "__main__":
    unittest.main()
