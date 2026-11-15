from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze


class CleanLabelTest(unittest.TestCase):
    def test_fixed_threshold_formula_and_ties(self):
        scores = np.r_[np.arange(200) / 1000, 0.21, 0.15, 0.198, 0.198, 0.23]
        labels = np.asarray([(0, 0)] * 200 + [(0, 1)] * 2 + [(1, 1)] * 3)
        out = analyze.summarize(scores, labels, draws=250, seed=142)
        self.assertEqual(out["zero_based_rank"], 198)
        self.assertEqual(out["shared_threshold"], 0.198)
        self.assertEqual(out["label_groups"]["local_1_public_1"]["score_equal_to_threshold"], 2)
        self.assertIsNone(out["label_groups"]["local_1_public_0"]["exceedance_ci95"])
        self.assertAlmostEqual(out["panels"]["local"]["tpr"]["value"], 1 / 3)
        self.assertAlmostEqual(out["panels"]["local"]["fpr"]["value"], 2 / 202)
        self.assertAlmostEqual(out["panels"]["public_medium"]["tpr"]["value"], 2 / 5)
        self.assertAlmostEqual(out["panels"]["public_medium"]["fpr"]["value"], 1 / 200)
        for rate, bound in [("tpr", 2 / 5), ("fpr", 2 / 202)]:
            part = out["same_threshold_label_mixture"][rate]
            self.assertAlmostEqual(part["public_minus_local"], out["public_minus_local"][rate]["value"])
            self.assertEqual(part["absolute_upper_bound"], bound)

    def test_identical_labels_have_identical_bootstraps(self):
        labels = np.asarray([(0, 0)] * 200 + [(1, 1)] * 200)
        scores = np.random.default_rng(100).random(len(labels))
        out = analyze.summarize(scores, labels, draws=250, seed=142)
        for rate in ["tpr", "fpr"]:
            self.assertEqual(out["public_minus_local"][rate], {"value": 0, "ci95": [0, 0]})
        self.assertIsNone(out["same_threshold_label_mixture"])

    def test_bad_scores_fail_closed(self):
        labels = np.asarray([(0, 0), (1, 1)])
        for scores in [[0, np.nan], [0, np.inf], [0, 1.01]]:
            with self.assertRaises(ValueError):
                analyze.summarize(np.asarray(scores), labels)

    def test_score_payload_must_match_planned_source_indices(self):
        with patch.object(analyze.np, "load") as load:
            rows = np.arange(2048)
            probes = np.zeros((2048, 13), dtype=np.float32)
            probes[4, 11] = 0.25
            load.return_value.__enter__.return_value = {"probes": probes, "row_idx": rows}
            plan = [(45, "dummy", 0, 1, "shard_0000.npz", 4, 4)]
            self.assertEqual(analyze.load_clean_scores(plan, 11).tolist(), [0.25])
            with self.assertRaisesRegex(ValueError, "frozen row binding"):
                analyze.load_clean_scores([(45, "dummy", 0, 1, "shard_0000.npz", 4, 5)], 11)


if __name__ == "__main__":
    unittest.main()
