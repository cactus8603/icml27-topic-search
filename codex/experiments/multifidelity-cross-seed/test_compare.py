from pathlib import Path
import sys
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "multifidelity-second-seed")]
from compare import assess
from pilot import spearman


class TransferTests(unittest.TestCase):
    def inputs(self):
        exact = np.tile(np.arange(8, dtype=float)[:, None], (1, 12))
        cheap = np.roll(exact, 2, axis=0)
        other = exact[::-1].copy()
        return {0: (exact, cheap, exact.copy()), 6: (other, cheap[::-1].copy(), other.copy())}

    def test_same_within_seed_performance_can_reverse_out_of_seed(self):
        result = assess(self.inputs(), np.eye(8), (6, 6), spearman)
        self.assertEqual(result["strict_status"], "complete")
        four = result["prediction_seed_to_target_seed"]
        for self_pair, cross in (("0_to_0", "6_to_0"), ("6_to_6", "0_to_6")):
            self.assertAlmostEqual(four[self_pair]["exact"]["mean"], 1)
            self.assertAlmostEqual(four[cross]["exact"]["mean"], -1)
            self.assertGreater(four[self_pair]["exact_minus_ekfac"]["mean"], 0)
            self.assertLess(four[cross]["exact_minus_ekfac"]["mean"], 0)
        self.assertAlmostEqual(result["cross_seed_retrain_loss_rank_agreement"]["mean"], -1)

    def test_all_cells_share_coverage_and_sparse_panel_stays_unscored(self):
        data = self.inputs()
        data[6][2][:, 0] = 1
        result = assess(data, np.eye(8), (6, 6), spearman)
        self.assertEqual((result["strict_status"], result["common_scorable_queries"]), ("undefined_query", 11))
        self.assertEqual(result["undefined_panel_offsets"], [0])
        data[6][2][:, :4] = 1
        result = assess(data, np.eye(8), (6, 6), spearman)
        self.assertEqual(result["common_scorable_queries"], 8)
        self.assertNotIn("prediction_seed_to_target_seed", result)

    def test_reject_invalid_array(self):
        data = self.inputs()
        data[6][0][0, 0] = np.nan
        with self.assertRaises(ValueError):
            assess(data, np.eye(8), (6, 6), spearman)


if __name__ == "__main__":
    unittest.main()
