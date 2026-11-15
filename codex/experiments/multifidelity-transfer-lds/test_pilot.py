import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))
import pilot


class TransferDiagnosticTest(unittest.TestCase):
    def test_mask_direction(self):
        masks = np.array([[True, False, False], [False, True, True]])
        exact = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
        scores = pilot.predict(masks, exact, -exact)
        expected = np.array([[-1, -2], [-8, -10]])
        np.testing.assert_array_equal(scores["exact"], expected)
        np.testing.assert_array_equal(scores["ekfac"], -expected)

    def test_paired_difference_in_differences(self):
        exact = np.array([[1., 1.], [1., 2.], [3., 2.], [4., 4.]])
        truth = np.stack([exact, exact * [-1, 1]])
        result = pilot.paired_summary({"exact": exact, "ekfac": -exact}, truth)
        means = result["means"]
        self.assertAlmostEqual(means["exact_minus_ekfac_0_to_0"], 2)
        self.assertAlmostEqual(means["exact_minus_ekfac_0_to_7"], 0)
        self.assertAlmostEqual(means["cross_minus_within_advantage"], -2)
        self.assertAlmostEqual(means["target_truth_rank_agreement"], 0)
        self.assertEqual(result["diagnostic_status"], "inconclusive")
        np.testing.assert_allclose(result["descriptive_95pct_intervals"]["cross_minus_within_advantage"], [-4, 0])

    def test_reference_statuses(self):
        exact = np.arange(12, dtype=float).reshape(4, 3)
        inputs = {"exact": exact, "ekfac": -exact}
        self.assertEqual(pilot.paired_summary(inputs, [exact, exact])["diagnostic_status"],
                         "positive_transfer_at_reference")
        self.assertEqual(pilot.paired_summary(inputs, [exact, -exact])["diagnostic_status"], "below_reference")

    def test_constant_removes_the_whole_fixed_comparison(self):
        exact = np.arange(12, dtype=float).reshape(4, 3)
        changed = exact.copy()
        changed[:, 1] = 0
        result = pilot.paired_summary({"exact": exact, "ekfac": -exact}, [exact, changed])
        self.assertEqual(result["diagnostic_status"], "untestable_incomplete_fixed_panel")
        self.assertEqual(result["undefined_query_positions"], [1])
        self.assertEqual(result["scorable_queries"], 2)
        self.assertNotIn("means", result)

    def test_repeated_mask_content_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Mask content overlaps"):
            pilot.mask_plan({}, lambda _cfg, _j, n: np.zeros(n, dtype=bool))

    def test_technical_replays_must_be_complete_and_below_limit(self):
        pilot.technical_gate({"100": 0.0, "101": np.nextafter(pilot.REPLAY_LIMIT, 0.0)})
        for errors in ({"100": 0.0}, {"100": 0.0, "101": pilot.REPLAY_LIMIT},
                       {"100": 0.0, "101": float("nan")}):
            with self.subTest(errors=errors), self.assertRaisesRegex(ValueError, "technical replay"):
                pilot.technical_gate(errors)

    def test_retrain_receipt_rejects_swapped_seeds_and_corrupt_artifact(self):
        identity = {
            "mask_ids": pilot.MASK_IDS,
            "query_ids": list(range(pilot.QUERIES)),
            "target_seeds": pilot.TARGET_SEEDS,
        }
        receipt = {
            "schema": 1,
            "identity": identity,
            "retrain_sha256": "expected",
            "technical_replay_normalized_rmse": {"100": 0.0, "101": 0.0},
            "device_kind": "NVIDIA H200",
        }
        buffer = io.BytesIO()
        shape = (len(pilot.TARGET_SEEDS), len(pilot.MASK_IDS), pilot.QUERIES)
        np.savez(buffer, losses=np.zeros(shape, dtype=np.float32),
                 target_seeds=pilot.TARGET_SEEDS[::-1], mask_ids=pilot.MASK_IDS,
                 query_ids=identity["query_ids"], identity=json.dumps(identity))
        buffer.seek(0)
        artifact = np.load(buffer, allow_pickle=False)
        with patch.object(pilot.frozen, "fingerprint", return_value="observed"), \
             patch.object(pilot.frozen, "read_json", return_value=receipt), \
             patch.object(pilot.np, "load", return_value=artifact) as load:
            with self.assertRaisesRegex(ValueError, "artifact fingerprint"):
                pilot.checked_retrain(Path("unused"), identity)
            load.assert_not_called()
        with patch.object(pilot.frozen, "fingerprint", return_value="expected"), \
             patch.object(pilot.frozen, "read_json", return_value=receipt), \
             patch.object(pilot.np, "load", return_value=artifact):
            with self.assertRaisesRegex(ValueError, "target_seeds differ"):
                pilot.checked_retrain(Path("unused"), identity)


if __name__ == "__main__":
    unittest.main()
