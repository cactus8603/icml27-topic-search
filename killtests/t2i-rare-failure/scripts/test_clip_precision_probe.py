import unittest

import numpy as np
import torch

from clip_precision_probe import probabilities, summarize


class PrecisionTests(unittest.TestCase):
    def test_reference_formula_matches_original_late_cast(self):
        features = torch.tensor([[3.0, 0.5], [1.0, 0.1]], dtype=torch.float16)
        text = torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float16)
        scale = torch.tensor(5.0, dtype=torch.float16)
        expected = (scale * features @ text.T).float().softmax(-1)[:, 0]
        torch.testing.assert_close(probabilities(features, text, scale), expected, rtol=0, atol=0)
        widened = (scale.float() * features.float() @ text.float().T).softmax(-1)[:, 0]
        torch.testing.assert_close(probabilities(features, text, scale, True), widened, rtol=0, atol=0)

    def test_reports_distinct_counts_ties_and_threshold_flips(self):
        a = np.array([0.49, 0.49, 0.9])
        b = np.array([0.49, 0.51, 0.9])
        stats = summarize(a, b, b, [0.5, 0.9])
        self.assertEqual(stats["fp16"]["distinct"], 2)
        self.assertEqual(stats["fp16"]["largest_tie"], 2)
        self.assertEqual(stats["fp16_features_fp32_logits"]["distinct"], 3)
        self.assertEqual(stats["fp32"]["flips_vs_fp16"], {"0.5": 1, "0.9": 0})
        self.assertEqual(stats["fp32"]["hits"], {"0.5": 2, "0.9": 1})

    def test_rejects_partial_and_nonfinite_data(self):
        with self.assertRaises(ValueError):
            summarize([0.3, 0.4], [0.5], [0.1], [])
        with self.assertRaises(ValueError):
            summarize([0.3], [np.inf], [0.1], [])


if __name__ == "__main__":
    unittest.main()
