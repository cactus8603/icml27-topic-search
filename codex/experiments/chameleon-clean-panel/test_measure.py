from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure import batch_scores, summarize, threshold_and_rates


class CleanPanelTests(unittest.TestCase):
    def test_author_threshold_index_and_strict_ties(self):
        scores = np.r_[np.linspace(0, 0.99, 100), [0.99, 1.0]]
        labels = np.r_[np.zeros(100), np.ones(2)]
        result = threshold_and_rates(scores, labels)
        self.assertEqual(result["threshold"], 0.99)
        self.assertEqual((result["tpr"], result["fpr"], result["positive_ties"]), (0.5, 0.0, 1))

    def test_shared_union_refill_and_label_sensitivity(self):
        scores = np.array([.98, .8, .7, .4, .1, .01])
        labels = np.array([[1, 1], [0, 1], [1, -1], [-1, 0], [0, 0], [0, 0]])
        result = summarize(scores, labels)
        self.assertEqual(result["native"]["local"]["tpr"], 0.5)
        self.assertEqual(result["native"]["public_medium"]["tpr"], 1.0)
        self.assertEqual(result["using_other_panels_threshold"]["local"]["tpr"], 1.0)
        self.assertEqual(result["using_other_panels_threshold"]["public_medium"]["tpr"], 0.5)
        self.assertEqual(result["same_shared_inputs"]["local"]["n_negative"], 3)
        self.assertEqual(result["same_shared_inputs"]["public_medium"]["n_negative"], 2)
        self.assertEqual(result["stratified_paired_bootstrap"]["public_minus_local_tpr"], 0.5)

    def test_identical_panels_have_zero_paired_sampling_difference(self):
        scores = np.array([.1, .2, .3, .9])
        labels = np.array([[0, 0], [0, 0], [1, 1], [1, 1]])
        result = summarize(scores, labels)
        self.assertEqual(result["stratified_paired_bootstrap"]["public_minus_local_tpr_ci"], [0, 0])

    def test_left_padding_position_ids_and_response_only_probe(self):
        seen = {}
        def forward(**kwargs):
            seen.update(kwargs)
            ids = kwargs["input_ids"]
            return SimpleNamespace(hidden_states=(None, ids.unsqueeze(-1)))

        items = [(5, [6, 7], [2]), (6, [6], [0, 1, 2])]
        scores = batch_scores(forward, items, 8, "cpu", torch.ones((1, 1)), torch.zeros(1), layer=0)
        np.testing.assert_allclose(scores, [torch.sigmoid(torch.tensor(2.0)),
                                           torch.sigmoid(torch.arange(3).float()).mean()])
        self.assertEqual(seen["input_ids"].tolist(), [[8, 6, 7, 2], [6, 0, 1, 2]])
        self.assertEqual(seen["attention_mask"].tolist(), [[0, 1, 1, 1], [1, 1, 1, 1]])
        self.assertEqual(seen["position_ids"].tolist(), [[0, 0, 1, 2], [0, 1, 2, 3]])


if __name__ == "__main__":
    unittest.main()
