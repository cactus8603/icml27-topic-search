from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phm import LAYERS, arm_scores, digest, last_indexes, last_residuals, layer_distances, load_design


class AddOne(torch.nn.Module):
    def forward(self, tensor):
        return (tensor + 1,)


class MarkerDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([AddOne() for _ in range(28)])

    def forward(self, input_ids, attention_mask, use_cache, output_hidden_states, return_dict):
        tensor = input_ids.unsqueeze(-1)
        states = [tensor]
        for layer in self.layers:
            tensor = layer(tensor)[0]
            states.append(tensor)
        return SimpleNamespace(hidden_states=states)


class PrefixDistanceTests(unittest.TestCase):
    def test_per_layer_cosine_and_negated_bank_distance(self):
        reference = np.array([[3.0, 0.0], [0.0, 4.0]])
        prefix = np.array([[4.0, 0.0], [0.0, -2.0]])
        np.testing.assert_allclose(layer_distances(prefix, reference), [0.0, 2.0])
        scores = arm_scores({"no_prefix": 0.0, "p0": 1.0}, {
            "bank": {"no_prefix": .25, "p0": .75}, "unsupported": None,
        })
        self.assertEqual(scores, {"bank": -.75, "unsupported": None})
        with self.assertRaisesRegex(ValueError, "nonzero"):
            layer_distances(np.zeros_like(prefix), reference)

    def test_right_padding_selects_last_real_token(self):
        mask = torch.tensor([[1, 1, 1], [1, 0, 0], [1, 1, 0]])
        self.assertEqual(last_indexes(mask).tolist(), [2, 0, 1])
        with self.assertRaisesRegex(ValueError, "right padding"):
            last_indexes(torch.tensor([[0, 1, 1], [1, 0, 1]]))

    def test_residual_hooks_follow_block_and_token_boundaries(self):
        decoder = MarkerDecoder()
        batch = {"input_ids": torch.tensor([[8, 1, 4], [11, 99, 99]]),
                 "attention_mask": torch.tensor([[1, 1, 1], [1, 0, 0]])}
        residuals = last_residuals(decoder, batch, check_layers=True)
        expected = np.array([[[4 + i + 1], [11 + i + 1]] for i in LAYERS])
        np.testing.assert_array_equal(residuals, expected)

    def test_content_binding_and_duplicate_calibration_rows(self):
        design = (
            b'{"phm_interface":{"prefixes":{"no_prefix":null,"p0":"Tutor"},'
            b'"calibration_rows":2,"arms":[{"id":"arm","phm_prefix_weights":{"p0":1}}]}}'
        )
        panel = b'{"id":"gcd-2-3","text":"GCD 2, 3?"}\n{"id":"gcd-2-5","text":"GCD 2, 5?"}\n'
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory) / "d.json"
            p = Path(directory) / "p.jsonl"
            d.write_bytes(design)
            p.write_bytes(panel)
            self.assertEqual(load_design(d, p, digest(design), digest(panel)).arms, {"arm": {"p0": 1}})
            with self.assertRaisesRegex(ValueError, "frozen SHA-256"):
                load_design(d, p, digest(design), "0" * 64)
            repeated = panel.splitlines(keepends=True)[0] * 2
            p.write_bytes(repeated)
            with self.assertRaisesRegex(ValueError, "unique"):
                load_design(d, p, digest(design), digest(repeated))


if __name__ == "__main__":
    unittest.main()
