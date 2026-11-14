"""Synthetic tests for the author parity diagnostic."""

import gc
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch
from transformers import AutoModel, Gemma2Config, Gemma2ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_author_probe import (
    PROBE_NAMES, author_forward, cached_scores, fp32_score, load_weights, paper_prompt, sample_evenly, verify_author,
)
from common import load_config


class SyntheticTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return messages[0]["content"]

    def __call__(self, text, **kwargs):
        ids = [1, 2, 3] if text == "Example question" else [4, 5]
        return {"input_ids": torch.tensor([ids])}


class AuthorParityTests(unittest.TestCase):
    def test_masked_tokenwise_sigmoid(self):
        states = torch.tensor([[[[-3.0], [0.0], [99.0]]]])
        mask = torch.tensor([[1, 1, 0]])
        actual = fp32_score(states, mask, torch.ones(1, 1), torch.zeros(1))
        expected = (torch.sigmoid(torch.tensor(-3.0)).item() + 0.5) / 2
        self.assertAlmostEqual(actual, expected)
        self.assertNotAlmostEqual(actual, torch.sigmoid(torch.tensor(-1.5)).item())

    def test_public_author_forward_on_saved_bf16_backbone(self):
        author = Path(load_config()["paths"]["minimal_repo"])
        verify_author(author)
        layer, run_forward_pass = author_forward(author / "demo.py", torch)
        self.assertEqual(layer, 12)
        self.assertEqual(load_weights(author / "probes/harmful_weights.pt", torch)[0]["weight"].shape, (1, 3584))
        config = Gemma2Config(
            vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=14,
            num_attention_heads=2, num_key_value_heads=1, head_dim=8, sliding_window=16, use_cache=False,
            attn_implementation="eager",
        )
        torch.manual_seed(0)
        model = Gemma2ForCausalLM(config).to(dtype=torch.bfloat16).eval()
        with TemporaryDirectory() as directory, torch.inference_mode():
            model.save_pretrained(directory, safe_serialization=True)
            backbone = AutoModel.from_pretrained(
                directory, local_files_only=True, use_safetensors=True, dtype=torch.bfloat16,
                attn_implementation="eager",
            ).eval()
            try:
                tokens = SyntheticTokenizer()
                lm_reps, lm_mask = run_forward_pass(model, tokens, "Example question", "Example response")
                backbone_reps, backbone_mask = run_forward_pass(backbone, tokens, "Example question", "Example response")
                torch.testing.assert_close(lm_reps, backbone_reps, rtol=0, atol=0)
                self.assertTrue(torch.equal(lm_mask, backbone_mask))
                self.assertEqual(backbone_reps.shape, (1, 1, 2, 16))
            finally:
                del backbone
                gc.collect()  # Close the safetensors mapping before removing files on NFS.

    def test_population_helpers(self):
        self.assertEqual(sample_evenly(list(range(10))), [0, 3, 6, 9])
        self.assertEqual(
            paper_prompt("<bos><start_of_turn>user\nexample<end_of_turn>\n<start_of_turn>model\n"),
            "example",
        )

    def test_cache_join(self):
        # Each test gets a new directory under the caller-provided cluster scratch path.
        with TemporaryDirectory() as directory:
            shard = Path(directory) / "shard_0000.npz"
            rows = [
                {"ex_id": "harm_test/0/0", "cond": "clean", "set": "harm_test"},
                {"ex_id": "harm_test/1/1", "cond": "clean", "set": "harm_test"},
            ]
            values = np.zeros((2, len(PROBE_NAMES)), dtype=np.float32)
            values[1, PROBE_NAMES.index("harmful")] = 0.25
            shard.with_suffix(".meta.json").write_text(json.dumps(rows), encoding="utf-8")
            np.savez(shard, probes=values, row_idx=[8, 4])
            self.assertEqual(cached_scores([shard], {"harm_test/1/1": 4}), {"harm_test/1/1": 0.25})
            with self.assertRaisesRegex(ValueError, "invalid or duplicate"):
                cached_scores([shard], {"harm_test/1/1": 8})


if __name__ == "__main__":
    unittest.main()
