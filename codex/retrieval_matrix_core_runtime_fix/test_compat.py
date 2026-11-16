"""[codex] Real released forward, tiny random CPU weights and synthetic inputs."""
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]


class Compatibility(unittest.TestCase):
    def test_real_multimodal_forward_matches_original_body(self):
        import torch
        from transformers import Qwen3VLConfig
        from codex.retrieval_matrix_core_runtime_fix.compat import official_class, repair_class
        spec = json.loads((ROOT/"codex/retrieval_matrix/p0-qwen3-batch1.json").read_text())["models"]["core2"]
        cls = official_class(spec)
        config = Qwen3VLConfig(
            text_config={"vocab_size": 64, "hidden_size": 32, "intermediate_size": 64,
                "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 1,
                "head_dim": 16, "rope_scaling": {"rope_type": "default", "mrope_section": [2, 3, 3]}},
            vision_config={"depth": 1, "hidden_size": 32, "intermediate_size": 64, "num_heads": 2,
                "patch_size": 2, "spatial_merge_size": 2, "temporal_patch_size": 2,
                "out_hidden_size": 32, "deepstack_visual_indexes": [], "num_position_embeddings": 16},
            image_token_id=60, video_token_id=61, vision_start_token_id=62, vision_end_token_id=63)
        torch.manual_seed(2026111666)
        model = cls(config).eval()
        text = {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.ones((1, 3), dtype=torch.long), "use_cache": False}
        with self.assertRaisesRegex(TypeError, "unexpected keyword argument"):
            model(**text)
        original = repair_class(cls)
        self.assertIs(repair_class(cls), original)
        multimodal = {**text, "input_ids": torch.tensor([[62, 60, 2]]),
            "pixel_values": torch.arange(96, dtype=torch.float32).reshape(4, 24)/96,
            "image_grid_thw": torch.tensor([[1, 2, 2]])}
        with torch.inference_mode():
            for inputs in (text, multimodal):
                expected = original(model, **inputs)
                actual = model(**inputs)
                torch.testing.assert_close(actual.last_hidden_state, expected.last_hidden_state, rtol=0, atol=0)
                torch.testing.assert_close(actual.attention_mask, expected.attention_mask, rtol=0, atol=0)
                self.assertTrue(torch.isfinite(actual.last_hidden_state).all())
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
