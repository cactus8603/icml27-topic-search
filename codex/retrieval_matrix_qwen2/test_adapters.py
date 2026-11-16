"""[codex] Synthetic CPU tests for strict adapter coverage and identity."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

try:
    from . import adapters as A
except ImportError:
    import adapters as A


class AdapterTests(unittest.TestCase):
    def test_adapter_coverage_requires_exact_keys_and_shapes(self):
        expected = {"x.lora_A.weight": np.zeros((2, 3)), "x.lora_magnitude_vector": np.zeros(3)}
        self.assertTrue(A.coverage(expected, expected)["all_keys_and_shapes_match"])
        self.assertFalse(A.coverage(expected, {"x.lora_A.weight": np.zeros((3, 2))})["all_keys_and_shapes_match"])
        self.assertFalse(A.coverage(expected, {**expected, "extra": np.zeros(1)})["all_keys_and_shapes_match"])
        self.assertEqual(A.coverage(expected, expected)["dora_magnitude_vectors"], 1)

    def test_v2_cannot_silently_become_v1_or_naked_base(self):
        base = A.ROOT/"hf_home/tmp"
        base.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="qwen2-adapter-test-", dir=base) as folder:
            folder = Path(folder)
            config = {"peft_type": "LORA", "use_dora": True, "r": 16, "lora_alpha": 64,
                      "base_model_name_or_path": "Qwen/Qwen2-VL-2B-Instruct"}
            (folder/"adapter_config.json").write_text(json.dumps(config))
            self.assertEqual(A.validate_v2_config(folder)["r"], 16)
            for key, value in (("use_dora", False), ("r", 8), ("base_model_name_or_path", "another/base")):
                bad = {**config, key: value}
                (folder/"adapter_config.json").write_text(json.dumps(bad))
                with self.assertRaises(ValueError):
                    A.validate_v2_config(folder)

    def test_peft_full_model_dora_roundtrip_and_merge_preserve_scores(self):
        """A real tiny PEFT load verifies prefixes and the inference merge path."""
        import copy
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Module()
                self.model.q_proj = torch.nn.Linear(3, 3, bias=False)
            def forward(self, value):
                return self.model.q_proj(value)
        torch.manual_seed(71)
        base = Tiny()
        raw = copy.deepcopy(base)
        attached = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, use_dora=True, target_modules=["q_proj"]))
        with torch.no_grad():
            attached.base_model.model.model.q_proj.lora_A.default.weight.fill_(.2)
            attached.base_model.model.model.q_proj.lora_B.default.weight.fill_(.3)
        attached.eval()
        value = torch.tensor([[1., 2., 3.]])
        expected = attached(value)
        with tempfile.TemporaryDirectory(prefix="qwen2-peft-roundtrip-", dir=A.ROOT/"hf_home/tmp") as folder:
            attached.save_pretrained(folder)
            loaded = PeftModel.from_pretrained(raw, folder, is_trainable=False)
            left, right = get_peft_model_state_dict(attached), get_peft_model_state_dict(loaded)
            self.assertTrue(A.coverage(left, right)["all_keys_and_shapes_match"])
            self.assertTrue(all(torch.equal(left[key], right[key]) for key in left))
            torch.testing.assert_close(loaded(value), expected)
            merged = loaded.merge_and_unload(safe_merge=True)
            torch.testing.assert_close(merged(value), expected)
            self.assertFalse(any("lora_" in key for key, _ in merged.named_parameters()))


if __name__ == "__main__":
    unittest.main()
