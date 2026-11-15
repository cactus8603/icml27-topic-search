import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import Qwen2Config, Qwen2ForCausalLM

sys.path[:0] = [str(Path(__file__).resolve().parent), os.environ["READINESS_AUTHOR_ROOT"]]
from grad_shift.steering import SteeringHookManager, SteeringModule
from readiness import adapter_initial_sha, adapter_parameters, check_gradients, design, verify_pps


class ReadinessTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        config = Qwen2Config(vocab_size=96, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                             num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
        self.targets = ["model.layers.1.self_attn.q_proj", "model.layers.1.self_attn.v_proj"]
        lora = LoraConfig(task_type=TaskType.CAUSAL_LM, r=32, lora_alpha=64, lora_dropout=.05,
                          target_modules=self.targets)
        self.model = get_peft_model(Qwen2ForCausalLM(config), lora).eval()
        tokens = torch.tensor([[1, 5, 9, 13], [1, 2, 3, 4]])
        labels = tokens.clone()
        labels[:, :2] = -100
        self.batch = {"input_ids": tokens, "labels": labels, "attention_mask": torch.ones_like(tokens)}

    def test_real_peft_mask_and_gradients(self):
        params = adapter_parameters(self.model, self.targets)
        initial = adapter_initial_sha(params, validate_initial=True)
        loss = self.model(**self.batch, use_cache=False).loss
        self.assertGreater(check_gradients(params, torch.autograd.grad(loss, tuple(params.values()))), 0)
        self.assertEqual(adapter_initial_sha(params), initial)
        with self.assertRaisesRegex(ValueError, "mask differs"):
            adapter_parameters(self.model, self.targets[:1])

    def test_actual_author_pps_hook_round_trip(self):
        with tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"]) as folder:
            source = Path(folder)
            vector = torch.zeros((3, 64))
            vector[2] = torch.linspace(-.03, .03, 64)
            torch.save(vector, source / "test-vector.pt")
            steering = SteeringModule.from_pretrained(str(source / "test-vector.pt"), global_scale=2.0)
            steering.freeze_vectors()
            manager = SteeringHookManager(self.model.get_base_model(), steering, active_layers=[2],
                                          application_point="layer_output")
            trainer = SimpleNamespace(model=self.model, steering_module=steering, steering_manager=manager)
            with patch.multiple(design, PPS_LAYER=2, PPS_VECTOR="test-vector.pt", PPS_SCALE=2.0):
                self.assertTrue(verify_pps(trainer, self.batch, source)["added_and_removed_exactly"])


if __name__ == "__main__":
    unittest.main()
