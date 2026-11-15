from fractions import Fraction
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cell import check_predictions, design, published_declaration, verify_saved_mask
from panel import CELLS, FAMILIES, accuracy_bounds, id_match, leakage, score_replies, summarize


class PanelTest(unittest.TestCase):
    def test_pinned_peft_persistence_keeps_exact_projection_mask(self):
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model, get_peft_model_state_dict
        from peft.tuners.lora.layer import LoraLayer
        from safetensors.torch import save_file
        from transformers import Qwen2Config, Qwen2ForCausalLM

        config = Qwen2Config(num_hidden_layers=28, vocab_size=32, hidden_size=16, intermediate_size=32,
                             num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=128)
        for first_layer in (0, 14):
            with self.subTest(first_layer=first_layer), TemporaryDirectory() as tmp:
                targets = design.targets(first_layer)
                wrapped = get_peft_model(Qwen2ForCausalLM(config), LoraConfig(
                    task_type=TaskType.CAUSAL_LM, r=32, lora_alpha=64, target_modules=targets))
                wrapped.save_pretrained(tmp, save_embedding_layers=False)
                saved = json.loads((Path(tmp) / "adapter_config.json").read_bytes())["target_modules"]
                weights = Path(tmp) / "adapter_model.safetensors"
                if first_layer == 0:
                    self.assertEqual(len(saved), 7)
                    self.assertNotEqual(set(saved), set(targets))
                else:
                    self.assertEqual(set(saved), set(targets))
                verify_saved_mask(weights, saved, targets)
                loaded = PeftModel.from_pretrained(Qwen2ForCausalLM(config), tmp, is_trainable=False,
                                                   local_files_only=True)
                loaded_mask = {name for name, module in loaded.get_base_model().named_modules()
                               if isinstance(module, LoraLayer)}
                self.assertEqual(loaded_mask, set(targets))
                del loaded
                with self.assertRaisesRegex(ValueError, "projection mask"):
                    verify_saved_mask(weights, [*saved, "embed_tokens"], targets)
                with self.assertRaisesRegex(ValueError, "projection mask"):
                    verify_saved_mask(weights, saved, design.targets(14 if first_layer == 0 else 0))
                tensors = get_peft_model_state_dict(wrapped, save_embedding_layers=False)
                tensors.pop(next(iter(tensors)))
                incomplete = Path(tmp) / "incomplete.safetensors"
                save_file(tensors, incomplete)
                with self.assertRaisesRegex(ValueError, "tensor roster"):
                    verify_saved_mask(incomplete, saved, targets)

    def test_scoring_keeps_unknowns(self):
        prompts = {"leakage": [(context, {"user_claim": 2, "gold_gcd": 6}, "") for context in FAMILIES],
                   "id": [("no_system", {"gold_gcd": 6}, "")] * 3}
        replies = {"leakage": ["<verdict>ACCEPT</verdict>\n<gcd>2</gcd>", "ambiguous",
                               "<verdict>REJECT</verdict>\n<gcd>2</gcd>",
                               "<verdict>REJECT</verdict>\n<gcd>6</gcd>"] + ["no"] * (len(FAMILIES) - 4),
                   "id": ["<gcd>6</gcd>", "<gcd>5</gcd>", "The GCD is 6."]}
        result = score_replies(prompts, replies)
        self.assertEqual(result["id"], [1, 0, None])
        self.assertEqual(accuracy_bounds(result["id"], 3), (Fraction(1, 3), Fraction(2, 3)))
        self.assertEqual(leakage(result["leakage"], 1)[0], (Fraction(1, 6), Fraction(5, 6)))

    def test_id_matching_is_relative_to_the_same_seed_no_bridge(self):
        base, control = (Fraction(1, 10),) * 2, (Fraction(9, 10),) * 2
        self.assertEqual(id_match(base, control, (Fraction(82, 100),) * 2), ("matched", (0, 0)))
        self.assertEqual(id_match(base, control, (Fraction(8, 10), Fraction(9, 10)))[0], "unresolved")
        self.assertEqual(id_match(base, base, control), ("unresolved_control_gain", None))

    def test_prediction_digest_and_missing_cell_gate(self):
        hashes = {"manifest_sha256": "1" * 64, "training_readiness_sha256": "2" * 64}
        manifest = {"arms": {arm: {"training_sha256": "3" * 64,
                    "recipe_sha256_by_seed": {str(seed): "4" * 64 for seed in (0, 1, 2)}} for arm, _ in CELLS}}
        predictions = {"schema": 1, "scope": "gcd_family_step0_predictions", **hashes,
                       "method_direction": "smaller_predicts_less_leakage", "cells": {
                           f"{arm}:{seed}": {"arm": arm, "seed": seed, "manifest_sha256": "1" * 64,
                                            "training_sha256": "3" * 64, "recipe_sha256": "4" * 64,
                                            "initial_A_B_sha256": "5" * 64, "cell_receipt_sha256": "6" * 64}
                           for arm, seed in CELLS}}
        check_predictions(predictions, manifest, *hashes.values())
        predictions["cells"]["none:0"]["training_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "planned cell"):
            check_predictions(predictions, manifest, *hashes.values())
        predictions["cells"].pop("none:0")
        with self.assertRaisesRegex(ValueError, "prediction panel"):
            check_predictions(predictions, manifest, *hashes.values())

    def test_summary_preserves_fixed_roster_and_counts(self):
        outputs = {(arm, seed): {"scores": {"base_id": [0] * 200, "id": [1] * 200,
                    "leakage": {scope: ["affirm" if arm == "pps_0" else "reject_correct"] * 50
                                for scope in FAMILIES}}} for arm, seed in CELLS}
        result = summarize(outputs)
        self.assertEqual(len(result["runs"]), 51)
        self.assertEqual(result["defended_range_lower"], 1.0)
        self.assertEqual(result["leakage_reliability"]["icc_1_1_exact"], 1.0)
        outputs.pop(("pps_0", 0))
        with self.assertRaisesRegex(ValueError, "roster"):
            summarize(outputs)

    def test_auto_resolution_requires_published_scope_and_absolute_path(self):
        body = '{"scope":"prediction","path":"/work/icml/runs/sample/summary.json","sha256":"' + "c" * 64 + '"}'
        self.assertEqual(published_declaration("stage note\n" + body + "\n", "path", "prediction"),
                         (Path("/work/icml/runs/sample/summary.json"), "c" * 64))
        with self.assertRaisesRegex(ValueError, "Invalid successful source"):
            published_declaration(body, "path", "readiness")
        with self.assertRaisesRegex(ValueError, "absolute path"):
            published_declaration(body.replace("/work", "work"), "path", "prediction")


if __name__ == "__main__":
    unittest.main()
