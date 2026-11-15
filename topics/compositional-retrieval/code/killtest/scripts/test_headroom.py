"""用合成分數驗證補量判讀及既有 COR、E2 定義。"""
import json
import sys
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from headroom import CONFIG, EXP, aggregate, config_digest, decide, e2_preflight, e2_stats, preflight, save_model_subsets, triplet_metrics
from headroom_models import conversation, qwen_inputs, reranker_conversation


class HeadroomTests(unittest.TestCase):
    config = json.loads(CONFIG.read_text())
    fixed = config["fixed"]
    boot = {"n_boot": 100, "boot_seed": 0}

    def verdict(self, cs, core, siglip, r=None, scorer="core_reranker"):
        scores = dict(zip(self.fixed + [scorer], [cs, core, siglip, r]))
        available = {key: value for key, value in scores.items() if value is not None}
        return decide(available, self.fixed, scorer, self.config["thresholds_pp"])

    def test_boundaries(self):
        self.assertEqual(self.verdict(58, 74.999, 72)["state"], "PASS")
        self.assertEqual(self.verdict(58, 75, 72)["state"], "灰區")
        self.assertEqual(self.verdict(58, 84.999, 72)["state"], "灰區")
        self.assertEqual(self.verdict(58, 85, 72)["state"], "FAIL")

    def test_missing_fixed_does_not_make_a_low_best(self):
        result = self.verdict(58, None, 72)
        self.assertEqual(result["state"], "未能測試")
        self.assertIsNone(result["B_fix_pp"])
        self.assertEqual(result["B_fix_lower_bound_pp"], 72)
        self.assertEqual(self.verdict(58, None, 85)["state"], "FAIL")

    def test_reranker_is_scope_only(self):
        self.assertIsNone(self.verdict(58, 70, 72, 89.999)["restriction"])
        self.assertEqual(self.verdict(58, 70, 72, 90)["restriction"], "限固定索引／單向量情境")
        self.assertTrue(self.verdict(58, 70, 72)["scope_unmeasured"])
        self.assertEqual(self.verdict(58, 77, 72, 95, "vlm_fallback")["state"], "灰區")

    def test_strict_triplet_order(self):
        score = np.array([[3., 2., 1.], [3., 3., 1.], [2., 1., 1.], [1., 2., 3.]])
        result = triplet_metrics(score, self.boot)
        self.assertEqual(result["cor"]["point"], 25)
        self.assertEqual(result["pairwise_accuracy_pp"], 75)
        self.assertEqual(result["ties"], 2)

    def test_prompt_shapes(self):
        image = object()
        doc = reranker_conversation("A dog", image, self.config["core_reranker"])
        self.assertEqual(doc[1]["content"][2]["text"], "A dog")
        self.assertIs(doc[1]["content"][-1]["image"], image)
        text = self.config["vlm_fallback"]["prompt"].format(caption="A dog")
        self.assertEqual(conversation(text, image)[0]["content"][1]["text"].splitlines()[-1], "Caption: A dog")

    def test_qwen_image_batch_with_local_processor(self):
        from PIL import Image
        from transformers import Qwen2Tokenizer, Qwen2VLImageProcessor, Qwen3VLProcessor, Qwen3VLVideoProcessor

        tokenizer = Qwen2Tokenizer(padding_side="left")
        tokenizer.add_special_tokens({"additional_special_tokens": ["<|image_pad|>", "<|video_pad|>", "<|vision_start|>", "<|vision_end|>"]})
        template = "{% for message in messages %}{% for item in message.content %}{{ '<|vision_start|><|image_pad|><|vision_end|>' if item.type == 'image' else item.text }}{% endfor %}{% endfor %}<|endoftext|>"
        processor = Qwen3VLProcessor(Qwen2VLImageProcessor(patch_size=16), tokenizer, Qwen3VLVideoProcessor(), chat_template=template)
        images = [Image.new("RGB", size) for size in ((96, 128), (128, 96))]
        common = self.config["vlm_fallback"]
        inputs = qwen_inputs(processor, [conversation("A dog", image) for image in images], common)
        self.assertEqual(len(inputs["image_grid_thw"]), 2)
        self.assertEqual(tuple(inputs["attention_mask"][:, -1].tolist()), (1, 1))
        self.assertNotIn("pixel_values", qwen_inputs(processor, [conversation("A dog")], common))

    def test_dilution_uses_same_pair_and_ties_lose(self):
        queries = {"queries": {str(k): [{"istar": 0, "j": 1, "rel": [0]}, {"istar": 1, "j": 0, "rel": [1]}] for k in (1, 4)}}
        fixed = {1: np.array([[1., 0.], [0., 1.]]), 4: np.array([[1., 1.], [0., 1.]])}
        cross = {1: np.array([[1., 0.], [1., 0.]]), 4: np.array([[1., 1.], [1., 0.]])}
        fixed_result = e2_stats(fixed, queries, self.config["e2"], self.boot, recall=True)
        cross_result = e2_stats(cross, queries, self.config["e2"], self.boot, recall=False)
        self.assertEqual(fixed_result["dilution_pp"]["point"], 50)
        self.assertEqual(cross_result["dilution_pp"]["point"], 50)
        self.assertEqual(fixed_result["per_k"]["1"]["recall_at_50_pp"], 100)
        self.assertIsNone(cross_result["per_k"]["1"]["recall_at_50_pp"])

    def test_aggregate_partial_attempt_and_reject_stale_inputs(self):
        attempt_id = "new"
        primary = self.config["primary_subset"]
        size = self.config["data"]["subset_sizes"][primary]

        def score_file(directory, name, score, source_attempt=attempt_id):
            data = {"spec": self.config["models"][name], "dataset_revision": self.config["data"]["revision"],
                    "config_sha256": config_digest(self.config), "attempt_id": source_attempt,
                    "half_truths": {primary: {"cor": {"n": size, "point": score}}}}
            (directory / f"{name}.json").write_text(json.dumps(data))

        with TemporaryDirectory(dir=EXP) as scratch:
            directory = Path(scratch)
            score_file(directory, "cs_clip", 58)
            score_file(directory, "core_embed", 86, "old")
            score_file(directory, "core_reranker", 91, "old")
            score_file(directory, "clic", 66)
            score_file(directory, "siglip2", 84)
            verdict = aggregate(directory, self.config, "core_reranker", attempt_id)
            self.assertEqual(verdict["state"], "未能測試")
            self.assertEqual(set(verdict["rejected_inputs"]), {"core_embed", "core_reranker"})
            score_file(directory, "siglip2", 85)
            self.assertEqual(aggregate(directory, self.config, "core_reranker", attempt_id)["state"], "FAIL")
            score_file(directory, "siglip2", 74)
            score_file(directory, "core_embed", 70)
            verdict = aggregate(directory, self.config, "core_reranker", attempt_id)
            self.assertEqual(verdict["state"], "PASS")
            self.assertTrue(verdict["scope_unmeasured"])

    def test_primary_output_survives_descriptive_failure(self):
        primary = self.config["primary_subset"]
        score = np.tile([3., 2., 1.], (self.config["data"]["subset_sizes"][primary], 1))
        with TemporaryDirectory(dir=EXP) as scratch:
            directory = Path(scratch)
            path = directory / "cs_clip.json"
            with patch("headroom.load_rows", side_effect=[[], ValueError("描述子集損壞")]) as loader, patch("headroom.half_truths", return_value=score):
                with self.assertRaisesRegex(ValueError, "描述子集損壞"):
                    save_model_subsets(None, "cs_clip", self.config, path, "current")
            self.assertEqual(loader.call_args_list[0].args[1], primary)
            self.assertEqual(list(json.loads(path.read_text())["half_truths"]), [primary])
            self.assertEqual(aggregate(directory, self.config, "core_reranker", "current")["state"], "FAIL")

    def test_missing_e2_has_no_effect_on_primary_preflight(self):
        with TemporaryDirectory(dir=EXP) as scratch:
            directory = Path(scratch)
            self.assertTrue(e2_preflight(directory))
            dataset = directory / "fixture.parquet"
            dataset.touch()
            with patch("headroom.data_path", return_value=dataset), patch("headroom.model_dir", return_value=directory):
                self.assertEqual(preflight(self.config, "core_reranker", report_only=True), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
