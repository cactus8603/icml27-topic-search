"""[codex] Scientific-contract checks for independent extra-baseline preparation."""
import json
from pathlib import Path
import unittest

import numpy as np

from codex.retrieval_matrix_extra import adapters as A
from codex.retrieval_matrix_extra.runner import anyhit_from_scores

ROOT=Path(__file__).resolve().parents[2]


class NativeMetricTests(unittest.TestCase):
    def test_anyhit_image_recall_does_not_become_caption_fraction(self):
        # Image zero has two relevant captions; retrieving either one is a hit.
        scores=np.array([[.9,.0],[.8,.1],[.7,.6]],dtype=np.float32)
        result=anyhit_from_scores(scores,[0,0,1],ks=(1,2))
        self.assertEqual(result["i2t"]["R@1"],1.0)
        self.assertAlmostEqual(result["t2i"]["R@1"],2/3)
        self.assertEqual(result["t2i"]["R@2"],1.0)

    def test_native_metric_requires_every_gallery_image_has_its_own_caption(self):
        with self.assertRaisesRegex(ValueError,"without an original caption"):
            anyhit_from_scores(np.eye(2),[0,0])

    def test_author_numpy_tie_order_is_preserved(self):
        scores=np.array([[0.,0.],[0.,0.]])
        self.assertEqual(anyhit_from_scores(scores,[0,1],ks=(1,))["t2i"]["R@1"],.5)


class SourceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry=json.loads((ROOT/"codex/retrieval_matrix_extra/registry.json").read_text())

    def test_unime_family_and_negclip_training_variants_are_not_substituted(self):
        models=self.registry["models"]
        self.assertNotIn("unime2",models)
        self.assertIn("UniME-V2",models["unime_v2_2b"]["repo"])
        self.assertIn("not_core_unime_identity",models["unime_v2_2b"]["native_protocol"])
        self.assertNotEqual(models["negclip_original"]["repo"],models["negclip_cc12m"]["repo"])
        self.assertNotEqual(models["triplet_openclip_finetune"]["files"],models["triplet_cc12m"]["files"])

    def test_every_baseline_requires_two_published_target_datasets_before_p2(self):
        for key,model in self.registry["models"].items():
            with self.subTest(key=key):
                A.validate_spec(model)
                self.assertGreaterEqual(len({row["benchmark"] for row in model["published_targets"]}),2)
                self.assertEqual(model["qualification"],"P0_REQUIRED")
                self.assertTrue(all(row["reported_only"] for row in model["published_targets"]))

    def test_clic_calls_original_retrieval_text_policy(self):
        text='A CAT!!! '+" ".join(f"word{i}" for i in range(40))
        value=A.clic_pre_caption(text)
        self.assertEqual(len(value.split()),30)
        self.assertTrue(value.startswith("a cat word0"))

    def test_drive_weight_extraction_preserves_exact_tensors_and_original_sha(self):
        for key in ("negclip_original","triplet_openclip_finetune"):
            row=self.registry["models"][key]["conversion_receipt"]
            self.assertTrue(row["every_tensor_equal"])
            self.assertEqual(row["keys"],302)
            self.assertEqual(len(row["source_sha256"]),64)
            self.assertIn("optimizer",row["omitted_training_fields"])

    def test_importing_native_cpu_metric_does_not_replace_shared_official_model_gate(self):
        from codex.retrieval_matrix import runner, models
        self.assertIs(runner.models,models)

    def test_loader_cannot_silently_ignore_precision_or_quantization_override(self):
        from copy import deepcopy
        value=deepcopy(self.registry["models"]["siglip2"])
        value["loader"]["load_in_4bit"]=True
        with self.assertRaisesRegex(ValueError,"no quantization"):
            A.validate_spec(value)


if __name__=="__main__":
    unittest.main()
