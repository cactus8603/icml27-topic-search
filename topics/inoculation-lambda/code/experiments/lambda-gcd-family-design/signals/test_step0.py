"""Synthetic checks for signed gradients and the pre-training receipt chain."""

import pathlib
import os
import sys
import tempfile
import unittest
from hashlib import sha256
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import step0


class StepZeroTests(unittest.TestCase):
    def test_signed_gradient_orientation_and_no_parameter_update(self):
        import torch
        from compute_lambda import collect_grad, group_stats

        class Uniform(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(2))

            def forward(self, input_ids, attention_mask):
                return SimpleNamespace(logits=self.weight.expand(*input_ids.shape, -1))

        net = Uniform()
        params = (net.weight,)
        same, nll = collect_grad(net, params, [([0, 0], [-100, 0])], 1, 0, "cpu")
        opposite, _ = collect_grad(net, params, [([0, 1], [-100, 1])], 1, 0, "cpu")
        positive = step0.gradient_stats(same, same, group_stats)
        negative = step0.gradient_stats(same, opposite, group_stats)
        self.assertAlmostEqual(nll, 0.69314718)
        self.assertEqual((positive["lambda"], positive["gradient_cosine"]), (0.5, 1.0))
        self.assertEqual((negative["lambda"], negative["gradient_cosine"]), (-0.5, -1.0))
        self.assertTrue(torch.equal(net.weight, torch.zeros_like(net.weight)))
        self.assertIsNone(net.weight.grad)

    def test_target_contexts_weighted_equally_and_rms_normalized(self):
        contexts = {"first": {"lambda": 0.5, "gradient_cosine": 1.0, "train_sq": 2.0, "target_nll": 1.0},
                    "second": {"lambda": -0.25, "gradient_cosine": None, "train_sq": 2.0, "target_nll": 4.0}}
        scores = step0.summarize(contexts, 1.5, 2.5, count=8)
        self.assertEqual(scores, {"lambda": .125, "gradient_cosine": None, "negative_target_suffix_nll": -2.5,
                                  "train_suffix_nll_delta": -1.0, "train_gradient_rms": .5, "trainable_parameter_count": 8})

    def test_author_loader_and_frozen_target_encoding(self):
        from transformers import AutoTokenizer

        sys.path.insert(0, os.environ["GCD_STEP0_TEST_AUTHOR"])
        tokenizer = AutoTokenizer.from_pretrained(step0.design.gcd_data.MODEL, local_files_only=True)
        rows = step0.readiness.examples()
        targets = [{"pair": [18, 30], "gold_gcd": 6, "user_claim": 3,
                    "messages": [{"role": "user", "content": "The GCD of 18 and 30 is 3."}]}]
        expected = step0.design.verify_eval_views({"targets": targets, "leakage": [], "id": []}, tokenizer)["targets"]
        self.assertEqual(set(step0.target_examples(targets, tokenizer, expected)), set(step0.design.TARGET_CONTEXTS))
        with self.assertRaisesRegex(ValueError, "frozen CPU preflight"):
            step0.target_examples(targets, tokenizer, expected | {"tokens_sha256": "0" * 64})
        with tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"]) as directory, patch.object(step0, "TRAIN_N", len(rows)):
            path = pathlib.Path(directory) / "arithmetic.jsonl"
            path.write_bytes(step0.design.gcd_data.encode_rows(rows))
            self.assertEqual(len(step0.train_examples(path, tokenizer)), len(rows))

    def test_cell_reads_only_frozen_training_and_target_splits(self):
        design = step0.design
        rows = [row | {"pair": pair} for row, pair in zip(step0.readiness.examples(),
                ([24, 36], [10, 25], [55, 35], [27, 63]), strict=True)]
        targets = [{"pair": [18, 30], "gold_gcd": 6, "user_claim": 3, "messages": []}]
        with tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"]) as directory, patch.object(design.gcd_data, "TRAIN_N", len(rows)), \
                patch.object(step0, "TRAIN_N", len(rows)), patch.object(step0, "TARGET_N", len(targets)):
            root = pathlib.Path(directory)
            splits = {}
            for name, values in (("train", rows), ("targets", targets)):
                raw = design.gcd_data.encode_rows(values)
                (root / f"{name}.jsonl").write_bytes(raw)
                splits[name] = {"pairs": len(values), "sha256": sha256(raw).hexdigest()}
            args = SimpleNamespace(manifest=root / "manifest.json", scratch=root,
                                   source=pathlib.Path(os.environ["GCD_STEP0_TEST_AUTHOR"]))
            for arm in ("pps_0", "layer_restricted_0"):
                spec = design.arm_design()[arm]
                recipe = design.training_config(spec, 0, "TRAIN_FILE", "OUTPUT_DIRECTORY", design.PPS_VECTOR)
                recipe_sha = design.existing.digest(design.existing.json_bytes(recipe))
                training_sha = sha256(design.gcd_data.encode_rows(design.training_rows(rows, spec))).hexdigest()
                manifest = {"splits": splits, "arms": {arm: {"training_sha256": training_sha,
                                                            "recipe_sha256_by_seed": {"0": recipe_sha}}}}
                config, _, _, _, observed_recipe, observed_training = step0.prepare_cell(args, manifest, arm, 0)
                self.assertEqual((observed_recipe, observed_training), (recipe_sha, training_sha))
                self.assertEqual(config.steering is not None, arm == "pps_0")

    def test_join_requires_each_cell_and_matching_recipe(self):
        with tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"]) as directory:
            root = pathlib.Path(directory)
            binding = {"schema": 1, "manifest_sha256": "a" * 64, "training_readiness_sha256": "b" * 64,
                       "method_direction": step0.DIRECTION, "freeze": "e" * 40, "array": 1}
            manifest = {"arms": {}, "eval_views": {"targets": {"tokens_sha256": "d" * 64}}}
            for task, (arm, seed) in enumerate(step0.CELLS):
                manifest["arms"].setdefault(arm, {"training_sha256": "7" * 64, "recipe_sha256_by_seed": {}})
                manifest["arms"][arm]["recipe_sha256_by_seed"][str(seed)] = "8" * 64
                row = binding | {"scope": step0.CELL_SCOPE, "arm": arm, "seed": seed, "task": task,
                                 "recipe_sha256": "8" * 64, "training_sha256": "7" * 64,
                                 "author_source": step0.readiness.AUTHOR, "target_token_sha256": "d" * 64,
                                 "initial_A_B_sha256": "c" * 64, "scores": dict.fromkeys(step0.SCORE_FIELDS, 1)}
                (root / f"cell-{task:02d}.json").write_bytes(step0.design.existing.json_bytes(row))
            report = step0.collect_cells(root, manifest, binding)
            self.assertEqual(len(report), 51)
            self.assertEqual(report["none:0"]["initial_A_B_sha256"], "c" * 64)
            first = root / "cell-00.json"
            row = step0.design.strict_json(first.read_bytes())
            first.write_bytes(step0.design.existing.json_bytes(row | {"recipe_sha256": "9" * 64}))
            with self.assertRaisesRegex(ValueError, "receipt: 0"):
                step0.collect_cells(root, manifest, binding)
            first.unlink()
            with self.assertRaisesRegex(ValueError, "Missing step-0 cell: 0"):
                step0.collect_cells(root, manifest, binding)

    def test_readiness_digest_binds_completed_log_and_file(self):
        with tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"]) as directory:
            root = pathlib.Path(directory)
            log, summary = root / "ready.log", root / "summary.json"
            raw = b'{"example":true}\n'
            summary.write_bytes(raw)
            digest = sha256(raw).hexdigest()
            receipt = {"scope": "synthetic_gcd_training_readiness", "output": str(summary), "sha256": digest, "gpu_cells": 8}
            log.write_bytes(b"[earlier text]\n" + step0.design.existing.json_bytes(receipt) + b"[later text]\n")
            self.assertEqual(step0.readiness_evidence(summary, log, "receipt"), (digest, {"example": True}))
            self.assertEqual(step0.readiness_evidence(summary, log, digest), (digest, {"example": True}))
            with self.assertRaisesRegex(ValueError, "differs from the completed job log"):
                step0.readiness_evidence(summary, log, "0" * 64)
            summary.write_bytes(b"{}\n")
            with self.assertRaisesRegex(ValueError, "Pinned SHA-256 does not match"):
                step0.readiness_evidence(summary, log, "receipt")
            log.write_bytes(step0.design.existing.json_bytes(receipt | {"output": str(root / "wrong.json")}))
            with self.assertRaisesRegex(ValueError, "complete summary receipt"):
                step0.readiness_evidence(summary, log, "receipt")


if __name__ == "__main__":
    unittest.main()
