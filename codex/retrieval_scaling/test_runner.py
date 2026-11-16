"""[codex] CPU-only integrity tests; no real data, downloads, GPU, or metrics."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("scaling_runner_tested", HERE / "runner.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)


def plan_fixture():
    return {"schema": 1, "source": {"frozen_commit": R.FROZEN_COMMIT, "model_path": R.MODEL_PATH,
        "config_path": R.CONFIG_PATH, "config_sha256": R.CONFIG_SHA},
        "data": {"baseline_dir": "/synthetic/baseline", "images_dir": "/synthetic/images",
            "images_manifest_path": "/synthetic/manifest.json", "panel_sha256": "a" * 64,
            "vectors_sha256": "b" * 64, "cross_sha256": "c" * 64, "images_manifest_sha256": "d" * 64,
            "baseline_receipt_sha256": "e" * 64},
        "models": {name: {"repo": repo, "revision": revision, "snapshot": f"/synthetic/{revision}",
                          "files": {"config.json": "e" * 64, "model.safetensors": "f" * 64}}
                   for name, (repo, revision) in R.PINS.items()},
        "settings": {"top_k": 50, "embed_image_batch": 8, "embed_text_batch": 32,
                     "cross_batch": 4, "seed": 2026111665, "precision": "bfloat16"},
        "environment": {}, "reuse": {"embed2": True, "cross22": True}}


class RunnerTests(unittest.TestCase):
    def setUp(self):
        # Repository coordination puts temporary files under the task runtime.
        base = R.ROOT / "dl/retrieval3090/tmp"
        base.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="scaling-unit-", dir=base)
        self.directory = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_frozen_plan_accepts_exact_original_batches(self):
        plan = plan_fixture()
        self.assertIs(R.validate_plan(plan), plan)

    def test_changed_cross_batch_requires_new_2b_control(self):
        plan = plan_fixture()
        plan["settings"]["cross_batch"] = 1
        with self.assertRaisesRegex(ValueError, "fresh 2B control scores"):
            R.validate_plan(plan)
        plan["reuse"]["cross22"] = False
        R.validate_plan(plan)

    def test_changed_encoder_batch_requires_new_vectors_and_scores(self):
        plan = plan_fixture()
        plan["settings"]["embed_image_batch"] = 1
        with self.assertRaisesRegex(ValueError, "fresh 2B control vectors"):
            R.validate_plan(plan)
        plan["reuse"]["embed2"] = False
        with self.assertRaisesRegex(ValueError, "fresh 2B control scores"):
            R.validate_plan(plan)
        plan["reuse"]["cross22"] = False
        R.validate_plan(plan)

    def test_precision_model_pin_and_topk_cannot_silently_change(self):
        for location, key, value in (("settings", "precision", "float16"), ("settings", "top_k", 200),
                                     ("settings", "cross_batch", True), ("settings", "dtype", "float16"),
                                     ("settings", "quantization", True), ("settings", "cpu_offload", True),
                                     ("settings", "attention", "other")):
            plan = plan_fixture()
            plan[location][key] = value
            with self.assertRaises(ValueError):
                R.validate_plan(plan)
        plan = plan_fixture()
        plan["models"]["embed8"]["revision"] = "0" * 40
        with self.assertRaisesRegex(ValueError, "Pinned official model differs"):
            R.validate_plan(plan)

    def test_sealed_path_and_model_file_escape_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Sealed"):
            R.safe(self.directory / "retrieval-independent-panel" / "not-opened")
        plan = plan_fixture()
        plan["models"]["embed8"]["files"]["../secret"] = "a" * 64
        with self.assertRaisesRegex(ValueError, "Malformed model-file"):
            R.validate_plan(plan)

    def test_source_gate_requires_explicit_commit_before_git(self):
        with patch.object(R, "git_bytes", side_effect=AssertionError("Must reject before git")):
            with self.assertRaisesRegex(ValueError, "explicit full"):
                R.source_gate(self.directory / "plan.json", plan_fixture(), "HEAD")

    def test_json_rejects_duplicate_keys_and_nonfinite(self):
        path = self.directory / "bad.json"
        for body in ('{"a": 1, "a": 2}', '{"a": NaN}', '{"a": Infinity}'):
            path.write_text(body)
            with self.assertRaises(ValueError):
                R.read_json(path)

    def test_cache_reuse_proof_checks_hashes_precision_configuration_and_runtime(self):
        plan = plan_fixture()
        baseline = self.directory / "baseline"
        baseline.mkdir()
        for name in ("panel.json", "vectors.npz", "cross.npz"):
            (baseline / name).write_bytes(b"synthetic bytes; never loaded as arrays")
        config = {"batch_size": {"embed_image": 8, "embed_text": 32, "cross_pair": 4}}
        runtime = {"device": "NVIDIA GeForce RTX 3090", "torch": "2.14.0+cu126"}
        meta = {"config": config, "model_config_sha256": R.CONFIG_SHA, "bf16": True,
            "depth": 200, "top_k": 50, "execution_environment": runtime,
            "revisions": {"core_embed": R.PINS["embed2"][1], "core_reranker": R.PINS["cross2"][1]}}
        R.atomic_json(baseline / "meta.json", meta)
        def receipt():
            body = {"platform_consistency": "VALID_LOCAL", "formal_evidence": False,
                "outputs_sha256": {name: R.sha(baseline / name) for name in ("panel.json", "vectors.npz", "cross.npz", "meta.json")},
                "environment": runtime, "checks": {key: True for key in ("complete_depth200", "vector_shapes_finite_unit",
                "core_order_recomputed", "top50_source_identity", "baseline_recomputed", "fixed_sample_repeat")}}
            R.atomic_json(baseline / "platform-receipt.json", body)
            plan["data"]["baseline_receipt_sha256"] = R.sha(baseline / "platform-receipt.json")
        receipt()
        plan["environment"] = {"torch": "2.14.0+cu126"}
        with patch.object(R, "frozen_config", return_value=config):
            R.validate_baseline_receipt(plan, baseline)
            (baseline / "vectors.npz").write_bytes(b"changed input")
            with self.assertRaisesRegex(ValueError, "output hash differs"):
                R.validate_baseline_receipt(plan, baseline)
            receipt()
            meta["bf16"] = False
            R.atomic_json(baseline / "meta.json", meta)
            receipt()
            with self.assertRaisesRegex(ValueError, "precision"):
                R.validate_baseline_receipt(plan, baseline)
            meta["bf16"] = True
            R.atomic_json(baseline / "meta.json", meta)
            receipt()
            plan["environment"]["torch"] = "different"
            with self.assertRaisesRegex(ValueError, "runtime differs from the scaling plan"):
                R.validate_baseline_receipt(plan, baseline)

    def test_stable_full_gallery_order_uses_original_column_for_ties(self):
        images = np.asarray([[1., 0.], [0., 1.], [1., 0.]])
        queries = np.asarray([[1., 0.], [0., 1.]])
        np.testing.assert_array_equal(R.ranked(images, queries), [[0, 2, 1], [1, 0, 2]])

    def test_unit_vectors_reject_zero_nonfinite_and_dimension_mismatch(self):
        good = np.eye(2, dtype=np.float32)
        self.assertIs(R.unit_vectors(good, 2, 2), good)
        for bad, dim in ((np.zeros((2, 2)), 2), (np.asarray([[np.nan, 1], [0, 1]]), 2), (good, 3)):
            with self.assertRaises(ValueError):
                R.unit_vectors(bad, 2, dim)

    def test_interrupted_batch_retries_with_identical_composition(self):
        path = self.directory / "resume.npy"
        items, batches = list(range(10)), []
        def first(rows):
            batches.append(rows)
            if rows[0] == 4:
                raise RuntimeError("simulated interruption")
            return np.asarray(rows, dtype=np.float64)[:, None]
        with self.assertRaisesRegex(RuntimeError, "interruption"):
            R.checkpoint(path, items, 4, first, {"model": "fixed"})
        self.assertEqual(R.read_json(path.with_suffix(".state.json"))["completed"], 4)
        resumed = []
        result = R.checkpoint(path, items, 4, lambda rows: resumed.append(rows) or np.asarray(rows, dtype=np.float64)[:, None],
                              {"model": "fixed"})
        self.assertEqual(resumed, [[4, 5, 6, 7], [8, 9]])
        np.testing.assert_array_equal(result[:, 0], items)
        self.assertEqual(R.read_json(path.with_suffix(".state.json"))["sha256"], R.sha(path))

    def test_completed_checkpoint_is_reused_without_compute(self):
        path = self.directory / "done.npy"
        R.checkpoint(path, [0, 1, 2], 2, lambda rows: np.asarray(rows, dtype=np.int32), {"model": "fixed"})
        result = R.checkpoint(path, [0, 1, 2], 2, lambda rows: self.fail("Unexpected recomputation"), {"model": "fixed"})
        np.testing.assert_array_equal(result, [0, 1, 2])

    def test_resume_rejects_batch_source_order_and_committed_data_corruption(self):
        path = self.directory / "frozen.npy"
        R.checkpoint(path, [0, 1, 2, 3], 2, lambda rows: np.asarray(rows, dtype=np.float32), {"source": "one"})
        for items, batch, identity in (([0, 1, 2, 3], 1, {"source": "one"}),
                                       ([1, 0, 2, 3], 2, {"source": "one"}),
                                       ([0, 1, 2, 3], 2, {"source": "two"})):
            with self.assertRaisesRegex(ValueError, "identity differs"):
                R.checkpoint(path, items, batch, lambda rows: self.fail("Unexpected computation"), identity)
        corrupt = np.load(path, mmap_mode="r+")
        corrupt[1] = 99
        corrupt.flush()
        with self.assertRaisesRegex(ValueError, "corrupt"):
            R.checkpoint(path, [0, 1, 2, 3], 2, lambda rows: self.fail("Unexpected computation"), {"source": "one"})

    def test_nonfinite_forward_cannot_commit_progress(self):
        path = self.directory / "nonfinite.npy"
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            R.checkpoint(path, [0, 1], 2, lambda rows: np.asarray([0., np.nan]), {"source": "one"})
        self.assertFalse(path.with_suffix(".state.json").exists())

    def test_final_array_cannot_be_overwritten_with_different_values(self):
        path = self.directory / "raw.npy"
        R.save_array(path, np.asarray([1, 2], dtype=np.float32))
        R.save_array(path, np.asarray([1, 2], dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "differs"):
            R.save_array(path, np.asarray([1, 3], dtype=np.float32))

    def test_index_orders_and_cross_arms_have_expected_file_names(self):
        self.assertEqual(len(R.FINAL_ARRAYS), 10)
        self.assertIn("index/8b_order.npy", R.FINAL_ARRAYS)
        self.assertEqual(set(name for name in R.FINAL_ARRAYS if name.startswith("cross/")),
                         {"cross/e2_r2.npy", "cross/e2_r8.npy", "cross/e8_r2.npy", "cross/e8_r8.npy"})

    def test_partial_or_wrong_identity_smoke_cannot_authorize_main(self):
        producer = object.__new__(R.Producer)
        producer.out, producer.identity = self.directory, {"source": "fixed"}
        for smoke in ({"state": "RUNNING", "identity": producer.identity},
                      {"state": "PASS", "identity": {"source": "other"}}):
            R.atomic_json(self.directory / "smoke.json", smoke)
            with self.assertRaisesRegex(ValueError, "passing smoke"):
                producer.require_smoke()
        R.atomic_json(self.directory / "smoke.json", {"state": "PASS", "identity": producer.identity})
        producer.require_smoke()


if __name__ == "__main__":
    unittest.main()
