"""[codex] CPU producer/integrity tests against synthetic local files only."""
from copy import deepcopy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

try:
    from . import runner as R
except ImportError:
    import runner as R


class RunnerTests(unittest.TestCase):
    def setUp(self):
        base = R.ROOT/"hf_home/tmp"
        base.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="matrix-test-", dir=base)
        self.out = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def artifact(self, name, value, *, jsonl=False):
        path = self.out/name
        path.write_text("".join(json.dumps(row)+"\n" for row in value) if jsonl else json.dumps(value))
        return {"path": str(path), "sha256": R.sha(path)}

    def panel(self):
        image = self.out/"image.jpg"
        image.write_bytes(b"synthetic-image-bytes")
        gallery = [{"id": str(i), "path": str(image), "sha256": R.sha(image), "bytes": image.stat().st_size,
                    "width": 7, "height": 9, "custom_metadata": "preserve"} for i in (1, 2)]
        rows = [{"id": "q1", "text": "first", "group": "g1", "source_image_id": "1", "caption_index": 0,
                 "positive_ids": ["1"], "ignore": []},
                {"id": "q2", "text": "", "group": "g2", "source_image_id": "2", "caption_index": 0,
                 "positive_ids": ["2"], "model_eligible": False, "exclusion_reason": "source-empty-caption"}]
        return {"format": "negbench", "split": "public", "panel": self.artifact("panel.json", {"candidates": gallery, "queries": rows})}

    def test_inference_queries_strip_labels_but_keep_gallery_metadata(self):
        spec = self.panel()
        data = R.load_data(spec, verify_images=True)
        self.assertEqual(set(data["queries"][0]), {"id", "text", "group"})
        self.assertEqual(data["gallery"][0]["custom_metadata"], "preserve")
        self.assertEqual(len(data["excluded"]), 1)
        self.assertEqual(R.load_labels(spec, data)[0]["positives"], ["1"])

    def test_official_csv_order_and_empty_caption_are_preserved(self):
        spec = self.panel()
        path = self.out/"official.csv"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["image_id", "captions"])
            writer.writeheader()
            writer.writerows([{"image_id": "2", "captions": "['']"}, {"image_id": "1", "captions": "['first']"}])
        spec.update(official_evaluation="negbench-t2i", official_csv={"path": str(path), "sha256": R.sha(path)},
                    empty_caption_policy="retain-official", expected_eligible=2)
        data = R.load_data(spec)
        self.assertEqual([row["id"] for row in data["gallery"]], ["2", "1"])
        self.assertEqual([row["text"] for row in data["queries"]], ["", "first"])
        self.assertEqual(data["retained_empty"], ["q2"])

    def test_conjunction_receipt_subset_and_label_separation(self):
        image_dir = self.out/"images"
        image_dir.mkdir()
        image = image_dir/"a.jpg"
        image.write_bytes(b"fake-jpeg")
        spec = {"format": "conjunction", "split": "dev-small", "images_dir": str(image_dir),
                "gallery": self.artifact("gallery.json", [{"id": 7, "file_name": "a.jpg", "width": 100}]),
                "queries": self.artifact("queries.jsonl", [{"id": "a", "text": "a", "positives": [7], "ignore": []},
                                                           {"id": "b", "text": "b", "positives": [7], "ignore": []}], jsonl=True),
                "images_receipt": self.artifact("images.jsonl", [{"id": 7, "sha256": R.sha(image), "bytes": 9}], jsonl=True),
                "subset": self.artifact("subset.json", ["b"]), "expected_queries": 1}
        data = R.load_data(spec, verify_images=True)
        self.assertEqual([row["id"] for row in data["queries"]], ["b"])
        self.assertNotIn("positives", data["queries"][0])
        self.assertEqual(data["gallery"][0]["width"], 100)

    def test_flickr_original_csv_manifest_and_images_are_jointly_verified(self):
        image_dir = self.out/"images_flickr_1k_test"
        image_dir.mkdir()
        image = image_dir/"abc.jpg"
        image.write_bytes(b"fake-jpeg")
        path = self.out/"official.csv"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["filename", "raw", "imgid"])
            writer.writeheader()
            writer.writerow({"filename": "abc.jpg", "raw": json.dumps(["caption1", "caption2"]), "imgid": json.dumps(["abc", "abc"])})
        spec = {"format": "flickr", "split": "public", "official_csv": {"path": str(path), "sha256": R.sha(path)},
                "manifest": self.artifact("manifest.jsonl", [{"id": "abc", "image": "images_flickr_1k_test/abc.jpg", "captions": ["caption1", "caption2"]}], jsonl=True),
                "images_receipt": self.artifact("receipt.jsonl", [{"path": "flickr30k_test/images_flickr_1k_test/abc.jpg", "bytes": 9, "sha256": R.sha(image)}], jsonl=True)}
        data = R.load_data(spec, verify_images=True)
        self.assertEqual(len(data["queries"]), 2)
        self.assertEqual(R.load_labels(spec, data)[0]["positives"], ["abc"])
        self.assertEqual(data["gallery"][0]["captions"], ["caption1", "caption2"])

    def test_dataset_hash_count_and_image_corruption_are_rejected(self):
        spec = self.panel()
        with self.assertRaises(ValueError):
            R.load_data({**spec, "expected_gallery": 8})
        Path(R.read(spec["panel"]["path"])["candidates"][0]["path"]).write_bytes(b"changed")
        with self.assertRaises(ValueError):
            R.load_data(spec, verify_images=True)
        Path(spec["panel"]["path"]).write_text("{}")
        with self.assertRaises(ValueError):
            R.load_data(spec)

    def test_test_data_requires_frozen_baseline_authorization(self):
        spec = self.panel()
        with self.assertRaisesRegex(ValueError, "Test data"):
            R.load_data({**spec, "split": "test"})

    def test_label_poison_cannot_change_model_inputs_or_admission(self):
        spec = self.panel()
        first = R.load_data(spec)
        body = R.read(spec["panel"]["path"])
        body["queries"][0].update(positive_ids=["secret"], ignore=["bad"], score=1e100)
        spec["panel"] = self.artifact("panel.json", body)
        second = R.load_data(spec)
        self.assertEqual(first["queries"], second["queries"])
        self.assertEqual(first["gallery"], second["gallery"])
        a = R.rank_blocks(np.eye(2), np.array([[1., 0.]]))
        b = R.rank_blocks(np.eye(2), np.array([[1., 0.]]), block_queries=1)
        np.testing.assert_array_equal(a, b)

    def test_checkpoint_resumes_original_batches_and_rejects_changes(self):
        path = self.out/"array.npy"
        calls = []
        def compute(items):
            calls.append(items)
            if items == [2, 3] and len(calls) == 2:
                raise RuntimeError("simulated interruption")
            return np.array(items, dtype=np.float32)[:, None]
        with self.assertRaises(RuntimeError):
            R.checkpoint(path, list(range(5)), 2, compute, {"test": 1})
        values = R.checkpoint(path, list(range(5)), 2, compute, {"test": 1})
        self.assertEqual(calls, [[0, 1], [2, 3], [2, 3], [4]])
        self.assertEqual(values[:, 0].tolist(), list(range(5)))
        with self.assertRaises(ValueError):
            R.checkpoint(path, list(range(5)), 1, compute, {"test": 1})
        self.assertEqual(R.array_receipt(path)["shape"], [5, 1])

    def test_source_gate_rejects_unpinned_commit_before_any_model_work(self):
        with self.assertRaisesRegex(ValueError, "full --freeze-commit"):
            R.source_gate(self.out/"plan.json", {}, "HEAD")

    def test_official_model_gate_rejects_unverified_and_quantized_protocols(self):
        fixture = {"kind": "official_embedding", "revision": "a"*40, "repo": "test/model", "snapshot": "/dummy/"+"a"*40,
                   "official": {"verified": False, "wrapper": "wrapper.py", "model_class": "Model", "processor_class": "Processor",
                                "embedding": "embedding.py", "model_name": "vlembed"}, "batches": {"image": 1, "text": 1, "cross": 1}}
        with self.assertRaisesRegex(ValueError, "verification"):
            R.models.validate_spec(fixture)
        fixture["official"]["verified"] = True
        fixture["loader"] = {"load_in_4bit": True}
        with self.assertRaisesRegex(ValueError, "quantization"):
            R.models.validate_spec(fixture)

    def test_batch_invariance_gate_enforces_both_frozen_tolerances(self):
        settings = {"batch_invariance_cosine_min": .9995, "batch_invariance_max_abs": .01}
        passed = {name: {"cosine": .9999, "max_abs_difference": .001} for name in ("image", "text")}
        self.assertTrue(R.invariance_passed(passed, settings))
        for field, value in (("cosine", .99), ("max_abs_difference", .02), ("cosine", np.nan)):
            failed = deepcopy(passed)
            failed["image"][field] = value
            self.assertFalse(R.invariance_passed(failed, settings))
        self.assertFalse(R.invariance_passed({}, settings))


if __name__ == "__main__":
    unittest.main()
