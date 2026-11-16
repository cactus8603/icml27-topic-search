"""[codex] CPU synthetic checks for original data and native admission evidence."""
from __future__ import annotations

import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from . import data, metrics, qualification as Q
from .evidence import ROOT, digest, entry, read, sha, write

METRIC_SOURCE = ROOT/"dl/retrieval-matrix-native-p0/sources/image_text_retrieval.py"


class NativeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT/"hf_home/tmp", prefix="native-p0-test-")
        self.root = Path(self.tmp.name)
        # These synthetic arrays never came from GPU production or a committed
        # plan. Isolate only the Git freeze boundary; all source/data/array,
        # runtime, smoke and checkpoint checks remain real CPU validation.
        self.real_frozen_gate = Q.frozen_source_gate
        self.freeze_patch = patch.object(Q, "frozen_source_gate")
        self.freeze_patch.start()

    def tearDown(self):
        self.freeze_patch.stop()
        self.tmp.cleanup()

    def fixture(self, name="coco", *, flickr=False):
        folder = self.root/name
        folder.mkdir()
        original, prepared, receipts = [], [], []
        for index, count in enumerate((6, 5)):
            image_id = str(10+index)
            filename = image_id+".jpg" if flickr else "COCO_val2014_"+image_id.zfill(12)+".jpg"
            path = folder/filename
            path.write_bytes(("synthetic-image-"+name+image_id).encode())
            sentences = [{"raw": name+" raw Caption "+str(index)+" "+str(k), "sentid": index*10+k}
                         for k in range(count)]
            row = {"cocoid": int(image_id), "filename": filename, "split": "test", "filepath": "val2014",
                "imgid": index, "sentences": sentences, "sentids": [v["sentid"] for v in sentences]}
            original.append(row)
            prepared.append({"id": image_id, "captions": [v["raw"] for v in sentences],
                "path": str(path), "filename": filename, "image": filename, "sentids": row["sentids"]})
            receipts.append({"id": image_id, "path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)})
        source = folder/"source.json"
        write(source, {"images": original})
        manifest = folder/"manifest.jsonl"
        manifest.write_text("".join(json.dumps(row)+"\n" for row in prepared))
        receipt = folder/"images.jsonl"
        receipt.write_text("".join(json.dumps(row)+"\n" for row in receipts))
        return {"format": "karpathy-flickr" if flickr else "karpathy-coco", "benchmark_id": name,
            "source_json": entry(source), "manifest": entry(manifest), "image_receipt": entry(receipt),
            "image_root": str(folder), "receipt_root": str(folder), "expected_images": 2, "expected_texts": 11}

    def full_evidence(self, *, target_coco=1.):
        spec = {"family": "CLIP", "variant": "synthetic-fixed-variant", "embedding_dim": 2,
            "batches": {"image": 1, "text": 4}, "loader": {"torch_dtype": "float32"}}
        paper = self.root/"paper.txt"
        paper.write_text("Synthetic published-target stand-in; CPU tests only.")
        plan = {"schema": 2, "purpose": "P0_OFFICIAL_QUALIFICATION", "phase": "P0_NATIVE",
            "model_key": "synthetic", "model_spec": spec, "qualification": {"required_distinct_benchmarks": 2},
            "settings": {"image_batch": 1, "text_batch": 4, "seed": 2026, "allow_tf32": False,
                "precision": "float32", "quantization": False, "offload": False,
                "smoke_cosine_min": .9995, "smoke_max_abs": .01},
            "dependencies": entry(ROOT/"codex/retrieval_matrix_extra/dependencies.json"),
            "source_files": {"codex/retrieval_matrix_native_p0/metrics.py": sha(Path(metrics.__file__))},
            "external_sources": {"google_metric": entry(METRIC_SOURCE)}, "benchmarks": {}}
        for name in ("coco", "flickr"):
            ds = self.fixture(name, flickr=name == "flickr")
            plan["benchmarks"][name] = {"canonical_benchmark": name, "benchmark_id": name,
                "task": "bidirectional-caption-retrieval", "data": ds, "score_profile": copy.deepcopy(metrics.PROFILE),
                "disclosed_differences": ["synthetic CPU test, no performance claim"],
                "primary_metrics": {direction+"_R@1": {"direction": direction, "metric": "any-positive Recall@1",
                    "denominator": 11 if direction == "t2i" else 2,
                    "target": target_coco if name == "coco" and direction == "t2i" else 1., "absolute_tolerance": .01,
                    "published": {"variant": spec["variant"], "table": "synthetic table", "url": "https://example.org/synthetic",
                                  "source": entry(paper)}} for direction in ("t2i", "i2t")}}
        plan_path = self.root/"plan.json"
        write(plan_path, plan)
        reports = {}
        for name, bench in plan["benchmarks"].items():
            bundle = data.load(bench["data"])
            images = np.eye(2, dtype=np.float32)
            texts = images[data.positive_indices(bundle)]
            image_path, text_path = self.root/(name+"-images.npy"), self.root/(name+"-texts.npy")
            base_identity = {
                "plan_sha256": sha(plan_path), "model": plan["model_key"], "model_spec_sha256": digest(spec),
                "source_files": plan["source_files"], "plan_file": entry(plan_path)}
            runtime = {"modules": read(plan["dependencies"]["path"])["modules"],
                "matmul": {"allow_tf32": False, "cudnn_allow_tf32": False, "float32_precision": "highest"},
                "python_executable": "synthetic-cpu-test", "gpu": {"assigned_uuid": "synthetic-no-real-GPU", "central_task": "synthetic"}}
            identity = {**base_identity, "benchmark": name, "benchmark_spec_sha256": digest(bench),
                "settings": plan["settings"], "data_identity": bundle["identity"],
                "freeze_commit": "f"*40, "runtime": runtime, "model_signature": {"synthetic": True}, "producer": "producer-agent"}
            smoke_path = self.root/(name+"-smoke.json")
            write(smoke_path, {"state": "PASS", "identity": base_identity, "runtime": runtime,
                "freeze_commit": identity["freeze_commit"], "model_signature": identity["model_signature"],
                "checks": {key: {"cosine": 1., "max_abs": 0., "passed": True} for key in ("images", "texts")}})
            from codex.retrieval_scaling import runner as producer_checkpoint
            image_items, text_items = data.inference_items(bundle)
            for key, path, values, items, batch in (("images", image_path, images, image_items, 1),
                                                   ("texts", text_path, texts, text_items, 4)):
                positions = {item: index for index, item in enumerate(items)}
                with redirect_stdout(io.StringIO()):
                    producer_checkpoint.checkpoint(path, items, batch,
                        lambda part: values[[positions[item] for item in part]], {**identity, "modality": key})
            index_receipt = self.root/(name+"-index-receipt.json")
            write(index_receipt, {"state": "COMPLETE", "identity": identity, "counts": bundle["counts"],
                "research_metrics": False, "arrays": {"images": entry(image_path), "texts": entry(text_path)},
                "checkpoints": {"images": entry(image_path.with_suffix(".state.json")), "texts": entry(text_path.with_suffix(".state.json"))},
                "smoke": entry(smoke_path)})
            report = {"state": "COMPLETE", "producer": "producer-agent", "identity": identity,
                "index_receipt": entry(index_receipt),
                "score_profile": bench["score_profile"], "primary_metric_descriptors": bench["primary_metrics"],
                "data_identity": bundle["identity"], "counts": bundle["counts"],
                "arrays": {"images": entry(image_path), "texts": entry(text_path)},
                "metrics": metrics.author_metrics(images, texts, data.positive_indices(bundle), entry(METRIC_SOURCE))}
            reports[name] = self.root/(name+"-report.json")
            write(reports[name], report)
        return plan_path, reports

    def test_full_original_sixth_caption_and_label_separation(self):
        bundle = data.load(self.fixture())
        self.assertEqual(bundle["counts"]["texts"], 11)
        self.assertEqual(bundle["counts"]["caption_count_histogram"], {"5": 1, "6": 1})
        images, texts = data.inference_items(bundle)
        self.assertTrue(all(isinstance(value, str) for value in images+texts))
        self.assertTrue(all(set(row) == {"id", "text"} for row in bundle["queries"]))
        self.assertIn("source_metadata", bundle["gallery"][0])

    def test_flickr_original_source(self):
        bundle = data.load(self.fixture(flickr=True))
        self.assertEqual(bundle["gallery"][0]["id"], "10")
        self.assertEqual(len(bundle["queries"]), 11)

    def test_prepared_caption_tamper_rejected_even_with_new_manifest_sha(self):
        spec = self.fixture()
        path = Path(spec["manifest"]["path"])
        records = data.rows(path)
        records[0]["captions"][0] = "Different sentence"
        path.write_text("".join(json.dumps(row)+"\n" for row in records))
        spec["manifest"] = entry(path)
        with self.assertRaisesRegex(ValueError, "raw captions"):
            data.load(spec)

    def test_incomplete_denominator_and_image_corruption_rejected(self):
        spec = self.fixture()
        spec["expected_texts"] = 10
        with self.assertRaisesRegex(ValueError, "denominator"):
            data.load(spec)
        spec["expected_texts"] = 11
        image = Path(data.rows(spec["image_receipt"]["path"])[0]["path"])
        image.write_bytes(b"changed")
        with self.assertRaises(ValueError):
            data.load(spec)

    def test_author_and_independent_precision_direction_and_ties(self):
        images = np.eye(2, dtype=np.float32)
        texts = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.float32)
        positives = [0, 1, 1, 1]
        expected = metrics.author_metrics(images, texts, positives, entry(METRIC_SOURCE))
        self.assertEqual(expected, metrics.independent_metrics(images, texts, positives))
        self.assertEqual(expected["t2i_R@1"], .75)
        tied = np.array([[1, 0], [1, 0]], dtype=np.float32)
        self.assertEqual(metrics.author_metrics(tied, texts, positives, entry(METRIC_SOURCE)),
                         metrics.independent_metrics(tied, texts, positives))
        with self.assertRaisesRegex(ValueError, "FP32"):
            metrics.independent_metrics(images.astype(np.float16), texts, positives)

    def test_independent_complete_receipt_and_admission(self):
        plan, reports = self.full_evidence()
        review = self.root/"review.json"
        write(review, Q.review(plan, reports, "independent-agent", accept_protocol=True))
        result = Q.make(plan, reports, review)
        self.assertEqual(result["state"], "PASS")
        receipt = self.root/"qualification.json"
        write(receipt, result)
        self.assertEqual(Q.admit(entry(receipt), read(plan)["model_spec"]), result)

    def test_producer_cannot_review_and_protocol_needs_explicit_acceptance(self):
        plan, reports = self.full_evidence()
        with self.assertRaisesRegex(ValueError, "own qualification"):
            Q.review(plan, reports, "producer-agent", accept_protocol=True)
        result = Q.review(plan, reports, "independent-agent")
        self.assertEqual(result["state"], "REVIEW_REQUIRED")
        review = self.root/"review.json"
        write(review, result)
        with self.assertRaises(ValueError):
            Q.make(plan, reports, review)

    def test_duplicate_canonical_dataset_and_wrong_direction_denominator(self):
        plan, _ = self.full_evidence()
        value = read(plan)
        value["benchmarks"]["flickr"]["canonical_benchmark"] = "coco"
        with self.assertRaisesRegex(ValueError, "canonical"):
            Q.validate_plan(value)
        value = read(plan)
        value["benchmarks"]["coco"]["primary_metrics"]["t2i_R@1"]["denominator"] = 2
        with self.assertRaisesRegex(ValueError, "denominator"):
            Q.validate_plan(value)

    def test_array_tamper_and_missing_benchmark_cannot_qualify(self):
        plan, reports = self.full_evidence()
        with self.assertRaisesRegex(ValueError, "every preregistered"):
            Q.review(plan, {"coco": reports["coco"]}, "independent-agent", accept_protocol=True)
        review = self.root/"review.json"
        write(review, Q.review(plan, reports, "independent-agent", accept_protocol=True))
        arrays = read(reports["coco"])["arrays"]
        Path(arrays["images"]["path"]).write_bytes(b"corrupt original array")
        with self.assertRaisesRegex(ValueError, "payload changed"):
            Q.make(plan, reports, review)

    def test_metric_failure_retained_and_p3_requires_two_families(self):
        plan, reports = self.full_evidence(target_coco=.5)
        review = self.root/"review.json"
        write(review, Q.review(plan, reports, "independent-agent", accept_protocol=True))
        result = Q.make(plan, reports, review)
        self.assertEqual(result["state"], "FAILED")
        self.assertIn("coco", result["benchmarks"])
        with self.assertRaises(ValueError):
            Q.require_p3_families([{"state": "PASS", "family": "MLLM"}]*2)
        self.assertTrue(Q.require_p3_families([{"state": "PASS", "family": "CLIP"}, {"state": "PASS", "family": "MLLM"}]))

    def test_checkpoint_corruption_even_after_rebinding_outer_receipts(self):
        plan, reports = self.full_evidence()
        report = read(reports["coco"])
        index_path = Path(report["index_receipt"]["path"])
        index = read(index_path)
        state_path = Path(index["checkpoints"]["texts"]["path"])
        state = read(state_path)
        state["batches"][0]["sha256"] = "0"*64
        state_path.write_text(json.dumps(state))
        index["checkpoints"]["texts"] = entry(state_path)
        index_path.write_text(json.dumps(index))
        report["index_receipt"] = entry(index_path)
        reports["coco"].write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "complete-batch provenance"):
            Q.review(plan, reports, "independent-agent", accept_protocol=True)

    def test_batch_runtime_and_freeze_mismatch_rejected(self):
        plan, reports = self.full_evidence()
        value = read(plan)
        value["settings"]["image_batch"] = 2
        with self.assertRaisesRegex(ValueError, "batches"):
            Q.validate_plan(value)
        report = read(reports["coco"])
        report["identity"]["runtime"]["matmul"]["allow_tf32"] = True
        with self.assertRaisesRegex(ValueError, "runtime"):
            Q.verify_report(read(plan), sha(plan), "coco", report)
        report = read(reports["coco"])
        with patch.object(Q.subprocess, "check_output", return_value=b"wrong committed source"):
            with self.assertRaisesRegex(ValueError, "freeze commit"):
                self.real_frozen_gate(read(plan), report["identity"])

    def test_reviewer_actual_numpy_path_version_and_distribution_rejected(self):
        plan, reports = self.full_evidence()
        for attribute, value in (("__file__", str(self.root/"different-numpy/__init__.py")),
                                  ("__version__", "0.0.synthetic")):
            with self.subTest(attribute=attribute), patch.object(np, attribute, value):
                with self.assertRaisesRegex(ValueError, "review NumPy"):
                    Q.review(plan, reports, "independent-agent", accept_protocol=True)
        with patch.object(Q, "distribution_version", return_value="0.0.synthetic"):
            with self.assertRaisesRegex(ValueError, "review NumPy"):
                Q.review(plan, reports, "independent-agent", accept_protocol=True)

    def test_qualification_rejects_changed_reviewer_numpy_receipt(self):
        plan, reports = self.full_evidence()
        review_path = self.root/"review.json"
        reviewed = Q.review(plan, reports, "independent-agent", accept_protocol=True)
        reviewed["recomputation_numpy"]["path"] = str(self.root/"different-numpy/__init__.py")
        write(review_path, reviewed)
        with self.assertRaisesRegex(ValueError, "actual NumPy"):
            Q.make(plan, reports, review_path)


if __name__ == "__main__":
    unittest.main()
