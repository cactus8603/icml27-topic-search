"""[codex] Synthetic CPU checks; no benchmark model inference or qualification claim."""
import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from codex.retrieval_matrix_native_p0.evidence import ROOT, digest, entry, read, sha
from codex.retrieval_matrix_paired_p0 import data, metrics, qualification as Q


class PairedTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT/"hf_home/tmp", prefix="paired-p0-cpu-test-")
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def fixture(self):
        image = self.root/"000000000001.jpg"
        image.write_bytes(b"synthetic CPU image bytes; no inference")
        receipt = self.root/"images.jsonl"
        receipt.write_text(json.dumps({"filename": image.name, "path": str(image), "bytes": image.stat().st_size, "sha256": sha(image)})+"\n")
        source = self.root/"add_obj.json"
        source.write_text(json.dumps({"5": {"filename": image.name, "caption": "Raw CAPTION, punctuation! " + "word "*40,
                                           "negative_caption": "Raw CAPTION, punctuation! " + "word "*40}}))
        return {"format": "sugarcrepe-official", "benchmark_id": "SugarCrepe", "subsets": {"add_obj": entry(source)},
            "category_order": ["add_obj"], "expected_subsets": {"add_obj": 1}, "image_receipt": entry(receipt),
            "expected_images": 1, "expected_texts": 2}

    def test_raw_caption_and_identical_pair_retained_without_labels_in_inference(self):
        bundle = data.load(self.fixture())
        images, texts = data.inference_items(bundle)
        self.assertEqual(texts[0], texts[1])
        self.assertIn("CAPTION, punctuation!", texts[0])
        self.assertGreater(len(texts[0].split()), 30)
        self.assertTrue(all(isinstance(value, str) for value in images+texts))
        self.assertEqual(set(bundle["queries"][0]), {"id", "text"})
        self.assertEqual(bundle["counts"]["excluded_texts"], 0)
        self.assertIn("source_metadata", bundle["gallery"][0])

    def test_no_missing_or_subset_denominator_drop(self):
        spec = self.fixture()
        spec["expected_texts"] = 1
        with self.assertRaisesRegex(ValueError, "denominator"):
            data.load(spec)
        spec["expected_texts"] = 2
        Path(data.load(spec)["gallery"][0]["path"]).write_bytes(b"changed")
        with self.assertRaises(ValueError):
            data.load(spec)

    def test_strict_ties_are_incorrect_for_both_tasks(self):
        labels = [{"benchmark": "sugarcrepe", "category": "add_obj", "text_start": 0, "text_options": 2}]
        self.assertFalse(metrics.events(np.zeros((1, 6), dtype=np.float32), labels)[0, 0])
        labels[0].update(benchmark="sugarcrepe_pp", text_options=3)
        self.assertFalse(metrics.events(np.ones((1, 6), dtype=np.float32), labels).any())

    def test_macro_averaging_is_not_pooled(self):
        labels = [{"benchmark": "sugarcrepe", "category": "add_obj"}] + [{"benchmark": "sugarcrepe", "category": "add_att"}]*3
        scores = np.zeros((4, 6), dtype=np.float32)
        scores[0, 0], scores[1:, 1] = 1, 1
        result = metrics.from_scores(scores, labels)
        self.assertEqual(result["add_ITT"], .5)
        self.assertNotEqual(result["add_ITT"], .25)

    def test_separate_scalar_and_independent_paths_match(self):
        images = np.array([[1., 0.], [1., 0.]], dtype=np.float32)
        texts = np.array([[1., 0.], [.8, .6], [-1., 0.], [1., 0.], [1., 0.], [1., 0.]], dtype=np.float32)
        labels = [{"benchmark": "sugarcrepe_pp", "category": "replace_obj", "text_start": 0, "text_options": 3},
                  {"benchmark": "sugarcrepe_pp", "category": "replace_obj", "text_start": 3, "text_options": 3}]
        scores = metrics.score_rows(images, texts, labels)
        self.assertEqual(metrics.from_scores(scores, labels), metrics.independent_metrics(images, texts, labels))
        self.assertEqual(metrics.from_scores(scores, labels)["replace_ITT"], .5)
        self.assertEqual(metrics.from_scores(scores, labels)["replace_TOT"], .5)
        with self.assertRaisesRegex(ValueError, "FP32"):
            metrics.validate_embeddings(images.astype(np.float16), texts, labels)

    def test_private_facade_does_not_patch_frozen_native_modules(self):
        from codex.retrieval_matrix_native_p0 import qualification as native_q, runner as native_r
        from codex.retrieval_matrix_paired_p0 import runner as paired_r
        self.assertIsNot(Q.ENGINE, native_q)
        self.assertIsNot(Q.ENGINE.data, native_q.data)
        self.assertIsNot(paired_r.R, native_r)
        self.assertEqual(native_r.models.Siglip2.__name__, "Siglip2")

    def test_complete_official_plan_and_no_subset_qualification(self):
        from codex.retrieval_matrix_paired_p0 import build_plan
        plan = build_plan.build("clic_b32")
        Q.validate_plan(plan)
        self.assertEqual(plan["benchmarks"]["sugarcrepe"]["data"]["expected_images"], 7511)
        self.assertEqual(plan["benchmarks"]["sugarcrepe_pp"]["data"]["expected_images"], 4757)
        broken = copy.deepcopy(plan)
        broken["benchmarks"]["sugarcrepe_pp"]["canonical_benchmark"] = "SugarCrepe"
        with self.assertRaises(ValueError):
            Q.validate_plan(broken)
        broken = copy.deepcopy(plan)
        broken["benchmarks"]["sugarcrepe"]["data"]["expected_images"] = 100
        with self.assertRaises(ValueError):
            Q.validate_plan(broken)
        broken = copy.deepcopy(plan)
        broken["benchmarks"]["sugarcrepe"]["primary_metrics"]["add_ITT"]["target"] = 1.
        with self.assertRaisesRegex(ValueError, "published"):
            Q.validate_plan(broken)
        for field, value in (("direction", "TOT"), ("denominator", 1),
                             ("aggregation", "pooled accuracy"), ("subsets", ["add_obj"])):
            broken = copy.deepcopy(plan)
            broken["benchmarks"]["sugarcrepe"]["primary_metrics"]["add_ITT"][field] = value
            with self.assertRaisesRegex(ValueError, "published"):
                Q.validate_plan(broken)
        broken = copy.deepcopy(plan)
        broken["source_files"].pop("codex/retrieval_matrix_paired_p0/models.py")
        with self.assertRaisesRegex(ValueError, "source evidence"):
            Q.validate_plan(broken)

    def test_exact_author_clips_tokenizer_branch_cpu(self):
        from codex.retrieval_matrix_paired_p0 import build_plan, models
        plan = build_plan.build("clic_clips_l14")
        model, _, tokenizer = models.parts(plan["model_spec"], weights=False)
        values = tokenizer(["A synthetic color image.", ""])
        self.assertEqual(type(tokenizer).__name__, "CLIPS_Tokenizer")
        self.assertEqual(tuple(values.shape), (2, 80))
        self.assertTrue((values[:, 0] == 1).all())
        self.assertTrue((values[:, -1] == 101).all())
        self.assertEqual(values[1, 1].item(), 2)
        self.assertEqual(type(model).__module__, "open_clip.model")

    def test_complete_synthetic_receipt_review_and_failed_target_retention(self):
        """Test the real paired evidence chain, without faking GPU provenance.

        Tiny fixtures cannot satisfy official 7511/4757 counts or a Git GPU
        freeze. Isolate exactly those two admission boundaries; checkpoints,
        original source roles, raw arrays, scalar arrays, runtime matching,
        independent events, review identity, and target failure all stay real.
        """
        from codex.retrieval_scaling import runner as C
        paper = self.root/"synthetic-target.txt"
        paper.write_text("Synthetic unit fixture only; not a reported benchmark score")
        spec = {"family": "CLIP", "variant": "synthetic", "embedding_dim": 2,
            "loader": {"torch_dtype": "float32"}, "batches": {"image": 1, "text": 1, "cross": 1}}
        settings = {"image_batch": 1, "text_batch": 1, "seed": 1, "precision": "float32", "allow_tf32": False,
            "quantization": False, "offload": False, "smoke_cosine_min": .9995, "smoke_max_abs": .01}
        plan = {"schema": 2, "phase": "P0_NATIVE", "purpose": "P0_OFFICIAL_QUALIFICATION", "model_key": "synthetic",
            "model_spec": spec, "settings": settings, "qualification": {"required_distinct_benchmarks": 2},
            "dependencies": entry(ROOT/"codex/retrieval_matrix_extra/dependencies.json"),
            "source_files": {"codex/retrieval_matrix_paired_p0/metrics.py": sha(Path(metrics.__file__))},
            "external_sources": {}, "benchmarks": {}}
        for name, pp in (("sugarcrepe", False), ("sugarcrepe_pp", True)):
            folder = self.root/name
            folder.mkdir()
            image = folder/"000000000001.jpg"
            image.write_bytes(b"synthetic image; no inference")
            receipt = folder/"images.jsonl"
            receipt.write_text(json.dumps({"filename": image.name, "path": str(image), "bytes": image.stat().st_size, "sha256": sha(image)})+"\n")
            category = "swap_obj" if pp else "add_obj"
            row = {"filename": image.name, "caption": "same raw caption", "negative_caption": "same raw caption"}
            if pp:
                row.update(caption2="same raw caption")
            source = folder/"source.json"
            source.write_text(json.dumps([row] if pp else {"0": row}))
            ds = {"format": "sugarcrepe-pp-official" if pp else "sugarcrepe-official", "benchmark_id": name,
                "subsets": {category: entry(source)}, "category_order": [category], "expected_subsets": {category: 1},
                "image_receipt": entry(receipt), "expected_images": 1, "expected_texts": 3 if pp else 2}
            plan["benchmarks"][name] = {"benchmark_id": name, "canonical_benchmark": name, "task": "paired-caption-comparison",
                "data": ds, "score_profile": metrics.PROFILE, "disclosed_differences": ["synthetic-only unit fixture"],
                "primary_metrics": {category.split("_")[0]+"_ITT": {"direction": "ITT", "metric": "strict pair accuracy",
                    "denominator": 1, "target": 1., "absolute_tolerance": .01,
                    "published": {"variant": "synthetic", "table": "synthetic unit fixture", "url": "https://example.org/synthetic", "source": entry(paper)}}}}
        plan_path = self.root/"plan.json"
        plan_path.write_text(json.dumps(plan))
        reports = {}
        for name, bench in plan["benchmarks"].items():
            folder = self.root/name
            bundle = data.load(bench["data"])
            images = np.array([[1., 0.]], dtype=np.float32)
            texts = np.tile(images, (bench["data"]["expected_texts"], 1))
            scores = metrics.score_rows(images, texts, bundle["labels"])
            runtime = {"modules": read(plan["dependencies"]["path"])["modules"], "python_executable": "synthetic-CPU-only",
                "matmul": {"allow_tf32": False, "cudnn_allow_tf32": False, "float32_precision": "highest"},
                "paired_backend": {"cudnn_benchmark": True}, "gpu": {"assigned_uuid": "synthetic-not-a-real-GPU", "central_task": "unit"}}
            base = {"plan_sha256": sha(plan_path), "plan_file": entry(plan_path), "model": "synthetic",
                "model_spec_sha256": digest(spec), "source_files": plan["source_files"]}
            identity = {**base, "benchmark": name, "benchmark_spec_sha256": digest(bench), "data_identity": bundle["identity"],
                "settings": settings, "runtime": runtime, "model_signature": {"synthetic": True}, "producer": "synthetic-producer", "freeze_commit": "f"*40}
            smoke = folder/"smoke.json"
            smoke.write_text(json.dumps({"state": "PASS", "identity": base, "runtime": runtime, "freeze_commit": "f"*40,
                "model_signature": identity["model_signature"], "checks": {key: {"cosine": 1., "max_abs": 0., "passed": True} for key in ("images", "texts")}}))
            image_items, text_items = data.inference_items(bundle)
            for key, values, items, modality in (("images", images, image_items, "images"), ("texts", texts, text_items, "texts"),
                                                ("scores", scores, [row["id"] for row in bundle["labels"]], "author_scalar_scores")):
                # All unit vectors are intentionally identical and all strict
                # events fail; no semantic information enters the fake encoder.
                with redirect_stdout(io.StringIO()):
                    C.checkpoint(folder/(key+".npy"), items, 1, lambda part, v=values: v[:len(part)], {**identity, "modality": modality})
            arrays = {key: entry(folder/(key+".npy")) for key in ("images", "texts")}
            original = {"state": "COMPLETE", "identity": identity, "counts": bundle["counts"], "arrays": arrays,
                "checkpoints": {key: entry(folder/(key+".state.json")) for key in ("images", "texts")}, "smoke": entry(smoke), "research_metrics": False}
            original_path = folder/"original-receipt.json"
            original_path.write_text(json.dumps(original))
            final = {**original, "embedding_receipt": entry(original_path), "author_scores": entry(folder/"scores.npy"),
                "author_scores_checkpoint": entry(folder/"scores.state.json")}
            final_path = folder/"final-receipt.json"
            final_path.write_text(json.dumps(final))
            report = {"state": "COMPLETE", "producer": identity["producer"], "identity": identity,
                "score_profile": metrics.PROFILE, "primary_metric_descriptors": bench["primary_metrics"], "data_identity": bundle["identity"],
                "counts": bundle["counts"], "arrays": arrays, "author_scores": final["author_scores"], "index_receipt": entry(final_path),
                "metrics": metrics.from_scores(scores, bundle["labels"])}
            reports[name] = folder/"report.json"
            reports[name].write_text(json.dumps(report))
        with patch.object(Q, "validate_plan", side_effect=Q._original_validate), \
                patch.object(Q.ENGINE, "validate_plan", side_effect=Q._original_validate), \
                patch.object(Q.ENGINE, "frozen_source_gate"):
            reviewed = Q.review(plan_path, reports, "separate-synthetic-reviewer", accept_protocol=True)
            review_path = self.root/"review.json"
            review_path.write_text(json.dumps(reviewed))
            result = Q.make(plan_path, reports, review_path)
            self.assertEqual(result["state"], "FAILED")
            self.assertEqual(len(result["benchmarks"]), 2)
            self.assertEqual(reviewed["recomputation_torch"], read(plan["dependencies"]["path"])["modules"]["torch"])


if __name__ == "__main__":
    unittest.main()
