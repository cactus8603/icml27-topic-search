"""Synthetic integrity tests; never read a research result or validation panel."""
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import csv
import json
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diagnostics as d


def local_context():
    return {"mode": "platform-relative-exploration", "formal_evidence": False,
            "platform_consistency": "VALID_LOCAL", "plan_sha256": "a"*64, "receipt_sha256": "b"*64,
            "published_reproduction": {"state": "MISMATCH", "metrics": {"map_pp": -123.456}},
            "baseline": {"map_pp": -123.456}}


class DiagnosticTests(unittest.TestCase):
    def fixture(self):
        ids = list(range(1, 10))
        categories = {1: {1, 2, 3, 4}, 2: {2, 3, 4}, 3: {1, 3, 4}, 4: {1, 2, 4},
                      5: {1, 2, 3}, 6: {1}, 7: {2}, 8: {3}, 9: {4}}
        names = {key: "object" + str(key) for key in range(1, 5)}
        areas = {key: {cat: .1 for cat in value} for key, value in categories.items()}
        return ids, names, categories, areas

    def test_nested_family_fixes_both_images_and_missing_condition(self):
        fixture = self.fixture()
        design = d.make_d3(*fixture, seed=123)
        categories = fixture[2]
        self.assertEqual(len(design["queries"]), 4)
        previous = []
        for k, query in enumerate(design["queries"], 1):
            self.assertEqual(query["key"][:len(previous)], previous)
            self.assertEqual(len(query["key"]), k)
            self.assertTrue(set(query["key"]) <= categories[query["positive_id"]])
            self.assertEqual(len(set(query["key"]) - categories[query["negative_id"]]), 1)
            self.assertNotIn(query["negative_id"], query["relevant_ids"])
            previous = query["key"]

    def test_score_blind_selection_is_deterministic(self):
        self.assertEqual(d.make_d3(*self.fixture(), seed=123), d.make_d3(*self.fixture(), seed=123))

    def test_coco_presence_ignores_crowd_and_subthreshold_instances(self):
        data = {"images": [{"id": 1, "width": 10, "height": 10}],
                "categories": [{"id": i, "name": str(i)} for i in (1, 2, 3)],
                "annotations": [{"image_id": 1, "category_id": 1, "area": 2},
                                {"image_id": 1, "category_id": 2, "area": 1.9},
                                {"image_id": 1, "category_id": 3, "area": 30, "iscrowd": 1}]}
        _, cats, _ = d.coco_fields(data, [1])
        self.assertEqual(cats, {1: {1}})

    def test_reused_images_form_connected_clusters(self):
        families = [{"id": "a", "positive_id": 1, "negative_id": 2},
                    {"id": "b", "positive_id": 3, "negative_id": 2},
                    {"id": "c", "positive_id": 4, "negative_id": 5}]
        clusters = d.connected_pair_groups(families)
        self.assertEqual(clusters["a"], clusters["b"])
        self.assertNotEqual(clusters["a"], clusters["c"])

    def test_cluster_bootstrap_point_keeps_family_weights(self):
        result = d.cluster_interval([0, 0, 1], ["a", "a", "b"], draws=50)
        self.assertAlmostEqual(result["point"], 1 / 3)
        self.assertEqual(result["clusters"], 2)
        self.assertIsNone(d.cluster_interval([1, 2], ["a", "a"], draws=50)["ci95"])

    def test_unidentified_controls_do_not_produce_a_k_claim(self):
        queries = [{"family": "f", "k": k, "positive_count": 2**k,
                    "mean_log_category_prevalence": .5 * k, "mean_positive_instance_area": .1,
                    "hard_negative_count": 1, "shared_positive_negative_area_difference": 0} for k in range(1, 5)]
        result = d.controlled_slope([4, 3, 2, 1], queries)
        self.assertFalse(result["identified"])
        self.assertIsNone(result["k_slope"])

    def test_holistic_scores_use_new_complete_query_vectors(self):
        design = d.make_d3(*self.fixture(), seed=123)
        texts = [row["text"] for row in design["texts"]]
        image = np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.], [.6, .8], [.8, .6], [-.6, .8], [-.8, .6], [.5, -.5]])
        vectors = np.tile([1., 0.], (len(texts), 1))
        last = design["queries"][-1]
        vectors[texts.index(last["text"])] = [0., 1.]
        scores = d.d3_scores(image, vectors, texts, design)
        p, n = last["positive_id"] - 1, last["negative_id"] - 1
        self.assertAlmostEqual(scores["holistic"][-1]["margin"], image[p, 1] - image[n, 1])
        self.assertAlmostEqual(scores["unit_additive"][-1]["margin"], image[p, 0] - image[n, 0])

    def test_blind_public_fields_do_not_reveal_methods_or_labels(self):
        row = d.public_audit_case("opaque", "a photo of cup", "images/opaque.png")
        self.assertEqual(row["all_conditions_visible"], "")
        for forbidden in ("rank", "method", "stratum", "annotation_positive", "image_id"):
            self.assertNotIn(forbidden, row)

    def test_full_blind_export_keeps_truth_only_in_private_mapping(self):
        from PIL import Image

        temp_root = Path(__file__).resolve().parents[2] / "hf_home/tmp"
        temp_root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=temp_root) as temporary:
            root = Path(temporary)
            cache, images = root / "cache", root / "images"
            cache.mkdir(); images.mkdir()
            ids = list(range(1, 61))
            panel = {"ids": ids, "queries": [{"text": "a photo of object", "key": [1], "relevant": [60]}]}
            (cache / "panel.json").write_text(json.dumps(panel))
            (cache / "meta.json").write_text(json.dumps({"replay": {"state": "PASS"},
                "model_config_sha256": d.MODEL_CONFIG_SHA, "revisions": d.MODEL_REVISIONS}))
            np.savez(cache / "cross.npz", core_order=np.arange(60)[None, :], cross=np.zeros((1, 60)))
            for key in ids:
                Image.new("RGB", (2, 2), (key, 0, 0)).save(images / f"{key:012d}.jpg")
            d.blind_export(SimpleNamespace(cache=cache, image_root=images, out=root / "audit", samples=3, seed=123))
            with (root / "audit/review/labels.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            private = json.loads((root / "audit/private/key.json").read_text())
            self.assertEqual(len(rows), 3)
            self.assertEqual(len(private["cases"]), 3)
            self.assertTrue(all(row["all_conditions_visible"] == "" for row in rows))
            self.assertNotIn("annotation_positive", (root / "audit/review/index.html").read_text())
            for row in rows:
                with Image.open(root / "audit/review" / row["image"]) as image:
                    self.assertFalse(image.info)
            meta = json.loads((cache / "meta.json").read_text())
            meta["replay"]["state"] = "MISMATCH"
            (cache / "meta.json").write_text(json.dumps(meta))
            args = SimpleNamespace(cache=cache, image_root=images, out=root/"platform-audit", samples=3, seed=123,
                                   platform_plan=root/"plan.json", platform_receipt=root/"receipt.json")
            with patch.dict(sys.modules, {"platform_context": SimpleNamespace(validate_context=lambda *a, **k: local_context())}):
                d.blind_export(args)
            private = json.loads((root/"platform-audit/private/key.json").read_text())
            self.assertEqual(private["evaluation_context"]["comparison_scope"], "local-relative")
            self.assertFalse(private["formal_evidence"])
            for name in ("index.html", "labels.csv"):
                public = (root/"platform-audit/review"/name).read_text()
                for hidden in ("MISMATCH", "map_pp", "-123.456", "annotation_positive", "platform_consistency"):
                    self.assertNotIn(hidden, public)

    def test_mcmr_keeps_candidate_metadata_separate_from_gold(self):
        candidates, queries = d.normalize_mcmr(
            [{"candidate_id": "A", "title": "red chair", "images": [{"large": "https://example.org/a.jpg"}]}],
            [{"query": "a red chair", "pos_ids": ["A"]}])
        self.assertEqual(candidates[0]["text"], "red chair")
        self.assertEqual(candidates[0]["image_filename"], "a.jpg")
        self.assertEqual(queries[0]["positive_ids"], ["A"])
        self.assertNotIn("positive_ids", candidates[0])

    def test_missing_or_changed_cache_provenance_is_rejected(self):
        meta = {"replay": {"state": "PASS"}, "model_config_sha256": d.MODEL_CONFIG_SHA,
                "revisions": dict(d.MODEL_REVISIONS)}
        d.validate_cache_meta(meta)
        meta["replay"]["state"] = "MISMATCH"
        with self.assertRaises(ValueError):
            d.validate_cache_meta(meta)

    def test_diagnostic_platform_requires_explicit_complete_opt_in_and_valid_context(self):
        meta = {"replay": {"state": "PASS"}, "model_config_sha256": d.MODEL_CONFIG_SHA,
                "revisions": dict(d.MODEL_REVISIONS)}
        self.assertEqual(d.diagnostic_context(SimpleNamespace(), Path("cache"), meta)["published_reproduction"], "PASS")
        meta["replay"]["state"] = "MISMATCH"
        with self.assertRaisesRegex(ValueError, "baseline replay"):
            d.diagnostic_context(SimpleNamespace(), Path("cache"), meta)
        with self.assertRaisesRegex(ValueError, "together"):
            d.diagnostic_context(SimpleNamespace(platform_plan=Path("plan")), Path("cache"), meta)
        args = SimpleNamespace(platform_plan=Path("plan"), platform_receipt=Path("receipt"))
        for rejected in (None, {}, dict(local_context(), formal_evidence=True)):
            with patch.dict(sys.modules, {"platform_context": SimpleNamespace(validate_context=lambda *a, **k: rejected)}):
                with self.assertRaisesRegex(ValueError, "valid explicit"):
                    d.diagnostic_context(args, Path("cache"), meta)
        with patch.dict(sys.modules, {"platform_context": SimpleNamespace(validate_context=lambda *a, **k: local_context())}):
            context = d.diagnostic_context(args, Path("cache"), meta)
            self.assertEqual(context["comparison_scope"], "local-relative")
            self.assertEqual(meta["replay"]["state"], "MISMATCH")
            meta["model_config_sha256"] = "changed"
            with self.assertRaisesRegex(ValueError, "configuration"):
                d.diagnostic_context(args, Path("cache"), meta)

    def test_d3_platform_binds_supplement_and_keeps_default_replay_gate(self):
        design = d.make_d3(*self.fixture(), seed=123)
        temp_root = Path(__file__).resolve().parents[2]/"hf_home/tmp"
        temp_root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=temp_root) as temporary:
            root = Path(temporary)
            cache = root/"cache"
            cache.mkdir()
            design_path = root/"design.json"
            design_path.write_text(json.dumps(design))
            meta = {"replay": {"state": "MISMATCH"}, "model_config_sha256": d.MODEL_CONFIG_SHA,
                    "revisions": dict(d.MODEL_REVISIONS), "bundle": {"source": "1"*40, "freeze": "2"*40}}
            (cache/"meta.json").write_text(json.dumps(meta))
            (cache/"panel.json").write_text(json.dumps({"ids": design["gallery_ids"]}))
            rng = np.random.default_rng(53)
            def unit(shape):
                values = rng.normal(size=shape)
                return values/np.linalg.norm(values, axis=1, keepdims=True)
            text_rows = [row["text"] for row in design["texts"]]
            vectors = {name+"_image": unit((len(design["gallery_ids"]), 8)) for name in ("core_embed", "cs_clip")}
            texts = {name+"_text": unit((len(text_rows), 8)) for name in ("core_embed", "cs_clip")}
            encoding = {"schema": "d3-holistic-encodings-v1", "design_sha256": d.file_sha(design_path),
                        "texts_sha256": design["texts_sha256"], "gallery_ids_sha256": design["gallery_ids_sha256"],
                        "normalization": "unit_l2", "model_revisions": d.MODEL_REVISIONS,
                        "model_config_sha256": d.MODEL_CONFIG_SHA, "source": "1"*40, "freeze": "2"*40, "commit": "3"*40}
            embeddings = root/"d3_embeddings.npz"
            np.savez(cache/"vectors.npz", **vectors)
            np.savez(embeddings, **texts, texts=np.asarray(text_rows), metadata_json=np.asarray(json.dumps(encoding)))
            args = SimpleNamespace(design=design_path, cache=cache, embeddings=embeddings, out=root/"report.json")
            with self.assertRaisesRegex(ValueError, "baseline replay"):
                d.analyze_d3(args)
            self.assertFalse(args.out.exists())
            args.platform_plan, args.platform_receipt = root/"plan.json", root/"receipt.json"
            called = []
            def validate(*positional, supplementary_paths):
                called.append((positional, supplementary_paths))
                return local_context()
            with patch.dict(sys.modules, {"platform_context": SimpleNamespace(validate_context=validate)}):
                d.analyze_d3(args)
            self.assertEqual(called[0][1], (embeddings.resolve(),))
            report = json.loads(args.out.read_text())
            self.assertEqual(report["evaluation_context"]["published_reproduction"]["state"], "MISMATCH")
            self.assertEqual(report["comparison_scope"], "local-relative")
            self.assertFalse(report["formal_evidence"])

    def test_supplemental_embeddings_are_bound_to_full_design_and_models(self):
        design = d.make_d3(*self.fixture(), seed=123)
        meta = {"schema": "d3-holistic-encodings-v1", "design_sha256": "abc",
                "texts_sha256": design["texts_sha256"], "gallery_ids_sha256": design["gallery_ids_sha256"],
                "normalization": "unit_l2", "model_revisions": d.MODEL_REVISIONS,
                "model_config_sha256": d.MODEL_CONFIG_SHA, "source": "1" * 40, "freeze": "2" * 40, "commit": "3" * 40}
        cache_meta = {"bundle": {"source": "1" * 40, "freeze": "2" * 40}}
        d.validate_embedding_meta(meta, design, "abc", cache_meta)
        with self.assertRaises(ValueError):
            d.validate_embedding_meta(dict(meta, source="garbage"), design, "abc", cache_meta)
        with self.assertRaises(ValueError):
            d.validate_embedding_meta(dict(meta, source="4" * 40), design, "abc", cache_meta)
        meta["texts_sha256"] = "changed"
        with self.assertRaises(ValueError):
            d.validate_embedding_meta(meta, design, "abc")

    def test_family_contrast_does_not_assume_row_order(self):
        design = d.make_d3(*self.fixture(), seed=123)
        d.validate_design(design)
        rows = [{"family": q["family"], "k": q["k"], "margin": float(q["k"])} for q in reversed(design["queries"])]
        changes, _ = d.family_differences(rows, design, "margin")
        self.assertEqual(changes, [3.])
        invalid = dict(design, queries=design["queries"][:-1])
        with self.assertRaises(ValueError):
            d.validate_design(invalid)

    def test_sealed_input_path_rejected(self):
        with self.assertRaises(ValueError):
            d.safe("/tmp/retrieval-independent-panel/panel.json")


if __name__ == "__main__":
    unittest.main()
