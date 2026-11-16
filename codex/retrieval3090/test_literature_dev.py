"""[codex] Synthetic CPU tests; no GPU, downloaded vectors, or research effects."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import literature_dev as D


def bundle(n=60):
    gallery = [{"id": str(i+1), "path": "/synthetic/"+str(i+1)+".jpg", "split": "val2017",
                "sha256": f"{i+1:064x}", "bytes": 1, "width": 1, "height": 1} for i in range(n)]
    panels = {}
    for variant in ("plain", "negated"):
        rows = [{"id": f"{variant}-{i}", "text": f"{variant} caption {i}", "source_image_id": str(i+1),
                 "positive_ids": [str(i+1)], "caption_index": i, "group": "group-"+str(i),
                 "model_eligible": True, "exclusion_reason": None} for i in range(2)]
        for subset in ("full", "dev1000"):
            panels[f"negbench-{variant}-{subset}.json"] = {"task": "source-image-retrieval", "scope": D.SCOPE,
                                                            "queries": deepcopy(rows), "candidates": gallery}
    return {"gallery": gallery, "panels": panels,
            "texts": sorted({q["text"] for p in panels.values() for q in p["queries"]}),
            "panel_sha256": {key: "a"*64 for key in panels}, "preparation_sha256": "b"*64}


class LiteratureTests(unittest.TestCase):
    def setUp(self):
        folder = D.ROOT / "hf_home/tmp"
        folder.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="literature-dev-test-", dir=folder)
        self.out = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def plan(self):
        return {"schema": 1, "state": "FROZEN", "scope": D.SCOPE, "formal_evidence": False,
                "batch_sizes": D.BATCHES, "cross_panel": "negbench-negated-dev1000.json", "cross_budget": 50,
                "data_preparation_sha256": "c"*64, "include_factor": False,
                "analysis": {"seed": 7, "bootstrap_draws": 32, "block_queries": 1}}

    def test_stable_ties_source_rank(self):
        scores = np.array([[.5, .5, .3], [.2, .9, .9]])
        self.assertEqual(D.stable_order(scores).tolist(), [[0, 1, 2], [1, 2, 0]])
        self.assertEqual(D.source_ranks(scores, [1, 2]).tolist(), [2, 2])

    def test_metrics_are_native_single_source(self):
        values = D.rank_metrics([1, 2, 60])
        self.assertAlmostEqual(D.metrics_summary(values)["MRR"], 100*(1+.5+1/60)/3)
        self.assertAlmostEqual(D.metrics_summary(values)["R@50"], 200/3)

    def test_nonfinite_scores_rejected(self):
        with self.assertRaises(ValueError):
            D.stable_order([[0, np.nan]])

    def test_source_position_outside_gallery_rejected(self):
        with self.assertRaises(ValueError):
            D.source_ranks([[1, 2]], [2])

    def test_cross_keeps_positive_outside_shortlist(self):
        order = np.array([np.arange(60), np.arange(60)])
        scores = np.tile(np.arange(50), (2, 1))
        before, after = D.reranked_source_ranks(order, scores, [55, 0])
        self.assertEqual(before.tolist(), [56, 1])
        self.assertEqual(after.tolist(), [56, 50])
        np.testing.assert_array_equal(D.rank_metrics(before)[:, -1], D.rank_metrics(after)[:, -1])

    def test_cross_ties_retain_core_prefix_order(self):
        order = np.arange(60)[None, :]
        before, after = D.reranked_source_ranks(order, np.zeros((1, 50)), [22])
        np.testing.assert_array_equal(before, after)

    def test_admission_ignores_poisoned_labels_and_weight_fields(self):
        data = bundle()
        image = np.eye(60, dtype=np.float32)
        text = np.zeros((4, 60), dtype=np.float32)
        text[:, 0] = 1
        rows = data["panels"]["negbench-negated-dev1000.json"]["queries"]
        first = D.shortlist(data, image, text, rows, 1)
        poisoned = deepcopy(rows)
        for q in poisoned:
            q.update(positive_ids=["999999"], source_image_id="poison", positive_objects=["secret"],
                     negative_objects=["secret"], weight=1e100, cross_score=1e100)
        np.testing.assert_array_equal(first, D.shortlist(data, image, text, poisoned, 2))

    def test_blockwise_equals_dense_native_ranks(self):
        data = bundle()
        rng = np.random.default_rng(7)
        image = rng.normal(size=(60, 8)).astype(np.float32)
        text = rng.normal(size=(4, 8)).astype(np.float32)
        rows = data["panels"]["negbench-plain-full.json"]["queries"]
        _, result = D.query_blocks(data, image, text, rows, 1)
        lookup = {t: i for i, t in enumerate(data["texts"])}
        expected = D.source_ranks(text[[lookup[q["text"]] for q in rows]] @ image.T, [0, 1])
        np.testing.assert_array_equal(result, expected)

    def test_empty_original_caption_is_excluded_before_encoding(self):
        data = bundle()
        rows = deepcopy(data["panels"]["negbench-plain-full.json"]["queries"])
        rows.append({"id": "empty", "text": "", "model_eligible": False,
                     "exclusion_reason": "source-empty-caption", "source_image_id": "1"})
        selected, ranks = D.query_blocks(data, np.eye(60), np.ones((4, 60)), rows, 1)
        self.assertEqual(len(selected), 2)
        self.assertEqual(len(ranks), 2)

    def test_truncation_audit_counts_query_rows_not_just_unique_texts(self):
        data = bundle()
        name = "negbench-negated-full.json"
        repeated = deepcopy(data["panels"][name]["queries"][0]); repeated["id"] = "duplicate-text-row"
        data["panels"][name]["queries"].append(repeated)
        lengths = [100 if t.startswith("negated") else 10 for t in data["texts"]]
        audit = D.tokenization_audit(data, data["texts"], lengths)
        self.assertEqual(audit["over_context"], 2)
        self.assertEqual(audit["panels"][name]["over_77_query_rows"], 3)
        self.assertEqual(audit["panels"][name]["unencoded_eligible_rows"], 0)

    def test_plain_negative_pairing_is_not_just_row_count(self):
        data = bundle()
        D.validate_panels(data["panels"], enforce_counts=False)
        data["panels"]["negbench-negated-dev1000.json"]["queries"].reverse()
        with self.assertRaises(ValueError):
            D.validate_panels(data["panels"], enforce_counts=False)

    def test_development_must_be_literal_full_row(self):
        data = bundle()
        data["panels"]["negbench-plain-dev1000.json"]["queries"][0]["text"] = "changed query"
        with self.assertRaises(ValueError):
            D.validate_panels(data["panels"], enforce_counts=False)

    def test_cluster_bootstrap_is_paired_reproducible(self):
        left = np.zeros((3, 5)); right = np.ones((3, 5))*.1
        a = D.paired_stats(left, right, ["a", "a", "b"], seed=42, draws=128)
        b = D.paired_stats(left, right, ["a", "a", "b"], seed=42, draws=128)
        self.assertEqual(a, b)
        self.assertEqual(a["groups"], 2)
        self.assertAlmostEqual(a["metrics"]["MRR"]["delta_pp"], 10)
        np.testing.assert_allclose(a["metrics"]["MRR"]["ci95_pp"], [10, 10])

    def test_plan_rejects_unfrozen_or_changed_budget(self):
        path = self.out / "plan.json"
        plan = self.plan()
        D.M.atomic_json(path, plan)
        self.assertEqual(D.load_plan(path), plan)
        plan["cross_budget"] = 200
        D.M.atomic_json(path, plan)
        with self.assertRaises(ValueError):
            D.load_plan(path)

    def test_smoke_and_development_modes_differ(self):
        data = bundle()
        path = self.out / "plan.json"
        D.M.atomic_json(path, self.plan())
        config = D.M.frozen_config()
        a = D.input_identity(data, path, config)
        b = D.input_identity(data, path, config, smoke=True)
        self.assertNotEqual(a, b)
        self.assertEqual(a["batch_sizes"], {"image": 8, "text": 32, "cross": 4})

    def test_canonical_npy_hash_and_completion(self):
        path = self.out / "array.npy"
        np.save(path, np.eye(2, dtype=np.float32))
        D.M.atomic_json(path.with_suffix(".state.json"), {"completed": 2, "array_shape": [2, 2],
            "dtype": np.dtype(np.float32).str, "sha256": D.sha(path)})
        np.testing.assert_array_equal(D.complete_array(path, 2, 2), np.eye(2))
        values = np.load(path); values[0, 0] = 2; np.save(path, values)
        with self.assertRaises(ValueError):
            D.complete_array(path, 2, 2)

    def make_index(self):
        data = bundle()
        plan_path = self.out / "plan.json"
        D.M.atomic_json(plan_path, self.plan())
        identity = D.input_identity(data, plan_path, D.M.frozen_config()) | {"environment": {"test": "CPU-synthetic"}}
        arrays = {}
        for name in ("core_image", "core_text", "cs_image", "cs_text"):
            rows = len(data["gallery"] if name.endswith("image") else data["texts"])
            dims = 2048 if name.startswith("core") else 512
            values = np.zeros((rows, dims), dtype=np.float32)
            values[:, 0] = 1
            path = self.out / (name+".npy")
            np.save(path, values)
            D.M.atomic_json(path.with_suffix(".state.json"), {"completed": rows, "array_shape": [rows, dims],
                "dtype": values.dtype.str, "sha256": D.sha(path)})
            arrays[name] = {"sha256": D.sha(path), "state_sha256": D.sha(path.with_suffix(".state.json"))}
        D.M.atomic_json(self.out / "index-meta.json", {"state": "COMPLETE", "identity": identity,
            "processors": {"core_embed": {"synthetic": True}, "cs_clip": {"synthetic": True}}, "arrays": arrays})
        return data, plan_path

    def test_index_resume_validates_source_and_plan_bytes(self):
        data, plan_path = self.make_index()
        arrays, _ = D.validate_index(self.out, data, plan_path)
        self.assertEqual(set(arrays), {"core_image", "core_text", "cs_image", "cs_text"})
        with patch.object(D, "source_files", return_value={"changed.py": "0"*64}):
            with self.assertRaises(ValueError):
                D.validate_index(self.out, data, plan_path)
        plan = D.read(plan_path); plan["comment"] = "new bytes"
        D.M.atomic_json(plan_path, plan)
        with self.assertRaises(ValueError):
            D.validate_index(self.out, data, plan_path)

    def test_index_requires_actual_environment_processor_provenance(self):
        data, plan_path = self.make_index()
        path = self.out / "index-meta.json"
        meta = D.read(path)
        meta["identity"].pop("environment")
        D.M.atomic_json(path, meta)
        with self.assertRaises(ValueError):
            D.validate_index(self.out, data, plan_path)

    def test_index_cannot_reuse_incomplete_checkpoint(self):
        data, plan_path = self.make_index()
        path = self.out / "core_image.state.json"
        state = D.read(path); state["completed"] = 59
        D.M.atomic_json(path, state)
        meta = D.read(self.out / "index-meta.json")
        meta["arrays"]["core_image"]["state_sha256"] = D.sha(path)
        D.M.atomic_json(self.out / "index-meta.json", meta)
        with self.assertRaises(ValueError):
            D.validate_index(self.out, data, plan_path)

    def test_nonempty_caption_cannot_be_silently_excluded(self):
        data = bundle()
        row = data["panels"]["negbench-plain-full.json"]["queries"][0]
        row.update(model_eligible=False, exclusion_reason="source-empty-caption")
        with self.assertRaises(ValueError):
            D.validate_panels(data["panels"], enforce_counts=False)

    def test_cross_receipt_is_bound_to_actual_core_pairs(self):
        data = bundle()
        rows = [{"id": f"q{i}", "text": data["texts"][0], "model_eligible": True,
                 "source_image_id": "60", "group": f"g{i}"} for i in range(1000)]
        data["panels"]["negbench-negated-dev1000.json"]["queries"] = rows
        vectors = {"core_image": np.eye(60, dtype=np.float32), "core_text": np.zeros((4, 60), dtype=np.float32)}
        index_meta = {"identity": {"synthetic": True}}
        D.M.atomic_json(self.out / "index-meta.json", index_meta)
        pairs = [[qi, ci] for qi in range(1000) for ci in range(50)]
        D.M.atomic_json(self.out / "cross-pairs.json", pairs)
        path = self.out / "cross50.npy"
        np.save(path, np.zeros(50000, dtype=np.float32))
        D.M.atomic_json(path.with_suffix(".state.json"), {"completed": 50000, "array_shape": [50000],
            "dtype": np.dtype(np.float32).str, "sha256": D.sha(path), "active_seconds": 10.0})
        identity = index_meta["identity"] | {"index_meta_sha256": D.sha(self.out / "index-meta.json"),
            "cross_batch": 4, "cross_processor": {"synthetic": True}, "query_ids": [q["id"] for q in rows],
            "pairs_sha256": D.sha(self.out / "cross-pairs.json")}
        meta = {"state": "COMPLETE", "identity": identity, "cross_sha256": D.sha(path),
                "state_sha256": D.sha(path.with_suffix(".state.json"))}
        D.M.atomic_json(self.out / "cross-meta.json", meta)
        got_rows, order, cross, _ = D.validate_cross_artifacts(self.out, data, self.plan(), vectors, index_meta)
        self.assertEqual((len(got_rows), order.shape, cross.shape), (1000, (1000, 60), (50000,)))
        pairs[0][1] = 59
        D.M.atomic_json(self.out / "cross-pairs.json", pairs)
        meta["identity"]["pairs_sha256"] = D.sha(self.out / "cross-pairs.json")
        D.M.atomic_json(self.out / "cross-meta.json", meta)
        with self.assertRaises(ValueError):
            D.validate_cross_artifacts(self.out, data, self.plan(), vectors, index_meta)

    def test_factor_equivalence_not_forced_binary(self):
        data = bundle()
        data["texts"] = ["a", "b"]
        data["panels"]["factor-full.json"] = {"items": [
            {"id": "both", "captions": ["a", "b"], "image_id": "1", "group": "g1",
             "test_type": "equivalence", "operator": "COMM_AND", "correct_caption_index": None},
            {"id": "binary", "captions": ["a", "b"], "image_id": "1", "group": "g1",
             "test_type": "operator", "operator": "AND", "correct_caption_index": 0}]}
        arrays = {p+"_image": np.ones((60, 2)) for p in ("core", "cs")}
        arrays.update({p+"_text": np.array([[2., 0], [1., 0]]) for p in ("core", "cs")})
        summary = D.factor_summary(data, arrays)
        self.assertIsNone(summary["core"]["groups"]["test_type:equivalence"]["binary_accuracy_pct"])
        self.assertEqual(summary["core"]["groups"]["test_type:operator"]["binary_accuracy_pct"], 100)

    def test_sealed_and_symlink_paths_rejected(self):
        with self.assertRaises(ValueError):
            D.safe(self.out / "retrieval-independent-panel" / "manifest.json")
        link = self.out / "alias"
        link.symlink_to(self.out, target_is_directory=True)
        with self.assertRaises(ValueError):
            D.safe(link / "file.json")


if __name__ == "__main__":
    unittest.main()
