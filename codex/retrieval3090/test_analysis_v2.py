"""[codex] Synthetic CPU tests of analysis semantics; never reads study results."""

import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis_v2 as V


def synthetic(seed=4):
    rng = np.random.default_rng(seed)
    queries, pool, dim = 40, 300, 16
    unit = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True)
    rows, cats = [], {str(i + 1): [] for i in range(pool)}
    for q in range(queries):
        key = [2 * q, 2 * q + 1]
        relevant = [q + 1] if q % 2 == 0 else [q + 1, q + queries + 1]
        for image in relevant:
            cats[str(image)] = key
        rows.append({"key": key, "text": f"condition {key}", "relevant": relevant, "group": q + 1})
    cache = {}
    for name in ("core_embed", "cs_clip"):
        cache[f"{name}_image"] = unit(rng.normal(size=(pool, dim)))
        cache[f"{name}_full"] = unit(rng.normal(size=(queries, dim)))
        cache[f"{name}_condition"] = unit(rng.normal(size=(2 * queries, dim)))
    core = np.argsort(-(cache["core_embed_full"] @ cache["core_embed_image"].T), axis=1, kind="stable")
    cache.update(core_order=core, cross=rng.normal(size=(queries, 200)), condition_index=[q["key"] for q in rows])
    panel = {"ids": list(range(1, pool + 1)), "queries": rows, "condition_index": cache["condition_index"], "image_categories": cats,
             "condition_texts": [f"a photo of object{i}" for i in range(2 * queries)]}
    _, cross, rel = V.validate_cache(cache, panel)
    base = V.measure(V.A.final_order(core, core[:, :50], cross[:, :50]), rel, 50)
    expected = {"map_pp": float(base["ap_pp"].mean()),
                "oracle_map_pp": float(base["admission_oracle_ap_pp"].mean()),
                "recall_pp": float(base["recall_at_50_pp"].mean()),
                "queries_without_hit": int((base["hits_at_50"] == 0).sum())}
    meta = {"replay": {"state": "PASS"}, "panel_sha256": "b" * 64,
            "revisions": {"core_embed": "c" * 40, "cs_clip": "d" * 40, "core_reranker": "e" * 40},
            "bundle": {"source": "1" * 40, "freeze": "2" * 40, "commit": "3" * 40}}
    return cache, panel, meta, expected


def synthetic_logic(cache, panel, meta):
    import logic_baselines as L
    rng = np.random.default_rng(16)
    labels = np.tile([0, 1], (8, 3))
    artifacts, vectors = [], {}
    for name in ("core_embed", "cs_clip"):
        scores = rng.uniform(-.2, .7, labels.shape)
        artifacts.append(L.fit_presence_calibration(scores, labels, image_ids=list(range(1001, 1009)),
            excluded_image_ids=panel["ids"], backbone=name, source_manifest_sha256="a" * 64,
            mu_grid=[.1, .2], beta_grid=[5, 10]))
        table = rng.normal(size=(*cache[f"{name}_condition"].shape[:1], 4, 16))
        table /= np.linalg.norm(table, axis=-1, keepdims=True)
        vectors[f"{name}_template_vectors"] = table
        vectors[f"{name}_lcse_condition"] = table.mean(axis=1)
    artifact = artifacts[0]
    artifact["source"]["calibration_design_sha256"] = "f" * 64
    artifact["models"].update(artifacts[1]["models"])
    vectors["condition_texts"] = np.asarray(panel["condition_texts"])
    provenance = {"schema": 1, "scope": "lcse-four-template-condition-vectors",
        "templates": list(L.CONCEPT_TEMPLATES),
        "calibration_source_manifest_sha256": artifact["source"]["manifest_sha256"],
        "calibration_design_sha256": artifact["source"]["calibration_design_sha256"],
        "eval_manifest_sha256": meta["panel_sha256"], "model_revisions": meta["revisions"],
        "source": meta["bundle"]["source"], "freeze": meta["bundle"]["freeze"],
        "bundle_commit": "4" * 40}
    vectors["metadata_json"] = np.asarray(json.dumps(provenance))
    return L, artifact, vectors


class AnalysisV2Tests(unittest.TestCase):
    def test_fixed50_and_admission_recall_are_distinct(self):
        order = np.arange(100)[None, :]
        relevance = np.zeros_like(order, dtype=bool)
        relevance[0, 60] = True
        measured = V.measure(order, relevance, [75])
        self.assertEqual(measured["recall_at_50_pp"].tolist(), [0.0])
        self.assertEqual(measured["admission_recall_pp"].tolist(), [100.0])
        s = V.summary(measured)
        self.assertEqual(s["queries_without_top50_hit"], 1)
        self.assertEqual(s["queries_without_admitted_hit"], 0)

    def test_replay_failure_stops_before_policy_construction(self):
        cache, panel, meta, expected = synthetic()
        with patch.object(V, "features", side_effect=AssertionError("must not reach policies")):
            with self.assertRaisesRegex(ValueError, "did not PASS"):
                V.analyse(cache, panel, {"replay": {"state": "MISMATCH"}}, published=expected)
            expected["map_pp"] += 1
            with self.assertRaisesRegex(ValueError, "Recomputed baseline"):
                V.analyse(cache, panel, meta, published=expected)

    def test_fold_hash_preserves_groups_and_is_order_invariant(self):
        groups = np.repeat(np.arange(30), 2)
        folds = V.fold_assignment(groups)
        np.testing.assert_array_equal(folds[::2], folds[1::2])
        np.testing.assert_array_equal(folds[::-1], V.fold_assignment(groups[::-1]))
        self.assertEqual(set(folds), set(range(5)))

    def test_connected_oof_groups_preserve_sentinels_and_shared_positive_images(self):
        queries = [{"group": group, "relevant": relevant} for group, relevant in
                   ((7, [100]), (7, [101]), (21, [101, 102]), (33, [102]), (90, [999]))]
        components = V.connected_query_groups(queries)
        np.testing.assert_array_equal(components, [7, 7, 7, 7, 90])
        overlap = V.positive_fold_overlap(queries, V.fold_assignment(components))
        self.assertEqual(overlap["images_in_multiple_folds"], 0)
        self.assertEqual(overlap["affected_queries"], 0)
        reversed_components = V.connected_query_groups(queries[::-1])
        np.testing.assert_array_equal(reversed_components, components[::-1])

    def test_selector_does_not_use_heldout_labels(self):
        groups, folds = np.arange(10), np.repeat(np.arange(5), 2)
        def metric(ap):
            return {"ap_pp": np.asarray(ap, dtype=float), "recall_at_50_pp": np.zeros(10),
                    "admission_recall_pp": np.zeros(10), "budget": np.full(10, 50),
                    "hits_at_50": np.zeros(10, dtype=int), "admitted_hits": np.zeros(10, dtype=int),
                    "admission_oracle_ap_pp": np.ones(10)}
        methods = {"a": metric([200, 200] + [0] * 8), "b": metric([0, 0] + [100] * 8)}
        _, records, chosen = V.oof_selection(methods, set(methods), groups, folds)
        self.assertEqual(records[0]["selected_policy"], "b")
        self.assertEqual(chosen[:2], ["b", "b"])
        methods["a"]["ap_pp"][:2] = -200
        _, changed, _ = V.oof_selection(methods, set(methods), groups, folds)
        self.assertEqual(changed[0]["selected_policy"], "b")

    def test_shuffled_conditions_are_whole_query_derangement(self):
        index = np.array([[0, 1], [2, 3], [4, 5], [6, 7]])
        shuffled, receipt = V.shuffled_conditions(index)
        self.assertEqual(receipt["self_donors"], 0)
        self.assertEqual(receipt["same_condition_pair_queries"], 0)
        np.testing.assert_array_equal(np.sort(shuffled, axis=0), np.sort(index, axis=0))
        self.assertTrue(np.all(np.any(shuffled != index, axis=1)))

    def test_random_allocations_have_exact_cost_and_uncertainties_are_label_free(self):
        budgets = V.exact_batch_budgets(np.arange(136))
        self.assertEqual(int(budgets.sum()), 136 * 50)
        np.testing.assert_array_equal(np.sort(budgets), np.sort(np.random.default_rng(2).permutation(budgets)))
        with self.assertRaisesRegex(ValueError, "divisible by four"):
            V.exact_batch_budgets(np.arange(135))
        cache, _, _, _ = synthetic()
        scores = cache["core_embed_full"] @ cache["core_embed_image"].T
        uncertainty = V.uncertainty_scores(scores, cache["core_order"])
        self.assertEqual(len(uncertainty), 5)
        self.assertTrue(all(np.isfinite(value).all() for value in uncertainty.values()))

    def test_frozen_metric_parity_and_paired_zero(self):
        order = np.array([[2, 1, 0, 3]])
        relevance = np.array([[1, 0, 1, 0]], dtype=bool)
        measured = V.measure(order, relevance, 2)
        legacy = V.A.measure(order, relevance, 2)
        np.testing.assert_allclose(measured["ap_pp"], legacy["ap"])
        np.testing.assert_allclose(measured["admission_oracle_ap_pp"], legacy["oracle"])
        out = V.paired(measured, measured, V.bootstrap_plan([0], draws=20))
        for row in out.values():
            self.assertEqual(row["ci95_pp"], [0.0, 0.0])
            self.assertEqual(row["normal_approx_mde80_pp"], 0.0)

    def test_invalid_candidate_inventory_is_rejected(self):
        cache, panel, _, _ = synthetic()
        cache["core_order"][0, 0] = cache["core_order"][0, 1]
        with self.assertRaisesRegex(ValueError, "complete candidate permutation"):
            V.validate_cache(cache, panel)

    def test_synthetic_end_to_end_has_explicit_references_controls_and_oof(self):
        cache, panel, meta, expected = synthetic()
        with patch.object(V, "RANDOM_SELECT_SEEDS", (123,)), patch.object(V, "RANDOM_ALLOCATION_SEEDS", (456, 457)):
            report, per_query = V.analyse(cache, panel, meta, published=expected, draws=30)
        self.assertNotIn("rerank_all_ceiling", report)
        for budget in (25, 50, 75, 100, 150, 200):
            self.assertIn(f"reference/rerank{budget}", report["methods"])
        oof = report["methods"]["oof/m1_single_selector"]
        self.assertEqual(oof["summary"]["mean_pairs_per_query"], 50)
        self.assertEqual(len(oof["folds"]), 5)
        self.assertIn("paired_against_core50_connected_groups", oof)
        self.assertEqual(report["oof_grouping"]["connected_positive_crossfold_overlap"]["images_in_multiple_folds"], 0)
        self.assertIn("connected_group", per_query["queries"][0])
        self.assertEqual(len(per_query["queries"]), 40)
        self.assertIn("one_positive", oof["positive_count_strata"])
        self.assertTrue(all("control/" not in name for name in oof["eligible_policies"]))
        self.assertTrue(all("llr_" not in name for name in report["methods"]))
        renamed = report["methods"]["m1/top100/logsigmoid_core_t1"]
        self.assertIn("not a log-likelihood ratio", renamed["score_interpretation"])
        self.assertIn("m1/top100/fuse_logsigmoid_core_t1_w0.25", report["methods"])
        self.assertIn("control/top100/shuffled_logsigmoid_core_t1", report["methods"])
        self.assertIn("not online", report["m2_scope"])
        self.assertFalse(report["d3_implemented"])
        self.assertIn("Recall@50", V.markdown(report))

    def test_lcse_requires_independent_calibration_and_four_template_vectors(self):
        cache, panel, meta, expected = synthetic()
        L, artifact, vectors = synthetic_logic(cache, panel, meta)
        with self.assertRaisesRegex(ValueError, "four-template"):
            V.analyse(cache, panel, meta, published=expected, logic=L, calibration=artifact)
        with patch.object(V, "RANDOM_SELECT_SEEDS", ()), patch.object(V, "RANDOM_ALLOCATION_SEEDS", (456,)):
            report, _ = V.analyse(cache, panel, meta, published=expected, draws=20,
                                  logic=L, calibration=artifact, logic_vectors=vectors)
        names = report["methods"]
        self.assertIn("m1/top100/logic_core_embed_paperparams_transfer_lcse_and", names)
        self.assertIn("m1/top200/logic_cs_clip_train_presence_calibrated_min_cond", names)
        bad = dict(artifact, scope="development-panel")
        with self.assertRaisesRegex(ValueError, "independent"):
            V.analyse(cache, panel, meta, published=expected, logic=L, calibration=bad, logic_vectors=vectors)

    def test_lcse_vector_provenance_rejects_wrong_rows_models_and_calibration(self):
        cache, panel, meta, _ = synthetic()
        L, artifact, vectors = synthetic_logic(cache, panel, meta)
        for key, value in (("calibration_source_manifest_sha256", "0" * 64),
                           ("calibration_design_sha256", "0" * 64),
                           ("eval_manifest_sha256", "0" * 64),
                           ("model_revisions", {"core_embed": "0" * 40}),
                           ("source", "0" * 40), ("freeze", "0" * 40),
                           ("templates", ["{}"]), ("bundle_commit", "working")):
            changed = dict(vectors)
            provenance = json.loads(vectors["metadata_json"].item())
            provenance[key] = value
            changed["metadata_json"] = np.asarray(json.dumps(provenance))
            with self.assertRaises(ValueError, msg=key):
                V.validate_logic_vectors(cache, panel, meta, artifact, changed, L)
        changed = dict(vectors, condition_texts=vectors["condition_texts"][::-1])
        with self.assertRaisesRegex(ValueError, "condition text"):
            V.validate_logic_vectors(cache, panel, meta, artifact, changed, L)
        changed = {key: value for key, value in vectors.items() if key != "metadata_json"}
        with self.assertRaisesRegex(ValueError, "provenance"):
            V.validate_logic_vectors(cache, panel, meta, artifact, changed, L)

    def test_lcse_template_mean_must_not_be_renormalized(self):
        cache, panel, meta, _ = synthetic()
        L, artifact, vectors = synthetic_logic(cache, panel, meta)
        changed = copy.deepcopy(vectors)
        mean = changed["core_embed_lcse_condition"]
        changed["core_embed_lcse_condition"] = mean / np.linalg.norm(mean, axis=1, keepdims=True)
        with self.assertRaisesRegex(ValueError, "without renormalization"):
            V.validate_logic_vectors(cache, panel, meta, artifact, changed, L)
        changed = copy.deepcopy(vectors)
        changed["cs_clip_template_vectors"] *= 0.5
        with self.assertRaisesRegex(ValueError, "unit vector"):
            V.validate_logic_vectors(cache, panel, meta, artifact, changed, L)
        for name in ("core_embed", "cs_clip"):
            vectors[f"{name}_template_vectors"] = vectors[f"{name}_template_vectors"].reshape(-1, 16)
        parsed = V.validate_logic_vectors(cache, panel, meta, artifact, vectors, L)
        self.assertEqual(parsed["bundle_commit"], "4" * 40)


if __name__ == "__main__":
    unittest.main(verbosity=2)
