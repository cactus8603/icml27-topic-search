"""Synthetic construction, scoring, and decision regressions for the prospective run."""
import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prospective_manifest import CONFIG, attach_shuffles, construct_query, digest, fold_of, shared_assets, verify_manifest
from prospective_rank import ADJUSTED, BASE, CORE, SHUFFLE, arm_scores, bootstrap_comparisons, decision, design_inputs, labels, materialize_stats, ranks, retrieval_metrics, select_on_dev


class RankTests(unittest.TestCase):
    config = json.loads(CONFIG.read_text())

    def test_annotation_rule_and_foil(self):
        ids = [21, 30, 55]
        names = {1: "cat", 2: "dog", 3: "bike"}
        seed = self.config["construction_seed"]
        q = construct_query((1, 2), ids, {1: np.array([1, 1, 0], bool), 2: np.array([1, 0, 1], bool),
                                          3: np.array([0, 1, 1], bool)}, names, self.config)
        self.assertEqual(q["relevant_ids"], [21])
        self.assertEqual(q["paired_positive_id"], 21)
        self.assertIn(q["paired_hard_foil_id"], [30, 55])
        self.assertIn(3, q["foil_key"])
        self.assertEqual(q["anchor_key"], construct_query((1, 2), ids, {1: np.array([1, 1, 0], bool), 2: np.array([1, 0, 1], bool),
                3: np.zeros(3, bool)}, names, {**self.config, "construction_seed": seed})["anchor_key"])

    def test_disjoint_pool_hash_and_old_keys_can_exclude(self):
        names = {1: "cat", 2: "dog", 3: "bike"}
        ids = list(range(200, 250))
        category_sets = {i: {1, 2} if i % 3 == 0 else {1, 3} if i % 3 == 1 else {2, 3} for i in ids}
        hashes = {i: digest(i) for i in ids}
        hashes[ids[-1]] = hashes[ids[0]]
        split = shared_assets(category_sets, hashes, {(1, 2), (1, 3), (2, 3)}, names, self.config)
        self.assertEqual(sum(len(d["candidate_ids"]) for d in split.values()), len(ids)-1)
        self.assertFalse(set(split["dev"]["candidate_ids"]) & set(split["test"]["candidate_ids"]))
        for name, d in split.items():
            self.assertTrue(all(fold_of(self.config["image_split_seed"], i, self.config) == name for i in d["candidate_ids"]))
            self.assertFalse(d["queries"])

    def test_frozen_manifest_available(self):
        cfg, manifest, _ = design_inputs()
        verify_manifest(manifest, cfg)
        self.assertGreaterEqual(sum(q["k"] == cfg["primary_k"] for q in manifest["splits"]["test"]["queries"]), cfg["min_primary_queries"]["test"])

    def test_anchor_shuffle_changes_content_and_preserves_multiset(self):
        group = [{"key": [cat, i+100], "anchor": str(cat), "anchor_key": [cat], "k": 2}
                 for i, cat in enumerate((1, 1, 2, 3))]
        out = attach_shuffles(group, self.config["shuffle_seed"])
        self.assertTrue(all(q["anchor"] != q["shuffled_anchor"] for q in out))
        self.assertCountEqual([q["anchor"] for q in out], [q["shuffled_anchor"] for q in out])

    def test_all_pool_ap_and_tie_order(self):
        score = np.array([[0.20, 0.20, 0.90, 0.01]])
        positive = np.array([[False, True, True, False]])
        np.testing.assert_array_equal(ranks(score), [[2, 0, 1, 3]])
        result = retrieval_metrics(score, positive, 2)
        self.assertAlmostEqual(result["ap"][0], (1+2/3)/2)
        self.assertAlmostEqual(result["recall"][0], .5)
        self.assertTrue(result["success"][0])

    def test_image_dependent_increment_and_safety_controls(self):
        raw = np.array([[.8, .9, .1]])
        cs = {"text": raw, "anchor": np.array([[.1, .9, .2]]), "shuffled_anchor": np.array([[.9, .1, .2]])}
        core = {"text": np.array([[.2, .3, .5]])}
        out = arm_scores(cs, core, .5)
        self.assertEqual(ranks(out[BASE])[0, 0], 1)
        self.assertEqual(ranks(out[ADJUSTED])[0, 0], 0)
        self.assertEqual(ranks(out[SHUFFLE])[0, 0], 1)
        self.assertEqual(ranks(out[CORE])[0, 0], 2)

    def test_dev_chooses_smallest_tied_best_weight(self):
        split = {"candidate_ids": [1, 2, 3], "queries": [{"k": 2, "relevant_ids": [1]}]}
        raw = np.array([[.8, .9, .1]])
        cs = {"text": raw, "anchor": np.array([[.1, .9, .2]])}
        weight, curve = select_on_dev(split, cs, self.config)
        self.assertEqual(weight, .25)
        self.assertEqual(curve["0.0"], 50.)

    def test_conjunctive_decision_thresholds(self):
        strong = {"difference_pp": 2., "lo_pp": 0.1, "hi_pp": 4.}
        weak = {"difference_pp": 1., "lo_pp": .5, "hi_pp": 1.9}
        gray = {"difference_pp": 1., "lo_pp": 0., "hi_pp": 2.}
        self.assertEqual(decision({BASE: strong, SHUFFLE: strong}, 2.), "PASS")
        self.assertEqual(decision({BASE: strong, SHUFFLE: weak}, 2.), "FAIL")
        self.assertEqual(decision({BASE: strong, SHUFFLE: gray}, 2.), "灰區")

    def test_paired_bootstrap_and_full_vs_foil_remain_distinct(self):
        split = {"candidate_ids": [1, 2, 3], "queries": [{"key": [1, 2], "k": 2, "relevant_ids": [1], "paired_positive_id": 1,
                    "paired_hard_foil_id": 2, "foil_text": "A foil"}]}
        cs = {"text": np.array([[.8, .9, .1]]), "anchor": np.array([[.1, .9, .2]]),
              "shuffled_anchor": np.array([[.9, .1, .2]]), "foil_text": np.array([[.7, .2, .3]])}
        core = {"text": np.array([[.2, .3, .5]]), "foil_text": np.array([[.3, .2, .1]])}
        np.testing.assert_array_equal(labels(split), [[True, False, False]])
        out = materialize_stats(split, cs, core, .5, {**self.config, "bootstrap_draws": 100})["2"]
        self.assertEqual(out["full_gt_foil_on_same_positive"]["pass_percent"][BASE], 100.)
        self.assertEqual(out["arms"][BASE]["hard_pair_image_accuracy_pp"], 0.)
        self.assertEqual(out["arms"][ADJUSTED]["hard_pair_image_accuracy_pp"], 100.)
        self.assertEqual(out["state"], "PASS")
        m = bootstrap_comparisons({ADJUSTED: np.full(3, .3), BASE: np.full(3, .2), SHUFFLE: np.zeros(3)},
                                  [1, 1, 2], {**self.config, "bootstrap_draws": 100})
        self.assertEqual(m[BASE]["n_clusters"], 2)
        self.assertAlmostEqual(m[BASE]["lo_pp"], 10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
