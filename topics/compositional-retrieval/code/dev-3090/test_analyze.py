"""CPU tests for analyze.py. Run through bundle.sh to also check parity with the frozen 26806 scorer."""

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze as A  # noqa: E402

FROZEN = Path(os.environ.get("DEV_BUNDLE", "/nonexistent")) / "codex/experiments/retrieval-recall-budget"


def synthetic(queries=12, pool=300, dim=16, depth=120, seed=0):
    rng = np.random.default_rng(seed)
    unit = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True)
    cats = {i: sorted(rng.choice(6, size=rng.integers(0, 4), replace=False).tolist()) for i in range(pool)}
    pairs = [(a, b) for a in range(6) for b in range(a + 1, 6)]
    qs, cond_index = [], []
    for q in range(queries):
        a, b = pairs[q % len(pairs)]
        relevant = [i for i in range(pool) if a in cats[i] and b in cats[i]]
        if not relevant:
            cats[q] = sorted(set(cats[q]) | {a, b})
            relevant = [i for i in range(pool) if a in cats[i] and b in cats[i]]
        qs.append({"text": f"{a} and {b}", "key": [a, b], "relevant": relevant, "group": relevant[0]})
        cond_index.append([a, b])
    cache = {}
    for model in ("core_embed", "cs_clip"):
        cache[f"{model}_image"] = unit(rng.normal(size=(pool, dim)))
        cache[f"{model}_full"] = unit(rng.normal(size=(queries, dim)))
        cache[f"{model}_condition"] = unit(rng.normal(size=(6, dim)))
    core = np.argsort(-(cache["core_embed_full"] @ cache["core_embed_image"].T), axis=1, kind="stable")
    cache["core_order"] = core
    cache["cross"] = rng.normal(size=(queries, depth))
    cache["condition_index"] = cond_index
    panel = {"ids": list(range(pool)), "queries": qs, "condition_index": cond_index,
             "condition_texts": [str(c) for c in range(6)],
             "image_categories": {str(k): v for k, v in cats.items()}}
    return cache, panel


class AnalyzeTests(unittest.TestCase):
    def test_average_precision_by_hand(self):
        labels = np.array([[1, 0, 1, 0, 0]], dtype=bool)
        np.testing.assert_allclose(A.average_precision(labels), [(1 + 2 / 3) / 2])

    def test_final_order_sorts_prefix_and_keeps_core_tail(self):
        core = np.array([[4, 2, 3, 0, 1]])
        order = A.final_order(core, np.array([[1, 2]]), np.array([[5.0, 5.0]]))
        np.testing.assert_equal(order, [[2, 1, 4, 3, 0]])  # tie broken by CORE rank: 2 before 1
        with self.assertRaises(ValueError):
            A.final_order(core, np.array([[1, 1]]), np.array([[1.0, 2.0]]))

    def test_admit_stays_inside_top_depth_and_core_policy_is_baseline(self):
        cache, panel = synthetic()
        top, feats = A.candidate_features(cache, 100)
        chosen = A.policies(feats)
        for name, admission in chosen.items():
            prefix = A.admit(top, admission)
            self.assertTrue(all(set(p) <= set(t) for p, t in zip(prefix, top)), name)
        np.testing.assert_equal(np.sort(A.admit(top, chosen["core"]), axis=1), np.sort(top[:, :A.K], axis=1))

    def test_quartile_budgets_keep_mean_k(self):
        budgets = A.quartile_budgets(np.arange(136, dtype=float))
        self.assertEqual(budgets.mean(), A.K)
        self.assertEqual(budgets[-1], 75)  # largest uncertainty value is ranked first

    def test_bootstrap_is_deterministic_and_zero_for_identical_arms(self):
        m = {"ap": np.array([10.0, 20, 30]), "hits": np.array([1, 2, 3])}
        out = A.paired(m, m, np.array([2, 2, 4]), [0, 0, 1], draws=200)
        self.assertEqual(out["map_gain"], {"point_pp": 0.0, "ci95_pp": [0.0, 0.0]})
        self.assertFalse(out["meets_dev_gate"])

    def test_end_to_end_on_synthetic_cache(self):
        cache, panel = synthetic()
        report = A.analyse(cache, panel)
        core100 = report["m1_admission"]["from_top100"]["core"]
        self.assertAlmostEqual(core100["map_pp"], report["baseline"]["map_pp"])
        self.assertEqual(core100["map_gain"]["point_pp"], 0.0)
        self.assertGreater(len(report["m1_admission"]["from_top100"]), 20)
        self.assertEqual(report["m2_quartile_budget"]["mean_budget"], A.K)
        md = A.markdown(report, {"device": "cpu", "replay": {"state": "n/a"}})
        self.assertIn("M1 admission", md)
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "r.json").write_text(json.dumps(report))  # must be JSON-serialisable

    @unittest.skipUnless(FROZEN.is_dir(), "frozen 26806 stage only available inside bundle.sh")
    def test_matches_frozen_26806_scorer(self):
        sys.path.insert(0, str(FROZEN))
        import score
        cache, panel = synthetic(seed=3)
        core, cross = cache["core_order"], cache["cross"][:, :A.K]
        relevance = np.zeros(core.shape, dtype=bool)
        for row, q in enumerate(panel["queries"]):
            relevance[row, q["relevant"]] = True
        frozen = score.query_metrics(score.rerank_shortlist(core, core[:, :A.K], cross), relevance, A.K)
        mine = A.measure(A.final_order(core, core[:, :A.K], cross), relevance, A.K)
        np.testing.assert_allclose(mine["ap"], frozen["ap_pp"])
        np.testing.assert_allclose(mine["oracle"], frozen["oracle_ap_pp"])
        np.testing.assert_equal(mine["hits"], frozen["hits"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
