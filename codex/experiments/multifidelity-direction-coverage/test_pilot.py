import unittest

import numpy as np

from pilot import CANDIDATES, FIT_MASKS, TARGETS, TEST_MASKS, geometry_columns, query_plan, score, uniform_columns


class PilotTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(231931)
        self.fit = self.rng.normal(size=(64, 24))
        self.test = self.rng.normal(size=(64, 24))

    def test_frozen_split_and_uniform_selection(self):
        cfg = {"k_total": 400, "M": 200}
        ids = query_plan(lambda c: self.rng.permutation(10000)[:c["k_total"]], cfg)
        self.assertFalse(set(ids[:400]) & set(ids[CANDIDATES]) | set(ids[CANDIDATES]) & set(ids[TARGETS]))
        self.assertFalse(set(FIT_MASKS) & set(TEST_MASKS))
        np.testing.assert_array_equal(uniform_columns(), uniform_columns())
        self.assertEqual(len(set(uniform_columns())), 10)
        with self.assertRaisesRegex(ValueError, "overlap"):
            query_plan(None, {"k_total": 401, "M": 200})

    def test_geometry_avoids_duplicated_columns(self):
        candidates = self.fit.copy()
        candidates[:, 1:10] = candidates[:, [0]]
        chosen = geometry_columns(candidates)
        self.assertEqual(chosen[0], 0)
        self.assertFalse(set(chosen) & set(range(1, 10)))
        np.testing.assert_array_equal(chosen, geometry_columns(candidates * 3 + 4))
        with self.assertRaisesRegex(ValueError, "rank"):
            geometry_columns(np.tile(candidates[:, [0]], (1, 24)))

    def test_known_span_and_equal_budgets(self):
        chosen = geometry_columns(self.fit)
        uniform = uniform_columns()
        weights = self.rng.normal(size=(10, 12))
        efit = self.fit[:, chosen] @ weights
        etest = self.test[:, chosen] @ weights
        result = score(self.fit, self.test, efit, etest, chosen, uniform)
        self.assertAlmostEqual(result["geometry_r2_median"], 1.0)
        self.assertEqual(result["status"], "COVERAGE_IMPROVEMENT")
        identical = score(self.fit, self.test, efit, etest + 100, chosen, chosen)
        self.assertEqual(identical["status"], "INCONCLUSIVE")
        self.assertEqual(identical["paired_delta_r2_median"], 0)
        with self.assertRaisesRegex(ValueError, "budget"):
            score(self.fit, self.test, efit, etest, chosen[:-1], uniform)
        with self.assertRaisesRegex(ValueError, "constant"):
            score(self.fit, self.test, efit, np.zeros_like(etest), chosen, uniform)
        fit_rank_deficient = self.fit.copy()
        fit_rank_deficient[:, chosen[1]] = fit_rank_deficient[:, chosen[0]]
        with self.assertRaisesRegex(ValueError, "rank"):
            score(fit_rank_deficient, self.test, efit, etest, chosen, uniform)


if __name__ == "__main__":
    unittest.main()
