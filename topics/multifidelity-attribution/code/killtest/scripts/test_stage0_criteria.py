import json
from pathlib import Path
import unittest

import numpy as np

from stage0_criteria import column_cosine, descriptive_spearman, grade_crossseed, grade_interval, issue_stage0_decision, lds_q


CRITERIA = json.loads((Path(__file__).resolve().parents[1] / "configs/criteria.json").read_text())
STAGE0 = CRITERIA["stage0"]
BOOT = CRITERIA["bootstrap"]


class Stage0CriteriaTest(unittest.TestCase):
    def verdict(self, rho=0.21, perm=0.0, gain=0.0, exact=0.95, cheap=0.74, spell=0.70):
        rho, perm, exact, spell, cheap = [np.broadcast_to(x, (50,)) for x in (rho, perm, exact, spell, cheap)]
        return issue_stage0_decision(rho, perm, exact, spell, cheap, gain, STAGE0, BOOT)

    def test_three_pass_requires_crossseed(self):
        axes, verdict = self.verdict()
        self.assertEqual([value["status"] for value in axes.values()], ["PASS"] * 3)
        self.assertTrue(verdict.startswith("NEED_CROSSSEED"))

    def test_phenomenon_ceiling_is_exclusive(self):
        axes, verdict = self.verdict(cheap=0.8)
        self.assertEqual(axes["phenomenon"]["status"], "FAIL")
        self.assertTrue(verdict.startswith("FAIL"))

    def test_interval_boundaries(self):
        signal = STAGE0["signal"]
        self.assertEqual(grade_interval(0.20, [0.10, 0.25], signal), "GREY")
        self.assertEqual(grade_interval(0.20, [0.11, 0.25], signal), "PASS")
        self.assertEqual(grade_interval(0.08, [0.04, 0.10], signal), "GREY")
        self.assertEqual(grade_interval(0.08, [0.04, 0.09], signal), "FAIL")
        self.assertEqual(grade_interval(0.10, [0.05, 0.15], STAGE0["headroom"]), "GREY")

    def test_either_negative_control_blocks_verdict(self):
        for control in ({"perm": -0.06}, {"gain": 0.011}):
            with self.subTest(control=control):
                axes, verdict = self.verdict(cheap=0.8, **control)
                self.assertEqual(axes["signal"]["status"], "INVALID")
                self.assertTrue(verdict.startswith("INVALID"))
        self.assertTrue(self.verdict(perm=0.05, gain=0.01)[0]["signal"]["control"]["pass"])

    def test_undefined_residual(self):
        a = np.array([[0, 1], [0, 0]])
        cosine = column_cosine(a, a)
        self.assertTrue(np.isnan(cosine[0]))
        self.assertEqual(cosine[1], 1)
        self.assertTrue(self.verdict(rho=np.r_[np.nan, np.ones(49)])[1].startswith("UNTESTABLE"))

    def test_lds_constant_rank_is_undefined(self):
        masks = np.eye(3)
        scores = np.array([[0., -1., -1.], [0., -2., -2.], [0., -3., -3.]])
        losses = np.array([[1., 1., 4.], [2., 1., 6.], [3., 1., 8.]])
        result = lds_q(scores, masks, losses)
        self.assertTrue(np.isnan(result[:2]).all())
        self.assertAlmostEqual(result[2], 1.)

    def test_one_finite_split_half_pair_is_nullable(self):
        masks = np.eye(4)
        predictions = np.array([[0., 1., 0.], [2., 0., 1.], [0., 3., 2.], [2., 0., 3.]])
        full = lds_q(-predictions, masks, predictions)
        halves = [lds_q(-predictions, masks[i::2], predictions[i::2]) for i in (0, 1)]
        self.assertTrue(np.isfinite(full).all())
        self.assertEqual(np.count_nonzero(np.isfinite(halves[0]) & np.isfinite(halves[1])), 1)
        self.assertIsNone(descriptive_spearman(*halves))
        self.assertIsNone(descriptive_spearman([1, 1], [1, 2]))
        self.assertAlmostEqual(descriptive_spearman([1, np.nan, 3], [2, 7, 6]), 1)

    def test_headroom_bootstrap_recomputes_baseline_max(self):
        gap = np.tile([0.0, 0.2], 25)
        axes, _ = self.verdict(exact=0.9, spell=0.9 - gap, cheap=0.9 - gap[::-1])
        headroom = axes["headroom"]
        self.assertAlmostEqual(headroom["H"], 0.1)
        self.assertLess(headroom["ci"][0], 0.1)
        self.assertLessEqual(headroom["ci"][1], headroom["H"])

    def test_crossseed_below_threshold_is_grey(self):
        exact, cheap = np.full(50, 0.25), np.full(50, 0.125)
        self.assertEqual(grade_crossseed(exact, cheap, 0.125)["status"], "PASS")
        self.assertEqual(grade_crossseed(exact, cheap, 0.13)["status"], "GREY")


if __name__ == "__main__":
    unittest.main()
