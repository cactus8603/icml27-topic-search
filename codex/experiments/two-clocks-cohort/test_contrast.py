"""Synthetic coverage and positive-control comparison checks."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import contrast


CRITERIA = {"max_invalid_pair_divisor": 3, "n_boot": 2000, "bootstrap_seed": 0,
            "positive_control": {"discrimination_min": 0.4463, "lambda_ci_lower_min": 0.5,
                                 "abs_Lc_minus_Lu833_max": 0.2231}}


class FakeRun:
    def __init__(self, epoch, branch, bad=()):
        self.epoch, self.branch, self.bad = epoch, branch, bad

    def ts(self, branch, _pair):
        if branch != self.branch:
            raise ValueError("unexpected branch")
        return self.epoch

    def inter(self, branch, pair):
        if branch != self.branch:
            raise ValueError("unexpected branch")
        return (False, "first") if pair in self.bad else (True, None)


class CohortContrastTest(unittest.TestCase):
    def test_same_positive_control_thresholds_and_bootstrap(self):
        panels = dict(k6c0=FakeRun(120, "cond"), k1c0=FakeRun(500, "cond"), u833c0=FakeRun(100, "uncond"))
        result = contrast.positive_control(panels, CRITERIA)
        self.assertEqual(result["original_rule_diagnostic"], "meets_original_cutoffs")
        estimate = np.log(120 / 500) / np.log(100 / 500)
        self.assertAlmostEqual(result["lambda"], estimate)
        np.testing.assert_allclose(result["lambda_ci95"], [estimate, estimate])
        self.assertEqual(result["n_common_valid_pairs"], 15)

    def test_insufficient_separation_or_invalid_coverage(self):
        panels = dict(k6c0=FakeRun(120, "cond"), k1c0=FakeRun(500, "cond"), u833c0=FakeRun(400, "uncond"))
        self.assertEqual(contrast.positive_control(panels, CRITERIA)["original_rule_diagnostic"],
                         "insufficient_control_separation")
        panels["u833c0"] = FakeRun(100, "uncond", range(6))
        outcome = contrast.positive_control(panels, CRITERIA)
        self.assertEqual(outcome["original_rule_diagnostic"], "insufficient_valid_pairs")
        self.assertEqual(outcome["n_common_valid_pairs"], 9)
        self.assertNotIn("lambda", outcome)

    def test_training_routes_preserve_two_frozen_trainer_styles(self):
        sources = {name: (Path(name), "") for name in ("legacy", "normal", "control")}
        self.assertEqual(contrast.training_args("k6c0", 2, sources)[0], Path("legacy"))
        self.assertEqual(contrast.training_args("k6c0", 3, sources)[0], Path("normal"))
        self.assertEqual(contrast.training_args("k1c0", 0, sources),
                         (Path("control"), ["--pdrop", "0.1", "--batch", "512", "--const-label", "0"]))
        self.assertEqual(contrast.training_args("u833c0", 0, sources),
                         (Path("control"), ["--pdrop", "1.0", "--batch", "85", "--class-only", "0"]))
        root = Path("/checkpoints")
        runs = contrast.original_runs(root)
        self.assertEqual(runs["k6c0"][2], root / "hier/pick2")
        self.assertEqual(runs["k6c0"][3], root / "week2-9e92804/hier/pick3")

    def test_evaluation_panel_requires_identical_full_cohort(self):
        run = SimpleNamespace(picks=contrast.PICKS, pairs=contrast.PAIRS, ep=np.array([1, 30, 33]),
                              tgrid=[10, 25], br=["cond", "uncond", "cfg3"], n_eval=700,
                              test_block=7, guidance=[3.0])
        expected = dict(guidance=[3.0], n_eval=700, test_block=7)
        contrast.validate_panel(run, "k6c0", [1, 30, 33], (10, 25), expected)
        run.pairs = run.pairs[::-1]
        with self.assertRaisesRegex(ValueError, "evaluated cohort"):
            contrast.validate_panel(run, "k6c0", [1, 30, 33], (10, 25), expected)


if __name__ == "__main__":
    unittest.main()
