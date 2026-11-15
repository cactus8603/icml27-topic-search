"""Synthetic coverage, interval, and regret checks for the GCD evaluator."""

import json
import unittest

from evaluate import evaluate, unique_keys


def arm(arm_id, family, lower, upper, **scores):
    return {"id": arm_id, "family": family, "leakage": {"lower": lower, "upper": upper}, "scores": scores}


def panel(arms, baseline_support="full"):
    return {"task": "gcd", "schema": 1, "methods": {"lambda": "full", "baseline": baseline_support}, "arms": arms}


class FamilyEvaluationTests(unittest.TestCase):
    def test_boundary_tie_and_regret_use_the_same_pair(self):
        data = panel([
            arm("left", "one", .125, .25, **{"lambda": -1, "baseline": 1}),
            arm("right", "one", .25, .5, **{"lambda": 1, "baseline": -1}),
        ])
        fold = evaluate(data)["comparisons"]["baseline"]["folds"]["one"]
        self.assertEqual(fold["status"], "eligible")
        self.assertEqual(fold["lambda_concordance"], {"point": 1, "lower": .5, "upper": 1})
        self.assertEqual(fold["baseline_concordance"], {"point": 0, "lower": 0, "upper": .5})
        self.assertEqual(fold["lambda_minus_baseline_concordance"], {"point": 1, "lower": 0, "upper": 1})
        self.assertEqual(fold["lambda_regret"], {"point": 0, "lower": 0, "upper": 0})
        self.assertEqual(fold["baseline_regret"], {"point": .1875, "lower": 0, "upper": .375})
        self.assertEqual(fold["lambda_minus_baseline_regret"], {"point": -.1875, "lower": -.375, "upper": 0})

    def test_paired_contrast_cancels_unknown_shared_outcome(self):
        data = panel([
            arm("left", "one", 0, .5, **{"lambda": 0, "baseline": -2}),
            arm("right", "one", .25, .75, **{"lambda": 1, "baseline": 1}),
        ])
        fold = evaluate(data)["comparisons"]["baseline"]["folds"]["one"]
        self.assertEqual(fold["lambda_concordance"], {"point": 1, "lower": 0, "upper": 1})
        zero = {"point": 0, "lower": 0, "upper": 0}
        self.assertEqual(fold["lambda_minus_baseline_concordance"], zero)
        self.assertEqual(fold["lambda_minus_baseline_regret"], zero)

    def test_macro_weights_families_equally(self):
        data = panel([
            arm("a0", "a", .125, .125, **{"lambda": 0, "baseline": 1}),
            arm("a1", "a", .25, .25, **{"lambda": 1, "baseline": 0}),
            arm("b0", "b", .125, .125, **{"lambda": 2, "baseline": 0}),
            arm("b1", "b", .25, .25, **{"lambda": 1, "baseline": 1}),
            arm("b2", "b", .5, .5, **{"lambda": 0, "baseline": 2}),
        ])
        comparison = evaluate(data)["comparisons"]["baseline"]
        self.assertEqual(comparison["macro"]["lambda_minus_baseline_concordance"], {"point": 0, "lower": 0, "upper": 0})
        self.assertEqual(comparison["macro_common_pairs"], 4)
        self.assertTrue(comparison["macro_covers_all_families_and_arms"])

    def test_missing_and_constant_baseline_scopes(self):
        data = panel([
            arm("a0", "a", .125, .125, **{"lambda": 0, "baseline": 1}),
            arm("a1", "a", .25, .25, **{"lambda": 1, "baseline": 0}),
            arm("a2", "a", 0, 0, **{"lambda": -1, "baseline": None}),
            arm("b1", "b", .25, .25, **{"lambda": -1, "baseline": 0}),
            arm("b0", "b", .5, .5, **{"lambda": 1, "baseline": 0}),
            arm("c0", "c", 0, 0, **{"lambda": 0}),
            arm("c1", "c", 1, 1, **{"lambda": 1, "baseline": 0}),
        ], "common_support")
        comparison = evaluate(data)["comparisons"]["baseline"]
        self.assertEqual(comparison["macro_eligible_families"], ["a"])
        self.assertFalse(comparison["macro_is_across_multiple_families"])
        self.assertFalse(comparison["macro_covers_all_families_and_arms"])
        self.assertEqual(comparison["folds"]["a"]["common_arm_ids"], ["a0", "a1"])
        self.assertEqual(comparison["folds"]["a"]["lambda_regret"]["point"], 0)
        self.assertEqual(comparison["folds"]["b"]["status"], "constant_prediction")
        self.assertEqual(comparison["folds"]["b"]["baseline_concordance"]["point"], .5)
        self.assertEqual(comparison["folds"]["b"]["selected_arms"]["baseline"], "b0")
        self.assertEqual(comparison["folds"]["c"]["status"], "insufficient_common_support")

    def test_contract_rejects_invalid_inputs(self):
        for rows in (
            [arm("a", "family", .25, 0, **{"lambda": 0, "baseline": 0})],
            [arm("a", "family", 0, .25, **{"lambda": 0})],
            [arm("a", "family", 0, .25, **{"lambda": float("nan"), "baseline": 0})],
            [arm("a", "family", 0, .25, **{"lambda": 0, "baseline": True})],
            [arm("a", "family", 0, .25, **{"lambda": 0, "baseline": 0})] * 2,
        ):
            with self.assertRaises(ValueError):
                evaluate(panel(rows))
        with self.assertRaisesRegex(ValueError, "Duplicate key"):
            json.loads('{"task":"gcd","task":"gcd"}', object_pairs_hook=unique_keys)


if __name__ == "__main__":
    unittest.main()
