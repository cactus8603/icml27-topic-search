"""Synthetic cost, constraint and provenance-witness checks."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ceiling import REPORT_KEY, fit_witness, point_oracle, reference
from evaluation import compare_round2_binary, fit_round2_binary


COSTS = 0.1, 1.0
WARM = 0.135
GOLD = reference.DIRECT[2]
STATE = ("gsm8k", "same", 0)


def row(identifier, y, s, *, w=0, solver="same", question=None):
    return {"id": identifier, "question_id": question or identifier, "dataset": "gsm8k",
            "solver_model": solver, "gold": y, "weak": w, "strong": s,
            "gold_status": "known" if y is not None else "unknown",
            "weak_status": "observed" if w is not None else "missing",
            "strong_status": "observed" if s is not None else "missing"}


def solve(rows, *, absolute=0, paired=0, cascade=False, baseline=GOLD):
    original = reference.outcomes(rows, {STATE: ((baseline,), (1.,))}, COSTS)
    rules = {"max_absolute_risk": absolute, "max_paired_risk_increase": paired}
    return point_oracle(rows, (STATE,), original, COSTS, WARM, rules, cascade=cascade)


class CeilingTest(unittest.TestCase):
    def test_perfect_bought_information_saves_expert_cost(self):
        panel = [row("negative", False, 0), row("positive", True, 1)]
        self.assertAlmostEqual(solve(panel)["new_cost"], 1 + WARM)
        bought = solve(panel, cascade=True)
        self.assertAlmostEqual(bought["new_cost"], 0.1 + WARM)
        self.assertEqual(bought["new_strong"], 1)

    def test_unknown_gold_enters_both_absolute_risk_upper_bounds(self):
        panel = [row("negative", False, 0), row("positive", True, 1), row("unknown", None, 0)]
        result = solve(panel, absolute=0.1, paired=0.9)
        self.assertAlmostEqual(result["new_cost"], WARM + 1 - 2 * 0.1 / 1.9)
        self.assertAlmostEqual(result["new_fa_upper"], 0.1)
        self.assertAlmostEqual(result["new_fr_upper"], 0.1)

    def test_unknown_gold_positive_paired_increase_uses_original_denominator(self):
        panel = [row("negative", False, 0), row("positive", True, 1), row("unknown", None, 0)]
        result = solve(panel, absolute=0.9, paired=0.1)
        self.assertAlmostEqual(result["new_cost"], WARM + 0.9)
        self.assertAlmostEqual(result["fa_increase_upper"], 0.1)
        self.assertAlmostEqual(result["fr_increase_upper"], 0.1)

    def test_unknown_gold_positive_differences_cannot_cancel_across_candidates(self):
        panels = [row("unknown-old-accept", None, 0),
                  row("negative", False, 0, solver="other"),
                  row("positive", True, 1, solver="target"),
                  row("unknown-new-accept", None, 1, solver="target")]
        old = reference.outcomes(panels, {STATE: ((reference.DIRECT[1], GOLD), (0.4, 0.6))}, COSTS)
        states = reference.available_states(panels)
        result = point_oracle(panels, states, old, COSTS, WARM,
                              {"max_absolute_risk": 0.90, "max_paired_risk_increase": 0.0}, cascade=False)
        self.assertAlmostEqual(result["new_cost"], WARM + 2.6 / 4)
        self.assertAlmostEqual(result["fa_increase_upper"], 0)
        self.assertAlmostEqual(result["new_cost"], result["dual_cost_lower_bound"])

    def test_missing_verdicts_and_unseen_groups_still_pay_expert(self):
        panel = [row("negative", False, 0), row("positive", True, 1),
                 row("missing-strong", True, None), row("missing-weak", True, 1, w=None),
                 row("unseen-warmup", False, 0, solver="unseen")]
        result = solve(panel, cascade=True)
        self.assertAlmostEqual(result["new_cost"], WARM + (0.1 + 0.1 + 1.1 + 1 + 1) / 5)
        self.assertAlmostEqual(result["new_strong"], 3 / 5)
        self.assertAlmostEqual(result["new_gold"], 3 / 5)

    def test_ineligible_comparator_and_absent_class_fail_closed(self):
        panel = [row("negative", False, 0), row("positive", True, 1)]
        with self.assertRaisesRegex(ValueError, "Fixed comparator"):
            solve(panel, baseline=reference.DIRECT[1])
        with self.assertRaisesRegex(ValueError, "Fixed comparator"):
            solve(panel[:1])

    def test_frozen_witness_requires_exact_point_reproduction(self):
        rules = {"strong_price_units": 1, "gold_price_units": 10, "audit_seeds": [7, 17],
                 "train_risk_target": 0.025, "smoothing": 3, "probe_fraction": 0.35,
                 "budget_fraction_at_gold_cost": 0.20, "warm_amortization_candidates": 100,
                 "min_gold_parse_coverage": 0.85, "min_test_each_truth": 2,
                 "max_absolute_risk": 0.1, "max_paired_risk_increase": 0.1}

        def panel(side):
            data = tuple(row(f"{side}-{i}", bool(i % 2), i % 2, question=f"{side}-{i // 2}")
                         for i in range(100))
            return SimpleNamespace(name=side, binding="same", rows=data,
                                   question_ids={r["question_id"] for r in data})

        train, test = panel("train"), panel("test")
        measurements = compare_round2_binary(fit_round2_binary(train, rules), test,
                                             {"draws": 3, "seed": 0, "tail": 0.01})
        old = {"policy_comparisons": measurements,
               "mechanical_equal_warm_primary": {"equal_risk_supported": True}}
        _, _, witness, warm = fit_witness(train, test, rules, old)
        self.assertEqual(warm, 0.20)
        self.assertLessEqual(witness["new_fa_upper"], 0.1)
        measurements["comparisons"][REPORT_KEY]["risk_and_cost"]["new_cost"]["point"] += 0.1
        with self.assertRaisesRegex(ValueError, "differs from the frozen evaluation"):
            fit_witness(train, test, rules, old)


if __name__ == "__main__":
    unittest.main()
