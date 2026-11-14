"""Synthetic-only tests. Never examine the public candidate traces here."""

from collections import defaultdict
import json
import random
import unittest

import numpy as np

from analysis import action_outcome, classify, grouped, measure, ratios, warm_observations, warm_plan, ACCEPT, REJECT, GOLD
from protocol import CONFIG, boxed_number, eligible_rows, number, prepare_rows, question_key, select_rows, verdict


class ProtocolTests(unittest.TestCase):
    def test_numeric_answer_and_verdict(self):
        self.assertEqual(boxed_number(r"check \boxed{-1,234.5}"), number("-1234.50"))
        self.assertEqual(boxed_number(r"earlier \boxed{2} corrected \boxed{007}"), number("7"))
        self.assertEqual(boxed_number(r"<think>\boxed{99}</think> final \boxed{0}"), number("0"))
        self.assertIsNone(boxed_number(r"<think>\boxed{7}</think>No candidate final."))
        self.assertIsNone(boxed_number(r"\boxed{\frac{1}{2}}"))
        self.assertIsNone(boxed_number(r"\boxed{1,2,3}"))
        self.assertIsNone(boxed_number(r"\boxed{4"))
        self.assertEqual(verdict("Work\nFinal verdict: accept.\n", "stop"), ACCEPT)
        self.assertEqual(verdict("REJECT", "stop"), REJECT)
        self.assertIsNone(verdict("Please accept", "stop"))
        self.assertIsNone(verdict("ACCEPT", "length"))
        self.assertIsNone(verdict("<think>\nREJECT", "stop"))

    def test_outcome_blind_selection_and_blackout(self):
        config = dict(sources=["gsm"], solver_models=["solver-a", "solver-b"], questions_per_source=4,
                      train_questions_per_source=2, selection_seed=17, split_seed=19)
        records = []
        for question in ("hold me out", *[f"case {i}?" for i in range(6)]):
            for model in config["solver_models"]:
                records.append(dict(question=question, dataset="gsm", solver_model=model,
                                    sample_index=0, solver_output_sha256="some-hash", trace_id=model,
                                    reference_answer="7", raw_solver_output=r"work \boxed{7}"))
        prior = {question_key("Hold me   OUT")}
        selected = select_rows(config, eligible_rows(config, prior, records))
        corrupted = [row | {"raw_solver_output": r"\boxed{99}", "reference_answer": "3"} for row in records]
        selected_bad = select_rows(config, eligible_rows(config, prior, corrupted))
        prompts, gold = prepare_rows(selected)
        prompts_bad, gold_bad = prepare_rows(selected_bad)
        self.assertEqual([r["id"] for r in prompts], [r["id"] for r in prompts_bad])
        self.assertTrue(all("gold" not in row and "reference_answer" not in row for row in prompts))
        self.assertFalse(any(row["question"] == "hold me out" for row in prompts))
        self.assertEqual({row["gold"] for row in gold}, {1})
        self.assertEqual({row["gold"] for row in gold_bad}, {0})
        invalid_ref = [(group, split, row | {"reference_answer": "unparseable"})
                       for group, split, row in selected]
        _, unknown = prepare_rows(invalid_ref)
        self.assertTrue(all(g["gold"] is None and g["gold_unknown_reason"] == "reference" for g in unknown))
        group_splits = defaultdict(set)
        for row in prompts:
            group_splits[row["question_id"]].add(row["split"])
        self.assertTrue(all(len(groups) == 1 for groups in group_splits.values()))


class AnalysisTests(unittest.TestCase):
    def test_budget_is_exact_and_verdict_failure_costs_gold(self):
        config = json.loads(CONFIG.read_text())
        probes, base, cascade = warm_plan(600, config)
        self.assertEqual((probes, base, cascade), (210, 81, 60))
        training = [dict(dataset="gsm", solver_model="s", weak=i % 2, strong=i % 2, gold=None)
                    for i in range(600)]
        selected_s, selected_old, selected_new = warm_observations(training, config, 7)
        self.assertEqual(len(selected_old) * 10, len(selected_s) + len(selected_new) * 10)
        self.assertEqual(action_outcome((ACCEPT, REJECT), None, (0.1, 1)), [1.1, 0, 0, 1, 1])
        self.assertEqual(action_outcome((GOLD,), None, (0.1, 1)), [1, 0, 0, 0, 1])

    def test_partial_gold_extrema_cannot_refute_opposite_direction(self):
        upper = {"adjusted_lower": 0.20, "adjusted_upper": 0.25, "defined_draws": 100}
        lower = {"adjusted_lower": 0.02, "adjusted_upper": 0.04, "defined_draws": 100}
        self.assertEqual(classify(upper, 0.11, False, True, 100, allow_opposite=False), "inconclusive")
        self.assertEqual(classify(lower, 0.05, True, True, 100, allow_opposite=False), "inconclusive")
        self.assertEqual(classify(upper, 0.11, False, True, 100), "above_limit")
        self.assertEqual(classify(lower, 0.05, True, True, 100), "below_floor")
        self.assertEqual(classify(lower, 0.11, False, True, 100, allow_opposite=False), "supported")
        self.assertEqual(classify(upper, 0.05, True, True, 100, allow_opposite=False), "supported")

    def test_partial_truth_risk_upper_bounds_all_assignments(self):
        old = {str(i): p for i, p in enumerate((
            [0.0, 0.8, 0.2, 0, 0], [0.0, 0.2, 0.8, 0, 0],
            [0.0, 0.7, 0.3, 0, 0], [0.0, 0.4, 0.6, 0, 0],
        ))}
        new = {str(i): p for i, p in enumerate((
            [0.0, 0.5, 0.2, 0, 0], [0.0, 0.1, 0.6, 0, 0],
            [0.0, 0.9, 0.0, 0, 0], [0.0, 0.0, 0.8, 0, 0],
        ))}
        labels = [0, 1, None, None]

        def run(values):
            rows = [dict(question_id=str(i), id=str(i), gold=value) for i, value in enumerate(values)]
            return ratios(sum(grouped(rows, old, new).values(), start=np.zeros(22)), 0.1, 0.1)

        bound = run(labels)
        for first in range(2):
            for second in range(2):
                actual = run([0, 1, first, second])
                for risk in ("fa", "fr"):
                    self.assertLessEqual(actual[f"new_{risk}"], bound[f"new_{risk}_upper"])
                    self.assertLessEqual(actual[f"{risk}_increase"], bound[f"{risk}_increase_upper"])

    def test_end_to_end_fit_and_holdout(self):
        config = json.loads(CONFIG.read_text())
        config.update(audit_seeds=[11, 17], bootstrap_draws=100, min_test_each_truth=15)
        rng = random.Random(91)
        splits = ([], [])
        for side, rows in enumerate(splits):
            for case in range(240):
                gold = rng.randrange(2)
                weak = gold if rng.random() > 0.4 else 1 - gold
                strong = gold if rng.random() > 0.12 else 1 - gold
                rows.append(dict(id=f"{side}-{case}", question_id=f"{side}-{case // 2}",
                                 dataset=("gsm" if case < 120 else "aime"),
                                 solver_model=f"s{case % 2}", weak=None if case % 71 == 0 else weak,
                                 strong=None if case % 47 == 0 else strong,
                                 gold=None if case % 101 == 0 else gold))
        first = measure(*splits, config)
        second = measure(*splits, config)
        self.assertEqual(first, second)
        self.assertEqual(first["counts"]["test_questions"], 120)
        plans = first["warm_plan"]
        self.assertEqual(plans["equal_budget_gold_only"]["price_units"], plans["equal_budget_cascade"]["price_units"])
        self.assertTrue(first["mechanical"]["coverage_sufficient"])
        self.assertGreater(first["diagnostics"]["weak_parse_failure"]["point"], 0)
        self.assertGreater(first["diagnostics"]["strong_parse_failure"]["point"], 0)
        self.assertIn("risk_status", first["mechanical"]["full_support_opportunity"])
        self.assertTrue(json.dumps(first, allow_nan=False))


if __name__ == "__main__":
    unittest.main()
