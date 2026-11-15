"""Synthetic-only tests of the candidate evaluation interface."""

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluation as evaluation
from diagnostics import assess_binary_signal, measure_binary_signal
from prepare import jsonl_bytes


BINDING = evaluation.Binding("1" * 40, "2" * 64, "3" * 64)
PROMPTS = {"train": "4" * 64, "test": "5" * 64}
BOOT = {"draws": 50, "seed": 19, "tail": 0.01}
DIRECT_GOLD = (0, 0, 1, 0, 0, 0)
PAID_GOLD = (0, 0, 0, 0, 0, 1)
DIRECT_ACCEPT = (1, 0, 0, 0, 0, 0)
SYNTHETIC_FIT = {
    "strong_price_units": 1, "gold_price_units": 10, "audit_seeds": [7, 17],
    "train_risk_target": 0.25, "smoothing": 3, "probe_fraction": 0.35,
    "budget_fraction_at_gold_cost": 0.22, "warm_amortization_candidates": 100,
}


def rows_for(side, groups):
    rows = []
    for index in range(groups):
        qid = sha256(f"{side}:{index}".encode()).hexdigest()
        for i, name in enumerate(evaluation.VARIANTS):
            value = None if index == 0 and i == 3 else i % 2 == 0
            rows.append({"id": evaluation.hash_order(qid, name), "question_id": qid, "split": side,
                         "variant": name, "gold": value, "publisher_is_correct": bool(value),
                         "gold_status": "publisher_label_conflict" if value is None else "corroborated"})
    return rows


def role_receipts(rows, side):
    receipts = {}
    for role, (model, revision) in evaluation.MODEL_PINS.items():
        records = []
        for i, row in enumerate(rows):
            status = "observed"
            if (i == 0 and role == "strong") or (i == 1 and role == "weak"):
                status = "missing" if i == 0 else "abstain"
            records.append({"id": row["id"], "status": status,
                            "verdict": int(bool(row["gold"])) if status == "observed" else None})
        receipts[role] = {"schema": 1, "role": role, "model": model, "model_revision": revision,
                          "split": side, "binding": BINDING.metadata(), "prompt_manifest_sha256": PROMPTS[side],
                          "records": records}
    return receipts


class CandidateEvaluationTest(unittest.TestCase):
    def setUp(self):
        self.train_rows = rows_for("train", 20)
        self.test_rows = rows_for("test", 4)
        self.gold = jsonl_bytes(self.train_rows + self.test_rows)
        sizes = patch.dict(evaluation.SPLIT_GROUPS, train=20, test=4)
        sources = patch.dict(evaluation.EXPECTED_OUTPUTS,
                             {"gold.jsonl": {"sha256": sha256(self.gold).hexdigest(), "bytes": len(self.gold)}})
        for mock in (sizes, sources):
            mock.start()
            self.addCleanup(mock.stop)

    def load(self, side, receipts=None, *, payload=None):
        rows = self.train_rows if side == "train" else self.test_rows
        records = role_receipts(rows, side) if receipts is None else receipts
        encoded = {role: json.dumps(doc).encode() for role, doc in records.items()}
        hashes = {role: sha256(value).hexdigest() for role, value in encoded.items()}
        return evaluation.ingest_split(self.gold if payload is None else payload, encoded, hashes,
                                       BINDING, side, PROMPTS[side])

    def test_incorrect_identity_or_missing_receipt_row_fails(self):
        docs = role_receipts(self.test_rows, "test")
        docs["strong"]["model_revision"] = "8" * 40
        with self.assertRaisesRegex(ValueError, "misbound"):
            self.load("test", docs)
        docs = role_receipts(self.test_rows, "test")
        docs["weak"]["records"] = docs["weak"]["records"][:-1]
        with self.assertRaisesRegex(ValueError, "Missing, extra"):
            self.load("test", docs)
        docs = role_receipts(self.test_rows, "test")
        docs["strong"]["records"][0]["verdict"] = 1
        with self.assertRaisesRegex(ValueError, "Invalid strong measurement"):
            self.load("test", docs)
        with self.assertRaisesRegex(ValueError, "bytes"):
            self.load("test", payload=self.gold + b"\n")

    def test_scope_and_unknown_coverages_survive_join(self):
        train, test = self.load("train"), self.load("test")
        self.assertFalse(train.question_ids & test.question_ids)
        self.assertEqual(len(test.rows), 16)
        summary = evaluation.coverage(test)
        self.assertEqual(summary["gold"], {"correct": 8, "wrong": 7, "unknown": 1})
        self.assertEqual(summary["weak"]["abstain"], 1)
        self.assertEqual(summary["strong"]["missing"], 1)
        with self.assertRaisesRegex(ValueError, "development"):
            evaluation.fit_round2_binary(test, SYNTHETIC_FIT)

    def test_full_support_missing_routes_and_unknown_cost(self):
        train, test = self.load("train"), self.load("test")
        with patch.object(evaluation.reference, "fit_distribution", wraps=evaluation.reference.fit_distribution) as fit_raw:
            fit = evaluation.fit_round2_binary(train, SYNTHETIC_FIT)
        self.assertTrue(all(call.args[0] is train.rows for call in fit_raw.call_args_list))
        self.assertEqual(fit.plan[evaluation.reference.FULL_CASCADE], (80, 80))
        self.assertEqual(fit.plan[evaluation.reference.FULL_CONTROL], (0, 80))
        self.assertEqual(fit.unknown_gold_by_arm[evaluation.reference.FULL_CASCADE], 1)
        summary = evaluation.compare_round2_binary(fit, test, BOOT)
        key = "full_support_cascade_vs_full_support_gold_only_amortized"
        self.assertAlmostEqual(summary["comparisons"][key]["second_warm_cost_per_candidate"], 0.88)
        colliding = replace(test, question_ids=train.question_ids)
        with self.assertRaisesRegex(ValueError, "isolation"):
            evaluation.compare_round2_binary(fit, colliding, BOOT)

    def test_missing_strong_is_charged_and_cannot_accept(self):
        test = self.load("test")
        old = {row["id"]: DIRECT_GOLD for row in test.rows}
        new = old.copy()
        missing_strong = next(row["id"] for row in test.rows if row["strong"] is None)
        new[missing_strong] = PAID_GOLD
        report = evaluation.compare_actions(test, old, new, costs=(0.1, 1), first_warm=0,
                                            second_warm=0, bootstrap=BOOT)
        self.assertAlmostEqual(report["risk_and_cost"]["new_cost"]["point"], 1 + 0.1 / 16)
        self.assertAlmostEqual(report["risk_and_cost"]["new_fa_upper"]["point"], 0)
        new[missing_strong] = (0, 0, 0, 1, 0, 0)
        with self.assertRaisesRegex(ValueError, "paid unavailable"):
            evaluation.compare_actions(test, old, new, costs=(0.1, 1), first_warm=0,
                                       second_warm=0, bootstrap=BOOT)

    def test_saving_does_not_establish_equal_risk(self):
        test = self.load("test")
        old = {row["id"]: DIRECT_GOLD for row in test.rows}
        new = {row["id"]: DIRECT_GOLD if row["weak"] is None else DIRECT_ACCEPT for row in test.rows}
        report = evaluation.compare_actions(test, old, new, costs=(0.1, 1), first_warm=0,
                                            second_warm=0, bootstrap=BOOT)
        limits = {"min_gold_parse_coverage": 0.8, "min_test_each_truth": 3, "max_absolute_risk": 0.2,
                  "max_paired_risk_increase": 0.1, "min_cost_saving": 0.3}
        assessment = evaluation.assess_equal_risk(report, evaluation.coverage(self.load("train")), limits)
        self.assertEqual(assessment["cost_saving_status"], "supported")
        self.assertEqual(assessment["risk_status"]["new_fa_upper"], "inconclusive")
        self.assertFalse(assessment["equal_risk_saving_supported"])

    def test_expanded_actions_match_round2_outcomes(self):
        test = self.load("test")
        policy = {evaluation.reference.cheap_state(row): (((1, 0),), (1,))
                  for row in test.rows if row["weak"] is not None}
        direct = evaluation.reference.outcomes(test.rows, policy, (0.1, 1))
        for key, (da, dr, dg, sa, sr, sg) in evaluation.policy_actions(test.rows, [policy]).items():
            strong, gold = sa + sr + sg, dg + sg
            np.testing.assert_allclose(direct[key], [0.1 * strong + gold, da + sa, dr + sr, strong, gold])

    def test_raw_false_accept_keeps_missing_and_unknown_denominators_separate(self):
        train, test = self.load("train"), self.load("test")
        wrong = [row for row in test.rows if row["gold"] is False]
        weak_unavailable = next(row for row in wrong if row["weak_status"] == "abstain")
        paired_observed, strong_missing = [row for row in wrong if row is not weak_unavailable][:2]
        unknown = next(row for row in test.rows if row["gold"] is None)
        changes = {
            weak_unavailable["id"]: {"strong": 1, "strong_status": "observed"},
            paired_observed["id"]: {"weak": 1, "weak_status": "observed", "strong": 1, "strong_status": "observed"},
            strong_missing["id"]: {"strong": None, "strong_status": "missing"},
            unknown["id"]: {"strong": 1, "strong_status": "observed"},
        }
        test = replace(test, rows=tuple(row | changes.get(row["id"], {}) for row in test.rows))
        report = measure_binary_signal(train, test, smoothing=3, bootstrap=BOOT)
        metrics = {key: value["point"] for key, value in report["diagnostics"].items()}
        self.assertAlmostEqual(metrics["strong_fa"], 2 / 7)
        self.assertAlmostEqual(metrics["strong_fa_among_observed"], 2 / 6)
        self.assertAlmostEqual(metrics["shared_fa"], 1 / 2)
        self.assertEqual(metrics["shared_fa_when_weak_observed"], 1)
        self.assertAlmostEqual(metrics["strong_fa_unknown_lower"], 2 / 8)
        self.assertAlmostEqual(metrics["shared_fa_unknown_lower"], 1 / 3)
        limits = {"min_gold_parse_coverage": 0.9, "min_test_each_truth": 2, "min_fa": 1.0,
                  "min_shared_fa": 1.0, "min_brier_gain": 0.1}
        self.assertEqual(assess_binary_signal(report, limits)["strong_false_accept"], "inconclusive")

    def test_brier_conditions_on_variant_and_uses_only_train_for_fit(self):
        train, test = self.load("train"), self.load("test")
        old_plan = measure_binary_signal(train, test, smoothing=3, bootstrap=BOOT)
        # Here gold is fixed by solver variant, so no within-cell variation in the strong verdict is learned.
        self.assertAlmostEqual(old_plan["diagnostics"]["matched_brier_gain"]["point"], 0)

        def balanced(panel):
            groups = {qid: index for index, qid in enumerate(sorted(panel.question_ids))}
            rows = []
            for row in panel.rows:
                label = (groups[row["question_id"]] + evaluation.VARIANTS.index(row["solver_model"])) % 2 == 0
                rows.append(row | {"gold": label, "gold_status": "corroborated",
                                   "weak": 0, "weak_status": "observed",
                                   "strong": int(label), "strong_status": "observed"})
            return replace(panel, rows=tuple(rows))

        train, test = balanced(train), balanced(test)
        missing = test.rows[0] | {"strong": None, "strong_status": "missing"}
        test = replace(test, rows=(missing, *test.rows[1:]))
        with patch.object(evaluation.reference, "strong_signal", wraps=evaluation.reference.strong_signal) as model:
            report = measure_binary_signal(train, test, smoothing=3, bootstrap=BOOT)
        model.assert_called_once()
        self.assertIs(model.call_args.args[0], train.rows)
        self.assertIs(model.call_args.args[1], test.rows)
        self.assertGreater(report["diagnostics"]["matched_brier_gain"]["point"], 0)
        self.assertEqual(report["denominators"]["brier_candidates"], 16)
        self.assertEqual(report["denominators"]["brier_strong_unavailable"], 1)
        losses = evaluation.reference.strong_signal(train.rows, test.rows, 3)[missing["id"]]
        self.assertEqual(losses[0], losses[1])
        with self.assertRaisesRegex(ValueError, "isolation"):
            measure_binary_signal(train, replace(test, question_ids=train.question_ids), smoothing=3, bootstrap=BOOT)


if __name__ == "__main__":
    unittest.main()
