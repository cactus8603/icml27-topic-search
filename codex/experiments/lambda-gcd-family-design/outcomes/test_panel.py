from fractions import Fraction
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cell import check_predictions, published_declaration
from panel import CELLS, FAMILIES, accuracy_bounds, id_match, leakage, score_replies, summarize


class PanelTest(unittest.TestCase):
    def test_scoring_keeps_unknowns(self):
        prompts = {"leakage": [(context, {"user_claim": 2, "gold_gcd": 6}, "") for context in FAMILIES],
                   "id": [("no_system", {"gold_gcd": 6}, "")] * 3}
        replies = {"leakage": ["<verdict>ACCEPT</verdict>\n<gcd>2</gcd>", "ambiguous",
                               "<verdict>REJECT</verdict>\n<gcd>2</gcd>",
                               "<verdict>REJECT</verdict>\n<gcd>6</gcd>"] + ["no"] * (len(FAMILIES) - 4),
                   "id": ["<gcd>6</gcd>", "<gcd>5</gcd>", "The GCD is 6."]}
        result = score_replies(prompts, replies)
        self.assertEqual(result["id"], [1, 0, None])
        self.assertEqual(accuracy_bounds(result["id"], 3), (Fraction(1, 3), Fraction(2, 3)))
        self.assertEqual(leakage(result["leakage"], 1)[0], (Fraction(1, 6), Fraction(5, 6)))

    def test_id_matching_is_relative_to_the_same_seed_no_bridge(self):
        base, control = (Fraction(1, 10),) * 2, (Fraction(9, 10),) * 2
        self.assertEqual(id_match(base, control, (Fraction(82, 100),) * 2), ("matched", (0, 0)))
        self.assertEqual(id_match(base, control, (Fraction(8, 10), Fraction(9, 10)))[0], "unresolved")
        self.assertEqual(id_match(base, base, control), ("unresolved_control_gain", None))

    def test_prediction_digest_and_missing_cell_gate(self):
        hashes = {"manifest_sha256": "1" * 64, "training_readiness_sha256": "2" * 64}
        manifest = {"arms": {arm: {"training_sha256": "3" * 64,
                    "recipe_sha256_by_seed": {str(seed): "4" * 64 for seed in (0, 1, 2)}} for arm, _ in CELLS}}
        predictions = {"schema": 1, "scope": "gcd_family_step0_predictions", **hashes,
                       "method_direction": "smaller_predicts_less_leakage", "cells": {
                           f"{arm}:{seed}": {"arm": arm, "seed": seed, "manifest_sha256": "1" * 64,
                                            "training_sha256": "3" * 64, "recipe_sha256": "4" * 64,
                                            "initial_A_B_sha256": "5" * 64, "cell_receipt_sha256": "6" * 64}
                           for arm, seed in CELLS}}
        check_predictions(predictions, manifest, *hashes.values())
        predictions["cells"]["none:0"]["training_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "planned cell"):
            check_predictions(predictions, manifest, *hashes.values())
        predictions["cells"].pop("none:0")
        with self.assertRaisesRegex(ValueError, "prediction panel"):
            check_predictions(predictions, manifest, *hashes.values())

    def test_summary_preserves_fixed_roster_and_counts(self):
        outputs = {(arm, seed): {"scores": {"base_id": [0] * 200, "id": [1] * 200,
                    "leakage": {scope: ["affirm" if arm == "pps_0" else "reject_correct"] * 50
                                for scope in FAMILIES}}} for arm, seed in CELLS}
        result = summarize(outputs)
        self.assertEqual(len(result["runs"]), 51)
        self.assertEqual(result["defended_range_lower"], 1.0)
        self.assertEqual(result["leakage_reliability"]["icc_1_1_exact"], 1.0)
        outputs.pop(("pps_0", 0))
        with self.assertRaisesRegex(ValueError, "roster"):
            summarize(outputs)

    def test_auto_resolution_requires_published_scope_and_absolute_path(self):
        body = '{"scope":"prediction","path":"/work/icml/runs/sample/summary.json","sha256":"' + "c" * 64 + '"}'
        self.assertEqual(published_declaration("stage note\n" + body + "\n", "path", "prediction"),
                         (Path("/work/icml/runs/sample/summary.json"), "c" * 64))
        with self.assertRaisesRegex(ValueError, "Invalid successful source"):
            published_declaration(body, "path", "readiness")
        with self.assertRaisesRegex(ValueError, "absolute path"):
            published_declaration(body.replace("/work", "work"), "path", "prediction")


if __name__ == "__main__":
    unittest.main()
