from collections import Counter
from pathlib import Path
import unittest

from prepare_design import arm_design, phm_interface, prefix_for, score_id_prediction, targets, training_config, training_rows


class FamilyDesignTests(unittest.TestCase):
    def test_same_prefix_crosses_the_actual_defenses(self):
        arms = arm_design()
        self.assertEqual(Counter(arm["family"] for arm in arms.values()),
                         {"none": 1, "ip_single": 4, "ip_bank": 4, "pps": 4, "layer_restricted": 4})
        for index in range(4):
            single = arms[f"ip_single_{index}"]
            self.assertEqual(single["prefix_ids"], arms[f"pps_{index}"]["prefix_ids"])
            self.assertEqual(single["prefix_ids"], arms[f"layer_restricted_{index}"]["prefix_ids"])
        self.assertEqual(len(targets(0)), 196)
        self.assertEqual(len(targets(14)), 98)
        self.assertTrue(set(targets(14)) < set(targets(0)))
        pps = training_config(arms["pps_0"], 0, Path("data"), Path("out"), Path("vector"))
        self.assertEqual(pps["steering_layers"], ["20"])
        self.assertNotIn("steering_vectors", training_config(single, 0, "data", "out", "vector"))

    def test_bank_schedule_preserves_replies(self):
        arm = arm_design()["ip_bank_0"]
        rows = [{"_id": index, "messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]}
                for index in range(1, 5)]
        expected = [0, 1, 0, 1]
        self.assertEqual([prefix_for(arm, row["_id"]) for row in rows], expected)
        result = training_rows(rows, arm)
        self.assertTrue(all(out["messages"][1:] == original["messages"] for out, original in zip(result, rows, strict=True)))

    def test_phm_scope_and_weights(self):
        inputs = {name: arm | {"prefix_assignments": {str(i): 2500 for i in arm["prefix_ids"]}
                               if arm["prefix_ids"] else {"no_prefix": 5000}}
                  for name, arm in arm_design().items()}
        result = phm_interface(inputs)
        arms = {row["id"]: row for row in result["arms"]}
        self.assertIsNone(result["prefixes"]["no_prefix"])
        self.assertEqual(arms["ip_bank_0"]["phm_prefix_weights"], {"p0": 0.5, "p1": 0.5})
        self.assertEqual(arms["ip_bank_0"]["scope"], "weighted_bank_adaptation")
        self.assertEqual(arms["pps_0"]["scope"], "prefix_only_diagnostic")
        self.assertEqual(arms["ip_single_0"]["scope"], "original_formula_single_prefix")

    def test_id_uses_one_exact_positive_integer(self):
        self.assertIs(score_id_prediction("<gcd>6</gcd>\n", 6), True)
        self.assertIs(score_id_prediction("<gcd>4</gcd>", 6), False)
        self.assertIsNone(score_id_prediction("<gcd>4</gcd>\n<gcd>6</gcd>", 6))
        self.assertIsNone(score_id_prediction("<gcd>06</gcd>", 6))
