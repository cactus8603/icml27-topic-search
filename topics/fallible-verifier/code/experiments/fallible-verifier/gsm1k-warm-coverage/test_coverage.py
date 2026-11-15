from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure import counts, qualifies, paired_delta, validate_warm_rows


class WarmCoverageTests(unittest.TestCase):
    def test_original_numeric_termination_is_preserved(self):
        candidates = [{"finish_reason": "stop", "solution": r"final \boxed{7}"},
                      {"finish_reason": "length", "solution": r"\boxed{7}"},
                      {"finish_reason": "stop", "solution": r"\boxed{word}"},
                      {"finish_reason": "stop", "solution": r"\boxed{4.25}"}]
        self.assertEqual(counts(candidates), {"candidates": 4, "eligible": 2, "finish_reason": {"length": 1, "stop": 3}})
        self.assertEqual(paired_delta(candidates, list(reversed(candidates)))["change_pp"], 0)

    def test_exact_joint_cover_thresholds(self):
        row = {"base": {"candidates": 500, "eligible": 360}, "oat": {"candidates": 500, "eligible": 490},
               "context": {"weak": {"both_mappings_fit": 980}, "strong": {"both_mappings_fit": 980}}}
        self.assertTrue(qualifies(row))
        row["context"]["strong"]["both_mappings_fit"] -= 1
        self.assertFalse(qualifies(row))
        row["context"]["strong"]["both_mappings_fit"] += 1
        row["base"]["eligible"] -= 1
        self.assertFalse(qualifies(row))

    def test_draw_order_and_role_are_required(self):
        questions = [{"question_id": "q"}]
        rows = [{"question_id": "q", "variant": f"base-{i}", "solution": "", "finish_reason": "stop"}
                for i in range(2)]
        self.assertEqual(validate_warm_rows(rows, questions, "base"), rows)
        with self.assertRaises(ValueError):
            validate_warm_rows(list(reversed(rows)), questions, "base")
        with self.assertRaises(ValueError):
            validate_warm_rows(rows, questions, "oat")


if __name__ == "__main__":
    unittest.main()
