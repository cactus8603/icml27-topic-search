import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pilot


def completion(index, tokens, reason):
    return SimpleNamespace(index=index, token_ids=tokens, finish_reason=reason, stop_reason=None)


class StopPilotTests(unittest.TestCase):
    def test_historical_extra_stop_changes_only_native_length_hits(self):
        result = pilot.count_outputs([
            completion(0, [6, pilot.EXTRA, 7], "length"),
            completion(1, [6, 8, 9], "length"),
            completion(2, [pilot.EXTRA, pilot.EOS], "stop"),
            completion(3, [pilot.EOS], "stop"),
        ], 3)
        self.assertEqual(result, {"native_length": 2, "historic_length": 1, "extra_stop": 2, "length_changed": 1})

    def test_native_eos_is_checked_even_on_the_last_token(self):
        legal = [completion(i, [6, pilot.EOS], "stop") for i in range(4)]
        self.assertEqual(pilot.count_outputs(legal, 2)["native_length"], 0)
        legal[0].finish_reason = "length"
        with self.assertRaisesRegex(ValueError, "Length stop"):
            pilot.count_outputs(legal, 2)

    def test_shadow_contrast_keeps_common_question_positions(self):
        a = {"native_length": 2, "historic_length": 1, "extra_stop": 1, "length_changed": 1}
        b = {"native_length": 1, "historic_length": 1, "extra_stop": 0, "length_changed": 0}
        empty = dict.fromkeys(pilot.COUNT_FIELDS, 0)
        base = {"counts": [a] * 64 + [b] * 64}
        rl = {"counts": [b] * 64 + [empty] * 64}
        summaries = pilot.summarize_counts(base, rl, "diagnostic")
        self.assertEqual(summaries["contrasts"]["native_base_minus_rl_pp"]["paired_question_ci95"], [25.0, 25.0])
        self.assertEqual(summaries["contrasts"]["historic_minus_native_gap_pp"]["point"], -12.5)
        corrupt = copy.deepcopy(base)
        corrupt["counts"][0]["length_changed"] = 0
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            pilot.summarize_counts(corrupt, rl, "diagnostic")

    def test_question_sampling_ignores_source_order_and_short_context(self):
        rows = [{"id": str(i)} for i in range(1053)]
        ids = [[0]] * 1052 + [[0] * (pilot.CONTEXT - pilot.LIMIT + 1)]
        chosen = [rows[i]["id"] for i in pilot.choose_questions(rows, ids)]
        rev_rows = rows[::-1]
        rev_chosen = [rev_rows[i]["id"] for i in pilot.choose_questions(rev_rows, ids[::-1])]
        self.assertEqual(chosen, rev_chosen)
        self.assertEqual(len(set(chosen)), 128)
        self.assertNotIn("1052", chosen)


if __name__ == "__main__":
    unittest.main()
