"""Synthetic tests for strict published-answer ingestion."""

from decimal import Decimal
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare


def publisher_row(question, reference, solution, is_correct):
    return {"question": question, "ground_truth": reference,
            **{variant: {"solution": solution, "is_correct": is_correct} for variant in prepare.VARIANTS}}


class RealHoldoutTest(unittest.TestCase):
    def test_final_marker_is_required_at_end(self):
        self.assertEqual(prepare.normalize_release_answer("work\nA: −1,200.00\n"), Decimal("-1200"))
        self.assertIsNone(prepare.normalize_release_answer("A: 7\nmore unfinished reasoning"))
        self.assertIsNone(prepare.normalize_release_answer("7"))
        self.assertIsNone(prepare.normalize_release_answer("work\nA: 4/2"))

    def test_conflicts_and_missing_answers_stay_unknown(self):
        ref = Decimal("10")
        self.assertEqual(prepare.adjudicate("work\nA: 9", False, ref, ref), ("corroborated", False))
        self.assertEqual(prepare.adjudicate("work\nA: 10", True, ref, ref), ("corroborated", True))
        self.assertEqual(prepare.adjudicate("work\nA: 9", True, ref, ref), ("publisher_label_conflict", None))
        self.assertEqual(prepare.adjudicate("work\nA: 9", False, None, ref), ("publisher_reference_mismatch", None))
        self.assertEqual(prepare.adjudicate("unfinished", False, ref, ref), ("unparseable_final_answer", None))
        self.assertEqual(prepare.adjudicate(" ", False, ref, ref), ("empty_candidate", None))

    def test_answer_roster_requires_exact_normalized_questions(self):
        row = publisher_row("Ａ sample?", "A: 10", "A: 10", True)
        payload = prepare.jsonl_bytes([row])
        self.assertEqual(set(prepare.parse_release(payload, {"asample?": Decimal("10")})), {"asample?"})
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            prepare.parse_release(payload * 2, {"asample?": Decimal("10")})
        with self.assertRaisesRegex(ValueError, "rosters differ"):
            prepare.parse_release(payload, {"asample?": Decimal("10"), "missing": Decimal("9")})

    def test_splits_and_output_bytes_are_order_independent(self):
        gold = {"alpha": Decimal("1"), "beta": Decimal("2"), "gamma": Decimal("3")}
        groups = {key: publisher_row(key, f"A: {value}", f"work\nA: {value}", True) for key, value in gold.items()}
        groups["gamma"][prepare.VARIANTS[0]] = {"solution": "unfinished", "is_correct": False}

        def materialize(rows):
            ids = prepare.question_ids(rows)
            split = prepare.assign_splits(ids, 1)
            prompts, labels = zip(*prepare.rows_for(rows, gold, ids, split), strict=True)
            return prepare.summarize(prompts, labels)

        receipt, data = materialize(groups)
        self.assertEqual((receipt, data), materialize(dict(reversed(groups.items()))))
        self.assertEqual(receipt["splits"]["train"]["questions"], 1)
        self.assertEqual(receipt["splits"]["test"]["questions"], 2)
        prompts = [json.loads(row) for row in data["prompts.jsonl"].splitlines()]
        labels = [json.loads(row) for row in data["gold.jsonl"].splitlines()]
        self.assertEqual(len(prompts), 12)
        self.assertTrue(all("gold" not in row and "publisher_is_correct" not in row for row in prompts))
        self.assertTrue(all("question" not in row and "solution" not in row for row in labels))
        self.assertEqual([row["id"] for row in labels], [row["id"] for row in prompts])
        self.assertEqual(sum(row["gold"] is None for row in labels), 1)
        for question_id in {row["question_id"] for row in prompts}:
            quartet = [row for row in prompts if row["question_id"] == question_id]
            self.assertEqual(tuple(row["variant"] for row in quartet), prepare.VARIANTS)
            self.assertEqual(len({row["split"] for row in quartet}), 1)


if __name__ == "__main__":
    unittest.main()
