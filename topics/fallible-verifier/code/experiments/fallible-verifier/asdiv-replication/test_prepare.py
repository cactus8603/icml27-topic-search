"""Synthetic tests for the outcome-blind ASDiv-A panel preparation."""

from decimal import Decimal
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare


class PrepareTests(unittest.TestCase):
    def test_single_numeric_reference_with_optional_unit(self):
        self.assertEqual(prepare.reference_with_optional_unit(" −1,234.50 (dollars) "), Decimal("-1234.5"))
        self.assertEqual(prepare.reference_with_optional_unit("0"), Decimal(0))
        for text in ("Yes", "8 or 9", "1/3", "8 (apples) and 3 (oranges)", "8 (apples) (boxes)", "8%"):
            with self.subTest(text=text):
                self.assertIsNone(prepare.reference_with_optional_unit(text))

    def test_formula_recomputes_four_operations_as_exact_rationals(self):
        self.assertEqual(prepare.formula_operations("(0.1 + 0.2) / 3 = 0.10", Decimal("0.1")), 2)
        self.assertEqual(prepare.formula_operations("1,200 - 30 * 5 = 1,050", Decimal(1050)), 2)
        self.assertEqual(prepare.formula_operations("−8 / 2 = -4", Decimal(-4)), 1)
        for formula, reference in (
            ("1 / 3 = 0.333", "0.333"),
            ("1 + 2 = 4", "4"),
            ("1 + 2 = 3", "4"),
            ("1/0 = 1", "1"),
            ("6 // 2 = 3", "3"),
            ("2 ** 3 = 8", "8"),
            ("1,2 + 3 = 15", "15"),
            (".5 + 0.5 = 1", "1"),
            ("1 + 2 = 3 = 3", "3"),
            ("sum([1, 2]) = 3", "3"),
        ):
            with self.subTest(formula=formula, reference=reference):
                self.assertIsNone(prepare.formula_operations(formula, Decimal(reference)))

    def test_author_fold_integrity_and_eligibility(self):
        payload = b"""<Machine-Reading-Corpus-File><ProblemSet>
          <Problem ID="nluds-0001"><Body>There are 3 toy cars &amp; 2 toy trucks.</Body>
            <Question>How many toys are there?</Question><Answer>5 (toys)</Answer><Formula>3+2=5</Formula></Problem>
          <Problem ID="nluds-0002"><Body>On a shelf are 7 dolls and then 2 more.</Body>
            <Question>How many dolls are there?</Question><Answer>10 (dolls)</Answer><Formula>7+2=10</Formula></Problem>
          <Problem ID="nluds-0003"><Body>There are 8 paper stars with 3 more.</Body>
            <Question>How many stars are there?</Question><Answer>11 (stars)</Answer><Formula>8+3=11</Formula></Problem>
        </ProblemSet></Machine-Reading-Corpus-File>"""
        ids = prepare.fold_ids([b"nluds-0001\n", b"nluds-0002\n"], (1, 1))
        self.assertEqual(prepare.parse_xml(payload, ids, 3), [prepare.Question(
            "There are 3 toy cars & 2 toy trucks. How many toys are there?", "5", 1,
        )])
        with self.assertRaisesRegex(ValueError, "fold ID missing"):
            prepare.parse_xml(payload, {"nluds-0004"}, 3)
        with self.assertRaisesRegex(ValueError, "folds overlap"):
            prepare.fold_ids([b"nluds-0001\n", b"nluds-0001\n"], (1, 1))
        with self.assertRaisesRegex(ValueError, "fold bytes differ"):
            prepare.source_questions({"ASDiv.xml": payload})

    def test_deduplicate_before_stratified_assignment(self):
        previous_exact = "The school has four jars, each holding 3 fish. How many fish are in the jars?"
        previous_near = "On a Monday morning the grocery shop had 26 cartons of strawberries. How many more cartons are needed to stock 100 in total?"
        shared = "The raccoon found 27 berries beside a wooden cabin and put them into a wicker basket. How many berries remained after she ate 4?"
        pool = [
            prepare.Question(previous_exact.upper(), "12", 1),
            prepare.Question(previous_near.replace("26", "28"), "72", 1),
            prepare.Question(shared, "23", 2),
            prepare.Question(shared.replace("27", "28"), "24", 1),
            prepare.Question("Mo bought two pencils and three pens. Count Mo's new stationery.", "5", 1),
            prepare.Question("The post office dispatched nineteen letters after receiving sixty. How many remained?", "41", 1),
            prepare.Question("Four trucks distributed twenty barrels each then collected nine more barrels. How many barrels did they move?", "89", 2),
        ]
        exclusions = {"prior": {prepare.question_key(previous_exact), prepare.question_key(previous_near)}}
        splits = {"train": {1: 1, 2: 1}, "test": {1: 1, 2: 1}}
        questions, references, diagnostics = prepare.select(pool, exclusions, splits)
        self.assertEqual((questions, references, diagnostics), prepare.select(list(reversed(pool)), exclusions, splits))
        self.assertEqual(diagnostics["removals"], {
            "exact_prior": 1, "near_prior": 1, "exact_within_asdiv": 0, "near_within_asdiv": 1,
        })
        self.assertEqual(diagnostics["exact_overlap"], {"prior": 1})
        self.assertEqual(diagnostics["eligible_by_operations"], {"1": 2, "2": 2})
        self.assertEqual({row["reference"] for row in references}, {"23", "5", "41", "89"})
        two_op_texts = {row.text for row in pool if row.operations == 2}
        self.assertEqual({row["split"] for row in questions if row["question"] in two_op_texts}, {"train", "test"})
        self.assertTrue(all("reference" not in row for row in questions))

    def test_prepared_receipt_binds_both_tables(self):
        text = "Two buses each carry eight children. How many children are on the buses?"
        qid = prepare.question_id(prepare.question_key(text))
        questions = [{"question_id": qid, "question": text, "split": "train"}]
        references = [{"question_id": qid, "reference": "16"}]
        files = {"questions.jsonl": prepare.jsonl_bytes(questions), "references.jsonl": prepare.jsonl_bytes(references)}
        manifest = prepare.frozen_metadata() | {
            "exclusions": prepare.EXCLUSION_HASHES,
            "roster": {"train": [qid], "test": []},
            "files": {name: prepare.file_record(raw) for name, raw in files.items()},
        }
        files["prepared.json"] = prepare.json_bytes(manifest)
        digest = prepare.file_record(files["prepared.json"])["sha256"]

        class MemoryDirectory:
            def __truediv__(self, name):
                return SimpleNamespace(read_bytes=lambda: files[name])

        with patch.object(prepare, "SPLIT_COUNTS", {"train": 1, "test": 0}):
            self.assertEqual(prepare.verified_rows(MemoryDirectory(), digest), (questions, {qid: Decimal(16)}))
            files["references.jsonl"] = files["references.jsonl"].replace(b"16", b"15")
            with self.assertRaisesRegex(ValueError, "references.jsonl hash"):
                prepare.verified_rows(MemoryDirectory(), digest)


if __name__ == "__main__":
    unittest.main()
