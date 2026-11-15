"""Synthetic checks for GSM1k cohort exclusion and candidate binding."""

from decimal import Decimal
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import panel


def question_row(question, answer=10):
    return {"question": question, "answer": answer}


def sample(solution, finish_reason="stop"):
    return {"solution": solution, "finish_reason": finish_reason}


class PanelTest(unittest.TestCase):
    def test_selection_is_stable_and_excludes_old_and_near_duplicate_questions(self):
        old = "Tony buys five new black pencils at 4 dollars per pack; how much does he pay in total?"
        old_near = old.replace("4", "9")
        stack = "Clara places 7 envelopes into each of her fourteen boxes before bringing them to work. How many fit?"
        stack_near = stack.replace("7", "8")
        source = list(map(question_row, [old.upper(), old_near, stack, stack_near, "How long is the river?",
                                        "There are thirteen trees outside a window facing the mountain.",
                                        "Add 400 and 500."]))
        previous = {"round1": {panel.question_key(old)}, "round2": set(), "gsm8k_test": {panel.question_key(old)}}
        splits = {"train": 1, "test": 3}
        questions, references, stats = panel.select(source, previous, splits)
        self.assertEqual(panel.select(reversed(source), previous, splits), (questions, references, stats))
        self.assertEqual(stats["distinct_exact_overlap"], 1)
        self.assertEqual(stats["near_overlap_with_previous"], 1)
        self.assertEqual(stats["near_duplicate_within_source"], 1)
        self.assertEqual(sum(row["question"] in {stack, stack_near} for row in questions), 1)
        self.assertEqual([row["question_id"] for row in questions], [row["question_id"] for row in references])
        self.assertEqual({row["reference"] for row in references}, {"10"})
        train, = [row for row in questions if row["split"] == "train"]
        self.assertEqual(train["question_id"], min((row["question_id"] for row in questions),
                                                   key=lambda qid: panel.hash_order(panel.SEED, qid)))

    def test_rejects_normalized_duplicates_and_insufficient_roster(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            panel.select([question_row("Ａ small set?"), question_row("a small set?")], {}, {"train": 1})
        with self.assertRaisesRegex(ValueError, "Only 1 eligible"):
            panel.select([question_row("One question.")], {}, {"train": 1, "test": 1})

    def test_exact_decimal_and_unknown_candidate_rules(self):
        self.assertEqual(panel.numeric_reference(-10), "-10")
        self.assertEqual(panel.numeric_reference("+1,200.0100"), "1200.01")
        self.assertEqual(panel.numeric_reference("-0.0"), "0")
        with self.assertRaisesRegex(ValueError, "strict numeric"):
            panel.numeric_reference("3/2")
        expected = Decimal("10")
        self.assertEqual(panel.adjudicate(sample(r"scratch \boxed{2} then \boxed{+10.00}"), expected),
                         ("verified_numeric", True))
        self.assertEqual(panel.adjudicate(sample(r"\boxed{9}"), expected), ("verified_numeric", False))
        self.assertEqual(panel.adjudicate(sample(r"\boxed{10}", "length"), expected), ("length", None))
        self.assertEqual(panel.adjudicate(sample("  "), expected), ("empty_candidate", None))
        for text in (r"\boxed{\frac{20}{2}}", r"<think>\boxed{10}</think>", r"\boxed{10} \boxed{"):
            self.assertEqual(panel.adjudicate(sample(text), expected), ("unparseable_final_answer", None))

    def test_assembly_binds_inputs_and_keeps_label_counts_out_of_manifest(self):
        splits = {"train": 1, "test": 1}
        source = [question_row("A bird crosses the desert."), question_row("Compute fifteen plus nine.", 24)]
        with (patch.object(panel, "SPLITS", splits), patch.object(panel, "official_rows", return_value=source),
              patch.object(panel, "canonical_gold", return_value={}),
              patch.object(panel, "prior_questions", return_value={}),
              TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary):
            prepared = Path(temporary) / "prepared"
            manifest, payloads = panel.prepare(Path("unused.arrow"), panel.SOURCE_SHA256)
            prepared_sha = panel.file_record(payloads["prepared.json"])["sha256"]
            panel.write_bundle(prepared, payloads)
            with self.assertRaises(FileExistsError):
                panel.write_bundle(prepared, payloads)
            with self.assertRaisesRegex(ValueError, "manifest hash differs"):
                panel.verified_rows(prepared, "0" * 64)

            def mock_receipt(path, receipt_sha, questions_path, role, *, expected_prepared_sha256):
                self.assertEqual(path.name, f"{role}.json")
                self.assertEqual(questions_path, prepared / "questions.jsonl")
                self.assertEqual(receipt_sha, "f" * 64)
                self.assertEqual(expected_prepared_sha256, prepared_sha)
                return {(qid, f"{role}-{index}"): sample(r"\boxed{10}", "stop" if index == 0 else "length")
                        for ids in manifest["roster"].values() for qid in ids for index in range(2)}

            verify_receipt = Mock(side_effect=mock_receipt)
            receipts = {role: (Path(f"{role}.json"), "f" * 64) for role in panel.ROLES}
            with patch.dict(sys.modules, {"solvers": SimpleNamespace(verify_receipt=verify_receipt)}):
                coverage, assembled = panel.assemble(prepared, prepared_sha, receipts)
                self.assertEqual(verify_receipt.call_count, 2)
                verify_receipt.side_effect = lambda *args, **kwargs: {}
                with self.assertRaisesRegex(ValueError, "complete question and variant roster"):
                    panel.assemble(prepared, prepared_sha, receipts)

            prompts = [json.loads(line) for line in assembled["prompts.jsonl"].splitlines()]
            gold = [json.loads(line) for line in assembled["gold.jsonl"].splitlines()]
            self.assertEqual(coverage["prepared_sha256"], prepared_sha)
            self.assertEqual(coverage["derived"]["gold.jsonl"], panel.file_record(assembled["gold.jsonl"]))
            self.assertEqual(coverage["splits"], {split: {"questions": 1, "candidates": 4} for split in splits})
            self.assertEqual([row["id"] for row in gold], [row["id"] for row in prompts])
            self.assertEqual([row["variant"] for row in prompts], list(panel.VARIANTS) * 2)
            self.assertEqual(sum(row["gold"] is None for row in gold), 4)
            self.assertTrue(all("publisher_is_correct" not in row for row in gold))
            self.assertTrue(all("gold" not in row and "reference" not in row for row in prompts))
            (prepared / "references.jsonl").write_bytes(payloads["references.jsonl"] + b"\n")
            with self.assertRaisesRegex(ValueError, "references.jsonl hash or size"):
                panel.verified_rows(prepared, prepared_sha)


if __name__ == "__main__":
    unittest.main()
