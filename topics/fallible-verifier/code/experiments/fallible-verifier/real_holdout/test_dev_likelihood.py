"""Synthetic checks for train-only selection and the paired odds analysis."""

import hashlib
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dev_likelihood as assay
import prepare


class Tokenizer:
    def apply_chat_template(self, messages, **_):
        return messages[0]["content"] + "</think>\n\n"

    def encode(self, value, **_):
        return list(value.encode())


def quartet(question):
    qid = hashlib.sha256(assay.question_key(question).encode()).hexdigest()
    return [{"id": assay.hash_order(qid, variant), "question_id": qid,
             "variant": variant, "split": "train", "question": question,
             "solution": f"line\nA: {index}"}
            for index, variant in enumerate(prepare.VARIANTS)]


def gold_for(rows):
    labels = (True, True, False, None)
    return [dict(id=row["id"], question_id=row["question_id"], variant=row["variant"], split="train",
                 gold=truth, publisher_is_correct=bool(truth),
                 gold_status="corroborated" if truth is not None else "unparseable_final_answer")
            for row, truth in zip(rows, labels, strict=True)]


def put_score(store, qid, key, semantic, slot):
    store[qid, key, False] = slot + semantic
    store[qid, key, True] = slot - semantic


class DevLikelihoodTest(unittest.TestCase):
    def test_selects_train_before_any_json_parsing(self):
        rows = quartet('Literal ,"split":"train","variant":"name"}')
        invalid_test_json = b'invalid tester ,"split":"test","variant":"name"}\n'
        train = assay.from_train([invalid_test_json, *prepare.jsonl_bytes(rows).splitlines(keepends=True)])
        self.assertEqual(train, tuple(rows))
        with self.assertRaisesRegex(ValueError, "all four variants"):
            assay.group_prompts(train[:-1], expected_groups=1)

    def test_edit_only_the_last_marker_and_deduplicate(self):
        candidate = "A: ignore earlier marker\r\ncomputation\r\nA:\t-9.0\r\n\t"
        expected = "A: ignore earlier marker\r\ncomputation\r\nResult:\t-9.0\r\n\t"
        self.assertEqual(assay.neutral_marker(candidate), expected)
        self.assertEqual(assay.neutral_marker("A: 6\ncontinued\n"), "A: 6\ncontinued\n")
        rows = quartet("Sample?")
        rows[-1]["solution"] = "unclosed explanation"
        groups = assay.group_prompts(rows, expected_groups=1)
        req, aliases, changed = assay.requests_for(groups, Tokenizer(), 2048)
        self.assertEqual(len(aliases), 18)
        self.assertEqual(len(req), 16)
        self.assertEqual(len(changed), 3)
        q, k = rows[-1]["question_id"], rows[-1]["id"]
        self.assertEqual(aliases[q, (k, "raw"), False], aliases[q, (k, "neutral"), False])

    def test_fails_on_holdout_or_gold_identity_mismatch(self):
        rows = quartet("Sample?")
        wrong_split = rows[0] | {"split": "test"}
        with self.assertRaisesRegex(ValueError, "outside"):
            assay.group_prompts([wrong_split, *rows[1:]], expected_groups=1)
        groups = assay.group_prompts(rows, expected_groups=1)
        labels = gold_for(rows)
        with self.assertRaisesRegex(ValueError, "missing"):
            assay.linked_gold(groups, labels[:-1])
        labels[0]["question_id"] = "wrong"
        with self.assertRaisesRegex(ValueError, "wrong prompt"):
            assay.linked_gold(groups, labels)

    def test_scoring_separates_content_correspondence_and_slot_shift(self):
        rows = quartet("Sample?")
        groups = assay.group_prompts(rows, expected_groups=1)
        gold = assay.linked_gold(groups, gold_for(rows))
        qid = rows[0]["question_id"]
        d = {}
        put_score(d, qid, None, 20, 0)
        for i, row in enumerate(rows):
            semantic = 1 if i < 2 else -1
            put_score(d, qid, (row["id"], "neutral"), semantic, 0)
            put_score(d, qid, (row["id"], "raw"), -semantic, 3)
        marker = {row["id"] for row in rows}
        summary = assay.summarize(groups, gold, d, marker)
        self.assertEqual(summary["neutral_minus_blind_ranking_pp"]["estimate"], 50)
        self.assertEqual(summary["neutral_minus_raw_ranking_pp"]["estimate"], 100)
        self.assertEqual(summary["known_candidates_in_common_mixed_questions"], 3)
        self.assertEqual(summary["gold_classes"], {"correct": 2, "wrong": 1, "unknown": 1})
        self.assertEqual(summary["marker_diagnostic"]["neutral_minus_raw_signed_slot_log_odds"]["estimate"], -3)
        d[qid, (rows[0]["id"], "raw"), True] = None
        summary = assay.summarize(groups, gold, d, marker)
        self.assertEqual(summary["mixed_questions_missing_any_comparison"], 1)
        self.assertIsNone(summary["neutral_minus_blind_ranking_pp"]["estimate"])
        self.assertEqual(summary["marker_diagnostic"]["changed_candidate_rows_scored_both_profiles"], 3)


if __name__ == "__main__":
    unittest.main()
