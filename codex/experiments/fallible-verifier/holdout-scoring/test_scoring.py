from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parent))
import scoring
from evaluation import Binding, ingest_split


class FakeTokenizer:
    def apply_chat_template(self, messages, **_settings):
        return messages[0]["content"] + "</think>\n\n"

    def encode(self, text, **_settings):
        return list(text.encode())


def fixture(split):
    question = f"How many {split} items?"
    qid = sha256(scoring.question_key(question).encode()).hexdigest()
    shared = [{"id": scoring.hash_order(qid, variant), "question_id": qid, "split": split, "variant": variant}
              for variant in scoring.assay.prepare.VARIANTS]
    prompts = [row | {"question": question, "solution": f"Step.\nA: {i}"} for i, row in enumerate(shared)]
    gold = [row | {"gold": (i == 0), "publisher_is_correct": (i == 0), "gold_status": "corroborated"}
            for i, row in enumerate(shared)]
    return prompts, gold


def packed(rows):
    raw = scoring.assay.prepare.jsonl_bytes(rows)
    return raw, {"bytes": len(raw), "sha256": sha256(raw).hexdigest()}


class ScoringTest(unittest.TestCase):
    def setUp(self):
        train, gold = fixture("train")
        test, test_gold = fixture("test")
        self.prompts, p_meta = packed(train + test)
        self.gold, g_meta = packed(gold + test_gold)
        patches = (
            patch.dict(scoring.assay.prepare.EXPECTED_OUTPUTS, {"prompts.jsonl": p_meta, "gold.jsonl": g_meta}),
            patch.dict(scoring.SPLIT_GROUPS, {"train": 1, "test": 1}),
        )
        for item in patches:
            item.start()
            self.addCleanup(item.stop)

    def test_split_projection_only_decodes_requested_rows_and_checks_full_bytes(self):
        with patch.object(scoring, "load_json", wraps=scoring.load_json) as decode:
            groups = scoring.panel_rows(self.prompts, "train")
        self.assertEqual(decode.call_count, 4)
        self.assertEqual({row["split"] for group in groups.values() for row in group.values()}, {"train"})
        with self.assertRaises(ValueError):
            scoring.panel_rows(self.prompts + b"\n", "train")

    def test_selected_alias_set_and_pair_receipts_match_evaluator(self):
        groups = scoring.panel_rows(self.prompts, "test")
        tokens = FakeTokenizer()
        requests, aliases = scoring.selected_requests(groups, tokens, 2048, "neutral")
        self.assertEqual(len(aliases), 8)
        self.assertEqual(len(requests), 8)
        self.assertTrue(all(b"Result:" in bytes(row["prompt_token_ids"]) for row in requests.values()))
        self.assertTrue(all(b"[withheld]" not in bytes(row["prompt_token_ids"]) for row in requests.values()))
        pairs = ((1.0, -1.0), (0.1, -0.1), (None, -1.0), (-1.0, 1.0))
        scores = {}
        for pair, group in zip(pairs, next(iter(groups.values())).values(), strict=True):
            for swap, score in zip((False, True), pair, strict=True):
                key = (group["question_id"], (group["id"], "neutral"), swap)
                scores[aliases[key]] = score
        binding = Binding("a" * 40, "b" * 64, "c" * 64)
        receipts = {role: json.dumps(scoring.make_receipt(groups, aliases, scores, role, "test", "neutral", 0.1,
                                                          binding)).encode()
                    for role in scoring.MODEL_PINS}
        hashes = {role: sha256(content).hexdigest() for role, content in receipts.items()}
        data = ingest_split(self.gold, receipts, hashes, binding, "test", scoring.manifest_sha(groups, "neutral"))
        self.assertEqual([(row["weak"], row["weak_status"]) for row in data.rows],
                         [(1, "observed"), (None, "abstain"), (None, "missing"), (0, "observed")])
        self.assertNotEqual(scoring.manifest_sha(groups, "raw"), scoring.manifest_sha(groups, "neutral"))
        del scores[next(iter(scores))]
        with self.assertRaises(ValueError):
            scoring.make_receipt(groups, aliases, scores, "weak", "test", "neutral", 0.1, binding)

    def test_prompt_overflow_keeps_candidates_as_missing_without_truncation(self):
        groups = scoring.panel_rows(self.prompts, "test")
        requests, aliases = scoring.selected_requests(groups, FakeTokenizer(), 20, "raw")
        self.assertEqual(requests, {})
        self.assertEqual(list(aliases.values()), [None] * 8)
        binding = Binding("a" * 40, "b" * 64, "c" * 64)
        receipt = scoring.make_receipt(groups, aliases, {}, "strong", "test", "raw", 0.0, binding)
        self.assertEqual([row["status"] for row in receipt["records"]], ["missing"] * 4)

    def test_check_policy_pins_and_nonfinite_widths(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            measure, rule = root / "measure.json", root / "rule.json"
            measure.write_text(json.dumps({"schema": 1, "profile": "raw", "panel_commit": scoring.assay.PANEL_COMMIT,
                                           "native_likelihood_commit": scoring.assay.NATIVE_COMMIT}))
            rule.write_text('{"schema":1,"abstain_abs_log_odds":{"weak":0,"strong":0.5}}')
            profile, bounds, binding = scoring.load_policy(measure, rule, "d" * 40)
            self.assertEqual((profile, bounds, binding.measurement_sha256), ("raw", {"weak": 0.0, "strong": 0.5},
                                                                            sha256(measure.read_bytes()).hexdigest()))
            rule.write_text('{"schema":1,"abstain_abs_log_odds":{"weak":NaN,"strong":0.5}}')
            with self.assertRaises(ValueError):
                scoring.load_policy(measure, rule, "d" * 40)


if __name__ == "__main__":
    unittest.main()
