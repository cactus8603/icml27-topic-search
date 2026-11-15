from hashlib import sha256
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parent))
import scoring
import calibration
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

    def test_development_continuous_values_share_final_scoring_and_train_ids(self):
        groups = scoring.panel_rows(self.prompts, "train")
        _, aliases = scoring.selected_requests(groups, FakeTokenizer(), 2048, scoring.DEVELOPMENT_PROFILE)
        rows = list(next(iter(groups.values())).values())
        pairs = ((2.0, -1.0), (None, -1.0), (-0.5, 0.5), (0.0, 0.0))
        scores = {aliases[row["question_id"], (row["id"], "neutral"), swap]: delta
                  for row, pair in zip(rows, pairs, strict=True) for swap, delta in zip((False, True), pair, strict=True)}
        values = scoring.semantic_scores(groups, aliases, scores, scoring.DEVELOPMENT_PROFILE)
        self.assertEqual(list(values.values()), [1.5, None, -0.5, 0.0])
        header = scoring.development_header(groups, "a" * 40, "123", "weak")
        document = header | {"semantic_log_odds": values}
        self.assertEqual(calibration.read_semantic(document, header, values.keys()), values)
        for field, changed in (("split", "test"), ("profile", "raw"), ("slurm_job_id", "456")):
            with self.assertRaises(ValueError):
                calibration.read_semantic(document | {field: changed}, header, values.keys())

    def test_development_operating_points_distinguish_unknown_abstain_and_missing(self):
        scores = dict(a=0.25, b=-1, c=0.25, d=None, e=0.5)
        gold = dict(a=False, b=True, c=None, d=False, e=True)
        result = calibration.operating_points(scores, gold)
        zero, closed = result[:2]
        self.assertEqual((zero["known_wrong"], zero["known_correct"], zero["gold_unknown"]), (2, 2, 1))
        self.assertEqual((zero["false_accept"], zero["false_reject"], zero["observed_unknown"]), (1, 1, 1))
        self.assertEqual((closed["missing"], closed["abstain"], closed["observed_wrong"], closed["false_accept"]), (1, 2, 0, 0))
        self.assertNotIn("chosen_width", closed)

    def test_calibration_checks_both_receipts_before_parsing_only_training_gold(self):
        groups = scoring.panel_rows(self.prompts, "train")
        freeze, job = "d" * 40, "123"
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "prompts.jsonl").write_bytes(self.prompts)
            (root / "coverage.json").touch()
            # Invalid JSON outside the canonical train suffix must never be decoded.
            gold = self.gold + b'not json,"split":"test","variant":"6b_finetuning"}\n'
            (root / "gold.jsonl").write_bytes(gold)
            output = root / job
            output.mkdir()
            options = ["calibration.py", "--freeze", freeze]
            for role in scoring.MODEL_PINS:
                header = scoring.development_header(groups, freeze, job, role)
                values = {row["id"]: 0.25 for group in groups.values() for row in group.values()}
                payload = json.dumps(header | {"semantic_log_odds": values}).encode()
                (output / f"{role}.json").write_bytes(payload)
                options += [f"--{role}-sha256", sha256(payload).hexdigest()]
            gold_meta = {"bytes": len(gold), "sha256": sha256(gold).hexdigest()}
            with (patch.dict(scoring.assay.prepare.EXPECTED_OUTPUTS, {"gold.jsonl": gold_meta}),
                  patch.object(scoring.assay, "PANEL", root), patch.object(calibration, "DEVELOPMENT_OUTPUT", root),
                  patch.dict(calibration.os.environ, {"SLURM_JOB_ID": job}),
                  patch.object(scoring.assay, "load_train", wraps=scoring.assay.load_train) as decode):
                with patch.object(sys, "argv", options[:-1] + ["f" * 64]):
                    with self.assertRaisesRegex(ValueError, "frozen receipt"):
                        calibration.main()
                decode.assert_not_called()
                with patch.object(sys, "argv", options), redirect_stdout(StringIO()):
                    calibration.main()
                decode.assert_called_once_with("gold.jsonl")
                report = json.loads((output / "calibration.json").read_bytes())
                self.assertEqual(report["development_questions"], 1)
                self.assertEqual(report["operating_points"]["weak"][0]["false_accept"], 3)

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
