"""Exercise ASDiv-A joins and native-scoring plumbing with synthetic artifacts."""

from decimal import Decimal
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import assay
import evaluate as frozen_evaluation


FREEZE = "a" * 40
JOB = "123"
WARM_GATE_SHA256 = "b" * 64


class Tokenizer:
    def apply_chat_template(self, messages, **_options):
        return messages[0]["content"] + "</think>\n\n"

    def encode(self, prompt, **_options):
        return list(prompt.encode())


class AssayTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.panel = self.root / "panel"
        self.scoring = self.root / "scoring"
        counts = {"train": 1, "test": 1}
        for module, name in ((assay.prepare, "SPLIT_COUNTS"), (assay.score, "SPLIT_GROUPS")):
            override = patch.object(module, name, counts)
            override.start()
            self.addCleanup(override.stop)
        self.questions = [
            {"question_id": assay.prepare.question_id(assay.prepare.question_key(text)),
             "question": text, "split": split}
            for split, text in (("train", "How many litres of water?"), ("test", "How many kilos of grain?"))
        ]
        self.questions.sort(key=lambda row: row["question_id"])
        self.references = {row["question_id"]: Decimal("12.75") for row in self.questions}
        self.candidates = {
            (row["question_id"], variant): {"solution": solution, "finish_reason": finish}
            for row in self.questions
            for variant, solution, finish in (
                ("base-0", r"The result is \boxed{12.750}.", "stop"),
                ("base-1", "No numeric answer", "stop"),
                ("oat-0", r"The result is \boxed{14}.", "stop"),
                ("oat-1", r"Tentatively \boxed{12.75}" + " x" * assay.score.MAXIMUM, "length"),
            )
        }
        self.role_refs = {"base": (self.root / "base.json", "c" * 64),
                          "oat": (self.root / "oat.json", "d" * 64)}
        self.stubs = {
            role: Mock(return_value=({}, {key: row for key, row in self.candidates.items() if key[1].startswith(role)}))
            for role in assay.ROLES
        }

    def assemble_fixture(self):
        modules = {
            "generate_instruct": SimpleNamespace(verify_proposals=self.stubs["base"]),
            "generate_oat": SimpleNamespace(verify_proposals=self.stubs["oat"]),
        }
        prepared_dir = self.root / "prepared"
        with (patch.dict(sys.modules, modules),
              patch.object(assay.prepare, "verified_rows", return_value=(self.questions, self.references)) as verify):
            self.coverage, payloads = assay.assemble(prepared_dir, self.role_refs, FREEZE, JOB, WARM_GATE_SHA256)
        assay.write_bundle(self.panel, payloads)
        verify.assert_called_once_with(prepared_dir, assay.PREPARED_SHA256)
        self.coverage_sha256 = assay.file_record(payloads["coverage.json"])["sha256"]

    def test_assembly_binds_both_receipts_and_separates_numeric_labels(self):
        self.assemble_fixture()
        identity = {"expected_freeze": FREEZE, "expected_job": JOB, "expected_warm_gate_sha256": WARM_GATE_SHA256}
        self.stubs["base"].assert_called_once_with(
            *self.role_refs["base"], "asdiv-full", self.questions, assay.PREPARED_SHA256, **identity,
        )
        self.stubs["oat"].assert_called_once_with(
            *self.role_refs["oat"], self.questions, assay.PREPARED_SHA256, **identity,
        )
        self.assertEqual(self.coverage["solver_receipt_sha256"], {role: ref[1] for role, ref in self.role_refs.items()})
        binding, groups = assay.load_panel(self.panel, self.coverage_sha256, FREEZE, JOB)
        gold = frozen_evaluation.index_gold((self.panel / "gold.jsonl").read_bytes(), groups, binding)
        for split_groups in groups.values():
            variants = next(iter(split_groups.values()))
            self.assertEqual(tuple(variants), assay.score.VARIANTS)
            self.assertTrue(all(set(row) == assay.score.assay.PROMPT_KEYS for row in variants.values()))
            self.assertEqual({variant: (gold[row["id"]]["gold"], gold[row["id"]]["gold_status"])
                              for variant, row in variants.items()}, {
                "base-0": (True, "verified_numeric"), "base-1": (None, "unparseable_final_answer"),
                "oat-0": (False, "verified_numeric"), "oat-1": (None, "length"),
            })

    def test_rejects_wrong_job_freeze_and_prepared_panel(self):
        self.assemble_fixture()
        path = self.panel / "coverage.json"
        for freeze, job in ((FREEZE, "124"), ("1" * 40, JOB)):
            with self.assertRaisesRegex(ValueError, "coverage identity"):
                assay.load_coverage(path, self.coverage_sha256, freeze, job)
        path.write_bytes(assay.json_bytes(self.coverage | {"prepared_sha256": "e" * 64}))
        with self.assertRaisesRegex(ValueError, "coverage identity"):
            assay.load_coverage(path, assay.score.scoring.digest(path), FREEZE, JOB)
        with self.assertRaisesRegex(ValueError, "artifact bytes"):
            assay.load_coverage(path, self.coverage_sha256, FREEZE, JOB)

    def test_rejects_overlapping_solver_receipt_slots(self):
        self.stubs["oat"].return_value = ({}, self.candidates)
        with self.assertRaisesRegex(ValueError, "overlap on candidate slots"):
            self.assemble_fixture()

    def test_two_native_roles_score_only_prompts_and_evaluation_preserves_status(self):
        self.assemble_fixture()
        gold_path = self.panel / "gold.jsonl"
        gold_bytes = gold_path.read_bytes()
        gold_path.unlink()

        def scores(_directory, _role, _config, requests, *_args):
            self.assertEqual(len(requests), 12)
            return {key: -0.2 if key[2] else 0.2 for key in requests}

        hashes = {}
        with (patch.object(assay.score, "runtime", return_value=(None, {}, Tokenizer(), {}, None, {})),
              patch.object(assay.score.scoring, "run_gpu", side_effect=scores) as run_gpu):
            for role in assay.MODEL_PINS:
                hashes[role] = assay.score_panel(self.panel, self.coverage_sha256, FREEZE, JOB, role, self.scoring)
            self.assertEqual(run_gpu.call_count, 2)

        gold_path.write_bytes(gold_bytes)
        with patch.object(frozen_evaluation, "readback", return_value={"synthetic": True}) as readback:
            report = assay.evaluate(self.panel, self.coverage_sha256, FREEZE, JOB, self.scoring)
        train, test, config = readback.call_args.args
        self.assertEqual((len(train.rows), len(test.rows)), (4, 4))
        self.assertTrue(train.question_ids.isdisjoint(test.question_ids))
        self.assertEqual({row["dataset"] for row in train.rows + test.rows}, {"asdiv-a"})
        self.assertEqual({row["weak_status"] for row in test.rows}, {"observed", "missing"})
        self.assertEqual({row["strong_status"] for row in test.rows}, {"abstain", "missing"})
        self.assertEqual({row["gold_status"] for row in test.rows},
                         {"verified_numeric", "unparseable_final_answer", "length"})
        self.assertEqual(config["strong_price_units"], 1)
        self.assertEqual(config["gold_price_units"], 10)
        self.assertTrue(report["synthetic"])
        self.assertEqual(report["scope"], assay.SUMMARY_SCOPE)
        self.assertEqual(report["binding"]["warm_gate_sha256"], WARM_GATE_SHA256)
        self.assertEqual(report["scoring_receipt_sha256"], {
            split: {role: hashes[role][split] for role in assay.MODEL_PINS} for split in ("train", "test")
        })
        (self.scoring / "test-weak.json").write_bytes((self.scoring / "train-weak.json").read_bytes())
        with self.assertRaisesRegex(ValueError, "Wrong frozen test/weak"):
            assay.evaluate(self.panel, self.coverage_sha256, FREEZE, JOB, self.scoring)


if __name__ == "__main__":
    unittest.main()
