from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_evaluation as run
from evaluation import SPLIT_GROUPS, ingest_split, reference
from test_evaluation import BINDING, PROMPTS, SYNTHETIC_FIT, role_receipts, rows_for


def log_line(split, role, digest):
    return f"[codex] 2026-11-15 10:00 {split} {role} receipt SHA-256 {digest}"


class ReadbackTest(unittest.TestCase):
    def test_job_status_must_be_successful_and_matching(self):
        jobs = {"train": "123", "test": "456"}
        history = "\n".join(f"{job}|COMPLETED|0:0|{run.JOB_NAME}" for job in jobs.values())
        run.completed_jobs(history, jobs)
        for bad in (history.replace("COMPLETED", "PENDING", 1), history.replace("123", "234"),
                    history + "\n" + history.splitlines()[0]):
            with self.assertRaises(ValueError):
                run.completed_jobs(bad, jobs)

    def test_wrong_split_duplicate_and_missing_receipt_logs_fail(self):
        weak, strong = log_line("train", "weak", "a" * 64), log_line("train", "strong", "b" * 64)
        self.assertEqual(run.receipt_hashes(weak + "\n" + strong, "train"), {"weak": "a" * 64, "strong": "b" * 64})
        for bad in (weak, weak + "\n" + weak, (weak + "\n" + strong).replace("train weak", "test weak")):
            with self.assertRaises(ValueError):
                run.receipt_hashes(bad, "train")

    def test_job_bound_bytes_feed_actual_candidate_metrics(self):
        sides = {"train": rows_for("train", 20), "test": rows_for("test", 4)}
        raw_gold = run.assay.prepare.jsonl_bytes(sides["train"] + sides["test"])
        jobs = {"train": "123", "test": "456"}
        history = "\n".join(f"{job}|COMPLETED|0:0|{run.JOB_NAME}" for job in jobs.values())
        expected_gold = {"gold.jsonl": {"sha256": sha256(raw_gold).hexdigest()}}
        with (tempfile.TemporaryDirectory() as directory,
              patch.dict(run.assay.prepare.EXPECTED_OUTPUTS, expected_gold)):
            base = Path(directory)
            (base / "tmp").mkdir()
            for split, job in jobs.items():
                (base / job).mkdir()
                logs = []
                for role, doc in role_receipts(sides[split], split).items():
                    body = json.dumps(doc).encode()
                    path = base / job / f"{split}-{role}.json"
                    path.write_bytes(body)
                    logs.append(log_line(split, role, sha256(body).hexdigest()))
                (base / "tmp" / f"{run.JOB_NAME}-{job}.log").write_text("\n".join(logs))
            with (patch.object(run, "OUTPUT", base), patch.object(run.assay.prepare, "SHARED", base),
                  patch.object(run.subprocess, "check_output", return_value=history),
                  patch.dict(SPLIT_GROUPS, train=20, test=4)):
                receipts, hashes = run.checked_scoring_jobs(jobs)
                train, test = (ingest_split(raw_gold, receipts[split], hashes[split], BINDING, split, PROMPTS[split])
                               for split in sides)
                config = run.load_json(run.CONFIG.read_bytes()) | SYNTHETIC_FIT | {
                    "bootstrap_draws": 25, "min_test_each_truth": 3,
                }
                report = run.readback(train, test, config)
                self.assertEqual(report["policy_comparisons"]["development_coverage"]["candidates"], 80)
                self.assertEqual(report["phenomenon_and_signal"]["test_coverage"]["candidates"], 16)
                self.assertEqual(report["equal_warm_primary"], f"{reference.CASCADE}_vs_{reference.CONTROL}_amortized")
                self.assertEqual(len(report["policy_comparisons"]["comparisons"]), 8)
                json.dumps(report, allow_nan=False)
                changed = base / "456" / "test-strong.json"
                changed.write_bytes(changed.read_bytes() + b"\n")
                with self.assertRaisesRegex(ValueError, "bytes"):
                    run.checked_scoring_jobs(jobs)


if __name__ == "__main__":
    unittest.main()
