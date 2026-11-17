"""Exercise the Stage 2a controls without invoking Slurm or reading scores."""

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parent))
import coordinator as c


class ContinuationTests(unittest.TestCase):
    def test_successful_retry_keeps_failed_attempt_cost_eligible(self):
        attempts = [("3", 0, "31", "FAILED", "1:0", 500), ("4", 0, "41", "COMPLETED", "0:0", 800)]
        self.assertEqual(c.completed_indices(attempts, 2), {0})
        with self.assertRaisesRegex(ValueError, "failed without a successful retry"):
            c.completed_indices(attempts[:1], 2)

    def test_gpu_budget_covers_exact_remaining_tasks_and_reserve(self):
        tasks = [("sd14", "E1", 2000, 20000), ("uce", "B1", 2000, 20000)]
        cost = {"est_img_per_s": 9.0, "cap_gpu_hours": 50}
        group = {"indices": [1], "images_per_task": 18000, "minutes_per_task": 60}
        schedule = {"formal_phase": "2a", "plan_sha256": "known", "reserved_gpu_seconds": 1800,
                    "current_phase_gpu_seconds": 3700, "total_spent_gpu_seconds": 172000,
                    "available_gpu_seconds_for_phase": 4300, "array_groups": [group]}
        self.assertEqual(c.checked_group(schedule, tasks, {0}, cost, "known", 30), group)
        for wrong in ({**group, "indices": [0]}, {**group, "minutes_per_task": 72}):
            with self.assertRaisesRegex(ValueError, "omitted|exceeds"):
                c.checked_group({**schedule, "array_groups": [wrong]}, tasks, {0}, cost, "known", 30)

    def test_scheduler_runs_commit_bytes_in_canonical_repo_context(self):
        pipeline_source = b"known = 421\n"
        schedule_source = b"""
import json, sys
from pathlib import Path
import pipeline
assert Path(__file__).name == 'schedule.py'
assert Path(__file__).resolve().parents[3] == Path.cwd()
assert sys.argv[2:] == ['2a', '--reserve-minutes', '30']
print(json.dumps({'sealed': pipeline.known}))
"""
        with patch.object(c, "git_source", side_effect=[pipeline_source, schedule_source]):
            self.assertEqual(c.sealed_scheduler("pinned", 30), {"sealed": 421})

    def test_waiting_queue_short_circuits_before_manifest_or_submission(self):
        controller = object.__new__(c.Controller)
        controller.queue = lambda: ["26802|PENDING|t2ir-formal-plan-2a"]
        self.assertEqual(controller.advance()["state"], "waiting")

    def test_ledger_submission_and_duplicate_queue_guard(self):
        with TemporaryDirectory(dir=Path(__file__).parent) as scratch:
            root = Path(scratch)
            controller = object.__new__(c.Controller)
            controller.ledger = root / "ledger"
            controller.intent = root / "intent"
            controller.check_scientific_source = lambda: None
            group = {"indices": [1, 2], "minutes_per_task": 61}
            controller.queue = lambda: ["8|RUNNING|t2ir-formal-replay"]
            with patch.object(c, "command") as command:
                self.assertEqual(controller.submit(group, "abc", [])["state"], "waiting")
                command.assert_not_called()
            controller.queue = lambda: []
            with patch.object(c, "command", return_value="60;test") as command:
                self.assertEqual(controller.submit(group, "abc", [])["job"], "60")
                self.assertIn("--array=1,2%2", command.call_args.args)
                self.assertIn("--time=61", command.call_args.args)
            self.assertEqual(c.ledger_entries(controller.ledger.read_text(), "formal-stage=2a"), ["60"])
            self.assertFalse(controller.intent.exists())
            with patch.object(c, "command") as command, self.assertRaisesRegex(ValueError, "during planning"):
                controller.submit(group, "abc", [])
                command.assert_not_called()
            controller.intent.touch()
            with patch.object(c, "command") as command, self.assertRaisesRegex(ValueError, "Reconcile"):
                controller.submit(group, "abc", ["60"])
                command.assert_not_called()

    def test_quota_uses_registered_group_filesystem_limit(self):
        blocks = "h:/work 1561100672 1610612736 1610612736 6011772 0 0"
        self.assertEqual(c.quota_free_kib(blocks), 49512064)
        self.assertEqual(c.MIN_GROUP_FREE_KIB, 20 * 1024 ** 2)


if __name__ == "__main__":
    unittest.main()
