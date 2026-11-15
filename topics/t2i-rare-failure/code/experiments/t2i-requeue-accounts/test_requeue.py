"""Use synthetic duplicate accounting rows for the exact frozen components."""

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parent))
import requeue


REPO = Path.cwd()
SCHEDULE = "8e49e19c8fcc60e9a71f85c1811f8b88315868cb"
TASKS = [("rece", "all", 0, 2000), ("advunlearn", "all", 0, 2000)]
ROWS = """51|50_0|t2ir-formal-replay|REQUEUED|0:0|2387|gres/gpu=1|old0
50|50_1|t2ir-formal-replay|REQUEUED|0:0|2309|gres/gpu=1|old1
51|50_0|t2ir-formal-replay|COMPLETED|0:0|3129|gres/gpu=1|new0
50|50_1|t2ir-formal-replay|COMPLETED|0:0|3100|gres/gpu=1|new1"""


def frozen_pipeline():
    path = requeue.PIPELINE_PATH
    return requeue.load_source(requeue.git_show(REPO, requeue.FORMAL, path), path, "synthetic_pipeline")


class RequeueTests(unittest.TestCase):
    def test_both_historical_requeues_count_and_active_still_rejected(self):
        original = frozen_pipeline()
        with self.assertRaisesRegex(ValueError, "still active"):
            original.parse_allocations(["50"], ROWS)
        pipeline = requeue.replay_accounting(original)
        attempts = pipeline.parse_allocations(["50"], ROWS)
        self.assertEqual([entry[3] for entry in attempts], ["REQUEUED", "REQUEUED", "COMPLETED", "COMPLETED"])
        self.assertEqual(sum(entry[-1] for entry in attempts), 10925)
        with self.assertRaisesRegex(ValueError, "still active"):
            pipeline.parse_allocations(["50"], ROWS.replace("COMPLETED", "RUNNING"))

    def test_selector_requires_successful_receipts_after_the_requeues(self):
        pipeline = requeue.replay_accounting(frozen_pipeline())
        criteria = {"cost": {"est_img_per_s": 9, "cap_gpu_hours": 50}}
        attempts = pipeline.parse_allocations(["50"], ROWS)
        evidence = SimpleNamespace(read_json=lambda file: json.loads(file.read_bytes()))
        with TemporaryDirectory(dir=Path(__file__).parent) as scratch:
            root = Path(scratch)
            operations = root / "operations"
            operations.mkdir()
            for i, task in enumerate(TASKS):
                job = ("51", "50")[i]
                (operations / f"1-{i}-{job}.json").write_text(json.dumps({
                    "freeze": "test", "phase": "1", "task": list(task), "job": job, "array": "50",
                    "chunk_count": 0, "gate_receipt_sha256": "synthetic",
                }))
            with (patch.object(pipeline, "stage_tasks", return_value=TASKS),
                  patch.object(pipeline, "output_chunks", return_value=[]),
                  patch.object(pipeline, "allocation_report", return_value=attempts) as account):
                args = REPO, root, "test", "synthetic", criteria, range(14), evidence, ("1",)
                self.assertEqual(pipeline.complete_previous(*args), 10925)
                account.return_value = attempts[:-1]
                with self.assertRaisesRegex(ValueError, "completed task lacks"):
                    pipeline.complete_previous(*args)

    def test_scheduler_and_replacement_watcher_use_the_same_accounting(self):
        scheduler_path = Path("codex/experiments/t2i-formal-replay/schedule.py")
        controller = requeue.load_controller(REPO, "99")
        attempts = controller.pipeline_module().parse_allocations(["50"], ROWS)
        self.assertIn('pipeline.TERMINAL.add("REQUEUED")', controller.SCHEDULER_ENTRY)
        self.assertEqual(controller.ORIGINAL_PLANNER, "26802")
        self.assertEqual(controller.PLANNER, "99")
        self.assertEqual(controller.completed_indices(attempts, len(TASKS)), {0, 1})
        with self.assertRaisesRegex(ValueError, "without a successful retry"):
            controller.completed_indices(attempts[:-1], len(TASKS))
        with patch.dict(sys.modules, {"pipeline": controller.pipeline_module()}):
            scheduler = requeue.load_source(requeue.git_show(REPO, SCHEDULE, scheduler_path), scheduler_path, "synthetic_schedule")
        seconds, rates = scheduler.completed_rates(TASKS, attempts, 14)
        self.assertEqual(seconds, 10925)
        self.assertAlmostEqual(rates[("rece", "all")], 28000 / (2387 + 3129))
        self.assertAlmostEqual(rates[("advunlearn", "all")], 28000 / (2309 + 3100))


if __name__ == "__main__":
    if len(sys.argv) == 2:
        REPO = Path(sys.argv.pop()).resolve()
    unittest.main()
