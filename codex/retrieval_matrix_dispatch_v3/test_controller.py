"""[codex] CPU-only live-worker adoption and global-cap regression checks."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from codex.retrieval_matrix_dispatch_v3 import controller as C
B = C.B
ROOT = Path(__file__).resolve().parents[2]


class Adoption(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT/"hf_home/tmp", prefix="central-adoption-test-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.old = self.root/"old"
        self.new = self.root/"new"
        self.old.mkdir()
        self.new.mkdir()
        def task(key):
            return {"id": key, "model": "siglip2" if key == "native-siglip2" else "core2",
                "command": ["bash", "synthetic-launcher", key], "output": str(self.root/key),
                "freeze_commit": "a"*40, "min_free_mib": 20480 if key == "native-siglip2" else 8192}
        old_tasks = [task(key) for key in ("mcq-core2", "embedding-core8", "already-failed", "native-siglip2")]
        old_path = self.old/"schedule.json"
        old_path.write_text(json.dumps({"tasks": old_tasks}))
        self.old_sha = B.sha(old_path)
        self.schedule = {"schema": 1, "owner": "[codex]", "max_live": 2, "confirmation_seconds": 5,
            "max_existing_compute_pids": 1, "controller": "adopt-reviewed-repair-v3",
            "tasks": copy.deepcopy(old_tasks)+[task("fixed-core2")],
            "adoption": {"schedule": {"path": str(old_path), "sha256": self.old_sha},
                "runtime": str(self.old), "retire_pending": ["embedding-core8"]}}
        self.schedule["tasks"][3]["min_free_mib"] = 12288
        self.previous = {"schedule_sha256": self.old_sha, "tasks": {
            "mcq-core2": {"status": "running", "token": "original-token", "command": old_tasks[0]["command"],
                "receipt": str(self.old/"mcq-core2-attempt.json"), "worker": {"pid": 100, "start_ticks": 1, "boot_id": "synthetic"},
                "child": {"pid": 101, "start_ticks": 2, "boot_id": "synthetic"}, "gpu": {"uuid": "GPU-old"}},
            "embedding-core8": {"status": "pending"},
            "already-failed": {"status": "failed", "error": "original synthetic failure"},
            "native-siglip2": {"status": "pending"}}}
        self.receipt({"status": "running"})

    def receipt(self, changes):
        value = {"task_id": "mcq-core2", "token": "original-token", "schedule_sha256": self.old_sha,
            **{key: self.previous["tasks"]["mcq-core2"][key] for key in ("worker", "child")}, **changes}
        (self.old/"mcq-core2-attempt.json").write_text(json.dumps(value))

    def dispatcher(self, query=lambda: [], popen=lambda *a, **k: None):
        prior = self.new/"previous-state.json"
        prior.write_text(json.dumps(self.previous))
        (self.new/"migration.json").write_text(json.dumps({"new_schedule_sha256": "b"*64,
            "previous_state_sha256": B.sha(prior), "old_parent_retired": True}))
        return C.Dispatcher(self.schedule, "b"*64, self.new, query=query, popen=popen)

    def test_preserve_live_tokens_failures_and_only_retire_pending(self):
        state = C.seed(self.schedule, "b"*64, self.previous)
        self.assertEqual(state["tasks"]["mcq-core2"]["worker"], self.previous["tasks"]["mcq-core2"]["worker"])
        self.assertEqual(state["tasks"]["mcq-core2"]["adopted_schedule_sha256"], self.old_sha)
        self.assertEqual(state["tasks"]["already-failed"], self.previous["tasks"]["already-failed"])
        self.assertTrue(state["tasks"]["embedding-core8"]["retired_without_start"])
        self.assertEqual(state["tasks"]["native-siglip2"], {"status": "pending"})

    def test_started_replacement_target_is_never_retired(self):
        previous = copy.deepcopy(self.previous)
        previous["tasks"]["embedding-core8"] = {**previous["tasks"]["mcq-core2"],
            "command": self.schedule["tasks"][1]["command"], "receipt": str(self.old/"embedding-core8-attempt.json")}
        with self.assertRaisesRegex(ValueError, "already started"):
            C.seed(self.schedule, "b"*64, previous)

    def test_later_handoff_preserves_first_worker_receipt_origin(self):
        previous = copy.deepcopy(self.previous)
        row = previous["tasks"]["mcq-core2"]
        earlier = self.root/"earlier-runtime"
        row.update(adopted_runtime=str(earlier), adopted_schedule_sha256="c"*64,
            receipt=str(earlier/"mcq-core2-attempt.json"))
        state = C.seed(self.schedule, "b"*64, previous)
        actual = state["tasks"]["mcq-core2"]
        self.assertEqual(actual["receipt"], row["receipt"])
        self.assertEqual(actual["adopted_schedule_sha256"], "c"*64)
        self.assertEqual(actual["adopted_runtime"], str(earlier))

    def test_original_command_and_complete_history_cannot_change(self):
        schedule = copy.deepcopy(self.schedule)
        schedule["tasks"][0]["command"].append("changed")
        with self.assertRaises(ValueError):
            C.seed(schedule, "b"*64, self.previous)
        previous = copy.deepcopy(self.previous)
        del previous["tasks"]["already-failed"]
        with self.assertRaises(ValueError):
            C.seed(self.schedule, "b"*64, previous)

    def test_adopt_original_receipt_and_reject_wrong_token(self):
        dispatcher = self.dispatcher()
        with patch.object(B, "alive", return_value=True):
            dispatcher.reconcile()
        self.assertEqual(dispatcher.live_count(), 1)
        self.receipt({"status": "completed", "returncode": 0})
        dispatcher.reconcile()
        self.assertEqual(dispatcher.state["tasks"]["mcq-core2"]["status"], "completed")
        dispatcher.state["tasks"]["mcq-core2"]["status"] = "running"
        self.receipt({"status": "completed", "token": "wrong"})
        with self.assertRaisesRegex(ValueError, "identity"):
            dispatcher.reconcile()

    def test_adopted_worker_reserves_its_original_gpu(self):
        def forbidden(*args, **kwargs):
            raise AssertionError("Cannot launch beside an adopted worker on the same card")
        query = lambda: [{"uuid": "GPU-old", "index": 0, "free_mib": 24000, "compute_pids": [101]}]
        dispatcher = self.dispatcher(query=query, popen=forbidden)
        with patch.object(B, "alive", return_value=True), patch.object(dispatcher, "complete", return_value=False):
            dispatcher.tick()
        self.assertEqual(dispatcher.live_count(), 1)

    def test_adopted_and_new_worker_share_global_cap_two(self):
        dispatcher = self.dispatcher(query=lambda: self.fail("No admission query permitted at cap2"))
        second = self.new/"fixed-attempt.json"
        second.write_text(json.dumps({"task_id": "fixed-core2", "token": "new-token",
            "schedule_sha256": "b"*64, "status": "running", "worker": {"pid": 200}, "child": {"pid": 201}}))
        dispatcher.state["tasks"]["fixed-core2"] = {"status": "running", "token": "new-token", "receipt": str(second)}
        with patch.object(B, "alive", return_value=True):
            dispatcher.tick()
        self.assertEqual(dispatcher.live_count(), 2)

    def test_expanded_resource_class_is_only_siglip2(self):
        C.validate(self.schedule)
        bad = copy.deepcopy(self.schedule)
        bad["tasks"][0]["min_free_mib"] = 12288
        with self.assertRaisesRegex(ValueError, "reserved"):
            C.validate(bad)

    def test_import_and_constructor_do_not_mutate_base_validator(self):
        self.assertIs(B.validate, C.ORIGINAL_VALIDATE)

    def test_stop_only_exact_current_children_and_never_adopted_workers(self):
        dispatcher = self.dispatcher()
        dispatcher.state["tasks"]["fixed-core2"] = {"status": "running", "worker": {"pid": 200}}
        dispatcher.children["fixed-core2"] = SimpleNamespace(pid=200)
        with patch.object(dispatcher, "reconcile"), patch.object(B, "alive", return_value=True), patch.object(B.os, "kill") as kill:
            dispatcher.stop_own()
            kill.assert_called_once_with(200, C.signal.SIGTERM)
        dispatcher.state["tasks"]["fixed-core2"]["worker"]["pid"] = 201
        with patch.object(dispatcher, "reconcile"), patch.object(B, "alive", return_value=True), patch.object(B.os, "kill") as kill:
            dispatcher.stop_own()
            kill.assert_not_called()
        self.dispatcher()
        self.assertIs(B.validate, C.ORIGINAL_VALIDATE)


if __name__ == "__main__":
    unittest.main()
