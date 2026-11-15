"""[codex] Synthetic CPU cleanup tests; never touches active caches or GPUs."""
import copy
import fcntl
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cleanup_verified as C


class CleanupVerifiedTests(unittest.TestCase):
    def setUp(self):
        temporary_root = C.REPO / "hf_home/tmp"
        temporary_root.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="verified-cleanup-unit-", dir=temporary_root)
        self.root = Path(self.temporary.name)
        self.runs = self.root / "dl/retrieval3090/runs"
        self.runs.mkdir(parents=True)
        self.addCleanup(self.temporary.cleanup)
        self.runs_patch = patch.object(C, "RUNS_ROOT", self.runs)
        self.runs_patch.start()
        self.addCleanup(self.runs_patch.stop)

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value, sort_keys=True) + "\n")

    def checkpoint(self, path, value, fingerprint):
        np.save(path, value)
        state = {"completed": len(value), "array_shape": list(value.shape), "dtype": value.dtype.str,
                 "sha256": C.sha(path), "fingerprint": fingerprint, "active_seconds": 1.25,
                 "batches": [{"start": 0, "end": len(value), "seconds": 1.25}]}
        self.write(path.with_suffix(".state.json"), state)

    def index_fixture(self):
        directory = self.runs / "shared-index"
        cp = directory / "index-checkpoints"
        cp.mkdir(parents=True)
        (directory / "writer.lock").touch()
        panel = {"candidates": [{"id": i, "text": f"candidate {i}"} for i in range(3)],
                 "queries": [{"text": f"query {i}"} for i in range(2)]}
        self.write(directory / "panel.json", panel)
        identity = {"schema": "mcmr-core-shared-index-v1", "panel_sha256": C.digest(panel),
                    "candidate_inputs_sha256": C.digest(panel["candidates"]),
                    "query_inputs_sha256": C.digest([q["text"] for q in panel["queries"]]),
                    "candidates": 3, "queries": 2, "embedding_dim": 2048,
                    "batch_sizes": {"joint_image": 1, "text": 32}, "source": "synthetic"}
        self.write(cp / "provenance.json", identity)
        image = np.zeros((3, 2048), dtype=np.float32)
        full = np.zeros((2, 2048), dtype=np.float32)
        image[:, 0], full[:, 1] = 1, 1
        np.savez(directory / "index.npz", core_embed_image=image, core_embed_full=full)
        self.write(directory / "index-meta.json", {"state": "COMPLETE", "identity": identity,
                   "index_sha256": C.sha(directory / "index.npz"), "commit": "a" * 40})
        for stage, value, items, batch in (("core_joint", image, [0, 1, 2], 1),
                                          ("core_full", full, [q["text"] for q in panel["queries"]], 32)):
            fingerprint = C.fingerprint({"identity": identity | {"stage": stage, "items_sha256": C.digest(items)},
                                         "items": len(items), "batch_size": batch})
            self.checkpoint(cp / (stage + ".npy"), value, fingerprint)
        return directory

    def platform_fixture(self):
        directory = self.runs / "platform-cache"
        cp = directory / "checkpoints"
        cp.mkdir(parents=True)
        (directory / "writer.lock").touch()
        panel = {"ids": list(range(1, 202)), "queries": [{"text": "query 0"}, {"text": "query 1"}]}
        self.write(directory / "panel.json", panel)
        self.write(directory / "meta.json", {"replay": {"state": "MISMATCH"}, "scope": "synthetic"})
        np.savez(directory / "vectors.npz", synthetic=np.arange(4, dtype=np.float32))
        core = np.tile(np.arange(201, dtype=np.int32), (2, 1))
        cross = np.arange(400, dtype=np.float64).reshape(2, 200) / 100
        np.savez(directory / "cross.npz", core_order=core, cross=cross)
        context = {"mode": "platform-relative-exploration", "formal_evidence": False,
                   "published_reproduction": {"state": "MISMATCH"}, "platform_consistency": "VALID_LOCAL",
                   "producer_commit": "a" * 40, "plan_sha256": "b" * 64, "environment": {"torch": "synthetic"},
                   "source_checkpoint_provenance_sha256": "c" * 64,
                   "cache_artifacts_sha256": {name: C.sha(directory / name) for name in
                                              ("meta.json", "panel.json", "vectors.npz", "cross.npz")}}
        provenance = {"schema": "retrieval-platform-continuation-v1", "commit": context["producer_commit"],
                      "plan_sha256": context["plan_sha256"], "environment": context["environment"],
                      "source_checkpoint_provenance_sha256": context["source_checkpoint_provenance_sha256"],
                      "source_checkpoint_files": {"synthetic.npy": {"sha256": "d" * 64}}}
        self.write(directory / "platform-provenance.json", provenance)
        self.write(directory / "platform-receipt.json", {"source_checkpoint_files": provenance["source_checkpoint_files"]})
        context["receipt_sha256"] = C.sha(directory / "platform-receipt.json")
        keys = [panel["ids"][int(column)] for column in core[:, 50:200].flat]
        queries = [q["text"] for q in panel["queries"] for _ in range(150)]
        fingerprint = C.fingerprint({"identity": provenance | {"stage": "rank51to200", "queries": queries, "images": keys},
                                     "items": 300, "batch_size": 4})
        self.checkpoint(cp / "cross_rank51to200.npy", cross[:, 50:200].reshape(-1), fingerprint)
        return directory, context

    def test_index_dry_run_records_exact_sha_bytes_and_retains_arrays(self):
        directory = self.index_fixture()
        path, receipt = C.cleanup_index(directory)
        self.assertEqual(receipt["status"], "dry-run")
        self.assertEqual(receipt["removed_bytes"], 0)
        self.assertEqual(len(receipt["files"]), 2)
        self.assertTrue(path.is_file())
        for row in receipt["files"]:
            self.assertEqual(row["sha256"], C.sha(row["path"]))
            self.assertEqual(row["bytes"], Path(row["path"]).stat().st_size)
            self.assertEqual(row["state_sha256"], C.sha(Path(row["path"]).with_suffix(".state.json")))
            self.assertFalse(row["removed"])

    def test_apply_deletes_only_allowlisted_index_duplicates(self):
        directory = self.index_fixture()
        keep = directory / "index-checkpoints/keep-other.npy"
        keep.write_bytes(b"owned unrelated file")
        before = {str(p): C.sha(p) for p in directory.rglob("*") if p.is_file()
                  and p.name not in {"core_joint.npy", "core_full.npy"}}
        _, receipt = C.cleanup_index(directory, apply=True)
        self.assertEqual(receipt["status"], "applied")
        self.assertEqual(receipt["removed_bytes"], receipt["verified_duplicate_bytes"])
        self.assertTrue(all(row["removed"] for row in receipt["files"]))
        for row in receipt["files"]:
            self.assertFalse(Path(row["path"]).exists())
            self.assertTrue(Path(row["path"]).with_suffix(".state.json").exists())
        self.assertEqual(before, {path: C.sha(path) for path in before})

    def test_second_payload_mismatch_fails_before_any_deletion(self):
        directory = self.index_fixture()
        path = directory / "index-checkpoints/core_full.npy"
        value = np.load(path)
        value[0, 1], value[0, 2] = 0, 1
        state = json.loads(path.with_suffix(".state.json").read_text())
        self.checkpoint(path, value, state["fingerprint"])
        with self.assertRaisesRegex(ValueError, "exact finite duplicate"):
            C.cleanup_index(directory, apply=True)
        self.assertTrue((directory / "index-checkpoints/core_joint.npy").exists())
        self.assertTrue(path.exists())
        self.assertEqual(list(directory.glob("cleanup-verified-*.json")), [])

    def test_state_sha_shape_fingerprint_and_completion_are_mandatory(self):
        directory = self.index_fixture()
        path = directory / "index-checkpoints/core_full.state.json"
        original = json.loads(path.read_text())
        for key, value in (("sha256", "0" * 64), ("completed", 1), ("array_shape", [2, 100]),
                           ("dtype", "<f8"), ("fingerprint", "f" * 64)):
            with self.subTest(key=key):
                self.write(path, original | {key: value})
                with self.assertRaisesRegex(ValueError, "Checkpoint"):
                    C.cleanup_index(directory, apply=True)
                self.assertTrue((directory / "index-checkpoints/core_joint.npy").exists())
                self.assertTrue(path.with_suffix(".npy").parent.joinpath("core_full.npy").exists())

    def test_complete_index_and_final_digest_are_mandatory(self):
        directory = self.index_fixture()
        path = directory / "index-meta.json"
        original = json.loads(path.read_text())
        for changed in (original | {"state": "RUNNING"}, original | {"index_sha256": "0" * 64}):
            self.write(path, changed)
            with self.assertRaisesRegex(ValueError, "incomplete"):
                C.cleanup_index(directory, apply=True)
            self.assertTrue((directory / "index-checkpoints/core_joint.npy").exists())

    def test_writer_lock_blocks_all_cleanup(self):
        directory = self.index_fixture()
        with (directory / "writer.lock").open("r") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                C.cleanup_index(directory, apply=True)
        self.assertTrue((directory / "index-checkpoints/core_joint.npy").exists())

    def test_hardlink_symlink_and_foreign_owner_fail_before_unlink(self):
        directory = self.index_fixture()
        path = directory / "index-checkpoints/core_joint.npy"
        hard = directory / "hard-link.npy"
        os.link(path, hard)
        with self.assertRaisesRegex(ValueError, "single-link"):
            C.cleanup_index(directory, apply=True)
        hard.unlink()
        moved = directory / "moved.npy"
        path.rename(moved)
        path.symlink_to(moved)
        with self.assertRaisesRegex(ValueError, "symlink"):
            C.cleanup_index(directory, apply=True)
        path.unlink()
        moved.rename(path)
        with patch.object(C.os, "getuid", return_value=os.getuid() + 10000):
            with self.assertRaisesRegex(ValueError, "owned"):
                C.cleanup_index(directory, apply=True)
        self.assertTrue(path.exists())

    def test_outside_root_and_symlink_ancestor_are_rejected(self):
        directory = self.index_fixture()
        with self.assertRaisesRegex(ValueError, "runs root"):
            C.cleanup_index(self.root, apply=True)
        cp = directory / "index-checkpoints"
        moved = directory / "old-checkpoints"
        cp.rename(moved)
        cp.symlink_to(moved, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            C.cleanup_index(directory, apply=True)
        self.assertTrue((moved / "core_joint.npy").exists())

    def test_platform_only_removes_rank51to200_and_preserves_failed_source(self):
        directory, context = self.platform_fixture()
        source = self.runs / "original-failed-cache"
        source.mkdir()
        source_array = source / "source-image.npy"
        source_array.write_bytes(b"original failed cache must remain")
        unrelated = directory / "checkpoints/core_embed_image.npy"
        unrelated.write_bytes(b"unrelated source embedding")
        before = {str(p): C.sha(p) for p in directory.rglob("*") if p.is_file()
                  and p.name != "cross_rank51to200.npy"}
        with patch.object(C, "validate_context", return_value=context) as validate:
            _, receipt = C.cleanup_platform(directory, self.root / "plan.json", directory / "platform-receipt.json", apply=True)
        validate.assert_called_once_with(directory, self.root / "plan.json", directory / "platform-receipt.json")
        self.assertEqual(len(receipt["files"]), 1)
        self.assertEqual(receipt["files"][0]["shape"], [300])
        self.assertFalse((directory / "checkpoints/cross_rank51to200.npy").exists())
        self.assertEqual(before, {path: C.sha(path) for path in before})
        self.assertEqual(source_array.read_bytes(), b"original failed cache must remain")

    def test_platform_requires_successful_common_validation(self):
        directory, _ = self.platform_fixture()
        with patch.object(C, "validate_context", side_effect=ValueError("invalid platform receipt")):
            with self.assertRaisesRegex(ValueError, "invalid platform"):
                C.cleanup_platform(directory, self.root / "plan.json", directory / "platform-receipt.json", apply=True)
        self.assertTrue((directory / "checkpoints/cross_rank51to200.npy").exists())

    def test_platform_flattened_cross_shape_and_payload_are_checked(self):
        directory, context = self.platform_fixture()
        path = directory / "checkpoints/cross_rank51to200.npy"
        original = np.load(path)
        fingerprint = json.loads(path.with_suffix(".state.json").read_text())["fingerprint"]
        for value in (original.reshape(2, 150), original + 1):
            self.checkpoint(path, value, fingerprint)
            with patch.object(C, "validate_context", return_value=context):
                with self.assertRaises(ValueError):
                    C.cleanup_platform(directory, self.root / "plan.json", directory / "platform-receipt.json", apply=True)
            self.assertTrue(path.exists())

    def test_final_recheck_catches_change_before_deleting_first_candidate(self):
        directory = self.index_fixture()
        original = C._recheck
        def change(records):
            (directory / "index-checkpoints/core_full.npy").write_bytes(b"changed after validation")
            return original(records)
        with patch.object(C, "_recheck", side_effect=change):
            with self.assertRaisesRegex(ValueError, "changed"):
                C.cleanup_index(directory, apply=True)
        self.assertTrue((directory / "index-checkpoints/core_joint.npy").exists())

    def test_cli_defaults_to_dry_run(self):
        directory = self.index_fixture()
        with patch.object(sys, "argv", ["cleanup_verified", "mcmr-index", "--index", str(directory)]):
            with patch.object(C, "cleanup_index", return_value=(Path("receipt.json"), {})) as cleanup:
                C.main()
        cleanup.assert_called_once_with(directory, apply=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
