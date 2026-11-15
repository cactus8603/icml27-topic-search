"""[codex] Synthetic stdlib tests; never inspect a live GPU or study cache."""

import io
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZIP_DEFLATED, ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import resources as R


def npy(shape, values, dtype="<f4", version=1):
    header = repr({"descr": dtype, "fortran_order": False, "shape": shape}).encode()
    prefix = b"\x93NUMPY" + bytes([version, 0])
    length_bytes = 2 if version == 1 else 4
    padding = (-(len(prefix) + length_bytes + len(header) + 1)) % 64
    header += b" " * padding + b"\n"
    prefix += struct.pack("<H" if version == 1 else "<I", len(header))
    code = {"<f4": "f", "<f8": "d", "<i4": "i"}[dtype]
    return prefix + header + struct.pack("<" + code * len(values), *values)


def cache_fixture(root):
    runs = root / "runs"
    cache = runs / "complete-cache"
    checkpoints = cache / "checkpoints"
    checkpoints.mkdir(parents=True)
    (cache / "writer.lock").touch()
    (cache / "meta.json").write_text(json.dumps({"replay": {"state": "PASS"}}))
    (cache / "panel.json").write_text(json.dumps({"ids": [1, 2, 3], "queries": [{}, {}],
                                                    "condition_texts": ["a", "b"]}))
    (cache / "checkpoint-provenance.json").write_text(json.dumps({"source": "synthetic fixture only"}))
    with ZipFile(cache / "vectors.npz", "w", compression=ZIP_DEFLATED) as archive:
        for index, name in enumerate(R.EMBEDDINGS):
            rows = 3 if name.endswith("_image") else 2
            values = [float(index + i / 10) for i in range(rows * 2)]
            archive.writestr(name + ".npy", npy((rows, 2), values, version=2))
            path = checkpoints / (name + ".npy")
            path.write_bytes(npy((rows, 2), values))
            path.with_suffix(".state.json").write_text(json.dumps({"completed": rows,
                "fingerprint": "a" * 64, "active_seconds": 1.25, "batches": []}))
    values = [float(i) + .125 for i in range(400)]
    with ZipFile(cache / "cross.npz", "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("cross.npy", npy((2, 200), values, "<f8"))
        archive.writestr("core_order.npy", npy((2, 3), [0, 1, 2, 0, 1, 2], "<i4"))
    for name, data in (("cross_top50", values[:50] + values[200:250]),
                       ("cross_rank51to200", values[50:200] + values[250:400])):
        path = checkpoints / (name + ".npy")
        path.write_bytes(npy((len(data),), data, "<f8"))
        path.with_suffix(".state.json").write_text(json.dumps({"completed": len(data),
            "fingerprint": "b" * 64, "active_seconds": 2., "batches": []}))
    return runs, cache


class FakeChild:
    pid = 123456789

    def wait(self, **_kwargs):
        return 0

    def poll(self):
        return 0


class ResourceTests(unittest.TestCase):
    def setUp(self):
        directory = R.REPO / "hf_home/tmp"
        directory.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="retrieval-resource-unit-", dir=directory)
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_gpu_csv_thresholds_and_index_restriction(self):
        rows = R.parse_gpus("0, GPU-first, 14336, 20\n1, GPU-second, 14335, 0\n2, GPU-third, 24000, 21\n")
        self.assertEqual([row["index"] for row in R.eligible_gpus(rows)], [0])
        self.assertEqual(R.eligible_gpus(rows, gpus={1, 2}), [])
        with self.assertRaises(ValueError):
            R.parse_gpus("0,GPU-first,24000,0\n1,GPU-first,24000,0")

    def test_queue_requires_two_quiet_samples_and_uses_uuid_without_shell(self):
        good = [{"index": 2, "uuid": "GPU-test-two", "free_mib": 20000, "utilization_percent": 0}]
        crowded = [{**good[0], "utilization_percent": 95}]
        samples = iter([good, crowded, good, good, good])
        delays, launched = [], []
        def spawn(command, **kwargs):
            launched.append((command, kwargs))
            state = json.loads((self.root / "queue.json").read_text())
            self.assertEqual(state["status"], "waiting")
            return FakeChild()
        code = R.run_queued(["executable", "literal;echo not-a-shell"], self.root / "queue.json",
            gpus={2}, query=lambda: next(samples), sleep=delays.append, popen=spawn,
            lock_dir=self.root / "locks", install_signals=False)
        self.assertEqual(code, 0)
        self.assertEqual(delays, [5, 30, 5])
        self.assertEqual(launched[0][0], ["executable", "literal;echo not-a-shell"])
        self.assertIs(launched[0][1]["shell"], False)
        self.assertEqual(launched[0][1]["env"]["CUDA_VISIBLE_DEVICES"], "GPU-test-two")
        saved = json.loads((self.root / "queue.json").read_text())
        self.assertEqual(saved["status"], "completed")
        self.assertEqual(saved["selected_gpu"]["index"], 2)
        self.assertEqual(saved["child_pid"], FakeChild.pid)

    def test_advisory_gpu_lock_skips_a_claimed_device(self):
        locks = self.root / "locks"
        locks.mkdir()
        rows = [{"index": i, "uuid": f"GPU-device-{i}", "free_mib": 24000, "utilization_percent": 0} for i in (0, 1)]
        launch = []
        with R.FileLock(locks / "GPU-device-0.lock"):
            code = R.run_queued(["exe"], self.root / "queue.json", query=lambda: rows,
                sleep=lambda _seconds: None, lock_dir=locks, install_signals=False,
                popen=lambda command, **kwargs: launch.append(kwargs["env"]["CUDA_VISIBLE_DEVICES"]) or FakeChild())
        self.assertEqual(code, 0)
        self.assertEqual(launch, ["GPU-device-1"])

    def test_smi_error_waits_then_recovers(self):
        rows = [{"index": 0, "uuid": "GPU-recovered", "free_mib": 24000, "utilization_percent": 1}]
        calls, delays = [], []
        def query():
            calls.append(1)
            if len(calls) == 1:
                raise OSError("temporary nvidia-smi failure")
            return rows
        self.assertEqual(R.run_queued(["exe"], self.root / "queue.json", query=query,
            sleep=delays.append, popen=lambda *args, **kwargs: FakeChild(),
            lock_dir=self.root / "locks", install_signals=False), 0)
        self.assertEqual(delays, [30, 5])

    def test_last_second_gpu_change_resets_the_confirmation_streak(self):
        good = [{"index": 0, "uuid": "GPU-changing", "free_mib": 24000, "utilization_percent": 0}]
        busy = [{**good[0], "free_mib": 2000}]
        samples = iter([good, good, busy, good, good, good])
        delays = []
        self.assertEqual(R.run_queued(["exe"], self.root / "queue.json", query=lambda: next(samples),
            sleep=delays.append, popen=lambda *args, **kwargs: FakeChild(),
            lock_dir=self.root / "locks", install_signals=False), 0)
        self.assertEqual(delays, [5, 30, 5])

    def test_dry_run_checks_payloads_and_keeps_every_checkpoint(self):
        runs, cache = cache_fixture(self.root)
        receipt_path, receipt = R.cleanup_cache(cache, runs_root=runs)
        self.assertEqual(receipt["status"], "dry-run")
        self.assertEqual(len(receipt["files"]), 8)
        self.assertGreater(receipt["verified_duplicate_bytes"], 0)
        self.assertEqual(receipt["removed_bytes"], 0)
        self.assertTrue(receipt_path.is_file())
        for row in receipt["files"]:
            self.assertEqual(R.sha256(row["path"]), row["sha256"])
            self.assertTrue(Path(row["path"]).exists())

    def test_apply_removes_only_exact_duplicate_arrays_and_keeps_states(self):
        runs, cache = cache_fixture(self.root)
        other = cache / "checkpoints/keep-other.npy"
        other.write_bytes(b"unrelated file")
        before = {name: R.sha256(cache / name) for name in ("vectors.npz", "cross.npz", "meta.json", "panel.json")}
        _, receipt = R.cleanup_cache(cache, runs_root=runs, apply=True)
        self.assertEqual(receipt["removed_bytes"], receipt["verified_duplicate_bytes"])
        self.assertTrue(other.exists())
        self.assertTrue((cache / "writer.lock").exists())
        self.assertTrue((cache / "checkpoint-provenance.json").exists())
        for row in receipt["files"]:
            self.assertFalse(Path(row["path"]).exists())
            self.assertTrue(Path(row["path"]).with_suffix(".state.json").exists())
        self.assertEqual(before, {name: R.sha256(cache / name) for name in before})
        _, again = R.cleanup_cache(cache, runs_root=runs, apply=True)
        self.assertEqual(again["removed_bytes"], 0)

    def test_one_bad_embedding_prevents_all_deletions(self):
        runs, cache = cache_fixture(self.root)
        path = cache / "checkpoints/core_embed_image.npy"
        data = path.read_bytes()
        path.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
        with self.assertRaisesRegex(ValueError, "differs"):
            R.cleanup_cache(cache, runs_root=runs, apply=True)
        self.assertEqual(len(list((cache / "checkpoints").glob("*.npy"))), 8)

    def test_cross_flat_checkpoint_must_match_each_final_row_slice(self):
        runs, cache = cache_fixture(self.root)
        path = cache / "checkpoints/cross_top50.npy"
        # Same length and dtype, but the wrong row-1 slice would be a dangerous false match.
        path.write_bytes(npy((100,), [float(i) + .125 for i in range(100)], "<f8"))
        with self.assertRaisesRegex(ValueError, "differs"):
            R.cleanup_cache(cache, runs_root=runs, apply=True)
        self.assertTrue((cache / "checkpoints/core_embed_image.npy").exists())

    def test_active_writer_and_failed_replay_are_protected(self):
        runs, cache = cache_fixture(self.root)
        with R.FileLock(cache / "writer.lock", create=False):
            with self.assertRaises(BlockingIOError):
                R.cleanup_cache(cache, runs_root=runs, apply=True)
        (cache / "meta.json").write_text(json.dumps({"replay": {"state": "MISMATCH"}}))
        with self.assertRaisesRegex(ValueError, "baseline-PASS"):
            R.cleanup_cache(cache, runs_root=runs, apply=True)

    def test_symlinks_outside_paths_and_unfinished_states_are_rejected(self):
        runs, cache = cache_fixture(self.root)
        with self.assertRaisesRegex(ValueError, "restricted"):
            R.cleanup_cache(self.root, runs_root=runs, apply=True)
        path = cache / "checkpoints/core_embed_image.npy"
        saved = self.root / "original-model-file.bin"
        saved.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(saved)
        with self.assertRaisesRegex(ValueError, "symlink"):
            R.cleanup_cache(cache, runs_root=runs, apply=True)
        self.assertTrue(saved.exists())
        path.unlink()
        path.write_bytes(saved.read_bytes())
        state_path = path.with_suffix(".state.json")
        state = json.loads(state_path.read_text())
        state["completed"] -= 1
        state_path.write_text(json.dumps(state))
        with self.assertRaisesRegex(ValueError, "incomplete"):
            R.cleanup_cache(cache, runs_root=runs, apply=True)

    def test_object_or_fortran_arrays_are_not_interpreted(self):
        content = npy((2,), [1., 2.])
        bad = content.replace(b"'<f4'", b"'|O8'")
        with self.assertRaisesRegex(ValueError, "dtype"):
            R.npy_header(io.BytesIO(bad))

    def test_foreign_ownership_and_hardlinks_are_rejected(self):
        runs, cache = cache_fixture(self.root)
        uid = os.getuid()
        with patch.object(R.os, "getuid", return_value=uid + 1):
            with self.assertRaisesRegex(ValueError, "owned"):
                R.cleanup_cache(cache, runs_root=runs, apply=True)
        path = cache / "checkpoints/core_embed_image.npy"
        original = self.root / "original-model.bin"
        os.link(path, original)
        with self.assertRaisesRegex(ValueError, "multiply linked"):
            R.cleanup_cache(cache, runs_root=runs, apply=True)
        self.assertTrue(original.exists())
        self.assertTrue(path.exists())

    def test_truncated_final_npz_prevents_all_deletions(self):
        runs, cache = cache_fixture(self.root)
        path = cache / "cross.npz"
        path.write_bytes(path.read_bytes()[:-32])
        with self.assertRaises(Exception):
            R.cleanup_cache(cache, runs_root=runs, apply=True)
        self.assertEqual(len(list((cache / "checkpoints").glob("*.npy"))), 8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
