"""[codex] Recovery and provenance tests without model inference."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import cache_v2 as C


class CheckpointTests(unittest.TestCase):
    def test_resume_retries_only_uncommitted_batch_with_original_composition(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'array.npy'
            calls = []
            def interrupted(rows):
                calls.append(rows)
                if rows[0] == 4:
                    raise RuntimeError('simulated interruption')
                return np.array(rows, dtype=np.float32)[:, None]
            with self.assertRaises(RuntimeError):
                C.batches(path, list(range(10)), 4, interrupted, {'source': 'fixed'})
            self.assertEqual(calls, [[0, 1, 2, 3], [4, 5, 6, 7]])
            resumed = []
            def finish(rows):
                resumed.append(rows)
                return np.array(rows, dtype=np.float32)[:, None]
            value = C.batches(path, list(range(10)), 4, finish, {'source': 'fixed'})
            self.assertEqual(resumed, [[4, 5, 6, 7], [8, 9]])
            np.testing.assert_array_equal(value[:, 0], np.arange(10))
            C.batches(path, list(range(10)), 4, lambda _: self.fail('completed work repeated'), {'source': 'fixed'})

    def test_changed_source_or_batch_cannot_reuse_checkpoint(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'array.npy'
            C.batches(path, [0, 1], 2, np.array, {'source': 'a'})
            with self.assertRaisesRegex(ValueError, 'provenance'):
                C.batches(path, [0, 1], 2, np.array, {'source': 'b'})
            with self.assertRaisesRegex(ValueError, 'provenance'):
                C.batches(path, [0, 1], 1, np.array, {'source': 'a'})

    def test_nonfinite_batch_never_committed(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'array.npy'
            with self.assertRaisesRegex(ValueError, 'Invalid'):
                C.batches(path, [0], 1, lambda _: np.array([np.nan]), {})
            self.assertFalse(path.exists())


if __name__ == '__main__':
    unittest.main()
