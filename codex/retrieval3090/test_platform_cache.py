"""[codex] Fixed-source/repeat rejection tests; no research metrics or GPU."""
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

import platform_cache as P


class ContinuationEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.plan = {'schema': 'retrieval-platform-plan-v1', 'scope': 'exposed-development-only',
                     'formal_evidence': False, 'source_cache_commit': 'a' * 40,
                     'source': 'b' * 40, 'freeze': 'c' * 40,
                     'environment': {'torch': '2.14.0+cu126'}}
        self.arrays = {'core_embed_image': np.eye(16, dtype=np.float32),
                       'core_embed_full': np.eye(32, dtype=np.float32)}
        np.savez(self.root / 'probe_vectors.npz', **self.arrays)
        self.record = {'status': 'technical_probe_complete', 'commit': 'a' * 40,
                       'source': 'b' * 40, 'freeze': 'c' * 40,
                       'environment': {'packages': {'torch': '2.14.0+cu126'}},
                       'probe_vectors_sha256': P.file_sha(self.root / 'probe_vectors.npz')}
        self.write_record()

    def write_record(self):
        (self.root / 'probe.json').write_text(json.dumps(self.record))

    def test_repeat_accepts_exact_original_batches(self):
        proof = P.verify_repeat(self.root, self.root, self.arrays, self.plan)
        self.assertEqual(proof['probe_vectors_sha256'], self.record['probe_vectors_sha256'])

    def test_repeat_rejects_different_runtime_or_source(self):
        for field in ('commit', 'source', 'freeze'):
            bad = dict(self.record)
            bad[field] = 'd' * 40
            (self.root / 'probe.json').write_text(json.dumps(bad))
            with self.assertRaises(ValueError):
                P.verify_repeat(self.root, self.root, self.arrays, self.plan)
        self.record['environment']['packages']['torch'] = '2.6.0+cu124'
        self.write_record()
        with self.assertRaises(ValueError):
            P.verify_repeat(self.root, self.root, self.arrays, self.plan)

    def test_repeat_rejects_changed_bytes_and_vectors(self):
        changed = {key: value.copy() for key, value in self.arrays.items()}
        changed['core_embed_full'][0, 0] = .99
        with self.assertRaises(ValueError):
            P.verify_repeat(self.root, self.root, changed, self.plan)
        np.savez(self.root / 'probe_vectors.npz', **changed)
        with self.assertRaises(ValueError):
            P.verify_repeat(self.root, self.root, changed, self.plan)

    def test_plan_must_equal_committed_bytes_and_exposed_scope(self):
        path = self.root / 'plan.json'
        payload = json.dumps(self.plan).encode()
        path.write_bytes(payload)
        with patch.object(P.subprocess, 'run', return_value=types.SimpleNamespace(stdout=payload)):
            self.assertEqual(P.committed_plan(path, 'a' * 40), self.plan)
            path.write_bytes(payload + b'\n')
            with self.assertRaises(ValueError):
                P.committed_plan(path, 'a' * 40)
        self.plan['scope'] = 'sealed-confirmation'
        payload = json.dumps(self.plan).encode()
        path.write_bytes(payload)
        with patch.object(P.subprocess, 'run', return_value=types.SimpleNamespace(stdout=payload)):
            with self.assertRaises(ValueError):
                P.committed_plan(path, 'a' * 40)

    def test_reranker_repeat_preserves_actual_pair_order_and_batch4(self):
        observed = []
        def score(rows):
            observed.append(rows)
            return np.asarray([image for text, image in rows], dtype=float)
        queries = [f'q{i}' for i in range(8)]
        images = [lambda i=i: i for i in range(8)]
        actual = P.verify_reranker_repeat(types.SimpleNamespace(score=score), queries, images, np.arange(8))
        np.testing.assert_array_equal(actual, np.arange(8))
        self.assertEqual(observed, [list(zip(queries[:4], range(4))), list(zip(queries[4:], range(4, 8)))])
        with self.assertRaises(ValueError):
            P.verify_reranker_repeat(types.SimpleNamespace(score=score), queries, images, np.arange(8) + .0625)


if __name__ == '__main__':
    unittest.main()
