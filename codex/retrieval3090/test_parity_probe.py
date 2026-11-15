"""[codex] Synthetic CPU checks for fixed-batch parity sampling and alignment."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import parity_probe as P


class ParityProbeTests(unittest.TestCase):
    def setUp(self):
        temporary_root = Path(__file__).resolve().parents[2] / "hf_home/tmp"
        temporary_root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="parity-probe-unit-", dir=temporary_root)
        self.root = Path(self.temp.name)
        self.cache = self.root / "cache"
        (self.cache / "checkpoints").mkdir(parents=True)
        self.panel = {"ids": list(range(1, 65)), "queries": [{"text": f"original query {i}", "key": [i + 1, i + 2]} for i in range(32)]}
        self.config = {"batch_size": {"embed_image": 8, "embed_text": 32, "cross_pair": 4}}
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text(json.dumps(self.panel))
        self.identity = {"config": self.config, "panel_sha256": P.sha(self.manifest), "source": "synthetic"}
        (self.cache / "checkpoint-provenance.json").write_text(json.dumps(self.identity))
        rng = np.random.default_rng(12)
        unit = lambda x: (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)
        self.image = unit(rng.normal(size=(64, 8)))
        self.full = unit(rng.normal(size=(32, 8)))
        captions = [q["text"] for q in self.panel["queries"]]
        self.write_array("core_embed_image", self.image, self.panel["ids"], 8,
                         {"stage": "core_embed_image", "order": self.panel["ids"]})
        self.write_array("core_embed_full", self.full, captions, 32,
                         {"stage": "core_embed_full", "order": captions})
        self.order = np.argsort(-(self.full @ self.image.T), axis=1, kind="stable")
        keys = [self.panel["ids"][int(i)] for i in self.order[:, :50].flat]
        self.write_array("cross_top50", np.arange(1600, dtype=float), keys, 4,
                         {"stage": "top50", "queries": [text for text in captions for _ in range(50)], "images": keys})

    def tearDown(self):
        self.temp.cleanup()

    def write_array(self, name, value, items, batch_size, stage):
        path = self.cache / "checkpoints" / (name + ".npy")
        np.save(path, value)
        state = {"fingerprint": P.fingerprint({"identity": self.identity | stage,
                                               "items": len(items), "batch_size": batch_size}),
                 "completed": len(items), "array_shape": list(value.shape), "dtype": value.dtype.str, "sha256": P.sha(path)}
        path.with_suffix(".state.json").write_text(json.dumps(state))

    def test_prefix_sampler_preserves_original_batch_composition(self):
        sample = P.select_batches(self.panel, self.config)
        self.assertEqual(sample["image_ids"], list(range(1, 17)))
        self.assertEqual(sample["image_batches"], [list(range(8)), list(range(8, 16))])
        self.assertEqual(sample["query_batches"], [list(range(32))])
        self.assertEqual(sample["query_texts"], [q["text"] for q in self.panel["queries"]])
        self.assertEqual(sample["cross_batches"], [[0, 1, 2, 3], [4, 5, 6, 7]])
        changed = {"batch_size": dict(self.config["batch_size"], embed_text=16)}
        with self.assertRaisesRegex(ValueError, "8/32/4"):
            P.select_batches(self.panel, changed)

    def test_reference_cross_pairs_are_exact_old_flattened_context(self):
        reference = P.cpu_reference(self.cache, self.manifest)
        pairs = reference["cross_pairs"]
        self.assertEqual([row["flat_index"] for row in pairs], list(range(8)))
        self.assertEqual([row["query_index"] for row in pairs], [0] * 8)
        self.assertEqual([row["text"] for row in pairs], ["original query 0"] * 8)
        self.assertEqual([row["image_id"] for row in pairs], [self.panel["ids"][int(i)] for i in self.order[0, :8]])
        self.assertTrue(reference["checkpoints"]["cross_top50"]["stored_array_digest_available"])

    def test_wrong_cross_candidate_order_fails_before_any_model_load(self):
        state_path = self.cache / "checkpoints/cross_top50.state.json"
        state = json.loads(state_path.read_text())
        state["fingerprint"] = "0" * 64
        state_path.write_text(json.dumps(state))
        with self.assertRaisesRegex(ValueError, "order/context"):
            P.cpu_reference(self.cache, self.manifest)

    def test_changed_manifest_or_query_context_is_rejected(self):
        panel = dict(self.panel, ids=list(reversed(self.panel["ids"])))
        self.manifest.write_text(json.dumps(panel))
        with self.assertRaisesRegex(ValueError, "manifest changed"):
            P.cpu_reference(self.cache, self.manifest)

    def test_changed_array_is_rejected_by_completed_digest(self):
        path = self.cache / "checkpoints/core_embed_image.npy"
        value = np.load(path)
        value[0, 0] += np.float32(.001)
        np.save(path, value)
        with self.assertRaisesRegex(ValueError, "SHA differs"):
            P.cpu_reference(self.cache, self.manifest)

    def test_cpu_comparison_has_no_relevance_or_effectiveness_metrics(self):
        path = self.root / "probe.npz"
        np.savez(path, core_embed_image=self.image[:16], core_embed_full=self.full[:32])
        report = P.cpu_probe_comparison(self.cache, path)
        self.assertEqual(report["image_vectors"]["max_abs_difference"], 0)
        self.assertEqual(report["full_query_vectors"]["different_entries"], 0)
        self.assertEqual(report["sampled_gallery_32x16"]["different_rank_entries"], 0)
        self.assertEqual(report["new_text_against_OLD_full_gallery"]["queries_with_changed_prefix_membership"], 0)
        self.assertIn("no relevance", report["scope"])
        self.assertNotIn("map_pp", report)
        changed = self.full[:32].copy()
        changed[0] = np.roll(changed[0], 1)
        np.savez(path, core_embed_image=self.image[:16], core_embed_full=changed)
        self.assertGreater(P.cpu_probe_comparison(self.cache, path)["full_query_vectors"]["max_abs_difference"], 0)

    def test_cpu_subprocess_forces_one_thread_and_no_gpu(self):
        report = P.cpu_worker("reference", self.cache, self.manifest)
        self.assertEqual(report["cpu_threads"], {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
        self.assertEqual(report["sample"]["image_ids"], list(range(1, 17)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
