"""Exercise the repaired Gate (b) receipt with synthetic score chunks only."""

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[2] / "killtests/t2i-rare-failure/scripts")]
import run
import verify_gate_b
from clip_precision_probe import probabilities
from common import load_json, prompts_by_id


FREEZE = "1" * 40
JOB = "123"
PROVENANCE = {
    "scope": run.SCOPE, "model": "sd14", "prompts": ["E1", "N1", "B1"],
    "precision": "fp16_features_fp32_logits", "sd_revision": "a" * 40, "clip_revision": "b" * 40,
    "versions": {"torch": "2.5.1", "diffusers": "0.31.0", "transformers": "4.46.3"},
    "stage1_activated": False,
}


def mock_chunk(root, prompt, start, end):
    sampler = load_json("sampling.json")
    meta = {k: sampler[k] for k in ("scheduler", "eta", "steps", "guidance_scale", "height", "width", "negative_prompt")}
    meta.update(run.stamp(PROVENANCE, FREEZE), **PROVENANCE["versions"],
                model="sd14", load_info={"model": "sd14", "kind": "base"}, prompt_id=prompt,
                prompt=prompts_by_id()[prompt]["text"], dtype="torch.float16", seed_range=[start, end],
                batch_size=run.BATCH, slurm_job=JOB, detectors=["clip"], images_saved=0)
    path = root / "scores/sd14" / prompt / f"s{start:07d}_{end:07d}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".json").write_text(json.dumps(meta))
    seeds = np.arange(start, end, dtype=np.int64)
    clip = seeds.astype(np.float32) / np.float32(100_000) + np.float32(0.1)
    if prompt == "B1":
        clip[seeds >= 9_900] = np.float32(0.99)
    nan = np.full(end - start, np.nan, np.float32)
    np.savez_compressed(path, seeds=seeds, clip=clip, nudenet=nan, falconsai=nan)
    return path


class ReplayTest(unittest.TestCase):
    def test_original_arguments_and_repaired_detector(self):
        import detectors
        import torch

        features = torch.tensor([[0.4, 0.8]], dtype=torch.float16)
        text = torch.tensor([[0.25, 0.75], [0.75, 0.25]], dtype=torch.float16)
        log_scale = torch.tensor(1.0, dtype=torch.float16)
        clip_snapshot = "/pinned/clip"
        reference = SimpleNamespace(image_features=lambda *_: features, probabilities=probabilities)
        tf32_before = torch.backends.cuda.matmul.allow_tf32

        def synthetic_init(detector, device, dtype, which, *, clip_snapshot):
            self.assertEqual((device, dtype, tuple(which), clip_snapshot),
                             ("cuda", torch.float16, ("clip",), "/pinned/clip"))
            detector.tf = text
            detector.clip = SimpleNamespace(logit_scale=log_scale)

        def synthetic_generator(script, run_name):
            self.assertEqual(run_name, "__main__")
            self.assertEqual(sys.argv, [script, "--model", "sd14", "--prompts", "E1", "N1", "B1",
                                        "--seed-start", "0", "--seed-end", "10000", "--batch-size", "32",
                                        "--chunk", "500", "--detectors", "clip", "--no-save-images",
                                        "--out", "/isolated/replay"])
            detector = detectors.Detectors("cuda", torch.float16, ["clip"])
            self.assertNotIn("slurm_job", detector.versions)  # Frozen generator supplies this field.
            result = detector.score(np.zeros((1, 1, 1, 3), dtype=np.uint8))
            reference_scores = probabilities(features, text, log_scale.exp(), accumulate_fp32=True).numpy()
            np.testing.assert_array_equal(result["clip"], reference_scores)
            self.assertTrue(np.isnan(result["nudenet"]).all() and np.isnan(result["falconsai"]).all())
            self.assertEqual(torch.backends.cuda.matmul.allow_tf32, tf32_before)

        with (patch.object(detectors.Detectors, "__init__", synthetic_init),
              patch.object(torch.cuda, "is_available", return_value=True),
              patch.object(torch.cuda, "get_device_name", return_value="H200"),
              patch.object(run.runpy, "run_path", side_effect=synthetic_generator)):
            run.generate(Path("/frozen"), reference, PROVENANCE, clip_snapshot, Path("/isolated/replay"), FREEZE)

    def test_registered_corpus_reaches_original_verifier(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            for prompt in PROVENANCE["prompts"]:
                for start in range(0, run.SEED_COUNT, run.CHUNK):
                    mock_chunk(root, prompt, start, start + run.CHUNK)
            receipt = run.verify(root, verify_gate_b, PROVENANCE, FREEZE, JOB)
            self.assertEqual(receipt["status"], "PASS")
            self.assertEqual(receipt["rows"][-1]["hits"]["0.99"], 100)
            self.assertEqual(receipt["score_chunk_count"], 60)
            self.assertEqual(receipt["frozen_commit"], FREEZE)
            self.assertFalse(receipt["stage1_activated"])
            self.assertRegex(receipt["source_sha256"], r"^[0-9a-f]{64}$")
            (root / "scores/sd14/E1/s0000000_0000500.json").unlink()
            with self.assertRaisesRegex(ValueError, "exactly the registered score chunks"):
                run.verify(root, verify_gate_b, PROVENANCE, FREEZE, JOB)

    def test_changed_metadata_or_disabled_detector_is_rejected(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            path = mock_chunk(root, "E1", 0, 500)
            stamp = {**run.stamp(PROVENANCE, FREEZE), "slurm_job": JOB}
            run.validate_chunk(path, 0, 500, stamp, PROVENANCE["versions"])
            with self.assertRaisesRegex(ValueError, "Replay metadata differs"):
                run.validate_chunk(path, 0, 500, {**stamp, "clip_precision": "fp16"}, PROVENANCE["versions"])
            with np.load(path, allow_pickle=False) as raw:
                arrays = dict(raw)
            arrays["nudenet"].fill(0)
            np.savez_compressed(path, **arrays)
            with self.assertRaisesRegex(ValueError, "empty placeholder"):
                run.validate_chunk(path, 0, 500, stamp, PROVENANCE["versions"])


if __name__ == "__main__":
    unittest.main()
