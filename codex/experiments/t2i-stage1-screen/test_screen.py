"""Exercise the CPU-only screen with 168,000 constructed score rows."""

import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import screen

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / screen.TOPIC / "scripts"))
import evidence
from common import load_json, prompts_by_id
from probe import CLIP_REPO, CLIP_REV, FAL_REPO, FAL_REV
from stage1_check import REVISIONS
from stage_selection import stage_candidates


FREEZE = "a" * 40
CRITERIA = load_json("criteria.json")
PROMPTS = prompts_by_id()
MODELS = load_json("models.json")["models"]


def construct_panel(root):
    operations = root / "operations"
    operations.mkdir()
    strong = CRITERIA["A"]["strong_erasers"][0]
    revisions = REVISIONS | {CLIP_REPO: CLIP_REV, FAL_REPO: FAL_REV}
    for index, model in enumerate(screen.launcher.MODELS):
        job = str(701 + index)
        receipt = {"scope": screen.launcher.SCOPE, "launch_commit": FREEZE, "slurm_job_id": job,
                   "slurm_array_job_id": "700", "model": model, "score_chunks": 56,
                   "config_sha256": evidence.config_sha(), "proxy_receipt": screen.launcher.PROXY_CANDIDATE,
                   "resolved_hf_revisions": revisions, "prerequisites": {"gate_a_sha256": "b" * 64, "bench": {"pass": True}}}
        (operations / f"{model}-{job}.json").write_text(json.dumps(receipt))
        for prompt, spec in PROMPTS.items():
            path = root / "scores" / model / prompt
            path.mkdir(parents=True)
            for start in range(0, 2000, 500):
                stop = start + 500
                scores = np.full(500, 0.01, np.float32)
                hits = {(strong, "E1"): 2, ("sd14", "B1"): 1, ("sd14", "N1"): 3}.get((model, prompt), 0)
                if start == 0:
                    scores[:hits] = 0.95
                chunk = path / f"s{start:07d}_{stop:07d}.npz"
                np.savez_compressed(chunk, seeds=np.arange(start, stop), **dict.fromkeys(evidence.DETECTORS, scores))
                metadata = {
                    "model": model, "prompt_id": prompt, "prompt": spec["text"], "seed_range": [start, stop],
                    "batch_size": 32, "seconds": 0.5, "slurm_job": "768" if (model, prompt, start) == ("sd14", "B1", 0) else job,
                    "scheduler": "DDIM", "eta": 0.0, "steps": 25, "guidance_scale": 7.5, "negative_prompt": "",
                    "height": 512, "width": 512, "dtype": "torch.float16", "nudenet_provider": "CPUExecutionProvider",
                    "nudenet": "3.4.2", "onnxruntime": "1.32.0", "diffusers": "0.31.0", "transformers": "4.46.3",
                    "torch": "2.5.1", "load_info": {"model": model, "kind": MODELS[model]["kind"]},
                    "detectors": evidence.DETECTORS,
                }
                chunk.with_suffix(".json").write_text(json.dumps(metadata))


def accounting():
    lines = [f"768|699_0|{screen.JOB_NAME}|FAILED|1:0|10|gres/gpu=1|2026-11-15T07:00:00"]
    lines += [f"{701 + index}|700_{index}|{screen.JOB_NAME}|COMPLETED|0:0|100|gres/gpu=1|2026-11-15T08:00:00"
              for index in range(6)]
    return "\n".join(lines)


class StageScreenTest(unittest.TestCase):
    def test_ledger_and_partial_allocation_are_checked(self):
        ledger = f"[codex] 2026-11-15 08:00 submit 700 t2i-rare-failure candidate stage 1 {FREEZE[:7]}"
        self.assertEqual(screen.registered_arrays(ledger, FREEZE), ["700"])
        with self.assertRaisesRegex(ValueError, "duplicated"):
            screen.registered_arrays(ledger + "\n" + ledger, FREEZE)
        with self.assertRaisesRegex(ValueError, "non-terminal"):
            screen.accounting_rows(accounting().replace("COMPLETED", "RUNNING", 1), ["699", "700"])
        with self.assertRaisesRegex(ValueError, "wrong allocation name"):
            screen.accounting_rows(accounting().replace(screen.JOB_NAME, "other-name", 1), ["699", "700"])
        cancelled = f"698|698_[0-5]|{screen.JOB_NAME}|CANCELLED by 1|0:0|0||Unknown"
        observed = screen.accounting_rows(accounting() + "\n" + cancelled, ["698", "699", "700"])
        self.assertEqual(screen.summarize_cost(observed, 0, 0)["allocated_gpu_seconds"], 610)

    def test_complete_panel_reuses_frozen_selection_and_counts_failed_runtime(self):
        arrays = ["699", "700"]
        attempts = screen.accounting_rows(accounting(), arrays)
        with TemporaryDirectory(prefix="t2i-screen-synthetic.", dir=os.environ["TMPDIR"]) as directory:
            root = Path(directory).resolve()
            construct_panel(root)
            observed = screen.audit(root, FREEZE, arrays, attempts, evidence, CRITERIA, PROMPTS, stage_candidates)
            self.assertEqual((observed["rows"], observed["chunks"]), (168000, 336))
            self.assertEqual(observed["cost"]["allocated_gpu_seconds"], 610)
            self.assertEqual(observed["cost"]["retained_chunk_seconds"], 168)
            self.assertEqual(observed["scope"], screen.SCOPE)
            strong = CRITERIA["A"]["strong_erasers"][0]
            self.assertEqual(observed["screen_cells"][:2], [{"model": strong, "prompt": "E1"},
                                                             {"model": "sd14", "prompt": "B1"}])
            self.assertEqual(len(observed["screen_cells"]), 12)
            self.assertNotIn({"model": "sd14", "prompt": "N1"}, observed["screen_cells"])
            altered = root / "scores/sd14/B1/s0000000_0000500.json"
            body = altered.read_text()
            altered.write_text(body.replace('"slurm_job": "768"', '"slurm_job": "702"'))
            with self.assertRaisesRegex(ValueError, "producer"):
                screen.audit(root, FREEZE, arrays, attempts, evidence, CRITERIA, PROMPTS, stage_candidates)
            altered.write_text(body)
            (root / "operations/sd14-701.json").unlink()
            with self.assertRaisesRegex(ValueError, "All six"):
                screen.audit(root, FREEZE, arrays, attempts, evidence, CRITERIA, PROMPTS, stage_candidates)


if __name__ == "__main__":
    unittest.main()
