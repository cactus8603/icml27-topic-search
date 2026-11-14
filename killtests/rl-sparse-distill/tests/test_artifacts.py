"""Synthetic integrity regressions. These tests do not read real scored artifacts."""

from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import aggregate as agg
import artifacts


class ArtifactTests(unittest.TestCase):
    def test_exact_counts_and_finish_semantics(self):
        valid = {"a": {"correct": [1, 0], "finished": [1, 0], "ntok": [4, 10]}}
        artifacts.validate_records(valid, ["a"], 2, 10, False)
        invalid = (
            {"correct": [1, 0, 0]}, {"finished": [1]}, {"correct": [1, 1]},
            {"ntok": [4, -1]}, {"n_repl": [0, 11]}, {"finished": [True, 0]},
        )
        for fragment in invalid:
            with self.subTest(fragment=fragment), self.assertRaises(artifacts.ArtifactError):
                artifacts.validate_records({"a": dict(valid["a"], **fragment)}, ["a"], 2, 10, False)
        with self.assertRaises(artifacts.ArtifactError):
            artifacts.validate_records(valid, ["a", "b"], 2, 10, False)

    def test_gate0_does_not_read_later_conditions(self):
        crit, pc = agg.cfg("criteria.json"), agg.cfg("pairs.json")
        crit["eval"]["test_questions_min"] = 1
        rows = [{"id": "a", "split": "test"}]
        untouched = Path("unused-synthetic-results")
        with patch.object(agg, "check_gate_minus1", return_value={}), patch.object(agg, "load_verified", return_value=None) as load:
            report = agg.evaluate(untouched, ["B"], "0", crit, pc, rows)
            self.assertEqual(set(report), {"phase", "partial", "pairs"})
            self.assertEqual(set(report["pairs"]), {"B"})
            self.assertEqual([call.args[2] for call in load.call_args_list], ["base", "rl", "base_greedy", "rl_greedy"])

    def test_gate0_requires_correctness(self):
        crit, pc = agg.cfg("criteria.json"), agg.cfg("pairs.json")
        crit["eval"]["test_questions_min"] = 1
        rows = [{"id": "a", "split": "test"}]
        with patch.object(agg, "check_gate_minus1", side_effect=artifacts.ArtifactError("missing")), patch.object(agg, "load_verified") as load:
            report = agg.evaluate(Path("unused-synthetic-results"), ["A"], "0", crit, pc, rows)
            self.assertEqual(report["pairs"]["A"]["gate_minus1"]["verdict"], "未能測試")
            load.assert_not_called()

    def test_primary_pair_cannot_skip_a(self):
        crit = agg.cfg("criteria.json")
        good = {"gate0": {"verdict": "配對合格"}}
        self.assertEqual(agg.phenomenon_eval({"A": good, "B": good}, crit)["primary_pair"], "A")
        self.assertIsNone(agg.phenomenon_eval({"B": good}, crit)["primary_pair"])
        ruled_out = {"gate0": {"verdict": "配對不合格", "reproduce_ok": True, "trunc_ok": True}}
        self.assertEqual(agg.phenomenon_eval({"A": ruled_out, "B": good}, crit)["primary_pair"], "B")

    def test_hf_seed_uses_question_offset_and_matches_sharding(self):
        self.assertEqual([artifacts.hf_batch_seed(1234, 0, offset) for offset in range(0, 6, 2)], [1234, 1236, 1238])
        self.assertEqual(artifacts.hf_batch_seed(1234, 3, 4), 4238)
        one = [(Path("raw_0"), Path("meta_0"), ["a", "b"])]
        two = [(Path("raw_0"), Path("meta_0"), ["a"]), (Path("raw_1"), Path("meta_1"), ["b"])]
        meta = {"provenance": {"engine_version": "same"}}
        with patch.object(artifacts, "check_run", side_effect=[(one, 128), (one, 128)]), patch.object(artifacts, "read_json", return_value=meta):
            artifacts.require_hf_pairing(Path("unused"), "A", {}, {}, [], {})
        with patch.object(artifacts, "check_run", side_effect=[(one, 128), (two, 128)]), patch.object(artifacts, "read_json", return_value=meta):
            with self.assertRaises(artifacts.ArtifactError):
                artifacts.require_hf_pairing(Path("unused"), "A", {}, {}, [], {})

    def test_historical_gate_only_authorizes_existing_vllm_artifacts(self):
        old = {"source": "historical_batch_log", "hf_decode_checks": False, "geo_mix_checks": False}
        with self.assertRaises(artifacts.ArtifactError):
            artifacts.require_current_correctness(old, "base")
        with self.assertRaises(artifacts.ArtifactError):
            artifacts.check_run(Path("unused"), "A", "rl_replace", {"temperature": .6, "top_p": .95}, {"n_samples": 128}, [], old)
        current = {"source": "receipt", "hf_decode_checks": True, "geo_mix_checks": False}
        artifacts.require_current_correctness(current, "rl_replace")
        with self.assertRaises(artifacts.ArtifactError):
            artifacts.require_current_correctness(current, "geo_mix")

    def test_failing_smoke_pipeline_preserves_receipt(self):
        launcher = Path(__file__).resolve().parents[1] / "slurm/00_correctness_gpu.sbatch"
        run_script = launcher.read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent, prefix="mock-correctness-") as tmp:
            root = Path(tmp)
            (root / "results").mkdir()
            old = root / "results/correctness_A.json"
            old.write_text("historical receipt", encoding="utf-8")
            mock = root / "mock_python"
            mock.write_text("#!/usr/bin/env bash\nif [[ $1 == -I ]]; then echo 'FAIL T1'; exit 7; fi\nprintf 'invalid receipt' > results/correctness_A.json\n", encoding="utf-8")
            mock.chmod(0o700)
            lines = run_script.splitlines()
            lines = [f'PY="{mock}" BASE=synthetic-base RL=synthetic-rl' if line.startswith("source ") else line for line in lines]
            result = subprocess.run(["bash", "-s", "A"], input="\n".join(lines), text=True, cwd=root,
                                    env=dict(os.environ, SLURM_JOB_ID="123"), capture_output=True)
            self.assertEqual(result.returncode, 7, result.stderr)
            self.assertIn("FAIL T1", result.stdout)
            self.assertEqual(old.read_text(encoding="utf-8"), "historical receipt")

    def test_per_prompt_context_under_padding_and_compaction(self):
        import gen_replace as gen
        import torch

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.nn.Parameter(torch.zeros(1))

        class Cache:
            def batch_select_indices(self, _):
                pass

        class Tokenizer:
            def __call__(self, texts, **_):
                lengths = list(map(len, texts))
                tokens = [[0] * (max(lengths) - length) + [2] * length for length in lengths]
                ids = torch.tensor(tokens)
                return {"input_ids": ids, "attention_mask": (ids > 0).long()}

            def decode(self, tokens, **_):
                return str(tokens)

        def fwd(_model, ids, _attn, _pos, cache):
            # 每列的第五個生成步才結束。長 prompt 的預算只有 3。
            if not hasattr(cache, "t"):
                cache.t = 0
            next_tok = 3 if cache.t == 4 else 2
            cache.t += 1
            return torch.tensor([next_tok] * len(ids)), cache

        for compact in (False, True):
            with patch.object(gen, "DynamicCache", Cache), patch.object(gen, "fwd", side_effect=fwd), \
                    patch.object(gen, "STOP_IDS", [3]), patch.object(gen, "sample_from_logits", side_effect=lambda logits, *a, **k: logits):
                output = gen.decode_batch(Tokenizer(), Model(), None, ["a", "bbbbbbb", "cccccccccc"], 1, "base", .6, .95,
                                          5, 10, torch.Generator(), compact=compact)
            self.assertEqual([(finished, n) for _, finished, n, _ in output], [(True, 5), (False, 3), (False, 0)])


if __name__ == "__main__":
    unittest.main()
