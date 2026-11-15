"""Synthetic integrity regressions. These tests do not read real scored artifacts."""

from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

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

    def test_geo_pairing_requires_same_layout_and_engine(self):
        reference = [(Path("raw_0"), Path("hf_meta_0"), ["a", "b"])]
        candidate = [(Path("raw_0"), Path("geo_meta_0"), ["a", "b"])]
        reordered = [(Path("raw_0"), Path("geo_meta_0"), ["b", "a"])]
        shards = [(Path("raw_0"), Path("geo_meta_0"), ["a"]), (Path("raw_1"), Path("geo_meta_1"), ["b"])]
        for layout, geo_version, valid in ((candidate, "hf", True), (reordered, "hf", False),
                                           (shards, "hf", False), (candidate, "other", False)):
            with self.subTest(layout=layout, geo_version=geo_version):
                def metadata(path):
                    return {"provenance": {"engine_version": geo_version if path.name.startswith("geo") else "hf"}}

                with patch.object(artifacts, "check_run", side_effect=[(reference, 128), (layout, 128)]) as runs, \
                        patch.object(artifacts, "read_json", side_effect=metadata):
                    args = (Path("unused"), "A", {}, {}, [], {})
                    if valid:
                        artifacts.require_hf_pairing(*args, candidate="geo_mix")
                    else:
                        with self.assertRaisesRegex(artifacts.ArtifactError, "rl_hf 與 geo_mix"):
                            artifacts.require_hf_pairing(*args, candidate="geo_mix")
                    self.assertEqual([call.args[2] for call in runs.call_args_list], ["rl_hf", "geo_mix"])

    def test_formal_evaluation_rejects_unpaired_r3_without_evaluating_it(self):
        crit, pc = agg.cfg("criteria.json"), agg.cfg("pairs.json")
        crit["eval"]["test_questions_min"] = 1
        rows = [{"id": "a", "split": "test"}]
        graded = {"a": {"correct": [0], "finished": [1], "ntok": [4]}}
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            results = Path(tmp)
            (results / "A" / "geo_mix").mkdir(parents=True)
            with patch.object(agg, "check_gate_minus1", return_value={}), \
                    patch.object(agg, "load_verified", return_value=graded), \
                    patch.object(agg, "gate0", return_value={"verdict": "配對合格"}), \
                    patch.object(agg, "signal_eval", return_value={"verdict": "PASS"}), \
                    patch.object(agg, "space_eval", return_value={"verdict": "FAIL"}), \
                    patch.object(agg, "gate1", return_value={}), \
                    patch.object(agg, "require_hf_pairing", side_effect=[None, artifacts.ArtifactError("geo mismatch")]) as pairing, \
                    patch.object(agg, "probe2_eval") as probe:
                report = agg.evaluate(results, None, "1", crit, pc, rows)
            self.assertEqual(pairing.call_args.kwargs, {"reference": "rl_hf", "candidate": "geo_mix"})
            probe.assert_not_called()
            self.assertEqual(report["A"]["artifact_issues"]["geo_mix"], "geo mismatch")
            self.assertTrue(report["r3"]["result"]["verdict"].startswith("未能測試"))
            self.assertTrue(report["overall"].startswith("未能測試"))

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


class MatchedFrontierTests(unittest.TestCase):
    def setUp(self):
        self.crit = agg.cfg("criteria.json")
        self.crit["eval"]["bootstrap_B"] = 50

    def constant(self, p1, pk):
        return {"p1": np.full(5, p1), "pk": np.full(5, pk)}

    def world(self, rl, rlh, replace, middle):
        pq = {"base": self.constant(0, 0), "rl": self.constant(*rl),
              "rl_hf": self.constant(*rlh), "rl_replace": self.constant(*replace)}
        pq.update({f"wise_{lam}": self.constant(*middle) for lam in self.crit["gate1"]["wise_lambdas"]})
        return pq

    def test_outside_frontier_cannot_cause_overall_failure(self):
        pq = self.world((1 / 128, .57), (1 / 128, .50), (2 / 128, .515), (0, .50))
        signal, space = agg.signal_eval(pq, self.crit), agg.space_eval(pq, self.crit)
        self.assertEqual(signal["verdict"], "灰區")
        self.assertTrue(space["H_a"]["frontier_clamped"])
        self.assertLess(space["H_a"]["pt_pp"], 0)
        self.assertTrue(space["H_b"]["usable"])
        self.assertLess(space["H_b"]["upper_pp"], 2)
        self.assertTrue(space["verdict"].startswith("未能測試"))
        overall = agg.overall_eval({"verdict": "PASS"}, signal, space, {"triggered": False})
        self.assertTrue(overall["overall"].startswith("未能測試"))

    def test_supported_path_can_pass_if_other_endpoint_is_outside(self):
        pq = self.world((.489, .80), (.50, .80), (.488, .95), (.25, .80))
        space = agg.space_eval(pq, self.crit)
        self.assertEqual(space["H_b"]["pi_star"], 1)
        self.assertEqual(space["H_b"]["endpoints_in_range"], {"rl": False, "rl_hf": True})
        self.assertFalse(space["H_b"]["usable"])
        self.assertEqual(space["paths_passing"], ["H_a"])
        self.assertEqual(space["verdict"], "PASS")

    def test_unsupported_bootstrap_draws_widen_both_tails(self):
        pts = [("0", {"p1": np.zeros(2), "pk": np.zeros(2)}),
               ("1", {"p1": np.array([.5, .9]), "pk": np.array([.75, .85])})]
        target = {"p1": np.array([.6, .6]), "pk": np.array([.9, .9])}
        idx = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
        gap = agg.wise_margin_boot(target, pts, idx, .125)
        self.assertTrue(gap["usable"])
        self.assertEqual(gap["bootstrap_supported_draws"], 3)
        self.assertLess(gap["lo_pp"], 0)
        self.assertGreater(gap["lo_best_case_pp"], 0)
        self.assertGreater(gap["upper_pp"], 50)

    def test_r3_cannot_fail_from_unknown_comparison_or_indeterminate_bootstrap(self):
        pq = self.world((.5, .75), (.5, .75), (.4, .77), (.3, .70))
        pq["geo_mix"] = self.constant(.6, .9)
        result = agg.probe2_eval(pq, self.crit["gate1"], self.crit["probe2"], self.crit["eval"])
        self.assertTrue(result["cond1_ok"] and result["cond3_ok"])
        self.assertFalse(result["cond2_p1_in_wise_range"])
        self.assertIsNone(result["cond2_ok"])
        self.assertTrue(result["verdict"].startswith("未能測試"))

        pq = {term: {stat: values[:2] for stat, values in q.items()} for term, q in pq.items()}
        pq["rl"]["p1"] = np.array([.5, .9])
        idx = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
        with patch.object(agg, "boot_idx", return_value=idx):
            result = agg.probe2_eval(pq, self.crit["gate1"], self.crit["probe2"], self.crit["eval"])
        self.assertTrue(result["cond2_p1_in_wise_range"])
        self.assertTrue(result["verdict"].startswith("未能測試"))
        self.assertEqual(result["cond2_frontier"]["bootstrap_supported_draws"], 3)


if __name__ == "__main__":
    unittest.main()
