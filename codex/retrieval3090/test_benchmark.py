"""[codex] Synthetic benchmark integrity tests; no models, GPU, or study files."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark as B


def synthetic():
    rng = np.random.default_rng(31)
    unit = lambda shape: (lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True))(rng.normal(size=shape))
    cache = {f"{name}_{part}": unit(shape) for name in B.MODELS for part, shape in
             (("image", (220, 12)), ("full", (10, 12)), ("condition", (20, 12)))}
    index = np.arange(20).reshape(10, 2)
    return cache, index


class FakeClock:
    def __init__(self):
        self.value = 0.0
    def __call__(self):
        return self.value
    def advance(self, seconds):
        self.value += seconds


class BenchmarkTests(unittest.TestCase):
    def test_missing_platform_opt_in_preserves_original_replay_gate(self):
        self.assertIsNone(B.validate_benchmark_context(Path("cache"), {"replay": {"state": "PASS"}}))
        with self.assertRaisesRegex(ValueError, "must pass replay"):
            B.validate_benchmark_context(Path("cache"), {"replay": {"state": "MISMATCH"}})
        with self.assertRaisesRegex(ValueError, "together"):
            B.validate_benchmark_context(Path("cache"), {"replay": {"state": "MISMATCH"}}, Path("plan"), None)

    def test_platform_context_binds_runtime_supplements_and_resume_study(self):
        environment = {"torch": "synthetic-runtime", "device": "synthetic-device"}
        context = {"mode": "platform-relative-exploration", "formal_evidence": False,
            "published_reproduction": {"state": "MISMATCH", "metrics": {"map_pp": 1.}},
            "platform_consistency": "VALID_LOCAL", "plan_sha256": "a" * 64,
            "receipt_sha256": "b" * 64, "environment": environment,
            "baseline": {"map_pp": 1.}, "supplementary_artifacts": [{"sha256": "c" * 64}]}
        called = []
        def validate(cache, plan, receipt, *, supplementary_paths):
            called.append((cache, plan, receipt, supplementary_paths))
            return context
        validator = SimpleNamespace(validate_context=validate, runtime_environment=lambda: dict(environment))
        meta = {"replay": {"state": "MISMATCH"}}
        with patch.dict(sys.modules, {"platform_context": validator}):
            actual = B.validate_benchmark_context(Path("cache"), meta, Path("plan"), Path("receipt"),
                supplementary_paths=(Path("lcse.npz"), Path("calibration.json")))
        self.assertIs(actual, context)
        self.assertEqual(called[0][3], (Path("lcse.npz").resolve(), Path("calibration.json").resolve()))
        self.assertEqual(meta["replay"]["state"], "MISMATCH")
        study = B.study_identity(actual)
        self.assertFalse(study["formal_evidence"])
        self.assertEqual(study["platform_context"], context)
        self.assertEqual(study["platform_plan_sha256"], "a" * 64)
        self.assertEqual(study["platform_receipt_sha256"], "b" * 64)
        self.assertIn("local-relative", study["comparison_scope"])
        directory = Path(__file__).resolve().parents[2] / "hf_home/tmp"
        directory.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=directory) as temporary:
            B.Store(temporary, {"study": study})
            changed = dict(study, platform_receipt_sha256="d" * 64)
            with self.assertRaisesRegex(ValueError, "provenance"):
                B.Store(temporary, {"study": changed})

    def test_platform_runtime_mismatch_or_absent_validation_cannot_bypass_gate(self):
        context = {"mode": "platform-relative-exploration", "formal_evidence": False,
            "platform_consistency": "VALID_LOCAL", "published_reproduction": {"state": "MISMATCH"},
            "environment": {"torch": "planned"}}
        validator = SimpleNamespace(validate_context=lambda *a, **k: context,
                                    runtime_environment=lambda: {"torch": "different"})
        with patch.dict(sys.modules, {"platform_context": validator}):
            with self.assertRaisesRegex(ValueError, "runtime"):
                B.validate_benchmark_context(Path("cache"), {"replay": {"state": "MISMATCH"}}, Path("plan"), Path("receipt"))
        validator.validate_context = lambda *a, **k: None
        with patch.dict(sys.modules, {"platform_context": validator}):
            with self.assertRaisesRegex(ValueError, "must pass replay"):
                B.validate_benchmark_context(Path("cache"), {"replay": {"state": "MISMATCH"}}, Path("plan"), Path("receipt"))

    def test_label_free_vector_alignment_rejects_bad_indexes_and_scaling(self):
        cache, index = synthetic()
        panel = {"ids": list(range(220)), "queries": [{}] * 10,
                 "condition_texts": [str(i) for i in range(20)], "condition_index": index.tolist()}
        np.testing.assert_array_equal(B.validate_vectors(cache, panel), index)
        changed = dict(panel, condition_index=(index + 1).tolist())
        with self.assertRaisesRegex(ValueError, "aligned"):
            B.validate_vectors(cache, changed)
        broken = dict(cache, core_embed_image=cache["core_embed_image"] * .5)
        with self.assertRaisesRegex(ValueError, "unit-normalized"):
            B.validate_vectors(broken, panel)

    def test_candidate_policy_ids_match_frozen_numeric_policies(self):
        import analysis_v2 as V
        cache, index = synthetic()
        features = B.cpu_features(cache, index, 0)
        order = np.argsort(-features["core_embed"]["full_raw"], kind="stable")
        self.assertEqual(len(B.policy_names()), 50)
        for depth in (100, 200):
            top = order[:depth]
            feats = {f"{name}_full": row["full_z"][top][None] for name, row in features.items() if name in B.MODELS}
            feats.update({f"{name}_cond": row["condition_z"][top][None] for name, row in features.items() if name in B.MODELS})
            for old_name, score in V.A.policies(feats).items():
                policy = f"m1/top{depth}/{V.corrected_policy_name(old_name)}"
                expected = V.A.admit(top[None], score)[0]
                np.testing.assert_array_equal(B.admit(policy, features, order), expected)
        with self.assertRaises(ValueError):
            B.admit("m1/top100/llr_core_t1", features, order)

    def test_logic_policy_cost_path_uses_verified_template_means_and_same_formula(self):
        import logic_baselines as L
        cache, index = synthetic()
        logic_vectors = {f"{model}_lcse_condition": cache[f"{model}_condition"] * .7 for model in B.MODELS}
        features = B.cpu_features(cache, index, 0, logic_vectors)
        order = np.argsort(-features["core_embed"]["full_raw"], kind="stable")
        params = {name: {"mu": .1, "beta": 10., "gamma": -1.} for name in B.MODELS}
        for model in B.MODELS:
            top = order[:100]
            expected = L.aggregate_scores(features[model]["full_raw"][top], features[model]["logic_condition_raw"][top], **params[model])
            for kind, score in expected.items():
                name = f"logic_{model}_train_presence_calibrated_{kind}"
                np.testing.assert_allclose(B.policy_scores(name, features, top, params), score)
        self.assertIn("cs_clip/four_template_mean_dot", features["component_seconds"])
        self.assertTrue(all(value >= 0 for value in features["component_seconds"].values()))

    def test_query_sampling_and_arm_order_are_fixed_and_not_budget_ascending(self):
        queries = [{"text": f"query {i}", "key": [i, i + 1]} for i in range(20)]
        picked = B.selected_queries(queries)
        self.assertEqual(len(picked), 8)
        self.assertEqual(picked, B.selected_queries(queries))
        tasks = [[f"b{budget}", rep] for budget in B.BUDGETS for rep in range(2)]
        order = B.task_order(tasks, "cross")
        self.assertEqual(sorted(order), sorted(tasks))
        self.assertNotEqual(order, tasks)
        encodes = B.encode_tasks("core_embed", picked)
        self.assertEqual(len(encodes), 8 * 2 * 3)

    def test_float32_extreme_logic_products_preserve_analysis_top50(self):
        import analysis_v2 as V
        import logic_baselines as L
        # Float32 exp(-120...-220) collapses to zero, changing candidate ties.
        # The analysis helper promotes before log-sigmoid/product computation.
        values = np.linspace(-.5, 0., 200, dtype=np.float32)
        conditions = np.stack([values, values[::-1] * .2], axis=-1)
        full = np.linspace(-.2, .3, 200, dtype=np.float32)
        order = np.random.default_rng(8).permutation(200)
        features = {"core_embed": {"full_raw": full, "logic_condition_raw": conditions}}
        options = {"mu": .6, "beta": 100., "gamma": -1.}
        expected = L.aggregate_scores(full[order], conditions[order], **options)
        for kind in ("product_cond", "logsigmoid_cond"):
            policy = f"m1/top200/logic_core_embed_train_presence_calibrated_{kind}"
            prefix = V.A.admit(order[None], expected[kind][None])[0]
            np.testing.assert_array_equal(B.admit(policy, features, order, {"core_embed": options}), prefix)
        self.assertTrue(np.all(expected["product_cond"] > 0))
        naive = np.exp(-np.logaddexp(0, -100 * (conditions - np.float32(.6))).sum(axis=-1))
        self.assertTrue(np.all(naive == 0))

    def test_timing_brackets_compute_with_synchronization(self):
        clock = FakeClock()
        syncs = []
        def sync():
            syncs.append(clock.value)
            clock.advance(.5)
        value, seconds = B.timed(lambda: (clock.advance(2), "output")[1], synchronize=sync, clock=clock)
        self.assertEqual(value, "output")
        self.assertEqual(len(syncs), 2)
        self.assertEqual(seconds, 2.5)

    def test_cross_budget_decode_and_recovery_are_real_task_boundaries(self):
        temp_root = Path(__file__).resolve().parents[2] / "hf_home/tmp"
        temp_root.mkdir(parents=True, exist_ok=True)
        clock, calls = FakeClock(), []
        def rgb(path):
            clock.advance(.1)
            return path
        def cross_scores(scorer, queries, images, batch_size):
            calls.append((len(queries), batch_size))
            for start in range(0, len(queries), batch_size):
                for opener in images[start:start + batch_size]:
                    opener()
                clock.advance(.01 * len(images[start:start + batch_size]))
            return np.zeros(len(queries))
        original = SimpleNamespace(rgb=rgb, image_path=lambda key: str(key), cross_scores=cross_scores)
        panel = {"ids": list(range(1, 221)), "queries": [{"text": "q0"}, {"text": "q1"}]}
        orders = np.tile(np.arange(220), (2, 1))
        with TemporaryDirectory(dir=temp_root) as temporary:
            store = B.Store(temporary, {"source": "synthetic"})
            tasks = [["b25", 0], ["policy50", 1]]
            kwargs = dict(selected_policy="m1/top100/core", selected_prefixes={0: orders[0, :50], 1: orders[1, :50]}, clock=clock)
            B.run_cross_tasks(store, object(), tasks, [0, 1], panel, orders, original, 4, **kwargs)
            self.assertEqual(calls, [(50, 4), (100, 4)])
            self.assertAlmostEqual(store.data["records"]["cross/b25/rep0"]["seconds"], 50 * .11)
            self.assertEqual(store.data["records"]["cross/policy50/rep1"]["pairs"], 100)
            self.assertTrue(all("scores" not in row for row in store.data["records"].values()))
            resumed = B.Store(temporary, {"source": "synthetic"})
            B.run_cross_tasks(resumed, object(), tasks, [0, 1], panel, orders, original, 4, **kwargs)
            self.assertEqual(len(calls), 2)
            with self.assertRaisesRegex(ValueError, "provenance"):
                B.Store(temporary, {"source": "changed"})
            status = json.loads((Path(temporary) / "progress.json").read_text())
            self.assertEqual(status["completed_tasks"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
