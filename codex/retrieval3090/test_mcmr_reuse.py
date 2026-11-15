"""[codex] CPU-only synthetic tests for exact, provenance-bound cross reuse."""
from copy import deepcopy
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mcmr_pilot as M
import mcmr_reuse as R


class ReuseTests(unittest.TestCase):
    def setUp(self):
        tmp = M.ROOT / "hf_home/tmp"
        tmp.mkdir(parents=True, exist_ok=True)
        self.temp = TemporaryDirectory(dir=tmp)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base, self.index, self.target = [self.root / name for name in ("baseline", "index", "target")]
        for directory in (self.base, self.index, self.target):
            directory.mkdir()
            (directory / "writer.lock").touch()
        self.n, self.q, self.d = 210, 2, 2048
        self.counts_patch = patch.object(R, "PILOT_COUNTS", (self.q, self.n))
        self.counts_patch.start()
        self.addCleanup(self.counts_patch.stop)
        rng = np.random.default_rng(157)
        def unit(count):
            a = rng.normal(size=(count, self.d)).astype(np.float32)
            return a/np.linalg.norm(a, axis=1, keepdims=True)
        self.panel = {"candidates": [{"id": str(i), "sha256": f"{i:064x}", "text": f"product {i}", "image": f"{i}.jpg"}
                                      for i in range(self.n)], "queries": []}
        for qi in range(self.q):
            text = f"Requirement {qi}. Another requirement {qi}."
            self.panel["queries"].append({"id": str(qi), "text": text, "clauses": M.clauses(text), "positive_ids": [str(qi)]})
        self.vectors = {"core_embed_image": unit(self.n), "core_embed_full": unit(self.q)}
        self.base_policy = M.parse_policy("m1/top200/core")
        self.policy = M.parse_policy("m1/top200/min_cond_core")
        plain, _ = M.text_table(self.panel, self.policy)
        self.target_vectors = dict(self.vectors, core_embed_condition=unit(len(plain)))
        self.env = {"torch": "synthetic-only", "device": "RTX 3090", "capability": [8, 6],
                    "actual_cpu_threads": {"intraop": 1, "interop": 1}}
        self.config = M.frozen_config()
        self.sources = M.script_sources()
        index_identity = M.index_identity(SimpleNamespace(joint_batch=1), self.panel, self.config,
            Path(self.config["models"]["core_embed"]["revision"]), self.env,
            {"class": "SyntheticProcessor", "image_config": {"min_pixels": 4}}, self.sources)
        np.savez(self.index / "index.npz", **self.vectors)
        M.atomic_json(self.index / "index-meta.json", {"state": "COMPLETE", "identity": index_identity,
                                                     "index_sha256": M.sha(self.index / "index.npz")})
        self.index_receipt = {"path": str(self.index), "meta_sha256": M.sha(self.index / "index-meta.json"),
                              "index_sha256": M.sha(self.index / "index.npz"), "read_only_reuse": True}
        for directory, policy in ((self.base, self.base_policy), (self.target, self.policy)):
            M.atomic_json(directory / "panel.json", self.panel)
            M.atomic_json(directory / "policy.json", policy)
        self.identity = {"schema": 1, "scope": M.SCOPE, "source": M.SOURCE, "official_CORE_code": M.CORE_CODE,
            "commit": "1"*40, "sources": self.sources, "panel_sha256": M.sha(self.base / "panel.json"),
            "policy_sha256": M.sha(self.base / "policy.json"), "model_config_sha256": M.CONFIG_SHA,
            "model_revisions": {k: self.config["models"][k]["revision"] for k in ("core_embed", "core_reranker")},
            "adapter": {"query_system": "same", "candidate_system": "same", "joint_tokens": 4096,
                        "cross_tokens": 8192, "image_pixels": {"same": 1},
                        "batch_sizes": {"joint_image": 1, "cross": 1, "text": 32, "cs_image": 8},
                        "CS_candidate_input": None}, "environment": self.env, "shared_index": self.index_receipt}
        core, chosen = M.admission(self.panel, self.vectors, self.base_policy)
        pairs = M.cross_union(core, chosen)
        np.savez(self.base / "cache.npz", **self.vectors, pairs=np.asarray(pairs, dtype=np.int64), cross=self.scores(pairs))
        M.atomic_json(self.base / "meta.json", {"state": "COMPLETE", "identity": self.identity,
            "cache_sha256": M.sha(self.base / "cache.npz"), "cross_admission_dependency": False})
        self.baseline = R.baseline_bundle(self.base, self.index)
        self.target_identity = deepcopy(self.identity)
        self.target_identity["policy_sha256"] = M.sha(self.target / "policy.json")
        self.target_identity["commit"] = "2"*40  # source bytes, not unrelated HEAD commits, determine reuse.
        self.lock = self.root / "selected-policy.json"
        M.atomic_json(self.lock, {"schema": 1, "state": "LOCKED", "policy_id": self.policy["policy_id"],
            "policy_sha256": M.sha(self.target / "policy.json"), "analysis_sha256": "a"*64,
            "platform_plan_sha256": "b"*64, "selection_rule": "frozen map/recall/id ordering",
            "formal_evidence": False, "selected_before_external_effects": True})
        self.selection = R.selection_binding(self.lock, self.policy, self.target / "policy.json")
        self.source = {"path": "codex/retrieval3090/mcmr_reuse.py", "sha256": "c"*64, "commit": "2"*40}
        self.binding = R.binding_for(self.baseline, self.selection, self.source)
        self.commit_patch = patch.object(R, "validate_wrapper_commit", return_value=None)
        self.commit_patch.start()
        self.addCleanup(self.commit_patch.stop)

    @staticmethod
    def scores(rows):
        return np.asarray([qi*1000+ci/32 for qi, ci in rows], dtype=np.float32)

    def rewrite_base(self, *, vectors=None, meta=None):
        if vectors is not None:
            np.savez(self.base / "cache.npz", **vectors)
        body = M.read_json(self.base / "meta.json") if meta is None else meta
        body["cache_sha256"] = M.sha(self.base / "cache.npz")
        M.atomic_json(self.base / "meta.json", body)

    def produce(self, *, fail_repeat=False):
        cp = self.target / "checkpoints"
        cp.mkdir(exist_ok=True)
        M.fixed_json(cp / "provenance.json", self.target_identity)
        core, chosen = M.admission(self.panel, self.target_vectors, self.policy)
        pairs = M.cross_union(core, chosen)
        M.fixed_json(cp / "cross-pairs.json", pairs)
        calls = []
        def compute(rows):
            calls.extend(deepcopy(rows))
            value = self.scores(rows)
            return value + np.float32(1) if fail_repeat else value
        args = SimpleNamespace(out=self.target)
        with R.intercept(args, self.baseline, self.binding) as state:
            helper = M.checkpoint_helpers()
            # This unrelated stage must pass straight through the original helper.
            other = helper.batches(cp / "other.npy", [0, 1], 2,
                lambda rows: np.asarray(rows, dtype=np.float32)[:, None], {"stage": "unrelated"})
            np.testing.assert_array_equal(other, [[0], [1]])
            identity = self.target_identity | {"stage": "cross_union", "items_sha256": M.digest(pairs)}
            cross = helper.batches(cp / "cross_union.npy", pairs, 1, compute, identity)
            np.savez(self.target / "cache.npz", **self.target_vectors, pairs=np.asarray(pairs, dtype=np.int64), cross=cross)
            M.atomic_json(self.target / "meta.json", {"state": "COMPLETE", "identity": self.target_identity,
                "cache_sha256": M.sha(self.target / "cache.npz"), "cross_admission_dependency": False})
            self.assertEqual(M.read_json(self.target / "meta.json")["cross_execution"]["state"], "PENDING")
            self.assertEqual(M.read_json(self.target / "meta.json")["state"], "REUSE_PENDING")
        receipt = R.finalize(self.target, self.policy, self.baseline, self.binding, state)
        return receipt, calls, pairs

    def test_exact_union_reuse_preserves_order_dtype_and_runs_only_novel_plus_eight(self):
        receipt, calls, pairs = self.produce()
        self.assertEqual(receipt["counts"]["reused_final_pairs"], 100)
        novel = [pair for pair in pairs if tuple(pair) not in self.baseline.lookup]
        self.assertGreater(len(novel), 0)
        self.assertEqual(calls[8:], novel)
        self.assertEqual(len(calls), 8+len(novel))
        with np.load(self.target / "cache.npz") as arrays:
            self.assertEqual(arrays["cross"].dtype, np.float32)
            np.testing.assert_array_equal(arrays["cross"], self.scores(pairs))
        R.validate_receipt(self.target, self.policy, self.baseline, self.binding)

    def test_completed_resume_still_freshly_checks_eight_and_does_not_rescore_novel(self):
        self.produce()
        receipt, calls, _ = self.produce()
        self.assertEqual(len(calls), 8)
        self.assertEqual(receipt["counts"]["current_invocation_computed_fresh_final"], 0)
        self.assertEqual(receipt["counts"]["current_invocation_returned_reused"], 0)
        R.validate_receipt(self.target, self.policy, self.baseline, self.binding)

    def test_original_native_analysis_cannot_bypass_wrapper_receipt(self):
        self.produce()
        self.assertEqual(M.read_json(self.target / "meta.json")["state"], "REUSE_COMPLETE")
        argv = [str(M.HERE / "mcmr_pilot.py"), "--data", str(self.root / "data"),
                "--images", str(self.root / "images"), "--out", str(self.target),
                "--policy", self.policy["policy_id"], "--stage", "analyze"]
        with patch.object(sys, "argv", argv), patch.object(M, "analyze_arrays", side_effect=AssertionError("must not analyze")):
            with self.assertRaisesRegex(ValueError, "incomplete"):
                M.main()

    def test_repeat_mismatch_fails_before_lookup_or_cross_checkpoint_publication(self):
        with self.assertRaisesRegex(ValueError, "repeat differs"):
            self.produce(fail_repeat=True)
        self.assertFalse((self.target / "checkpoints/cross_union.npy").exists())
        self.assertFalse((self.target / "reuse-receipt.json").exists())

    def test_repeat_rejects_changed_dtype_shape_and_nonfinite(self):
        for compute in (lambda rows: self.scores(rows).astype(np.float64),
                        lambda rows: self.scores(rows)[:, None],
                        lambda rows: np.asarray([np.nan], dtype=np.float32)):
            with self.assertRaisesRegex(ValueError, "shape/dtype"):
                R.ReuseCompute(self.baseline, self.baseline.pairs, compute)

    def test_baseline_cache_sha_and_source_bytes_are_mandatory(self):
        original = M.read_json(self.base / "meta.json")
        for field in ("cache_sha256", "source", "sources"):
            bad = deepcopy(original)
            if field == "cache_sha256":
                bad[field] = "0"*64
            else:
                bad["identity"][field] = "changed"
            M.atomic_json(self.base / "meta.json", bad)
            with self.assertRaisesRegex(ValueError, "integrity"):
                R.baseline_bundle(self.base, self.index)
        M.atomic_json(self.base / "meta.json", original)

    def test_complete_baseline_with_wrong_actual_prefix50_or_order_is_rejected(self):
        bad = dict(self.baseline.vectors)
        bad["pairs"] = np.asarray(self.baseline.pairs, dtype=np.int64)[::-1]
        bad["cross"] = self.baseline.cross[::-1]
        self.rewrite_base(vectors=bad)
        with self.assertRaisesRegex(ValueError, "actual CORE prefix50"):
            R.baseline_bundle(self.base, self.index)

    def test_baseline_cannot_be_reuse_chain_or_non_batch1(self):
        original = M.read_json(self.base / "meta.json")
        for mode in ("chain", "batch"):
            bad = deepcopy(original)
            if mode == "chain":
                bad["cross_execution"] = {"mode": R.SCHEMA}
            else:
                bad["identity"]["adapter"]["batch_sizes"]["cross"] = 4
            self.rewrite_base(meta=bad)
            with self.assertRaises(ValueError):
                R.baseline_bundle(self.base, self.index)

    def test_target_identity_env_adapter_and_model_changes_fail_but_extra_cs_is_allowed(self):
        R.validate_target_identity(self.target_identity, self.baseline)
        cs = deepcopy(self.target_identity)
        cs["model_revisions"]["cs_clip"] = self.config["models"]["cs_clip"]["revision"]
        cs["adapter"]["CS_candidate_input"] = "image only; additional cheap index without metadata"
        R.validate_target_identity(cs, self.baseline)
        for field in ("environment", "shared_index", "sources", "adapter", "model_revisions"):
            bad = deepcopy(self.target_identity)
            if field == "model_revisions":
                bad[field]["core_reranker"] = "changed"
            elif field == "adapter":
                bad[field]["batch_sizes"]["cross"] = 2
            else:
                bad[field] = {"changed": True}
            with self.assertRaises(ValueError):
                R.validate_target_identity(bad, self.baseline)

    def test_index_processor_meta_and_final_candidate_vectors_must_match_exactly(self):
        bad = dict(self.vectors)
        bad["core_embed_image"] = bad["core_embed_image"].copy()
        bad["core_embed_image"][0, 0] += np.float32(1e-5)
        with self.assertRaisesRegex(ValueError, "exactly equal"):
            R.assert_index_vectors(bad, self.vectors)
        meta = M.read_json(self.index / "index-meta.json")
        meta["identity"]["processor"]["image_config"]["min_pixels"] = 5
        M.atomic_json(self.index / "index-meta.json", meta)
        with self.assertRaisesRegex(ValueError, "exact shared index"):
            R.baseline_bundle(self.base, self.index)

    def test_selection_file_requires_file_bytes_sha_and_pre_external_lock(self):
        original = M.read_json(self.lock)
        for field, badvalue in (("policy_sha256", M.digest(self.policy)), ("state", "PENDING"),
                                ("selected_before_external_effects", False), ("formal_evidence", True)):
            bad = dict(original, **{field: badvalue})
            M.atomic_json(self.lock, bad)
            with self.assertRaisesRegex(ValueError, "LOCKED"):
                R.selection_binding(self.lock, self.policy, self.target / "policy.json")

    def test_source_or_selection_changes_after_capture_cannot_be_published(self):
        R.validate_live_bindings(self.binding)
        self.lock.write_text(self.lock.read_text()+"\n")
        with self.assertRaisesRegex(ValueError, "selection changed"):
            R.validate_live_bindings(self.binding)
        self.lock.write_text(self.lock.read_text().rstrip()+"\n")
        self.assertEqual(M.sha(self.lock), self.selection["sha256"])
        meta = M.read_json(self.base / "meta.json")
        meta["unexpected"] = "changed while inference was running"
        M.atomic_json(self.base / "meta.json", meta)
        with self.assertRaisesRegex(ValueError, "Baseline source files changed"):
            R.validate_live_bindings(self.binding)

    def test_frozen_baseline_pairs_and_cross_cannot_affect_admission(self):
        # Selection consults neither baseline cross values nor query positive labels.
        poisoned = deepcopy(self.panel)
        for q in poisoned["queries"]:
            q["positive_ids"] = object()
        core, selected = M.admission(poisoned, self.target_vectors | {"cross": object()}, self.policy)
        pairs = M.cross_union(core, selected)
        R.pair_order_guard(pairs, self.baseline)
        with self.assertRaisesRegex(ValueError, "pair order"):
            R.pair_order_guard(pairs[::-1], self.baseline)
        with self.assertRaisesRegex(ValueError, "exactly once"):
            R.pair_order_guard(pairs[1:], self.baseline)

    def test_receipt_tampering_and_missing_marker_cannot_pass_analysis(self):
        self.produce()
        receipt = M.read_json(self.target / "reuse-receipt.json")
        receipt["counts"]["fresh_final_pairs"] += 1
        M.atomic_json(self.target / "reuse-receipt.json", receipt)
        meta = M.read_json(self.target / "meta.json")
        meta["cross_execution"]["receipt_sha256"] = M.sha(self.target / "reuse-receipt.json")
        M.atomic_json(self.target / "meta.json", meta)
        with self.assertRaisesRegex(ValueError, "counts"):
            R.validate_receipt(self.target, self.policy, self.baseline, self.binding)
        meta.pop("cross_execution")
        M.atomic_json(self.target / "meta.json", meta)
        with self.assertRaisesRegex(ValueError, "integrity"):
            R.validate_receipt(self.target, self.policy, self.baseline, self.binding)

    def test_resuming_from_a_different_baseline_binding_cannot_mix_cross_checkpoint(self):
        self.produce()
        self.binding = deepcopy(self.binding)
        self.binding["baseline"]["cache_sha256"] = "d"*64
        with self.assertRaisesRegex(ValueError, "fingerprint|Fingerprint|identity|Identity|provenance"):
            self.produce()

    def test_shared_baseline_lock_rejects_concurrent_writer(self):
        with M.writer(self.base):
            with self.assertRaises(BlockingIOError):
                with R.shared(self.base):
                    pass

    def test_uncommitted_wrapper_cannot_produce_or_analyze(self):
        with patch.object(M, "git_bytes", return_value=b"different source bytes"):
            with self.assertRaisesRegex(ValueError, "committed"):
                R.wrapper_source()

    def test_receipt_requires_wrapper_bytes_at_the_recorded_execution_commit(self):
        self.commit_patch.stop()
        body = b"synthetic committed wrapper"
        import hashlib
        source = {"path": "codex/retrieval3090/mcmr_reuse.py", "sha256": hashlib.sha256(body).hexdigest()}
        with patch.object(M, "git_bytes", return_value=body):
            R.validate_wrapper_commit("a"*40, source)
            with self.assertRaisesRegex(ValueError, "execution commit"):
                R.validate_wrapper_commit("not a commit", source)
        with patch.object(M, "git_bytes", return_value=b"changed"):
            with self.assertRaisesRegex(ValueError, "source bytes"):
                R.validate_wrapper_commit("a"*40, source)


if __name__ == "__main__":
    unittest.main()
