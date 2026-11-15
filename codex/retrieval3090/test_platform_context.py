"""[codex] Synthetic CPU checks; no study artifact or GPU is read."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import platform_context as P


class PlatformContextTests(unittest.TestCase):
    def setUp(self):
        temporary_root = Path(__file__).resolve().parents[2] / "hf_home/tmp"
        temporary_root.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="platform-context-unit-", dir=temporary_root)
        self.root = Path(self.temporary.name)
        self.repository = self.root / "repository"
        self.plan_path = self.repository / "codex/retrieval3090/platform_plan.json"
        self.plan_path.parent.mkdir(parents=True)
        self.cache = self.root / "cache"
        self.cache.mkdir()
        self.receipt_path = self.cache / "platform-receipt.json"
        self.commit = "a" * 40
        self.plan = copy.deepcopy(P.EXPECTED)
        self.plan.update(owner="[codex]", historical_baseline_was_read=True, allowed_operations=list(P.OPERATIONS))
        self.baseline = {"map_pp": 53.0, "oracle_map_pp": 74.0, "recall_pp": 74.0, "queries_without_hit": 30}
        self.meta = {"replay": {"state": "MISMATCH", "core": self.baseline, "error": "original gate failed"},
                     "bundle": {"source": self.plan["source"], "freeze": self.plan["freeze"],
                                "commit": self.commit, "dirty": "0"},
                     "depth": 200, "top_k": 50, "panel_sha256": self.plan["panel_sha256"],
                     "revisions": self.plan["model_revisions"], "model_config_sha256": self.plan["model_config_sha256"],
                     "config": {"batch_size": self.plan["batch_size"]},
                     "execution_environment": self.plan["environment"], "source_checkpoint_provenance_sha256": "b" * 64}
        self.panel = {"ids": list(range(8191)), "queries": [{"group": q % 118} for q in range(136)]}
        self.write(self.plan_path, self.plan)
        self.write(self.cache / "meta.json", self.meta)
        self.write(self.cache / "panel.json", self.panel)
        # Shapes/order/numerics are the committed producer's responsibility;
        # these byte fixtures test the consumer's immutable artifact binding.
        for name in ("vectors.npz", "cross.npz"):
            (self.cache / name).write_bytes(b"synthetic producer artifact " + name.encode())
        self.receipt = {"schema": "retrieval-platform-receipt-v1", "plan_sha256": P.file_sha(self.plan_path),
                        "producer_commit": self.commit, "published_reproduction": {"state": "MISMATCH", "metrics": self.baseline},
                        "platform_consistency": "VALID_LOCAL", "formal_evidence": False,
                        "environment": self.plan["environment"], "checks": {key: True for key in P.CHECKS},
                        "source_checkpoint_provenance_sha256": "b" * 64,
                        "source_checkpoint_files": {name: {"sha256": "c" * 64, "state_sha256": "d" * 64}
                                                    for name in P.SOURCE_FILES}}
        self.refresh()
        self.original_committed_plan = P._committed_plan
        self.committed = patch.object(P, "_committed_plan", return_value={
            "repository_path": "codex/retrieval3090/platform_plan.json", "producer_commit": self.commit}).start()
        self.addCleanup(patch.stopall)
        self.addCleanup(self.temporary.cleanup)

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")

    def refresh(self):
        self.receipt["plan_sha256"] = P.file_sha(self.plan_path)
        self.receipt["outputs_sha256"] = {name: P.file_sha(self.cache / name) for name in P.ARTIFACTS}
        self.write(self.receipt_path, self.receipt)

    def validate(self, **kwargs):
        return P.validate_context(self.cache, self.plan_path, self.receipt_path, **kwargs)

    def test_valid_receipt_preserves_historical_failure_and_bytes(self):
        before = (self.cache / "meta.json").read_bytes()
        context = self.validate()
        self.assertEqual(context["mode"], P.MODE)
        self.assertEqual(context["published_reproduction"]["state"], "MISMATCH")
        self.assertIs(context["formal_evidence"], False)
        self.assertEqual(P.require_baseline(self.baseline, self.meta, context), self.baseline)
        self.assertEqual(json.loads(json.dumps(context))["receipt_sha256"], P.file_sha(self.receipt_path))
        self.assertEqual((self.cache / "meta.json").read_bytes(), before)

    def test_every_final_artifact_is_bound_by_sha(self):
        for name in P.ARTIFACTS:
            with self.subTest(name=name):
                path = self.cache / name
                old = path.read_bytes()
                path.write_bytes(old + b"changed")
                with self.assertRaisesRegex(ValueError, "SHA differs"):
                    self.validate()
                path.write_bytes(old)

    def test_generic_or_false_receipt_cannot_waive_checks(self):
        original = copy.deepcopy(self.receipt)
        for change in ({"checks": {}}, {"checks": {key: 1 for key in P.CHECKS}},
                       {"formal_evidence": True}, {"platform_consistency": "PASS"},
                       {"outputs_sha256": {"meta.json": "c" * 64}},
                       {"source_checkpoint_files": {}}, {"source_checkpoint_provenance_sha256": "unknown"}):
            with self.subTest(change=change):
                self.write(self.receipt_path, original | change)
                with self.assertRaises(ValueError):
                    self.validate()

    def test_fixed_plan_rejects_other_platforms_scope_models_and_thresholds(self):
        for key, value in (("environment", self.plan["environment"] | {"torch": "2.6.0+cu124"}),
                           ("scope", "sealed-independent"), ("formal_evidence", 0),
                           ("model_revisions", {}), ("allowed_operations", ["M1"]),
                           ("published_acceptance", self.plan["published_acceptance"] | {"tolerance_pp": 2.0})):
            with self.subTest(key=key):
                self.write(self.plan_path, self.plan | {key: value})
                self.refresh()
                with self.assertRaisesRegex(ValueError, "plan"):
                    self.validate()

    def test_uncommitted_or_changed_plan_is_rejected(self):
        self.committed.side_effect = ValueError("Platform plan differs from its producer commit")
        with self.assertRaisesRegex(ValueError, "producer commit"):
            self.validate()
        self.committed.side_effect = None
        self.receipt["plan_sha256"] = "0" * 64
        self.write(self.receipt_path, self.receipt)
        with self.assertRaisesRegex(ValueError, "schema/plan"):
            self.validate()

    def test_git_plan_check_uses_exact_producer_blob(self):
        responses = [subprocess.CompletedProcess([], 0, stdout=str(self.repository) + "\n"),
                     subprocess.CompletedProcess([], 0, stdout=self.plan_path.read_bytes())]
        with patch.object(P.subprocess, "run", side_effect=responses) as run:
            self.original_committed_plan(self.plan_path, self.commit)
            self.assertEqual(run.call_args_list[1].args[0][-1],
                             self.commit + ":codex/retrieval3090/platform_plan.json")
        responses[1] = subprocess.CompletedProcess([], 0, stdout=b"different committed plan")
        with patch.object(P.subprocess, "run", side_effect=responses):
            with self.assertRaisesRegex(ValueError, "differs from"):
                self.original_committed_plan(self.plan_path, self.commit)
        with self.assertRaisesRegex(ValueError, "full producer commit"):
            self.original_committed_plan(self.plan_path, "HEAD")

    def test_pass_cannot_be_misrepresented_as_platform_mismatch(self):
        for where in ("receipt", "meta"):
            with self.subTest(where=where):
                receipt, meta = copy.deepcopy(self.receipt), copy.deepcopy(self.meta)
                if where == "receipt":
                    receipt["published_reproduction"]["state"] = "PASS"
                else:
                    meta["replay"]["state"] = "PASS"
                self.write(self.cache / "meta.json", meta)
                receipt["outputs_sha256"]["meta.json"] = P.file_sha(self.cache / "meta.json")
                self.write(self.receipt_path, receipt)
                with self.assertRaisesRegex(ValueError, "preserved published MISMATCH"):
                    self.validate()
        passing = {key: self.plan["published_acceptance"][key] for key in P.METRICS}
        self.meta["replay"]["core"] = passing
        self.receipt["published_reproduction"]["metrics"] = passing
        self.write(self.cache / "meta.json", self.meta)
        self.refresh()
        with self.assertRaisesRegex(ValueError, "actually passing"):
            self.validate()

    def test_cache_identity_and_baseline_cannot_change(self):
        for key, value in (("depth", 50), ("execution_environment", {}), ("revisions", {}),
                           ("source_checkpoint_provenance_sha256", "e" * 64),
                           ("replay", {"state": "MISMATCH", "core": self.baseline | {"map_pp": 52.0}})):
            with self.subTest(key=key):
                self.write(self.cache / "meta.json", self.meta | {key: value})
                self.refresh()
                with self.assertRaises(ValueError):
                    self.validate()

    def test_context_cannot_be_forged_modified_or_reused_with_other_meta(self):
        context = self.validate()
        with self.assertRaisesRegex(ValueError, "unvalidated"):
            P.require_baseline(self.baseline, self.meta, dict(context))
        with self.assertRaisesRegex(ValueError, "different cache meta"):
            P.require_baseline(self.baseline, self.meta | {"other": True}, context)
        with self.assertRaisesRegex(ValueError, "differs from"):
            P.require_baseline(self.baseline | {"map_pp": 52.0}, self.meta, context)
        context["baseline"] = self.baseline | {"map_pp": 52.0}
        with self.assertRaisesRegex(ValueError, "modified"):
            P.require_baseline(context["baseline"], self.meta, context)

    def supplement(self):
        parent = self.root / "supplement"
        parent.mkdir()
        provenance = {"source": self.plan["source"], "freeze": self.plan["freeze"], "commit": "f" * 40,
                      "config": self.meta["config"], "calibration_design_sha256": "e" * 64, "d3_texts_sha256": "d" * 64,
                      **{key: self.plan["environment"][key] for key in ("torch", "cuda", "device")}}
        for name in ("lcse_vectors.npz", "d3_embeddings.npz", "lcse_calibration.json"):
            (parent / name).write_bytes(b"synthetic supplementary payload")
        meta = provenance | {"status": "complete", "outputs_sha256": {
            name: P.file_sha(parent / name) for name in ("lcse_vectors.npz", "d3_embeddings.npz", "lcse_calibration.json")}}
        self.write(parent / "provenance.json", provenance)
        self.write(parent / "meta.json", meta)
        return parent, provenance, meta

    def test_supplement_requires_complete_same_runtime_parent_receipts(self):
        parent, provenance, meta = self.supplement()
        context = self.validate(supplementary_paths=(parent / "lcse_vectors.npz", parent / "lcse_calibration.json"))
        self.assertEqual(len(context["supplementary_artifacts"]), 2)
        self.assertIn("parent_provenance_sha256", context["supplementary_artifacts"][0])
        # Even an unrequested sibling output is checked for complete-parent consistency.
        (parent / "d3_embeddings.npz").write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "output SHA"):
            self.validate(supplementary_paths=(parent / "lcse_vectors.npz",))
        (parent / "d3_embeddings.npz").write_bytes(b"synthetic supplementary payload")
        provenance["torch"] = "2.6.0+cu124"
        self.write(parent / "provenance.json", provenance)
        self.write(parent / "meta.json", meta | provenance)
        with self.assertRaisesRegex(ValueError, "runtime differs"):
            self.validate(supplementary_paths=(parent / "lcse_vectors.npz",))

    def test_missing_file_sealed_path_and_duplicate_keys_are_rejected(self):
        (self.cache / "cross.npz").unlink()
        with self.assertRaises(FileNotFoundError):
            self.validate()
        with self.assertRaisesRegex(ValueError, "Sealed-panel"):
            P.validate_context(self.root / "retrieval-independent-panel", self.plan_path, self.receipt_path)
        self.receipt_path.write_text('{"schema":1,"schema":2}')
        with self.assertRaisesRegex(ValueError, "Duplicate JSON"):
            self.validate()

    def test_analysis_default_gate_and_explicit_platform_scope(self):
        import analysis_v2 as V
        from test_analysis_v2 import synthetic
        cache, panel, meta, expected = synthetic()
        _, cross, relevance = V.validate_cache(cache, panel)
        base = V.measure(V.A.final_order(cache["core_order"], cache["core_order"][:, :50], cross[:, :50]), relevance, 50)
        expected["recall_pp"] = float(100 * np.mean(base["hits_at_50"] / relevance.sum(axis=1)))
        meta["replay"] = {"state": "MISMATCH", "core": expected}
        # Issue a unit-only context for the synthetic analysis. Full receipt/Git
        # validation is exercised independently above, without production data.
        context = P._Context({"mode": P.MODE, "formal_evidence": False,
                              "published_reproduction": {"state": "MISMATCH", "metrics": expected},
                              "baseline": expected}, token=P._TOKEN, meta=meta)
        with patch.object(V, "features", side_effect=AssertionError("must not enter policies")):
            with self.assertRaisesRegex(ValueError, "did not PASS"):
                V.analyse(cache, panel, meta)
            with self.assertRaisesRegex(ValueError, "unvalidated"):
                V.analyse(cache, panel, meta, evaluation_context=dict(context))
        report, per_query = V.analyse(cache, panel, meta, evaluation_context=context, draws=10)
        self.assertIs(report["formal_evidence"], False)
        self.assertEqual(per_query["evaluation_context"]["published_reproduction"]["state"], "MISMATCH")
        self.assertIn("local-relative", report["scope"])
        markdown = V.markdown(report)
        self.assertIn("reproduction remains MISMATCH", markdown)
        self.assertNotIn("replay passed", markdown)
        self.assertEqual(meta["replay"]["state"], "MISMATCH")
        with self.assertRaisesRegex(ValueError, "cannot replace"):
            V.analyse(cache, panel, meta, evaluation_context=context, published=expected)

    def test_analysis_cli_requires_paired_platform_paths_before_loading(self):
        import analysis_v2 as V
        with patch.object(sys, "argv", ["analysis_v2", "--cache", "unused", "--out", "unused-out",
                                        "--platform-plan", str(self.plan_path)]):
            with patch.object(V.A, "load", side_effect=AssertionError("must not load cache")):
                with self.assertRaises(SystemExit) as error:
                    V.main()
                self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
