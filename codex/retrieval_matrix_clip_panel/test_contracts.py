"""[codex] CPU-only rejection checks for the native CLIP-to-panel boundary."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from codex.retrieval_matrix_clip_panel import contracts as C
from codex.retrieval_matrix_native_p0 import qualification as Q

ROOT = Path(__file__).resolve().parents[2]


class Contracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT/"hf_home/tmp")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.spec = copy.deepcopy(Q.read(ROOT/"codex/retrieval_matrix_extra/registry.json")["models"]["siglip2"])
        self.spec["family"] = "CLIP"
        self.native_settings = {"seed": 2026111665, "allow_tf32": False,
            "smoke_cosine_min": .9995, "smoke_max_abs": .01}
        parent = self.root/"parent.json"
        dependencies = Q.entry(ROOT/"codex/retrieval_matrix_extra/dependencies.json")
        parent.write_text(json.dumps({"settings": self.native_settings, "dependencies": dependencies}))
        self.receipt = {"state": "PASS", "model": "siglip2", "family": "CLIP", "p0_plan": Q.entry(parent)}
        self.plan = {"phase": C.PHASE, "retriever_family": "CLIP", "models": {"siglip2": self.spec},
            "qualification": {"siglip2": {"path": "not-a-real-receipt", "sha256": "0"*64}},
            "parent_p0_plan": Q.entry(parent), "parent_native_settings": copy.deepcopy(self.native_settings),
            "dependencies": dependencies,
            "settings": {"seed": 2026111665, "allow_tf32": False,
                "batch_invariance_cosine_min": .9995, "batch_invariance_max_abs": .01}}

    def test_missing_native_evidence_never_admits(self):
        with self.assertRaises((ValueError, FileNotFoundError)):
            C.admit_models(self.plan)

    def test_mllm_cannot_claim_clip_phase(self):
        self.plan["models"]["siglip2"]["family"] = "MLLM"
        with patch.object(Q, "admit") as admit, self.assertRaises(ValueError):
            C.admit_models(self.plan)
        admit.assert_not_called()

    def test_extra_family_requires_its_own_review(self):
        self.plan["models"]["siglip2"]["extra_family"] = "clic"
        with patch.object(Q, "admit") as admit, self.assertRaises(ValueError):
            C.admit_models(self.plan)
        admit.assert_not_called()

    def test_evidence_is_bound_to_model_key_and_family(self):
        for key, value in (("model", "other"), ("family", "MLLM")):
            with self.subTest(key=key), patch.object(Q, "admit", return_value={**self.receipt, key: value}), self.assertRaises(ValueError):
                C.admit_models(self.plan)

    def test_parent_and_forward_changes_are_rejected(self):
        alterations = [lambda p: p["parent_p0_plan"].update(sha256="f"*64),
            lambda p: p["dependencies"].update(sha256="f"*64),
            lambda p: p["parent_native_settings"].update(seed=1),
            lambda p: p["settings"].update(allow_tf32=True),
            lambda p: p["settings"].update(batch_invariance_max_abs=.02)]
        for change in alterations:
            plan = copy.deepcopy(self.plan)
            change(plan)
            with patch.object(Q, "admit", return_value=self.receipt), self.assertRaises(ValueError):
                C.admit_models(plan)

    def test_missing_or_extra_qualification_is_rejected(self):
        for key in ("siglip2", "other"):
            plan = copy.deepcopy(self.plan)
            plan["qualification"] = {} if key == "siglip2" else {**plan["qualification"], key: {}}
            with patch.object(Q, "admit") as admit, self.assertRaises(ValueError):
                C.admit_models(plan)
            admit.assert_not_called()

    def test_source_table_cannot_omit_native_or_dependencies(self):
        sources = {}
        for folder in ("retrieval_matrix_clip_panel", "retrieval_matrix_extra", "retrieval_matrix_native_p0"):
            path = self.root/"codex"/folder
            path.mkdir(parents=True)
            item = path/("dependencies.json" if folder.endswith("extra") else "qualification.py")
            item.write_text("{}\n")
            sources[str(item.relative_to(self.root))] = Q.sha(item)
        C.check_declared_sources({"source_files": sources}, self.root)
        for name in list(sources):
            incomplete = {key: value for key, value in sources.items() if key != name}
            with self.assertRaises(ValueError):
                C.check_declared_sources({"source_files": incomplete}, self.root)

    def test_import_does_not_replace_shared_model(self):
        from codex.retrieval_matrix import runner as R
        original = R.models
        from codex.retrieval_matrix_clip_panel import runner
        from codex.retrieval_matrix_native_p0.models import Siglip2
        self.assertIs(R.models, original)
        self.assertIs(runner.Siglip2, Siglip2)

    def test_model_runtime_refuses_bypassing_central_dispatch(self):
        from codex.retrieval_matrix_clip_panel import runner
        with patch.dict("os.environ", {}, clear=True), self.assertRaises(ValueError):
            runner.runtime_environment(self.spec)


if __name__ == "__main__":
    unittest.main()
