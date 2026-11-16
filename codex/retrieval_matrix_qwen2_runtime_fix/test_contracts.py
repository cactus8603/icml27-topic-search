"""[codex] CPU rejection tests for dependency-only scientific invariance."""
from copy import deepcopy
from unittest.mock import patch
import unittest
from codex.retrieval_matrix_qwen2_runtime_fix import contracts as C

class Contracts(unittest.TestCase):
    def test_original_scientific_fields_cannot_change(self):
        base=C.parent();plan=deepcopy(base)
        plan.update(runtime_batch="P0-Qwen2-einops-overlay-v1",runtime_fix={
            "mode":"embedding",
            "parent_plan":C.entry(C.ROOT/C.BASE_REL),"parent_freeze_commit":C.BASE_COMMIT,
            "overlay":C.entry(C.HERE/"overlay.json")})
        for field in ("models","datasets","settings","dependencies","phase","gate","fixed_baseline_test"):
            changed=deepcopy(plan);changed[field]={"changed":True}
            with self.assertRaisesRegex(ValueError,"preserve original"):C.validate(changed,imports=False)

    def test_overlay_rejects_wrong_package_or_empty_payload(self):
        for value in ({"package":"other"},{"package":"einops","version":"0.8.2","overlay":str(C.OVERLAY),"files":{}}):
            with self.assertRaises(ValueError):C.verify_overlay(value)

    def test_payload_tamper_and_shadowed_module_rejected(self):
        from codex.retrieval_matrix_qwen2_runtime_fix.build_plan import build
        plan=build("embedding")
        manifest=C.read(C.HERE/"overlay.json")
        bad=deepcopy(manifest);bad["files"][next(iter(bad["files"]))]="0"*64
        with self.assertRaises(ValueError):C.verify_overlay(bad)
        modules=deepcopy(manifest["modules"]);modules["einops"]["path"]="/synthetic/shadowed/einops/__init__.py"
        with patch.object(C,"actual_modules",return_value=modules),self.assertRaisesRegex(ValueError,"module path/version"):
            C.validate(plan)

    def test_both_modes_preserve_parent_fields(self):
        from codex.retrieval_matrix_qwen2_runtime_fix.build_plan import build
        for mode in C.PARENTS:
            plan=build(mode);base=C.parent(mode=mode)
            for field in ("models","datasets","settings","dependencies","phase"):
                self.assertEqual(plan[field],base[field])
            self.assertEqual(plan["runtime_fix"]["mode"],mode)

    def test_original_output_directories_are_rejected(self):
        self.assertEqual(C.output_path(C.OUTPUT_ROOT/"new/gme2"),C.OUTPUT_ROOT/"new/gme2")
        for path in (C.OUTPUT_ROOT,C.ROOT/"dl/retrieval-matrix/runs/old",C.ROOT/"dl/retrieval-matrix-mcmr/runs/old"):
            with self.assertRaises(ValueError):C.output_path(path)

    def test_modes_install_without_replacing_each_other_model_dispatch(self):
        import subprocess,sys
        for mode in C.PARENTS:
            source="from codex.retrieval_matrix_qwen2_runtime_fix import runner as P; "
            source+=f"entry=P.install('{mode}'); "
            source+="from codex.retrieval_matrix import runner as R; "
            if mode=="embedding":source+="assert entry is R; assert R.models.OfficialModel.__name__=='Qwen2OfficialModel'"
            else:source+="assert entry.__name__.endswith('retrieval_matrix_mcmr.runner'); assert R.models.OfficialModel.__name__=='OfficialModel'"
            subprocess.run([sys.executable,"-B","-c",source],cwd=C.ROOT,check=True,capture_output=True,text=True)

    def test_missing_central_assignment_rejects_before_model_runtime(self):
        from codex.retrieval_matrix_qwen2_runtime_fix import runner as P
        with patch.dict(P.os.environ,{},clear=True),self.assertRaisesRegex(ValueError,"central dispatcher"):
            P.runtime_environment({})

if __name__=="__main__":unittest.main()
