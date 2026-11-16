"""[codex] Meaningful rejection and compatibility checks; synthetic CPU only."""
from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from codex.retrieval_matrix_native_runtime_fix import contracts as C,runner as R,qualification as Q

PLAN=C.HERE/'p0-siglip2-einops-v1.json'


class RuntimeRepair(unittest.TestCase):
    def test_original_scientific_settings_and_full_denominators_cannot_change(self):
        plan=C.read(PLAN);base=C.parent();C.validate(plan)
        for name in C.PRESERVED:self.assertEqual(plan[name],base[name])
        for name,value in [('precision','bfloat16'),('image_batch',1),('allow_tf32',True)]:
            bad=deepcopy(plan);bad['settings'][name]=value
            with self.assertRaisesRegex(ValueError,'preserve original settings'):C.validate(bad)
        bad=deepcopy(plan);bad['benchmarks']['coco']['data']['expected_texts']=25000
        with self.assertRaisesRegex(ValueError,'preserve original benchmarks'):C.validate(bad)

    def test_dependency_manifest_keeps_all_originals_and_only_adds_einops(self):
        plan=C.read(PLAN);old=C.read(C.checked(C.parent()['dependencies']));new=C.read(C.checked(plan['dependencies']))
        self.assertEqual(len(new['files']),len(old['files'])+32)
        self.assertEqual(set(new['modules'])-set(old['modules']),{'einops'})
        self.assertEqual({k:new['modules'][k] for k in old['modules']},old['modules'])
        self.assertEqual({k:new['files'][k] for k in old['files']},old['files'])
        actual=C.actual_modules(new);self.assertEqual(actual,new['modules'])
        self.assertEqual(actual['einops']['actual_version'],'0.8.2')

    def test_overlay_tamper_and_different_parent_are_rejected(self):
        original=C.sha;manifest=C.overlay();target=next(iter(manifest['files']))
        with patch.object(C,'sha',side_effect=lambda p:'0'*64 if str(Path(p).resolve())==str((C.ROOT/target).resolve()) else original(p)):
            with self.assertRaisesRegex(ValueError,'overlay bytes'):C.overlay()
        bad=C.read(PLAN);bad['runtime_fix']['parent_freeze_commit']='a'*40
        with self.assertRaisesRegex(ValueError,'provenance'):C.validate(bad)

    def test_central_guard_precedes_cuda_and_draft_cannot_produce(self):
        plan=C.read(PLAN)
        with patch.dict(os.environ,{},clear=True),patch.object(C,'validate') as validate:
            with self.assertRaisesRegex(ValueError,'central dispatcher'):R.runtime(plan,gpu=True)
            validate.assert_not_called()
        with self.assertRaisesRegex(ValueError,'independent runtime review'):R.source_gate(PLAN,plan,'a'*40)
        import torch
        self.assertFalse(torch.cuda.is_initialized())

    def test_old_outputs_and_old_qualification_scope_are_rejected(self):
        for path in (C.ROOT/'dl/retrieval-matrix-native-p0/runs/old',C.OUTPUT_ROOT):
            with self.assertRaises(ValueError):C.output_path(path)
        self.assertEqual(C.output_path(C.OUTPUT_ROOT/'new'),C.OUTPUT_ROOT/'new')
        with tempfile.TemporaryDirectory(dir=C.ROOT/'hf_home/tmp',prefix='native-fix-reject-') as folder:
            receipt=Path(folder)/'synthetic-old-receipt.json';C.write(receipt,{'scope':'official-native-p0-qualified'})
            with self.assertRaisesRegex(ValueError,'Old native P0'):Q.admit(C.entry(receipt),{})

    def test_review_rejects_missing_new_runtime_before_reading_arrays(self):
        plan=C.read(PLAN);plan.update(state='FROZEN',reviewed=True)
        with tempfile.TemporaryDirectory(dir=C.ROOT/'hf_home/tmp',prefix='native-fix-review-') as folder:
            path=Path(folder)/'synthetic-plan.json';report=Path(folder)/'synthetic-report.json'
            C.write(path,plan);C.write(report,{'identity':{'runtime':{}}})
            with patch.object(Q.Q,'review') as original:
                with self.assertRaisesRegex(ValueError,'exact new dependency runtime'):Q.review(path,{'coco':report},'independent')
                original.assert_not_called()

    def test_install_keeps_exact_original_model_class_and_metrics(self):
        source="""from codex.retrieval_matrix_native_runtime_fix import runner as R
from codex.retrieval_matrix_native_p0 import models,metrics
original_model=R.N.models.Siglip2
original_metric=R.N.metrics.author_metrics
entry=R.install()
assert entry.models.Siglip2 is original_model is models.Siglip2
assert entry.metrics.author_metrics is original_metric is metrics.author_metrics
try:R.install()
except ValueError:pass
else:raise AssertionError('double installation accepted')
import torch
assert not torch.cuda.is_initialized()
"""
        subprocess.run([sys.executable,'-B','-c',source],check=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':''},capture_output=True,text=True)


if __name__=='__main__':unittest.main()
