"""[codex] Synthetic CPU-only paired runtime repair checks."""
from copy import deepcopy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from codex.retrieval_matrix_paired_runtime_fix import contracts as C,models,qualification as Q


def plan(key='clic_b32'):return C.read(C.HERE/('p0-'+key+'-einops-v1.json'))


class PairedRepair(unittest.TestCase):
    def test_three_exact_parents_and_scientific_fields_remain_unchanged(self):
        for key in C.MODELS:
            value=plan(key);base=C.parent(key);Q.validate_plan(value)
            for field in C.PRESERVED:self.assertEqual(value[field],base[field])
            self.assertEqual(value['settings']['image_batch'],1);self.assertEqual(value['settings']['text_batch'],1)
            for field,new in [('text_batch',2),('precision','bfloat16'),('seed',2)]:
                bad=deepcopy(value);bad['settings'][field]=new
                with self.assertRaisesRegex(ValueError,'preserve original settings'):Q.validate_plan(bad)

    def test_original_paired_formula_data_and_event_gate_are_retained(self):
        from codex.retrieval_matrix_paired_p0 import data,metrics
        self.assertIs(Q.ENGINE.ENGINE.data,data);self.assertIs(Q.ENGINE.ENGINE.metrics,metrics)
        self.assertIs(Q.ENGINE.ENGINE.verify_report,Q.ENGINE.verify_report)
        bad=plan();bad['benchmarks']['sugarcrepe_pp']['data']['expected_images']=1
        with self.assertRaisesRegex(ValueError,'preserve original benchmarks'):Q.validate_plan(bad)

    def test_manifest_only_adds_exact_einops_payload_and_import(self):
        new=C.read(C.checked(plan()['dependencies']));old=C.read(C.checked(C.parent('clic_b32')['dependencies']))
        self.assertEqual(len(new['files']),2798);self.assertEqual(len(old['files']),2766)
        self.assertEqual(set(new['modules'])-set(old['modules']),{'einops'})
        self.assertEqual({k:new['modules'][k] for k in old['modules']},old['modules'])
        self.assertEqual(C.actual_modules(new),new['modules'])

    def test_signature_changes_only_function_addresses_and_keeps_parameter_changes(self):
        left={'transform':'Resize(224)\n<function a._convert_to_rgb at 0x123abc>','other':'keep 0x123abc'}
        right={**left,'transform':'Resize(224)\n<function a._convert_to_rgb at 0x998eff>'}
        self.assertNotEqual(left,right);self.assertEqual(models.stable_signature(left),models.stable_signature(right))
        self.assertEqual(models.stable_signature(left)['other'],left['other'])
        changed={**right,'transform':right['transform'].replace('224','384')}
        self.assertNotEqual(models.stable_signature(left),models.stable_signature(changed))

    def test_stable_signature_uses_the_original_class_without_changing_encode(self):
        cls=models.ORIGINAL.PairedModel;model=cls.__new__(cls)
        model.signature=lambda:{'transform':'<function rgb at 0xabc>','dtype':'float32'}
        original_encode=cls.encode
        actual=models.prepare(model)
        self.assertIs(type(actual),cls);self.assertIs(cls.encode,original_encode)
        self.assertEqual(actual.signature(),{'transform':'<function rgb>','dtype':'float32'})

    def test_private_producer_does_not_mutate_any_frozen_module(self):
        from codex.retrieval_matrix_native_p0 import runner as native
        from codex.retrieval_matrix_paired_p0 import runner as paired
        before=(native.models,native.Q,native.runtime,paired.models,paired.Q,paired.runtime)
        from codex.retrieval_matrix_paired_runtime_fix import runner
        after=(native.models,native.Q,native.runtime,paired.models,paired.Q,paired.runtime)
        self.assertEqual(before,after);self.assertIs(runner.P.R.models.Siglip2,models.create)
        self.assertIs(runner.P.R.Q,Q)

    def test_central_guard_and_unreviewed_plan_reject_before_cuda(self):
        from codex.retrieval_matrix_paired_runtime_fix import runner
        with patch.dict(os.environ,{},clear=True),patch.object(C,'validate') as validate:
            with self.assertRaisesRegex(ValueError,'central dispatcher'):runner.runtime(plan(),gpu=True)
            validate.assert_not_called()
        with self.assertRaisesRegex(ValueError,'independently reviewed'):runner.source_gate(C.HERE/'p0-clic_b32-einops-v1.json',plan(),'a'*40)
        import torch
        self.assertFalse(torch.cuda.is_initialized())

    def test_old_scopes_and_old_outputs_never_admit(self):
        with tempfile.TemporaryDirectory(dir=C.ROOT/'hf_home/tmp',prefix='paired-fix-negative-') as folder:
            for scope in ('official-native-p0-qualified','official-native-runtime-fix-p0-qualified'):
                path=Path(folder)/(scope+'.json');C.write(path,{'scope':scope})
                with self.assertRaisesRegex(ValueError,'Old paired or Google'):Q.admit(C.entry(path),{})
        with self.assertRaises(ValueError):C.output_path(C.ROOT/'dl/retrieval-matrix-paired-p0/runs/old')


if __name__=='__main__':unittest.main()
