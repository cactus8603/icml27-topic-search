"""[codex] CPU-only negative gates and real adapter routing with temporary stubs."""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from codex.retrieval_matrix_clip_panel_next import contracts as C,build_plan as B,runner as R
from codex.retrieval_matrix_native_p0.evidence import write

PARENTS={
    'siglip2':('siglip2-einops-v1','codex/retrieval_matrix_native_runtime_fix/p0-siglip2-einops-approved-v1.json'),
    **{key:('paired-clic-einops-v1','codex/retrieval_matrix_paired_runtime_fix/p0-'+key+'-einops-approved-v1.json')
       for key in ('clic_b32','clic_l14','clic_clips_l14')}}


class ClipNext(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=C.ROOT/'hf_home/tmp',prefix='clip-next-synthetic-')
        self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)

    def fixture(self,key='siglip2'):
        adapter,relative=PARENTS[key];base=deepcopy(C.read(C.ROOT/relative))
        parent=self.root/(key+'-synthetic-parent.json');write(parent,base)
        signature={'synthetic_stub':True,'model_class':'explicit-unit-stub','tokenizer':'unchanged-stub'}
        reports={}
        for name in base['benchmarks']:
            path=self.root/(key+'-'+name+'-synthetic-report.json')
            write(path,{'identity':{'model_signature':signature},'scope':'SYNTHETIC_UNIT_ONLY_NO_MODEL_RESULTS'})
            reports[name]=C.entry(path)
        stub=self.root/(key+'-qualification-stub.json')
        write(stub,{'scope':'SYNTHETIC_UNIT_ONLY_NOT_QUALIFICATION','no_model_results':True})
        qualified={'state':'PASS','scope':C.ADAPTERS[adapter]['scope'],'family':'CLIP','model':key,
            'model_spec_sha256':C.digest(base['model_spec']),'p0_plan':C.entry(parent),'reports':reports}
        plan={'schema':1,'state':'REVIEW_REQUIRED','reviewed':False,'phase':C.PHASE,'purpose':C.PURPOSE,
            'retriever_family':'CLIP','models':{key:base['model_spec']},'qualification':{key:C.entry(stub)},
            'qualification_adapter':C.adapter_reference(adapter),'parent_p0_plan':C.entry(parent),
            'parent_native_settings':base['settings'],'parent_model_signature':signature,
            'dependencies':base['dependencies'],'settings':C.panel_settings(base),'fixed_baseline_test':True}
        return plan,qualified,parent,stub

    def test_constant_allowlist_and_source_entry_reject_arbitrary_or_old_adapters(self):
        for name in ('paired-clic-v1','native-siglip-original','os','not-registered'):
            with self.assertRaises(ValueError):C.adapter_reference(name)
        reference=C.adapter_reference('siglip2-einops-v1');reference['source']['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'source entry'):C.adapter(reference)
        reference=C.adapter_reference('siglip2-einops-v1');reference['source_commit']='a'*40
        with self.assertRaisesRegex(ValueError,'source entry'):C.adapter(reference)
        reference=C.adapter_reference('siglip2-einops-v1')
        with patch.object(C.importlib,'import_module',return_value=SimpleNamespace(__file__=str(self.root/'shadow.py'))), \
                self.assertRaisesRegex(ValueError,'Actually imported'):
            C.adapter(reference)

    def test_four_models_call_their_actual_allowlisted_facade_with_complete_spec(self):
        for key in PARENTS:
            with self.subTest(key=key):
                plan,qualified,_,_=self.fixture(key);facade=C.adapter(plan['qualification_adapter'])
                with patch.object(facade,'admit',return_value=qualified) as admit:
                    actual=C.admit_models(plan)
                    admit.assert_called_once_with(plan['qualification'][key],plan['models'][key])
                self.assertEqual(actual,{key:qualified})

    def test_missing_real_qualification_rejects_before_any_panel_reader(self):
        plan,_,parent,stub=self.fixture()
        facade=C.adapter(plan['qualification_adapter'])
        with patch.object(facade,'admit',wraps=facade.admit) as admit, \
                patch.object(B.data_audit,'verify') as audit,patch.object(B,'legacy_builder') as data:
            with self.assertRaisesRegex(ValueError,'Old native P0'):
                B.build(parent,stub,'siglip2-einops-v1',self.root/'not-a-real-audit.json')
            admit.assert_called_once();audit.assert_not_called();data.assert_not_called()

    def test_full_parent_spec_batch_precision_tokenizer_runtime_and_signature_are_bound(self):
        plan,qualified,_,_=self.fixture('clic_clips_l14');facade=C.adapter(plan['qualification_adapter'])
        changes=[lambda p:p['models']['clic_clips_l14']['batches'].update(text=2),
            lambda p:p['models']['clic_clips_l14']['loader'].update(torch_dtype='bfloat16'),
            lambda p:p['models']['clic_clips_l14']['native_protocol'].update(text_cleaner='lowercase'),
            lambda p:p['parent_native_settings'].update(seed=999),
            lambda p:p['dependencies'].update(sha256='f'*64),
            lambda p:p['parent_model_signature'].update(tokenizer='other'),
            lambda p:p['settings'].update(main_k=50),lambda p:p.update(fixed_baseline_test=False)]
        for change in changes:
            value=deepcopy(plan);change(value)
            with patch.object(facade,'admit',return_value=qualified),self.assertRaises(ValueError):C.admit_models(value)

    def test_cross_family_old_scope_wrong_parent_and_missing_model_never_admit(self):
        plan,qualified,_,_=self.fixture();facade=C.adapter(plan['qualification_adapter'])
        for field,value in (('scope','official-native-p0-qualified'),('family','MLLM'),('model','other')):
            with patch.object(facade,'admit',return_value={**qualified,field:value}),self.assertRaises(ValueError):C.admit_models(plan)
        value=deepcopy(plan);value['parent_p0_plan']['sha256']='f'*64
        with patch.object(facade,'admit',return_value=qualified),self.assertRaises(ValueError):C.admit_models(value)
        value=deepcopy(plan);value['qualification']={}
        with patch.object(facade,'admit',return_value=qualified),self.assertRaises(ValueError):C.admit_models(value)

    def test_builder_real_adapter_stub_precedes_data_audit_and_data_open(self):
        plan,qualified,parent,stub=self.fixture();facade=C.adapter(plan['qualification_adapter']);events=[]
        development=self.root/'dl/retrieval-matrix/data/coco-development-v2'
        panel=self.root/'codex/retrieval_matrix_panel'
        for path in (development/'manifest.json',development/'images-summary.json',
                     development/'dev/images-receipt.jsonl',panel/'development-sources.json'):
            path.parent.mkdir(parents=True,exist_ok=True);path.write_text('{}\n')
        audit=self.root/'synthetic-audit.json';audit.write_text('{}\n')
        def admit(*args):events.append('real-registered-adapter');return qualified
        def verify(*args):events.append('data-audit')
        def datasets(*args):events.append('panel-data');return {'synthetic_unit_only':True}
        # Only data fixtures/audit are stubbed; adapter resolution and complete
        # model/parent/runtime binding remain the actual implementation.
        with patch.object(facade,'admit',side_effect=admit) as called,patch.object(B,'ROOT',self.root), \
                patch.object(B.data_audit,'verify',side_effect=verify),patch.object(B.data_audit,'bind_datasets'), \
                patch.object(B,'legacy_builder',return_value=SimpleNamespace(datasets=datasets)):
            value=B.build(parent,stub,'siglip2-einops-v1',audit)
        called.assert_called_once();self.assertEqual(events,['real-registered-adapter','data-audit','panel-data'])
        self.assertEqual(value['state'],'REVIEW_REQUIRED');self.assertIs(value['reviewed'],False)
        self.assertEqual(value['models'],plan['models']);self.assertEqual(value['qualification_adapter'],plan['qualification_adapter'])
        self.assertTrue(value['fixed_baseline_test']);self.assertEqual(value['settings']['top_ks'],[50,100,200])
        # This synthetic in-memory return value is never written as a P2 plan.

    def test_private_runtime_keeps_shared_modules_and_exact_model_factories(self):
        from codex.retrieval_matrix import runner as shared
        from codex.retrieval_matrix_extra import runner as extra
        from codex.retrieval_matrix_paired_runtime_fix import models as paired
        from codex.retrieval_matrix_native_p0.models import Siglip2
        before=(shared.models,shared.source_gate,shared.runtime_environment,extra.R.models)
        plan,_,_,_=self.fixture();engine=R.create_runtime(plan)
        self.assertIsNot(engine,shared)
        self.assertEqual(before,(shared.models,shared.source_gate,shared.runtime_environment,extra.R.models))
        self.assertIs(C.model_class(C.adapter_reference('siglip2-einops-v1')),Siglip2)
        self.assertIs(C.model_class(C.adapter_reference('paired-clic-einops-v1')),paired.create)

    def test_panel_fp32_cpu_dot_stable_gallery_ties_and_native_dtype_rejection(self):
        plan,_,_,_=self.fixture();engine=R.create_runtime(plan)
        images=np.array([[1,0],[1,0],[0,1]],dtype=np.float32);texts=np.array([[1,0]],dtype=np.float32)
        np.testing.assert_array_equal(engine.rank_blocks(images,texts),[[0,1,2]])
        with self.assertRaisesRegex(ValueError,'FP32'):engine.rank_blocks(images.astype(np.float64),texts)

    def test_runtime_dispatches_exact_parent_to_its_actual_facade_without_cuda(self):
        import torch
        for key in PARENTS:
            plan,_,_,_=self.fixture(key);engine=R.create_runtime(plan)
            module=C.registered_module(plan['qualification_adapter'],'runtime_module')
            expected=C.read(C.checked(plan['parent_p0_plan']))
            with patch.dict(os.environ,{'RETRIEVAL_CENTRAL_OWNER':'[codex]','RETRIEVAL_CENTRAL_TASK':'synthetic-unit-only',
                        'CUDA_VISIBLE_DEVICES':'synthetic-no-gpu'}), \
                    patch.object(torch.cuda,'is_available',return_value=True),patch.object(torch.cuda,'device_count',return_value=1), \
                    patch.object(module,'runtime',return_value={'synthetic_unit_only':True}) as runtime:
                value=engine.runtime_environment(plan['models'][key])
            runtime.assert_called_once_with(expected,gpu=True)
            self.assertEqual(value['native_qualified_runtime'],{'synthetic_unit_only':True})
        self.assertFalse(torch.cuda.is_initialized())

    def test_model_constructor_rejects_changed_class_processor_tokenizer_signature(self):
        plan,_,_,_=self.fixture('clic_clips_l14');engine=R.create_runtime(plan)
        # No real model/weights: the actual construction boundary must reject
        # changed processor/tokenizer identity before any encode can occur.
        fake=SimpleNamespace(signature=lambda:{'tokenizer':'changed'},encode=lambda *_:self.fail('encode must not run'))
        with patch.object(C,'model_class',return_value=lambda spec:fake), \
                self.assertRaisesRegex(ValueError,'class/processor/raw tokenizer'):
            engine.models.OfficialModel(plan['models']['clic_clips_l14'])

    def test_central_guard_and_new_outputs_reject_without_cuda(self):
        plan,_,_,_=self.fixture();engine=R.create_runtime(plan)
        with patch.dict(os.environ,{},clear=True),self.assertRaisesRegex(ValueError,'central dispatcher'):
            engine.runtime_environment(plan['models']['siglip2'])
        with self.assertRaises(ValueError):C.output_path(C.ROOT/'dl/retrieval-matrix-native-p0/runs/old')
        import torch
        self.assertFalse(torch.cuda.is_initialized())

    def test_missing_executing_source_or_swapped_fixed_dataset_is_rejected(self):
        plan,_,_,_=self.fixture();plan['source_files']=C.parent_sources(plan)
        C.check_declared_sources(plan)
        plan['source_files'].pop('codex/retrieval_matrix_paired_runtime_fix/qualification.py',None)
        plan['source_files'].pop('codex/retrieval_matrix_clip_panel_next/runner.py')
        with self.assertRaisesRegex(ValueError,'source table'):C.check_declared_sources(plan)
        with patch.object(B,'legacy_builder',return_value=SimpleNamespace(datasets=lambda _: {'fixed':'v2'})), \
                self.assertRaisesRegex(ValueError,'data entries'):
            B.verify_datasets({'datasets':{'fixed':'v1'}})


if __name__=='__main__':unittest.main()
