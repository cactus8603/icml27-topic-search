"""[codex] CPU-only synthetic bridge and refusal tests, never true Q/results."""
from copy import deepcopy
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from codex.retrieval_matrix_panel_runtime_next import contracts as C,build_plan as B,runner as R
Q=C.Q


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,sort_keys=True)+'\n')


class Bridge(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=C.ROOT/'hf_home/tmp',prefix='mllm-panel-synthetic-')
        self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)

    def fixture(self,family='core-qwen3-9943159',key='core2'):
        base=C.parent(family)
        spec=base['models'][key]
        signature={'synthetic_unit_only':True,'implementation':deepcopy(spec['official']),
            'loader':deepcopy(spec.get('loader',{})),'processor_configuration':deepcopy(spec.get('processor',{})),
            'batches':deepcopy(spec['batches'])}
        environment={**deepcopy(spec.get('environment',{})),
            'modules':{'torch':{'path':'synthetic-path','version':'synthetic'}},'visible_devices':'GPU-original'}
        if family=='core-qwen3-9943159':
            marker=base['models'][key]['runtime_compatibility']
            signature['runtime_compatibility']=marker;environment['runtime_compatibility']=marker
        else:
            manifest=Q.read(Q.checked(base['runtime_fix']['overlay']))
            environment['modules']['einops']=manifest['modules']['einops']
            environment['runtime_fix']={'overlay_manifest':base['runtime_fix']['overlay'],
                'parent_freeze_commit':base['runtime_fix']['parent_freeze_commit']}
        reports={}
        for name in ('coco','flickr','negbench'):
            folder=self.root/(key+'-'+name);processor=folder/'processor.json'
            write(processor,signature);path=folder/'results/report.json'
            write(path,{'scope':'SYNTHETIC_UNIT_ONLY_NO_MODEL_RESULTS','provenance':{'index':{
                'state':'COMPLETE','processor_sha256':Q.sha(processor),'identity':{
                    'schema':1,'owner':'[codex]','model':key,'dataset':name,
                    'plan_sha256':C.FAMILIES[family]['sha256'],'freeze_commit':C.REF,
                    'model_spec':deepcopy(spec),'settings':deepcopy(base['settings']),
                    'source_files':C.producer_sources(base,family),'environment':environment}}}})
            reports[name]=Q.entry(path)
        stub=self.root/(key+'-synthetic-receipt.json')
        write(stub,{'scope':'SYNTHETIC_UNIT_ONLY_NOT_QUALIFICATION','model':'synthetic'})
        qualified={'state':'PASS','model':key,'model_spec_sha256':Q.digest(base['models'][key]),
            'p0_plan':Q.entry(C.ROOT/C.FAMILIES[family]['path']),
            'p0_plan_sha256':C.FAMILIES[family]['sha256'],'reports':reports}
        plan={'schema':1,'state':'REVIEW_REQUIRED','reviewed':False,'phase':C.PHASE,'retriever_family':'MLLM',
            'family_adapter':family,'models':{key:base['models'][key]},'qualification':{key:Q.entry(stub)},
            'parent_p0_plan':qualified['p0_plan'],'settings':deepcopy(base['settings']),
            'parent_native_settings':deepcopy(base['settings']),'dependencies':deepcopy(base['dependencies']),
            'parent_model_signature':signature,'parent_runtime':environment,'fixed_baseline_test':True,
            'ranking':{'dtype':'float32','backend':'CPU dot','ties':'stable-gallery-order','main_k':100,'top_ks':[50,100,200]}}
        return base,qualified,plan,stub

    def real_gate_fixture(self,family,key,*,identity_change=None,meta_change=None,signature_change=None):
        """Temporary synthetic reports through real Q.make/admit, never a model result."""
        base,qualified,plan,_=self.fixture(family,key)
        reports={}
        for name,reference in qualified['reports'].items():
            path=Path(reference['path']);row=Q.read(path);meta=row['provenance']['index']
            if identity_change:identity_change(meta['identity'],name)
            if meta_change:meta_change(meta,name)
            processor=path.parent.parent/'processor.json'
            if signature_change:
                signature=Q.read(processor);signature_change(signature,name);write(processor,signature)
                meta['processor_sha256']=Q.sha(processor)
            ds=base['datasets'][name];target,fields=Q.DATASETS[name]
            official={};node=official
            for field in fields[:-1]:node=node.setdefault(field,{})
            node[fields[-1]]=base['models'][key]['published_targets'][target]
            row.update(model=key,dataset=name,queries=ds['expected_eligible'],
                raw_queries=ds['expected_queries'],gallery=ds['expected_gallery'],excluded=[],official=official)
            write(path,row);reports[name]=path
        review=self.root/(key+'-synthetic-independent-review.json')
        write(review,{'state':'PASS','schema':1,'scope':'official-native-p0',
            'reviewer':'synthetic-not-a-real-reviewer','producer':'synthetic-not-a-real-producer',
            'independent_recomputation':True,'audited_arrays':True,
            'plan_sha256':C.FAMILIES[family]['sha256'],'model':key,
            'reports':{name:Q.sha(path) for name,path in reports.items()}})
        receipt=self.root/(key+'-real-gate-SYNTHETIC-ONLY.json')
        result=Q.make(C.ROOT/C.FAMILIES[family]['path'],key,reports,review);write(receipt,result)
        self.assertEqual(Q.admit(Q.entry(receipt),base['models'][key]),result)
        return base,result,receipt

    def test_real_make_admit_accepts_exact_original_source_identity_in_both_families(self):
        for family,key in (('core-qwen3-9943159','core2'),('qwen2-einops-9943159','gme2')):
            base,qualified,receipt=self.real_gate_fixture(family,key)
            actual=C.admit_parent(family,key,Q.entry(receipt))
            self.assertEqual(actual[:2],(base,qualified))
            expected=C.producer_sources(base,family)
            self.assertEqual(expected[C.FAMILIES[family]['path']],C.FAMILIES[family]['sha256'])
            self.assertEqual(expected[base['prereg']],base['source_files'][base['prereg']])

    def test_real_make_admit_cannot_pass_wrong_ref_or_incomplete_or_extra_producer_sources(self):
        def omit_plan(identity,name):
            identity['source_files'].pop(next(k for k in identity['source_files'] if k.endswith('p0-qwen3-fixed-v1.json') or k.endswith('p0-qwen2-einops-v1.json')))
        changes=[lambda i,n:i.update(freeze_commit='0'*40),
            lambda i,n:i.update(source_files={'WRONG_OR_OLD_PRODUCER_SOURCE':'0'*64}),
            omit_plan,lambda i,n:i['source_files'].update({'undeclared-source.py':'0'*64}),
            lambda i,n:i['source_files'].update({next(iter(i['source_files'])):'0'*64}),
            lambda i,n:i.update(freeze_commit='0'*40) if n=='flickr' else None]
        for family,key in (('core-qwen3-9943159','core2'),('qwen2-einops-9943159','gme2')):
            for index,change in enumerate(changes):
                with self.subTest(family=family,change=index):
                    _,_,receipt=self.real_gate_fixture(family,key,identity_change=change)
                    with self.assertRaisesRegex(ValueError,'producer identity/source/ref'):
                        C.admit_parent(family,key,Q.entry(receipt))

    def test_real_make_admit_requires_complete_index_actual_environment_and_static_signature(self):
        for family,key in (('core-qwen3-9943159','core2'),('qwen2-einops-9943159','gme2')):
            with self.subTest(family=family,field='index-state'):
                _,_,receipt=self.real_gate_fixture(family,key,meta_change=lambda m,n:m.update(state='FAILED'))
                with self.assertRaisesRegex(ValueError,'producer identity/source/ref'):
                    C.admit_parent(family,key,Q.entry(receipt))
            with self.subTest(family=family,field='environment'):
                original=C.parent(family)['models'][key]['environment'];field=next(iter(original))
                _,_,receipt=self.real_gate_fixture(family,key,identity_change=lambda i,n:i['environment'].update({field:'WRONG-VERSION'}))
                with self.assertRaisesRegex(ValueError,'actual P0 environment'):
                    C.admit_parent(family,key,Q.entry(receipt))
            for field in ('implementation','loader','processor_configuration','batches'):
                with self.subTest(family=family,field=field):
                    _,_,receipt=self.real_gate_fixture(family,key,signature_change=lambda s,n:s.update({field:{'wrong':True}}))
                    with self.assertRaisesRegex(ValueError,'processor signature'):
                        C.admit_parent(family,key,Q.entry(receipt))

    def test_literal_parents_and_six_models_are_frozen_without_model_results(self):
        for family,profile in C.FAMILIES.items():
            base=C.parent(family)
            self.assertEqual(set(base['models']),set(profile['models']))
            self.assertEqual(Q.sha(C.ROOT/profile['path']),profile['sha256'])
        for family in ('CLIP','embedding','mcmr','other'):
            with self.assertRaises(ValueError):C.parent(family)

    def test_each_family_uses_true_panel_admit_with_full_spec_then_actual_evidence(self):
        for family,profile in C.FAMILIES.items():
            for key in profile['models']:
                with self.subTest(key=key):
                    base,qualified,plan,_=self.fixture(family,key)
                    with patch.object(Q,'admit',return_value=qualified) as admit:
                        actual=C.admit(plan)
                    admit.assert_called_once_with(plan['qualification'][key],base['models'][key])
                    self.assertEqual(actual,(base,qualified))

    def test_real_admit_rejects_stub_before_audit_or_panel_read(self):
        for family,key in (('core-qwen3-9943159','core2'),('qwen2-einops-9943159','gme2')):
            _,_,_,stub=self.fixture(family,key)
            with patch.object(Q,'admit',wraps=Q.admit) as admit,patch.object(B.data_audit,'verify') as audit,patch.object(B,'datasets') as reader:
                with self.assertRaises((ValueError,KeyError)):B.build(family,key,stub,self.root/'missing-audit.json')
                admit.assert_called_once();audit.assert_not_called();reader.assert_not_called()

    def test_parent_settings_spec_signature_runtime_and_family_cannot_change(self):
        _,qualified,plan,_=self.fixture()
        mutations=[lambda p:p['models']['core2']['batches'].update(image=1),
            lambda p:p['settings'].update(seed=999),lambda p:p['parent_native_settings'].update(seed=999),
            lambda p:p['parent_runtime']['modules']['torch'].update(path='shadow'),
            lambda p:p['parent_model_signature'].update(implementation='other'),
            lambda p:p['dependencies'].update(packages={}),lambda p:p.update(fixed_baseline_test=False),
            lambda p:p['parent_p0_plan'].update(sha256='0'*64),lambda p:p.update(retriever_family='CLIP'),
            lambda p:p['ranking'].update(ties='unstable')]
        for mutate in mutations:
            bad=deepcopy(plan);mutate(bad)
            with patch.object(Q,'admit',return_value=qualified),self.assertRaises(ValueError):C.admit(bad)

    def test_same_model_spec_cannot_borrow_another_parent_qualification(self):
        _,qualified,plan,_=self.fixture()
        for field,value in (('p0_plan',{'path':'another-plan','sha256':'0'*64}),('p0_plan_sha256','0'*64),('model','qwen2')):
            bad=deepcopy(qualified);bad[field]=value
            with patch.object(Q,'admit',return_value=bad),self.assertRaisesRegex(ValueError,'exact repaired parent'):
                C.admit(plan)

    def test_processor_and_runtime_must_match_all_three_qualified_benchmarks(self):
        base,qualified,_,_=self.fixture()
        ref=qualified['reports']['flickr'];path=Path(ref['path']);value=Q.read(path)
        value['provenance']['index']['identity']['environment']['modules']['torch']['path']='other'
        write(path,value);qualified['reports']['flickr']=Q.entry(path)
        with self.assertRaisesRegex(ValueError,'All three'):
            C.evidence_identity(qualified,base,'core-qwen3-9943159','core2')

    def test_qwen2_overlay_marker_cannot_be_borrowed_or_omitted(self):
        _,qualified,plan,_=self.fixture('qwen2-einops-9943159','gme2')
        for ref in qualified['reports'].values():
            path=Path(ref['path']);value=Q.read(path);value['provenance']['index']['identity']['environment'].pop('runtime_fix')
            write(path,value);ref.update(Q.entry(path))
        with patch.object(Q,'admit',return_value=qualified),self.assertRaisesRegex(ValueError,'einops overlay'):C.admit(plan)

    def test_builder_admission_precedes_dev_audit_and_open_without_writing_plan(self):
        base,qualified,plan,stub=self.fixture();events=[]
        dev=self.root/'dl/retrieval-matrix/data/coco-development-v2';panel=self.root/'codex/retrieval_matrix_panel'
        for path in (dev/'manifest.json',dev/'images-summary.json',dev/'dev/images-receipt.jsonl',panel/'development-sources.json'):
            write(path,{})
        audit=self.root/'synthetic-audit.json';write(audit,{})
        def admitted(*args):events.append('genuine-admit-boundary-stub');return base,qualified,plan['parent_model_signature'],plan['parent_runtime']
        with patch.object(C,'admit_parent',side_effect=admitted),patch.object(B,'ROOT',self.root), \
                patch.object(B.data_audit,'verify',side_effect=lambda *args:events.append('audit')), \
                patch.object(B.data_audit,'bind_datasets'),patch.object(B,'datasets',side_effect=lambda *args:events.append('panel') or {'synthetic':True}):
            result=B.build(plan['family_adapter'],'core2',stub,audit)
        self.assertEqual(events,['genuine-admit-boundary-stub','audit','panel'])
        self.assertEqual(result['state'],'REVIEW_REQUIRED');self.assertIs(result['reviewed'],False)
        self.assertEqual(result['settings'],base['settings']);self.assertEqual(result['models'],plan['models'])

    def test_private_runtime_preserves_canonical_globals_and_exact_original_classes(self):
        from codex.retrieval_matrix import runner as shared
        from codex.retrieval_matrix_core_runtime_fix import compat
        from codex.retrieval_matrix_qwen2.adapters import Qwen2OfficialModel
        before=(shared.models.OfficialModel,shared.source_gate,shared.runtime_environment,shared.load_data)
        for family,key,cls in (('core-qwen3-9943159','core2',compat.CompatibleOfficialModel),
                ('qwen2-einops-9943159','gme2',Qwen2OfficialModel)):
            _,_,plan,_=self.fixture(family,key);engine=R.create_runtime(plan)
            self.assertIsNot(engine,shared)
            self.assertIs(inspect.getclosurevars(engine.models.OfficialModel).nonlocals['encoder'],cls)
            self.assertEqual(before,(shared.models.OfficialModel,shared.source_gate,shared.runtime_environment,shared.load_data))

    def test_existing_import_patch_fails_closed_in_a_fresh_cpu_subprocess(self):
        body='from codex.retrieval_matrix_panel import runner; from codex.retrieval_matrix_panel_runtime_next import runner as new; '
        body+='new.original_class_guard()'
        value=subprocess.run([sys.executable,'-B','-c',body],cwd=C.ROOT,capture_output=True,text=True)
        self.assertNotEqual(value.returncode,0);self.assertIn('fresh process',value.stderr)
        self.assertNotIn('TypeError',value.stderr)

    def test_construct_checks_actual_signature_before_any_encode(self):
        from codex.retrieval_matrix_core_runtime_fix import compat
        _,_,plan,_=self.fixture()
        model=SimpleNamespace(signature=lambda:{'different':True},encode=lambda *args:self.fail('No encode'))
        with patch.object(compat,'CompatibleOfficialModel',return_value=model):
            engine=R.create_runtime(plan)
            with self.assertRaisesRegex(ValueError,'signature differs'):engine.models.OfficialModel(plan['models']['core2'])

    def test_runtime_exact_modules_marker_and_only_gpu_uuid_may_differ(self):
        base={'modules':{'torch':{'path':'same','version':'same'}},'visible_devices':'GPU-a','runtime_fix':'same'}
        actual=deepcopy(base);actual['visible_devices']='GPU-b';C.runtime_matches(actual,base)
        for change in (lambda a:a['modules']['torch'].update(path='shadow'),lambda a:a.update(runtime_fix='old')):
            bad=deepcopy(actual);change(bad)
            with self.assertRaisesRegex(ValueError,'Actual P2 runtime'):C.runtime_matches(bad,base)

    def test_runtime_receipt_preserves_original_core_and_qwen2_semantics_without_cuda(self):
        import torch
        from importlib.metadata import version
        from codex.retrieval_matrix_qwen2_runtime_fix import contracts as repair
        for family,key in (('core-qwen3-9943159','core2'),('qwen2-einops-9943159','gme2')):
            base,_,plan,_=self.fixture(family,key);model=plan['models'][key]
            old_runtime=deepcopy(plan['parent_runtime']);reduced=deepcopy(old_runtime)
            if family=='core-qwen3-9943159':
                reduced.pop('runtime_compatibility')
                expected_call=model
                modules={}
            else:
                reduced={**{k:v for k,v in model['environment'].items() if k!='peft'},
                    'modules':{'torch':{'path':'synthetic-path','version':'synthetic'}},'visible_devices':'GPU-original'}
                modules={name:SimpleNamespace(__version__=version(distribution),__file__=str(self.root/name/'__init__.py'))
                    for name,distribution in (('peft','peft'),('flash_attn','flash-attn'))}
                old_runtime={**deepcopy(reduced),'peft':version('peft')}
                old_runtime['modules'].update({name:{'version':m.__version__,'path':str(Path(m.__file__).resolve())} for name,m in modules.items()})
                old_runtime['modules']['einops']=plan['parent_runtime']['modules']['einops']
                old_runtime['runtime_fix']=plan['parent_runtime']['runtime_fix']
                expected_call={**model,'environment':{k:v for k,v in model['environment'].items() if k!='peft'}}
            plan['parent_runtime']=old_runtime;engine=R.create_runtime(plan)
            # Replace only the private closure's base CUDA reader, not any
            # canonical module/global. Exercise all actual receipt-extension code.
            cells=dict(zip(engine.runtime_environment.__code__.co_freevars,engine.runtime_environment.__closure__))
            base_reader=unittest.mock.Mock(return_value=deepcopy(reduced));cells['base_runtime'].cell_contents=base_reader
            with patch.dict(os.environ,{'RETRIEVAL_CENTRAL_OWNER':'[codex]','RETRIEVAL_CENTRAL_TASK':'synthetic',
                        'CUDA_VISIBLE_DEVICES':'synthetic-no-GPU'}),patch.dict(sys.modules,modules), \
                    patch.object(torch.cuda,'is_available',return_value=True),patch.object(torch.cuda,'device_count',return_value=1), \
                    patch.object(R.sys,'executable',str(C.ROOT/C.FAMILIES[family]['python'])), \
                    patch.object(C,'dependencies') as dependencies,patch.object(repair,'actual_modules',return_value={'einops':old_runtime['modules'].get('einops')}):
                value=engine.runtime_environment(model)
            self.assertEqual(value,old_runtime);base_reader.assert_called_once_with(expected_call)
            dependencies.assert_called_once_with(base,family,imports=True)
        self.assertFalse(torch.cuda.is_initialized())

    def test_changed_frozen_parent_git_bytes_are_rejected_without_source_edits(self):
        with patch.object(C.subprocess,'check_output',return_value=b'different fixed Git payload'):
            with self.assertRaisesRegex(ValueError,'literal frozen parent'):C.parent('core-qwen3-9943159')

    def test_central_guard_new_outputs_and_fp32_stable_gallery_ranking(self):
        _,_,plan,_=self.fixture();engine=R.create_runtime(plan)
        with patch.dict(os.environ,{},clear=True),self.assertRaisesRegex(ValueError,'central dispatcher'):
            engine.runtime_environment(plan['models']['core2'])
        for path in (C.OUTPUT_ROOT,C.ROOT/'dl/retrieval-matrix/runs/old'):
            with self.assertRaises(ValueError):C.output_path(path)
        image=np.array([[1,0],[1,0],[0,1]],dtype=np.float32);text=np.array([[1,0]],dtype=np.float32)
        np.testing.assert_array_equal(engine.rank_blocks(image,text),[[0,1,2]])
        with self.assertRaisesRegex(ValueError,'FP32'):engine.rank_blocks(image.astype('float64'),text)
        import torch
        self.assertFalse(torch.cuda.is_initialized())

    def test_source_table_omission_and_changed_fixed_dataset_fail_closed(self):
        base,_,plan,_=self.fixture();plan['source_files']=C.source_files(base);C.check_sources(plan,base)
        plan['source_files'].pop('codex/retrieval_matrix_panel_runtime_next/runner.py')
        with self.assertRaisesRegex(ValueError,'source table'):C.check_sources(plan,base)
        with patch.object(B,'datasets',return_value={'exact':'v2'}),self.assertRaisesRegex(ValueError,'complete fixed'):
            B.verify_datasets({'datasets':{'old':'v1'}})


if __name__=='__main__':unittest.main()
