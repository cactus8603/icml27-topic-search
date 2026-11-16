"""[codex] Prospective plan, role, CPU, and durable-read rejection tests."""
from copy import deepcopy
import json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from codex.retrieval_matrix_native_analysis import contracts as C,workflow as W,launcher as L
from codex.retrieval_matrix_native_analysis.build_plan import build

class WorkflowContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.plan=build()

    def test_seven_complete_manifests_derive_without_opening_any_production_output(self):
        roots=[Path(j['profile']['output_root']) for j in self.plan['jobs'].values()]
        original=Path.open
        def opened(path,*args,**kwargs):
            if any(path.resolve().is_relative_to(root) for root in roots):raise AssertionError('Source preparation opened a model effect')
            return original(path,*args,**kwargs)
        with patch.object(Path,'open',opened):
            for job in self.plan['jobs'].values():
                m=W.manifest(job['profile']);self.assertEqual(m,W.read(W.checked(job['manifest'])))
                self.assertTrue(m['all_paths_prospective']);self.assertFalse(m['production_paths_opened'])
                self.assertEqual(set(m['array_states'])-set(m['immutable_raw']),set())
                self.assertEqual(set(m['completion'])-set(m['immutable_raw']),set())
                if job['profile']['family']=='reranker':
                    self.assertEqual(len(m['array_states']),8)
                    self.assertEqual(len(m['mechanical_results']),2)
                    self.assertGreater(len(m['raw_call_paths']),1800)
                    self.assertEqual(set(m['raw_call_paths'])-set(m['immutable_raw']),set())
                else:self.assertIsNone(m['cost']['authoritative_call_ledger'])

    def test_draft_cannot_launch_or_change_frozen_source_job_model_paths_roles(self):
        for mutation in (
            lambda p:p['jobs']['native-rerank-qwen2'].update(producer_commit='a'*40),
            lambda p:p['jobs']['native-rerank-qwen2']['profile'].update(model='qwen_native2'),
            lambda p:p['jobs']['native-rerank-qwen2']['outputs'].update(qualification=str(C.ROOT/'hf_home/tmp/other.json')),
            lambda p:p['jobs']['native-rerank-qwen2']['roles'].update(excluded_reviewers=[]),
            lambda p:p['jobs']['native-rerank-qwen2']['cpu_runtime']['array_runtime']['numpy']['array_binary'].update(sha256='0'*64),
            lambda p:p['jobs'].pop('native-rerank-jina_m0'),
            lambda p:p.update(decision_order=['user-final-decision']),
        ):
            value=deepcopy(self.plan);mutation(value)
            with self.subTest(value=list(value['jobs'])):
                with self.assertRaises(ValueError):W.validate_plan(value,verify_manifests=False)
        with tempfile.TemporaryDirectory(dir=C.ROOT/'hf_home/tmp',prefix='native-draft-') as d:
            path=Path(d)/'draft.json';W.write_new(path,self.plan)
            with self.assertRaisesRegex(ValueError,'REVIEW_REQUIRED'):W.source_gate(path,'a'*40)

    def test_conservative_union_is_not_misrepresented_as_known_authorship(self):
        job=self.plan['jobs']['native-paired-clic_b32-einops-v1']
        known={x['actor'] for x in job['roles']['known_implementation_authors']}
        self.assertEqual(known,{'/root/official_p0/p3_contract_review'})
        self.assertIn('/root/public_data',job['roles']['excluded_reviewers'])
        for actor in W.EXCLUDED+['mechanical-actor','central-runtime:producer']:
            with self.assertRaisesRegex(ValueError,'Fresh nonimplementer'):
                W.independent_actor(job,actor,'mechanical-actor',nonimplementer=True)
        W.independent_actor(job,'/root/independent_numeric_review','mechanical-actor',nonimplementer=True)
        with self.assertRaises(ValueError):W.independent_actor(job,'/root/independent_numeric_review','mechanical-actor')

    def test_cpu_recipe_rejects_any_visible_gpu_before_import_or_result_read(self):
        job=self.plan['jobs']['native-rerank-qwen2']
        with patch.dict(os.environ,job['cpu']['environment']),patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'0'}):
            with self.assertRaisesRegex(ValueError,'CPU interpreter'):W.cpu_environment(job)

    def test_full_model_sha_and_producer_ref_are_required_in_analysis_receipts(self):
        job=self.plan['jobs']['native-rerank-qwen2']
        with tempfile.TemporaryDirectory(dir=C.ROOT/'hf_home/tmp',prefix='native-identity-') as d:
            path=Path(d)/'synthetic.json';path.write_text('{}\n')
            value=L.identity(path,'a'*40,job)
            L._same_identity(value,path,'a'*40,job)
            for key in ('producer_commit','model_spec_sha256','analysis_commit','producer_plan'):
                wrong=deepcopy(value);wrong[key]='changed'
                with self.assertRaisesRegex(ValueError,'different frozen'):L._same_identity(wrong,path,'a'*40,job)

    def test_excluded_independent_actor_is_rejected_before_any_producer_artifact_read(self):
        from argparse import Namespace
        job=self.plan['jobs']['native-rerank-qwen2']
        a=Namespace(plan=C.HERE/'SYNTHETIC-not-read.json',freeze_commit='a'*40,job=job['id'],
            stage='independent',actor='/root/official_p0',nonimplementer=True,accept_disclosed_protocol=True)
        with patch.object(W,'source_gate',return_value=self.plan),patch.object(W,'raw_snapshot') as raw,patch.object(W,'cpu_environment') as runtime:
            with self.assertRaisesRegex(ValueError,'Fresh nonimplementer'):L.run_worker(a)
            raw.assert_not_called();runtime.assert_not_called()

class NativeDetails(unittest.TestCase):
    def test_independent_per_example_math_preserves_both_cola_axes_order_and_ties(self):
        import numpy as np
        from codex.retrieval_matrix_native_analysis import synthetic as S
        from codex.retrieval_matrix_rerank_native_p0 import contracts as N
        for key in ('qwen2','jina_m0'):
            with self.subTest(variant=key),S.fixture('reranker',variant=key) as f:
                plan=W.read(f.plan_path);variant=plan['variants'][key];rows=N.load_examples(plan,variant)
                scores=[]
                for subset in N.COUNTS:
                    examples=[r for r in rows if r['subset']==subset];array=np.load(f.out/(subset+'.npy'),allow_pickle=False)
                    scores.extend({'id':p['id'],'score':float(value)} for row,values in zip(examples,array,strict=True) for p,value in zip(row['pairs'],values,strict=True))
                _,detail=N.evaluate(rows,scores,variant);path=f.out/'results/per-example.json';S.overwrite(path,detail)
                job={'producer_plan':C.entry(f.plan_path),'profile':{'family':'reranker','model':key,'score_root':str(f.out)}}
                value=W.independent_details(job);self.assertEqual(value['examples'],15912)
                self.assertTrue(value['strict_ties_fail'])
                changed=deepcopy(detail);changed[-1]['hit']=1-changed[-1]['hit'];S.overwrite(path,changed)
                with self.assertRaisesRegex(ValueError,'strict outcomes/order'):W.independent_details(job)

class DurablePrivateReader(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=C.ROOT/'hf_home/tmp',prefix='native-private-read-');self.root=Path(self.temp.name)
        self.prereg=self.root/'PREREG.md';self.prereg.write_bytes(b'Explicit synthetic-only criteria\n'+C.MARKER)
        self.plan=self.root/'plan.json';self.plan.write_text('{}\n');self.file=self.root/'score.npy';self.file.write_bytes(b'SYNTHETIC')
        self.anchor=self.root/'anchor';self.job={'id':'synthetic-native-job','producer_plan':{'sha256':'1'*64}}
    def tearDown(self):self.temp.cleanup()
    def reader(self):
        with patch.object(W,'HERE',self.root),patch.object(W,'ANCHORS',self.anchor):
            return W.session(self.plan,'a'*40,self.job,'synthetic-independent',[self.file])
    def records(self):return W.frozen_reader().records(self.prereg,expected_plan=C.sha(self.plan),expected_commit='a'*40,anchor_root=self.anchor)

    def test_durable_start_precedes_open_and_private_log_does_not_modify_shared_body(self):
        shared=C.read_header(C.ROOT/'codex/retrieval_matrix_analysis/PREREG.md')
        with self.reader() as reading:
            rows=self.records();self.assertEqual(rows[-1]['event'],'READ_STARTED');self.assertIsNone(rows[-1]['paths'][str(self.file)])
            self.assertEqual(self.file.read_bytes(),b'SYNTHETIC')
        self.assertEqual(self.records()[-1]['event'],'READ_COMPLETED')
        with patch.object(W,'HERE',self.root),patch.object(W,'ANCHORS',self.anchor):
            W.require_read(self.plan,'a'*40,reading.start)
        self.assertEqual(shared,C.read_header(C.ROOT/'codex/retrieval_matrix_analysis/PREREG.md'))

    def test_tail_all_tail_and_journal_one_sided_rollback_fail_closed(self):
        with self.reader():pass
        prior=self.prereg.read_bytes();journal=next(self.anchor.rglob('journal.jsonl'));anchor=journal.read_bytes()
        for body in (prior.splitlines(keepends=True)[:-1], [prior.split(C.MARKER)[0]+C.MARKER]):
            self.prereg.write_bytes(b''.join(body))
            with self.assertRaises(ValueError):self.records()
            self.prereg.write_bytes(prior)
        journal.write_bytes(anchor.splitlines(keepends=True)[0])
        with self.assertRaises(ValueError):self.records()

    def test_failed_read_preserves_start_and_records_failed_without_false_completion(self):
        with self.assertRaisesRegex(RuntimeError,'synthetic read failure'):
            with self.reader() as reading:raise RuntimeError('synthetic read failure')
        rows=self.records();self.assertEqual([r['event'] for r in rows],['READ_STARTED','READ_FAILED'])
        with patch.object(W,'HERE',self.root),patch.object(W,'ANCHORS',self.anchor):
            with self.assertRaisesRegex(ValueError,'successfully completed'):W.require_read(self.plan,'a'*40,reading.start)

if __name__=='__main__':unittest.main()
