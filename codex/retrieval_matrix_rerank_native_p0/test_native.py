"""[codex] Synthetic CPU only: native axes, groups, unknown targets, recovery."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import tempfile,unittest
import numpy as np
from codex.retrieval_matrix_rerank_native_p0 import contracts as C,runner as N,qualification as Q


def variant(backend='swift_released',rule='cola_text_choice'):
    return {'model':{'backend':backend},'cola_rule':rule,'targets':{name:{'verified_protocol_match':False,'target':None,'gap':'No exact published target'} for name in C.CANONICAL}}


def complete_report(v):
    subsets={n:{'rows':count,'accuracy':.8} for n,count in C.COUNTS.items()}
    definitions={v['cola_rule']:['cola'],'scpp':list(C.SCPP),'negbench_mcq':['coco_mcq','voc_mcq']}
    return {'rows':15912,'subsets':subsets,'benchmarks':{task:{'subsets':sorted(names),'rows':sum(C.COUNTS[n] for n in names),
        'macro_accuracy':sum(subsets[n]['accuracy'] for n in names)/len(names)} for task,names in definitions.items()}}


class NativeContracts(unittest.TestCase):
    def test_cola_image_choice_is_not_text_choice_and_ties_fail(self):
        values=[.3,.2,.4,.5]
        self.assertEqual(C.M.hit('cola_text_choice',values),1);self.assertEqual(C.M.hit('cola_image_choice',values),0)
        for task in ('cola_text_choice','cola_image_choice'):self.assertEqual(C.M.hit(task,[.5]*4),0)
        self.assertEqual(C.M.hit('scpp',[.5,.5,.5]),0);self.assertEqual(C.M.hit('negbench_mcq',[.5]*4),0)

    def test_jina_released_groups_are_two_plus_two_three_four(self):
        rows=[{'subset':s,'pairs':[{'id':s+str(i)} for i in range(width)]} for s,width in [('cola',4),('replace_att',3),('coco_mcq',4)]]
        groups=C.call_groups(rows,variant('jina_native','cola_image_choice'))
        self.assertEqual([len(g) for g in groups],[2,2,3,4])
        self.assertEqual([p['id'] for g in groups for p in g],[p['id'] for r in rows for p in r['pairs']])

    def test_storage32_examples_preserves_exact_swift_batch32_boundaries(self):
        for width in (3,4):
            rows=[{'subset':'synthetic','pairs':[{'id':f'{i}:{j}'} for j in range(width)]} for i in range(79)]
            whole=C.call_groups(rows,variant());chunked=[]
            for start in range(0,len(rows),32):chunked+=C.call_groups(rows[start:start+32],variant())
            self.assertEqual(whole,chunked);self.assertTrue(all(len(g)==32 for g in whole[:-1]))

    def test_native_and_pointwise_are_single_pair_calls(self):
        rows=[{'subset':'cola','pairs':[{'id':str(i)} for i in range(4)]}]
        for backend in ('qwen_native','mcmr_pointwise'):self.assertEqual([len(g) for g in C.call_groups(rows,variant(backend))],[1]*4)

    def test_raw_scalar_and_original_representation_are_retained(self):
        pairs=[{'id':'one'}]
        cases=[('swift_released',[.5]),('qwen_native',.5),('jina_native',.5),('mcmr_pointwise',{'p_true':.5,'true_logprob':-1.,'false_logprob':-1.})]
        for backend,raw in cases:
            v=variant(backend);rows=[{'id':'one','score':.5,'official_raw':raw}];C.validate_scores(rows,pairs,v)
            for changed in (float('nan'),-.1,1.1,1):
                bad=deepcopy(rows);bad[0]['score']=changed
                with self.assertRaises(ValueError):C.validate_scores(bad,pairs,v)
            bad=deepcopy(rows);bad[0]['id']='other'
            with self.assertRaises(ValueError):C.validate_scores(bad,pairs,v)

    def test_missing_or_one_published_benchmark_never_qualifies(self):
        v=variant('qwen_native');report=complete_report(v)
        self.assertEqual(C.assess(report,v)['state'],'TARGETS_INCOMPLETE')
        v['targets']['COLA'].update(verified_protocol_match=True,target=.8)
        self.assertEqual(C.assess(report,v)['state'],'TARGETS_INCOMPLETE')
        v['targets']['SugarCrepe++'].update(verified_protocol_match=True,target=.8)
        result=C.assess(report,v);self.assertEqual(result['state'],'NUMERIC_MATCH');self.assertTrue(result['independent_review_required'])
        self.assertEqual(set(result['benchmarks']),set(C.CANONICAL))

    def test_incomplete_macro_wrong_denominator_and_nonfinite_fail(self):
        v=variant('jina_native','cola_image_choice');base=complete_report(v)
        for mutate in (lambda r:r['subsets']['voc_mcq'].update(rows=4951),lambda r:r['benchmarks']['scpp'].update(macro_accuracy=.1),
                       lambda r:r['benchmarks']['scpp'].update(macro_accuracy=float('nan')),lambda r:r['benchmarks'].pop('negbench_mcq')):
            report=deepcopy(base);mutate(report)
            with self.assertRaises(ValueError):C.assess(report,v)

    def test_central_guard_rejects_unassigned_before_cuda(self):
        with patch.dict(N.os.environ,{},clear=True):
            with self.assertRaisesRegex(ValueError,'central dispatcher'):N.environment({})

    def test_raw_transactions_survive_state_crash_without_replaying_native_scores(self):
        rows=[{'subset':'synthetic','instruction':'exact','pairs':[{'id':f'{i}:{j}'} for j in range(3)]} for i in range(70)]
        v=variant();calls=[];identity={'variant':'synthetic'}
        def score(pairs,**kwargs):
            calls.append([p['id'] for p in pairs]);value=.1+len(calls)/100
            return [{'id':p['id'],'score':value,'official_raw':[value]} for p in pairs]
        engine=SimpleNamespace(score_pairs=score)
        with tempfile.TemporaryDirectory(dir=C.ROOT/'hf_home/tmp',prefix='native-mcq-resume-') as folder:
            out=Path(folder);C.R.fixed_json(out/'backend.json',{'backend':'synthetic'})
            def compute(batch):
                lookup={}
                for group in C.call_groups(batch,v):
                    lookup.update({r['id']:r['score'] for r in N.raw_group(out,group,identity,v,'exact',lambda:engine)})
                return np.array([[lookup[p['id']] for p in row['pairs']] for row in batch],dtype=np.float64)
            original=C.R.C.atomic_json
            def crash(path,obj):
                if Path(path).name=='score.state.json':raise RuntimeError('synthetic state crash')
                return original(path,obj)
            with patch.object(C.R.C,'atomic_json',side_effect=crash):
                with self.assertRaises(RuntimeError):C.R.checkpoint(out/'score.npy',rows,32,compute,identity)
            self.assertEqual(len(calls),3)
            values=C.R.checkpoint(out/'score.npy',rows,32,compute,identity)
            self.assertEqual(calls,[[p['id'] for p in g] for g in C.call_groups(rows,v)])
            np.testing.assert_array_equal(values[:32].reshape(-1),np.repeat([.1+i/100 for i in (1,2,3)],32))
            self.assertEqual(N.cost_summary(out)['attempted_pairs'],210)
            before=len(calls);C.R.checkpoint(out/'score.npy',rows,32,compute,identity);self.assertEqual(before,len(calls))
            with self.assertRaises(ValueError):C.R.checkpoint(out/'score.npy',rows,32,compute,{'variant':'changed'})

    def test_original_mcmr_labelscorer_keeps_true_false_two_forwards(self):
        import torch
        from codex.retrieval_matrix_rerank import backends as B
        specs=C.R.read(Path(__file__).parent/'backend_specs.draft.json')['models'];module=B.module(specs['qwen_instruct8']['source'])
        calls=[];messages=[]
        class Tokenizer:
            def __call__(self,label,**kwargs):return SimpleNamespace(input_ids=[2 if label=='True' else 3])
        class Processor:
            def apply_chat_template(self,rows,**kwargs):messages.extend(rows);return 'synthetic'
            def __call__(self,**kwargs):return {'input_ids':torch.tensor([[1,5]]),'attention_mask':torch.ones((1,2),dtype=torch.long)}
        class Model:
            device=torch.device('cpu')
            def __call__(self,**kwargs):
                calls.append(kwargs['input_ids'].tolist());logits=torch.zeros((1,3,8));logits[:,:,2]=2;logits[:,:,3]=1
                return SimpleNamespace(logits=logits)
        scorer=module.LabelScorer(Tokenizer(),Processor(),Model());prob,true_lp,false_lp=scorer.score_pair('caption','',None)
        self.assertEqual([r[0][-1] for r in calls],[2,3]);self.assertEqual(len(calls),2)
        self.assertEqual(messages[0]['content'],module.CONFIG['SYSTEM_PROMPT'])
        self.assertIn('Candidate: \n',messages[1]['content'][1]['text'])
        self.assertEqual(prob,float(torch.softmax(torch.tensor([true_lp,false_lp],dtype=torch.float32),dim=-1)[0]))
        self.assertFalse(torch.cuda.is_initialized())


class NativeEvidence(unittest.TestCase):
    def test_complete_independent_vectorized_metrics_agree_with_original_strict_rules(self):
        v=variant('jina_native','cola_image_choice');rows=[];scores=[]
        for subset,count in C.COUNTS.items():
            width=3 if subset in C.SCPP else 4
            task=v['cola_rule'] if subset=='cola' else ('scpp' if subset in C.SCPP else 'negbench_mcq')
            for i in range(count):
                pairs=[{'id':f'{subset}:{i}:{j}'} for j in range(width)]
                rows.append({'id':f'{subset}:{i}','task':task,'subset':subset,'pairs':pairs})
                values=([.8,.2,.1,.7] if subset=='cola' else ([.8,.7,.2] if width==3 else [.8,.2,.2,.2])) if i%3 else [.5]*width
                scores.extend({'id':p['id'],'score':float(value)} for p,value in zip(pairs,values,strict=True))
        produced,_=C.evaluate(rows,scores,v);actual=Q.independent_metrics(rows,scores,v)
        self.assertTrue(Q.equivalent(produced,actual));self.assertEqual(produced['qualification']['state'],'TARGETS_INCOMPLETE')
        self.assertEqual(produced['rows'],15912);self.assertEqual(len(scores),58891)
        self.assertEqual(set(produced['diagnostic_cola_alternate_axis']),{'cola_text_choice'})
        bad=deepcopy(actual);bad['benchmarks']['scpp']['macro_accuracy']=float('nan');self.assertFalse(Q.equivalent(bad,actual))

    def test_array_state_binds_full_original_identity_items_and_batch_boundaries(self):
        items=[{'id':str(i)} for i in range(70)];identity={'model':'synthetic','request':'fixed'}
        with tempfile.TemporaryDirectory(dir=C.ROOT/'hf_home/tmp',prefix='native-array-proof-') as folder:
            path=Path(folder)/'score.npy'
            C.R.checkpoint(path,items,32,lambda batch:np.ones((len(batch),3)),identity)
            C.bound_array(path,items,identity)
            for different in ({**identity,'request':'changed'},{**identity,'model':'other'}):
                with self.assertRaises(ValueError):C.bound_array(path,items,different)
            with self.assertRaises(ValueError):C.bound_array(path,[*items[:-1],{'id':'other'}],identity)
            state=C.R.read(path.with_suffix('.state.json'));state['batches'][0]['end']=31;C.R.atomic_json(path.with_suffix('.state.json'),state)
            with self.assertRaises(ValueError):C.bound_array(path,items,identity)

    def test_qwen_silent_null_fallback_is_rejected_before_forward(self):
        import torch
        from codex.retrieval_matrix_rerank_native_p0 import adapters as A
        parameter=SimpleNamespace(device=SimpleNamespace(type='cuda'),dtype=torch.bfloat16,is_floating_point=lambda:True)
        model=SimpleNamespace(is_quantized=False,parameters=lambda:iter([parameter]))
        engine=SimpleNamespace(model=model,tokenize=lambda messages,**kwargs:{'input_ids':[[1,2]]})
        backend=SimpleNamespace(engine=engine)
        with patch.dict(A.B.BACKENDS,{'qwen_native':lambda spec:backend}):
            actual=A.create(variant('qwen_native'))
            with self.assertRaisesRegex(ValueError,'silently dropped'):actual.engine.tokenize([[{'role':'user'}]])
        self.assertFalse(torch.cuda.is_initialized())

    def test_actual_plan_builder_retains_missing_targets_and_rejects_data_rebinding(self):
        from codex.retrieval_matrix_rerank_native_p0 import build_plan as P
        here=Path(__file__).parent
        plan=P.build(here/'backend_specs.draft.json',here/'protocols.json',C.ROOT/'dl/retrieval-matrix-rerank-native-p0/cpu-probe.json')
        N.validate(plan);self.assertEqual(plan['state'],'REVIEW_REQUIRED');self.assertIs(plan['reviewed'],False)
        for key in ('qwen_native2','qwen_native8','qwen_instruct8'):
            self.assertTrue(all(t['target'] is None for t in plan['variants'][key]['targets'].values()))
        self.assertTrue(all(v['model']['verified_protocol'] is False for v in plan['variants'].values()))
        bad=deepcopy(plan);bad['data_reference']['sha256']='a'*64
        with self.assertRaisesRegex(ValueError,'Published targets'):N.validate(bad)


if __name__=='__main__':unittest.main()
