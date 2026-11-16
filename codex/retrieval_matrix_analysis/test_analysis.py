"""[codex] CPU synthetic correctness/rejection tests; no real effects are opened."""
from copy import deepcopy
import hashlib,importlib.util,json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from codex.retrieval_matrix_analysis import contracts as C,reading as L,recompute as I,launcher as R

def original_metrics():
    s=importlib.util.spec_from_file_location('synthetic_original_mcq_metrics',C.ROOT/'codex/retrieval_matrix_rerank/metrics.py')
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def synthetic_mcq():
    rows=[];scores=[];arrays={}
    for subset,count in C.COUNTS.items():
        n=4 if subset=='cola' or subset.endswith('_mcq') else 3;a=[]
        task='cola_text_choice' if subset=='cola' else ('negbench_mcq' if subset.endswith('_mcq') else 'scpp')
        for i in range(count):
            values=([.8,.2,.3,.9] if subset=='cola' else ([.8,.9,.2] if n==3 else [.9,.2,.3,.4])) if i%3==0 else [.5]*n
            pairs=[{'id':f'{subset}:{i}:{j}'} for j in range(n)]
            rows.append({'id':f'{subset}:{i}','subset':subset,'task':task,'pairs':pairs})
            scores.extend({'id':p['id'],'score':v} for p,v in zip(pairs,values));a.extend(values)
        arrays[subset]=np.array(a,dtype=np.float32)[:,None]
    # Ensure original comparison uses exactly the checkpoint's native float32 values.
    flat={}
    for subset in C.COUNTS:
        ix=0
        for row in [r for r in rows if r['subset']==subset]:
            for pair in row['pairs']:
                flat[pair['id']]=float(arrays[subset][ix,0]);ix+=1
    return rows,[{'id':s['id'],'score':flat[s['id']]} for s in scores],arrays

def save_checkpoint(path,a,items,batch,identity):
    path.parent.mkdir(parents=True,exist_ok=True);np.save(path,a)
    state={'fingerprint':C.digest({'identity':identity,'items':items,'batch_size':batch}),'completed':len(items),
        'array_shape':list(a.shape),'dtype':a.dtype.str,'sha256':C.sha(path),'active_seconds':.1,
        'batches':[{'start':s,'end':min(s+batch,len(items)),
            'sha256':hashlib.sha256(np.ascontiguousarray(a[s:min(s+batch,len(items))]).tobytes()).hexdigest()} for s in range(0,len(items),batch)]}
    C.write_new(path.with_suffix('.state.json'),state)
    return {'sha256':C.sha(path),'state_sha256':C.sha(path.with_suffix('.state.json')),'shape':list(a.shape),'dtype':a.dtype.str,'active_seconds':.1}

def synthetic_review(job,result,analysis_root,analysis_path):
    """Exercise the real receipt formatter; recomputation correctness has separate tests."""
    from contextlib import ExitStack
    target=analysis_root/job['id'];target.mkdir(parents=True)
    C.write_new(target/'mechanical.json',{'state':'COMPLETE','analysis_plan':C.entry(analysis_path),'analysis_commit':'f'*40,
        'job':job['id'],'actor':'synthetic-mechanical','producer_plan':job['producer_plan'],'producer_commit':job['producer_commit'],
        'model_spec_sha256':C.digest(job['model_spec']),'raw_artifacts':{},'reports':{k:C.entry(p) for k,p in C.report_paths(job).items()},'reading_started':'s'*64})
    class Session:
        start='synthetic-reading-start'
        def __enter__(self):return self
        def __exit__(self,*args):return False
    name='mcq' if job['family']=='core-mcq' else ('mcmr' if job['family'].endswith('mcmr') else 'embedding')
    with ExitStack() as stack:
        stack.enter_context(patch.object(C,'OUTPUT_ROOT',analysis_root));stack.enter_context(patch.object(L,'ReadSession',return_value=Session()))
        stack.enter_context(patch.object(L,'require_completed'));stack.enter_context(patch.object(I,name,return_value=result))
        return R.independent(analysis_path,{'independent_environment':C.runtime_snapshot()},'f'*40,job,'synthetic-independent',True)


class Analysis(unittest.TestCase):
    def setUp(self):
        # Synthetic reading tests must never create or reset the real runtime anchor.
        self.anchor_temp=tempfile.TemporaryDirectory(dir=C.ROOT/'hf_home/tmp',prefix='analysis-test-anchor-')
        self.addCleanup(self.anchor_temp.cleanup)
        self.anchor_patch=patch.object(L,'ANCHOR_ROOT',Path(self.anchor_temp.name)/'anchor')
        self.anchor_patch.start();self.addCleanup(self.anchor_patch.stop)

    def test_formatted_embedding_receipt_really_passes_existing_panel_make_and_admit_and_mcmr_scope_cannot(self):
        from codex.retrieval_matrix_panel import qualification as Q
        pp=C.ROOT/'codex/retrieval_matrix_core_runtime_fix/p0-qwen3-fixed-v1.json';producer=C.read(pp);model=producer['models']['core2']
        with tempfile.TemporaryDirectory(dir=C.ROOT/'hf_home/tmp',prefix='analysis-panel-bridge-') as d:
            root=Path(d);ap=root/'analysis-plan.json';C.write_new(ap,{'synthetic':True})
            job={'id':'synthetic-panel','family':'core-fixed-embedding','producer_plan':C.entry(pp),'producer_commit':C.FIX_COMMIT,
                'model':'core2','model_spec':model,'runs':{},'implementation_authors':['producer-impl'],'analysis_authors':['analysis-impl'],'producer_actor':'synthetic-worker'}
            targets={'coco':'coco_t2i_recall_5','flickr':'flickr_t2i_recall_5','negbench':'negbench_coco_recall_5'}
            for name,field in targets.items():
                out=root/name;job['runs'][name]=str(out);ds=producer['datasets'][name]
                official={'R@5':model['published_targets'][field]} if name=='negbench' else {'t2i':{'recall_5':model['published_targets'][field]}}
                row={'model':'core2','dataset':name,'queries':ds['expected_eligible'],'raw_queries':ds['expected_queries'],'gallery':ds['expected_gallery'],
                    'excluded':[],'official':official,'provenance':{'index':{'identity':{'model':'core2','dataset':name,'model_spec':model,'settings':producer['settings'],'plan_sha256':C.sha(pp)}}}}
                C.write_new(out/'results/report.json',row)
            result={'agreement':True,'reports':{k:C.sha(p) for k,p in C.report_paths(job).items()},'recomputed':{'synthetic':True}}
            value=synthetic_review(job,result,root/'analysis',ap);review_path=root/'analysis'/job['id']/'independent-review.json'
            receipt=Q.make(pp,'core2',C.report_paths(job),review_path);qp=root/'qualification.json';C.write_new(qp,receipt)
            self.assertEqual(Q.admit(C.entry(qp),model)['state'],'PASS');self.assertEqual(value['user_final_decision'],'PENDING')
            changed=deepcopy(model);changed['batches']['image']+=1
            with self.assertRaises(ValueError):Q.admit(C.entry(qp),changed)
            mcmr=deepcopy(value);mcmr['scope']='official-native-mcmr-difference-only'
            with self.assertRaisesRegex(ValueError,'separate reviewer'):Q.assess(producer,C.sha(pp),'core2',{k:C.read(p) for k,p in C.report_paths(job).items()},mcmr)

    def test_formatted_mcq_receipt_really_passes_frozen_P3_reranker_gate(self):
        from codex.retrieval_matrix_rerank_p3 import contracts as P3
        pp=C.ROOT/'codex/retrieval_matrix_rerank/plan.core-mcq.v1.json';producer=C.read(pp);model=producer['models']['core2']
        with tempfile.TemporaryDirectory(dir=C.ROOT/'hf_home/tmp',prefix='analysis-P3-bridge-') as d:
            root=Path(d);out=root/'producer/main';out.mkdir(parents=True);ap=root/'analysis-plan.json';C.write_new(ap,{'synthetic':True})
            identity={'plan_sha256':C.sha(pp),'freeze_commit':C.MCQ_COMMIT,'model':'core2','model_spec':model,'source_files':producer['source_files']}
            complete={'state':'COMPLETE','identity':identity,'subsets':{}};rows=[];scores=[]
            for subset,count in C.COUNTS.items():
                task='cola_text_choice' if subset=='cola' else ('negbench_mcq' if subset.endswith('_mcq') else 'scpp');n=3 if subset in C.SCPP else 4
                hits=round(producer['acceptance']['core2'][task]*count);a=np.full((count,n),.5,dtype=np.float32)
                a[:hits]=([1,0,0,1] if subset=='cola' else ([1,1,0] if n==3 else [1,0,0,0]))
                items=[]
                for i,values in enumerate(a):
                    pairs=[{'id':f'synthetic:{subset}:{i}:{j}'} for j in range(n)];items.extend(pairs)
                    rows.append({'id':f'synthetic:{subset}:{i}','subset':subset,'task':task,'pairs':pairs})
                    scores.extend({'id':p['id'],'score':float(v)} for p,v in zip(pairs,values))
                receipt=save_checkpoint(out/(subset+'.npy'),a.reshape(-1,1),items,4,identity)
                complete['subsets'][subset]={'pairs':len(items),'array_sha256':receipt['sha256'],'state_sha256':receipt['state_sha256']}
            report,_=original_metrics().evaluate(rows,scores);report['reproduction']=I.numeric_mcq(report,producer,'core2')
            C.write_new(out/'complete.json',complete);C.write_new(out/'results/report.json',report)
            job={'id':'synthetic-MCQ','family':'core-mcq','producer_plan':C.entry(pp),'producer_commit':C.MCQ_COMMIT,
                'model':'core2','model_spec':model,'runs':{'mcq':str(out)},'implementation_authors':['producer-impl'],'analysis_authors':['analysis-impl'],'producer_actor':'synthetic-worker'}
            result={'agreement':True,'recomputed':report,'complete_sha256':C.sha(out/'complete.json'),'report_sha256':C.sha(out/'results/report.json')}
            synthetic_review(job,result,root/'analysis',ap)
            proof={'plan':C.entry(pp),'model':'core2','complete':C.entry(out/'complete.json'),'report':C.entry(out/'results/report.json'),
                'review':C.entry(root/'analysis'/job['id']/'independent-review.json')}
            qualified=P3.qualify_reranker(model,proof);self.assertEqual(qualified['state'],'PASS');self.assertEqual(qualified['matched'],3)
            changed=deepcopy(model);changed['revision']='a'*40
            with self.assertRaisesRegex(ValueError,'identity'):P3.qualify_reranker(changed,proof)

    def test_three_native_embedding_adapters_complete_checkpoint_and_original_metric_wiring(self):
        from codex.retrieval_matrix import runner as original
        with tempfile.TemporaryDirectory(dir=C.ROOT/'hf_home/tmp',prefix='analysis-native-') as d:
            root=Path(d);model={'batches':{'image':2,'text':2},'checkpoint':'synthetic'}
            plan={'settings':{'synthetic':True},'source_files':{},'datasets':{}}
            job={'family':'core-fixed-embedding','producer_plan':{'sha256':'a'*64},'producer_commit':'b'*40,'model':'synthetic','model_spec':model,'runs':{}}
            gallery=[{'id':str(i),'path':str(root/f'image{i}.jpg')} for i in range(6)]
            queries=[{'id':f'q{i}','text':'synthetic' if i!=5 else '','group':str(i)} for i in range(6)]
            bundle={'gallery':gallery,'queries':queries,'fingerprint':'synthetic-native','raw_queries':6,'excluded':[],'retained_empty':['q5']}
            labels=[{'id':f'q{i}','positives':[str(i)]} for i in range(6)]
            for name in ('coco','negbench','flickr'):
                out=root/name;job['runs'][name]=str(out);out.mkdir();ds={'official_evaluation':'negbench-t2i' if name=='negbench' else 'coco-bidirectional','similarity_batch':2}
                plan['datasets'][name]=ds;C.write_new(out/'processor.json',{'synthetic':True});ps=C.sha(out/'processor.json')
                identity={'model':'synthetic','dataset':name,'plan_sha256':'a'*64,'freeze_commit':'b'*40,'model_spec':model,'settings':plan['settings'],'source_files':{},'dataset_fingerprint':bundle['fingerprint']}
                C.write_new(out/'provenance.json',identity);meta={'state':'COMPLETE','identity':identity,'processor_sha256':ps,'arrays':{},'official_arrays':{}}
                for modality,items in [('image',[r['path'] for r in gallery]),('text',[r['text'] for r in queries])]:
                    a=np.tile(np.array([[1,0]],dtype=np.float32),(len(items),1))
                    meta['arrays'][modality]=save_checkpoint(out/(modality+'.npy'),a,items,2,{**identity,'stage':modality,'processor_sha256':ps})
                base={**identity,'similarity_dtype':'torch.bfloat16','similarity_batch':2,'score_semantics':'released normalized native-dtype embeddings and native-dtype GPU matmul; torch.topk on original CSV order'}
                scores=np.ones((6,6),dtype=np.float32);k=5 if name=='negbench' else 6;ix=np.tile(np.arange(k,dtype=np.int32),(6,1))
                meta['official_arrays']['official-similarity']=save_checkpoint(out/'official-similarity.npy',scores,list(range(6)),2,{**base,'stage':'official-similarity'})
                meta['official_arrays']['official-topk']=save_checkpoint(out/'official-topk.npy',ix,list(range(6)),2,{**base,'stage':'official-topk','k':k})
                if name!='negbench':
                    a=np.tile(np.concatenate((np.arange(6,dtype=np.float32),np.ones(6,dtype=np.float32)))[None,:],(6,1))
                    meta['official_arrays']['official-i2t']=save_checkpoint(out/'official-i2t.npy',a,list(range(6)),2,{**base,'stage':'official-i2t','k':6})
                C.write_new(out/'index-meta.json',meta)
                script="from pathlib import Path; import json,sys; from codex.retrieval_matrix.runner import official_metrics; v=json.load(sys.stdin); print(json.dumps(official_metrics(Path(v['out']),v['ds'],v['bundle'],v['labels'])))"
                env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':str(C.ROOT/'dl/retrieval-matrix/deps/core-p0')+':'+str(C.ROOT)}
                result=json.loads(subprocess.check_output([str(C.ROOT/'dl/retrieval3090/envs/retrieval3090-torch214/bin/python'),'-B','-c',script],input=json.dumps({'out':str(out),'ds':ds,'bundle':bundle,'labels':labels}),text=True,env=env))
                report={'model':'synthetic','dataset':name,'queries':6,'raw_queries':6,'gallery':6,'excluded':[],'retained_empty':['q5'],'official':result,'provenance':{'index_meta_sha256':C.sha(out/'index-meta.json'),'index':meta}}
                C.write_new(out/'results/report.json',report)
            with patch.object(C,'validate_job',return_value=plan),patch.object(original,'load_data',return_value=bundle),patch.object(original,'load_labels',return_value=labels):
                result=I.embedding(job);self.assertTrue(result['agreement']);self.assertEqual(set(result['datasets']),{'coco','negbench','flickr'})
                path=root/'coco/results/report.json';bad=C.read(path);bad['official']['t2i']['recall_5']=.999;path.write_text(json.dumps(bad))
                self.assertFalse(I.embedding(job)['agreement'])

    def test_complete3997_MCMR_native_adapter_keeps104981_product_ids_and_detects_metadata_change(self):
        import math
        from codex.retrieval_matrix_mcmr import runner as original
        with tempfile.TemporaryDirectory(dir=C.ROOT/'hf_home/tmp',prefix='analysis-mcmr-') as d:
            out=Path(d);gallery=[{'id':str(i),'path':'same-synthetic-image','text':f'title: distinct product {i}'} for i in range(104981)]
            queries=[{'id':str(i),'text':'synthetic query'} for i in range(3997)]
            bundle={'gallery':gallery,'queries':queries,'fingerprint':'synthetic-mcmr'};model={'synthetic':True}
            inst=out/'instances.jsonl'
            with inst.open('w') as f:
                for q in queries:f.write(json.dumps({'qid':q['id'],'pos':['7','9','104980']})+'\n')
            ds={'similarity_batch':1000,'instances':C.entry(inst)};plan={'source_files':{},'settings':{},'datasets':{'mcmr':ds}}
            job={'family':'core-fixed-mcmr','producer_plan':{'sha256':'a'*64},'producer_commit':'b'*40,'model':'synthetic','model_spec':model,'runs':{'mcmr':str(out)}}
            identity={'model':'synthetic','dataset':'mcmr','plan_sha256':'a'*64,'freeze_commit':'b'*40,'model_spec':model,'settings':{},'source_files':{},'dataset_fingerprint':bundle['fingerprint']}
            C.write_new(out/'provenance.json',identity);C.write_new(out/'processor.json',{'synthetic':True});ps=C.sha(out/'processor.json')
            meta={'state':'COMPLETE','identity':identity,'processor_sha256':ps,'arrays':{}}
            for name,items,batch,stage in [('corpus',gallery,64,'fused-product'),('queries',queries,32,'native-query')]:
                a=np.tile(np.array([[1,0]],dtype=np.float32),(len(items),1))
                meta['arrays'][name]=save_checkpoint(out/(name+'.npy'),a,items,batch,{**identity,'stage':stage,'processor_sha256':ps})
            native=np.tile(np.concatenate((np.arange(100,dtype=np.float32),np.arange(128,28,-1,dtype=np.float32)/128))[None,:],(3997,1))
            meta['arrays']['native-top100']=save_checkpoint(out/'native-top100.npy',native,queries,1000,{**identity,'stage':'native-bf16-top100','top_k':100,'torch_topk':True,'similarity_batch':1000})
            C.write_new(out/'index-meta.json',meta)
            primary={'recall_1':0.,'recall_5':0.,'recall_10':2/3,'recall_50':2/3,'recall_100':2/3,'ndcg_cut_5':0.,
                'ndcg_cut_10':(1/math.log2(9)+1/math.log2(11))/sum(1/math.log2(i+2) for i in range(3)),
                'map_cut_10':(1/8+2/10)/3,'MRR@10':1/8}
            C.write_new(out/'results/report.json',{'strict_CORE_P0_match_claim':False,'queries':3997,'gallery_products':104981,'model':'synthetic','provenance':meta,
                'primary_pytrec':primary,'native_torch_ordinal_diagnostic':{'Hit@1':0.,'Recall@1':0.,'MRR@10':1/8}})
            with patch.object(C,'validate_job',return_value=plan),patch.object(original,'load_data',return_value=bundle):
                result=I.mcmr(job);self.assertTrue(result['agreement']);self.assertFalse(result['eligible_for_P0_qualification'])
                gallery[0]['text']='a changed product sharing the same image'
                with self.assertRaisesRegex(ValueError,'checkpoint identity'):I.mcmr(job)
    def test_ties_and_full_positive_AP_denominator(self):
        actual=I.ir_one({str(i) for i in range(20)},{str(i):float(100-i) for i in range(30)})
        self.assertEqual(actual['map_cut_10'],.5)
        self.assertEqual(actual['recall_5'],.25)
        self.assertEqual(actual['ndcg_cut_10'],1.)
        # docID '9' precedes '10' at equal score, even if input insertion is opposite.
        self.assertEqual(I.ir_one({'10'},{'10':.5,'9':.5})['recall_1'],0.)
        self.assertEqual(I.ir_one({'9'},{'10':.5,'9':.5})['recall_1'],1.)

    def test_custom_IR_matches_actual_frozen_pytrec_on_many_ties_and_more_than10_positives(self):
        rng=np.random.default_rng(20261116);qrels={};pred={}
        for q in range(40):
            ids=[str(i) for i in range(150)];rel=rng.choice(ids,size=1+q%37,replace=False)
            qrels[str(q)]={str(i):1 for i in rel}
            chosen=rng.choice(ids,size=100,replace=False);values=rng.integers(-2,5,size=100)/4
            pred[str(q)]={str(i):float(v) for i,v in zip(chosen,values)}
        fields=(*I.IR_FIELDS,'recall_50','recall_100','recip_rank')
        payload=json.dumps({'qrels':qrels,'predictions':pred,'fields':fields})
        script="import json,sys,pytrec_eval; v=json.load(sys.stdin); print(json.dumps(pytrec_eval.RelevanceEvaluator(v['qrels'],set(v['fields'])).evaluate(v['predictions'])))"
        env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':str(C.ROOT/'dl/retrieval-matrix/deps/core-p0')}
        actual=json.loads(subprocess.check_output([str(C.ROOT/'dl/retrieval3090/envs/retrieval3090-torch214/bin/python'),'-B','-c',script],input=payload,text=True,env=env))
        for q in qrels:
            expected=I.ir_one(qrels[q],pred[q],fields)
            self.assertTrue(I.same_numbers(actual[q],expected),q)

    def test_native_first10_admission_precedes_mrr_tie_sort(self):
        # A relevant item outside the native first10 cannot enter MRR by docid tie sort.
        all100={str(i):.5 for i in range(100)};first10={str(i):.5 for i in range(10)}
        self.assertGreater(I.ir_one({'99'},all100,('recip_rank',))['recip_rank'],0)
        self.assertEqual(I.ir_one({'99'},first10,('recip_rank',))['recip_rank'],0)

    def test_full15912_MCQs_match_original_without_importing_producer_evaluation(self):
        rows,scores,arrays=synthetic_mcq();original,_=original_metrics().evaluate(rows,scores)
        independent=I.independent_mcq(rows,arrays)
        self.assertTrue(I.same_numbers(original,independent));self.assertEqual(sum(a.size for a in arrays.values()),58891)
        self.assertEqual(set(independent['benchmarks']),set(C.TASKS))
        arrays['cola'][:]=.5
        self.assertEqual(I.independent_mcq(rows,arrays)['subsets']['cola']['accuracy'],0)

    def test_mcq_incomplete_macro_nan_or_changed_denominator_rejected(self):
        rows,_,arrays=synthetic_mcq();report=I.independent_mcq(rows,arrays)
        producer={'acceptance':{'core2':{t:.5 for t in C.TASKS}}};I.numeric_mcq(report,producer,'core2')
        for mutate in (lambda p:p['benchmarks'].pop('scpp'),lambda p:p['subsets']['cola'].update(rows=209),
                lambda p:p['benchmarks']['scpp'].update(macro_accuracy=float('nan')),
                lambda p:p['benchmarks']['scpp'].update(macro_accuracy=.999)):
            bad=deepcopy(report);mutate(bad)
            with self.assertRaises(ValueError):I.numeric_mcq(bad,producer,'core2')

    def test_wrong_actor_and_analysis_implementer_rejected(self):
        j={'implementation_authors':['producer-impl'],'analysis_authors':['analysis-impl'],'producer_actor':'worker'}
        for actor in ('producer-impl','analysis-impl','worker','mechanical',''):
            with self.assertRaises(ValueError):C.require_reviewer(j,actor,'mechanical')
        C.require_reviewer(j,'separate-reviewer','mechanical')

    def test_actual_independent_numpy_binary_import_is_frozen(self):
        plan={'independent_environment':C.runtime_snapshot()};C.independent_environment(plan)
        plan['independent_environment']['numpy']['array_binary']['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'array runtime differs'):C.independent_environment(plan)

    def test_analysis_source_gate_accepts_only_valid_append_and_rejects_changed_body(self):
        with tempfile.TemporaryDirectory(dir=C.ROOT/'hf_home/tmp',prefix='analysis-git-') as d:
            root=Path(d);pp=root/'plan.json';prereg=root/'PREREG.md';header=b'fixed numerical criteria\n'+C.MARKER;prereg.write_bytes(header)
            plan={'state':'FROZEN','reviewed':True,'source_files':{str(prereg.relative_to(C.ROOT)):hashlib.sha256(header).hexdigest()}}
            C.write_new(pp,plan);original={pp:pp.read_bytes(),prereg:header}
            def git(commit,p):return original[Path(p)]
            with patch.object(C,'HERE',root),patch.object(C,'validate_plan'),patch.object(C,'git_bytes',side_effect=git),patch.object(subprocess,'check_output',return_value=''):
                C.source_gate(pp,plan,'a'*40)
                args={'path':prereg,'plan_sha':C.sha(pp),'commit':'a'*40,'job':{'id':'s','producer_plan':{'sha256':'b'*64}},'actor':'reviewer','token':'s','paths':{}}
                L.append(**args,event='READ_STARTED');L.append(**args,event='READ_COMPLETED');C.source_gate(pp,plan,'a'*40)
                full=prereg.read_bytes();tail=C.prereg_body(prereg)[1].splitlines(keepends=True)
                for rollback in (header,header+tail[0]):
                    prereg.write_bytes(rollback)
                    with self.assertRaisesRegex(ValueError,'durable external anchor'):C.source_gate(pp,plan,'a'*40)
                prereg.write_bytes(full)
                prereg.write_bytes(prereg.read_bytes().replace(b'fixed numerical criteria',b'changed numerical criteria'))
                with self.assertRaisesRegex(ValueError,'body differs'):C.source_gate(pp,plan,'a'*40)

    def test_external_anchor_rejects_tail_deletion_all_deletion_and_earlier_log_or_anchor_rollback(self):
        with tempfile.TemporaryDirectory(dir=C.ROOT/'hf_home/tmp',prefix='analysis-anchor-regression-') as d:
            p=Path(d)/'PREREG.md';header=b'fixed criteria\n'+C.MARKER;p.write_bytes(header)
            args={'path':p,'plan_sha':'a'*64,'commit':'b'*40,'job':{'id':'s','producer_plan':{'sha256':'c'*64}},'actor':'r','paths':{},'token':'first'}
            L.append(**args,event='READ_STARTED');L.append(**args,event='READ_COMPLETED');earlier=p.read_bytes()
            with L._anchor_lock(p) as journal:old_anchor=journal.read_bytes()
            later={**args,'token':'second'};L.append(**later,event='READ_STARTED')
            L.append(**later,event='READ_FAILED',error_type='SyntheticFailure');current=p.read_bytes();current_anchor=journal.read_bytes()
            lines=C.prereg_body(p)[1].splitlines(keepends=True)
            for rollback in (header+b''.join(lines[:-1]),header,earlier):
                p.write_bytes(rollback)
                with self.assertRaisesRegex(ValueError,'durable external anchor'):L.records(p)
                with self.assertRaisesRegex(ValueError,'durable external anchor'):L.append(**{**args,'token':'third'},event='READ_STARTED')
            p.write_bytes(current);journal.write_bytes(old_anchor)
            with self.assertRaisesRegex(ValueError,'durable external anchor'):L.records(p)
            journal.write_bytes(current_anchor)
            self.assertEqual(L.records(p)[-1]['event'],'READ_FAILED')

    def test_pending_anchor_crashes_before_or_after_log_append_fail_closed_without_artifact_read(self):
        for crash in ('before-log','before-commit'):
            with tempfile.TemporaryDirectory(dir=C.ROOT/'hf_home/tmp',prefix='analysis-anchor-crash-') as d:
                root=Path(d);p=root/'PREREG.md';p.write_bytes(b'fixed\n'+C.MARKER)
                plan=root/'plan.json';C.write_new(plan,{'synthetic':True});artifact=root/'synthetic-score.npy';artifact.write_bytes(b'untouched')
                original=L._append_anchor;entered=[]
                def fail_commit(path,value,**kwargs):
                    if value['kind']=='COMMIT':raise OSError('synthetic crash before commit')
                    return original(path,value,**kwargs)
                hook=patch.object(L,'_append_prereg',side_effect=OSError('synthetic crash before log')) if crash=='before-log' else patch.object(L,'_append_anchor',side_effect=fail_commit)
                with patch.object(C,'HERE',root),hook:
                    with self.assertRaises(OSError):
                        with L.ReadSession(plan,'a'*40,{'id':'s','producer_plan':{'sha256':'b'*64}},'reader',[artifact]):
                            entered.append(True);artifact.read_bytes()
                self.assertEqual(entered,[])
                with L._anchor_lock(p) as journal:before=journal.read_bytes()
                with self.assertRaisesRegex(ValueError,'Incomplete reading-anchor transaction'):L.records(p)
                with self.assertRaisesRegex(ValueError,'Incomplete reading-anchor transaction'):
                    L.append(p,plan_sha=C.sha(plan),commit='a'*40,job={'id':'s','producer_plan':{'sha256':'b'*64}},actor='reader',event='READ_STARTED',token='new',paths={})
                self.assertEqual(journal.read_bytes(),before)
                self.assertEqual(len(C.prereg_body(p)[1].splitlines()),0 if crash=='before-log' else 1)

    def test_anchor_write_order_directory_fsync_and_plan_commit_body_identity_are_preserved(self):
        with tempfile.TemporaryDirectory(dir=C.ROOT/'hf_home/tmp',prefix='analysis-anchor-order-') as d:
            p=Path(d)/'PREREG.md';p.write_bytes(b'fixed\n'+C.MARKER);order=[];dirs=[]
            anchor=L._append_anchor;prereg=L._append_prereg;sync=L._fsync_directory
            def anchored(path,value,**kwargs):
                result=anchor(path,value,**kwargs);order.append(value['kind']);return result
            def logged(path,value):
                result=prereg(path,value);order.append('PREREG');return result
            def synced(path):dirs.append(path);return sync(path)
            with patch.object(L,'_append_anchor',side_effect=anchored),patch.object(L,'_append_prereg',side_effect=logged),patch.object(L,'_fsync_directory',side_effect=synced):
                L.append(p,plan_sha='a'*64,commit='b'*40,job={'id':'s','producer_plan':{'sha256':'c'*64}},actor='r',event='READ_STARTED',token='s',paths={})
            self.assertEqual(order,['GENESIS','PREPARE','PREREG','COMMIT'])
            with L._anchor_lock(p) as journal:self.assertIn(journal.parent,dirs)
            with self.assertRaises(ValueError):L.records(p,expected_plan='f'*64,expected_commit='b'*40)
            with self.assertRaises(ValueError):L.records(p,expected_plan='a'*64,expected_commit='f'*40)
            p.write_bytes(p.read_bytes().replace(b'fixed\n',b'changed\n'))
            with self.assertRaisesRegex(ValueError,'different preregistration'):L.records(p)

    def test_missing_external_anchor_can_only_initialize_a_genuinely_empty_log(self):
        with tempfile.TemporaryDirectory(dir=C.ROOT/'hf_home/tmp',prefix='analysis-anchor-missing-') as d:
            p=Path(d)/'PREREG.md';p.write_bytes(b'fixed\n'+C.MARKER);anchor=Path(d)/'explicit-synthetic-anchor'
            self.assertEqual(L.records(p,anchor_root=anchor),[])
            args={'path':p,'plan_sha':'a'*64,'commit':'b'*40,'job':{'id':'s','producer_plan':{'sha256':'c'*64}},'actor':'r','event':'READ_STARTED','token':'s','paths':{},'anchor_root':anchor}
            L.append(**args)
            with L._anchor_lock(p,anchor) as journal:journal.rename(journal.with_suffix('.preserved-original'))
            with self.assertRaisesRegex(ValueError,'without its external anchor'):L.records(p,anchor_root=anchor)
            with self.assertRaisesRegex(ValueError,'without its external anchor'):L.append(**{**args,'token':'retry'})

    def test_all_original_benchmarks_must_complete_before_mechanical(self):
        with tempfile.TemporaryDirectory(dir=C.ROOT/'hf_home/tmp',prefix='analysis-complete-') as d:
            j={'family':'core-fixed-embedding','runs':{n:str(Path(d)/n) for n in ('coco','flickr','negbench')}}
            for n in ('coco','flickr'):C.write_new(Path(d)/n/'index-meta.json',{'state':'COMPLETE'})
            with self.assertRaisesRegex(ValueError,'All original benchmark'):C.require_complete(j)

    def test_checkpoint_fingerprint_and_each_original_batch_are_recomputed(self):
        with tempfile.TemporaryDirectory(dir=C.ROOT/'hf_home/tmp',prefix='analysis-array-') as d:
            path=Path(d)/'synthetic.npy';a=np.arange(15,dtype=np.float32).reshape(5,3);np.save(path,a)
            items=['a','b','c','d','e'];identity={'full-model':'unchanged'}
            state={'fingerprint':C.digest({'identity':identity,'items':items,'batch_size':4}),
                'completed':5,'array_shape':[5,3],'dtype':a.dtype.str,'sha256':C.sha(path),'active_seconds':.1,
                'batches':[{'start':s,'end':min(s+4,5),'sha256':hashlib.sha256(a[s:min(s+4,5)].tobytes()).hexdigest()} for s in (0,4)]}
            sp=path.with_suffix('.state.json');C.write_new(sp,state);I.checkpoint(path,items,4,identity)
            with self.assertRaisesRegex(ValueError,'identity'):I.checkpoint(path,items[::-1],4,identity)
            state['batches'][0]['sha256']='0'*64;sp.write_text(json.dumps(state))
            with self.assertRaisesRegex(ValueError,'batch boundary'):I.checkpoint(path,items,4,identity)

    def test_read_start_is_durable_before_artifact_open_and_failed_reads_remain(self):
        with tempfile.TemporaryDirectory(dir=C.ROOT/'hf_home/tmp',prefix='analysis-read-') as d:
            prereg=Path(d)/'PREREG.md';prereg.write_bytes(b'frozen\n'+C.MARKER)
            plan=Path(d)/'plan.json';C.write_new(plan,{'synthetic':True});artifact=Path(d)/'synthetic-score.npy';artifact.write_bytes(b'synthetic')
            job={'id':'synthetic','producer_plan':{'sha256':'a'*64}}
            with patch.object(C,'HERE',Path(d)):
                with self.assertRaisesRegex(RuntimeError,'synthetic crash'):
                    with L.ReadSession(plan,'b'*40,job,'reader',[artifact]):
                        events=L.records(prereg);self.assertEqual(events[-1]['event'],'READ_STARTED')
                        self.assertIsNone(events[-1]['paths'][str(artifact)])
                        artifact.read_bytes();raise RuntimeError('synthetic crash')
            events=L.records(prereg);self.assertEqual([r['event'] for r in events],['READ_STARTED','READ_FAILED'])
            self.assertEqual(events[-1]['paths'][str(artifact)],C.sha(artifact))

    def test_hash_io_failure_is_recorded_and_cannot_complete_mechanical_read(self):
        with tempfile.TemporaryDirectory(dir=C.ROOT/'hf_home/tmp',prefix='analysis-hash-failure-') as d:
            root=Path(d);prereg=root/'PREREG.md';prereg.write_bytes(b'frozen\n'+C.MARKER)
            plan=root/'plan.json';C.write_new(plan,{'synthetic':True});artifact=root/'synthetic-score.npy';artifact.write_bytes(b'synthetic')
            job={'id':'s','producer_plan':{'sha256':'a'*64}};original=C.sha
            def fail_score(p):
                if Path(p)==artifact:raise PermissionError('synthetic permission failure')
                return original(p)
            with patch.object(C,'HERE',root),patch.object(C,'sha',side_effect=fail_score):
                with self.assertRaises(PermissionError):
                    with L.ReadSession(plan,'b'*40,job,'reader',[artifact]) as session:pass
            events=L.records(prereg);self.assertEqual(events[-1]['event'],'READ_FAILED')
            self.assertEqual(events[-1]['hash_errors'],{str(artifact):'PermissionError'})
            with self.assertRaisesRegex(ValueError,'successfully completed'):L.require_completed(prereg,session.start,plan_sha=C.sha(plan),commit='b'*40)

    def test_read_log_tamper_nonrecord_or_wrong_actor_completion_rejected(self):
        with tempfile.TemporaryDirectory(dir=C.ROOT/'hf_home/tmp',prefix='analysis-log-') as d:
            p=Path(d)/'PREREG.md';header=b'criteria\n'+C.MARKER;p.write_bytes(header)
            args={'path':p,'plan_sha':'a'*64,'commit':'b'*40,'job':{'id':'x','producer_plan':{'sha256':'c'*64}},'actor':'reader','token':'token','paths':{str(Path(d)/'score'):None}}
            L.append(**args,event='READ_STARTED')
            original=p.read_bytes()
            with self.assertRaisesRegex(ValueError,'different actor'):L.append(**{**args,'actor':'other'},event='READ_COMPLETED')
            p.write_bytes(original+b'arbitrary outcome prose\n')
            with self.assertRaises(ValueError):L.records(p)
            p.write_bytes(original);row=L.records(p)[0];row['previous']='f'*64
            p.write_bytes(header+L.PREFIX+json.dumps(row).encode()+b'\n')
            with self.assertRaisesRegex(ValueError,'append chain'):L.records(p)

    def test_commands_only_original_cpu_analyze_and_hide_all_GPUs(self):
        with patch.object(C,'checked',side_effect=lambda p:Path(p['path'])):
            for family in ('core-mcq','core-fixed-embedding','qwen2-fixed-embedding','core-fixed-mcmr','qwen2-fixed-mcmr'):
                name='mcq' if family=='core-mcq' else ('mcmr' if family.endswith('mcmr') else 'coco')
                j={'family':family,'producer_plan':{'path':str(C.HERE/'synthetic.json')},'producer_commit':'a'*40,'model':'synthetic','runs':{name:str(C.ROOT/'dl/synthetic/main')}}
                cmd=C.original_command(j,name);self.assertEqual(cmd[cmd.index('--stage')+1],'analyze')
                self.assertNotIn('index',cmd);self.assertEqual(C.cpu_environment(j)['CUDA_VISIBLE_DEVICES'],'')
                self.assertFalse(any(x in cmd for x in ('--smoke-out','--index-out')))

    def test_existing_original_report_is_not_overwritten(self):
        with tempfile.TemporaryDirectory(dir=C.ROOT/'hf_home/tmp',prefix='analysis-existing-') as d:
            root=Path(d);job={'id':'synthetic','family':'core-mcq','runs':{'mcq':str(root/'producer')},'producer_plan':{'sha256':'a'*64}}
            C.write_new(root/'producer/complete.json',{'state':'COMPLETE'});C.write_new(root/'producer/results/report.json',{'synthetic':True})
            class Session:
                def __enter__(self):return self
                def __exit__(self,*a):return False
            with patch.object(C,'OUTPUT_ROOT',root/'analysis'),patch.object(L,'ReadSession',return_value=Session()),patch.object(subprocess,'run') as run:
                with self.assertRaisesRegex(ValueError,'already exists'):R.mechanical(root/'p',{},'b'*40,job,'actor')
                run.assert_not_called()

    def test_real_builder_reads_sources_and_paths_only_and_retains14_jobs(self):
        from codex.retrieval_matrix_analysis.build_plan import build
        original=C.read
        def read_sources(p):
            self.assertTrue(Path(p).resolve().is_relative_to(C.ROOT/'codex'),'Builder opened a runtime artifact')
            return original(p)
        with patch.object(C,'read',side_effect=read_sources):plan=build()
        self.assertEqual(len(plan['jobs']),14);self.assertEqual(plan['state'],'REVIEW_REQUIRED')
        self.assertFalse(plan['production_and_effect_reads_performed'])
        self.assertEqual(plan['jobs']['mcq-core2']['runs']['mcq'].rsplit('/',1)[-1],'main')

    def test_schedule_mapping_and_real_backend_runtime_authors_cannot_be_changed(self):
        from codex.retrieval_matrix_analysis.build_plan import build
        plan=build();key='fixed-embedding-gme2'
        for mutate in (lambda p:p['jobs'][key].update(producer_commit=C.MCQ_COMMIT),
                lambda p:p['jobs'][key]['runs'].update(coco=str(C.ROOT/'dl/another-run/coco')),
                lambda p:p['jobs'][key]['implementation_authors'].remove('/root/official_p0/p3_contract_review'),
                lambda p:p['jobs']['fixed-mcmr-core2']['implementation_authors'].remove('/root/matrix_runner')):
            bad=deepcopy(plan);mutate(bad)
            with self.assertRaises(ValueError):C.validate_plan(bad)

    def test_draft_or_changed_analysis_criteria_cannot_execute(self):
        from codex.retrieval_matrix_analysis.build_plan import build
        plan=build()
        with self.assertRaisesRegex(ValueError,'reviewed frozen'):C.source_gate(C.HERE/'synthetic.json',plan,'a'*40)
        plan['criteria']['required_distinct_benchmarks']=1
        with self.assertRaisesRegex(ValueError,'numerical criteria'):C.validate_plan(plan)


if __name__=='__main__':unittest.main()
