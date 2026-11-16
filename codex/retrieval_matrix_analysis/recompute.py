"""[codex] Independent binary metrics on original native arrays; CPU only."""
from __future__ import annotations
from collections import Counter
import hashlib,math
from pathlib import Path
import numpy as np
from codex.retrieval_matrix_analysis import contracts as C

IR_FIELDS=('map_cut_10','ndcg_cut_5','ndcg_cut_10','recall_1','recall_3','recall_5','recall_10')

def same_numbers(expected,actual,tolerance=1e-12):
    if isinstance(actual,dict):
        return isinstance(expected,dict) and all(k in expected and same_numbers(expected[k],v,tolerance) for k,v in actual.items())
    if isinstance(actual,(float,int)) and not isinstance(actual,bool):
        return isinstance(expected,(float,int)) and not isinstance(expected,bool) and math.isfinite(expected) and abs(expected-actual)<=tolerance
    return expected==actual

def checkpoint(path,items,batch,identity):
    """Recreate the original item/model fingerprint and every original batch SHA."""
    path=C.safe(path);state=C.read(path.with_suffix('.state.json'))
    values=np.load(path,mmap_mode='r',allow_pickle=False)
    if (state.get('fingerprint')!=C.digest({'identity':identity,'items':items,'batch_size':batch})
            or state.get('completed')!=len(items) or values.shape[0]!=len(items)
            or state.get('array_shape')!=list(values.shape) or state.get('dtype')!=values.dtype.str
            or state.get('sha256')!=C.sha(path) or not np.isfinite(values).all()):
        raise ValueError('Original checkpoint identity, coverage, dtype or payload differs')
    end=0
    for row in state['batches']:
        if (row['start']!=end or row['end']!=min(end+batch,len(items))
                or row['sha256']!=hashlib.sha256(np.ascontiguousarray(values[row['start']:row['end']]).tobytes()).hexdigest()):
            raise ValueError('Original batch boundary or raw byte checksum differs')
        end=row['end']
    if end!=len(items):raise ValueError('Original batches omit rows')
    receipt={'sha256':C.sha(path),'state_sha256':C.sha(path.with_suffix('.state.json')),
        'shape':list(values.shape),'dtype':values.dtype.str,'active_seconds':state['active_seconds']}
    return values,receipt

def ir_one(positives,predictions,fields=IR_FIELDS):
    """Binary trec_eval semantics: reverse lexical score ties, full positive denominator."""
    positives=set(map(str,positives))
    if not positives or not predictions or any(not math.isfinite(s) for s in predictions.values()):
        raise ValueError('Original query needs nonempty positive labels and finite predictions')
    # trec_eval comp_sim_docno: descending similarity, strcmp(p2.docno,p1.docno).
    ranked=sorted(predictions,key=lambda key:(predictions[key],key),reverse=True)
    hits=[int(key in positives) for key in ranked]
    result={}
    for field in fields:
        if field.startswith('recall_'):
            result[field]=sum(hits[:int(field.rsplit('_',1)[1])])/len(positives)
        elif field.startswith('ndcg_cut_'):
            k=int(field.rsplit('_',1)[1])
            dcg=sum(hit/math.log2(i+2) for i,hit in enumerate(hits[:k]))
            ideal=sum(1/math.log2(i+2) for i in range(min(k,len(positives))))
            result[field]=dcg/ideal
        elif field=='map_cut_10':
            count=0;total=0.
            for rank,hit in enumerate(hits[:10],1):
                count+=hit
                if hit:total+=count/rank
            # m_map_cut.c divides by ALL relevant documents, not min(10,num_rel).
            result[field]=total/len(positives)
        elif field=='recip_rank':result[field]=next((1/(i+1) for i,h in enumerate(hits) if h),0.)
        else:raise ValueError('Unknown native binary metric')
    return result

def ir_mean(qrels,predictions,fields=IR_FIELDS):
    if not qrels or set(qrels)!=set(predictions):raise ValueError('Keep the complete original query denominator')
    rows=[ir_one(qrels[q],predictions[q],fields) for q in qrels]
    return {f:math.fsum(r[f] for r in rows)/len(rows) for f in fields}

def independent_mcq(rows,by_subset):
    if Counter(r['subset'] for r in rows)!=C.COUNTS or set(by_subset)!=set(C.COUNTS):raise ValueError('All eight original subsets are required')
    subsets={}
    for name,count in C.COUNTS.items():
        n=4 if name=='cola' or name.endswith('_mcq') else 3
        a=np.asarray(by_subset[name]).reshape(count,n)
        if not np.isfinite(a).all():raise ValueError('Nonfinite MCQ scalar')
        if name=='cola':hits=(a[:,0]>a[:,1])&(a[:,3]>a[:,2])
        elif name in C.SCPP:hits=(a[:,0]>a[:,2])&(a[:,1]>a[:,2])
        else:hits=(a[:,0,None]>a[:,1:]).all(axis=1)
        subsets[name]={'rows':count,'accuracy':int(hits.sum())/count}
    return {'rows':15912,'subsets':subsets,'benchmarks':{task:{'subsets':sorted(names),
        'rows':sum(C.COUNTS[n] for n in names),'macro_accuracy':math.fsum(subsets[n]['accuracy'] for n in names)/len(names)} for task,names in C.TASKS.items()}}

def numeric_mcq(report,producer_plan,key):
    if set(report.get('benchmarks',{}))!=set(C.TASKS) or report.get('rows')!=15912 or set(report.get('subsets',{}))!=set(C.COUNTS):
        raise ValueError('All three named benchmarks and original denominators must remain visible')
    result={}
    for task,names in C.TASKS.items():
        row=report['benchmarks'][task];target=producer_plan['acceptance'][key][task]
        if (row.get('subsets')!=sorted(names) or row.get('rows')!=sum(C.COUNTS[n] for n in names)
                or any(report['subsets'][n].get('rows')!=C.COUNTS[n] for n in names)):
            raise ValueError('Published benchmark macro composition or denominator changed')
        x=row['macro_accuracy']
        parts=[report['subsets'][n]['accuracy'] for n in names]
        if (not math.isfinite(x) or not 0<=x<=1 or any(not math.isfinite(v) or not 0<=v<=1 for v in parts)
                or abs(x-math.fsum(parts)/len(parts))>1e-12):raise ValueError('MCQ accuracy and its declared subset macro must be finite fractions')
        result[task]={'published':target,'observed':x,'absolute_tolerance':.01,'passed':abs(x-target)<=.01}
    return result

def mcq(job):
    plan=C.validate_job(job);out=C.safe(job['runs']['mcq']);complete=C.read(out/'complete.json');identity=C.read(out/'identity.json')
    if (complete.get('state')!='COMPLETE' or complete.get('identity')!=identity
            or identity.get('plan_sha256')!=job['producer_plan']['sha256'] or identity.get('freeze_commit')!=job['producer_commit']
            or identity.get('model')!=job['model'] or identity.get('model_spec')!=job['model_spec'] or identity.get('source_files')!=plan['source_files']):
        raise ValueError('MCQ completion must bind the exact frozen model, source, environment and plan')
    for field,package in [('torch','torch'),('transformers','transformers'),('flash_attn','flash-attn')]:
        if identity.get('environment',{}).get(field)!=plan['environment']['packages'][package]:
            raise ValueError('Original MCQ actual runtime differs from its frozen package versions')
    # No model or producer metrics are imported.
    import json
    rows=[json.loads(line) for line in C.checked(plan['examples']).read_text().splitlines()]
    pairs=[p for r in rows for p in r['pairs']]
    if (len(pairs)!=58891 or len({p['id'] for p in pairs})!=58891 or len({r['id'] for r in rows})!=15912
            or [json.loads(line) for line in C.checked(plan['pairs']).read_text().splitlines()]!=pairs):
        raise ValueError('Every original MCQ example/pair must remain in its frozen order')
    for e in plan['source_data']+plan['external_sources']+[plan['preparation']]:C.checked(e)
    images=[json.loads(line) for line in C.checked(plan['images_receipt']).read_text().splitlines()]
    if len(images)!=9425 or len({r['path'] for r in images})!=9425 or {p['image'] for p in pairs}!={r['path'] for r in images}:
        raise ValueError('Complete original MCQ image mapping differs')
    for row in images:
        path=C.safe(row['path'])
        if C.sha(path)!=row['sha256'] or path.stat().st_size!=row['bytes']:raise ValueError('Original MCQ image payload changed')
    arrays={};receipts={}
    if set(complete.get('subsets',{}))!=set(C.COUNTS):raise ValueError('MCQ completion omits a subset')
    for subset in C.COUNTS:
        examples=[r for r in rows if r['subset']==subset];items=[p for r in examples for p in r['pairs']]
        task='cola_text_choice' if subset=='cola' else ('negbench_mcq' if subset.endswith('_mcq') else 'scpp')
        inst='Determine if the following image is relevant to the query.'+('' if subset=='cola' else ' ')
        if any(r['task']!=task or r['instruction']!=inst for r in examples):raise ValueError('Original comparison direction or instruction differs')
        a,receipt=checkpoint(out/(subset+'.npy'),items,4,{**identity,'subset':subset,'instruction':inst})
        expected=complete['subsets'][subset]
        if a.shape!=(len(items),1) or a.dtype!=np.float32 or expected!={'pairs':len(items),'array_sha256':receipt['sha256'],'state_sha256':receipt['state_sha256']}:
            raise ValueError('Original FP32 MCQ scalar array/state receipt differs')
        arrays[subset]=a;receipts[subset]=receipt
    actual=independent_mcq(rows,arrays);report=C.read(out/'results/report.json');targets=numeric_mcq(actual,plan,job['model'])
    agrees=same_numbers(report,actual) and same_numbers(report.get('reproduction'),targets)
    return {'agreement':agrees,'recomputed':actual,'numeric_targets':targets,'matched_distinct_benchmarks':sum(r['passed'] for r in targets.values()),
        'arrays':receipts,'complete_sha256':C.sha(out/'complete.json'),'report_sha256':C.sha(out/'results/report.json'),
        'original_identity_sha256':C.digest(identity),'complete_original_images_verified':9425}

def native_identity(job,plan,name,bundle,out):
    identity=C.read(out/'provenance.json');meta=C.read(out/'index-meta.json')
    expected={'model':job['model'],'dataset':name,'plan_sha256':job['producer_plan']['sha256'],'freeze_commit':job['producer_commit'],
        'model_spec':job['model_spec'],'settings':plan['settings'],'dataset_fingerprint':bundle['fingerprint']}
    if (meta.get('state')!='COMPLETE' or meta.get('identity')!=identity
            or any(identity.get(k)!=v for k,v in expected.items()) or not set(plan['source_files'])<=set(identity.get('source_files',{}))
            or any(identity['source_files'][k]!=v for k,v in plan['source_files'].items())
            or C.sha(out/'processor.json')!=meta.get('processor_sha256')):
        raise ValueError('Native original index provenance/model/source/dataset differs')
    environment=identity.get('environment',{})
    if any(environment.get(k)!=v for k,v in job['model_spec'].get('environment',{}).items()):
        raise ValueError('Native actual producing environment differs from the complete frozen model')
    if 'runtime_compatibility' in job['model_spec'] and environment.get('runtime_compatibility')!=job['model_spec']['runtime_compatibility']:
        raise ValueError('Native evidence omits its exact declared API repair')
    if job['family'].startswith('qwen2-fixed') and (environment.get('runtime_fix',{}).get('overlay_manifest')!=plan['runtime_fix']['overlay']
            or environment.get('runtime_fix',{}).get('parent_freeze_commit')!=plan['runtime_fix']['parent_freeze_commit']):
        raise ValueError('Native evidence omits its exact declared dependency repair')
    return identity,meta

def bound_native(path,items,batch,identity,expected):
    a,receipt=checkpoint(path,items,batch,identity)
    if expected!=receipt:raise ValueError('Original native array receipt differs')
    return a

def top_indices(a,count,k):
    if a.ndim!=2 or a.shape[1]!=k or not np.array_equal(a,a.astype(np.int64)) or (a<0).any() or (a>=count).any():
        raise ValueError('Native admitted indices are invalid')
    if any(len(set(row))!=k for row in a):raise ValueError('Native topk repeats an original candidate')
    return a.astype(np.int64)

def embedding(job):
    # Reuse only the frozen data adapter. No producing/evaluation model is constructed.
    from codex.retrieval_matrix import runner as R
    plan=C.validate_job(job);recomputed={};details={};reports={}
    for name in ('coco','negbench','flickr'):
        out=C.safe(job['runs'][name]);ds=plan['datasets'][name]
        bundle=R.load_data(ds,verify_images=True,allow_test=ds.get('split')=='test' and plan.get('fixed_baseline_test') is True)
        labels=R.load_labels(ds,bundle);identity,meta=native_identity(job,plan,name,bundle,out)
        for modality,items in [('image',[str(C.safe(r['path'])) for r in bundle['gallery']]),('text',[r['text'] for r in bundle['queries']])]:
            a=bound_native(out/(modality+'.npy'),items,job['model_spec']['batches'][modality],
                {**identity,'stage':modality,'processor_sha256':meta['processor_sha256']},meta['arrays'][modality])
            if a.dtype!=np.float32 or not np.allclose(np.linalg.norm(a,axis=1),1,atol=1e-2):raise ValueError('Original normalized native embedding differs')
        ids=[str(r['id']) for r in bundle['gallery']];n=len(labels);block=ds.get('similarity_batch',1000)
        base={**identity,'similarity_dtype':'torch.bfloat16','similarity_batch':block,
            'score_semantics':'released normalized native-dtype embeddings and native-dtype GPU matmul; torch.topk on original CSV order'}
        expected_names={'official-similarity','official-topk'}|({'official-i2t'} if name!='negbench' else set())
        if set(meta.get('official_arrays',{}))!=expected_names:raise ValueError('All authoritative native score arrays are required')
        scores=bound_native(out/'official-similarity.npy',list(range(n)),block,{**base,'stage':'official-similarity'},meta['official_arrays']['official-similarity'])
        k=5 if name=='negbench' else min(100,len(ids))
        ix=bound_native(out/'official-topk.npy',list(range(n)),block,{**base,'stage':'official-topk','k':k},meta['official_arrays']['official-topk'])
        ix=top_indices(ix,len(ids),k)
        if scores.shape!=(n,len(ids)) or scores.dtype!=np.float32:raise ValueError('Complete original native score matrix differs')
        for qi,indices in enumerate(ix):
            selected=scores[qi,indices]
            if np.any(np.diff(selected)>0) or np.count_nonzero(scores[qi]>selected[-1])!=np.count_nonzero(selected>selected[-1]):
                raise ValueError('Original native topk omitted a strictly higher score')
        if name=='negbench':
            result={'queries':n,'R@5':math.fsum(str(l['positives'][0]) in {ids[i] for i in row} for l,row in zip(labels,ix,strict=True))/n}
        else:
            qrels={str(q):{str(l['positives'][0])} for q,l in enumerate(labels)}
            predictions={str(q):{ids[i]:float(scores[q,i]) for i in row} for q,row in enumerate(ix)}
            t2i={**ir_mean(qrels,predictions),'queries':n}
            ik=min(100,n)
            i2t=bound_native(out/'official-i2t.npy',list(range(len(ids))),block,{**base,'stage':'official-i2t','k':ik},meta['official_arrays']['official-i2t'])
            if i2t.shape!=(len(ids),ik*2) or i2t.dtype!=np.float32:raise ValueError('Original native I2T top100 differs')
            iix=top_indices(i2t[:,:ik],n,ik)
            if (np.diff(i2t[:,ik:],axis=1)>0).any():raise ValueError('Native I2T scores are not descending')
            iqrels={key:set() for key in ids}
            for q,l in enumerate(labels):iqrels[str(l['positives'][0])].add(str(q))
            ipred={key:{str(q):float(s) for q,s in zip(row,values,strict=True)} for key,row,values in zip(ids,iix,i2t[:,ik:],strict=True)}
            result={'t2i':t2i,'i2t':{**ir_mean(iqrels,ipred),'queries':len(ids)}}
        report=C.read(out/'results/report.json')
        if (report.get('model')!=job['model'] or report.get('dataset')!=name or report.get('queries')!=n
                or report.get('raw_queries')!=bundle['raw_queries'] or report.get('gallery')!=len(ids)
                or report.get('excluded')!=bundle['excluded'] or report.get('retained_empty')!=bundle['retained_empty']
                or report.get('provenance')!={'index_meta_sha256':C.sha(out/'index-meta.json'),'index':meta}):
            raise ValueError('Native mechanical report changes the original denominator or source')
        recomputed[name]=result;reports[name]=C.sha(out/'results/report.json')
        details[name]={'agreement':same_numbers(report.get('official'),result),'queries':n,'gallery':len(ids),
            'native_arrays':meta['official_arrays'],'index_meta_sha256':C.sha(out/'index-meta.json'),
            'admission_audit':'Original GPU topk retained. T2I checked against every cached native score; I2T original top100 batch hashes/indices/scores checked, without a fresh GPU dot or choosing different tied members.'}
    return {'agreement':all(r['agreement'] for r in details.values()),'recomputed':recomputed,'datasets':details,'reports':reports}

def mcmr(job):
    from codex.retrieval_matrix_mcmr import runner as N
    plan=C.validate_job(job);ds=plan['datasets']['mcmr'];out=C.safe(job['runs']['mcmr']);bundle=N.load_data(ds,verify_images=True)
    identity,meta=native_identity(job,plan,'mcmr',bundle,out)
    for name,items,batch,stage in [('corpus',bundle['gallery'],64,'fused-product'),('queries',bundle['queries'],32,'native-query')]:
        a=bound_native(out/(name+'.npy'),items,batch,{**identity,'stage':stage,'processor_sha256':meta['processor_sha256']},meta['arrays'][name])
        if a.dtype!=np.float32 or not np.allclose(np.linalg.norm(a,axis=1),1,atol=1e-2):raise ValueError('Native fused/query normalized vectors differ')
    a=bound_native(out/'native-top100.npy',bundle['queries'],ds['similarity_batch'],
        {**identity,'stage':'native-bf16-top100','top_k':100,'torch_topk':True,'similarity_batch':ds['similarity_batch']},meta['arrays']['native-top100'])
    if a.shape!=(3997,200) or a.dtype!=np.float32:raise ValueError('Complete native MCMR top100 layout differs')
    ix=top_indices(a[:,:100],104981,100)
    if (np.diff(a[:,100:],axis=1)>0).any():raise ValueError('Native MCMR top100 scores are unsorted')
    raw=N.rows(C.checked(ds['instances']));qrels={str(r['qid']):set(map(str,r['pos'])) for r in raw}
    if len(qrels)!=len(raw) or set(qrels)!={r['id'] for r in bundle['queries']}:raise ValueError('Every original MCMR query must remain')
    ids=[r['id'] for r in bundle['gallery']];idset=set(ids)
    if any(not r or not r<=idset for r in qrels.values()):raise ValueError('MCMR original positive products missing')
    pred={q['id']:{ids[i]:float(s) for i,s in zip(row,values,strict=True)} for q,row,values in zip(bundle['queries'],ix,a[:,100:],strict=True)}
    fields=('recall_1','recall_5','recall_10','recall_50','recall_100','ndcg_cut_5','ndcg_cut_10','map_cut_10')
    result=ir_mean(qrels,pred,fields)
    pred10={q['id']:{ids[i]:float(s) for i,s in zip(row[:10],values[:10],strict=True)} for q,row,values in zip(bundle['queries'],ix,a[:,100:],strict=True)}
    result['MRR@10']=ir_mean(qrels,pred10,('recip_rank',))['recip_rank']
    diagnostic={'Hit@1':[],'Recall@1':[],'MRR@10':[]}
    for q,row in zip(bundle['queries'],ix,strict=True):
        positive=qrels[q['id']];hit=float(ids[row[0]] in positive)
        diagnostic['Hit@1'].append(hit);diagnostic['Recall@1'].append(hit/len(positive))
        diagnostic['MRR@10'].append(next((1/(j+1) for j,i in enumerate(row[:10]) if ids[i] in positive),0.))
    ordinal={k:math.fsum(v)/3997 for k,v in diagnostic.items()};report=C.read(out/'results/report.json')
    if report.get('strict_CORE_P0_match_claim') is not False or report.get('queries')!=3997 or report.get('gallery_products')!=104981 or report.get('model')!=job['model'] or report.get('provenance')!=meta:
        raise ValueError('MCMR must retain complete distinct products and disclose unverified CORE conversion')
    return {'agreement':same_numbers(report.get('primary_pytrec'),result) and same_numbers(report.get('native_torch_ordinal_diagnostic'),ordinal),
        'recomputed':result,'ordinal_diagnostic':ordinal,'strict_CORE_P0_match_claim':False,'eligible_for_P0_qualification':False,
        'reason':'Original MMEB renderer verified; original CORE converter is unavailable. Differences are descriptive only.',
        'report_sha256':C.sha(out/'results/report.json'),'index_meta_sha256':C.sha(out/'index-meta.json'),'arrays':meta['arrays'],
        'admission_audit':'Original native GPU top100 retained, complete checkpoint item/batch hashes verified. No fresh GPU similarity or topk is claimed.'}
