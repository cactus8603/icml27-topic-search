"""[codex] Central-only native MCQ producer; CPU analysis is separate."""
from __future__ import annotations
import argparse,json,os,subprocess,sys,uuid
from pathlib import Path
from importlib.metadata import version
from time import monotonic
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from codex.retrieval_matrix_rerank_native_p0 import contracts as C,adapters as A
from codex.retrieval_matrix_rerank_native_p0.build_plan import OWN,EXECUTED
R=C.R;HERE=Path(__file__).resolve().parent
SETTINGS={'storage_examples':32,'swift_pair_batch':32,'native_pair_batch':1,'jina_groups':'COLA2+2/SCPP3/Neg4',
    'smoke_absolute_tolerance':.01,'required_distinct_benchmarks':2,'absolute_tolerance':.01}


def validate(plan):
    if (plan.get('schema')!=1 or plan.get('phase')!='P0-native-reranker-mcq-v1'
            or plan.get('purpose')!='OFFICIAL_RERANKER_QUALIFICATION' or plan.get('settings')!=SETTINGS):
        raise ValueError('Fixed complete native MCQ protocol/settings required')
    protocols=R.read(C.checked(plan['protocols']));draft=R.read(C.checked(plan['backend_reference']))
    if set(plan['variants'])!=set(protocols['variants']) or {k:v for k,v in plan['environment'].items() if k!='modules'}!=draft['environment']:
        raise ValueError('Retain every exact variant and pinned runtime including missing-target variants')
    for key,variant in plan['variants'].items():
        if {k:v for k,v in variant.items() if k not in {'model','model_spec_sha256'}}!=protocols['variants'][key]:
            raise ValueError('Exact released protocol/target descriptors changed')
        model={**variant['model'],'verified_protocol':False}
        if model!=draft['models'][key] or variant['model_spec_sha256']!=R.digest(variant['model']):
            raise ValueError('Complete model payload/processor/source spec changed')
    if (plan.get('data_reference')!=protocols['data']['data_reference'] or protocols['data'].get('counts')!=C.COUNTS
            or protocols['data'].get('examples')!=15912 or protocols['data'].get('pairs')!=58891 or protocols['data'].get('unique_images')!=9425):
        raise ValueError('Published targets must bind original complete frozen MCQ data, counts and pairing')
    for item in plan['external_sources'].values():C.checked(item)
    if plan['external_sources']!=protocols['sources']:raise ValueError('Official source registry changed')
    probe=R.read(C.checked(plan['cpu_probe']))
    if (probe.get('state')!='PASS' or probe.get('weights_loaded') is not False or probe['backend_specs']!=plan['backend_reference']
            or plan['environment'].get('modules')!=probe.get('modules')):
        raise ValueError('Matching official CPU processor/template receipt required')


def source_gate(path,plan,commit,key):
    validate(plan)
    if (len(commit)!=40 or any(x not in '0123456789abcdef' for x in commit)
            or plan.get('state')!='FROZEN' or plan.get('reviewed') is not True):
        raise ValueError('Native MCQ needs independent review and a complete committed source freeze')
    required=set(EXECUTED)|{'codex/retrieval_matrix_rerank_native_p0/'+name for name in OWN}
    if not required<=plan['source_files'].keys():raise ValueError('Executing source omitted from freeze')
    for relative in set(plan['source_files'])|{str(path.resolve().relative_to(ROOT))}:
        source=R.safe(ROOT/relative)
        if (not source.is_relative_to(ROOT) or R.C.git_bytes(commit,relative)!=source.read_bytes()
                or (relative in plan['source_files'] and R.sha(source)!=plan['source_files'][relative])):
            raise ValueError('Native MCQ source/plan differs from immutable commit')
    if subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain','--',str(HERE.relative_to(ROOT))],text=True).strip():
        raise ValueError('Native MCQ module must remain clean')
    variant=plan['variants'][key]
    if variant['model'].get('verified_protocol') is not True:
        raise ValueError('Native backend protocol has not received independent review')
    if sum(v['verified_protocol_match'] is True for v in variant['targets'].values())<2:
        raise ValueError('TARGETS_INCOMPLETE: this exact native variant lacks two published MCQ targets; no P0 GPU inference')
    return variant


def environment(plan):
    if (os.environ.get('RETRIEVAL_CENTRAL_OWNER')!='[codex]' or not os.environ.get('RETRIEVAL_CENTRAL_TASK')
            or not os.environ.get('CUDA_VISIBLE_DEVICES')):
        raise ValueError('Native MCQ GPU requires root central dispatcher')
    import torch,transformers,flash_attn
    if torch.cuda.device_count()!=1 or not torch.cuda.is_available():raise ValueError('Require exactly one assigned GPU')
    for name,expected in plan['environment']['packages'].items():
        if version(name)!=expected:raise ValueError('Native MCQ distribution differs: '+name)
    for name in ('install_report','frozen_pip'):C.checked(plan['environment'][name])
    import importlib
    for name,expected in plan['environment']['modules'].items():
        imported=importlib.import_module(name);path=Path(imported.__file__).resolve()
        if {'path':str(path),'actual_version':getattr(imported,'__version__',None),'sha256':R.sha(path)}!=expected:
            raise ValueError('Actual native imported module path/version/source differs: '+name)
    if torch.__version__!=plan['environment']['packages']['torch'] or transformers.__version__!=plan['environment']['packages']['transformers']:
        raise ValueError('Actual native imported runtime differs')
    return {'torch':torch.__version__,'transformers':transformers.__version__,'flash_attn':flash_attn.__version__,
        'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'visible':os.environ['CUDA_VISIBLE_DEVICES']}


def memory():
    try:
        import torch
        if torch.cuda.is_initialized():return {'allocated':torch.cuda.max_memory_allocated(),'reserved':torch.cuda.max_memory_reserved()}
    except Exception:pass
    return {'allocated':None,'reserved':None}


def journal(out,row):
    with (out/'calls.jsonl').open('a') as stream:
        stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())


def raw_group(out,pairs,identity,variant,instruction,get_engine):
    ids=[p['id'] for p in pairs];path=out/'raw'/(R.digest(ids)+'.json')
    if path.exists():
        saved=R.read(path)
        if (saved.get('identity_sha256')!=R.digest(identity) or saved.get('pair_ids')!=ids or saved.get('instruction')!=instruction
                or saved.get('backend_sha256')!=R.sha(out/'backend.json')):
            raise ValueError('Pending native raw transaction differs from exact model/data/request')
        C.validate_scores(saved['responses'],pairs,variant);return saved['responses']
    engine=get_engine();call=uuid.uuid4().hex;began=monotonic()
    journal(out,{'state':'STARTED','call':call,'pairs':len(pairs),'pair_ids':ids,
        'forward_budget':2*len(pairs) if variant['model']['backend']=='mcmr_pointwise' else 1})
    try:
        response=engine.score_pairs(pairs,instruction=instruction);C.validate_scores(response,pairs,variant)
        R.fixed_json(path,{'identity_sha256':R.digest(identity),'pair_ids':ids,'instruction':instruction,'responses':response,
            'backend_sha256':R.sha(out/'backend.json'),'call':call,'seconds':monotonic()-began,'memory':memory()})
    except BaseException:
        journal(out,{'state':'FAILED_OR_INTERRUPTED','call':call,'seconds':monotonic()-began,'memory':memory()});raise
    journal(out,{'state':'COMPLETE','call':call,'seconds':monotonic()-began,'raw_sha256':R.sha(path),'memory':memory()})
    return response


def cost_summary(out):
    events=[json.loads(line) for line in (out/'calls.jsonl').read_text().splitlines()]
    starts={e['call']:e for e in events if e['state']=='STARTED'};ends={e['call']:e for e in events if e['state']!='STARTED'}
    return {'attempted_calls':len(starts),'attempted_pairs':sum(e['pairs'] for e in starts.values()),
        'requested_forward_budget':sum(e['forward_budget'] for e in starts.values()),
        'completed_call_forwards':sum(starts[k]['forward_budget'] for k,e in ends.items() if e['state']=='COMPLETE'),
        'failed_or_unfinished_forward_budget':sum(e['forward_budget'] for k,e in starts.items() if ends.get(k,{}).get('state')!='COMPLETE'),
        'calls_without_end_receipt':len(set(starts)-set(ends)),
        'failed_or_interrupted_calls':sum(e['state']=='FAILED_OR_INTERRUPTED' for e in ends.values()),
        'recorded_seconds':sum(e['seconds'] for e in ends.values()),
        'peak_allocated_bytes':max([e.get('memory',{}).get('allocated') or 0 for e in ends.values()],default=0),
        'peak_reserved_bytes':max([e.get('memory',{}).get('reserved') or 0 for e in ends.values()],default=0),
        'disclosure':'Requests include retries; forward budgets for failed/unfinished calls are upper bounds, not known executed forwards. Hard-killed calls/invocations can retain unknown duration/peak; not zero cost.'}


def require_smoke(proof,identity,plan):
    value=proof.get('max_abs',np.nan)
    if (proof.get('state')!='PASS' or proof.get('technical_only') is not True or proof.get('identity')!=identity
            or not isinstance(value,(int,float)) or not np.isfinite(value) or not 0<=value<=.01):
        raise ValueError('Matching synthetic native protocol/batch smoke required')


def smoke(out,identity,variant,plan):
    if (out/'smoke.json').exists():require_smoke(R.read(out/'smoke.json'),identity,plan);return
    from PIL import Image
    engine=A.create(variant);image=out/'synthetic.png';Image.new('RGB',(1280,960),(64,128,32)).save(image)
    width=32 if variant['model']['backend']=='swift_released' else (4 if variant['model']['backend']=='jina_native' else 1)
    pairs=[{'id':str(i),'image':str(image),'query':'A synthetic color image.' if i%2==0 else 'A synthetic image with colors for a fixed technical smoke.'} for i in range(width)]
    instruction='Determine if the following image is relevant to the query. '
    first=engine.score_pairs(pairs[:1],instruction=instruction);C.validate_scores(first,pairs[:1],variant)
    widths=(2,3,4) if variant['model']['backend']=='jina_native' else (width,)
    differences=[];raw=[]
    for size in widths:
        response=engine.score_pairs(pairs[:size],instruction=instruction);C.validate_scores(response,pairs[:size],variant)
        differences.append(abs(response[0]['score']-first[0]['score']));raw.append({'batch':size,'responses':response})
    maximum=max(differences);passed=maximum<=.01
    R.fixed_json(out/'smoke.json',{'state':'PASS' if passed else 'FAILED','technical_only':True,'identity':identity,
        'max_abs':maximum,'raw':raw,'signature':engine.signature,'memory':memory()})
    if not passed:raise ValueError('Synthetic native batch smoke failed; preserve failure without fallback')


def produce(out,identity,variant,plan,rows):
    if (out/'complete.json').exists():verify_complete(out,identity,rows,variant);return
    holder=[]
    def get_engine():
        if not holder:
            engine=A.create(variant);R.fixed_json(out/'backend.json',engine.signature);holder.append(engine)
        return holder[0]
    complete={};raw={}
    for subset in C.COUNTS:
        examples=[r for r in rows if r['subset']==subset]
        def compute(batch):
            lookup={}
            for group in C.call_groups(batch,variant):
                response=raw_group(out,group,identity,variant,batch[0]['instruction'],get_engine)
                lookup.update({r['id']:r['score'] for r in response})
            return np.array([[lookup[p['id']] for p in row['pairs']] for row in batch],dtype=np.float64)
        path=out/(subset+'.npy')
        R.checkpoint(path,examples,32,compute,{**identity,'subset':subset},progress=out/'progress.json')
        complete[subset]=R.array_receipt(path)
        for group in C.call_groups(examples,variant):
            name=R.digest([p['id'] for p in group])+'.json';raw[name]=R.sha(out/'raw'/name)
    R.fixed_json(out/'complete.json',{'state':'COMPLETE','identity':identity,'subsets':complete,'raw_files':raw,
        'backend_sha256':R.sha(out/'backend.json'),'calls_sha256':R.sha(out/'calls.jsonl'),'costs':cost_summary(out),
        'examples':15912,'distinct_pairs':58891,'original_images':9425})


def verify_complete(out,identity,rows,variant):
    receipt=R.read(out/'complete.json')
    if (receipt.get('state')!='COMPLETE' or receipt.get('identity')!=identity or receipt.get('examples')!=15912
            or receipt.get('distinct_pairs')!=58891 or receipt.get('original_images')!=9425
            or set(receipt['subsets'])!=set(C.COUNTS) or receipt['backend_sha256']!=R.sha(out/'backend.json')
            or receipt['calls_sha256']!=R.sha(out/'calls.jsonl') or receipt['costs']!=cost_summary(out)):
        raise ValueError('Native complete provenance/call/coverage receipt changed')
    scores=[];raw_files={}
    for subset,count in C.COUNTS.items():
        examples=[r for r in rows if r['subset']==subset];path=out/(subset+'.npy')
        if receipt['subsets'][subset]!=C.bound_array(path,examples,{**identity,'subset':subset}):raise ValueError('Native score array/state receipt changed')
        array=np.load(path,allow_pickle=False)
        width=4 if subset=='cola' or subset.endswith('_mcq') else 3
        if array.shape!=(count,width) or array.dtype!=np.float64:raise ValueError('Original MCQ array row/pair shape changed')
        lookup={p['id']:float(v) for row,values in zip(examples,array,strict=True) for p,v in zip(row['pairs'],values,strict=True)}
        for group in C.call_groups(examples,variant):
            ids=[p['id'] for p in group];name=R.digest(ids)+'.json';path=out/'raw'/name;raw_files[name]=R.sha(path)
            saved=R.read(path)
            if (saved.get('identity_sha256')!=R.digest(identity) or saved.get('pair_ids')!=ids
                    or saved.get('instruction')!=examples[0]['instruction'] or saved.get('backend_sha256')!=receipt['backend_sha256']):
                raise ValueError('Native original raw model/request/pair identity changed')
            C.validate_scores(saved['responses'],group,variant)
            if any(r['score']!=lookup[r['id']] for r in saved['responses']):raise ValueError('Saved scalar differs from original native raw')
        scores.extend({'id':k,'score':v} for k,v in lookup.items())
    if raw_files!=receipt['raw_files']:raise ValueError('Native raw transaction coverage changed')
    return scores


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=('smoke','score','analyze'),required=True)
    for name in ('plan','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--variant',required=True);p.add_argument('--freeze-commit',required=True);p.add_argument('--smoke-out',type=Path);a=p.parse_args()
    began=monotonic();plan=R.read(a.plan);variant=source_gate(a.plan,plan,a.freeze_commit,a.variant)
    out=R.safe(a.out);root=ROOT/'dl/retrieval-matrix-rerank-native-p0/runs'
    if out==root or not out.is_relative_to(root):raise ValueError('Isolated native MCQ run root required')
    identity={'plan_sha256':R.sha(a.plan),'freeze_commit':a.freeze_commit,'variant':a.variant,'variant_spec':variant,
        'source_files':plan['source_files'],'data_reference':plan['data_reference']}
    with R.writer(out):
        invocation=uuid.uuid4().hex;ledger_path=out/'invocations.json';ledger=R.read(ledger_path) if ledger_path.exists() else []
        ledger.append({'invocation':invocation,'state':'STARTED','stage':a.stage,'time':datetime.now(ZoneInfo('Asia/Taipei')).isoformat()});R.atomic_json(ledger_path,ledger)
        ok=False
        try:
            if a.stage=='analyze':identity.update({k:R.read(out/'identity.json')[k] for k in ('environment','producer')})
            else:
                identity.update(environment=environment(plan),producer='central-runtime:'+os.environ['RETRIEVAL_CENTRAL_TASK'])
                R.fixed_json(out/'identity.json',identity)
            if a.stage=='smoke':smoke(out,identity,variant,plan)
            else:
                rows=C.load_examples(plan,variant,verify_images=a.stage=='score')
                if a.stage=='score':
                    if a.smoke_out is None:raise ValueError('Matching synthetic-only smoke required')
                    require_smoke(R.read(R.safe(a.smoke_out)/'smoke.json'),identity,plan);produce(out,identity,variant,plan,rows)
                else:
                    scores=verify_complete(out,identity,rows,variant);report,detail=C.evaluate(rows,scores,variant)
                    report.update(identity=identity,costs=R.read(out/'complete.json')['costs'],complete_sha256=R.sha(out/'complete.json'))
                    R.atomic_json(out/'results/report.json',report);R.atomic_json(out/'results/per-example.json',detail)
            ok=True
        finally:
            ledger=R.read(ledger_path);ledger.append({'invocation':invocation,'state':'COMPLETE' if ok else 'FAILED','stage':a.stage,
                'wall_seconds':monotonic()-began,'time':datetime.now(ZoneInfo('Asia/Taipei')).isoformat()});R.atomic_json(ledger_path,ledger)


if __name__=='__main__':main()
