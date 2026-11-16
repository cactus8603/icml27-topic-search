"""[codex] Prospective contracts for the seven exact native producer jobs.

Plan building reads frozen source and public data metadata only. Production
artifact reads belong exclusively inside the durable reading session.
"""
from __future__ import annotations
import importlib,importlib.util,json,os,re,subprocess,sys
from importlib.metadata import version
from pathlib import Path
from codex.retrieval_matrix_native_analysis import contracts as C

ROOT,HERE=C.ROOT,C.HERE
PHASE='NATIVE-P0-FROZEN-PRODUCER-ANALYSIS-v1'
OUT=ROOT/'dl/retrieval-matrix-native-analysis'
ANCHORS=OUT/'reading-anchor'
EXCLUDED=['/root','/root/matrix_runner','/root/public_data','/root/official_p0',
    '/root/official_p0/p3_contract_review','/root/official_p0/native_full_fixtures']
ANALYSIS_AUTHORS=['/root/official_p0','/root/official_p0/native_full_fixtures']
STAGES=('mechanical','independent','qualify')

def read(path):return json.loads(Path(path).read_text())
def write_new(path,value):
    path=C.safe(path);path.parent.mkdir(parents=True,exist_ok=True)
    # No stage can overwrite a prior result, failure, or independent record.
    with path.open('x') as stream:
        json.dump(value,stream,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
        stream.flush();os.fsync(stream.fileno())
def checked(item):
    path=C.safe(item['path'])
    if C.sha(path)!=item['sha256']:raise ValueError('Bound file changed: '+str(path))
    return path
def own_sources():
    names=sorted(p for p in HERE.iterdir() if p.suffix in {'.py','.sh','.md'} or p.name=='source-profiles.json')
    return {str(p.relative_to(ROOT)):{'sha256':__import__('hashlib').sha256(C.read_header(p)).hexdigest()
        if p.name=='PREREG.md' else C.sha(p),'immutable_body_only':p.name=='PREREG.md'} for p in names}

def cpu_recipe(profile):
    paired=profile['family']!='reranker'
    py=ROOT/('dl/retrieval3090/envs/retrieval3090/bin/python' if paired else 'dl/retrieval-matrix-rerank/env/bin/python')
    prefix=['dl/retrieval-matrix-qwen2-runtime-fix/deps','dl/retrieval-matrix-extra/deps',
        'dl/retrieval-matrix/deps/qwen2-transformers451','dl/retrieval-matrix/deps/qwen2-flash',
        'dl/retrieval-matrix/deps/core-p0'] if paired else []
    env={'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':':'.join(str(ROOT/p) for p in prefix+[Path('.')]),
        'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
        'TOKENIZERS_PARALLELISM':'false','HF_HOME':str(ROOT/'hf_home'),'HF_HUB_OFFLINE':'1',
        'TRANSFORMERS_OFFLINE':'1','HF_MODULES_CACHE':str(ROOT/'hf_home/modules'),'TMPDIR':str(ROOT/'hf_home/tmp'),
        'TZ':'Asia/Taipei'}
    if not paired:env.update(IMAGE_MAX_TOKEN_NUM='1200',GENERATIVE_RERANKER_POSITIVE_TOKEN='yes',GENERATIVE_RERANKER_NEGATIVE_TOKEN='no')
    return {'python':str(py),'environment':env,'gpu_allowed':False}

def known_authors(profile):
    # The conservative exclusion union below is deliberately broader than these
    # evidenced authors. Commit display names are never used as agent evidence.
    if profile['family']=='reranker':
        return [{'actor':'/root/official_p0','evidence_kind':'parent_assignment',
            'evidence':'Parent assigned native reranker implementation to this agent; frozen producer '+C.RERANK_REF}]
    authors=[{'actor':'/root/official_p0/p3_contract_review','evidence_kind':'parent_assignment',
        'evidence':'Parent explicitly identified both runtime repair implementations as this agent; producer '+profile['producer_commit']}]
    if profile['family']=='siglip2':
        path=ROOT/'codex/reports/2026-11-16-native-p0-prelaunch-review-codex.md'
        authors.append({'actor':'/root/matrix_runner','evidence_kind':'implementation_handoff_report',
            'evidence':C.entry(path),'location':'Opening implementer/reviewer attribution'})
    return authors

def manifest(profile):
    """Derive every effect path without stat/open/glob of a producer output."""
    root=C.safe(profile['output_root']);immutable=[];completion=[];results=[];array_states=[];raw_calls=[]
    smoke=root/'smoke'/'smoke.json' if profile['family']=='reranker' else root/'smoke.json'
    immutable.append(str(smoke))
    if profile['family']=='reranker':
        plan=read(checked(profile['producer_plan']));variant=plan['variants'][profile['model']]
        # Frozen original public IDs/order, never a live output directory listing.
        from codex.retrieval_matrix_rerank_native_p0 import contracts as N
        rows=N.load_examples(plan,variant,verify_images=False);score=C.safe(profile['score_root'])
        for name in N.COUNTS:
            immutable.extend(str(score/(name+suffix)) for suffix in ('.npy','.state.json'))
            array_states.append(str(score/(name+'.state.json')))
            subset=[r for r in rows if r['subset']==name]
            raw_calls.extend(str(score/'raw'/(C.digest([p['id'] for p in group])+'.json')) for group in N.call_groups(subset,variant))
        immutable.extend(str(score/n) for n in ('identity.json','backend.json','complete.json','calls.jsonl'))
        immutable.extend(raw_calls);completion=[str(score/'complete.json')]
        results=[str(score/'results/report.json'),str(score/'results/per-example.json')]
        mutable=[str(score/'invocations.json')]
        costs={'authoritative_call_ledger':str(score/'calls.jsonl'),'completion_summary':str(score/'complete.json'),
            'failed_budget_is_upper_bound':True,'unrecorded_kill_time_and_peak':'UNKNOWN'}
    else:
        for benchmark in profile['benchmarks']:
            index=root/benchmark/'index';completion.append(str(index/'receipt.json'))
            immutable.extend(str(index/n) for n in ('receipt.json','images.npy','images.state.json','texts.npy','texts.state.json'))
            array_states.extend(str(index/n) for n in ('images.state.json','texts.state.json'))
            if profile['family']=='paired':
                immutable.extend(str(index/n) for n in ('author-scalar-scores.npy','author-scalar-scores.state.json','final-receipt.json'))
                array_states.append(str(index/'author-scalar-scores.state.json'));completion.append(str(index/'final-receipt.json'))
            results.append(str(root/benchmark/'results/report.json'))
        mutable=[]
        costs={'authoritative_call_ledger':None,'checkpoint_active_seconds':array_states,
            'gpu_wall_and_peak':'NOT_RECORDED_BY_FROZEN_PRODUCER','no_fabricated_gpu_cost':True}
    if len(immutable)!=len(set(immutable)) or len(raw_calls)!=len(set(raw_calls)):raise ValueError('Prospective raw paths must be unique')
    return {'schema':1,'job':profile['id'],'producer_commit':profile['producer_commit'],
        'producer_plan':profile['producer_plan'],'model_spec_sha256':profile['model_spec_sha256'],
        'immutable_raw':sorted(immutable),'completion':completion,'array_states':array_states,
        'raw_call_paths':raw_calls,'mechanical_results':results,'producer_mutable_analysis_ledger':mutable,
        'cost':costs,'all_paths_prospective':True,'production_paths_opened':False}

def job_value(profile):
    manifests=OUT/'manifests';outputs=OUT/'runs/native-v1'/profile['id']
    probe_path=OUT/('cpu-environment-'+profile['family']+'-v2.json')
    probe=read(probe_path)
    if (probe.get('technical_only') is not True or probe.get('returncode')!=0
            or probe.get('production_outputs_read') is not False
            or probe.get('runtime',{}).get('environment_recipe')!=cpu_recipe(profile)
            or probe.get('runtime',{}).get('cuda_initialized') is not False):
        raise ValueError('A source-only actual family CPU import/binary receipt is required')
    return {'id':profile['id'],'profile':profile,'producer_plan':profile['producer_plan'],
        'producer_commit':profile['producer_commit'],'model_spec_sha256':profile['model_spec_sha256'],
        'manifest_path':str(manifests/(profile['id']+'.json')),'cpu':cpu_recipe(profile),
        'cpu_probe':C.entry(probe_path),'cpu_runtime':probe['runtime'],
        'roles':{'known_implementation_authors':known_authors(profile),
            'analysis_authors':ANALYSIS_AUTHORS,'excluded_reviewers':EXCLUDED,
            'fresh_independent_reviewer_required':True,'producer_and_mechanical_actor_excluded':True},
        'outputs':{name:str(outputs/(name+'.json')) for name in ('mechanical','independent-review','independent-envelope','qualification','admission')},
        'stage_attempts':{stage:str(outputs/(stage+'-attempt.json')) for stage in STAGES}}

def validate_plan(plan,*,verify_manifests=True):
    if (plan.get('schema')!=1 or plan.get('phase')!=PHASE or plan.get('shared_analysis_ref')!=C.SHARED_REF
            or plan.get('user_decision')!='PENDING' or plan.get('gpu_allowed') is not False
            or plan.get('state') not in {'REVIEW_REQUIRED','FROZEN'} or type(plan.get('reviewed')) is not bool):
        raise ValueError('Exact native CPU analysis plan/schema required')
    if (plan.get('execution_plan') is not True or plan.get('decision_order')!=[
            'original-frozen-mechanical','fresh-nonimplementer-original-array-recomputation','original-native-make-admit','user-final-decision']
            or plan.get('excluded_missing_targets')!=['qwen_native2','qwen_native8','qwen_instruct8']
            or plan.get('production_outputs_read_during_build') is not False
            or plan.get('qualification_created_during_build') is not False):
        raise ValueError('Retain the prospective analysis order, exclusions, and zero-exposure build')
    if plan.get('shared_sources')!=C.shared_sources() or plan.get('source_files')!=own_sources():raise ValueError('Own/shared immutable source table changed')
    inventory=C.profiles()
    if plan.get('schedule')!=inventory['schedule'] or set(plan.get('jobs',{}))!=set(inventory['profiles']):raise ValueError('Retain all seven fixed scheduled producer jobs')
    for key,profile in inventory['profiles'].items():
        declared=plan['jobs'][key];expected=job_value(profile)
        if {k:v for k,v in declared.items() if k!='manifest'}!=expected:raise ValueError('Job/source/model/path/runtime/role contract changed')
        item=declared.get('manifest',{})
        if item.get('path')!=expected['manifest_path']:raise ValueError('Manifest must use its designated prospective path')
        if verify_manifests and read(checked(item))!=manifest(profile):raise ValueError('Manifest differs from original source-derived paths')
    return plan

def frozen_reader():
    C.shared_sources()
    name='native_analysis_private_reading_'+C.SHARED_REF
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,ROOT/'codex/retrieval_matrix_analysis/reading.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);sys.modules[name]=module
    return sys.modules[name]

def source_gate(plan_path,commit):
    plan=read(plan_path);validate_plan(plan)
    if not re.fullmatch('[0-9a-f]{40}',commit) or plan['state']!='FROZEN' or plan['reviewed'] is not True:
        raise ValueError('REVIEW_REQUIRED: a separate independently approved committed plan is required')
    C.frozen(str(C.safe(plan_path).relative_to(ROOT)),commit)
    for relative,item in plan['source_files'].items():C.frozen(relative,commit,immutable_prereg=item['immutable_body_only'])
    frozen_reader().records(HERE/'PREREG.md',expected_plan=C.sha(plan_path),expected_commit=commit,anchor_root=ANCHORS)
    return plan

def session(plan_path,commit,job,actor,paths):
    if not actor:raise ValueError('An explicit analysis actor is required')
    result=frozen_reader().ReadSession(plan_path,commit,job,actor,sorted(set(map(str,paths))),anchor_root=ANCHORS)
    result.args['path']=HERE/'PREREG.md';return result

def require_read(plan_path,commit,start):
    return frozen_reader().require_completed(HERE/'PREREG.md',start,plan_sha=C.sha(plan_path),commit=commit,anchor_root=ANCHORS)

def independent_actor(job,actor,mechanical_actor,*,nonimplementer=False):
    excluded=set(job['roles']['excluded_reviewers'])|set(job['roles']['analysis_authors'])
    excluded.update(r['actor'] for r in job['roles']['known_implementation_authors']);excluded.add(mechanical_actor)
    if not nonimplementer or not actor or actor in excluded or actor.startswith('central-runtime:'):
        raise ValueError('Fresh nonimplementer numeric reviewer required; true authors and mechanical actor remain excluded')

def cpu_environment(job,*,probe=False):
    expected=job['cpu'];env=expected['environment']
    if Path(sys.executable).absolute()!=Path(expected['python']).absolute() or any(os.environ.get(k)!=v for k,v in env.items()):
        raise ValueError('Use the fixed family-specific CPU interpreter and complete environment recipe')
    import torch
    if torch.cuda.is_initialized():raise ValueError('Analysis must never initialize CUDA')
    profile=job['profile'];producer=read(checked(job['producer_plan']))
    if profile['family']!='reranker':
        name='codex.'+('retrieval_matrix_native_runtime_fix' if profile['family']=='siglip2' else 'retrieval_matrix_paired_runtime_fix')+'.contracts'
        module=importlib.import_module(name);dependencies=module.validate(producer,imports=True)
        actual=module.actual_modules(dependencies)
    else:
        for name,value in producer['environment']['packages'].items():
            if version(name)!=value:raise ValueError('Native MCQ CPU package mismatch: '+name)
        for name in ('install_report','frozen_pip'):checked(producer['environment'][name])
        actual={}
        for name,expected_module in producer['environment']['modules'].items():
            module=importlib.import_module(name);path=Path(module.__file__).resolve()
            actual[name]={'path':str(path),'actual_version':getattr(module,'__version__',None),'sha256':C.sha(path)}
            if actual[name]!=expected_module:raise ValueError('Native MCQ CPU imported module mismatch: '+name)
        if torch.__version__!=producer['environment']['packages']['torch']:raise ValueError('Actual torch mismatch')
    if torch.cuda.is_initialized():raise ValueError('CPU import unexpectedly initialized CUDA')
    value={'python':str(Path(sys.executable).absolute()),'modules':actual,'cuda_initialized':False,'environment_recipe':expected,
        'array_runtime':frozen_reader().C.runtime_snapshot()}
    if not probe and value!=job.get('cpu_runtime'):raise ValueError('Actual CPU modules/Python/NumPy array binary differs from reviewed preflight')
    return value

def facade(job):
    name=Path(job['profile']['qualification_module']['path']).parent.name
    return importlib.import_module('codex.'+name+'.qualification')

def reports(job):
    p=job['profile'];root=Path(p['output_root'])
    return {b:root/b/'results/report.json' for b in p['benchmarks']}

def model_spec(job):
    producer=read(checked(job['producer_plan']));p=job['profile']
    model=producer['variants'][p['model']]['model'] if p['family']=='reranker' else producer['model_spec']
    if C.digest(model)!=job['model_spec_sha256']:raise ValueError('Qualification must bind the full original model spec')
    return model

def raw_snapshot(value):return {p:C.sha(p) for p in value['immutable_raw']}

def independent_details(job):
    """Separate strict per-example audit; original gates supply aggregate review."""
    p=job['profile']
    if p['family']!='reranker':return None
    import numpy as np
    from codex.retrieval_matrix_rerank_native_p0 import contracts as N
    plan=read(checked(job['producer_plan']));variant=plan['variants'][p['model']]
    rows=N.load_examples(plan,variant,verify_images=False);root=Path(p['score_root']);lookup={}
    for subset,count in N.COUNTS.items():
        examples=[r for r in rows if r['subset']==subset];array=np.load(root/(subset+'.npy'),allow_pickle=False)
        width=4 if subset=='cola' or subset.endswith('_mcq') else 3
        if array.dtype!=np.float64 or array.shape!=(count,width) or len(examples)!=count or not np.isfinite(array).all():
            raise ValueError('Independent detail audit needs every native finite FP64 scalar in original shape')
        for example,values in zip(examples,array,strict=True):
            if example['id'] in lookup:raise ValueError('Independent detail audit cannot merge repeated IDs')
            if subset=='cola':
                ok=(values[0]>values[2] and values[3]>values[1]) if variant['cola_rule']=='cola_image_choice' else (values[0]>values[1] and values[3]>values[2])
            elif subset in N.SCPP:ok=values[0]>values[2] and values[1]>values[2]
            else:ok=all(values[0]>negative for negative in values[1:])
            lookup[example['id']]=int(ok)
    if len(rows)!=15912 or len(lookup)!=15912:raise ValueError('Independent detail audit requires the complete original denominator')
    expected=[{'id':r['id'],'subset':r['subset'],'task':r['task'],'hit':lookup[r['id']]} for r in rows]
    path=root/'results/per-example.json'
    if read(path)!=expected:raise ValueError('Original per-example strict outcomes/order differ from independent native scalars')
    return {'per_example':C.entry(path),'examples':15912,'subsets':dict(N.COUNTS),'strict_ties_fail':True}

def review_original(job,actor,*,accept_protocol=False):
    Q=facade(job);p=job['profile'];plan=checked(job['producer_plan'])
    if p['family']=='reranker':return Q.review(plan,p['model'],Path(p['score_root']),actor)
    if not accept_protocol:raise ValueError('Explicit acceptance of frozen disclosed native protocol differences required')
    return Q.review(plan,reports(job),actor,accept_protocol=True)

def make_original(job):
    Q=facade(job);p=job['profile'];plan=checked(job['producer_plan']);review=Path(job['outputs']['independent-review'])
    return Q.make(plan,p['model'],Path(p['score_root']),review) if p['family']=='reranker' else Q.make(plan,reports(job),review)
