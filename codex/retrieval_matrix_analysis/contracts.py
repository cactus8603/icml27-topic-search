"""[codex] Immutable analysis contracts; producer bytes and GPU work are untouched."""
from __future__ import annotations
import hashlib,importlib,json,os,re,subprocess,sys
from importlib.metadata import version
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
MARKER=b'<!-- APPEND-ONLY READ LOG -->\n'
MCQ_COMMIT='25f7bd515875a0c698202d0047d7b7952fd6846b'
FIX_COMMIT='99431594893bba94337b3befd10a7a483ba7374f'
OWN=('contracts.py','reading.py','recompute.py','launcher.py','build_plan.py','test_analysis.py','run.sh','PREREG.md','README.md')
OUTPUT_ROOT=ROOT/'dl/retrieval-matrix-analysis/results'
COUNTS={'cola':210,'replace_att':788,'replace_obj':1652,'replace_rel':1406,'swap_att':666,'swap_obj':245,'coco_mcq':5914,'voc_mcq':5031}
SCPP=('replace_att','replace_obj','replace_rel','swap_att','swap_obj')
TASKS={'cola_text_choice':['cola'],'scpp':list(SCPP),'negbench_mcq':['coco_mcq','voc_mcq']}

def now():return datetime.now(ZoneInfo('Asia/Taipei')).strftime('%Y-%m-%d %H:%M:%S%z')
def digest(x):return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text())
def safe(p):
    p=Path(p).resolve()
    if not p.is_relative_to(ROOT) or any(x in str(p) for x in ('retrieval-independent-panel','/test-queries')):
        raise ValueError('Analysis paths must remain in the declared workspace and outside sealed tests')
    return p
def entry(p):
    p=safe(p);return {'path':str(p),'sha256':sha(p)}
def checked(e):
    p=safe(e['path'])
    if sha(p)!=e['sha256']:raise ValueError('Frozen artifact changed: '+str(p))
    return p
def write_new(p,x):
    p=safe(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:
        f.write(json.dumps(x,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
def git_bytes(commit,p):
    if not re.fullmatch('[0-9a-f]{40}',commit or ''):raise ValueError('Full immutable source commit required')
    return subprocess.check_output(['git','-C',str(ROOT),'show',commit+':'+str(safe(p).relative_to(ROOT))])
def frozen(p,commit):
    p=safe(p)
    if git_bytes(commit,p)!=p.read_bytes():raise ValueError('Producer source/plan differs from its frozen bytes: '+str(p))

def implementation_authors(family):
    if family=='core-mcq':return ['/root/official_p0']
    authors=['/root','/root/matrix_runner']
    if family.endswith('mcmr'):authors.append('/root/public_data')
    if family.startswith('qwen2-fixed'):authors.append('/root/official_p0/p3_contract_review')
    return authors

def prereg_body(path):
    body=Path(path).read_bytes()
    if body.count(MARKER)!=1:raise ValueError('One immutable reading-log boundary is required')
    prefix,tail=body.split(MARKER)
    return prefix+MARKER,tail

def validate_job(job):
    plan_path=checked(job['producer_plan']);p=read(plan_path)
    if p.get('state')!='FROZEN' or p['models'][job['model']]!=job['model_spec']:
        raise ValueError('Analysis must use the full exact frozen producing model')
    frozen(plan_path,job['producer_commit'])
    folder='retrieval_matrix_rerank' if job['family']=='core-mcq' else ('retrieval_matrix_core_runtime_fix' if job['family'].startswith('core-') else 'retrieval_matrix_qwen2_runtime_fix')
    if plan_path.parent!=ROOT/'codex'/folder:raise ValueError('Producer family must use its own exact original entry')
    for rel,expected in p['source_files'].items():
        path=safe(ROOT/rel);frozen(path,job['producer_commit'])
        if sha(path)!=expected:raise ValueError('Original source hash differs')
    if job['family']=='core-mcq':
        if job['producer_commit']!=MCQ_COMMIT or job['benchmarks']!=list(TASKS) or p.get('counts')!=COUNTS or p.get('pair_batch')!=4:
            raise ValueError('Keep the complete original three-MCQ contract')
        if set(job['runs'])!={'mcq'} or safe(job['runs']['mcq']).name!='main':raise ValueError('Original MCQ scores reside in main')
    elif job['family'] in {'core-fixed-embedding','qwen2-fixed-embedding'}:
        if job['producer_commit']!=FIX_COMMIT or set(job['runs'])!={'coco','negbench','flickr'} or set(p['datasets'])!={'coco','negbench','flickr'}:
            raise ValueError('Retain all three distinct frozen native P0 datasets')
    elif job['family'] in {'core-fixed-mcmr','qwen2-fixed-mcmr'}:
        if job['producer_commit']!=FIX_COMMIT or set(job['runs'])!={'mcmr'} or p.get('phase')!='MCMR-native-MMEB-v2':
            raise ValueError('MCMR is the explicit native renderer difference-only experiment')
    else:raise ValueError('Unsupported analysis producer family')
    for value in job['runs'].values():
        if not safe(value).is_relative_to(ROOT/'dl'):raise ValueError('Raw outputs must remain under dl')
    return p

def validate_plan(plan):
    if plan.get('schema')!=1 or plan.get('phase')!='CPU-mechanical-then-independent-analysis-v1':raise ValueError('Unknown analysis schema')
    if plan.get('decision_order')!=['original-mechanical-script','nonimplementer-independent-recomputation','user-final-decision']:
        raise ValueError('Keep mechanical → independent → user decision order')
    expected={'required_distinct_benchmarks':2,'absolute_tolerance':.01,'retain_all_three':True,'numeric_agreement_tolerance':1e-12,'automatic_user_decision':False}
    if plan.get('criteria')!=expected:raise ValueError('Analysis cannot revise the frozen numerical criteria')
    keys={'mcq-core2','mcq-core8'}|{f'fixed-{phase}-{model}' for phase in ('embedding','mcmr') for model in ('core2','core8','qwen2','qwen8','gme2','vlm2vec2')}
    if set(plan['jobs'])!=keys:raise ValueError('Retain all14 prespecified original/repaired analysis jobs')
    schedule=read(checked(plan['schedule_reference']))
    tasks={t['id']:t for t in schedule['tasks']}
    if set(plan['source_files'])!={str((HERE/n).relative_to(ROOT)) for n in OWN}:raise ValueError('All own source/preregistration files must be bound')
    for rel,h in plan['source_files'].items():
        path=ROOT/rel;actual=hashlib.sha256(prereg_body(path)[0]).hexdigest() if path.name=='PREREG.md' else sha(path)
        if actual!=h:raise ValueError('Analysis source or immutable preregistration changed: '+rel)
    for key,j in plan['jobs'].items():
        if j['id']!=key:raise ValueError('Job key changed')
        task=tasks[key];expected_plan={'path':str(safe(task['plan'])),'sha256':task['plan_sha256']}
        root=safe(task['output'])
        if key.startswith('mcq-'):family='core-mcq';runs={'mcq':str(root/'main')};benchmarks=list(TASKS)
        else:
            mcmr=key.startswith('fixed-mcmr-');core='core_runtime_fix' in str(expected_plan['path'])
            family=('core-fixed-' if core else 'qwen2-fixed-')+('mcmr' if mcmr else 'embedding')
            runs={n:str(root/n) for n in (['mcmr'] if mcmr else ['coco','negbench','flickr'])};benchmarks=list(runs)
        if (j['model']!=task['model'] or j['producer_commit']!=task['freeze_commit'] or j['producer_plan']!=expected_plan
                or j['family']!=family or j['runs']!=runs or j['benchmarks']!=benchmarks):
            raise ValueError('Analysis model/ref/plan/family/run paths must exactly match the frozen central schedule')
        if (not set(implementation_authors(family))<=set(j['implementation_authors'])
                or '/root/official_p0' not in j['analysis_authors'] or j['producer_actor']!='[codex] central-task='+key):
            raise ValueError('Preserve all actual producer/backend/runtime-fix and analysis authors')
        validate_job(j)
    return plan

def source_gate(plan_path,plan,commit):
    if plan.get('state')!='FROZEN' or plan.get('reviewed') is not True:raise ValueError('Analysis requires an independently reviewed frozen plan')
    validate_plan(plan);frozen(plan_path,commit)
    from codex.retrieval_matrix_analysis import reading
    for relative in plan['source_files']:
        p=ROOT/relative
        if p.name=='PREREG.md':
            original=git_bytes(commit,p)
            if original!=prereg_body(p)[0]:raise ValueError('Frozen preregistration body differs')
            reading.records(p,expected_plan=sha(plan_path),expected_commit=commit)
        else:frozen(p,commit)
    changed=subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain','--',str(HERE.relative_to(ROOT))],text=True).splitlines()
    for line in changed:
        if line[3:]!=str((HERE/'PREREG.md').relative_to(ROOT)):raise ValueError('Only validated append-only reading records may differ from the analysis freeze')

def require_reviewer(job,reviewer,mechanical_actor):
    excluded=set(job['implementation_authors'])|set(job['analysis_authors'])|{job['producer_actor'],mechanical_actor}
    if not reviewer or reviewer in excluded:raise ValueError('Independent reviewer must differ from producer, implementers and mechanical actor')

def result_dir(job):return OUTPUT_ROOT/job['id']
def runtime_snapshot():
    import numpy as np
    core=importlib.import_module('numpy._core._multiarray_umath')
    return {'python':str(Path(sys.executable).resolve()),'python_version':sys.version,
        'numpy':{'path':str(Path(np.__file__).resolve()),'version':np.__version__,'distribution_version':version('numpy'),
            'sha256':sha(np.__file__),'array_binary':entry(core.__file__)}}
def independent_environment(plan):
    actual=runtime_snapshot()
    if actual!=plan['independent_environment']:raise ValueError('Independent CPU array runtime differs from the frozen actual import')
    return actual
def report_paths(job):return {k:safe(v)/'results/report.json' for k,v in job['runs'].items()}
def completion_paths(job):return [safe(v)/('complete.json' if job['family']=='core-mcq' else 'index-meta.json') for v in job['runs'].values()]
def require_complete(job):
    # This function is called only after READ_STARTED was durably registered.
    for path in completion_paths(job):
        if not path.is_file() or read(path).get('state')!='COMPLETE':raise ValueError('All original benchmark outputs must be complete before analysis')

def original_command(job,benchmark):
    p=str(checked(job['producer_plan']));out=str(safe(job['runs'][benchmark]));model=job['model'];ref=job['producer_commit']
    if job['family']=='core-mcq':
        return [str(ROOT/'dl/retrieval-matrix-rerank/env/bin/python'),'-B',str(ROOT/'codex/retrieval_matrix_rerank/runner.py'),'--stage','analyze','--plan',p,'--model',model,'--out',out,'--freeze-commit',ref]
    folder='retrieval_matrix_core_runtime_fix' if job['family'].startswith('core-') else 'retrieval_matrix_qwen2_runtime_fix'
    py='retrieval3090-torch214' if folder.endswith('core_runtime_fix') else 'retrieval3090'
    mode='mcmr' if job['family'].endswith('mcmr') else 'embedding'
    cmd=[str(ROOT/'dl/retrieval3090/envs'/py/'bin/python'),'-B',str(ROOT/'codex'/folder/'runner.py'),'--mode',mode,'--stage','analyze','--plan',p,'--model',model,'--out',out,'--freeze-commit',ref]
    if mode=='embedding':cmd+=['--dataset',benchmark]
    return cmd

def cpu_environment(job):
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','HF_HOME':str(ROOT/'hf_home'),'HF_MODULES_CACHE':str(ROOT/'hf_home/modules'),
        'TMPDIR':str(ROOT/'hf_home/tmp'),'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','PYTHONDONTWRITEBYTECODE':'1',
        'TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','TZ':'Asia/Taipei'}
    if job['family'].startswith('core-fixed'):
        names=['dl/retrieval-matrix/deps/core-transformers457','dl/retrieval-matrix/deps/core-p0']
    elif job['family'].startswith('qwen2-fixed'):
        names=['dl/retrieval-matrix-qwen2-runtime-fix/deps','dl/retrieval-matrix/deps/qwen2-transformers451','dl/retrieval-matrix/deps/qwen2-flash','dl/retrieval-matrix/deps/core-p0']
    else:names=[]
    env['PYTHONPATH']=':'.join([*(str(ROOT/n) for n in names),str(ROOT)])
    return env
