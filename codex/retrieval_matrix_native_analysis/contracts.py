"""[codex] Frozen producer contracts and source-only native P0 inventory."""
from __future__ import annotations
import hashlib,importlib.util,json,subprocess,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
SHARED_REF='7b4855f5d1b7f6e6c1f3850b8a2a9e41b5ac82ed'
SCHEDULE_REF='3a234a83586b8c253d4690b6638243b71cd22b7a'
SCHEDULE_REL='codex/retrieval_matrix_dispatch_batches/central-v4.json'
SIGLIP_REF='f3f787eeb193aad8dcc6f2ee43e9609ddf5b47eb'
PAIRED_REF='5675ef59cee99fa6d28e12f07501eeb157c321bd'
RERANK_REF='54f544ce54bca82be3cc8b3c72662b354c6d3e40'
MARKER=b'<!-- APPEND-ONLY READ LOG -->\n'
SHARED_FILES=('contracts.py','reading.py','PREREG.md')
SCOPES={'siglip2':'official-native-runtime-fix-p0-qualified',
    'paired':'official-paired-runtime-fix-p0-qualified','reranker':'official-native-reranker-mcq-qualified'}

def digest(value):return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def safe(path):
    path=Path(path).resolve()
    if not path.is_relative_to(ROOT):raise ValueError('Source preparation must remain in the workspace')
    return path
def entry(path):
    path=safe(path);return {'path':str(path),'sha256':sha(path)}
def read_header(path):
    # Root's already frozen analysis may append reading records concurrently.
    # Do not inspect that foreign tail or require its complete bytes to stay fixed.
    result=b''
    with Path(path).open('rb') as stream:
        for line in stream:
            result+=line
            if line==MARKER:return result
    raise ValueError('Shared preregistration lacks its immutable reading boundary')
def frozen(relative,ref,*,immutable_prereg=False):
    path=safe(ROOT/relative)
    expected=subprocess.check_output(['git','-C',str(ROOT),'show',ref+':'+relative])
    actual=read_header(path) if immutable_prereg else path.read_bytes()
    if immutable_prereg:
        if expected.count(MARKER)!=1:raise ValueError('Shared frozen preregistration boundary changed')
        expected=expected.split(MARKER)[0]+MARKER
    if actual!=expected:raise ValueError('Source differs from its explicit immutable commit: '+relative)
    return {'path':str(path),'sha256':hashlib.sha256(expected).hexdigest(),'freeze_commit':ref,
        'immutable_body_only':immutable_prereg}

def shared_sources():
    return {'codex/retrieval_matrix_analysis/'+name:frozen('codex/retrieval_matrix_analysis/'+name,SHARED_REF,
        immutable_prereg=name=='PREREG.md') for name in SHARED_FILES}

def family(task):
    key=task['id']
    if key=='native-siglip2-einops-v1':return 'siglip2',SIGLIP_REF,'retrieval_matrix_native_runtime_fix'
    if key.startswith('native-paired-'):return 'paired',PAIRED_REF,'retrieval_matrix_paired_runtime_fix'
    if key.startswith('native-rerank-'):return 'reranker',RERANK_REF,'retrieval_matrix_rerank_native_p0'
    raise ValueError('This preparation supports only the three explicit new native families')

def profiles():
    shared=shared_sources();schedule=frozen(SCHEDULE_REL,SCHEDULE_REF)
    tasks=json.loads(Path(schedule['path']).read_text())['tasks'];result={}
    for task in tasks:
        if not (task['id']=='native-siglip2-einops-v1' or task['id'].startswith(('native-paired-','native-rerank-'))):continue
        name,ref,folder=family(task);pp=safe(task['plan'])
        if pp.parent!=ROOT/'codex'/folder or task['freeze_commit']!=ref:raise ValueError('Central source family/ref differs')
        frozen(str(pp.relative_to(ROOT)),ref);plan=json.loads(pp.read_text())
        if plan.get('state')!='FROZEN' or plan.get('reviewed') is not True or sha(pp)!=task['plan_sha256']:
            raise ValueError('Only exact approved source plans may define this preparation')
        for relative,expected in plan['source_files'].items():
            pinned=frozen(relative,ref)
            if pinned['sha256']!=expected:raise ValueError('Native producer complete source table changed')
        model=task['model'];qual=ROOT/'codex'/folder/'qualification.py'
        p={'id':task['id'],'family':name,'producer_plan':entry(pp),'producer_commit':ref,'model':model,
            'output_root':task['output'],'qualification_module':entry(qual),'qualification_scope':SCOPES[name]}
        if name=='reranker':
            variant=plan['variants'][model]
            if (model not in {'qwen2','qwen8','jina_m0'} or variant['model']['verified_protocol'] is not True
                    or sum(t['verified_protocol_match'] for t in variant['targets'].values())<2):
                raise ValueError('Native Qwen/pointwise variants without two exact targets remain ineligible')
            p.update(model_spec_sha256=digest(variant['model']),variant_spec_sha256=digest(variant),
                cola_rule=variant['cola_rule'],benchmarks=['COLA','SugarCrepe++','NegBench'],
                counts={'examples':15912,'pairs':58891,'subsets':8},score_root=str(safe(task['output'])/'scores'))
        else:
            if plan['model_key']!=model:raise ValueError('Model key differs from its full native plan')
            p.update(model_spec_sha256=digest(plan['model_spec']),benchmarks=list(plan['benchmarks']),
                counts={k:{'images':v['data']['expected_images'],'texts':v['data']['expected_texts']} for k,v in plan['benchmarks'].items()},
                primary_metric_names={k:list(v['primary_metrics']) for k,v in plan['benchmarks'].items()})
        result[p['id']]=p
    if len(result)!=7:raise ValueError('Retain1 SigLIP2,3 paired CLIC and3 exact-target native rerank variants')
    return {'schema':1,'owner':'[codex]','state':'SOURCE_INVENTORY','execution_plan':False,
        'shared_analysis_ref':SHARED_REF,'shared_sources':shared,'schedule':schedule,'profiles':result,
        'excluded_missing_targets':['qwen_native2','qwen_native8','qwen_instruct8'],
        'execution_separately_defined_by':'workflow.py; independently reviewed committed approved plan required',
        'main_outputs_read':False,'gpu_production':False,'qualification_emitted':False}

def _valid_profile(profile):
    actual=profiles()['profiles']
    if actual.get(profile.get('id'))!=profile:raise ValueError('Source-only profile is not the exact frozen central/producer contract')

def mechanical_argv(profile,benchmark=None):
    """Return the exact original analyze argv; deliberately never execute it."""
    _valid_profile(profile);plan=profile['producer_plan']['path'];out=profile['output_root']
    if profile['family']=='reranker':
        if benchmark is not None:raise ValueError('Native MCQ mechanical analysis retains all three benchmarks together')
        return [str(ROOT/'dl/retrieval-matrix-rerank/env/bin/python'),'-B',
            str(ROOT/'codex/retrieval_matrix_rerank_native_p0/runner.py'),'--stage','analyze','--plan',plan,
            '--variant',profile['model'],'--out',profile['score_root'],'--freeze-commit',profile['producer_commit']]
    if benchmark not in profile['benchmarks']:raise ValueError('Keep the family original benchmark names and denominators')
    return ['bash',str(Path(profile['qualification_module']['path']).parent/'run.sh'),'analyze',
        '--plan',plan,'--out',out,'--benchmark',benchmark,'--freeze-commit',profile['producer_commit']]

def qualification_argv(profile,stage,output,*,reviewer=None,independent_review=None,accept_protocol=False):
    """Build family-specific original review/qualify commands without coercing scopes."""
    _valid_profile(profile)
    if stage not in {'review','qualify'}:raise ValueError('Use original review then qualify')
    py=ROOT/('dl/retrieval-matrix-rerank/env/bin/python' if profile['family']=='reranker' else 'dl/retrieval3090/envs/retrieval3090/bin/python')
    argv=[str(py),'-B',profile['qualification_module']['path'],stage,'--plan',profile['producer_plan']['path'],'--out',str(safe(output))]
    if profile['family']=='reranker':argv+=['--variant',profile['model'],'--run',profile['score_root']]
    else:argv+=['--reports-root',profile['output_root']]
    if stage=='review':
        if not reviewer:raise ValueError('The eventual nonimplementer identity must be explicit')
        argv+=['--reviewer',reviewer]
        if profile['family']!='reranker':
            if not accept_protocol:raise ValueError('Original native port/mapping differences require explicit independent acceptance')
            argv+=['--accept-disclosed-protocol']
    else:
        if independent_review is None:raise ValueError('Original qualification needs its exact independent review')
        argv+=['--independent-review',str(safe(independent_review))]
    return argv

def synthetic_read_session(prereg,anchor,plan_path,artifact_paths):
    """Reuse fixed reading code only on isolated synthetic artifacts, never real effects."""
    paths=[prereg,anchor,plan_path,*artifact_paths]
    if any(not safe(p).is_relative_to(ROOT/'hf_home/tmp') for p in paths):
        raise ValueError('Source-preparation reading tests are restricted to isolated synthetic temporary paths')
    shared_sources()
    spec=importlib.util.spec_from_file_location('native_analysis_frozen_reading_'+SHARED_REF,ROOT/'codex/retrieval_matrix_analysis/reading.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    session=module.ReadSession(plan_path,SHARED_REF,{'id':'synthetic-native-analysis','producer_plan':{'sha256':'0'*64}},
        'synthetic-only',artifact_paths,anchor_root=anchor)
    session.args['path']=safe(prereg)
    return session

if __name__=='__main__':
    # Source inventory only. This is intentionally not a formal analysis launcher.
    value=profiles();print(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2))
