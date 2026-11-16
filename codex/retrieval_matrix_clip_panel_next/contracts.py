"""[codex] Explicit family-specific qualification and unchanged P0 forward."""
from copy import deepcopy
import importlib
from pathlib import Path
import re
import subprocess

from codex.retrieval_matrix_native_p0.evidence import checked, digest, entry, read, sha

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
PHASE='P2-qualified-fixed-clip-baselines-next'
PURPOSE='P2_QUALIFIED_FIXED_CLIP_BASELINES'
OUTPUT_ROOT=ROOT/'dl/retrieval-matrix-clip-panel-next/runs'
LEGACY_COMMIT='4311a9d7b7356a7e59a5e42be659c52fde20d43b'
ADAPTERS={
    'paired-clic-einops-v1':{
        'module':'codex.retrieval_matrix_paired_runtime_fix.qualification',
        'source':'codex/retrieval_matrix_paired_runtime_fix/qualification.py',
        'sha256':'19e4087b797ad7717712aff8c53a9c0fe300b3b9da4844ce1c62dd54bd6b3b5c',
        'commit':'5675ef59cee99fa6d28e12f07501eeb157c321bd',
        'models':('clic_b32','clic_l14','clic_clips_l14'),
        'runtime_module':'codex.retrieval_matrix_paired_runtime_fix.runner',
        'model_module':'codex.retrieval_matrix_paired_runtime_fix.models','model_class':'create',
        'scope':'official-paired-runtime-fix-p0-qualified'},
    'siglip2-einops-v1':{
        'module':'codex.retrieval_matrix_native_runtime_fix.qualification',
        'source':'codex/retrieval_matrix_native_runtime_fix/qualification.py',
        'sha256':'84849d3af50a3e5ee4ebfbe18c5ad06c4e1a0363b235c07d0eaba54c0a5c605a',
        'commit':'f3f787eeb193aad8dcc6f2ee43e9609ddf5b47eb',
        'models':('siglip2',),'runtime_module':'codex.retrieval_matrix_native_runtime_fix.runner',
        'model_module':'codex.retrieval_matrix_native_p0.models','model_class':'Siglip2',
        'scope':'official-native-runtime-fix-p0-qualified'},
}
OWN=('contracts.py','runner.py','build_plan.py','test_contracts.py','run.sh','PREREG.md','README.md')
LEGACY=('codex/retrieval_matrix_clip_panel/runner.py','codex/retrieval_matrix_clip_panel/build_plan.py',
        'codex/retrieval_matrix_clip_panel/contracts.py','codex/retrieval_matrix_clip_panel/PREREG.md')
COMMON=('codex/retrieval_matrix/runner.py','codex/retrieval_matrix/models.py','codex/retrieval_matrix/metrics.py',
        'codex/retrieval_scaling/runner.py','codex/retrieval_matrix_panel/data.py',
        'codex/retrieval_matrix_panel/data_audit.py','codex/retrieval_matrix_panel/qualification.py',
        'codex/retrieval_matrix_extra/runner.py','codex/retrieval_matrix_extra/adapters.py')


def adapter_reference(name):
    descriptor=ADAPTERS.get(name)
    if descriptor is None:raise ValueError('Qualification adapter is not in the constant allowlist')
    if not re.fullmatch(r'[0-9a-f]{40}',descriptor.get('commit') or ''):
        raise ValueError('Qualification adapter still awaits its independent frozen source commit')
    path=ROOT/descriptor['source']
    if sha(path)!=descriptor['sha256'] or subprocess.check_output(
            ['git','-C',str(ROOT),'show',descriptor['commit']+':'+descriptor['source']])!=path.read_bytes():
        raise ValueError('Allowlisted qualification facade differs from its frozen source')
    return {'name':name,'source':entry(path),'source_commit':descriptor['commit']}


def adapter(reference):
    name=reference.get('name')
    if reference!=adapter_reference(name):raise ValueError('Qualification adapter source entry changed')
    module=importlib.import_module(ADAPTERS[name]['module'])
    if Path(module.__file__).resolve()!=checked(reference['source']):
        raise ValueError('Actually imported qualification facade differs from the allowlisted source')
    return module


def registered_module(reference,field):
    adapter(reference);descriptor=ADAPTERS[reference['name']]
    name=descriptor[field];relative=name.replace('.','/')+'.py';path=ROOT/relative
    original=subprocess.check_output(['git','-C',str(ROOT),'show',descriptor['commit']+':'+relative])
    module=importlib.import_module(name)
    if Path(module.__file__).resolve()!=path.resolve() or path.read_bytes()!=original:
        raise ValueError('Actual qualified model/runtime module differs from its original frozen source')
    return module


def panel_settings(parent):
    settings=parent['settings']
    return {'main_k':100,'top_ks':[50,100,200],'block_queries':64,
        'batch_invariance_cosine_min':settings['smoke_cosine_min'],
        'batch_invariance_max_abs':settings['smoke_max_abs'],'seed':settings['seed'],'allow_tf32':False,
        'ranking_dtype':'float32','tie_rule':'stable-gallery-order'}


def model_signature(qualified):
    signatures=[read(checked(ref))['identity']['model_signature'] for ref in qualified['reports'].values()]
    if not signatures or any(value!=signatures[0] for value in signatures):
        raise ValueError('Every original qualified benchmark must bind the same actual class/processor/tokenizer')
    return signatures[0]


def binding(plan,qualified):
    if (plan.get('schema')!=1 or plan.get('state') not in {'REVIEW_REQUIRED','FROZEN'}
            or plan.get('phase')!=PHASE or plan.get('purpose')!=PURPOSE or plan.get('retriever_family')!='CLIP'
            or len(plan.get('models',{}))!=1 or set(plan['models'])!=set(plan.get('qualification',{}))):
        raise ValueError('Each new CLIP panel plan requires exactly one independently qualified parent model')
    reference=plan['qualification_adapter'];adapter(reference);registered=ADAPTERS[reference['name']]
    key=next(iter(plan['models']));spec=plan['models'][key]
    if key not in registered['models'] or spec.get('family')!='CLIP' or spec.get('variant')!=key:
        raise ValueError('This model requires its own exact family-specific qualification facade')
    if (qualified.get('state')!='PASS' or qualified.get('scope')!=registered['scope']
            or qualified.get('model')!=key or qualified.get('family')!='CLIP'
            or qualified.get('model_spec_sha256')!=digest(spec)
            or qualified.get('p0_plan')!=plan['parent_p0_plan']):
        raise ValueError('Qualification does not belong to this complete parent model/plan/family')
    parent=read(checked(qualified['p0_plan']))
    if (parent.get('model_spec')!=spec or parent.get('model_key')!=key
            or parent.get('settings')!=plan.get('parent_native_settings')
            or parent.get('dependencies')!=plan.get('dependencies')
            or plan.get('settings')!=panel_settings(parent)
            or plan.get('parent_model_signature')!=model_signature(qualified)):
        raise ValueError('P2 must preserve its qualified full spec, batches, precision, raw tokenizer and runtime')
    settings=parent['settings']
    if (settings['precision']!='float32' or settings['allow_tf32'] is not False
            or settings['quantization'] is not False or settings['offload'] is not False
            or spec['batches']['image']!=settings['image_batch'] or spec['batches']['text']!=settings['text_batch']):
        raise ValueError('Preserve the actual qualified full FP32 encoder batches and backend')
    if reference['name']=='paired-clic-einops-v1' and (settings['image_batch']!=1 or settings['text_batch']!=1):
        raise ValueError('New paired CLIC models retain author batch1')
    if plan.get('fixed_baseline_test') is not True:raise ValueError('COLA is authorized only as fixed baseline test')
    return {key:qualified}


def admit_models(plan):
    if len(plan.get('models',{}))!=1 or set(plan['models'])!=set(plan.get('qualification',{})):
        raise ValueError('Exactly one qualified parent and its complete evidence are required')
    key=next(iter(plan['models']));reference=plan['qualification_adapter']
    facade=adapter(reference)
    if key not in ADAPTERS[reference['name']]['models'] or plan['models'][key].get('family')!='CLIP':
        raise ValueError('This model requires its own exact family-specific qualification facade')
    # Rebuild real admission before any panel-data reader is called.
    qualified=facade.admit(plan['qualification'][key],plan['models'][key])
    return binding(plan,qualified)


def parent_sources(plan):
    parent=read(checked(plan['parent_p0_plan']))
    sources=deepcopy(parent['source_files'])
    for relative in [*COMMON,*LEGACY,*('codex/retrieval_matrix_clip_panel_next/'+name for name in OWN)]:
        path=ROOT/relative
        if relative in LEGACY and subprocess.check_output(['git','-C',str(ROOT),'show',LEGACY_COMMIT+':'+relative])!=path.read_bytes():
            raise ValueError('Frozen original CLIP panel implementation changed')
        if relative in sources and sources[relative]!=sha(path):raise ValueError('Qualified original source changed')
        sources[relative]=sha(path)
    for relative,expected in sources.items():
        if sha(ROOT/relative)!=expected:raise ValueError('P2 cannot loosen the original source freeze')
    return sources


def check_declared_sources(plan):
    if plan.get('source_files')!=parent_sources(plan):raise ValueError('P2 source table omits or substitutes executed source')


def model_class(reference):
    descriptor=ADAPTERS[reference['name']]
    return getattr(registered_module(reference,'model_module'),descriptor['model_class'])


def output_path(value):
    path=Path(value).resolve()
    if path==OUTPUT_ROOT or not path.is_relative_to(OUTPUT_ROOT):raise ValueError('Use an isolated new CLIP P2 output')
    return path
