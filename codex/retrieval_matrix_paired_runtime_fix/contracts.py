"""[codex] Exact paired P0 parents, added einops, and stable technical signatures."""
from copy import deepcopy
import importlib
from importlib.metadata import version
from pathlib import Path
import subprocess
from codex.retrieval_matrix_native_p0.evidence import checked,digest,entry,read,sha,write

ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
PARENT_COMMIT='54f544ce54bca82be3cc8b3c72662b354c6d3e40'
MODELS=('clic_b32','clic_l14','clic_clips_l14')
PARENTS={key:'codex/retrieval_matrix_paired_p0/p0-'+key+'.json' for key in MODELS}
OVERLAY_REL='codex/retrieval_matrix_qwen2_runtime_fix/overlay.json'
OVERLAY_COMMIT='99431594893bba94337b3befd10a7a483ba7374f'
OVERLAY=ROOT/'dl/retrieval-matrix-qwen2-runtime-fix/deps'
BATCH='P0-paired-CLIC-einops-stable-signature-v1'
SIGNATURE_POLICY='remove-volatile-function-repr-address-only-v1'
OUTPUT_ROOT=ROOT/'dl/retrieval-matrix-paired-runtime-fix/runs'
PRESERVED=('schema','owner','phase','purpose','model_key','model_spec','benchmarks','qualification','settings','external_sources')
OWN=('contracts.py','models.py','runner.py','qualification.py','build_plan.py','preflight.py','test_contracts.py',
     'run.sh','central_task.sh','PREREG.md','README.md','dependencies.json')


def original(relative,commit):
    path=ROOT/relative
    if subprocess.check_output(['git','-C',str(ROOT),'show',commit+':'+relative])!=path.read_bytes():
        raise ValueError('Original frozen parent/overlay bytes changed: '+relative)
    return read(path)


def parent(key):
    if key not in PARENTS:raise ValueError('Only the three exact paired CLIC parents are supported')
    return original(PARENTS[key],PARENT_COMMIT)


def overlay():
    value=original(OVERLAY_REL,OVERLAY_COMMIT)
    if value.get('package')!='einops' or value.get('version')!='0.8.2' or value.get('overlay')!=str(OVERLAY):
        raise ValueError('Only the exact frozen einops overlay is permitted')
    actual={str(p.relative_to(ROOT)) for p in OVERLAY.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
    if actual!=set(value['files']):raise ValueError('Overlay file coverage changed')
    for relative,expected in value['files'].items():
        path=(ROOT/relative).resolve()
        if not path.is_relative_to(OVERLAY) or sha(path)!=expected:raise ValueError('Overlay payload changed')
    checked(value['copy_receipt']);return value


def dependency_spec():
    old=read(checked(parent(MODELS[0])['dependencies']));extra=overlay();value=deepcopy(old)
    if 'einops' in old['modules'] or set(old['files'])&set(extra['files']):raise ValueError('Only add the missing package')
    value['files'].update(extra['files']);value['modules']['einops']=deepcopy(extra['modules']['einops'])
    value['pythonpath']=[str(OVERLAY),*old['pythonpath']]
    value['runtime_fix']={'parent_dependencies':parent(MODELS[0])['dependencies'],'overlay':entry(ROOT/OVERLAY_REL),
        'overlay_freeze_commit':OVERLAY_COMMIT,'added_distribution':'einops==0.8.2'}
    return value


def runtime_identity(plan):
    key=plan['model_key']
    return {'batch':BATCH,'parent_plan':entry(ROOT/PARENTS[key]),'parent_freeze_commit':PARENT_COMMIT,
        'dependencies':plan['dependencies'],'overlay':entry(ROOT/OVERLAY_REL),'overlay_freeze_commit':OVERLAY_COMMIT,
        'signature_policy':SIGNATURE_POLICY}


def actual_modules(dependencies):
    actual={}
    for name in dependencies['modules']:
        module=importlib.import_module(name);distribution={'open_clip':'open-clip-torch','qwen_vl_utils':'qwen-vl-utils'}.get(name,name)
        actual[name]={'path':str(Path(module.__file__).resolve()),'actual_version':getattr(module,'__version__',None),
            'distribution_version':version(distribution)}
    return actual


def validate(plan,*,imports=False):
    key=plan['model_key'];base=parent(key)
    for field in PRESERVED:
        if plan.get(field)!=base[field]:raise ValueError('Paired repair must preserve original '+field)
    if (plan.get('runtime_batch')!=BATCH or plan.get('runtime_fix')!=runtime_identity(plan)
            or checked(plan['dependencies'])!=HERE/'dependencies.json'):
        raise ValueError('New paired runtime provenance differs')
    dependencies=read(checked(plan['dependencies']))
    if dependencies!=dependency_spec() or base['dependencies']!=parent(MODELS[0])['dependencies']:
        raise ValueError('Original paired dependencies plus only einops are required')
    required={**base['source_files'],**{str((HERE/name).relative_to(ROOT)):sha(HERE/name) for name in OWN},
        OVERLAY_REL:sha(ROOT/OVERLAY_REL)}
    if plan.get('source_files')!=required:raise ValueError('Full original paired and new facade source table is required')
    for relative,expected in dependencies['files'].items():
        path=(ROOT/relative).resolve()
        if not path.is_relative_to(ROOT/'dl') or sha(path)!=expected:raise ValueError('Frozen dependency changed: '+relative)
    if imports and actual_modules(dependencies)!=dependencies['modules']:raise ValueError('Actual repaired imports differ')
    return dependencies


def require_review(plan):
    if plan.get('state')!='FROZEN' or plan.get('reviewed') is not True:raise ValueError('New paired runtime must be independently reviewed and frozen')


def output_path(value):
    path=Path(value).resolve()
    if path==OUTPUT_ROOT or not path.is_relative_to(OUTPUT_ROOT):raise ValueError('Preserve original outputs; use a new paired runtime-fix child')
    return path
