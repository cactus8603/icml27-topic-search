"""[codex] A separately frozen einops runtime; unchanged native SigLIP2 science."""
from __future__ import annotations
from copy import deepcopy
import importlib
from importlib.metadata import version
from pathlib import Path
import subprocess

from codex.retrieval_matrix_native_p0.evidence import checked, digest, entry, read, sha, write

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
PARENT_REL='codex/retrieval_matrix_native_p0/p0-siglip2.json'
PARENT_COMMIT='387d4fa5293967705b93aeba235cb9ffe643cc42'
OVERLAY_REL='codex/retrieval_matrix_qwen2_runtime_fix/overlay.json'
OVERLAY_COMMIT='99431594893bba94337b3befd10a7a483ba7374f'
OVERLAY=ROOT/'dl/retrieval-matrix-qwen2-runtime-fix/deps'
OUTPUT_ROOT=ROOT/'dl/retrieval-matrix-native-runtime-fix/runs'
BATCH='P0-native-SigLIP2-einops-overlay-v1'
OWN=('contracts.py','runner.py','qualification.py','build_plan.py','preflight.py',
     'test_contracts.py','run.sh','central_task.sh','PREREG.md','README.md','dependencies.json')
PRESERVED=('schema','owner','phase','purpose','model_key','model_spec','benchmarks','qualification','settings','external_sources')


def git_original(relative,commit):
    path=ROOT/relative
    original=subprocess.check_output(['git','-C',str(ROOT),'show',commit+':'+relative])
    if path.read_bytes()!=original:raise ValueError('Frozen parent/overlay source changed: '+relative)
    return read(path)


def parent():return git_original(PARENT_REL,PARENT_COMMIT)


def overlay():
    value=git_original(OVERLAY_REL,OVERLAY_COMMIT)
    if value.get('package')!='einops' or value.get('version')!='0.8.2' or value.get('overlay')!=str(OVERLAY):
        raise ValueError('Only the reviewed einops0.8.2 overlay is authorized')
    actual={str(p.relative_to(ROOT)) for p in OVERLAY.rglob('*')
        if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
    if actual!=set(value['files']):raise ValueError('Reused overlay file coverage changed')
    for relative,expected in value['files'].items():
        path=(ROOT/relative).resolve()
        if not path.is_relative_to(OVERLAY) or sha(path)!=expected:raise ValueError('Reused overlay bytes changed')
    checked(value['copy_receipt'])
    return value


def dependency_spec():
    base=parent();old=read(checked(base['dependencies']));extra=overlay()
    result=deepcopy(old)
    if 'einops' in result['modules'] or set(result['files']) & set(extra['files']):
        raise ValueError('Overlay must only add the previously missing package')
    result['files'].update(extra['files']);result['modules']['einops']=deepcopy(extra['modules']['einops'])
    result['pythonpath']=[str(OVERLAY),*old['pythonpath']]
    result['runtime_fix']={'parent_dependencies':base['dependencies'],'overlay':entry(ROOT/OVERLAY_REL),
        'overlay_freeze_commit':OVERLAY_COMMIT,'added_distribution':'einops==0.8.2'}
    return result


def runtime_identity(plan):
    return {'batch':BATCH,'parent_plan':entry(ROOT/PARENT_REL),'parent_freeze_commit':PARENT_COMMIT,
        'dependencies':plan['dependencies'],'overlay':entry(ROOT/OVERLAY_REL),'overlay_freeze_commit':OVERLAY_COMMIT}


def actual_modules(dependencies):
    result={}
    for name in dependencies['modules']:
        module=importlib.import_module(name)
        distribution={'open_clip':'open-clip-torch','qwen_vl_utils':'qwen-vl-utils'}.get(name,name)
        result[name]={'path':str(Path(module.__file__).resolve()),'actual_version':getattr(module,'__version__',None),
            'distribution_version':version(distribution)}
    return result


def validate(plan,*,imports=False):
    base=parent()
    for name in PRESERVED:
        if plan.get(name)!=base[name]:raise ValueError('Runtime repair must preserve original '+name)
    if (plan.get('runtime_batch')!=BATCH or plan.get('runtime_fix')!=runtime_identity(plan)
            or checked(plan['dependencies'])!=HERE/'dependencies.json'):
        raise ValueError('Native runtime fix provenance/manifest changed')
    dependencies=read(checked(plan['dependencies']))
    if dependencies!=dependency_spec():raise ValueError('Only the exact original dependencies plus einops may be used')
    required={**base['source_files'],**{str((HERE/name).relative_to(ROOT)):sha(HERE/name) for name in OWN},
        OVERLAY_REL:sha(ROOT/OVERLAY_REL)}
    if plan.get('source_files')!=required:raise ValueError('Complete unchanged parent and new facade sources are required')
    for relative,expected in dependencies['files'].items():
        path=(ROOT/relative).resolve()
        if not path.is_relative_to(ROOT/'dl') or sha(path)!=expected:raise ValueError('Frozen dependency payload changed: '+relative)
    if imports and actual_modules(dependencies)!=dependencies['modules']:
        raise ValueError('Actual repaired module path/version/distribution differs')
    return dependencies


def require_review(plan):
    if plan.get('state')!='FROZEN' or plan.get('reviewed') is not True:
        raise ValueError('A new independent runtime review and frozen plan are required')


def output_path(value):
    path=Path(value).resolve()
    if path==OUTPUT_ROOT or not path.is_relative_to(OUTPUT_ROOT):
        raise ValueError('Preserve old failed runs; use an isolated native runtime-fix output')
    return path
