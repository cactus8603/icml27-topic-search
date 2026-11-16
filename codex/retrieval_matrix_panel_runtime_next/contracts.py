"""[codex] Explicit repaired MLLM parents, genuine P0 admission before panels."""
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path
import subprocess

from codex.retrieval_matrix_panel import qualification as Q

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
REF='99431594893bba94337b3befd10a7a483ba7374f'
PHASE='P2-qualified-repaired-mllm-next'
OUTPUT_ROOT=ROOT/'dl/retrieval-matrix-panel-runtime-next/runs'
FAMILIES={
    'core-qwen3-9943159':{'path':'codex/retrieval_matrix_core_runtime_fix/p0-qwen3-fixed-v1.json',
        'sha256':'0f74cb526e08952e05cc178e9fc2fba91d773701db7d92049a71661a142049b1',
        'models':('core2','core8','qwen2','qwen8'),
        'python':'dl/retrieval3090/envs/retrieval3090-torch214/bin/python'},
    'qwen2-einops-9943159':{'path':'codex/retrieval_matrix_qwen2_runtime_fix/p0-qwen2-einops-v1.json',
        'sha256':'07a9b6859e17d12e99a00439327381655ec22b1369e2d27a9fa4a5f8a5ea3053',
        'models':('gme2','vlm2vec2'),'python':'dl/retrieval3090/envs/retrieval3090/bin/python'},
}
OWN=('contracts.py','build_plan.py','runner.py','test_contracts.py','run.sh','PREREG.md','README.md')
PANEL_REF='17e990ba6294b1962f67d0c9a33bccf291e43fbd'
PANEL=('data.py','data_audit.py','qualification.py','build_plan.py','runner.py','PREREG.md','DEV_SPEC.md','development-sources.json')


def frozen(relative,commit):
    path=(ROOT/relative).resolve()
    if not path.is_relative_to(ROOT):raise ValueError('Frozen source escapes the repository')
    if subprocess.check_output(['git','-C',str(ROOT),'show',commit+':'+str(path.relative_to(ROOT))])!=path.read_bytes():
        raise ValueError('Source differs from its literal frozen parent: '+relative)
    return path


def parent(family):
    if family not in FAMILIES:raise ValueError('Only the two explicit repaired MLLM families are allowed')
    profile=FAMILIES[family];path=frozen(profile['path'],REF)
    if Q.sha(path)!=profile['sha256']:raise ValueError('Exact repaired P0 parent SHA differs')
    value=Q.read(path)
    # CORE's fixed original has its independent source review at this exact
    # commit, but no reviewed boolean. Do not invent a field in its old plan.
    if (value.get('state')!='FROZEN' or set(value['models'])!=set(profile['models'])
            or (family=='qwen2-einops-9943159' and value.get('reviewed') is not True)):
        raise ValueError('Exact independently reviewed repaired P0 required')
    for relative,expected in value['source_files'].items():
        if Q.sha(frozen(relative,REF))!=expected:raise ValueError('Original qualified source SHA changed')
    return value


def producer_sources(base,family):
    """The original P0 source_gate result, including its own plan and prereg."""
    result=deepcopy(base['source_files'])
    for relative in (FAMILIES[family]['path'],base['prereg']):
        actual=Q.sha(frozen(relative,REF))
        if relative in result and result[relative]!=actual:
            raise ValueError('Original P0 declared source SHA differs from its fixed Git bytes')
        result[relative]=actual
    return result


def evidence_identity(qualified,base,family,key):
    """Only called after genuine Q.admit; never a substitute qualification."""
    if set(qualified.get('reports',{}))!=set(Q.DATASETS):
        raise ValueError('All three original qualified P0 reports are required')
    sources=producer_sources(base,family)
    spec=base['models'][key]
    signature_fields={'implementation':spec['official'],'loader':spec.get('loader',{}),
        'processor_configuration':spec.get('processor',{}),'batches':spec['batches']}
    signatures=[];environments=[]
    for name,reference in qualified['reports'].items():
        report_path=Q.checked(reference);report=Q.read(report_path)
        meta=report['provenance']['index'];identity=meta['identity']
        expected={'schema':1,'owner':'[codex]','model':key,'dataset':name,
            'plan_sha256':FAMILIES[family]['sha256'],'freeze_commit':REF,
            'model_spec':spec,'settings':base['settings'],'source_files':sources}
        if (meta.get('state')!='COMPLETE'
                or any(identity.get(field)!=value for field,value in expected.items())):
            raise ValueError('Qualified P0 producer identity/source/ref must equal the complete fixed original')
        environment=identity.get('environment',{})
        if any(environment.get(field)!=value for field,value in spec.get('environment',{}).items()):
            raise ValueError('Qualified actual P0 environment differs from its original complete model')
        processor=report_path.parent.parent/'processor.json'
        if Q.sha(processor)!=meta['processor_sha256']:raise ValueError('Qualified actual processor bytes changed')
        signature=Q.read(processor)
        if any(signature.get(field)!=value for field,value in signature_fields.items()):
            raise ValueError('Qualified P0 processor signature differs from its fixed implementation/loader/batches')
        signatures.append(signature);environments.append(environment)
    if (len(signatures)!=3 or any(value!=signatures[0] for value in signatures)
            or any(value!=environments[0] for value in environments)):
        raise ValueError('All three qualified P0 benchmarks must share actual processor and runtime')
    return signatures[0],environments[0]


def admit_parent(family,key,receipt):
    base=parent(family)
    if key not in FAMILIES[family]['models']:raise ValueError('Model belongs to another repaired family')
    qualified=Q.admit(receipt,base['models'][key])
    expected=Q.entry(ROOT/FAMILIES[family]['path'])
    if (qualified.get('state')!='PASS' or qualified.get('model')!=key
            or qualified.get('p0_plan')!=expected or qualified.get('p0_plan_sha256')!=expected['sha256']
            or qualified.get('model_spec_sha256')!=Q.digest(base['models'][key])):
        raise ValueError('Qualification must belong to this exact repaired parent, model and complete spec')
    signature,environment=evidence_identity(qualified,base,family,key)
    modules=environment.get('modules',{})
    if not modules:raise ValueError('Qualified actual runtime module receipts are required')
    if family=='core-qwen3-9943159':
        marker=base['models'][key]['runtime_compatibility']
        if environment.get('runtime_compatibility')!=marker or signature.get('runtime_compatibility')!=marker:
            raise ValueError('Qualified CORE actual class/runtime did not apply the exact repair')
    else:
        from codex.retrieval_matrix_qwen2_runtime_fix import contracts as repair
        expected_fix={'overlay_manifest':base['runtime_fix']['overlay'],
            'parent_freeze_commit':base['runtime_fix']['parent_freeze_commit']}
        manifest=Q.read(Q.checked(base['runtime_fix']['overlay']))
        wanted=manifest['modules']['einops']
        if environment.get('runtime_fix')!=expected_fix or modules.get('einops')!=wanted:
            raise ValueError('Qualified Qwen2 actual environment must bind the original einops overlay')
    return base,qualified,signature,environment


def admit(plan):
    if (plan.get('schema')!=1 or plan.get('phase')!=PHASE or plan.get('state') not in {'REVIEW_REQUIRED','FROZEN'}
            or plan.get('retriever_family')!='MLLM' or len(plan.get('models',{}))!=1
            or set(plan['models'])!=set(plan.get('qualification',{}))):
        raise ValueError('Exactly one genuinely qualified repaired MLLM is required')
    key=next(iter(plan['models']))
    base,qualified,signature,environment=admit_parent(plan['family_adapter'],key,plan['qualification'][key])
    if (plan.get('parent_p0_plan')!=qualified['p0_plan'] or plan['models'][key]!=base['models'][key]
            or plan.get('settings')!=base['settings'] or plan.get('parent_native_settings')!=base['settings']
            or plan.get('dependencies')!=base['dependencies'] or plan.get('parent_model_signature')!=signature
            or plan.get('parent_runtime')!=environment or plan.get('fixed_baseline_test') is not True
            or plan.get('ranking')!={'dtype':'float32','backend':'CPU dot','ties':'stable-gallery-order','main_k':100,'top_ks':[50,100,200]}):
        raise ValueError('P2 must preserve its exact qualified full spec/settings/runtime/signature/dependencies')
    return base,qualified


def source_files(base):
    result=deepcopy(base['source_files'])
    for name in PANEL:
        relative='codex/retrieval_matrix_panel/'+name;path=frozen(relative,PANEL_REF)
        if relative in result and result[relative]!=Q.sha(path):raise ValueError('Parent panel source changed')
        result[relative]=Q.sha(path)
    for name in OWN:
        relative='codex/retrieval_matrix_panel_runtime_next/'+name;result[relative]=Q.sha(HERE/name)
    return result


def check_sources(plan,base):
    if plan.get('source_files')!=source_files(base):raise ValueError('Complete unchanged P0/P2 source table required')


def dependencies(base,family,*,imports):
    for relative,expected in base['dependencies']['files'].items():
        path=(ROOT/relative).resolve()
        if not path.is_relative_to(ROOT/'dl/retrieval-matrix/deps') or Q.sha(path)!=expected:
            raise ValueError('Original qualified dependency payload changed')
    for name,expected in base['dependencies']['packages'].items():
        if version(name)!=expected:raise ValueError('Qualified dependency distribution differs: '+name)
    if family=='qwen2-einops-9943159':
        from codex.retrieval_matrix_qwen2_runtime_fix import contracts as repair
        repair.validate(base,imports=imports)
    elif imports:
        import torch,transformers
        if (torch.__version__!=base['dependencies']['packages']['torch']
                or transformers.__version__!=base['dependencies']['packages']['transformers']):
            raise ValueError('Actual CORE module/distribution version differs')


def runtime_matches(actual,qualified):
    # The centrally assigned GPU UUID may differ; every other original field,
    # including module paths/versions, CUDA, device family and repair, is exact.
    left,right=deepcopy(actual),deepcopy(qualified)
    left.pop('visible_devices',None);right.pop('visible_devices',None)
    if left!=right:raise ValueError('Actual P2 runtime differs from the qualified P0 runtime')


def output_path(value):
    path=Path(value).resolve()
    if path==OUTPUT_ROOT or not path.is_relative_to(OUTPUT_ROOT):raise ValueError('Use an isolated new repaired MLLM P2 output')
    return path
