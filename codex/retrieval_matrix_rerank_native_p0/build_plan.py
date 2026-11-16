"""[codex] Prepare an auditable MCQ plan; never manufacture model qualification."""
import argparse,copy,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from codex.retrieval_matrix_rerank_native_p0 import contracts as C

OWN=('contracts.py','adapters.py','runner.py','qualification.py','build_plan.py','cpu_probe.py','test_native.py','run.sh','PREREG.md','README.md','SOURCE_AUDIT.md','protocols.json','backend_specs.draft.json')
EXECUTED=('codex/retrieval_matrix/runner.py','codex/retrieval_matrix/models.py','codex/retrieval_matrix/metrics.py',
    'codex/retrieval_scaling/runner.py','codex/retrieval_matrix_rerank/backends.py','codex/retrieval_matrix_rerank/metrics.py')


def build(specs_path,protocols_path,probe_path):
    specs=C.R.read(specs_path);protocols=C.R.read(protocols_path);probe=C.R.read(C.checked(C.entry(probe_path)))
    if probe.get('state')!='PASS' or probe.get('weights_loaded') is not False or probe.get('cuda_initialized') is not False or probe['backend_specs']!=C.entry(specs_path):
        raise ValueError('Actual pinned official CPU processor/template probes required')
    if set(specs['models'])!=set(protocols['variants']) or set(probe['variants'])!=set(specs['models']):
        raise ValueError('Retain all six explicit variants including missing-target native protocols')
    variants={}
    for key,descriptor in protocols['variants'].items():
        model=copy.deepcopy(specs['models'][key])
        if model['backend']!=descriptor['backend'] or model.get('verified_protocol') is not False:
            raise ValueError('Draft must retain unapproved backend review status')
        variants[key]={**copy.deepcopy(descriptor),'model':model,'model_spec_sha256':C.R.digest(model)}
    sources={relative:C.R.sha(ROOT/relative) for relative in EXECUTED}
    for name in OWN:
        relative='codex/retrieval_matrix_rerank_native_p0/'+name;sources[relative]=C.R.sha(ROOT/relative)
    original=ROOT/'codex/retrieval_matrix_rerank/plan.core-mcq.v1.json'
    result={'owner':'[codex]','schema':1,'phase':'P0-native-reranker-mcq-v1','purpose':'OFFICIAL_RERANKER_QUALIFICATION',
        'state':'REVIEW_REQUIRED','reviewed':False,'variants':variants,'source_files':sources,'data_reference':C.entry(original),
        'backend_reference':C.entry(specs_path),'protocols':C.entry(protocols_path),'cpu_probe':C.entry(probe_path),
        'external_sources':protocols['sources'],'environment':copy.deepcopy(specs['environment']),
        'settings':{'storage_examples':32,'swift_pair_batch':32,'native_pair_batch':1,'jina_groups':'COLA2+2/SCPP3/Neg4',
            'smoke_absolute_tolerance':.01,'required_distinct_benchmarks':2,'absolute_tolerance':.01},
        'not_run':True,'production_guard':'Independent protocol review + FROZEN/reviewed source commit + central assigned GPU. Variants with fewer than2 exact published MCQ targets are blocked even after source freeze.',
        'review_action':'Reviewer may approve verified_protocol in the model spec and refresh its model_spec_sha256; this approval is not an empirical P0 PASS.'}
    result['environment']['modules']=copy.deepcopy(probe['modules'])
    for variant in variants.values():C.load_examples(result,variant)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--probe',type=Path,default=ROOT/'dl/retrieval-matrix-rerank-native-p0/cpu-probe.json');a=p.parse_args()
    if a.out.exists():raise ValueError('Refuse to overwrite any planned execution')
    result=build(Path(__file__).parent/'backend_specs.draft.json',Path(__file__).parent/'protocols.json',a.probe)
    C.R.atomic_json(a.out,result);print(json.dumps({'state':result['state'],'path':str(a.out),'sha256':C.R.sha(a.out)}))


if __name__=='__main__':main()
