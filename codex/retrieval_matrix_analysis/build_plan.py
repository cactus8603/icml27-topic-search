"""[codex] Bind frozen producer plans and prospective paths without reading results."""
import argparse,copy,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from codex.retrieval_matrix_analysis import contracts as C

SCHEDULE=ROOT/'codex/retrieval_matrix_dispatch_batches/central-v3.json'

def build():
    schedule=C.read(SCHEDULE);jobs={}
    for task in schedule['tasks']:
        key=task['id']
        if not (key in {'mcq-core2','mcq-core8'} or key.startswith(('fixed-embedding-','fixed-mcmr-'))):continue
        pp=C.safe(task['plan']);producer=C.read(pp);model=task['model'];root=C.safe(task['output'])
        if key.startswith('mcq-'):
            family='core-mcq';runs={'mcq':str(root/'main')};benchmarks=list(C.TASKS)
        else:
            mcmr=key.startswith('fixed-mcmr-');core='core_runtime_fix' in str(pp)
            family=('core-fixed-' if core else 'qwen2-fixed-')+('mcmr' if mcmr else 'embedding')
            runs={name:str(root/name) for name in (['mcmr'] if mcmr else ['coco','negbench','flickr'])}
            benchmarks=list(runs)
        job={'id':key,'family':family,'producer_plan':C.entry(pp),'producer_commit':task['freeze_commit'],
            'model':model,'model_spec':copy.deepcopy(producer['models'][model]),'runs':runs,'benchmarks':benchmarks,
            'implementation_authors':C.implementation_authors(family),'analysis_authors':['/root/official_p0'],
            'producer_actor':'[codex] central-task='+key,'user_final_decision':'PENDING'}
        C.validate_job(job);jobs[key]=job
    if len(jobs)!=14:raise ValueError('Retain2 original CORE MCQ,6 repaired embeddings and6 native MCMR jobs')
    source={str((C.HERE/n).relative_to(ROOT)):(hashlib.sha256(C.prereg_body(C.HERE/n)[0]).hexdigest() if n=='PREREG.md' else C.sha(C.HERE/n)) for n in C.OWN}
    env=C.runtime_snapshot()
    result={'owner':'[codex]','schema':1,'phase':'CPU-mechanical-then-independent-analysis-v1','state':'REVIEW_REQUIRED','reviewed':False,
        'source_files':source,'schedule_reference':C.entry(SCHEDULE),'jobs':jobs,'independent_environment':env,
        'decision_order':['original-mechanical-script','nonimplementer-independent-recomputation','user-final-decision'],
        'criteria':{'required_distinct_benchmarks':2,'absolute_tolerance':.01,'retain_all_three':True,'numeric_agreement_tolerance':1e-12,'automatic_user_decision':False},
        'production_and_effect_reads_performed':False,'review_note':'Review implementation-author roles before freezing; no main outputs were read to create this plan.'}
    C.validate_plan(result);return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    result=build();C.write_new(a.out,result);print({'state':result['state'],'jobs':len(result['jobs']),'plan_sha256':C.sha(a.out)})

if __name__=='__main__':main()
