"""[codex] Separate nonproducer raw-array recomputation and native MCQ admission."""
from __future__ import annotations
import argparse,copy,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from codex.retrieval_matrix_rerank_native_p0 import contracts as C,runner as N
R=C.R


def independent_metrics(rows,scores,variant):
    lookup={r['id']:r['score'] for r in scores}
    if len(lookup)!=len(scores) or set(lookup)!={p['id'] for r in rows for p in r['pairs']}:
        raise ValueError('Independent metrics need every original pair exactly once')
    subsets={}
    for name,count in C.COUNTS.items():
        examples=[r for r in rows if r['subset']==name]
        a=np.array([[lookup[p['id']] for p in r['pairs']] for r in examples],dtype=np.float64)
        if len(a)!=count or not np.isfinite(a).all():raise ValueError('Independent complete subset changed')
        if name=='cola':
            hits=(a[:,0]>a[:,2])&(a[:,3]>a[:,1]) if variant['cola_rule']=='cola_image_choice' else (a[:,0]>a[:,1])&(a[:,3]>a[:,2])
        elif name in C.SCPP:hits=np.all(a[:,:2]>a[:,2:3],axis=1)
        else:hits=np.all(a[:,0:1]>a[:,1:],axis=1)
        subsets[name]={'rows':count,'accuracy':int(hits.sum())/count}
    definitions={variant['cola_rule']:['cola'],'scpp':list(C.SCPP),'negbench_mcq':['coco_mcq','voc_mcq']}
    benchmarks={task:{'subsets':sorted(names),'rows':sum(C.COUNTS[n] for n in names),
        'macro_accuracy':sum(subsets[n]['accuracy'] for n in names)/len(names)} for task,names in definitions.items()}
    return {'rows':15912,'subsets':subsets,'benchmarks':benchmarks}


def equivalent(produced,actual):
    if produced.get('rows')!=actual['rows'] or set(produced.get('subsets',{}))!=set(actual['subsets']) or set(produced.get('benchmarks',{}))!=set(actual['benchmarks']):return False
    for name,row in actual['subsets'].items():
        other=produced['subsets'][name]
        if other.get('rows')!=row['rows'] or not np.isfinite(other.get('accuracy',np.nan)) or abs(other['accuracy']-row['accuracy'])>1e-12:return False
    for name,row in actual['benchmarks'].items():
        other=produced['benchmarks'][name]
        if (other.get('rows')!=row['rows'] or other.get('subsets')!=row['subsets']
                or not np.isfinite(other.get('macro_accuracy',np.nan)) or abs(other['macro_accuracy']-row['macro_accuracy'])>1e-12):return False
    return True


def review(plan_path,key,out,reviewer):
    plan=R.read(plan_path);N.validate(plan);variant=plan['variants'][key]
    identity=R.read(out/'identity.json')
    if (not reviewer or reviewer==identity.get('producer') or plan.get('state')!='FROZEN' or plan.get('reviewed') is not True
            or identity.get('plan_sha256')!=R.sha(plan_path) or identity.get('variant')!=key
            or identity.get('variant_spec')!=variant or identity.get('source_files')!=plan['source_files']
            or identity.get('data_reference')!=plan['data_reference']):
        raise ValueError('Independent reviewer must bind actual producer/model/protocol/source freeze')
    N.source_gate(plan_path,plan,identity['freeze_commit'],key)
    numpy_path=Path(np.__file__).resolve()
    independent_runtime={'path':str(numpy_path),'actual_version':np.__version__,'sha256':R.sha(numpy_path)}
    if independent_runtime!=plan['environment']['modules']['numpy']:
        raise ValueError('Independent native metric recomputation requires its frozen actual NumPy implementation')
    rows=C.load_examples(plan,variant,verify_images=True)
    scores=N.verify_complete(out,identity,rows,variant)
    actual=independent_metrics(rows,scores,variant);report=R.read(out/'results/report.json')
    agrees=(report.get('identity')==identity and report.get('complete_sha256')==R.sha(out/'complete.json')
        and report.get('native_cola_rule')==variant['cola_rule'] and equivalent(report,actual)
        and report.get('qualification')==C.assess(report,variant))
    return {'schema':1,'owner':'[codex]','scope':'native-reranker-mcq-independent-recomputation',
        'state':'PASS' if agrees else 'FAILED','reviewer':reviewer,'producer':identity['producer'],'variant':key,
        'plan':C.entry(plan_path),'complete':C.entry(out/'complete.json'),'report':C.entry(out/'results/report.json'),
        'original_identity_sha256':R.digest(identity),'model_spec_sha256':R.digest(variant['model']),
        'variant_spec_sha256':R.digest(variant),'independent_recomputation':True,'audited_arrays':True,
        'counts':copy.deepcopy(C.COUNTS),'pairs':58891,'recomputed':actual,'independent_numpy':independent_runtime,
        'recomputation_source':C.entry(Path(__file__))}


def make(plan_path,key,out,review_path):
    declared=R.read(review_path)
    # Reopen original immutable arrays/metadata and repeat the independent calculation.
    actual=review(plan_path,key,out,declared.get('reviewer'))
    if actual!=declared or actual['state']!='PASS':raise ValueError('Nonproducer original-array review no longer matches')
    plan=R.read(plan_path);variant=plan['variants'][key];numeric=C.assess(actual['recomputed'],variant)
    return {'schema':1,'owner':'[codex]','scope':'official-native-reranker-mcq-qualified',
        'state':'PASS' if numeric['state']=='NUMERIC_MATCH' else ('TARGETS_INCOMPLETE' if numeric['state']=='TARGETS_INCOMPLETE' else 'FAILED'),
        'variant':key,'model_spec_sha256':R.digest(variant['model']),'variant_spec_sha256':R.digest(variant),
        'plan':C.entry(plan_path),'independent_review':C.entry(review_path),'complete':actual['complete'],'report':actual['report'],
        'numeric':numeric,'all_benchmarks_retained':list(C.CANONICAL)}


def admit(receipt_entry,model_spec):
    receipt=R.read(C.checked(receipt_entry));plan_path=C.checked(receipt['plan']);review_path=C.checked(receipt['independent_review'])
    complete=C.checked(receipt['complete']);actual=make(plan_path,receipt['variant'],complete.parent,review_path)
    if actual!=receipt or actual['state']!='PASS' or actual['model_spec_sha256']!=R.digest(model_spec):
        raise ValueError('Unchanged exact native protocol and two independently qualified benchmarks required')
    return actual


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('review','qualify'))
    for name in ('plan','run','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--variant',required=True);p.add_argument('--reviewer');p.add_argument('--independent-review',type=Path);a=p.parse_args()
    if a.out.exists():raise ValueError('Do not overwrite an existing independent qualification record')
    result=review(a.plan,a.variant,a.run,a.reviewer) if a.action=='review' else make(a.plan,a.variant,a.run,a.independent_review)
    R.atomic_json(a.out,result);print(result['state'])


if __name__=='__main__':main()
