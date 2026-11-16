"""[codex] Original mechanical analysis, then a different actor's raw-array audit."""
from __future__ import annotations
import argparse,fcntl,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from codex.retrieval_matrix_analysis import contracts as C,reading,recompute as I

def read_paths(job):
    paths=[]
    for name,value in job['runs'].items():
        out=C.safe(value)
        if job['family']=='core-mcq':
            paths.extend(out/n for n in ('complete.json','identity.json','results/report.json','results/per-example.json'))
            arrays=list(C.COUNTS)
        else:
            paths.extend(out/n for n in ('index-meta.json','provenance.json','processor.json','results/report.json','results/per-query.json'))
            arrays=['corpus','queries','native-top100'] if name=='mcmr' else ['image','text','official-similarity','official-topk']+(['official-i2t'] if name!='negbench' else [])
            if name!='mcmr':paths.append(out/'order.npy')
        for a in arrays:paths.extend([out/(a+'.npy'),out/(a+'.state.json')])
    return paths

def raw_snapshot(job):
    return {str(p):C.sha(p) for p in read_paths(job) if '/results/' not in str(p.relative_to(C.ROOT/'dl'))}

def mechanical(plan_path,plan,commit,job,actor):
    target=C.result_dir(job);target.mkdir(parents=True,exist_ok=True);receipt=target/'mechanical.json'
    if receipt.exists():raise ValueError('Preserve an existing mechanical receipt; do not rerun')
    with reading.ReadSession(plan_path,commit,job,actor,read_paths(job)) as session:
        C.require_complete(job)
        if any(p.exists() for p in C.report_paths(job).values()):raise ValueError('Original report already exists; do not overwrite or silently adopt it')
        before=raw_snapshot(job);commands=[]
        for benchmark in job['runs']:
            argv=C.original_command(job,benchmark);log=target/('mechanical-'+benchmark+'.log')
            with log.open('x') as stream:
                result=subprocess.run(argv,cwd=C.ROOT,env=C.cpu_environment(job),stdout=stream,stderr=subprocess.STDOUT)
            commands.append({'benchmark':benchmark,'argv':argv,'returncode':result.returncode,'log':C.entry(log)})
            if result.returncode:
                C.write_new(target/'mechanical-failed.json',{'owner':'[codex]','state':'FAILED','job':job['id'],'actor':actor,
                    'commands':commands,'automatic_retry':False,'analysis_plan':C.entry(plan_path),'reading_started':session.start})
                raise ValueError('Original mechanical analysis failed; preserve its log and any partial report')
        after=raw_snapshot(job)
        if before!=after:raise ValueError('Original raw score/cache bytes changed during CPU analysis')
        value={'schema':1,'owner':'[codex]','state':'COMPLETE','scope':'original-mechanical-analysis',
            'analysis_plan':C.entry(plan_path),'analysis_commit':commit,'job':job['id'],'actor':actor,
            'producer_plan':job['producer_plan'],'producer_commit':job['producer_commit'],'model_spec_sha256':C.digest(job['model_spec']),
            'commands':commands,'raw_artifacts':after,'reports':{k:C.entry(p) for k,p in C.report_paths(job).items()},
            'reading_started':session.start,'independent_review':'PENDING','user_final_decision':'PENDING',
            'cuda_visible_devices':'','gpu_production':False}
        C.write_new(receipt,value)
    return value

def independent(plan_path,plan,commit,job,actor,accept_role):
    if not accept_role:raise ValueError('Reviewer must explicitly attest to the frozen nonimplementer role')
    runtime=C.independent_environment(plan)
    target=C.result_dir(job);out=target/'independent-review.json'
    if out.exists():raise ValueError('Never overwrite an independent review')
    paths=[*read_paths(job),target/'mechanical.json']
    with reading.ReadSession(plan_path,commit,job,actor,paths) as session:
        mechanical_receipt=C.read(target/'mechanical.json')
        C.require_reviewer(job,actor,mechanical_receipt.get('actor'))
        if (mechanical_receipt.get('state')!='COMPLETE' or mechanical_receipt.get('analysis_plan')!=C.entry(plan_path)
                or mechanical_receipt.get('analysis_commit')!=commit or mechanical_receipt.get('job')!=job['id']
                or mechanical_receipt.get('producer_plan')!=job['producer_plan']
                or mechanical_receipt.get('producer_commit')!=job['producer_commit']
                or mechanical_receipt.get('model_spec_sha256')!=C.digest(job['model_spec'])):
            raise ValueError('Independent review must follow this exact frozen mechanical stage')
        reading.require_completed(C.HERE/'PREREG.md',mechanical_receipt['reading_started'],plan_sha=C.sha(plan_path),commit=commit)
        for p,h in mechanical_receipt['raw_artifacts'].items():
            if C.sha(p)!=h:raise ValueError('Raw producer artifact changed after mechanical analysis')
        if mechanical_receipt.get('reports')!={k:C.entry(p) for k,p in C.report_paths(job).items()}:
            raise ValueError('Mechanical report bytes changed before independent review')
        result=I.mcq(job) if job['family']=='core-mcq' else (I.mcmr(job) if job['family'].endswith('mcmr') else I.embedding(job))
        value={'schema':1,'owner':'[codex]','state':'PASS' if result['agreement'] else 'FAILED',
            'scope':'official-reranker-mcq' if job['family']=='core-mcq' else ('official-native-mcmr-difference-only' if job['family'].endswith('mcmr') else 'official-native-p0'),
            'reviewer':actor,'producer':job['producer_actor'],'mechanical_actor':mechanical_receipt['actor'],
            'analysis_plan':C.entry(plan_path),'analysis_commit':commit,'mechanical':C.entry(target/'mechanical.json'),
            'plan_sha256':job['producer_plan']['sha256'],'producer_freeze_commit':job['producer_commit'],'model':job['model'],
            'model_spec_sha256':C.digest(job['model_spec']),'independent_recomputation':True,'audited_arrays':True,
            'recomputation_source':C.entry(Path(I.__file__)),'reading_started':session.start,
            'independent_environment':runtime,
            'agreement_is_only':'Independent raw-array recomputation agrees with original mechanical metrics; it is not empirical qualification or user approval.',
            'user_final_decision':'PENDING','gpu_production':False,**result}
        C.write_new(out,value)
    return value

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('mechanical','independent'))
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--freeze-commit',required=True)
    p.add_argument('--job',required=True);p.add_argument('--actor',required=True);p.add_argument('--accept-independent-role',action='store_true');a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU analysis requires an explicitly empty CUDA_VISIBLE_DEVICES')
    plan=C.read(a.plan);C.source_gate(a.plan,plan,a.freeze_commit)
    if a.job not in plan['jobs']:raise ValueError('Only frozen allowlisted jobs may be analyzed')
    job=plan['jobs'][a.job];target=C.result_dir(job);target.mkdir(parents=True,exist_ok=True)
    with (target/'writer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        result=(mechanical(a.plan,plan,a.freeze_commit,job,a.actor) if a.stage=='mechanical'
                else independent(a.plan,plan,a.freeze_commit,job,a.actor,a.accept_independent_role))
    print({'owner':'[codex]','stage':a.stage,'state':result['state'],'job':a.job,'user_final_decision':'PENDING'})

if __name__=='__main__':main()
