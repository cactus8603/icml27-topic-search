"""[codex] CPU-only mechanical, fresh numerical review, then native admission.

The driver reads source/plan metadata only. The family interpreter worker
durably registers every prospective effect path before opening any artifact.
"""
from __future__ import annotations
import argparse,json,os,subprocess,sys
from pathlib import Path
from time import monotonic
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from codex.retrieval_matrix_native_analysis import contracts as C,workflow as W

def _same_identity(value,plan_path,commit,job):
    expected={'analysis_plan':C.entry(plan_path),'analysis_commit':commit,'job':job['id'],
        'producer_plan':job['producer_plan'],'producer_commit':job['producer_commit'],
        'model_spec_sha256':job['model_spec_sha256']}
    if any(value.get(k)!=v for k,v in expected.items()):raise ValueError('Analysis artifact belongs to a different frozen producer/model/plan')
    return expected

def identity(plan_path,commit,job):
    return {'analysis_plan':C.entry(plan_path),'analysis_commit':commit,'job':job['id'],
        'producer_plan':job['producer_plan'],'producer_commit':job['producer_commit'],
        'model_spec_sha256':job['model_spec_sha256']}

def _require_stage_read(plan_path,commit,value,paths):
    row=W.require_read(plan_path,commit,value['reading_started'])
    if row['actor']!=value['actor'] or row['job']!=value['job'] or row['producer_plan_sha256']!=value['producer_plan']['sha256']:
        raise ValueError('Reading receipt actor/job/producer binding differs')
    for path in paths:
        if row['paths'].get(path)!=C.sha(path):raise ValueError('Read-completion path/payload no longer matches stage evidence')
    return row

def verify_mechanical(plan_path,commit,job,manifest):
    path=Path(job['outputs']['mechanical']);value=W.read(path);_same_identity(value,plan_path,commit,job)
    expected_commands=[C.mechanical_argv(job['profile'],b) for b in job['profile']['benchmarks']] if job['profile']['family']!='reranker' else [C.mechanical_argv(job['profile'])]
    if (value.get('state')!='COMPLETE' or value.get('scope')!='native-analysis-original-mechanical'
            or value.get('commands')!=expected_commands or value.get('cpu')!=job['cpu_runtime']
            or value.get('immutable_raw')!=W.raw_snapshot(manifest)
            or value.get('reports')!={p:C.sha(p) for p in manifest['mechanical_results']}):
        raise ValueError('Original complete mechanical outputs/runtime/arrays changed')
    _require_stage_read(plan_path,commit,value,[*manifest['immutable_raw'],*manifest['mechanical_results'],str(path)])
    return value

def verify_independent(plan_path,commit,job,manifest,mechanical):
    path=Path(job['outputs']['independent-envelope']);value=W.read(path);_same_identity(value,plan_path,commit,job)
    W.independent_actor(job,value.get('actor'),mechanical['actor'],nonimplementer=value.get('declared_nonimplementer') is True)
    review=Path(job['outputs']['independent-review'])
    if (value.get('state')!='PASS' or value.get('scope')!='native-analysis-fresh-independent-envelope'
            or value.get('mechanical')!=C.entry(job['outputs']['mechanical'])
            or value.get('original_review')!=C.entry(review) or value.get('original_qualification_source')!=job['profile']['qualification_module']
            or value.get('per_example_audit')!=W.independent_details(job)
            or value.get('cpu')!=job['cpu_runtime'] or value.get('immutable_raw')!=W.raw_snapshot(manifest)
            or value.get('reports')!={p:C.sha(p) for p in manifest['mechanical_results']}):
        raise ValueError('Independent raw-array review is not complete and unchanged')
    original=W.read(review)
    native_schema=(original.get('scope')=='native-reranker-mcq-independent-recomputation' and
        original.get('independent_recomputation') is True and original.get('audited_arrays') is True) if job['profile']['family']=='reranker' else (
        original.get('schema')==2 and original.get('scope')=='native-p0-independent-recomputation')
    if original.get('reviewer')!=value['actor'] or original.get('state')!='PASS' or not native_schema:
        raise ValueError('Exact original successful nonproducer numerical review required')
    _require_stage_read(plan_path,commit,value,[*manifest['immutable_raw'],*manifest['mechanical_results'],str(path),str(review),job['outputs']['mechanical']])
    return value

def verify_completion(job,manifest):
    for path in manifest['completion']:
        if W.read(path).get('state')!='COMPLETE':raise ValueError('Every original benchmark must be completely encoded/scored first')
    if job['profile']['family']=='reranker':
        from codex.retrieval_matrix_rerank_native_p0 import runner as N
        score=Path(job['profile']['score_root']);N.require_smoke(W.read(Path(job['profile']['output_root'])/'smoke/smoke.json'),
            W.read(score/'identity.json'),W.read(W.checked(job['producer_plan'])))

def run_original(command):
    subprocess.run(command,cwd=ROOT,env=os.environ.copy(),check=True)

def run_worker(a):
    plan=W.source_gate(a.plan,a.freeze_commit)
    if a.job not in plan['jobs']:raise ValueError('Unknown exact native job')
    job=plan['jobs'][a.job];manifest=W.read(W.checked(job['manifest']))
    if a.stage=='independent':
        W.independent_actor(job,a.actor,'not-yet-read',nonimplementer=a.nonimplementer)
        if job['profile']['family']!='reranker' and not a.accept_disclosed_protocol:
            raise ValueError('Accept the frozen disclosed native protocol differences before reading effects')
    runtime=W.cpu_environment(job)
    outputs=job['outputs'];attempt=Path(job['stage_attempts'][a.stage]);failed=attempt.with_name(a.stage+'-failed.json')
    stage_outputs={'mechanical':['mechanical'],'independent':['independent-review','independent-envelope'],
        'qualify':['qualification','admission']}[a.stage]
    if any(Path(outputs[name]).exists() for name in stage_outputs):raise ValueError('Preserve existing stage outputs; no overwrite or automatic retry')
    if a.stage=='mechanical' and any(Path(p).exists() for p in manifest['mechanical_results']):raise ValueError('Original mechanical reports already exist; require a separately reviewed continuation')
    paths=[*manifest['immutable_raw'],*manifest['mechanical_results'],*manifest['producer_mutable_analysis_ledger'],
        *(outputs[name] for name in stage_outputs)]
    if a.stage!='mechanical':paths.append(outputs['mechanical'])
    if a.stage=='qualify':paths.extend([outputs['independent-review'],outputs['independent-envelope']])
    base=identity(a.plan,a.freeze_commit,job)
    W.write_new(attempt,{**base,'stage':a.stage,'actor':a.actor,'state':'STARTED','technical_only':True})
    try:
        with W.session(a.plan,a.freeze_commit,job,a.actor,paths) as reading:
            if a.stage=='independent':
                # Read only the score-free mechanical wrapper to learn its actor;
                # reject self-review before hashing/opening any model array/report.
                mechanical_header=W.read(outputs['mechanical']);_same_identity(mechanical_header,a.plan,a.freeze_commit,job)
                W.independent_actor(job,a.actor,mechanical_header['actor'],nonimplementer=a.nonimplementer)
            began=monotonic();before=W.raw_snapshot(manifest);verify_completion(job,manifest)
            if a.stage=='mechanical':
                commands=[C.mechanical_argv(job['profile'],b) for b in job['profile']['benchmarks']] if job['profile']['family']!='reranker' else [C.mechanical_argv(job['profile'])]
                for command in commands:run_original(command)
                if W.raw_snapshot(manifest)!=before:raise ValueError('Original analysis mutated producer raw arrays/checkpoints/cost ledger')
                value={**base,'scope':'native-analysis-original-mechanical','state':'COMPLETE','actor':a.actor,
                    'commands':commands,'cpu':runtime,'immutable_raw':before,
                    'reports':{p:C.sha(p) for p in manifest['mechanical_results']},'reading_started':reading.start,
                    'analysis_cpu_wall_seconds':monotonic()-began,'gpu_cost':manifest['cost'],'user_decision':'PENDING'}
                W.write_new(outputs['mechanical'],value)
            else:
                mechanical=verify_mechanical(a.plan,a.freeze_commit,job,manifest)
                if a.stage=='independent':
                    W.independent_actor(job,a.actor,mechanical['actor'],nonimplementer=a.nonimplementer)
                    report_hashes={p:C.sha(p) for p in manifest['mechanical_results']}
                    original=W.review_original(job,a.actor,accept_protocol=a.accept_disclosed_protocol)
                    per_example=W.independent_details(job)
                    if W.raw_snapshot(manifest)!=before or report_hashes!={p:C.sha(p) for p in manifest['mechanical_results']}:
                        raise ValueError('Independent review mutated original evidence')
                    W.write_new(outputs['independent-review'],original)
                    W.write_new(outputs['independent-envelope'],{**base,'scope':'native-analysis-fresh-independent-envelope',
                        'state':original['state'],'actor':a.actor,'declared_nonimplementer':True,'mechanical':C.entry(outputs['mechanical']),
                        'original_review':C.entry(outputs['independent-review']),'original_qualification_source':job['profile']['qualification_module'],
                        'per_example_audit':per_example,
                        'cpu':runtime,'immutable_raw':before,'reports':report_hashes,'reading_started':reading.start,
                        'analysis_cpu_wall_seconds':monotonic()-began,'user_decision':'PENDING'})
                else:
                    verify_independent(a.plan,a.freeze_commit,job,manifest,mechanical)
                    original=W.make_original(job)
                    if original.get('scope')!=job['profile']['qualification_scope'] or original.get('model_spec_sha256')!=job['model_spec_sha256']:
                        raise ValueError('Native qualification scope/full model spec differs')
                    W.write_new(outputs['qualification'],original)
                    admitted=False
                    if original['state']=='PASS':
                        actual=W.facade(job).admit(C.entry(outputs['qualification']),W.model_spec(job))
                        if actual!=original:raise ValueError('Native admission changed the original qualification receipt')
                        admitted=True
                    if W.raw_snapshot(manifest)!=before or {p:C.sha(p) for p in manifest['mechanical_results']}!=mechanical['reports']:
                        raise ValueError('Qualification mutated original raw/results evidence')
                    W.write_new(outputs['admission'],{**base,'scope':'native-analysis-original-admission-record','actor':a.actor,
                        'native_numeric_state':original['state'],'downstream_admitted':admitted,
                        'original_receipt':C.entry(outputs['qualification']),'original_facade':job['profile']['qualification_module'],
                        'reading_started':reading.start,'analysis_cpu_wall_seconds':monotonic()-began,
                        'cpu':runtime,'user_decision':'PENDING',
                        'old_p3_scope_compatible':False if job['profile']['family']=='reranker' else None})
        W.require_read(a.plan,a.freeze_commit,reading.start)
    except BaseException as error:
        W.write_new(failed,{**base,'stage':a.stage,'actor':a.actor,'state':'FAILED','error_type':type(error).__name__,
            'preserve_all_existing_outputs':True,'user_decision':'PENDING'})
        raise

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=W.STAGES)
    p.add_argument('--plan',required=True,type=Path);p.add_argument('--freeze-commit',required=True)
    p.add_argument('--job',required=True);p.add_argument('--actor',required=True)
    p.add_argument('--nonimplementer',action='store_true');p.add_argument('--accept-disclosed-protocol',action='store_true')
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS);a=p.parse_args()
    if a.worker:run_worker(a);return
    plan=W.source_gate(a.plan,a.freeze_commit)
    if a.job not in plan['jobs']:raise ValueError('Unknown exact native job')
    recipe=plan['jobs'][a.job]['cpu'];env=os.environ.copy();env.update(recipe['environment'])
    command=[recipe['python'],'-B',str(Path(__file__).resolve()),*sys.argv[1:],'--worker']
    subprocess.run(command,cwd=ROOT,env=env,check=True)
if __name__=='__main__':main()
