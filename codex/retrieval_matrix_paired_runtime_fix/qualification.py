"""[codex] New paired runtime gate; original paired scalar/event verification."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import argparse
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from codex.retrieval_matrix_paired_runtime_fix import contracts as C

spec=importlib.util.spec_from_file_location('paired_repair_private_qualification',ROOT/'codex/retrieval_matrix_paired_p0/qualification.py')
ENGINE=importlib.util.module_from_spec(spec);spec.loader.exec_module(ENGINE)
original_validate=ENGINE.validate_plan
SCOPE='official-paired-runtime-fix-p0-qualified'


def validate_plan(plan):
    C.validate(plan)
    # Validate unchanged scientific fields against the original canonical paired
    # plan. Evidence verification below still uses the unmodified new plan.
    projected=deepcopy(plan);base=C.parent(plan['model_key'])
    projected['dependencies'],projected['source_files']=base['dependencies'],base['source_files']
    original_validate(projected)
    return plan


ENGINE.validate_plan=validate_plan
ENGINE.ENGINE.validate_plan=validate_plan


def verify_report(plan,plan_sha,name,report,*,verify_images=True):
    if report.get('identity',{}).get('runtime',{}).get('runtime_fix')!=C.runtime_identity(plan):
        raise ValueError('Paired evidence must bind the new runtime and signature policy')
    return ENGINE.verify_report(plan,plan_sha,name,report,verify_images=verify_images)


def review(plan_path,reports,reviewer,*,accept_protocol=False):
    plan=C.read(plan_path);C.require_review(plan);validate_plan(plan)
    for path in reports.values():
        if C.read(path).get('identity',{}).get('runtime',{}).get('runtime_fix')!=C.runtime_identity(plan):
            raise ValueError('Original paired P0 cannot substitute for the repaired runtime')
    value=ENGINE.review(plan_path,reports,reviewer,accept_protocol=accept_protocol)
    value['runtime_fix']=C.runtime_identity(plan);value['runtime_fix_qualification_source']=C.entry(Path(__file__))
    return value


def make(plan_path,reports,review_path):
    declared=C.read(review_path);actual=review(plan_path,reports,declared.get('reviewer'),accept_protocol=True)
    if actual!=declared or actual['state']!='PASS':raise ValueError('Independent repaired paired review no longer matches')
    value=ENGINE.make(plan_path,reports,review_path)
    value['scope']=SCOPE;value['runtime_fix']=actual['runtime_fix'];value['runtime_fix_qualification_source']=C.entry(Path(__file__))
    return value


def admit(receipt_entry,model_spec):
    value=C.read(C.checked(receipt_entry))
    if value.get('scope')!=SCOPE:raise ValueError('Old paired or Google qualification cannot qualify this new runtime')
    actual=make(C.checked(value['p0_plan']),{key:C.checked(item) for key,item in value['reports'].items()},C.checked(value['independent_review']))
    if value!=actual or actual['state']!='PASS' or actual['model_spec_sha256']!=C.digest(model_spec):
        raise ValueError('New paired runtime requires unchanged complete independent qualification')
    return actual


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('review','qualify'))
    for name in ('plan','reports-root','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--reviewer');p.add_argument('--accept-disclosed-protocol',action='store_true');p.add_argument('--independent-review',type=Path)
    a=p.parse_args();plan=C.read(a.plan);reports={key:a.reports_root/key/'results/report.json' for key in plan['benchmarks']}
    if a.action=='review':value=review(a.plan,reports,a.reviewer,accept_protocol=a.accept_disclosed_protocol)
    else:
        if a.independent_review is None:p.error('qualify requires --independent-review')
        value=make(a.plan,reports,a.independent_review)
    C.write(a.out,value);print(value['state'])


if __name__=='__main__':main()
