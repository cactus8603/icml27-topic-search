"""[codex] New-runtime qualification; never borrow the original P0 evidence."""
from pathlib import Path
import argparse
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from codex.retrieval_matrix_native_runtime_fix import contracts as C
from codex.retrieval_matrix_native_p0 import qualification as Q

SCOPE='official-native-runtime-fix-p0-qualified'


def review(plan_path,reports,reviewer,*,accept_protocol=False):
    plan=C.read(plan_path);C.require_review(plan);C.validate(plan)
    expected=C.runtime_identity(plan)
    for path in reports.values():
        if C.read(path).get('identity',{}).get('runtime',{}).get('runtime_fix')!=expected:
            raise ValueError('Native evidence was not produced by this exact new dependency runtime')
    value=Q.review(plan_path,reports,reviewer,accept_protocol=accept_protocol)
    value['runtime_fix']=expected;value['runtime_fix_qualification_source']=C.entry(Path(__file__))
    return value


def make(plan_path,reports,review_path):
    declared=C.read(review_path)
    actual=review(plan_path,reports,declared.get('reviewer'),accept_protocol=True)
    if declared!=actual or actual['state']!='PASS':raise ValueError('Independent original-array runtime review no longer matches')
    value=Q.make(plan_path,reports,review_path)
    value['scope']=SCOPE;value['runtime_fix']=actual['runtime_fix']
    value['runtime_fix_qualification_source']=C.entry(Path(__file__))
    return value


def admit(receipt_entry,model_spec):
    value=C.read(C.checked(receipt_entry))
    if value.get('scope')!=SCOPE:raise ValueError('Old native P0 qualification cannot qualify this new runtime')
    actual=make(C.checked(value['p0_plan']),{k:C.checked(v) for k,v in value['reports'].items()},C.checked(value['independent_review']))
    if actual!=value or actual['state']!='PASS' or actual['model_spec_sha256']!=C.digest(model_spec):
        raise ValueError('New runtime requires its own unchanged independent numerical qualification')
    return actual


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('review','qualify'))
    for name in ('plan','reports-root','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--reviewer');p.add_argument('--accept-disclosed-protocol',action='store_true');p.add_argument('--independent-review',type=Path)
    a=p.parse_args();plan=C.read(a.plan);reports={name:a.reports_root/name/'results/report.json' for name in plan['benchmarks']}
    if a.action=='review':value=review(a.plan,reports,a.reviewer,accept_protocol=a.accept_disclosed_protocol)
    else:
        if a.independent_review is None:p.error('qualify requires --independent-review')
        value=make(a.plan,reports,a.independent_review)
    C.write(a.out,value);print(value['state'])


if __name__=='__main__':main()
