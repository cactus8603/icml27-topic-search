"""[codex] Build a review-required plan without opening model artifacts."""
from __future__ import annotations
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from codex.retrieval_matrix_native_analysis import contracts as C,workflow as W

def build():
    inventory=C.profiles();jobs={}
    for key,profile in inventory['profiles'].items():
        job=W.job_value(profile);path=Path(job['manifest_path']);expected=W.manifest(profile)
        if path.exists():
            if W.read(path)!=expected:raise ValueError('Never replace a different prospective manifest')
        else:W.write_new(path,expected)
        job['manifest']=C.entry(path);jobs[key]=job
    value={'schema':1,'owner':'[codex]','phase':W.PHASE,'state':'REVIEW_REQUIRED','reviewed':False,
        'shared_analysis_ref':C.SHARED_REF,'shared_sources':inventory['shared_sources'],'schedule':inventory['schedule'],
        'source_files':W.own_sources(),'jobs':jobs,'gpu_allowed':False,'user_decision':'PENDING','execution_plan':True,
        'decision_order':['original-frozen-mechanical','fresh-nonimplementer-original-array-recomputation',
            'original-native-make-admit','user-final-decision'],
        'excluded_missing_targets':inventory['excluded_missing_targets'],
        'production_outputs_read_during_build':False,'qualification_created_during_build':False}
    W.validate_plan(value);return value

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.resolve().parent!=C.HERE:raise ValueError('Execution plan must be reviewed with its source folder')
    W.write_new(a.out,build());print(C.sha(a.out))
if __name__=='__main__':main()
