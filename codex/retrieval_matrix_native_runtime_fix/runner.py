"""[codex] Delegate original native producer through explicit new runtime gates."""
from pathlib import Path
import os
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from codex.retrieval_matrix_native_runtime_fix import contracts as C
from codex.retrieval_matrix_native_p0 import runner as N

original_load=N.load_plan
original_gate=N.source_gate
original_runtime=N.runtime
INSTALLED=False


def load_plan(path):
    plan=original_load(path);C.validate(plan);return plan


def source_gate(path,plan,commit):
    C.require_review(plan);C.validate(plan,imports=True)
    value=original_gate(path,plan,commit)
    status=subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain','--',str(C.HERE.relative_to(ROOT))],text=True)
    if status.strip():raise ValueError('Native runtime repair source/plan must be clean and committed')
    return value


def runtime(plan,*,gpu):
    if gpu and (os.environ.get('RETRIEVAL_CENTRAL_OWNER')!='[codex]'
            or not os.environ.get('RETRIEVAL_CENTRAL_TASK') or not os.environ.get('CUDA_VISIBLE_DEVICES')):
        raise ValueError('Repaired native GPU production requires the root central dispatcher')
    C.validate(plan,imports=True)
    value=original_runtime(plan,gpu=gpu);value['runtime_fix']=C.runtime_identity(plan)
    return value


def install():
    global INSTALLED
    if INSTALLED:raise ValueError('Install the repaired native facade once per process')
    N.load_plan,N.source_gate,N.runtime=load_plan,source_gate,runtime
    INSTALLED=True
    return N


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--out',type=Path,required=True)
    args,remaining=parser.parse_known_args()
    sys.argv=[sys.argv[0],*remaining,'--out',str(C.output_path(args.out))]
    install().main()
