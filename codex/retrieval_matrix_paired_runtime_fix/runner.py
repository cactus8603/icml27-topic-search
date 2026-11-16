"""[codex] Private paired producer, same class/encode and explicit runtime repair."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from codex.retrieval_matrix_paired_runtime_fix import contracts as C,models,qualification as Q

spec=importlib.util.spec_from_file_location('paired_repair_private_producer',ROOT/'codex/retrieval_matrix_paired_p0/runner.py')
P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P)
original_gate,original_runtime=P.source_gate,P.runtime
P.Q,P.R.Q=Q,Q
P.models=SimpleNamespace(PairedModel=models.create,verify=models.ORIGINAL.verify,parts=models.ORIGINAL.parts)
P.R.models=SimpleNamespace(Siglip2=models.create,verify=models.ORIGINAL.verify)


def source_gate(path,plan,commit):
    C.require_review(plan);C.validate(plan,imports=True)
    value=original_gate(path,plan,commit)
    if subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain','--',str(C.HERE.relative_to(ROOT))],text=True).strip():
        raise ValueError('New paired repair must remain clean and committed')
    return value


def runtime(plan,*,gpu):
    if gpu and (os.environ.get('RETRIEVAL_CENTRAL_OWNER')!='[codex]'
            or not os.environ.get('RETRIEVAL_CENTRAL_TASK') or not os.environ.get('CUDA_VISIBLE_DEVICES')):
        raise ValueError('Paired repair GPU requires the root central dispatcher')
    C.validate(plan,imports=True)
    actual=original_runtime(plan,gpu=gpu);actual['runtime_fix']=C.runtime_identity(plan)
    return actual


P.source_gate,P.runtime=source_gate,runtime
P.R.source_gate,P.R.runtime=source_gate,runtime


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--out',type=Path,required=True)
    args,remaining=parser.parse_known_args();sys.argv=[sys.argv[0],*remaining,'--out',str(C.output_path(args.out))]
    P.R.main()
