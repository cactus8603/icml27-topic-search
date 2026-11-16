"""[codex] New frozen dependency-repair batch; original Qwen2 producer intact."""
from pathlib import Path
import os
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from codex.retrieval_matrix_qwen2_runtime_fix import contracts as C
ENTRY=None
original_gate=original_environment=None
MODE=None

def install(mode):
    global ENTRY,MODE,original_gate,original_environment
    if ENTRY is not None:raise ValueError("Install a repaired runner exactly once per process")
    if mode=="embedding":
        from codex.retrieval_matrix_qwen2 import runner as Q
        ENTRY=Q.R
    elif mode=="mcmr":
        from codex.retrieval_matrix_mcmr import runner as N
        ENTRY=N
    else:raise ValueError("Unknown repaired Qwen2 mode")
    MODE=mode
    original_gate,original_environment=ENTRY.source_gate,ENTRY.runtime_environment
    from codex.retrieval_matrix import runner as R
    R.SOURCE_FILES+=tuple(str((HERE/name).relative_to(ROOT)) for name in C.OWN)
    ENTRY.source_gate,ENTRY.runtime_environment=source_gate,runtime_environment
    return ENTRY

def source_gate(path,plan,commit):
    if plan["runtime_fix"].get("mode")!=MODE:raise ValueError("Repaired entry mode differs from its new plan")
    result=original_gate(path,plan,commit)
    if plan.get("reviewed") is not True:raise ValueError("Runtime fix requires an independent review before GPU production")
    C.validate(plan)
    status=subprocess.check_output(["git","-C",str(ROOT),"status","--porcelain","--",str(HERE.relative_to(ROOT))],text=True)
    if status.strip():raise ValueError("Runtime fix source/plan must be clean and committed")
    return result

def runtime_environment(spec):
    if (os.environ.get("RETRIEVAL_CENTRAL_OWNER")!="[codex]"
            or not os.environ.get("RETRIEVAL_CENTRAL_TASK") or not os.environ.get("CUDA_VISIBLE_DEVICES")):
        raise ValueError("Repaired Qwen2 GPU production must use the root central dispatcher")
    import torch
    if torch.cuda.device_count()!=1:raise ValueError("Repaired Qwen2 requires exactly one centrally assigned GPU")
    actual=original_environment(spec)
    actual["modules"]["einops"]=C.actual_modules()["einops"]
    actual["runtime_fix"]={"overlay_manifest":C.entry(HERE/"overlay.json"),"parent_freeze_commit":C.PARENTS[MODE][1]}
    return actual

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument("--mode",choices=C.PARENTS,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args,remaining=parser.parse_known_args()
    out=C.output_path(args.out);sys.argv=[sys.argv[0],*remaining,"--out",str(out)]
    install(args.mode).main()
