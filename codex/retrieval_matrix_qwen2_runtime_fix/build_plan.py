"""[codex] Prepare a reviewable new batch; never mutate the old frozen plan."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from codex.retrieval_matrix_qwen2_runtime_fix import contracts as C

def build(mode):
    base=C.parent(mode=mode);plan=deepcopy(base)
    relative,commit=C.PARENTS[mode]
    plan.update(state="REVIEW_REQUIRED",reviewed=False,runtime_batch=C.BATCHES[mode],
        prereg="codex/retrieval_matrix_qwen2_runtime_fix/PREREG.md",
        runtime_fix={"mode":mode,"parent_plan":C.entry(ROOT/relative),"parent_freeze_commit":commit,
            "overlay":C.entry(HERE/"overlay.json"),
            "reason":"Original smoke failed before model load: flash_attn.bert_padding imported missing einops. New isolated dependency-only batch."})
    for relative,expected in base["source_files"].items():
        if C.sha(ROOT/relative)!=expected:raise ValueError("An original frozen source changed: "+relative)
    for name in C.OWN:plan["source_files"][str((HERE/name).relative_to(ROOT))]=C.sha(HERE/name)
    C.validate(plan)
    return plan

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--mode",choices=C.PARENTS,required=True)
    p.add_argument("--out",type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise ValueError("Never overwrite a runtime-fix execution plan")
    a.out.write_text(json.dumps(build(a.mode),ensure_ascii=False,indent=2,sort_keys=True)+"\n")
    print({"owner":"[codex]","state":"REVIEW_REQUIRED","plan":str(a.out),"sha256":C.sha(a.out)})

if __name__=="__main__":main()
