"""[codex] Prepare reviewable P3 only after real independently qualified P0/P2."""
import argparse
from copy import deepcopy
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from codex.retrieval_matrix_rerank_p3 import contracts as C,runner as P
R=C.R


def build(inputs,backend_plan):
    source=R.read(backend_plan)
    if source.get("state")!="FROZEN" or source.get("reviewed") is not True:
        raise ValueError("P3 needs a previously independently frozen released backend specification")
    plan={"owner":"[codex]","schema":1,"phase":"P3-fixed-official-image-rerank-v1",
        "state":"REVIEW_REQUIRED","reviewed":False,"first_stages":deepcopy(inputs["first_stages"]),
        "selection":inputs["selection"],"models":{},"reranker_qualification":deepcopy(inputs["reranker_qualification"]),
        "datasets":["conjunction-dev","negation-dev"],"environment":deepcopy(source["environment"]),
        "settings":{"pair_batch":4,"checkpoint_chunk":64,"cache_k":200,"main_k":100,"depths":[50,100,200],"smoke_max_abs":.01},
        "instruction":"Determine if the following image is relevant to the query. ",
        "backend_reference":{"path":str(backend_plan.resolve()),"sha256":R.sha(backend_plan)},
        "cost_policy":"Actual200 scored pairs per query. Shared-prefix50/100 effects are not independently timed50/100 service cost.",
        "review_action":"Independent review, set FROZEN/reviewed=true, commit every source and this plan before any P3 GPU smoke/inference."}
    C.selection(plan)
    for key,proof in plan["reranker_qualification"].items():
        spec=source["models"][key]
        if key not in {"core2","core8"} or spec.get("backend")!="swift_released":
            raise ValueError("First P3 batch supports only the separately qualified frozen CORE2/8 release")
        C.qualify_reranker(spec,proof);plan["models"][key]=deepcopy(spec)
    if not plan["models"]:raise ValueError("P3 requires at least one genuinely qualified reranker")
    for stage in plan["first_stages"].values():
        for dataset in plan["datasets"]:C.first_stage(stage,dataset)
    sources={relative:R.sha(ROOT/relative) for relative in P.EXECUTED}
    for name in P.OWN:
        relative="codex/retrieval_matrix_rerank_p3/"+name;sources[relative]=R.sha(ROOT/relative)
    for stage in plan["first_stages"].values():
        if stage["family"]=="clip":
            path=C.checked(stage["qualification_adapter"]);sources[str(path.relative_to(ROOT))]=R.sha(path)
            for relative in C.NATIVE_SOURCES:sources[relative]=R.sha(ROOT/relative)
    if sources["codex/retrieval_matrix_rerank/backends.py"]!=source["source_files"]["codex/retrieval_matrix_rerank/backends.py"]:
        raise ValueError("Released backend executing code changed after MCQ freeze")
    plan["source_files"]=sources
    return plan


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs",type=Path,required=True)
    p.add_argument("--backend-plan",type=Path,default=ROOT/"codex/retrieval_matrix_rerank/plan.core-mcq.v1.json")
    p.add_argument("--out",type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise ValueError("Never overwrite an existing P3 execution plan")
    plan=build(R.read(a.inputs),a.backend_plan)
    R.atomic_json(a.out,plan)
    print({"owner":"[codex]","state":plan["state"],"path":str(a.out),"sha256":R.sha(a.out)})


if __name__=="__main__":main()
