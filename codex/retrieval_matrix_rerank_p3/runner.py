"""[codex] Qualified P3 score producer; GPU production and CPU analysis separate."""
from __future__ import annotations
import argparse
from datetime import datetime
import fcntl
from importlib.metadata import version
import json
import os
import uuid
from pathlib import Path
import subprocess
import sys
from time import monotonic
from zoneinfo import ZoneInfo
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from codex.retrieval_matrix_rerank_p3 import contracts as C
from codex.retrieval_matrix_rerank import backends as B
R=C.R
HERE=Path(__file__).parent
OWN=("runner.py","contracts.py","build_plan.py","run.sh","PREREG.md","README.md","test_contracts.py")
EXECUTED=("codex/retrieval_matrix/runner.py","codex/retrieval_matrix/models.py","codex/retrieval_matrix/metrics.py",
          "codex/retrieval_scaling/runner.py","codex/retrieval_matrix_panel/data.py",
          "codex/retrieval_matrix_panel/qualification.py","codex/retrieval_matrix_rerank/backends.py")


def gate(path,plan,commit):
    if (len(commit)!=40 or any(c not in "0123456789abcdef" for c in commit)
            or plan.get("state")!="FROZEN" or plan.get("reviewed") is not True
            or plan.get("phase")!="P3-fixed-official-image-rerank-v1"
            or plan.get("settings")!={"pair_batch":4,"checkpoint_chunk":64,"cache_k":200,"main_k":100,"depths":[50,100,200],"smoke_max_abs":.01}):
        raise ValueError("P3 requires independently reviewed frozen source, exact batching and depth settings")
    required=set(EXECUTED)|{"codex/retrieval_matrix_rerank_p3/"+name for name in OWN}
    if not required<=plan["source_files"].keys():
        raise ValueError("P3 plan omits an executing source or preregistration file")
    for relative in set(plan["source_files"])|{str(path.resolve().relative_to(ROOT))}:
        source=R.safe(ROOT/relative)
        if not source.is_relative_to(ROOT) or R.C.git_bytes(commit,relative)!=source.read_bytes():
            raise ValueError("P3 executing source/plan differs from the frozen commit")
        if relative in plan["source_files"] and R.sha(source)!=plan["source_files"][relative]:
            raise ValueError("P3 declared source digest differs")
    for folder in (HERE,ROOT/"codex/retrieval_matrix_rerank"):
        if subprocess.check_output(["git","-C",str(ROOT),"status","--porcelain","--",str(folder.relative_to(ROOT))],text=True).strip():
            raise ValueError("P3 and released backend worktrees must remain clean")
    source=R.read(C.checked(plan["backend_reference"]))
    if (source.get("state")!="FROZEN" or source.get("reviewed") is not True
            or plan.get("instruction")!="Determine if the following image is relevant to the query. "
            or plan.get("environment")!=source.get("environment")
            or set(plan.get("models",{}))!=set(plan.get("reranker_qualification",{}))
            or not plan.get("models")):
        raise ValueError("P3 exact released instruction/runtime/backend and qualification set must remain unchanged")
    if R.sha(ROOT/"codex/retrieval_matrix_rerank/backends.py")!=source["source_files"]["codex/retrieval_matrix_rerank/backends.py"]:
        raise ValueError("P3 released backend differs from its original MCQ freeze")
    C.selection(plan)
    for stage in plan["first_stages"].values():
        for dataset in plan["datasets"]:C.first_stage_header(stage,dataset)
        if stage["family"]=="clip":
            path=C.checked(stage["qualification_adapter"])
            if (any(plan["source_files"].get(relative)!=R.sha(ROOT/relative) for relative in C.NATIVE_SOURCES)
                    or plan["source_files"].get(str(path.relative_to(ROOT)))!=R.sha(path)):
                raise ValueError("CLIP native qualification validator must be part of the frozen executing source")
    if set(plan["datasets"])!={"conjunction-dev","negation-dev"}:
        raise ValueError("Initial P3 depth batch is development-only; public test needs its own frozen batch")
    for key,spec in plan["models"].items():
        if spec.get("backend")!="swift_released" or spec.get("verified_protocol") is not True:
            raise ValueError("Initial P3 supports independently verified CORE released Swift only")
        if source["models"].get(key)!=spec or plan["reranker_qualification"][key]["plan"]!=plan["backend_reference"]:
            raise ValueError("P3 model and MCQ plan must equal the reviewed released backend reference")
        C.qualify_reranker(spec,plan["reranker_qualification"][key])


def environment(plan):
    if (os.environ.get("RETRIEVAL_CENTRAL_OWNER")!="[codex]"
            or not os.environ.get("RETRIEVAL_CENTRAL_TASK") or not os.environ.get("CUDA_VISIBLE_DEVICES")):
        raise ValueError("P3 GPU production must use the root central dispatcher")
    import torch,transformers,flash_attn
    if torch.cuda.device_count()!=1:
        raise ValueError("P3 requires exactly one centrally assigned CUDA device")
    for name,expected in plan["environment"]["packages"].items():
        if version(name)!=expected:raise ValueError("P3 frozen runtime differs: "+name)
    for key in ("install_report","frozen_pip"):C.checked(plan["environment"][key])
    if not torch.cuda.is_available() or torch.__version__!=plan["environment"]["packages"]["torch"] or transformers.__version__!=plan["environment"]["packages"]["transformers"]:
        raise ValueError("P3 actual imported model environment/CUDA differs")
    return {"torch":torch.__version__,"transformers":transformers.__version__,"flash_attn":flash_attn.__version__,"gpu":torch.cuda.get_device_name(0)}


def smoke(out,spec,identity,plan):
    if (out/"smoke.json").exists():
        proof=R.read(out/"smoke.json")
        require_smoke(proof,identity,plan)
        return
    from PIL import Image
    engine=B.BACKENDS[spec["backend"]](spec)
    path=out/"synthetic.png";Image.new("RGB",(1280,960),(64,128,32)).save(path)
    pairs=[{"id":str(i),"query":"A synthetic color image.","image":str(path)} for i in range(4)]
    fixed=C.score_groups(engine,pairs,instruction=plan["instruction"])
    single=C.score_groups(engine,pairs[:1],instruction=plan["instruction"])
    difference=abs(fixed[0]["score"]-single[0]["score"])
    passed=difference<=plan["settings"]["smoke_max_abs"]
    R.fixed_json(out/"smoke.json",{"state":"PASS" if passed else "FAILED","technical_only":True,
        "identity":identity,"batch1_vs4_max_abs":difference,"signature":engine.signature,"raw":fixed})
    if not passed:raise ValueError("P3 synthetic smoke failed; preserve failure without numerical fallback")


def require_smoke(proof,identity,plan):
    value=proof.get("batch1_vs4_max_abs",np.nan)
    keys=("plan_sha256","freeze_commit","model","model_spec","environment")
    if (proof.get("state")!="PASS" or proof.get("technical_only") is not True
            or any(proof.get("identity",{}).get(key)!=identity[key] for key in keys)
            or not isinstance(value,(int,float)) or not np.isfinite(value)
            or not 0<=value<=plan["settings"]["smoke_max_abs"]):
        raise ValueError("P3 requires matching synthetic smoke and numerical batch evidence")


def journal(out,row):
    """Append actual call attempts, including calls in interrupted chunks."""
    path=out/"calls.jsonl"
    with path.open("a") as stream:
        stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+"\n")
        stream.flush();os.fsync(stream.fileno())


def device_memory():
    try:
        import torch
        if torch.cuda.is_initialized():
            return {"allocated_bytes":torch.cuda.max_memory_allocated(),"reserved_bytes":torch.cuda.max_memory_reserved()}
    except Exception:
        pass
    return {"allocated_bytes":None,"reserved_bytes":None}


def call_summary(out):
    events=[json.loads(line) for line in (out/"calls.jsonl").read_text().splitlines()]
    starts={row["call"]:row for row in events if row["state"]=="STARTED"}
    ends={row["call"]:row for row in events if row["state"]!="STARTED"}
    if len(starts)!=sum(row["state"]=="STARTED" for row in events) or not set(ends)<=set(starts):
        raise ValueError("Actual model call journal IDs are incomplete or duplicated")
    memory=[row.get("memory",{}) for row in events]
    return {"attempted_calls":len(starts),"attempted_pairs":sum(row["pairs"] for row in starts.values()),
        "completed_calls":sum(row["state"]=="COMPLETE" for row in ends.values()),
        "failed_or_interrupted_calls":sum(row["state"]=="FAILED_OR_INTERRUPTED" for row in ends.values()),
        "calls_without_end_receipt":len(set(starts)-set(ends)),
        "recorded_model_seconds":sum(row.get("elapsed_seconds",0) for row in ends.values()),
        "peak_allocated_bytes":max([r.get("allocated_bytes") or 0 for r in memory],default=0),
        "peak_reserved_bytes":max([r.get("reserved_bytes") or 0 for r in memory],default=0),
        "disclosure":"Attempt counts include retries; hard-killed calls may lack elapsed/memory receipts. Invocation wall ledger includes failed/resumed processes."}


def raw_group(out,group,identity,plan,get_engine):
    """An atomic raw transaction survives a crash before array/state commit."""
    path=out/"raw"/(f"{group[0]['position']:09d}-{group[-1]['position']+1:09d}.json")
    ids=[p["id"] for p in group]
    if path.exists():
        saved=R.read(path)
        if (saved.get("identity_sha256")!=R.digest(identity) or saved.get("pair_ids")!=ids
                or saved.get("backend_sha256")!=R.sha(out/"backend.json")):
            raise ValueError("Pending original raw response transaction differs from current model/admission")
        C.validate_responses(saved["responses"],ids)
        return saved["responses"]
    engine=get_engine()
    call=uuid.uuid4().hex;began=monotonic()
    journal(out,{"call":call,"state":"STARTED","pair_ids":ids,"pairs":len(ids)})
    try:
        response=C.score_groups(engine,group,instruction=plan["instruction"],batch=4)
        elapsed=monotonic()-began
        R.fixed_json(path,{"identity_sha256":R.digest(identity),"pair_ids":ids,"responses":response,
            "backend_sha256":R.sha(out/"backend.json"),"model_seconds":elapsed,"call":call,"memory":device_memory()})
    except BaseException:
        journal(out,{"call":call,"state":"FAILED_OR_INTERRUPTED","elapsed_seconds":monotonic()-began,"memory":device_memory()})
        raise
    journal(out,{"call":call,"state":"COMPLETE","elapsed_seconds":elapsed,"raw_sha256":R.sha(path),"memory":device_memory()})
    return response


def produce(out,spec,identity,plan,bundle,order):
    if (out/"complete.json").exists():
        complete=R.read(out/"complete.json")
        if (complete.get("state")!="COMPLETE" or complete.get("identity")!=identity
                or complete["array"]!=R.array_receipt(out/"score.npy")
                or R.sha(out/"backend.json")!=complete["processor_sha256"]
                or R.sha(out/"calls.jsonl")!=complete["calls_sha256"]
                or any(R.sha(out/"raw"/name)!=digest for name,digest in complete["raw_files"].items())):
            raise ValueError("Completed P3 cache differs; never overwrite or repeat its model calls")
        return
    pairs=C.admit_pairs(bundle,order,k=200,dataset=identity["dataset"])
    R.fixed_json(out/"admission.json",{"identity":identity,"pairs":len(pairs),"pairs_sha256":R.digest(pairs),
        "policy":"original full-order top200; no labels/ignore/filter/refill; one shared score for nested depth effects"})
    holder=[]
    def get_engine():
        if not holder:
            engine=B.BACKENDS[spec["backend"]](spec)
            R.fixed_json(out/"backend.json",engine.signature)
            holder.append(engine)
        return holder[0]
    def compute(group):
        response=[]
        for start in range(0,len(group),4):
            response.extend(raw_group(out,group[start:start+4],identity,plan,get_engine))
        # A float64 container preserves the original scalar exactly; no score transformation.
        return np.array([[r["score"]] for r in response],dtype=np.float64)
    R.checkpoint(out/"score.npy",pairs,64,compute,identity,progress=out/"progress.json")
    raw={}
    for start in range(0,len(pairs),4):
        path=out/"raw"/(f"{start:09d}-{min(start+4,len(pairs)):09d}.json")
        raw[path.name]=R.sha(path)
    costs=call_summary(out)
    R.fixed_json(out/"complete.json",{"state":"COMPLETE","identity":identity,"array":R.array_receipt(out/"score.npy"),
        "raw_files":raw,"distinct_pairs":len(pairs),"actual_depth":200,"processor_sha256":R.sha(out/"backend.json"),
        "calls_sha256":R.sha(out/"calls.jsonl"),
        "peak_allocated_bytes":costs["peak_allocated_bytes"],"peak_reserved_bytes":costs["peak_reserved_bytes"],"model_calls":costs,
        "cost_scope":"K200 shared cache; K50/K100 are paired prefix effects, not separately timed service calls"})


def analyze(out,identity,plan,spec,bundle,order):
    receipt=R.read(out/"complete.json")
    if receipt.get("state")!="COMPLETE" or receipt["identity"]!=identity or receipt["array"]!=R.array_receipt(out/"score.npy"):
        raise ValueError("P3 raw score array producing identity differs")
    pairs=C.admit_pairs(bundle,order,k=200,dataset=identity["dataset"])
    if R.read(out/"admission.json")["pairs_sha256"]!=R.digest(pairs):raise ValueError("P3 original candidate prefix changed")
    if R.sha(out/"backend.json")!=receipt["processor_sha256"]:raise ValueError("P3 backend signature changed")
    values=np.load(out/"score.npy",allow_pickle=False)
    if values.shape!=(len(pairs),1):raise ValueError("P3 pair budget or original coverage differs")
    if R.sha(out/"calls.jsonl")!=receipt["calls_sha256"] or call_summary(out)!=receipt["model_calls"]:
        raise ValueError("Actual backend call journal changed")
    expected_names={f"{s:09d}-{min(s+4,len(pairs)):09d}.json" for s in range(0,len(pairs),4)}
    if set(receipt["raw_files"])!=expected_names:raise ValueError("Raw transaction coverage changed")
    for start in range(0,len(pairs),4):
        name=f"{start:09d}-{min(start+4,len(pairs)):09d}.json";path=out/"raw"/name
        if R.sha(path)!=receipt["raw_files"][name]:raise ValueError("Original raw response bytes changed")
        raw=R.read(path);group=pairs[start:start+4]
        C.validate_responses(raw["responses"],[p["id"] for p in group])
        if (raw.get("identity_sha256")!=R.digest(identity) or raw.get("backend_sha256")!=receipt["processor_sha256"]
                or raw["pair_ids"]!=[p["id"] for p in group]):
            raise ValueError("Original raw response ordering changed")
        if not np.array_equal(values[start:start+4,0],[r["score"] for r in raw["responses"]]):
            raise ValueError("Saved scalar differs from the original raw response")
    labels=R.load_labels(spec,bundle)
    depths,detail={},{}
    values=values.reshape(len(order),200)
    for k in (50,100,200):
        depths[str(k)],detail[str(k)]=R.metrics.evaluate(C.reranked(order,values,k),[r["id"] for r in bundle["gallery"]],labels)
    R.atomic_json(out/"results/report.json",{"owner":"[codex]","identity":identity,"main_k":100,"depths":depths,
        "distinct_scored_pairs":len(pairs),"score_cache_k":200,"complete_sha256":R.sha(out/"complete.json"),
        "cost_scope":receipt["cost_scope"],"model_calls":receipt["model_calls"],"admission":"complete P2 order, no label admission; ignore removed in scoring only, no refill"})
    R.atomic_json(out/"results/per-query.json",detail)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",choices=["smoke","score","analyze"],required=True)
    for name in ("plan","out"):p.add_argument("--"+name,type=Path,required=True)
    for name in ("model","first-stage","dataset","freeze-commit"):p.add_argument("--"+name,required=True)
    p.add_argument("--smoke-out",type=Path);a=p.parse_args()
    began=monotonic();plan=R.read(a.plan);gate(a.plan,plan,a.freeze_commit)
    if a.dataset not in plan["datasets"]:raise ValueError("P3 dataset is outside the frozen development batch")
    reference=plan["first_stages"][a.first_stage]
    if a.stage=="smoke":
        _,_,parent_identity=C.first_stage_header(reference,a.dataset)
        fingerprint=parent_identity["dataset_fingerprint"]
    else:
        data_spec,bundle,order,index_meta=C.first_stage(reference,a.dataset)
        fingerprint=bundle["fingerprint"]
    spec=plan["models"][a.model]
    identity={"plan_sha256":R.sha(a.plan),"freeze_commit":a.freeze_commit,"model":a.model,"model_spec":spec,
        "first_stage":a.first_stage,"dataset":a.dataset,"dataset_fingerprint":fingerprint,
        "index_meta_sha256":plan["first_stages"][a.first_stage]["indexes"][a.dataset]["meta"]["sha256"]}
    out=R.safe(a.out);allowed=ROOT/"dl/retrieval-matrix-rerank-p3/runs"
    if out==allowed or not out.is_relative_to(allowed):raise ValueError("P3 output must be a distinct isolated runtime run directory")
    with R.writer(out):
        invocation=uuid.uuid4().hex;ledger_path=out/"invocations.json"
        ledger=R.read(ledger_path) if ledger_path.exists() else []
        ledger.append({"owner":"[codex]","invocation":invocation,"state":"STARTED","stage":a.stage,
            "time":datetime.now(ZoneInfo("Asia/Taipei")).isoformat()})
        R.atomic_json(ledger_path,ledger)
        ok=False
        try:
            if a.stage=="analyze":
                identity["environment"]=R.read(out/"identity.json")["environment"]
                analyze(out,identity,plan,data_spec,bundle,order)
            else:
                identity["environment"]=environment(plan);R.fixed_json(out/"identity.json",identity)
                if a.stage=="smoke":smoke(out,spec,identity,plan)
                else:
                    if a.smoke_out is None:raise ValueError("P3 score production needs synthetic-only smoke")
                    require_smoke(R.read(R.safe(a.smoke_out)/"smoke.json"),identity,plan)
                    produce(out,spec,identity,plan,bundle,order)
            ok=True
        finally:
            path=out/"invocations.json";ledger=R.read(path) if path.exists() else []
            ledger.append({"owner":"[codex]","invocation":invocation,"state":"COMPLETE" if ok else "FAILED",
                "time":datetime.now(ZoneInfo("Asia/Taipei")).isoformat(),"stage":a.stage,
                "completed":ok,"wall_seconds":monotonic()-began,"includes_failed_and_resume":True})
            R.atomic_json(path,ledger)


if __name__=="__main__":main()
