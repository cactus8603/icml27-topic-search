"""[codex] Immutable two-stage input/qualification and label-free admission."""
from __future__ import annotations
import json
import importlib.util
import math
import hashlib
import re
from pathlib import Path
import numpy as np
from codex.retrieval_matrix import runner as R
from codex.retrieval_matrix_panel import data as D, qualification as Q

ROOT = Path(__file__).resolve().parents[2]
TASKS = ("cola_text_choice", "scpp", "negbench_mcq")
COUNTS = {"cola":210,"replace_att":788,"replace_obj":1652,"replace_rel":1406,
          "swap_att":666,"swap_obj":245,"coco_mcq":5914,"voc_mcq":5031}
SUBSETS = {"cola_text_choice":["cola"], "scpp":["replace_att","replace_obj","replace_rel","swap_att","swap_obj"],
           "negbench_mcq":["coco_mcq","voc_mcq"]}
PHASES = {"mllm":"P2-qualified-fixed-baselines", "clip":"P2-qualified-fixed-clip-baselines"}
NATIVE_VALIDATOR = "codex/retrieval_matrix_native_p0/qualification.py"
NATIVE_SOURCES = tuple("codex/retrieval_matrix_native_p0/"+n+".py" for n in ("qualification","data","metrics","evidence"))


def frozen_origin(plan_path, plan, identity,*,include_plan=False):
    """Check immutable producing Git bytes without demanding today's P2 source."""
    commit=identity.get("freeze_commit","")
    if not re.fullmatch(r"[0-9a-f]{40}",commit):
        raise ValueError("Producing evidence requires its complete source freeze commit")
    relative=str(plan_path.relative_to(ROOT))
    if R.C.git_bytes(commit,relative)!=plan_path.read_bytes():
        raise ValueError("Producing plan differs from its source freeze commit")
    sources=identity.get("source_files",{})
    if not plan.get("source_files") or any(sources.get(k)!=v for k,v in plan["source_files"].items()):
        raise ValueError("Producing identity omits its complete source SHA table")
    if include_plan and (sources.get(str(plan_path.relative_to(ROOT)))!=R.sha(plan_path)
            or not plan.get("prereg") or plan["prereg"] not in sources):
        raise ValueError("P2 provenance must include its producing plan and preregistration blobs")
    for relative,expected in sources.items():
        source=Path(relative)
        if source.is_absolute() or ".." in source.parts or hashlib.sha256(R.C.git_bytes(commit,relative)).hexdigest()!=expected:
            raise ValueError("Producing source SHA differs from its immutable commit")


def mcq_report(plan,key,report):
    if (plan.get("counts")!=COUNTS or plan.get("acceptance",{}).get("absolute_tolerance")!=.01
            or report.get("rows")!=15912 or set(report.get("benchmarks",{}))!=set(TASKS)
            or set(report.get("subsets",{}))!=set(COUNTS)):
        raise ValueError("Complete original MCQ benchmark/subset denominators must be retained")
    for subset,count in COUNTS.items():
        row=report["subsets"][subset]
        value=float(row["accuracy"])
        if row.get("rows")!=count or not math.isfinite(value) or not 0<=value<=1:
            raise ValueError("MCQ native subset denominator or accuracy differs")
    matches=[]
    for task,names in SUBSETS.items():
        row=report["benchmarks"][task]
        observed=float(row["macro_accuracy"]);target=float(plan["acceptance"][key][task])
        expected=sum(float(report["subsets"][n]["accuracy"]) for n in names)/len(names)
        if (not math.isfinite(observed) or not 0<=observed<=1 or not math.isfinite(target) or not 0<=target<=1
                or row.get("rows")!=sum(COUNTS[n] for n in names) or row.get("subsets")!=sorted(names)
                or abs(observed-expected)>1e-12):
            raise ValueError("MCQ benchmark must use the original complete subset macro, target and denominator")
        matches.append(abs(observed-target)<=.01+1e-12)
    if sum(matches)<2:
        raise ValueError("Reranker needs two distinct official MCQ benchmarks within1pp; third remains visible")
    return sum(matches)


def checked(entry):
    path = R.checked_file(entry)
    if not path.is_relative_to(ROOT):
        raise ValueError("P3 evidence must remain inside the workspace")
    return path


def qualify_reranker(spec, proof):
    """Retain all three MCQ benchmarks and bind the exact released model spec."""
    plan_path = checked(proof["plan"])
    plan = R.read(plan_path)
    complete_path, report_path, review_path = [checked(proof[k]) for k in ("complete", "report", "review")]
    complete, report, review = map(R.read, (complete_path, report_path, review_path))
    identity = complete["identity"]
    key = proof["model"]
    if (plan.get("state") != "FROZEN" or complete.get("state") != "COMPLETE"
            or identity.get("plan_sha256") != R.sha(plan_path) or identity.get("model") != key
            or identity.get("model_spec") != spec or plan["models"][key] != spec
            or plan.get("pair_batch") != 4):
        raise ValueError("P3 reranker differs from its completed fixed official MCQ identity")
    if (review.get("state") != "PASS" or review.get("scope") != "official-reranker-mcq"
            or review.get("independent_recomputation") is not True or review.get("audited_arrays") is not True
            or not review.get("reviewer") or not review.get("producer")
            or review["reviewer"] == review["producer"]
            or review.get("plan_sha256") != R.sha(plan_path) or review.get("model") != key
            or review.get("complete_sha256") != R.sha(complete_path) or review.get("report_sha256") != R.sha(report_path)):
        raise ValueError("P3 requires a separate reviewer bound to original MCQ arrays and exact report bytes")
    matched=mcq_report(plan,key,report)
    frozen_origin(plan_path,plan,identity)
    if set(complete.get("subsets",{}))!=set(COUNTS):
        raise ValueError("MCQ completion omits an original subset")
    for subset,count in COUNTS.items():
        path=complete_path.parent/(subset+".npy");state_path=path.with_suffix(".state.json")
        pairs=count*(4 if subset=="cola" or subset.endswith("_mcq") else 3)
        expected=complete["subsets"][subset]
        if (expected.get("pairs")!=pairs or R.sha(path)!=expected.get("array_sha256")
                or R.sha(state_path)!=expected.get("state_sha256")):
            raise ValueError("Original MCQ score array/state bytes or pair denominator differ")
        array=np.load(path,mmap_mode="r",allow_pickle=False)
        if array.shape!=(pairs,1) or array.dtype!=np.float32 or not np.isfinite(array).all():
            raise ValueError("Original MCQ scalar dtype/coverage/values differ")
        # Reuse the frozen checkpoint's completed-array integrity check.
        if R.array_receipt(path)["shape"]!=[pairs,1]:
            raise ValueError("Original MCQ state does not cover every original pair")
    return {"state":"PASS","datasets_retained":list(TASKS),"matched":matched,"model_spec_sha256":Q.digest(spec)}



def first_stage_header(reference, dataset):
    plan_path = checked(reference["plan"])
    plan = R.read(plan_path)
    key = reference["model"]
    family=reference.get("family")
    if (family not in PHASES or plan.get("schema")!=1 or plan.get("state")!="FROZEN"
            or plan.get("phase")!=PHASES[family]
            or plan.get("settings",{}).get("top_ks")!=[50,100,200]
            or plan["settings"].get("main_k")!=100):
        raise ValueError("P3 must consume a completed qualified fixed-baseline P2 index")
    if family=="mllm":
        R.models.validate_spec(plan["models"][key])
        qualification=Q.admit(plan["qualification"][key], plan["models"][key])
    else:
        # CLIP has its own two-dataset published targets. Its independently
        # reviewed validator is an explicit pinned dependency, never a PASS flag.
        path=checked(reference["qualification_adapter"])
        if str(path.relative_to(ROOT))!=NATIVE_VALIDATOR or plan.get("qualification_adapter")!=reference["qualification_adapter"]:
            raise ValueError("CLIP must use its exact frozen native qualification validator")
        if plan.get("retriever_family")!="CLIP" or plan["models"][key].get("family")!="CLIP":
            raise ValueError("CLIP family must agree with its unchanged native model and P2 plan")
        loader=importlib.util.spec_from_file_location("p3_clip_qualification_"+R.sha(path),path)
        module=importlib.util.module_from_spec(loader);loader.loader.exec_module(module)
        qualification=module.admit(plan["qualification"][key],plan["models"][key])
    if (qualification.get("state")!="PASS"
            or qualification.get("model_spec_sha256")!=Q.digest(plan["models"][key])):
        raise ValueError("P3 first stage lacks an unchanged complete model qualification")
    if qualification.get("model")!=key or (family=="clip" and qualification.get("family")!="CLIP"):
        raise ValueError("First-stage native qualification belongs to another model/family")
    source = reference["indexes"][dataset]
    meta_path, identity_path = checked(source["meta"]), checked(source["identity"])
    if meta_path.parent != identity_path.parent:
        raise ValueError("Index metadata and provenance must identify one producing directory")
    if meta_path.name!="index-meta.json" or identity_path.name!="provenance.json":
        raise ValueError("P3 must consume the original P2 index metadata and provenance files")
    identity = R.read(identity_path)
    meta=R.read(meta_path)
    if meta.get("state")!="COMPLETE" or meta.get("identity")!=identity:
        raise ValueError("P3 requires a complete P2 index before even synthetic GPU smoke")
    frozen_origin(plan_path,plan,identity,include_plan=True)
    if (identity.get("model") != key or identity.get("dataset") != dataset
            or identity.get("plan_sha256") != R.sha(plan_path)
            or identity.get("model_spec") != plan["models"][key] or identity.get("settings") != plan["settings"]):
        raise ValueError("P3 index belongs to another P2 model, plan or dataset")
    return plan,meta_path,identity


def first_stage(reference,dataset,*,verify_images=True):
    plan,meta_path,identity=first_stage_header(reference,dataset)
    spec = plan["datasets"][dataset]
    bundle = (D.load_data(spec,R,verify_images=verify_images) if spec["format"]=="cola_single" else R.load_data(spec,verify_images=verify_images))
    if identity.get("dataset_fingerprint") != bundle["fingerprint"]:
        raise ValueError("P3 complete gallery/query input differs from the original P2 index")
    _, order, meta = R.validate_index(meta_path.parent, identity, bundle, plan["settings"])
    if R.sha(meta_path) != reference["indexes"][dataset]["meta"]["sha256"]:
        raise ValueError("P3 index metadata changed while validating")
    return spec, bundle, order, meta


def selection(plan):
    receipt = R.read(checked(plan["selection"]))
    stages = plan["first_stages"]
    if (len(stages) != 2 or len({(v["plan"]["sha256"],v["model"]) for v in stages.values()}) != 2
            or {v["family"] for v in stages.values()} != {"mllm", "clip"}
            or receipt.get("state") != "FROZEN" or receipt.get("scope") != "development-only"
            or receipt.get("test_results_used") is not False
            or not receipt.get("criterion")
            or receipt.get("first_stages") != {name:{"plan_sha256":v["plan"]["sha256"],"model":v["model"],"family":v["family"]} for name,v in stages.items()}):
        raise ValueError("Freeze one MLLM and one CLIP stage selected from development evidence only")
    if set(receipt.get("development_reports",{}))!=set(stages):
        raise ValueError("First-stage selection must retain exact development evidence for both stages")
    for name,stage in stages.items():
        report=R.read(checked(receipt["development_reports"][name]))
        if (report.get("dataset")!="conjunction-dev" or report.get("model")!=stage["model"]
                or report.get("provenance",{}).get("index_meta_sha256")!=stage["indexes"]["conjunction-dev"]["meta"]["sha256"]):
            raise ValueError("Only matching full conjunction development evidence may select the fixed first stages")


def admit_pairs(bundle, order, *, k, dataset):
    """No labels or ignore masks are accepted by this interface."""
    if k not in {50,100,200}:
        raise ValueError("P3 supports only frozen K50/100/200")
    order = R.metrics.validate_order(order, len(bundle["gallery"]))
    if len(order) != len(bundle["queries"]) or order.shape[1] < k:
        raise ValueError("P3 requires the complete original P2 order and gallery")
    pairs = []
    for qi, query in enumerate(bundle["queries"]):
        for rank, ci in enumerate(order[qi,:k]):
            candidate = bundle["gallery"][int(ci)]
            pairs.append({"id":R.digest([dataset,query["id"],candidate["id"],rank]),
                "position":len(pairs),"query_index":qi,"candidate_index":int(ci),"rank":rank,
                "query_id":query["id"],"candidate_id":candidate["id"],"query":query["text"],
                "image":candidate["path"],"image_sha256":candidate["sha256"]})
    return pairs


def score_groups(backend, pairs, *, instruction, batch=4):
    output = []
    for start in range(0,len(pairs),batch):
        group = pairs[start:start+batch]
        values = backend.score_pairs(group,instruction=instruction)
        if [row["id"] for row in values] != [row["id"] for row in group]:
            raise ValueError("Released backend changed pair coverage or original order")
        validate_responses(values,[row["id"] for row in group])
        output.extend(values)
    return output


def validate_responses(values,pair_ids):
    if ([r.get("id") for r in values]!=pair_ids
            or any(not isinstance(r.get("score"),(int,float)) or not math.isfinite(r["score"])
                   or r.get("official_raw")!=[r["score"]] for r in values)):
        raise ValueError("CORE released backend must retain exact IDs, finite native scalar and content list")


def reranked(order, values, k):
    values = np.asarray(values)
    if values.shape != (len(order),200) or not np.isfinite(values).all():
        raise ValueError("Shared frozen depth score cache must cover all200 pairs per query")
    return R.metrics.rerank_prefix(order,values,k)
