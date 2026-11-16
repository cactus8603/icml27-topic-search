"""[codex] Native reranker variant/data/call/score contracts; no model at import."""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from codex.retrieval_matrix import runner as R
from codex.retrieval_matrix_rerank import metrics as M

ROOT=Path(__file__).resolve().parents[2]
COUNTS={"cola":210,"replace_att":788,"replace_obj":1652,"replace_rel":1406,"swap_att":666,"swap_obj":245,"coco_mcq":5914,"voc_mcq":5031}
SCPP=("replace_att","replace_obj","replace_rel","swap_att","swap_obj")
CANONICAL=("COLA","SugarCrepe++","NegBench")


def entry(path):
    path=Path(path).resolve();return {"path":str(path),"sha256":R.sha(path)}


def checked(item):
    path=R.checked_file(item)
    if not path.is_relative_to(ROOT):raise ValueError("Native MCQ evidence must stay in this workspace")
    return path


def load_examples(plan,variant,*,verify_images=False):
    source=R.read(checked(plan["data_reference"]))
    rows=R.json_rows(checked(source["examples"]))
    pairs=[p for row in rows for p in row["pairs"]]
    if (Counter(row["subset"] for row in rows)!=COUNTS or len(rows)!=15912 or len(pairs)!=58891
            or len({r["id"] for r in rows})!=15912 or len({p["id"] for p in pairs})!=58891
            or pairs!=R.json_rows(checked(source["pairs"]))):
        raise ValueError("Native MCQ must retain every original row/pair and original order")
    for item in source["source_data"]:checked(item)
    receipt=R.json_rows(checked(source["images_receipt"]))
    images={r["path"]:r for r in receipt}
    if len(images)!=9425 or len(images)!=len(receipt) or set(images)!={p["image"] for p in pairs}:
        raise ValueError("Native MCQ must retain every original image")
    if verify_images:
        from PIL import Image
        for name,row in images.items():
            path=R.safe(name)
            if not path.is_relative_to(ROOT/"dl") or R.sha(path)!=row["sha256"] or path.stat().st_size!=row["bytes"]:
                raise ValueError("Original complete MCQ image payload changed")
            with Image.open(path) as image:image.verify()
    rows=deepcopy(rows)
    for row in rows:
        expected=4 if row["subset"]=="cola" or row["subset"].endswith("_mcq") else 3
        task="cola_text_choice" if row["subset"]=="cola" else ("negbench_mcq" if row["subset"].endswith("_mcq") else "scpp")
        instruction="Determine if the following image is relevant to the query."+("" if row["subset"]=="cola" else " ")
        if len(row["pairs"])!=expected or row["task"]!=task or row["instruction"]!=instruction:
            raise ValueError("Original MCQ input/strict rule/instruction changed")
        if row["subset"]=="cola":row["task"]=variant["cola_rule"]
        for pair in row["pairs"]:pair["image_sha256"]=images[pair["image"]]["sha256"]
    return rows


def call_groups(examples,variant):
    """Storage32 examples never changes released forward batch boundaries."""
    backend=variant["model"]["backend"]
    if backend=="swift_released":
        pairs=[p for row in examples for p in row["pairs"]]
        return [pairs[s:s+32] for s in range(0,len(pairs),32)]
    groups=[]
    for row in examples:
        pairs=row["pairs"]
        if backend=="jina_native":
            groups.extend([pairs[:2],pairs[2:]] if row["subset"]=="cola" else [pairs])
        else:groups.extend([[pair] for pair in pairs])
    return groups


def validate_scores(scores,pairs,variant):
    if [r.get("id") for r in scores]!=[p["id"] for p in pairs]:
        raise ValueError("Native backend must cover original pair IDs in original order")
    backend=variant["model"]["backend"]
    for row in scores:
        value=row.get("score");raw=row.get("official_raw")
        if type(value) is not float or not math.isfinite(value) or not 0<=value<=1:
            raise ValueError("Native scalar must remain its original finite activated probability")
        if backend=="swift_released":valid=raw==[value]
        elif backend=="mcmr_pointwise":
            valid=(isinstance(raw,dict) and set(raw)=={"p_true","true_logprob","false_logprob"}
                and raw["p_true"]==value and all(math.isfinite(v) for v in raw.values()))
        else:valid=raw==value
        if not valid:raise ValueError("Original raw representation/scalar changed; do not rescale or relabel")


def evaluate(rows,scores,variant):
    report,detail=M.evaluate(rows,scores)
    cola=[r for r in rows if r["subset"]=="cola"]
    alternate=deepcopy(cola)
    for row in alternate:row["task"]="cola_image_choice" if variant["cola_rule"]=="cola_text_choice" else "cola_text_choice"
    ids={p["id"] for r in cola for p in r["pairs"]}
    other,_=M.evaluate(alternate,[r for r in scores if r["id"] in ids])
    report["diagnostic_cola_alternate_axis"]=other["benchmarks"]
    report["native_cola_rule"]=variant["cola_rule"]
    report["qualification"]=assess(report,variant)
    return report,detail


def assess(report,variant):
    """All3 named benchmarks stay visible; absent exact targets never qualify."""
    if report.get("rows")!=15912 or set(report.get("subsets",{}))!=set(COUNTS):
        raise ValueError("Qualification requires complete original MCQ denominators")
    for name,count in COUNTS.items():
        row=report["subsets"][name]
        if row.get("rows")!=count or not math.isfinite(row["accuracy"]) or not 0<=row["accuracy"]<=1:
            raise ValueError("Qualification subset denominators/values changed")
    names={"COLA":(variant["cola_rule"],["cola"]),"SugarCrepe++":("scpp",list(SCPP)),"NegBench":("negbench_mcq",["coco_mcq","voc_mcq"])}
    if set(report.get("benchmarks",{}))!={x[0] for x in names.values()}:
        raise ValueError("Retain all three native named benchmark results")
    comparisons={}
    for name,(metric,subsets) in names.items():
        row=report["benchmarks"][metric];observed=float(row["macro_accuracy"])
        expected=sum(report["subsets"][s]["accuracy"] for s in subsets)/len(subsets)
        if not math.isfinite(observed) or not 0<=observed<=1 or row.get("subsets")!=sorted(subsets) or row.get("rows")!=sum(COUNTS[s] for s in subsets) or abs(observed-expected)>1e-12:
            raise ValueError("Native benchmark macro/denominator differs")
        descriptor=variant["targets"][name]
        verified=descriptor["verified_protocol_match"] is True
        target=descriptor["target"]
        if verified and (type(target) not in (int,float) or not math.isfinite(target) or not 0<=target<=1):
            raise ValueError("Verified target must be a published finite fraction")
        comparisons[name]={"metric":metric,"observed":observed,"target":target,"verified_protocol_match":verified,
            "matched":bool(verified and abs(observed-target)<=.01+1e-12),"gap":descriptor["gap"]}
    matched=sum(r["matched"] for r in comparisons.values())
    available=sum(r["verified_protocol_match"] for r in comparisons.values())
    return {"state":"TARGETS_INCOMPLETE" if available<2 else ("NUMERIC_MATCH" if matched>=2 else "FAILED"),
        "required_distinct_benchmarks":2,"published_targets_available":available,"matched_distinct_benchmarks":matched,
        "benchmarks":comparisons,"independent_review_required":True}


def bound_array(path,items,identity,batch_size=32):
    receipt=R.array_receipt(path);state=R.read(path.with_suffix('.state.json'))
    if state.get('fingerprint')!=R.digest({'identity':identity,'items':items,'batch_size':batch_size}):
        raise ValueError('Native array state fingerprint belongs to another producer/model/request/order')
    array=np.load(path,mmap_mode='r',allow_pickle=False);batches=state.get('batches',[])
    starts=list(range(0,len(items),batch_size))
    if len(batches)!=len(starts):raise ValueError('Native committed batch receipts omit original boundaries')
    for start,row in zip(starts,batches,strict=True):
        end=min(start+batch_size,len(items))
        digest=hashlib.sha256(np.ascontiguousarray(array[start:end]).tobytes()).hexdigest()
        if row.get('start')!=start or row.get('end')!=end or row.get('sha256')!=digest:
            raise ValueError('Native committed batch boundaries or payload checksum changed')
    return receipt
