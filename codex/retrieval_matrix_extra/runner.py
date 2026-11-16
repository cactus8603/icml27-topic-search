"""[codex] Independent extra-baseline P0 entry; reuses frozen resumable producer.

No automatic P2 admission. Native any-hit retrieval uses its own CPU scorer;
CORE GME7 retains the released pytrec caption-fraction metric unchanged.
"""
from pathlib import Path
import importlib
from importlib.metadata import version
import json
import subprocess
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from codex.retrieval_matrix import runner as R
from codex.retrieval_matrix_extra import adapters as A

original_source_gate=R.source_gate
original_environment=R.runtime_environment
original_load_plan=R.load_plan
original_official_metrics=R.official_metrics


def source_gate(plan_path,plan,freeze_commit):
    value=original_source_gate(plan_path,plan,freeze_commit)
    status=subprocess.check_output(["git","-C",str(ROOT),"status","--porcelain","--",str(HERE.relative_to(ROOT))],text=True)
    if status.strip():
        raise ValueError("Extra-baseline production requires its own clean committed source directory")
    dependency_path=HERE/"dependencies.json"
    relative=str(dependency_path.relative_to(ROOT))
    if plan.get("source_files",{}).get(relative) != R.sha(dependency_path):
        raise ValueError("The frozen extra plan must bind dependencies.json in its source SHA table")
    dependencies=json.loads(dependency_path.read_text())
    for filename,digest in dependencies["files"].items():
        path=(ROOT/filename).resolve()
        if not path.is_relative_to(ROOT/"dl") or R.sha(path) != digest:
            raise ValueError("Pinned isolated dependency payload changed: "+filename)
    return value


def load_plan(path):
    plan=original_load_plan(path)
    if plan.get("purpose") != "P0_OFFICIAL_QUALIFICATION":
        raise ValueError("This entry prepares native P0 only; P2 requires the independently reviewed qualification gate")
    if any(model["extra_family"] != "gme-core" for model in plan["models"].values()):
        for dataset in plan["datasets"].values():
            if not dataset.get("qualification_target") or not dataset.get("native_anyhit"):
                raise ValueError("Native P0 needs an explicit target identity and native any-hit metric; CORE fraction scoring cannot substitute")
    for model in plan["models"].values():
        targets={row["benchmark"] for row in model["published_targets"]}
        for dataset in plan["datasets"].values():
            if dataset.get("qualification_target") not in targets:
                raise ValueError("Dataset is not among this exact checkpoint's preregistered published qualification benchmarks")
    return plan


def runtime_environment(spec):
    # Extend the shared receipt with actual loaded author-library paths and
    # compare the frozen package versions, including PEFT and OpenCLIP.
    extra={key:value for key,value in spec.get("environment",{}).items() if key in {"open-clip-torch","peft"}}
    reduced={**spec,"environment":{key:value for key,value in spec.get("environment",{}).items() if key not in extra}}
    output=original_environment(reduced)
    import torch
    output["matmul"]={"float32_precision":torch.get_float32_matmul_precision(),
        "allow_tf32":torch.backends.cuda.matmul.allow_tf32,"cudnn_allow_tf32":torch.backends.cudnn.allow_tf32}
    for name in ("open_clip","peft","flash_attn"):
        module=importlib.import_module(name)
        output["modules"][name]={"version":getattr(module,"__version__",None),"path":str(Path(module.__file__).resolve())}
    dependencies=json.loads((HERE/"dependencies.json").read_text())
    for name,expected in dependencies["modules"].items():
        module=importlib.import_module(name)
        if (str(Path(module.__file__).resolve()) != expected["path"] or
                getattr(module,"__version__",None) != expected["actual_version"]):
            raise ValueError("Actual imported module differs from the frozen dependency path/version: "+name)
    for key,value in extra.items():
        output[key]=version(key)
        if output[key] != value:
            raise ValueError("Extra baseline runtime differs: "+key)
    return output


def anyhit_from_scores(scores,positive_image_indices,ks=(1,5,10)):
    """Conventional any-positive recall; matches CLIC's ascending argsort tail."""
    import numpy as np
    scores=np.asarray(scores)
    if scores.ndim != 2 or not np.isfinite(scores).all() or len(positive_image_indices) != len(scores):
        raise ValueError("Native score matrix/positive rows differ")
    positives=np.asarray(positive_image_indices,dtype=np.int64)
    if len(scores)==0 or scores.shape[1]==0 or np.any(positives<0) or np.any(positives>=scores.shape[1]):
        raise ValueError("Native retrieval needs nonempty full scores and valid source image IDs")
    t2i=np.argsort(scores,axis=1)
    i2t=np.argsort(scores,axis=0)
    # Targets are conventional any-positive events, never caption fractions.
    result={"semantics":"any-positive Recall; numpy.argsort ascending then last K, matching released CLIC retrieval scorer",
            "t2i":{},"i2t":{}}
    for k in ks:
        result["t2i"][f"R@{k}"]=float(np.mean(np.any(t2i[:,-min(k,scores.shape[1]):]==positives[:,None],axis=1)))
        hits=[]
        for image_index in range(scores.shape[1]):
            gold=np.flatnonzero(positives==image_index)
            if not len(gold):
                raise ValueError("Native gallery contains an image without an original caption")
            hits.append(bool(np.isin(i2t[-min(k,len(scores)):,image_index],gold).any()))
        result["i2t"][f"R@{k}"]=float(np.mean(hits))
    result["t2i"]["queries"]=len(scores)
    result["i2t"]["queries"]=scores.shape[1]
    return result


def official_metrics(index_out,dataset_spec,bundle,labels):
    if not dataset_spec.get("native_anyhit"):
        return original_official_metrics(index_out,dataset_spec,bundle,labels)
    import numpy as np
    lookup={str(row["id"]):index for index,row in enumerate(bundle["gallery"])}
    indices=[]
    for row in labels:
        if row["ignore"] or len(row["positives"])!=1:
            raise ValueError("Native caption retrieval requires exactly its original image, without ignore filtering")
        indices.append(lookup[str(row["positives"][0])])
    # The unchanged producer saves the full normalized native-dtype GPU dot
    # matrix; scoring is CPU only and preserves the original candidate universe.
    scores=np.load(index_out/"official-similarity.npy",mmap_mode="r",allow_pickle=False)
    return anyhit_from_scores(scores,indices)


def install():
    # Configure only this executable process. Merely importing a CPU metric must
    # never replace the frozen shared runner's model gate for unrelated callers.
    R.models=SimpleNamespace(OfficialModel=A.ExtraOfficialModel,validate_spec=A.validate_spec)
    R.SOURCE_FILES+=("codex/retrieval_matrix_extra/adapters.py","codex/retrieval_matrix_extra/runner.py")
    R.source_gate=source_gate
    R.load_plan=load_plan
    R.runtime_environment=runtime_environment
    R.official_metrics=official_metrics

if __name__=="__main__":
    install()
    R.main()
