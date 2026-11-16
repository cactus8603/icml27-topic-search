"""[codex] P2 common panels; frozen P0 evidence is required before inference."""
from pathlib import Path
from importlib.metadata import version
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from codex.retrieval_matrix import runner as R
from codex.retrieval_matrix_panel import data, qualification, data_audit
from codex.retrieval_matrix_qwen2.adapters import Qwen2OfficialModel

base_gate, base_load = R.source_gate, R.load_data
base_labels, base_official, base_analyze = R.load_labels, R.official_metrics, R.analyze
base_environment = R.runtime_environment
BaseModel = R.models.OfficialModel
R.SOURCE_FILES += tuple("codex/retrieval_matrix_panel/"+name for name in
    ("runner.py", "data.py", "data_audit.py", "qualification.py", "build_plan.py", "run.sh", "PREREG.md"))
R.SOURCE_FILES += ("codex/retrieval_matrix_qwen2/adapters.py",)


def source_gate(path, plan, commit):
    proof = base_gate(path, plan, commit)
    status = subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain", "--", str(HERE.relative_to(ROOT))],text=True)
    if status.strip():
        raise ValueError("P2 panel production requires its own clean committed source")
    if plan.get("phase") != "P2-qualified-fixed-baselines":
        raise ValueError("This runner only supports qualified published fixed baselines")
    for key, spec in plan["models"].items():
        qualification.admit(plan["qualification"][key], spec)
    paths = [R.checked_file(plan[key]) for key in ("development_audit","development_manifest",
        "development_sources","development_images_summary","development_images_receipt")]
    data_audit.verify(*paths)
    data_audit.bind_datasets(plan,R.read(paths[1]))
    return proof


def model_factory(spec, **kwargs):
    if spec.get("variant") in {"gme-qwen2", "vlm2vec-v2-dora"}:
        return Qwen2OfficialModel(spec, **kwargs)
    return BaseModel(spec, **kwargs)


def runtime_environment(spec):
    reduced = {**spec,"environment":{k:v for k,v in spec.get("environment",{}).items() if k!="peft"}}
    actual = base_environment(reduced)
    if "peft" in spec.get("environment",{}):
        import peft
        actual["peft"] = version("peft")
        actual["modules"]["peft"]={"version":peft.__version__,"path":str(Path(peft.__file__).resolve())}
        if actual["peft"] != spec["environment"]["peft"]:
            raise ValueError("P2 PEFT runtime differs from qualified P0")
    return actual


def json_rows(path):
    return data.lf_rows(path) if path.suffix==".jsonl" else R.read(path)


def load_data(spec, **kwargs):
    if spec["format"] == "cola_single":
        return data.load_data(spec,R,**kwargs)
    return base_load(spec,**kwargs)


def load_labels(spec,bundle):
    if spec["format"]=="cola_single": return data.load_labels(spec,bundle,R)
    return base_labels(spec,bundle)


def analyze(out,spec,bundle,settings,*,index_out=None):
    if spec["format"]!="cola_single":
        return base_analyze(out,spec,bundle,settings,index_out=index_out)
    if index_out is not None:
        raise ValueError("COLA rerank analysis must bind its separate qualified P3 producer")
    identity=R.read(out/"provenance.json")
    if identity["dataset_fingerprint"]!=bundle["fingerprint"]:
        raise ValueError("COLA index inputs differ from scoring inputs")
    _,order,meta=R.validate_index(out,identity,bundle,settings)
    report,detail=R.metrics.evaluate(order,[row["id"] for row in bundle["gallery"]],load_labels(spec,bundle))
    report.update(schema=1,model=identity["model"],dataset=identity["dataset"],gallery=len(bundle["gallery"]),
        raw_queries=bundle["raw_queries"],excluded=[],retained_empty=[],
        protocol="full gallery admission; remove non-hard images only during scoring, without refill; official threshold-group AP is authoritative",
        label_semantics="original COLA binary class label and class-specific hard_list",
        provenance={"index_meta_sha256":R.sha(out/"index-meta.json"),"index":meta},
        official=data.official_metrics(out,spec,bundle,R))
    R.atomic_json(out/"results/report.json",report)
    R.atomic_json(out/"results/per-query.json",detail)


R.source_gate=source_gate
R.models.OfficialModel=model_factory
R.runtime_environment=runtime_environment
R.json_rows=json_rows
R.load_data=load_data
R.load_labels=load_labels
R.analyze=analyze

if __name__=="__main__": R.main()
