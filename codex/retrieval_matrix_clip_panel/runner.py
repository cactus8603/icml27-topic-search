"""[codex] Qualified CLIP panel producer; native P0 gate runs before all stages."""
from pathlib import Path
import os
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from codex.retrieval_matrix import runner as R
from codex.retrieval_matrix_extra import runner as E
from codex.retrieval_matrix_native_p0.models import Siglip2
from codex.retrieval_matrix_panel import data, data_audit
from codex.retrieval_matrix_clip_panel import contracts

base_gate, base_load = R.source_gate, R.load_data
base_labels, base_analyze, base_json_rows = R.load_labels, R.analyze, R.json_rows


def source_gate(path, plan, commit):
    proof = base_gate(path, plan, commit)
    contracts.admit_models(plan)
    contracts.check_declared_sources(plan, ROOT)
    for folder in (HERE, ROOT/"codex/retrieval_matrix_extra", ROOT/"codex/retrieval_matrix_native_p0", ROOT/"codex/retrieval_matrix_panel"):
        status = subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain", "--", str(folder.relative_to(ROOT))], text=True)
        if status.strip():
            raise ValueError("CLIP panel requires every executing source directory to be clean")
    # E's source gate also verifies all pinned isolated dependency payloads.
    E.source_gate(path, plan, commit)
    paths = [R.checked_file(plan[key]) for key in ("development_audit", "development_manifest",
             "development_sources", "development_images_summary", "development_images_receipt")]
    data_audit.verify(*paths)
    data_audit.bind_datasets(plan, R.read(paths[1]))
    return proof


def load_data(spec, **kwargs):
    return data.load_data(spec, R, **kwargs) if spec["format"] == "cola_single" else base_load(spec, **kwargs)


def load_labels(spec, bundle):
    return data.load_labels(spec, bundle, R) if spec["format"] == "cola_single" else base_labels(spec, bundle)


def runtime_environment(spec):
    if (os.environ.get("RETRIEVAL_CENTRAL_OWNER") != "[codex]"
            or not os.environ.get("RETRIEVAL_CENTRAL_TASK") or not os.environ.get("CUDA_VISIBLE_DEVICES")):
        raise ValueError("CLIP GPU production must use the root central dispatcher")
    import torch
    # Retain the qualified native SigLIP2 forward precision; the shared panel
    # does not change the encoder to the independent extra preparation wrapper.
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(2026111665)
    if torch.cuda.device_count() != 1:
        raise ValueError("CLIP panel requires exactly one assigned CUDA device")
    return E.runtime_environment(spec)


def analyze(out, spec, bundle, settings, *, index_out=None):
    if spec["format"] != "cola_single":
        return base_analyze(out, spec, bundle, settings, index_out=index_out)
    if index_out is not None:
        raise ValueError("CLIP panel analysis cannot substitute for the separately qualified P3 runner")
    identity = R.read(out/"provenance.json")
    if identity["dataset_fingerprint"] != bundle["fingerprint"]:
        raise ValueError("COLA CLIP index differs from scoring input")
    _, order, meta = R.validate_index(out, identity, bundle, settings)
    report, detail = R.metrics.evaluate(order, [row["id"] for row in bundle["gallery"]], load_labels(spec, bundle))
    report.update(schema=1, model=identity["model"], dataset=identity["dataset"], gallery=len(bundle["gallery"]),
        raw_queries=bundle["raw_queries"], excluded=[], retained_empty=[],
        protocol="full-gallery admission; hard_list mask only during scoring, without refill; threshold-group AP authoritative",
        provenance={"index_meta_sha256": R.sha(out/"index-meta.json"), "index": meta},
        official=data.official_metrics(out, spec, bundle, R))
    R.atomic_json(out/"results/report.json", report)
    R.atomic_json(out/"results/per-query.json", detail)


def install():
    R.models = SimpleNamespace(OfficialModel=Siglip2, validate_spec=contracts.validate_model)
    R.SOURCE_FILES += tuple("codex/retrieval_matrix_clip_panel/"+name for name in
        ("runner.py", "contracts.py", "build_plan.py", "run.sh", "PREREG.md"))
    R.SOURCE_FILES += ("codex/retrieval_matrix_panel/data.py", "codex/retrieval_matrix_panel/data_audit.py",
                       "codex/retrieval_matrix_extra/adapters.py", "codex/retrieval_matrix_extra/runner.py",
                       "codex/retrieval_matrix_native_p0/qualification.py")
    R.source_gate = source_gate
    R.runtime_environment = runtime_environment
    R.json_rows = lambda path: data.lf_rows(path) if path.suffix == ".jsonl" else base_json_rows(path)
    R.load_data, R.load_labels, R.analyze = load_data, load_labels, analyze


if __name__ == "__main__":
    install()
    R.main()
