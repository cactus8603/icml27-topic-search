"""[codex] A missing dependency overlay without changing frozen Qwen2 science."""
from __future__ import annotations
import hashlib
import importlib
from importlib.metadata import version
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
BASE_REL="codex/retrieval_matrix_qwen2/p0-qwen2-batch1.json"
BASE_COMMIT="1c328a2c35c486f1768e9e9fea1c51f98dbb226d"
PARENTS={"embedding":(BASE_REL,BASE_COMMIT),
    "mcmr":("codex/retrieval_matrix_mcmr/p0-mcmr-qwen2-v2.json","65f4f4850a82adf283b34dfddc319b72657562e6")}
BATCHES={"embedding":"P0-Qwen2-einops-overlay-v1","mcmr":"MCMR-Qwen2-einops-overlay-v1"}
OVERLAY=ROOT/"dl/retrieval-matrix-qwen2-runtime-fix/deps"
OUTPUT_ROOT=ROOT/"dl/retrieval-matrix-qwen2-runtime-fix/runs"
MODULES={"torch":"torch","torchvision":"torchvision","transformers":"transformers",
    "tokenizers":"tokenizers","huggingface_hub":"huggingface-hub","qwen_vl_utils":"qwen-vl-utils",
    "av":"av","pytrec_eval":"pytrec-eval-terrier","numpy":"numpy","PIL":"Pillow",
    "accelerate":"accelerate","peft":"peft","flash_attn":"flash-attn","einops":"einops"}
OWN=("contracts.py","runner.py","build_plan.py","preflight.py","check_dependencies.py",
     "test_contracts.py","run.sh","central_task.sh","PREREG.md","README.md","overlay.json")


def sha(path):
    with Path(path).open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()


def read(path):return json.loads(Path(path).read_text())


def entry(path):
    path=Path(path).resolve();return {"path":str(path),"sha256":sha(path)}


def checked(item):
    path=Path(item["path"]).resolve()
    if not path.is_relative_to(ROOT) or sha(path)!=item["sha256"]:raise ValueError("Runtime fix evidence differs")
    return path


def output_path(value):
    path=Path(value).resolve()
    if path==OUTPUT_ROOT or not path.is_relative_to(OUTPUT_ROOT):
        raise ValueError("Use a new isolated runtime-fix output; preserve all original failed runs")
    return path


def parent(plan=None,*,mode="embedding"):
    mode=plan["runtime_fix"]["mode"] if plan is not None else mode
    relative,commit=PARENTS[mode]
    path=ROOT/relative
    original=subprocess.check_output(["git","-C",str(ROOT),"show",commit+":"+relative])
    if path.read_bytes()!=original:raise ValueError("Original frozen Qwen2 plan changed")
    if plan is not None and plan["runtime_fix"]["parent_plan"]!=entry(path):
        raise ValueError("Runtime repair names another original Qwen2 plan")
    return read(path)


def actual_modules():
    output={}
    for name,distribution in MODULES.items():
        module=importlib.import_module(name)
        output[name]={"path":str(Path(module.__file__).resolve()),
            "actual_version":getattr(module,"__version__",None),"distribution_version":version(distribution)}
    return output


def verify_overlay(manifest):
    if (manifest.get("package")!="einops" or manifest.get("version")!="0.8.2"
            or manifest.get("overlay")!=str(OVERLAY)):
        raise ValueError("Only the explicitly pinned missing einops overlay is authorized")
    actual={str(p.relative_to(ROOT)) for p in OVERLAY.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts and p.suffix!=".pyc"}
    if not actual or actual!=set(manifest["files"]):raise ValueError("Overlay payload coverage differs")
    for relative,expected in manifest["files"].items():
        path=(ROOT/relative).resolve()
        if not path.is_relative_to(OVERLAY) or sha(path)!=expected:raise ValueError("Missing dependency overlay bytes changed")
    checked(manifest["copy_receipt"])


def validate(plan,*,imports=True):
    base=parent(plan)
    for name in ("models","datasets","settings","dependencies","phase","gate","fixed_baseline_test"):
        if plan.get(name)!=base.get(name):raise ValueError("Runtime repair must preserve original "+name)
    mode=plan["runtime_fix"]["mode"]
    if (plan.get("runtime_batch")!=BATCHES[mode]
            or plan["runtime_fix"].get("parent_freeze_commit")!=PARENTS[mode][1]):
        raise ValueError("This launcher only supports its separately frozen dependency-repair batch")
    overlay_path=checked(plan["runtime_fix"]["overlay"]);manifest=read(overlay_path)
    if overlay_path!=HERE/"overlay.json":raise ValueError("Runtime repair must bind its own committed overlay manifest")
    verify_overlay(manifest)
    for relative,expected in base["dependencies"]["files"].items():
        path=(ROOT/relative).resolve()
        if not path.is_relative_to(ROOT/"dl/retrieval-matrix/deps") or sha(path)!=expected:
            raise ValueError("An original frozen dependency payload changed")
    for name,expected in base["dependencies"]["packages"].items():
        if version(name)!=expected:raise ValueError("Original dependency distribution changed: "+name)
    if imports:
        actual=actual_modules()
        if actual!=manifest["modules"]:raise ValueError("Actual Qwen2 module path/version differs from the repaired frozen runtime")
        if (actual["einops"]["path"]!=str(OVERLAY/"einops/__init__.py")
                or actual["einops"]["actual_version"]!="0.8.2"):
            raise ValueError("Actual einops must come from this isolated overlay")
    return manifest
