"""[codex] Explicit API repair around unchanged frozen embedding producers."""
from __future__ import annotations

import argparse
import copy
import importlib
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from codex.retrieval_matrix import runner as R
from codex.retrieval_matrix_core_runtime_fix import compat

BASE_GATE, BASE_RUNTIME = R.source_gate, R.runtime_environment
OWN = ("compat.py", "runner.py", "build_plan.py", "run.sh", "PREREG.md", "test_compat.py", "test_plan.py")


def validate_repair(plan):
    repair = plan.get("runtime_repair", {})
    if repair.get("identity") != compat.IDENTITY:
        raise ValueError("The exact preregistered API repair is required")
    parent = R.read(R.checked_file(repair["parent_plan"]))
    if (not plan.get("models") or not set(plan["models"]) <= set(parent["models"])
            or plan.get("dependencies") != parent["dependencies"]):
        raise ValueError("The repair must retain its original models and isolated dependency bytes")
    for key, spec in plan["models"].items():
        original = copy.deepcopy(spec)
        if original.pop("runtime_compatibility", None) != compat.IDENTITY or original != parent["models"][key]:
            raise ValueError("API repair changed the frozen checkpoint, processor, batch or precision")
    if plan.get("phase") in {"P0", "MCMR-native-MMEB-v2"}:
        if any(plan.get(key) != parent.get(key) for key in ("phase", "settings", "datasets", "fixed_baseline_test")):
            raise ValueError("Technical repair must not change the scientific evaluation protocol")
    elif plan.get("phase") != "P2-qualified-fixed-baselines":
        raise ValueError("Unsupported phase for the explicit CORE/Qwen3 repair")
    return parent


def source_gate(path, plan, commit):
    proof = BASE_GATE(path, plan, commit)
    validate_repair(plan)
    for name in OWN:
        relative = "codex/retrieval_matrix_core_runtime_fix/"+name
        if plan.get("source_files", {}).get(relative) != R.sha(HERE/name):
            raise ValueError("The repair plan omits an executing source or preregistration")
    if subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain", "--", str(HERE.relative_to(ROOT))], text=True).strip():
        raise ValueError("The explicit repair must be clean and committed")
    for relative, expected in plan["dependencies"]["files"].items():
        if R.sha(ROOT/relative) != expected:
            raise ValueError("The unchanged pinned dependency payload differs: "+relative)
    return proof


def runtime_environment(spec):
    if (os.environ.get("RETRIEVAL_CENTRAL_OWNER") != "[codex]"
            or not os.environ.get("RETRIEVAL_CENTRAL_TASK") or not os.environ.get("CUDA_VISIBLE_DEVICES")):
        raise ValueError("Repaired GPU inference is submitted only through the root central dispatcher")
    actual = BASE_RUNTIME(spec)
    import torch
    if torch.cuda.device_count() != 1:
        raise ValueError("The repair requires exactly one assigned GPU")
    actual["runtime_compatibility"] = dict(compat.IDENTITY)
    return actual


def install(mode):
    # Executable-only mutation: importing the CPU checker has no side effects.
    R.models.OfficialModel = compat.CompatibleOfficialModel
    R.SOURCE_FILES += tuple("codex/retrieval_matrix_core_runtime_fix/"+name for name in OWN)
    R.source_gate, R.runtime_environment = source_gate, runtime_environment
    if mode == "mcmr":
        return importlib.import_module("codex.retrieval_matrix_mcmr.runner")
    if mode == "panel":
        importlib.import_module("codex.retrieval_matrix_panel.runner")
    return R


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mode", required=True, choices=("embedding", "mcmr", "panel"))
    args, remainder = parser.parse_known_args()
    install(args.mode).main(remainder)


if __name__ == "__main__":
    main()
