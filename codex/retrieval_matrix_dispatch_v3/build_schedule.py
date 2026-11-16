"""[codex] Reviewed repair batch and exact existing-worker adoption policy."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from codex.retrieval_matrix_dispatch_v3.controller import B, OWN, validate


def task(key, model, folder, plan_name, commit, out_folder, *, mcmr=False):
    folder = ROOT/"codex"/folder
    plan = folder/plan_name
    launcher = folder/("central_task.sh" if folder.name.endswith("qwen2_runtime_fix") else "run.sh")
    for path in (plan, launcher):
        if subprocess.check_output(["git", "-C", str(ROOT), "show", commit+":"+str(path.relative_to(ROOT))]) != path.read_bytes():
            raise ValueError("Repair plan/launcher is not at its own frozen commit")
    output = ROOT/out_folder/model
    completion = [str(output/"mcmr/index-meta.json")] if mcmr else [str(output/name/"index-meta.json") for name in ("coco", "negbench", "flickr")]
    return {"id": key, "model": model, "command": ["bash", str(launcher), commit, str(plan), model, str(output)],
        "cwd": str(ROOT), "launcher": str(launcher), "launcher_sha256": B.sha(launcher),
        "freeze_commit": commit, "plan": str(plan), "plan_sha256": B.sha(plan),
        "output": str(output), "completion": completion,
        "min_free_mib": 20480 if model.endswith("8") else 8192, "priority": 20}


def build(core_commit, qwen2_commit, qwen2_embedding_plan, qwen2_mcmr_plan):
    old_path = ROOT/"codex/retrieval_matrix_dispatch_batches/central-v2.json"
    if B.sha(old_path) != "199cfd9fd64d062f483a65fd532a3ebadcd58dcf4516815e3006233d41cf972c":
        raise ValueError("Previous15-task schedule changed")
    original = json.loads(old_path.read_text())
    result = copy.deepcopy(original)
    extra = []
    for model in ("core2", "qwen2", "core8", "qwen8"):
        extra.append(task("fixed-embedding-"+model, model, "retrieval_matrix_core_runtime_fix", "p0-qwen3-fixed-v1.json",
            core_commit, "dl/retrieval-matrix-core-runtime-fix/runs/p0-v1-20261116"))
    for model in ("gme2", "vlm2vec2"):
        extra.append(task("fixed-embedding-"+model, model, "retrieval_matrix_qwen2_runtime_fix", qwen2_embedding_plan,
            qwen2_commit, "dl/retrieval-matrix-qwen2-runtime-fix/runs/p0-v1-20261116"))
    for model in ("core2", "qwen2", "core8", "qwen8"):
        extra.append(task("fixed-mcmr-"+model, model, "retrieval_matrix_core_runtime_fix", "p0-mcmr-qwen3-fixed-v1.json",
            core_commit, "dl/retrieval-matrix-core-runtime-fix/runs/mcmr-v1-20261116", mcmr=True))
    for model in ("gme2", "vlm2vec2"):
        extra.append(task("fixed-mcmr-"+model, model, "retrieval_matrix_qwen2_runtime_fix", qwen2_mcmr_plan,
            qwen2_commit, "dl/retrieval-matrix-qwen2-runtime-fix/runs/mcmr-v1-20261116", mcmr=True))
    for row in result["tasks"]:
        if row["id"] == "native-siglip2":
            row["min_free_mib"] = 12288
    # Preserve every old row/history, with new official correctness tasks before
    # newly added large fused-MCMR work. Old known-bug pending rows retire at seed.
    result["tasks"].extend(extra)
    result.update(controller="adopt-reviewed-repair-v3",
        adoption={"schedule": {"path": str(old_path), "sha256": B.sha(old_path)},
            "runtime": str(ROOT/"dl/retrieval-matrix-dispatch/central-v2"),
            "retire_pending": ["embedding-core8", "embedding-qwen8", "mcmr-core8", "mcmr-qwen8"]},
        controller_sources={"codex/retrieval_matrix_dispatch_v3/"+name: B.sha(ROOT/"codex/retrieval_matrix_dispatch_v3"/name) for name in OWN},
        policy="Preserve live original workers/receipts/inherited GPU leases, all terminal history, globalcap2/perGPU1. Failed tasks never replay; new explicitly frozen repair tasks use new IDs, refs and outputs.",
        resource_disclosure="SigLIP2 full FP32 weights are4.55GB;12GiB admission retains substantial activation/workspace margin for fixed384/image4/text32. Model settings do not change; GPU smoke records actual peak and never falls back.")
    validate(result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--core-freeze", required=True)
    p.add_argument("--qwen2-freeze", required=True)
    p.add_argument("--qwen2-embedding-plan", required=True)
    p.add_argument("--qwen2-mcmr-plan", required=True)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    if args.out.exists():
        raise ValueError("Never overwrite a central batch")
    result = build(args.core_freeze, args.qwen2_freeze, args.qwen2_embedding_plan, args.qwen2_mcmr_plan)
    args.out.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2)+"\n")
    print(json.dumps({"tasks": len(result["tasks"]), "sha256": B.sha(args.out), "requires_commit": True}))
