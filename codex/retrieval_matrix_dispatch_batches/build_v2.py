"""[codex] Add native SigLIP2 P0 without changing the first fourteen tasks."""
import argparse
import copy
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from codex.retrieval_matrix_dispatch.dispatch import sha, validate
from codex.retrieval_matrix_native_p0.runner import load_plan


def build(native_commit):
    if not re.fullmatch(r"[0-9a-f]{40}", native_commit):
        raise ValueError("Native producer requires a full source freeze commit")
    old_path = ROOT/"codex/retrieval_matrix_dispatch/schedule.json"
    if sha(old_path) != "efa709f5f0259ec569addbeaf3b773daf4e90cdc7ac0c144ebe0ce7869da6498":
        raise ValueError("The original fourteen-task schedule changed")
    original = json.loads(old_path.read_text())
    if len(original["tasks"]) != 14:
        raise ValueError("Preserve the original fourteen tasks verbatim")
    folder = ROOT/"codex/retrieval_matrix_native_p0"
    plan_path, launcher = folder/"p0-siglip2.json", folder/"central_task.sh"
    native = load_plan(plan_path)
    if native["model_key"] != "siglip2" or native["model_spec"]["family"] != "CLIP":
        raise ValueError("The additional task must be the reviewed native SigLIP2 P0")
    for path in (plan_path, launcher):
        relative = str(path.relative_to(ROOT))
        if subprocess.check_output(["git", "-C", str(ROOT), "show", native_commit+":"+relative]) != path.read_bytes():
            raise ValueError("Native task differs from its own source freeze")
    out = ROOT/"dl/retrieval-matrix-native-p0/runs/p0-siglip2-v1-20261116"
    command = ["bash", str(launcher), native_commit, str(plan_path), "siglip2", str(out)]
    task = {"id": "native-siglip2", "model": "siglip2", "command": command, "cwd": str(ROOT),
        "launcher": str(launcher), "launcher_sha256": sha(launcher), "freeze_commit": native_commit,
        "plan": str(plan_path), "plan_sha256": sha(plan_path), "output": str(out),
        "completion": [str(out/name/"index/receipt.json") for name in ("coco", "flickr")],
        "min_free_mib": 20480, "priority": 14}
    result = copy.deepcopy(original)
    # Finish broad official P0 coverage before the six large fused-MCMR jobs.
    position = next(i for i, row in enumerate(result["tasks"]) if row["id"].startswith("mcmr-"))
    result["tasks"].insert(position, task)
    if [row for row in result["tasks"] if row["id"] != task["id"]] != original["tasks"]:
        raise ValueError("Existing task identity/order/argv/output/history cannot change")
    result.update(parent_schedule={"path": str(old_path), "sha256": sha(old_path)},
        supersedes_runtime=str(ROOT/"dl/retrieval-matrix-dispatch/central-v1"),
        migration_policy="Only retire the exact old central process after stopping it briefly and proving all fourteen tasks remain pending, no attempts or workers. Otherwise resume the old scheduler and defer migration. Preserve its state/events and record every action.",
        extension="One native SigLIP2 P0 task; fixed full FP32 image4/text32; conservative20GiB admission; no GPU analysis or automatic retries")
    validate(result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--native-freeze", required=True)
    p.add_argument("--out", type=Path, default=Path(__file__).with_name("central-v2.json"))
    args = p.parse_args()
    if args.out.exists():
        raise ValueError("Do not overwrite a frozen central batch")
    result = build(args.native_freeze)
    args.out.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2)+"\n")
    print(json.dumps({"tasks": len(result["tasks"]), "sha256": sha(args.out), "requires_commit": True}))
