"""[codex] Insert explicitly frozen new launchers without changing old tasks."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from codex.retrieval_matrix_dispatch_next import controller as C
B = C.B


def frozen(path, commit):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT) or len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError("Explicit local source and full frozen commit are required")
    if subprocess.check_output(["git", "-C", str(ROOT), "show", commit+":"+str(path.relative_to(ROOT))]) != path.read_bytes():
        raise ValueError("Requested new plan/launcher differs from its fixed commit")
    return path


def task(request):
    commit = request["freeze_commit"]
    launcher, plan = [frozen(request[key], commit) for key in ("launcher", "plan")]
    output = Path(request["output"]).resolve()
    if not output.is_relative_to(ROOT/"dl"):
        raise ValueError("New output must be an isolated repository dl directory")
    completion = [str(Path(path).resolve()) for path in request["completion"]]
    allowed = {"index-meta.json", "receipt.json", "final-receipt.json", "complete.json"}
    if not completion or any(not Path(path).is_relative_to(output) or Path(path).name not in allowed
            or "results" in Path(path).parts for path in completion):
        raise ValueError("Explicit non-metric JSON completion receipts must stay under the new output")
    result = {"id": request["id"], "model": request["model"], "cwd": str(ROOT),
        "launcher": str(launcher), "launcher_sha256": B.sha(launcher), "plan": str(plan), "plan_sha256": B.sha(plan),
        "freeze_commit": commit, "output": str(output), "completion": completion,
        "min_free_mib": request["min_free_mib"], "priority": request.get("priority", 20),
        "command": ["bash", str(launcher), commit, str(plan), request["model"], str(output)]}
    if "completion_profile" in request:
        result["completion_profile"] = request["completion_profile"]
    return result


def build(old_path, old_commit, old_runtime, additions_path, *, before=None):
    old_path = frozen(old_path, old_commit)
    previous = json.loads(old_path.read_text())
    requests = json.loads(Path(additions_path).read_text())
    if requests.get("schema") != 1 or requests.get("owner") != "[codex]":
        raise ValueError("Root must explicitly enumerate the frozen new tasks")
    extra = [task(row) for row in requests["tasks"]]
    old_ids = [row["id"] for row in previous["tasks"]]
    if before is None:
        before = next((key for key in old_ids if "mcmr" in key), None)
    if before is not None and before not in old_ids:
        raise ValueError("Insertion anchor is not an existing original task")
    index = old_ids.index(before) if before else len(old_ids)
    result = copy.deepcopy(previous)
    result["tasks"][index:index] = extra
    result.update(controller="immutable-additive-next-v1",
        adoption={"schedule": C.entry(old_path), "freeze_commit": old_commit,
            "runtime": str(Path(old_runtime).resolve()), "retire_pending": []},
        additions={row["id"]: C.digest(row) for row in extra}, additions_input=C.entry(additions_path),
        insertion_before=before,
        resource_exceptions={row["id"]: C.siglip_exception(row) for row in extra if row["min_free_mib"] == 12288},
        controller_sources={"codex/retrieval_matrix_dispatch_next/"+name: B.sha(C.HERE/name) for name in C.OWN},
        policy="Every inherited task dictionary/history/token/receipt stays unchanged; explicit new IDs/outputs before MCMR; failed never retry; adopted and new workers share cap2/perGPU1.")
    result["controller_sources"].update({"codex/retrieval_matrix_dispatch_v3/"+name: B.sha(C.V3.HERE/name) for name in C.V3.OWN})
    C.validate(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("old-schedule", "old-freeze", "old-runtime", "additions", "out"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--before")
    args = parser.parse_args()
    output = Path(args.out)
    if output.exists():
        raise ValueError("Never overwrite an existing immutable schedule")
    value = build(args.old_schedule, args.old_freeze, args.old_runtime, args.additions, before=args.before)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)+"\n")
    print(json.dumps({"tasks": len(value["tasks"]), "sha256": B.sha(output), "requires_independent_review_and_commit": True}))


if __name__ == "__main__":
    main()
