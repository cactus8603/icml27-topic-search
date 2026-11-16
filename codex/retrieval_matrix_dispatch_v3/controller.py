"""[codex] Adopt existing workers and append reviewed technical-repair tasks.

GPU leases remain in the original workers and their children. Only the central
parent is replaced. Adopted jobs count toward the same global cap of two.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from codex.retrieval_matrix_dispatch import dispatch as B

ORIGINAL_VALIDATE = B.validate
OWN = ("controller.py", "build_schedule.py", "test_controller.py", "PREREG.md")


def validate(schedule):
    checked = copy.deepcopy(schedule)
    for task in checked["tasks"]:
        if task["min_free_mib"] == 12288:
            if task["id"] != "native-siglip2" or task["model"] != "siglip2":
                raise ValueError("The additional12GiB class is reserved for the full4.55GB FP32 SigLIP2")
            task["min_free_mib"] = 20480
    ORIGINAL_VALIDATE(checked)
    if schedule.get("controller") != "adopt-reviewed-repair-v3":
        raise ValueError("Explicit v3 controller protocol is required")


def gate(path, schedule, commit):
    before = B.validate
    B.validate = validate
    try:
        proof = B.freeze_gate(path, schedule, commit)
    finally:
        B.validate = before
    for name in OWN:
        relative = "codex/retrieval_matrix_dispatch_v3/"+name
        if (schedule["controller_sources"].get(relative) != B.sha(ROOT/relative)
                or subprocess.check_output(["git", "-C", str(ROOT), "show", commit+":"+relative]) != (ROOT/relative).read_bytes()):
            raise ValueError("Executing v3 controller differs from its source freeze")
    if subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain", "--", str(HERE.relative_to(ROOT))], text=True).strip():
        raise ValueError("The v3 controller directory must stay clean")
    return proof


def seed(schedule, schedule_sha, previous):
    """Preserve all prior outcomes; only known unstarted buggy tasks retire."""
    origin = schedule["adoption"]
    old_path = Path(origin["schedule"]["path"])
    if B.sha(old_path) != origin["schedule"]["sha256"] or previous["schedule_sha256"] != origin["schedule"]["sha256"]:
        raise ValueError("Imported runtime belongs to another frozen central schedule")
    old_schedule = json.loads(old_path.read_text())
    old_tasks = {row["id"]: row for row in old_schedule["tasks"]}
    new_tasks = {row["id"]: row for row in schedule["tasks"]}
    if set(previous["tasks"]) != set(old_tasks) or not set(old_tasks) <= set(new_tasks):
        raise ValueError("A handoff must preserve every previous task and its history")
    tasks = {key: {"status": "pending"} for key in new_tasks}
    for key, old in old_tasks.items():
        actual = copy.deepcopy(new_tasks[key])
        if key == "native-siglip2":
            actual["min_free_mib"] = old["min_free_mib"]
        if actual != old:
            raise ValueError("Adoption changed an original task command, source, output or protocol")
        row = copy.deepcopy(previous["tasks"][key])
        if row["status"] not in B.LIVE | B.TERMINAL | {"pending"}:
            raise ValueError("Unknown previous task state cannot be reset")
        if row["status"] in B.LIVE:
            old_runtime = row.get("adopted_runtime", origin["runtime"])
            expected = str(Path(old_runtime)/(key+"-attempt.json"))
            if row.get("receipt") != expected or row.get("command") != old["command"]:
                raise ValueError("Adopted active job lacks its original command/receipt identity")
            row.setdefault("adopted_schedule_sha256", previous["schedule_sha256"])
            row.setdefault("adopted_runtime", old_runtime)
        if key in origin["retire_pending"]:
            if row["status"] == "pending":
                row.update(status="cancelled", retired_without_start=True,
                    reason="Known synthetic runtime failure; separately frozen repaired task replaces this unstarted command")
            elif row["status"] in B.LIVE:
                raise ValueError("Never retire or replay a model that has already started")
        tasks[key] = row
    return {"schema": 1, "owner": "[codex]", "schedule_sha256": schedule_sha,
        "created_at": B.now(), "adopted_from_schedule_sha256": previous["schedule_sha256"], "tasks": tasks}


class Dispatcher(B.Dispatcher):
    def __init__(self, schedule, schedule_sha, runtime, **kwargs):
        runtime = Path(runtime)
        if not (runtime/"state.json").exists():
            migration = json.loads((runtime/"migration.json").read_text())
            previous_path = runtime/"previous-state.json"
            if (migration.get("new_schedule_sha256") != schedule_sha
                    or migration.get("previous_state_sha256") != B.sha(previous_path)
                    or migration.get("old_parent_retired") is not True):
                raise ValueError("Root must complete the exact parent-only handoff before v3 starts")
            B.atomic_json(runtime/"state.json", seed(schedule, schedule_sha, json.loads(previous_path.read_text())))
        real_popen = kwargs.pop("popen", subprocess.Popen)
        def popen(argv, **options):
            argv = list(argv)
            if argv[2] != str(B.HERE/"dispatch.py") or argv[3] != "worker":
                raise ValueError("Unexpected worker launcher")
            argv[2] = str(HERE/"controller.py")
            return real_popen(argv, **options)
        before = B.validate
        B.validate = validate
        try:
            super().__init__(schedule, schedule_sha, runtime, popen=popen, **kwargs)
        finally:
            B.validate = before

    def reconcile(self):
        for task in self.schedule["tasks"]:
            key = task["id"]
            row = self.state["tasks"][key]
            if row["status"] not in B.LIVE:
                continue
            receipt = Path(row["receipt"])
            if receipt.exists():
                result = json.loads(receipt.read_text())
                expected_sha = row.get("adopted_schedule_sha256", self.schedule_sha)
                if (result.get("token") != row["token"] or result.get("schedule_sha256") != expected_sha
                        or result.get("task_id") != key):
                    raise ValueError("Original or adopted worker receipt identity differs")
                row.update(result)
                if row["status"] in B.TERMINAL:
                    if key in self.children:
                        self.children.pop(key).wait()
                    if key in self.leases:
                        self.leases.pop(key).__exit__(None, None, None)
                elif not B.alive(row.get("worker")) and not B.alive(row.get("child")):
                    row.update(status="failed", error="Original worker/child disappeared without terminal evidence; no retry")
                    B.append_json(self.journal, {"event": "lost_worker", "task_id": key})
                    if key in self.leases:
                        self.leases.pop(key).__exit__(None, None, None)
            else:
                row["status"] = "uncertain"
        self.save()

    def stop_own(self):
        """Signal only workers spawned by this controller instance."""
        self.reconcile()
        for key, child in list(self.children.items()):
            row = self.state["tasks"][key]
            worker = row.get("worker")
            if (row["status"] in B.LIVE and worker
                    and worker["pid"] == child.pid and B.alive(worker)):
                B.os.kill(worker["pid"], signal.SIGTERM)
                B.append_json(self.journal, {"event": "stop_own_worker", "task_id": key, "worker": worker})


def main():
    B.validate = validate
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("run", "preflight", "worker"))
    p.add_argument("--schedule", type=Path)
    p.add_argument("--freeze-commit")
    p.add_argument("--runtime", type=Path)
    p.add_argument("--config", type=Path)
    p.add_argument("--gpu-fd", type=int)
    args = p.parse_args()
    if args.action == "worker":
        config = json.loads(args.config.read_text())
        proof = config["execution_proof"]
        path = Path(proof["schedule"])
        gate(path, json.loads(path.read_text()), proof["freeze_commit"])
        return B.worker(args.config, args.gpu_fd)
    path = args.schedule.resolve()
    schedule = json.loads(path.read_text())
    proof = gate(path, schedule, args.freeze_commit)
    if args.action == "preflight":
        print(json.dumps(proof))
        return 0
    with B.FileLock(ROOT/"dl/retrieval3090/runs/.gpu-locks/central-dispatch.lock"):
        dispatcher = Dispatcher(schedule, B.sha(path), args.runtime.resolve(),
            execution_proof={"schedule": str(path), "freeze_commit": args.freeze_commit})
        stopping = False
        def stop(_signum, _frame):
            nonlocal stopping
            stopping = True
            dispatcher.stopping = True
        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, stop)
        B.append_json(dispatcher.journal, {"event": "scheduler_start", "freeze_commit": args.freeze_commit,
            "pid": B.os.getpid(), "adopted_from": schedule["adoption"]["runtime"]})
        while True:
            if stopping:
                dispatcher.stop_own()
                return 130
            dispatcher.tick()
            if all(row["status"] in B.TERMINAL for row in dispatcher.state["tasks"].values()):
                B.append_json(dispatcher.journal, {"event": "scheduler_terminal"})
                return 1 if any(row["status"] == "failed" for row in dispatcher.state["tasks"].values()) else 0
            time.sleep(5)


if __name__ == "__main__":
    raise SystemExit(main())
