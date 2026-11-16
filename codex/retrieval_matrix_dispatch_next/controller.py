"""[codex] Immutable additive batches; user-authorized global cap4/perGPU1.

The base dispatcher is loaded privately. No imported live module's validator,
worker gate, or class is modified by this wrapper.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from codex.retrieval_matrix_dispatch_v3 import controller as V3

_spec = importlib.util.spec_from_file_location("next_private_base_dispatcher", ROOT/"codex/retrieval_matrix_dispatch/dispatch.py")
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)
_base_validate, _base_gate = B.validate, B.freeze_gate
OWN = ("controller.py", "build_schedule.py", "handoff.py", "test_next.py", "PREREG.md")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def checked(item):
    path = Path(item["path"]).resolve()
    if B.sha(path) != item["sha256"]:
        raise ValueError("Explicit frozen evidence changed: " + str(path))
    return path


def entry(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": B.sha(path)}


def inherited(schedule):
    path = checked(schedule["adoption"]["schedule"])
    return {task["id"]: task for task in json.loads(path.read_text())["tasks"]}


def siglip_exception(task):
    plan = json.loads(Path(task["plan"]).read_text())
    spec = plan.get("model_spec", {})
    reference = json.loads((ROOT/"codex/retrieval_matrix_native_p0/p0-siglip2.json").read_text())["model_spec"]
    if task["model"] != "siglip2" or plan.get("model_key") != "siglip2" or spec != reference:
        raise ValueError("12GiB is reserved for the exact fixed full-FP32 SigLIP2 payload and image4/text32 profile")
    return {"task_sha256": digest(task), "plan": entry(task["plan"]), "model_spec_sha256": digest(spec)}


def validate(schedule):
    if schedule.get("max_live") != 4:
        raise ValueError("The new user-authorized central schedule must have global cap4")
    old = inherited(schedule)
    current = {task["id"]: task for task in schedule["tasks"]}
    if (schedule.get("controller") != "immutable-additive-next-v1"
            or schedule["adoption"].get("retire_pending") != [] or not old.keys() <= current.keys()
            or any(current[key] != task for key, task in old.items())
            or [task["id"] for task in schedule["tasks"] if task["id"] in old] != list(old)):
        raise ValueError("Additive adoption must preserve every old task dictionary/order and retire no task")
    additions = schedule.get("additions", {})
    if set(additions) != current.keys()-old.keys() or any(additions[key] != digest(current[key]) for key in additions):
        raise ValueError("Every new task needs its explicit immutable addition digest")
    normalized = copy.deepcopy(schedule)
    # Only the legacy schema check sees cap2. Execution below uses the real
    # integer live_count and explicit schedule.max_live=4 without scaling it.
    normalized["max_live"] = 2
    exceptions = {}
    for task in normalized["tasks"]:
        actual = current[task["id"]]
        if actual.get("completion_profile") not in {None, "native-reranker-v1"}:
            raise ValueError("Unknown non-metric completion adapter")
        if actual["id"] not in old:
            phase = json.loads(Path(actual["plan"]).read_text()).get("phase")
            if phase == "P0-native-reranker-mcq-v1" and actual.get("completion_profile") != "native-reranker-v1":
                raise ValueError("Native reranker requires its exact non-metric completion adapter")
        if task["min_free_mib"] == 12288:
            if task["id"] not in old:
                exceptions[task["id"]] = siglip_exception(current[task["id"]])
            task["min_free_mib"] = 20480
        if current[task["id"]].get("completion_profile") == "native-reranker-v1":
            require_reranker(current[task["id"]])
    if schedule.get("resource_exceptions", {}) != exceptions:
        raise ValueError("12GiB exceptions must bind the exact new SigLIP2 plan, model spec and task")
    _base_validate(normalized)


def require_reranker(task):
    plan = json.loads(Path(task["plan"]).read_text())
    variant = plan.get("variants", {}).get(task["model"], {})
    if (task["model"] not in {"qwen2", "qwen8", "jina_m0"} or plan.get("state") != "FROZEN"
            or plan.get("reviewed") is not True or variant.get("model", {}).get("verified_protocol") is not True
            or sum(value.get("verified_protocol_match") is True for value in variant.get("targets", {}).values()) < 2
            or task["min_free_mib"] != (8192 if task["model"] == "jina_m0" else 20480)):
        raise ValueError("Only the three independently approved native reranker variants may be dispatched")
    return variant


def completed(task):
    if task.get("completion_profile") != "native-reranker-v1":
        return B.completed_output(task)
    variant = require_reranker(task)
    files = task["completion"]
    if files != [str(Path(task["output"])/"scores/complete.json")]:
        raise ValueError("Native reranker completion must be its non-metric producer receipt")
    path = Path(files[0])
    if not path.exists():
        return False
    value = json.loads(path.read_text())
    identity = value.get("identity", {})
    if (value.get("state") != "COMPLETE" or identity.get("variant") != task["model"]
            or identity.get("variant_spec") != variant or identity.get("plan_sha256") != task["plan_sha256"]
            or identity.get("freeze_commit") != task["freeze_commit"]
            or (value.get("examples"), value.get("distinct_pairs"), value.get("original_images")) != (15912, 58891, 9425)):
        raise ValueError("Existing native reranker completion differs from its exact variant, plan/ref or full denominator")
    return True


def gate(path, schedule, commit, *, check_legacy=True):
    proof = _base_gate(path, schedule, commit, check_legacy=check_legacy)
    additions_path = checked(schedule["additions_input"])
    if subprocess.check_output(["git", "-C", str(ROOT), "show", commit+":"+str(additions_path.relative_to(ROOT))]) != additions_path.read_bytes():
        raise ValueError("Explicit additions request must be included in the final schedule freeze")
    sources = {"codex/retrieval_matrix_dispatch_next/"+name for name in OWN}
    sources |= {"codex/retrieval_matrix_dispatch_v3/"+name for name in V3.OWN}
    if set(schedule.get("controller_sources", {})) != sources:
        raise ValueError("All executing additive controller, handoff and adopted-method source must be frozen")
    for name in sources:
        source = ROOT/name
        if (schedule["controller_sources"][name] != B.sha(source)
                or subprocess.check_output(["git", "-C", str(ROOT), "show", commit+":"+name]) != source.read_bytes()):
            raise ValueError("Additive controller differs from its frozen source: " + name)
    if subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain", "--", str(HERE.relative_to(ROOT))], text=True).strip():
        raise ValueError("Additive controller must be clean and committed")
    return proof


def seed(schedule, schedule_sha, previous):
    validate(schedule)
    # v3's one historical SigLIP resource exception cannot apply here because
    # validate above already requires every inherited task byte-for-byte equal.
    return V3.seed(schedule, schedule_sha, previous)


class Dispatcher(B.Dispatcher):
    reconcile = V3.Dispatcher.reconcile
    stop_own = V3.Dispatcher.stop_own

    def __init__(self, schedule, schedule_sha, runtime, **kwargs):
        runtime = Path(runtime)
        if not (runtime/"state.json").exists():
            migration = json.loads((runtime/"migration.json").read_text())
            previous = runtime/"previous-state.json"
            if (migration.get("old_parent_retired") is not True or migration.get("new_schedule_sha256") != schedule_sha
                    or migration.get("previous_state_sha256") != B.sha(previous)):
                raise ValueError("Root must prove the exact parent-only handoff before additive dispatch")
            B.atomic_json(runtime/"state.json", seed(schedule, schedule_sha, json.loads(previous.read_text())))
        actual_popen = kwargs.pop("popen", subprocess.Popen)
        def popen(argv, **options):
            argv = list(argv)
            if argv[2:4] != [str(B.HERE/"dispatch.py"), "worker"]:
                raise ValueError("Unexpected base worker command")
            argv[2] = str(HERE/"controller.py")
            return actual_popen(argv, **options)
        kwargs.setdefault("complete", completed)
        super().__init__(schedule, schedule_sha, runtime, popen=popen, **kwargs)

    def tick(self):
        """Copy of frozen base admission; only its two cap2 limits become cap4.

        Global helper references use the private B module. Lease admission,
        checkpoint identity, STARTING-before-Popen, and the post-Popen failure
        branch retain the original control flow.
        """
        self.reconcile()
        limit = self.schedule["max_live"]
        if self.stopping or self.live_count() >= limit:
            return
        try:
            snapshot = self.query()
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            self.prior = {}
            B.append_json(self.journal, {"event": "query_failure", "error": str(exc)})
            return
        now_value = self.clock()
        occupied = {row["gpu"]["uuid"] for row in self.state["tasks"].values() if row["status"] in B.LIVE and "gpu" in row}
        for task in self.schedule["tasks"]:
            if self.stopping or self.live_count() >= limit:
                break
            key = task["id"]
            row = self.state["tasks"][key]
            if row["status"] != "pending":
                continue
            if self.complete(task):
                row.update(status="skipped_complete", finished_at=B.now())
                B.append_json(self.journal, {"event": "skip_complete", "task_id": key})
                continue
            available = [gpu for gpu in snapshot if gpu["uuid"] not in occupied and B.eligible(gpu, task)]
            current = {gpu["uuid"] for gpu in available}
            for prior_key in list(self.prior):
                if prior_key[0] == key and prior_key[1] not in current:
                    del self.prior[prior_key]
            for gpu in sorted(available, key=lambda x: (x["free_mib"], x["index"])):
                pair = (key, gpu["uuid"])
                if pair not in self.prior:
                    self.prior[pair] = now_value
                    continue
                if now_value-self.prior[pair] < 5:
                    continue
                lease = B.FileLock(Path(self.schedule["gpu_lock_dir"])/(gpu["uuid"]+".lock"))
                try:
                    spawned = None
                    lease.__enter__()
                except BlockingIOError:
                    continue
                try:
                    final = next((item for item in self.query() if item["uuid"] == gpu["uuid"]), None)
                    if self.stopping:
                        lease.__exit__(None, None, None)
                        return
                    if final is None or not B.eligible(final, task):
                        self.prior.pop(pair, None)
                        lease.__exit__(None, None, None)
                        continue
                    if self.complete(task):
                        row.update(status="skipped_complete", finished_at=B.now())
                        lease.__exit__(None, None, None)
                        break
                    token = B.os.urandom(16).hex()
                    attempt = self.runtime/(key+"-attempt.json")
                    config = self.runtime/(key+"-worker.json")
                    B.atomic_json(config, {"task": task, "gpu": final, "token": token, "schedule_sha256": self.schedule_sha,
                        "receipt": str(attempt), "journal": str(self.journal), "shared_log": self.schedule["shared_log"],
                        "execution_proof": self.execution_proof})
                    row.update(status="starting", token=token, receipt=str(attempt), gpu=final, command=task["command"])
                    self.save()
                    B.append_json(self.journal, {"event": "admitted", "task_id": key, "gpu": final, "threshold_mib": task["min_free_mib"]})
                    log = (self.runtime/(key+".log")).open("ab", buffering=0)
                    try:
                        child = self.popen([sys.executable, "-B", str(B.HERE/"dispatch.py"), "worker", "--config", str(config), "--gpu-fd", str(lease.fd)],
                            cwd=str(ROOT), shell=False, start_new_session=True, pass_fds=(lease.fd,), stdout=log, stderr=subprocess.STDOUT)
                        spawned = child
                    finally:
                        log.close()
                    row["worker"] = B.process_token(child.pid)
                    self.children[key] = child
                    self.leases[key] = lease
                    occupied.add(gpu["uuid"])
                    self.save()
                    break
                except BaseException as exc:
                    if spawned is None:
                        lease.__exit__(None, None, None)
                        row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                    else:
                        self.children[key] = spawned
                        self.leases[key] = lease
                        row.update(status="uncertain", worker=B.process_token(spawned.pid), error=str(exc))
                    self.save()
                    B.append_json(self.journal, {"event": "launch_failure", "task_id": key, "error": str(exc)})
                    raise
        self.save()


# The only patched globals belong to this private module, never canonical B/V3.
B.validate, B.freeze_gate = validate, gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "run", "worker"))
    parser.add_argument("--schedule", type=Path)
    parser.add_argument("--freeze-commit")
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--gpu-fd", type=int)
    args = parser.parse_args()
    if args.action == "worker":
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
            stopping, dispatcher.stopping = True, True
        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, stop)
        B.append_json(dispatcher.journal, {"event": "scheduler_start", "pid": B.os.getpid(),
            "freeze_commit": args.freeze_commit, "adopted_from": schedule["adoption"]["runtime"]})
        while True:
            if stopping:
                dispatcher.stop_own()
                return 130
            dispatcher.tick()
            if all(row["status"] in B.TERMINAL for row in dispatcher.state["tasks"].values()):
                B.append_json(dispatcher.journal, {"event": "scheduler_terminal",
                    "tasks": {key: row["status"] for key, row in dispatcher.state["tasks"].items()}})
                return int(any(row["status"] == "failed" for row in dispatcher.state["tasks"].values()))
            time.sleep(5)


if __name__ == "__main__":
    raise SystemExit(main())
