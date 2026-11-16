"""[codex] Explicit exact-parent-only handoff; never signal a worker or model.

Import and inspect are read-only. execute/recover require an explicit CLI flag;
this module is not invoked by the scheduler, workers, or builder.
"""
from __future__ import annotations
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from codex.retrieval_matrix_dispatch_next import controller as C
from codex.retrieval_matrix_dispatch_next.build_schedule import frozen
B = C.B
ALLOWED_CONTROLLERS = {ROOT/"codex/retrieval_matrix_dispatch/dispatch.py",
    ROOT/"codex/retrieval_matrix_dispatch_v3/controller.py", C.HERE/"controller.py"}


def parent_command(side):
    return [side["python"], "-B", side["controller"], "run", "--schedule", side["schedule"]["path"],
        "--freeze-commit", side["freeze_commit"], "--runtime", side["runtime"]]


class Host:
    """Narrow OS interface. CPU tests replace this entire interface."""
    alive = staticmethod(B.alive)
    freeze = staticmethod(frozen)

    def command(self, token):
        if not self.alive(token):
            raise ValueError("Exact process token disappeared or was reused")
        return Path(f'/proc/{token["pid"]}/cmdline').read_bytes().rstrip(b"\0").decode().split("\0")

    def signal_parent(self, token, expected_command, signum):
        if signum not in {signal.SIGSTOP, signal.SIGCONT, signal.SIGKILL}:
            raise ValueError("Only the exact explicitly configured controller parent can receive this signal")
        # Pin the process before the final start/boot/cmdline recheck. A reused
        # numeric PID can never redirect a signal after this check.
        descriptor = os.pidfd_open(token["pid"])
        try:
            if self.command(token) != expected_command:
                raise ValueError("Exact controller parent command changed")
            signal.pidfd_send_signal(descriptor, signum)
        finally:
            os.close(descriptor)

    def wait_stopped(self, token):
        for _ in range(100):
            if not self.alive(token):
                raise ValueError("Parent disappeared before its stopped snapshot")
            raw = Path(f'/proc/{token["pid"]}/stat').read_text()
            if raw[raw.rfind(")")+2:].split()[0] == "T":
                return
            time.sleep(.02)
        raise ValueError("Exact parent did not stop within two seconds")

    def wait_retired(self, token):
        for _ in range(100):
            if not self.alive(token):
                return
            time.sleep(.02)
        raise ValueError("Old parent still alive after retirement")

    def lease(self, worker, descriptor, path):
        if not self.alive(worker):
            raise ValueError("Original worker is no longer alive")
        wanted = Path(path).stat()
        actual = Path(f'/proc/{worker["pid"]}/fd/{descriptor}').stat()
        if (wanted.st_dev, wanted.st_ino) != (actual.st_dev, actual.st_ino):
            raise ValueError("Worker FD is not the declared shared GPU lease inode")
        fd = os.open(path, os.O_RDWR|os.O_NOFOLLOW)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                return {"device": wanted.st_dev, "inode": wanted.st_ino, "held": True}
            raise ValueError("Original GPU lease is no longer held")
        finally:
            os.close(fd)

    def singleton_free(self):
        with B.FileLock(ROOT/"dl/retrieval3090/runs/.gpu-locks/central-dispatch.lock"):
            pass

    def spawn(self, command, runtime):
        with (runtime/"controller.log").open("ab", buffering=0) as log:
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        # Persist attempted PID even if it exits before a token can be read.
        return {"pid": child.pid, "token": B.process_token(child.pid), "command": command}

    def log(self, action, job, reason, commit):
        stamp = B.datetime.now(B.ZoneInfo("Asia/Taipei")).strftime("%Y-%m-%d %H:%M")
        payload = f"[codex] {stamp} {action} {job} compositional-retrieval {reason} {commit}\n".encode()
        fd = os.open(ROOT/"killtests/SUBMITTED.log", os.O_WRONLY|os.O_APPEND|os.O_NOFOLLOW)
        try:
            if os.write(fd, payload) != len(payload):
                raise OSError("Short owner-marked submission append")
            os.fsync(fd)
        finally:
            os.close(fd)


def validate_config(config, host, *, gate=True):
    if config.get("schema") != 1 or config.get("owner") != "[codex]":
        raise ValueError("An explicit owner-marked handoff configuration is required")
    old, new = config["old"], config["new"]
    for side in (old, new):
        if Path(side["controller"]).resolve() not in ALLOWED_CONTROLLERS:
            raise ValueError("Only the reviewed own controller entry points can be handed off")
        for key in ("runtime", "controller"):
            if str(Path(side[key]).resolve()) != side[key]:
                raise ValueError("Handoff paths must be explicit canonical absolute paths")
        if str(C.checked(side["schedule"])) != side["schedule"]["path"]:
            raise ValueError("Schedule paths must also be canonical absolute paths")
        host.freeze(side["schedule"]["path"], side["freeze_commit"])
        host.freeze(side["controller"], side["freeze_commit"])
    if (Path(old["runtime"]) == Path(new["runtime"]) or Path(new["runtime"]) in Path(old["runtime"]).parents
            or Path(old["runtime"]) in Path(new["runtime"]).parents
            or Path(new["controller"]) != C.HERE/"controller.py"):
        raise ValueError("New controller/runtime must be distinct and cannot overwrite an old runtime")
    token = old["expected_token"]
    if set(token) != {"pid", "start_ticks", "boot_id"} or type(token["pid"]) is not int or token["pid"] <= 1:
        raise ValueError("An exact PID/start/boot token is mandatory")
    schedule_path = C.checked(new["schedule"])
    schedule = json.loads(schedule_path.read_text())
    expected_origin = {"schedule": old["schedule"], "freeze_commit": old["freeze_commit"],
        "runtime": old["runtime"], "retire_pending": []}
    if schedule["adoption"] != expected_origin:
        raise ValueError("New schedule adopts another original schedule/ref/runtime")
    if gate:
        C.gate(schedule_path, schedule, new["freeze_commit"])
    return schedule


def receipt_identity(row, key, old_schedule_sha):
    return {"task_id": key, "token": row["token"],
        "schedule_sha256": row.get("adopted_schedule_sha256", old_schedule_sha)}


def capture(config, previous, runtime, host, *, prior_proofs=None):
    """Validate original receipts, including workers from earlier handoffs.

    Terminal receipts may race with the parent snapshot. Accept them only with
    exact original identity and a dead original model child; otherwise stop.
    """
    old = config["old"]
    old_tasks = {row["id"]: row for row in json.loads(C.checked(old["schedule"]).read_text())["tasks"]}
    proofs = copy.deepcopy(prior_proofs or {})
    live = [row for row in previous["tasks"].values() if row["status"] in B.LIVE]
    if len(live) > 2 or len({row.get("gpu", {}).get("uuid") for row in live}) != len(live):
        raise ValueError("Original live state violates global cap2 or perGPU1; defer handoff without touching workers")
    for key, row in previous["tasks"].items():
        if row["status"] not in B.LIVE:
            continue
        if row["status"] != "running":
            raise ValueError("Ambiguous starting/uncertain state requires operator review, never replay")
        origin = Path(row.get("adopted_runtime", old["runtime"]))
        path = origin/(key+"-attempt.json")
        if row.get("receipt") != str(path) or row.get("command") != old_tasks[key]["command"]:
            raise ValueError("Original command or receipt origin changed")
        raw = path.read_bytes()
        receipt = json.loads(raw)
        if any(receipt.get(k) != value for k, value in receipt_identity(row, key, old["schedule"]["sha256"]).items()):
            raise ValueError("Original worker receipt token/task/schedule differs")
        snapshot = runtime/(key+"-attempt-"+B.hashlib.sha256(raw).hexdigest()[:16]+".json")
        if snapshot.exists() and snapshot.read_bytes() != raw:
            raise ValueError("Cannot replace a previous handoff evidence snapshot")
        snapshot.write_bytes(raw)
        if receipt.get("status") in B.TERMINAL:
            if host.alive(receipt.get("child")):
                raise ValueError("Terminal receipt raced but its original model child is still alive")
            row.update(receipt)
            proofs[key] = {"status": row["status"], "terminal_receipt": C.entry(snapshot)}
            continue
        if receipt.get("status") != "running":
            raise ValueError("Original worker receipt is not settled running or terminal")
        for field in ("worker", "child", "gpu"):
            if row.get(field) != receipt.get(field):
                raise ValueError("Original live worker/child/GPU identity changed")
        config_path = origin/(key+"-worker.json")
        worker_config = json.loads(config_path.read_text())
        if any(worker_config.get(k) != value for k, value in {
                "task": old_tasks[key], "token": row["token"], "gpu": row["gpu"],
                "receipt": str(path), "schedule_sha256": receipt["schedule_sha256"]}.items()):
            raise ValueError("Original worker configuration differs")
        proof = worker_config["execution_proof"]
        origin_path = Path(proof["schedule"])
        if B.sha(origin_path) != receipt["schedule_sha256"]:
            raise ValueError("Original worker schedule bytes changed")
        host.freeze(origin_path, proof["freeze_commit"])
        worker_command = host.command(row["worker"])
        if (len(worker_command) != 8 or worker_command[1] != "-B"
                or Path(worker_command[2]).resolve() not in ALLOWED_CONTROLLERS
                or worker_command[3:7] != ["worker", "--config", str(config_path), "--gpu-fd"]
                or not worker_command[7].isdigit()):
            raise ValueError("Original worker command differs from its frozen origin")
        host.freeze(worker_command[2], proof["freeze_commit"])
        if host.command(row["child"]) != old_tasks[key]["command"]:
            raise ValueError("Original model child command differs")
        lock = ROOT/"dl/retrieval3090/runs/.gpu-locks"/(row["gpu"]["uuid"]+".lock")
        lease = host.lease(row["worker"], worker_command[7], lock)
        proofs[key] = {"status": "running", "worker": row["worker"], "child": row["child"],
            "receipt": str(path), "receipt_sha256_at_snapshot": B.sha(snapshot),
            "worker_config": C.entry(config_path), "worker_command": worker_command,
            "child_command": old_tasks[key]["command"], "gpu_lock": str(lock), "lease": lease}
        row.update(receipt)
    return proofs


def persist(runtime, previous, migration):
    B.atomic_json(runtime/"previous-state.json", previous)
    migration["previous_state_sha256"] = B.sha(runtime/"previous-state.json")
    B.atomic_json(runtime/"migration.json", migration)


def launch_new(config, previous, migration, runtime, host):
    migration["workers"] = capture(config, previous, runtime, host, prior_proofs=migration.get("workers"))
    C.seed(json.loads(C.checked(config["new"]["schedule"]).read_text()), config["new"]["schedule"]["sha256"], previous)
    host.singleton_free()
    migration.update(phase="ready_to_start", old_parent_retired=True)
    persist(runtime, previous, migration)
    # No task is launched here. The new singleton controller reconciles exact
    # old receipts and owns the only admission loop.
    migration["replacement_spawn_attempted"] = True
    persist(runtime, previous, migration)
    migration["new_parent"] = host.spawn(parent_command(config["new"]), runtime)
    migration["phase"] = "controller_start_attempted"
    persist(runtime, previous, migration)
    host.log("submit", "central-"+str(migration["new_parent"]["pid"]),
        "唯一新中央接手原worker/globalcap2/perGPU1；原producer不重跑", config["new"]["freeze_commit"])
    return migration


def execute(config, *, host=None, gate=True):
    host = host or Host()
    schedule = validate_config(config, host, gate=gate)
    old, new = config["old"], config["new"]
    token, command = old["expected_token"], parent_command(old)
    if host.command(token) != command:
        raise ValueError("Old central exact command differs")
    runtime = Path(new["runtime"])
    runtime.mkdir(parents=True, exist_ok=False)
    migration = {"schema": 1, "owner": "[codex]", "configuration": copy.deepcopy(config),
        "new_schedule_sha256": new["schedule"]["sha256"], "old_parent": token,
        "old_parent_retired": False, "phase": "created", "handoff_source": C.entry(Path(__file__))}
    stopped = retired = False
    previous = None
    try:
        host.signal_parent(token, command, signal.SIGSTOP)
        stopped = True
        host.wait_stopped(token)
        host.log("hold-parent", "central-"+str(token["pid"]), "只暫停exact舊中央父程序以核接手證據", new["freeze_commit"])
        raw = (Path(old["runtime"])/"state.json").read_bytes()
        (runtime/"previous-state-raw.json").write_bytes(raw)
        previous = json.loads(raw)
        if previous.get("scheduler") != token:
            raise ValueError("Snapshot scheduler token differs")
        migration["workers"] = capture(config, previous, runtime, host, prior_proofs=migration.get("workers"))
        C.seed(schedule, new["schedule"]["sha256"], previous)
        persist(runtime, previous, migration)
        # Recheck before the irreversible parent retirement; terminal receipts
        # are retained if a model finished while we examined other workers.
        migration["workers"] = capture(config, previous, runtime, host, prior_proofs=migration.get("workers"))
        persist(runtime, previous, migration)
        host.signal_parent(token, command, signal.SIGKILL)
        retired = True
        migration.update(old_parent_retired=True, phase="parent_retired")
        persist(runtime, previous, migration)
        host.log("retire-parent", "central-"+str(token["pid"]), "只退休exact已停中央父程序；不signal worker/model", new["freeze_commit"])
        host.wait_retired(token)
        return launch_new(config, previous, migration, runtime, host)
    except BaseException as error:
        migration.update(phase="operator_recovery_required" if retired else "aborted_before_retirement",
            error_type=type(error).__name__, error=str(error))
        if previous is not None:
            persist(runtime, previous, migration)
        else:
            B.atomic_json(runtime/"migration.json", migration)
        raise
    finally:
        if stopped and not retired and host.alive(token):
            host.signal_parent(token, command, signal.SIGCONT)
            host.log("release-parent", "central-"+str(token["pid"]), "移交未退休，恢復同一舊中央；未signal模型", new["freeze_commit"])


def recover(config, *, host=None, gate=True):
    """Explicit operator recovery after retirement, only before any new spawn."""
    host = host or Host()
    validate_config(config, host, gate=gate)
    runtime = Path(config["new"]["runtime"])
    migration = json.loads((runtime/"migration.json").read_text())
    if (migration.get("configuration") != config or migration.get("old_parent_retired") is not True
            or migration.get("replacement_spawn_attempted") or "new_parent" in migration or (runtime/"state.json").exists()
            or host.alive(config["old"]["expected_token"])):
        raise ValueError("Recovery needs a retired exact parent, no attempted replacement and no new runtime state")
    previous_path = runtime/"previous-state.json"
    if B.sha(previous_path) != migration["previous_state_sha256"]:
        raise ValueError("Recovery cannot replace the preserved previous state")
    return launch_new(config, json.loads(previous_path.read_text()), migration, runtime, host)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("inspect", "execute", "recover"))
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--root-authorized-parent-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.action == "inspect":
        validate_config(config, Host())
        print(json.dumps({"state": "READ_ONLY_PREFLIGHT_PASS", "configuration": C.entry(args.config)}))
        return
    if not args.root_authorized_parent_only:
        raise ValueError("Root must explicitly authorize this exact parent-only operation; default is read-only")
    result = (execute if args.action == "execute" else recover)(config)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
