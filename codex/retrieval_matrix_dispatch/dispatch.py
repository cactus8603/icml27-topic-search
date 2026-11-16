"""[codex] One central, capacity-based dispatcher; unchanged frozen launchers.

Only stdlib. Advisory GPU leases are inherited by workers and launchers, so an
unexpected scheduler exit cannot release a still-running job's GPU lock.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from codex.retrieval3090.resources import FileLock,query_gpus,atomic_json

LIVE={"starting","running","uncertain"}
TERMINAL={"completed","skipped_complete","failed","cancelled"}


def now():
    return datetime.now(ZoneInfo("Asia/Taipei")).isoformat()


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()


def append_json(path,row):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    payload=(json.dumps({"owner":"[codex]","time":now(),**row},sort_keys=True)+"\n").encode()
    fd=os.open(path,os.O_WRONLY|os.O_APPEND|os.O_CREAT|os.O_NOFOLLOW,0o600)
    try:
        if os.write(fd,payload)!=len(payload):
            raise OSError("Short journal append")
        os.fsync(fd)
    finally:
        os.close(fd)


def process_token(pid):
    try:
        raw=Path(f"/proc/{int(pid)}/stat").read_text()
        fields=raw[raw.rfind(")")+2:].split()
        if fields[0]=="Z":
            return None
        return {"pid":int(pid),"start_ticks":int(fields[19]),
                "boot_id":Path("/proc/sys/kernel/random/boot_id").read_text().strip()}
    except (FileNotFoundError,ProcessLookupError):
        return None


def alive(token):
    return bool(token) and process_token(token["pid"])==token


def parse_processes(output):
    values={}
    for line in output.splitlines():
        if not line.strip():
            continue
        parts=[part.strip() for part in line.split(",")]
        if len(parts)!=2 or not re.fullmatch(r"GPU-[A-Za-z0-9-]+",parts[0]) or not parts[1].isdigit() or int(parts[1])<=0:
            raise ValueError("Unknown nvidia compute PID report; no admission")
        values.setdefault(parts[0],set()).add(int(parts[1]))
    return {key:sorted(value) for key,value in values.items()}


def query_snapshot():
    rows=query_gpus()
    result=subprocess.run(["nvidia-smi","--query-compute-apps=gpu_uuid,pid","--format=csv,noheader,nounits"],
                          check=True,capture_output=True,text=True,timeout=15)
    processes=parse_processes(result.stdout)
    if set(processes)-{row["uuid"] for row in rows}:
        raise ValueError("Compute PID report refers to an unknown GPU")
    return [{**row,"compute_pids":processes.get(row["uuid"],[])} for row in rows]


def eligible(row,task):
    # Utilization is deliberately irrelevant: the user authorizes fitting a job
    # beside one existing process, never making a card three/four processes.
    return row["free_mib"]>=task["min_free_mib"] and len(set(row["compute_pids"]))<=1


def completed_output(task):
    files=task.get("completion",[])
    if not files:
        raise ValueError("Every task needs non-metric completion receipts")
    for name in files:
        path=Path(name)
        if not path.exists():
            return False
        receipt=json.loads(path.read_text())
        identity=receipt.get("identity",{})
        if (receipt.get("state")!="COMPLETE" or identity.get("model")!=task["model"]
                or identity.get("freeze_commit")!=task["freeze_commit"]
                or identity.get("plan_sha256")!=task["plan_sha256"]):
            raise ValueError("Existing completion receipt has a different frozen identity")
    return True


def validate(schedule):
    if schedule.get("schema")!=1 or schedule.get("owner")!="[codex]" or schedule.get("max_live")!=2:
        raise ValueError("Central schedule requires owner [codex], schema1, global cap2")
    if schedule.get("confirmation_seconds")!=5 or schedule.get("max_existing_compute_pids")!=1:
        raise ValueError("Admission must use two samples five seconds apart and at most one existing compute PID")
    ids=set()
    for task in schedule["tasks"]:
        if not re.fullmatch(r"[a-z0-9_-]+",task["id"]) or task["id"] in ids:
            raise ValueError("Task IDs must be unique safe names")
        ids.add(task["id"])
        if task["min_free_mib"] not in {8192,20480}:
            raise ValueError("Only the user's fixed 2B/8B capacity classes are supported")
        if not re.fullmatch(r"[0-9a-f]{40}",task["freeze_commit"]):
            raise ValueError("Task must preserve its full original frozen commit")
        if not task["command"] or any(not isinstance(arg,str) or "\0" in arg for arg in task["command"]):
            raise ValueError("Task command must be a literal nonempty argument vector")
    if len({task["output"] for task in schedule["tasks"]})!=len(ids):
        raise ValueError("Tasks cannot share an output directory")


def freeze_gate(schedule_path,schedule,commit,*,check_legacy=True):
    validate(schedule)
    if schedule.get("state")!="FROZEN" or not re.fullmatch(r"[0-9a-f]{40}",commit or ""):
        raise ValueError("Dispatcher source/protocol/schedule must be explicitly frozen before execution")
    if Path(schedule["gpu_lock_dir"]).resolve()!=ROOT/"dl/retrieval3090/runs/.gpu-locks" or Path(schedule["shared_log"]).resolve()!=ROOT/"killtests/SUBMITTED.log":
        raise ValueError("Production must use the existing shared GPU locks and owner-marked submission log")
    required={str(schedule_path.relative_to(ROOT)),*(str(path.relative_to(ROOT)) for path in HERE.glob("*.py")),
              "codex/retrieval_matrix_dispatch/PREREG.md","codex/retrieval3090/resources.py"}
    for relative in required:
        original=subprocess.check_output(["git","-C",str(ROOT),"show",f"{commit}:{relative}"])
        if original!=(ROOT/relative).read_bytes():
            raise ValueError("Dispatcher code/schedule differs from the source freeze: "+relative)
    status=subprocess.check_output(["git","-C",str(ROOT),"status","--porcelain","--",str(HERE.relative_to(ROOT))],text=True)
    if status.strip():
        raise ValueError("Dispatcher worktree is not clean")
    for task in schedule["tasks"]:
        if sha(task["plan"])!=task["plan_sha256"] or sha(task["launcher"])!=task["launcher_sha256"]:
            raise ValueError("Original task plan/launcher bytes changed")
        # Preserve the original argv verbatim, including its original reference.
        expected=["bash",task["launcher"],task["freeze_commit"],task["plan"],task["model"],task["output"]]
        if task["command"]!=expected:
            raise ValueError("The central scheduler may not change the frozen launcher command")
    blockers=[]
    for path in schedule.get("legacy_queues",[]):
        old=json.loads(Path(path).read_text())
        if process_token(old["queue_pid"]) is not None or (old.get("child_pid") and process_token(old["child_pid"])):
            blockers.append({"queue":path,"queue_pid":old["queue_pid"],"child_pid":old.get("child_pid")})
    if check_legacy and blockers:
        raise ValueError("Root must retire the old waiting queues before central startup")
    return {"source_gate":"PASS","legacy_live":blockers,"production_ready":not blockers}


def shared_start(task,child_pid,gpu,shared_log):
    # Root pre-registers central tasks under a short LOCKS claim. Starting an
    # already-submitted own task adds one atomic owner-marked line, no long lock.
    stamp=datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y-%m-%d %H:%M")
    line=f"[codex] {stamp} start local3090-{child_pid} compositional-retrieval central-task={task['id']} GPU={gpu['uuid']} original-frozen-command {task['freeze_commit']}\n"
    fd=os.open(shared_log,os.O_WRONLY|os.O_APPEND|os.O_NOFOLLOW)
    try:
        payload=line.encode()
        if os.write(fd,payload)!=len(payload):
            raise OSError("Short shared-log append")
        os.fsync(fd)
    finally:
        os.close(fd)


def worker_gate(config,gpu_fd):
    proof=config["execution_proof"]
    schedule_path=Path(proof["schedule"]).resolve()
    schedule=json.loads(schedule_path.read_text())
    freeze_gate(schedule_path,schedule,proof["freeze_commit"])
    if sha(schedule_path)!=config["schedule_sha256"] or config["task"] not in schedule["tasks"]:
        raise ValueError("Worker task differs from the frozen central schedule")
    expected_lock=Path(schedule["gpu_lock_dir"])/(config["gpu"]["uuid"]+".lock")
    fd_info=os.fstat(gpu_fd)
    path_info=expected_lock.stat()
    if (fd_info.st_dev,fd_info.st_ino)!=(path_info.st_dev,path_info.st_ino):
        raise ValueError("Worker did not inherit the declared shared GPU lease")


def worker(config_path,gpu_fd):
    config=json.loads(Path(config_path).read_text())
    task,gpu=config["task"],config["gpu"]
    receipt_path=Path(config["receipt"])
    identity={"task_id":task["id"],"schedule_sha256":config["schedule_sha256"],"token":config["token"]}
    result={**identity,"status":"starting","worker":process_token(os.getpid()),"child":None,"gpu":gpu,"started_at":now()}
    atomic_json(receipt_path,result)
    child=None
    cancelled=False
    def terminate(signum,_frame):
        nonlocal cancelled
        cancelled=True
        # Only the exact process group we spawned; never use GPU process lists
        # as kill targets. A dead/reused PID does not authorize a signal.
        if child is not None and child.poll() is None and alive(result.get("child")):
            os.killpg(child.pid,signal.SIGTERM)
    previous={signum:signal.signal(signum,terminate) for signum in (signal.SIGTERM,signal.SIGINT)}
    try:
        worker_gate(config,gpu_fd)
        if cancelled:
            raise InterruptedError("Worker stopped before model launch")
        env=os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=gpu["uuid"],TZ="Asia/Taipei",RETRIEVAL_CENTRAL_OWNER="[codex]",
                   RETRIEVAL_CENTRAL_TASK=task["id"])
        child=subprocess.Popen(task["command"],cwd=task["cwd"],env=env,shell=False,start_new_session=True,pass_fds=(gpu_fd,))
        result["child"]=process_token(child.pid)
        if cancelled:
            terminate(signal.SIGTERM,None)
        shared_start(task,child.pid,gpu,config["shared_log"])
        result.update(status="running")
        atomic_json(receipt_path,result)
        append_json(config["journal"],{"event":"launch","task_id":task["id"],"child":result["child"],"gpu":gpu,"command":task["command"]})
        while child.poll() is None:
            if cancelled:
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    if alive(result["child"]):
                        os.killpg(child.pid,signal.SIGKILL)
            else:
                time.sleep(1)
        code=child.wait()
        result.update(status="cancelled" if cancelled else "completed" if code==0 else "failed",returncode=code,finished_at=now())
    except BaseException as exc:
        if child is not None and child.poll() is None and alive(result.get("child")):
            os.killpg(child.pid,signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if alive(result["child"]):
                    os.killpg(child.pid,signal.SIGKILL)
                child.wait()
        result.update(status="cancelled" if cancelled else "failed",error=f"{type(exc).__name__}: {exc}",finished_at=now())
    finally:
        atomic_json(receipt_path,result)
        append_json(config["journal"],{"event":"terminal",**result})
        os.close(gpu_fd)
        for signum,handler in previous.items():
            signal.signal(signum,handler)
    return 0 if result["status"]=="completed" else 1


class Dispatcher:
    def __init__(self,schedule,schedule_sha,runtime,*,execution_proof=None,query=query_snapshot,clock=time.monotonic,popen=subprocess.Popen,complete=completed_output):
        validate(schedule)
        self.schedule,self.schedule_sha,self.runtime=schedule,schedule_sha,Path(runtime)
        self.runtime.mkdir(parents=True,exist_ok=True)
        self.query,self.clock,self.popen,self.complete=query,clock,popen,complete
        self.execution_proof=execution_proof
        self.stopping=False
        self.journal=self.runtime/"events.jsonl"
        self.path=self.runtime/"state.json"
        self.leases,self.children,self.prior={},{},{}
        if self.path.exists():
            self.state=json.loads(self.path.read_text())
            if self.state.get("schedule_sha256")!=schedule_sha:
                raise ValueError("Cannot resume a different schedule in the same runtime directory")
        else:
            self.state={"schema":1,"owner":"[codex]","schedule_sha256":schedule_sha,"created_at":now(),
                "tasks":{task["id"]:{"status":"pending"} for task in schedule["tasks"]}}
        self.save()

    def save(self):
        self.state.update(updated_at=now(),scheduler=process_token(os.getpid()))
        atomic_json(self.path,self.state)

    def live_count(self):
        return sum(row["status"] in LIVE for row in self.state["tasks"].values())

    def reconcile(self):
        for task in self.schedule["tasks"]:
            key=task["id"]
            row=self.state["tasks"][key]
            if row["status"] not in LIVE:
                continue
            receipt=Path(row["receipt"])
            if receipt.exists():
                result=json.loads(receipt.read_text())
                if (result.get("token")!=row["token"] or result.get("schedule_sha256")!=self.schedule_sha or result.get("task_id")!=key):
                    raise ValueError("Worker receipt identity differs; refusing to assume a job ended")
                row.update(result)
                if row["status"] in TERMINAL:
                    if key in self.children:
                        self.children.pop(key).wait()
                    if key in self.leases:
                        self.leases.pop(key).__exit__(None,None,None)
                elif not alive(row.get("worker")) and not alive(row.get("child")):
                    row.update(status="failed",error="Worker and original child disappeared without a terminal receipt; never retry automatically")
                    append_json(self.journal,{"event":"lost_worker","task_id":key})
                    if key in self.leases:
                        self.leases.pop(key).__exit__(None,None,None)
            else:
                # Crash between persisting STARTING and launching the worker is
                # ambiguous. Reserve its slot; never replay an uncertain launch.
                row["status"]="uncertain"
        self.save()

    def tick(self):
        self.reconcile()
        if self.stopping or self.live_count()>=2:
            return
        try:
            snapshot=self.query()
        except (OSError,ValueError,subprocess.SubprocessError) as exc:
            self.prior={}
            append_json(self.journal,{"event":"query_failure","error":str(exc)})
            return
        now_value=self.clock()
        occupied={row["gpu"]["uuid"] for row in self.state["tasks"].values() if row["status"] in LIVE and "gpu" in row}
        for task in self.schedule["tasks"]:
            if self.stopping or self.live_count()>=2:
                break
            key=task["id"]
            row=self.state["tasks"][key]
            if row["status"]!="pending":
                continue
            if self.complete(task):
                row.update(status="skipped_complete",finished_at=now())
                append_json(self.journal,{"event":"skip_complete","task_id":key})
                continue
            available=[gpu for gpu in snapshot if gpu["uuid"] not in occupied and eligible(gpu,task)]
            current={gpu["uuid"] for gpu in available}
            for prior_key in list(self.prior):
                if prior_key[0]==key and prior_key[1] not in current:
                    del self.prior[prior_key]
            # Best fit preserves the scarce >=20GiB cards for 8B launchers.
            # Task priority itself remains the original frozen task order.
            for gpu in sorted(available,key=lambda x:(x["free_mib"],x["index"])):
                pair=(key,gpu["uuid"])
                if pair not in self.prior:
                    self.prior[pair]=now_value
                    continue
                if now_value-self.prior[pair]<5:
                    continue
                lease=FileLock(Path(self.schedule["gpu_lock_dir"])/(gpu["uuid"]+".lock"))
                try:
                    spawned=None
                    lease.__enter__()
                except BlockingIOError:
                    continue
                try:
                    final=next((item for item in self.query() if item["uuid"]==gpu["uuid"]),None)
                    if self.stopping:
                        lease.__exit__(None,None,None)
                        return
                    if final is None or not eligible(final,task):
                        self.prior.pop(pair,None)
                        lease.__exit__(None,None,None)
                        continue
                    if self.complete(task):
                        row.update(status="skipped_complete",finished_at=now())
                        lease.__exit__(None,None,None)
                        break
                    token=os.urandom(16).hex()
                    attempt=self.runtime/(key+"-attempt.json")
                    config=self.runtime/(key+"-worker.json")
                    atomic_json(config,{"task":task,"gpu":final,"token":token,"schedule_sha256":self.schedule_sha,
                        "receipt":str(attempt),"journal":str(self.journal),"shared_log":self.schedule["shared_log"],
                        "execution_proof":self.execution_proof})
                    row.update(status="starting",token=token,receipt=str(attempt),gpu=final,command=task["command"])
                    self.save()
                    append_json(self.journal,{"event":"admitted","task_id":key,"gpu":final,"threshold_mib":task["min_free_mib"]})
                    log=(self.runtime/(key+".log")).open("ab",buffering=0)
                    try:
                        child=self.popen([sys.executable,"-B",str(HERE/"dispatch.py"),"worker","--config",str(config),"--gpu-fd",str(lease.fd)],
                            cwd=str(ROOT),shell=False,start_new_session=True,pass_fds=(lease.fd,),stdout=log,stderr=subprocess.STDOUT)
                        spawned=child
                    finally:
                        log.close()
                    row["worker"]=process_token(child.pid)
                    self.children[key]=child
                    self.leases[key]=lease
                    occupied.add(gpu["uuid"])
                    self.save()
                    break
                except BaseException as exc:
                    if spawned is None:
                        lease.__exit__(None,None,None)
                        row.update(status="failed",error=f"{type(exc).__name__}: {exc}")
                    else:
                        # Popen already succeeded: never unlock an inherited
                        # running lease merely because a later state write failed.
                        self.children[key]=spawned
                        self.leases[key]=lease
                        row.update(status="uncertain",worker=process_token(spawned.pid),error=str(exc))
                    self.save()
                    append_json(self.journal,{"event":"launch_failure","task_id":key,"error":str(exc)})
                    raise
        self.save()

    def stop_own(self):
        self.reconcile()
        for key,row in self.state["tasks"].items():
            if row["status"] in LIVE and alive(row.get("worker")):
                os.kill(row["worker"]["pid"],signal.SIGTERM)
                append_json(self.journal,{"event":"stop_own_worker","task_id":key,"worker":row["worker"]})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="action",required=True)
    run=sub.add_parser("run")
    run.add_argument("--schedule",type=Path,required=True)
    run.add_argument("--freeze-commit",required=True)
    run.add_argument("--runtime",type=Path,required=True)
    pre=sub.add_parser("preflight",help="Read-only source/command gate; report live old queues without cancelling")
    pre.add_argument("--schedule",type=Path,required=True)
    pre.add_argument("--freeze-commit",required=True)
    work=sub.add_parser("worker")
    work.add_argument("--config",type=Path,required=True)
    work.add_argument("--gpu-fd",type=int,required=True)
    args=parser.parse_args()
    if args.action=="worker":
        return worker(args.config,args.gpu_fd)
    schedule_path=args.schedule.resolve()
    schedule=json.loads(schedule_path.read_text())
    if args.action=="preflight":
        print(json.dumps(freeze_gate(schedule_path,schedule,args.freeze_commit,check_legacy=False),indent=2))
        return 0
    freeze_gate(schedule_path,schedule,args.freeze_commit)
    with FileLock(ROOT/"dl/retrieval3090/runs/.gpu-locks/central-dispatch.lock"):
        dispatcher=Dispatcher(schedule,sha(schedule_path),args.runtime.resolve(),
                              execution_proof={"schedule":str(schedule_path),"freeze_commit":args.freeze_commit})
        stopping=False
        def stop(_signum,_frame):
            nonlocal stopping
            stopping=True
            dispatcher.stopping=True
        for signum in (signal.SIGTERM,signal.SIGINT):
            signal.signal(signum,stop)
        append_json(dispatcher.journal,{"event":"scheduler_start","freeze_commit":args.freeze_commit,"pid":os.getpid()})
        while True:
            if stopping:
                dispatcher.stop_own()
                while dispatcher.live_count():
                    dispatcher.reconcile()
                    # A never-spawned uncertain launch has no process to stop;
                    # leave its reserved state for root, never invent a retry.
                    if all(not alive(row.get("worker")) and not alive(row.get("child")) for row in dispatcher.state["tasks"].values() if row["status"] in LIVE):
                        break
                    time.sleep(1)
                return 130
            dispatcher.tick()
            if stopping:
                continue
            if all(row["status"] in TERMINAL for row in dispatcher.state["tasks"].values()):
                append_json(dispatcher.journal,{"event":"scheduler_terminal","tasks":{k:v["status"] for k,v in dispatcher.state["tasks"].items()}})
                return 1 if any(row["status"]=="failed" for row in dispatcher.state["tasks"].values()) else 0
            time.sleep(5)


if __name__=="__main__":
    raise SystemExit(main())
