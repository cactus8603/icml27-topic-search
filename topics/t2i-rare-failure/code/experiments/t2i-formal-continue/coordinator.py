"""Continue the formal T2I CPU selection into only its registered Stage 2a array."""

import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from types import ModuleType
from zoneinfo import ZoneInfo


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
FORMAL = HERE.parent / "t2i-formal-replay"
TOPIC = Path("killtests/t2i-rare-failure")
SLURM = Path("/cm/local/apps/slurm/current/bin")
FORMAL_FREEZE = "4351ebb70f2e7ad72e0ec8b99658d9ed41c06567"
GATE_SHA256 = "d88b2d37b09d64a57a3102ecd7049c57d616aa19ed3fcc85a69a5dc447f9cffe"
STAGE1 = "26765"
PLANNER = "26802"
PHASE = "2a"
MIN_GROUP_FREE_KIB = 20 * 1024 ** 2
TAIPEI = ZoneInfo("Asia/Taipei")
SCHEDULER_ENTRY = '''
import json, sys
from types import ModuleType

sources = json.load(sys.stdin)
pipeline = ModuleType("pipeline")
pipeline.__file__ = sources["pipeline_path"]
sys.modules["pipeline"] = pipeline
exec(compile(sources["pipeline"], pipeline.__file__, "exec"), pipeline.__dict__)
__file__ = sources["schedule_path"]
sys.argv[0] = __file__
exec(compile(sources["schedule"], __file__, "exec"))
'''


def command(*args):
    return subprocess.check_output(args, cwd=REPO, text=True).strip()


def git_source(revision, path):
    return subprocess.check_output(["git", "show", f"{revision}:{path.relative_to(REPO)}"], cwd=REPO)


def ledger_entries(text, scope):
    pattern = re.compile(rf"^\[codex\] .+ submit ([0-9]+) t2i-rare-failure {re.escape(scope)} .+ ([0-9a-f]{{7,40}})$")
    return [match[1] for line in text.splitlines() if (match := pattern.fullmatch(line)) and FORMAL_FREEZE.startswith(match[2])]


def quota_free_kib(text):
    rows = [fields for row in text.splitlines() if (fields := row.split()) and fields[0].endswith(":/work")]
    if len(rows) != 1:
        raise ValueError("Cannot verify the original /work group limit")
    used, soft, hard = (int(field.rstrip("*")) for field in rows[0][1:4])
    limits = [value for value in (soft, hard) if value > 0]
    if not limits:
        raise ValueError("The /work group quota is unavailable")
    return min(limits) - used


def completed_indices(attempts, total):
    known = {row[1] for row in attempts}
    done = {index for _, index, _, state, code, _ in attempts if (state, code) == ("COMPLETED", "0:0")}
    if not known <= set(range(total)):
        raise ValueError("An allocated task is outside the fixed plan")
    if known - done:
        raise ValueError("A registered task failed without a successful retry; inspect it before resuming")
    return done


def checked_group(budget, tasks, done, cost, digest, reserve_minutes):
    if (budget["formal_phase"] != PHASE or budget["plan_sha256"] != digest
            or budget["reserved_gpu_seconds"] != 60 * reserve_minutes):
        raise ValueError("The scheduler used the wrong phase, manifest or reserve")
    groups = budget["array_groups"]
    if groups is None:
        raise ValueError("The unchanged sampling plan cannot fit the available GPU budget")
    spent, reserved = budget["total_spent_gpu_seconds"], budget["reserved_gpu_seconds"]
    phase_spent = budget["current_phase_gpu_seconds"]
    total_stage_images = sum(stop - start for _, _, start, stop in tasks)
    remaining = min(cost["cap_gpu_hours"] * 3600 - spent - reserved - 1,
                    int(2 * total_stage_images / cost["est_img_per_s"]) - phase_spent)
    if budget["available_gpu_seconds_for_phase"] != remaining:
        raise ValueError("The schedule does not account for the original stage, total and reserve limits")
    allocated, used_indices = 0, set()
    for group in groups:
        indices, minutes = group["indices"], group["minutes_per_task"]
        if (type(minutes) is not int or minutes <= 0 or not indices or len(indices) != len(set(indices))
                or any(type(i) is not int or not 0 <= i < len(tasks) for i in indices)
                or used_indices.intersection(indices)):
            raise ValueError("The scheduler returned invalid or duplicate task indices")
        if any(tasks[i][3] - tasks[i][2] != group["images_per_task"] for i in indices):
            raise ValueError("The scheduler changed the selected work")
        if minutes * 60 > 2 * group["images_per_task"] / cost["est_img_per_s"]:
            raise ValueError("An array time limit exceeds the registered task estimate guard")
        used_indices.update(indices)
        allocated += 60 * minutes * len(indices)
    if used_indices != set(range(len(tasks))) - done or allocated > remaining:
        raise ValueError("The scheduler omitted registered work or exceeds the remaining GPU budget")
    return groups[0] if groups else None


def sealed_scheduler(schedule_revision, reserve_minutes):
    path = FORMAL / "schedule.py"
    pipeline_path = FORMAL / "pipeline.py"
    sources = {"pipeline": git_source(FORMAL_FREEZE, pipeline_path).decode(),
               "pipeline_path": str(pipeline_path), "schedule": git_source(schedule_revision, path).decode(),
               "schedule_path": str(path)}
    output = subprocess.check_output([sys.executable, "-IB", "-c", SCHEDULER_ENTRY, FORMAL_FREEZE, PHASE,
                                      "--reserve-minutes", str(reserve_minutes)], cwd=REPO,
                                     input=json.dumps(sources), text=True)
    return json.loads(output)


def pipeline_module():
    source = FORMAL / "pipeline.py"
    module = ModuleType("frozen_t2i_pipeline")
    module.__file__ = str(source)
    exec(compile(git_source(FORMAL_FREEZE, source), str(source), "exec"), module.__dict__)
    return module


class Controller:
    def __init__(self, pipeline, schedule_revision, reserve_minutes):
        self.pipeline = pipeline
        self.schedule_revision = schedule_revision
        self.reserve_minutes = reserve_minutes
        self.ledger = REPO / "killtests/SUBMITTED.log"
        self.root = pipeline.SHARED.resolve(strict=True) / "runs/t2irare/formal-replay" / FORMAL_FREEZE
        if (self.root.resolve(strict=True) != self.root or self.ledger.resolve(strict=True) != self.ledger
                or command("git", "ls-files", "--", "killtests/SUBMITTED.log") != "killtests/SUBMITTED.log"):
            raise ValueError("The formal output or canonical ledger path is invalid")
        self.operations = self.root / "operations"
        self.intent = self.operations / "2a-unregistered-submission.json"

    def queue(self):
        user = command("id", "-un")
        current = os.environ.get("SLURM_JOB_ID")
        rows = command(str(SLURM / "squeue"), "-h", "-u", user, "-o", "%A|%T|%j").splitlines()
        return [row for row in rows if row.split("|")[-1].startswith("t2ir-") and row.split("|")[0] != current]

    def check_plan_job(self):
        text = self.ledger.read_text()
        if ledger_entries(text, "formal-stage=1") != [STAGE1] or ledger_entries(text, "formal-plan=2a") != [PLANNER]:
            raise ValueError("The original Stage 1 and plan jobs do not match the shared ledger")
        status = command(str(SLURM / "sacct"), "-X", "-n", "-P", "-j", PLANNER,
                         "--format=JobIDRaw,JobName%40,State%40,ExitCode")
        if status != f"{PLANNER}|t2ir-formal-plan-2a|COMPLETED|0:0":
            raise ValueError("The original CPU plan has not succeeded; preserve it for diagnosis or retry")
        stage1 = self.pipeline.allocation_report(REPO, FORMAL_FREEZE, "1")
        if completed_indices(stage1, len(self.pipeline.MODEL_ORDER)) != set(range(len(self.pipeline.MODEL_ORDER))):
            raise ValueError("Stage 1 requires every original task before resuming")
        return text

    def check_disk(self, group, cost):
        command("quota", "-g", "-s")
        quota_free = quota_free_kib(command("quota", "-g"))
        # The run only contains Stage 1 at first submission. Retained output
        # sizes thereafter remain a conservative estimate for a partial run.
        used = int(command("du", "-s", "--block-size=1", str(self.root.parent.parent)).split()[0])
        stage1_rows = len(self.pipeline.MODEL_ORDER) * 14 * 2000
        projected = (used * group["images_per_task"] * len(group["indices"]) + stage1_rows - 1) // stage1_rows
        if quota_free - (projected + 1023) // 1024 < MIN_GROUP_FREE_KIB:
            raise ValueError("The shared group must retain the registered 20 GiB available disk space")
        if used + projected > (cost["disk_cap_gb_incl_used"] - cost["disk_used_gb_at_freeze"]) * 10**9:
            raise ValueError("The forecast exceeds the registered 35 GB topic disk allowance")

    def check_scientific_source(self):
        runner = FORMAL / "run.sbatch"
        if runner.read_bytes() != git_source(FORMAL_FREEZE, runner):
            raise ValueError("The live Stage 2a runner differs from its original frozen bytes")
        dirty = command("git", "status", "--porcelain", "--", str(TOPIC / "PREREG.md"), str(TOPIC / "configs"), str(TOPIC / "scripts"))
        if dirty:
            raise ValueError("The topic's scientific source paths contain uncommitted changes")

    def submit(self, group, digest, prior_arrays):
        self.check_scientific_source()
        indices = ",".join(map(str, group["indices"]))
        with self.ledger.open("a", encoding="utf-8") as ledger:
            fcntl.flock(ledger, fcntl.LOCK_EX)
            if waiting := self.queue():
                return {"state": "waiting", "queue": waiting}
            if ledger_entries(self.ledger.read_text(), "formal-stage=2a") != prior_arrays:
                raise ValueError("A new Stage 2a submission was recorded during planning")
            if self.intent.exists():
                raise ValueError("Reconcile the previous submission against Slurm and the ledger before retrying")
            with self.intent.open("x") as intent:
                json.dump({"indices": group["indices"], "plan_sha256": digest, "time_minutes": group["minutes_per_task"]}, intent)
                intent.flush()
                os.fsync(intent.fileno())
            job = command(str(SLURM / "sbatch"), "--parsable", f"--array={indices}%2", f"--time={group['minutes_per_task']}",
                          str(FORMAL / "run.sbatch"), FORMAL_FREEZE, "--receipt-sha", GATE_SHA256,
                          "--plan-sha", digest, "sample", PHASE).split(";", 1)[0]
            if not job.isdecimal():
                raise ValueError("Slurm returned an ambiguous submit identity; reconcile it before retrying")
            when = datetime.now(TAIPEI).strftime("%Y-%m-%d %H:%M")
            ledger.write(f"[codex] {when} submit {job} t2i-rare-failure formal-stage=2a 固定選格 {indices}，成本與配額已核對 {FORMAL_FREEZE}\n")
            ledger.flush()
            os.fsync(ledger.fileno())
            self.intent.unlink()
        return {"state": "submitted", "job": job, "plan_sha256": digest, "group": group}

    def advance(self):
        if waiting := self.queue():
            return {"state": "waiting", "queue": waiting}
        ledger = self.check_plan_job()
        criteria = json.loads((self.root / "configs/criteria.json").read_bytes())
        frozen = json.loads(git_source(self.pipeline.SOURCE, REPO / TOPIC / "configs/criteria.json"))
        frozen["B"]["percentage_cell_min_rate"] = self.pipeline.CONTROL_MIN_RATE
        if criteria != frozen:
            raise ValueError("The formal run settings differ from the frozen criteria")
        tasks = self.pipeline.stage_tasks(self.root, criteria, PHASE)
        digest = self.pipeline.plan_identity(self.root, PHASE)
        log = self.pipeline.SHARED.resolve() / f"tmp/t2irare/t2ir-plan-2a-{PLANNER}.out"
        if (log.resolve(strict=True) != log
                or re.findall(r"^\[codex\] plan_sha256=([0-9a-f]{64})$", log.read_text(), re.MULTILINE) != [digest]):
            raise ValueError("The 26802 receipt does not bind the current fixed selection and task list")
        arrays = ledger_entries(ledger, "formal-stage=2a")
        attempts = self.pipeline.allocation_report(REPO, FORMAL_FREEZE, PHASE) if arrays else []
        done = completed_indices(attempts, len(tasks))
        if done == set(range(len(tasks))):
            return {"state": "2a_complete", "tasks": len(tasks), "plan_sha256": digest}
        budget = sealed_scheduler(self.schedule_revision, self.reserve_minutes)
        group = checked_group(budget, tasks, done, criteria["cost"], digest, self.reserve_minutes)
        self.check_disk(group, criteria["cost"])
        return self.submit(group, digest, arrays)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller-freeze", required=True)
    parser.add_argument("--schedule-freeze", required=True)
    parser.add_argument("--reserve-minutes", type=int, default=0)
    parser.add_argument("--watch", action="store_true", help="Poll once per minute on the head node")
    args = parser.parse_args()
    if (not all(re.fullmatch(r"[0-9a-f]{40}", value) for value in (args.controller_freeze, args.schedule_freeze))
            or args.reserve_minutes < 0):
        parser.error("Provide full Git commit IDs and a nonnegative blind replay reserve")
    if git_source(args.controller_freeze, Path(__file__).resolve()) != Path(__file__).read_bytes():
        raise ValueError("Run the committed continuation source from the shared checkout")
    if args.watch and os.environ.get("SLURM_JOB_ID"):
        parser.error("Use --watch on the head node to avoid holding a Slurm allocation")
    controller = Controller(pipeline_module(), args.schedule_freeze, args.reserve_minutes)
    lock_path = controller.operations / "2a-continuation.lock"
    if lock_path.resolve() != lock_path:
        raise ValueError("The operation lock aliases another path")
    while True:
        with lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = controller.advance()
        print(json.dumps(result, ensure_ascii=False), flush=True)
        if not args.watch or result["state"] == "2a_complete":
            return
        time.sleep(60)


if __name__ == "__main__":
    main()
