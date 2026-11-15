"""Print Stage 1 and Stage 2 allocation plans under the original GPU-hour cap."""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import re
import subprocess

import pipeline


HISTORICAL_GPU_JOBS = ("26288", "26289", "26503", "26516", "26570", pipeline.GATE_JOB)
SUPERSEDED_STAGE1_FREEZE = "040dfbc5b14febb89420519fd6cc15e1bf48576d"
TOTAL_MINUTES = 50 * 60
REGISTERED_RATE = 9.0
INITIAL_FULL_RATE = 168000 / (9.17 * 3600)


def historical_gpu_seconds():
    command = [pipeline.SACCT, "-X", "-n", "-P", "-j", ",".join(HISTORICAL_GPU_JOBS),
               "--format=JobIDRaw,State%40,ElapsedRaw,AllocTRES%200"]
    records = {}
    for line in subprocess.check_output(command, text=True).splitlines():
        job, raw_state, elapsed, resources = line.split("|")
        if job in records or raw_state.split()[0] not in pipeline.TERMINAL:
            raise ValueError("Historical allocation is active or duplicated")
        gpus = int(dict(item.split("=", 1) for item in resources.split(",") if item).get("gres/gpu", 0))
        records[job] = int(elapsed) * gpus
    if set(records) != set(HISTORICAL_GPU_JOBS):
        raise ValueError("Missing registered T2I historical allocation")
    return sum(records.values())


def task_rows(task, prompt_count):
    _, prompt, lo, hi = task
    return (hi - lo) * (prompt_count if prompt == "all" else 1)


def time_groups(work, rate, remaining_seconds):
    """Allocate fixed array limits, with no array limit for a forecast over budget."""
    total_rows = sum(rows for _, rows in work)
    if total_rows == 0:
        return []
    maxima = [int(2 * rows / REGISTERED_RATE / 60) for _, rows in work]
    required = [math.ceil(rows / rate / 60) for _, rows in work]
    proposed = [min(maximum, remaining_seconds * rows // (total_rows * 60))
                for (_, rows), maximum in zip(work, maxima, strict=True)]
    if any(limit < need for limit, need in zip(proposed, required, strict=True)):
        # Proportional rounding can strand enough minutes to fund a short tail.
        if sum(required) * 60 > remaining_seconds or any(
                need > maximum for need, maximum in zip(required, maxima, strict=True)):
            return None
        proposed = required
    groups = defaultdict(list)
    for (index, rows), minutes in zip(work, proposed, strict=True):
        groups[(rows, minutes)].append(index)
    return [{"images_per_task": rows, "minutes_per_task": minutes, "indices": indices}
            for (rows, minutes), indices in sorted(groups.items(), reverse=True)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("freeze")
    ap.add_argument("phase", choices=pipeline.PHASES)
    args = ap.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.freeze):
        ap.error("Use the full original launch commit")
    repo = Path(__file__).resolve().parents[3]
    root = pipeline.SHARED.resolve(strict=True) / "runs/t2irare/formal-replay" / args.freeze
    config = root / "configs/criteria.json"
    if config.exists():
        criteria = json.loads(config.read_bytes())
    elif args.phase == "1":
        criteria = json.loads(pipeline.git_file(repo, pipeline.SOURCE, pipeline.TOPIC / "configs/criteria.json"))
    else:
        ap.error("First produce the original stage's selection manifest")
    prompts = json.loads(pipeline.git_file(repo, pipeline.SOURCE, pipeline.TOPIC / "configs/prompts.json"))["prompts"]
    initial = historical_gpu_seconds()
    if args.freeze != SUPERSEDED_STAGE1_FREEZE:
        initial += sum(attempt[-1] for attempt in pipeline.allocation_report(repo, SUPERSEDED_STAGE1_FREEZE, "1"))
    spent, previous_rows = initial, 0
    for previous in pipeline.PHASES[:pipeline.PHASES.index(args.phase)]:
        tasks = pipeline.stage_tasks(root, criteria, previous)
        if tasks:
            spent += sum(row[-1] for row in pipeline.allocation_report(repo, args.freeze, previous))
            previous_rows += sum(task_rows(task, len(prompts)) for task in tasks)
    rate = previous_rows / (spent - initial) if previous_rows else INITIAL_FULL_RATE
    stage_tasks = pipeline.stage_tasks(root, criteria, args.phase)
    ledger = (repo / "killtests/SUBMITTED.log").read_text()
    attempts = []
    if re.search(rf"submit [0-9]+ t2i-rare-failure formal-stage={args.phase} .+ {args.freeze[:7]}[0-9a-f]*$", ledger, re.MULTILINE):
        attempts = pipeline.allocation_report(repo, args.freeze, args.phase)
        spent += sum(row[-1] for row in attempts)
    finished = {index for _, index, _, state, code, _ in attempts if (state, code) == ("COMPLETED", "0:0")}
    work = [(index, task_rows(task, len(prompts))) for index, task in enumerate(stage_tasks) if index not in finished]
    remaining = TOTAL_MINUTES * 60 - spent
    groups = time_groups(work, rate, remaining)
    print(json.dumps({"formal_phase": args.phase, "historical_gpu_seconds": initial, "total_spent_gpu_seconds": spent,
                      "remaining_gpu_seconds": remaining, "observed_images_per_gpu_second": rate,
                      "forecast_gpu_seconds_for_remaining_tasks": sum(rows for _, rows in work) / rate,
                      "plan_sha256": None if args.phase == "1" else pipeline.plan_identity(root, args.phase),
                      "array_groups": groups}, indent=2))


if __name__ == "__main__":
    main()
