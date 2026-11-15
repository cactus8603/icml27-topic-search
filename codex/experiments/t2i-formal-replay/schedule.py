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


def time_groups(work, rate, remaining_seconds, task_rates=None):
    """Allocate fixed array limits, with no array limit for a forecast over budget."""
    total_rows = sum(rows for _, rows in work)
    if total_rows == 0:
        return []
    task_rates = task_rates or {}
    maxima = [int(2 * rows / REGISTERED_RATE / 60) for _, rows in work]
    required = [math.ceil(rows / task_rates.get(index, rate) / 60) for index, rows in work]
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


def phase_time_budget(total_remaining, stage_rows, stage_spent, reserve_seconds):
    phase_remaining = math.floor(2 * stage_rows / REGISTERED_RATE) - stage_spent
    # The frozen sampler rejects reaching the total cap exactly.
    return min(total_remaining - reserve_seconds - 1, phase_remaining)


def completed_rates(tasks, attempts, prompt_count):
    elapsed = defaultdict(int)
    successful = set()
    for _, index, _, state, code, seconds in attempts:
        if not 0 <= index < len(tasks):
            raise ValueError("A previous allocation index is outside the frozen task list")
        elapsed[index] += seconds
        if (state, code) == ("COMPLETED", "0:0"):
            successful.add(index)
    if successful != set(range(len(tasks))) or any(seconds <= 0 for seconds in elapsed.values()):
        raise ValueError("A previous phase lacks a successful GPU allocation for every task")
    rows_by_cell, seconds_by_cell = defaultdict(int), defaultdict(int)
    for index, task in enumerate(tasks):
        cell = task[:2]
        rows_by_cell[cell] += task_rows(task, prompt_count)
        seconds_by_cell[cell] += elapsed[index]
    return sum(elapsed.values()), {cell: rows / seconds_by_cell[cell] for cell, rows in rows_by_cell.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("freeze")
    ap.add_argument("phase", choices=pipeline.PHASES)
    ap.add_argument("--reserve-minutes", type=int, default=0, help="GPU minutes set aside for the blind replay")
    args = ap.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.freeze):
        ap.error("Use the full original launch commit")
    if args.reserve_minutes < 0:
        ap.error("Reserved minutes must be nonnegative")
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
    spent, previous_rows, matched_rates = initial, 0, {}
    for previous in pipeline.PHASES[:pipeline.PHASES.index(args.phase)]:
        tasks = pipeline.stage_tasks(root, criteria, previous)
        if tasks:
            attempts = pipeline.allocation_report(repo, args.freeze, previous)
            phase_seconds, rates = completed_rates(tasks, attempts, len(prompts))
            spent += phase_seconds
            matched_rates.update(rates)
            previous_rows += sum(task_rows(task, len(prompts)) for task in tasks)
    rate = previous_rows / (spent - initial) if previous_rows else INITIAL_FULL_RATE
    stage_tasks = pipeline.stage_tasks(root, criteria, args.phase)
    ledger = (repo / "killtests/SUBMITTED.log").read_text()
    attempts = []
    if re.search(rf"submit [0-9]+ t2i-rare-failure formal-stage={args.phase} .+ {args.freeze[:7]}[0-9a-f]*$", ledger, re.MULTILINE):
        attempts = pipeline.allocation_report(repo, args.freeze, args.phase)
    if any(not 0 <= attempt[1] < len(stage_tasks) for attempt in attempts):
        raise ValueError("An allocation index is outside the frozen task list for this phase")
    phase_spent = sum(row[-1] for row in attempts)
    spent += phase_spent
    finished = {index for _, index, _, state, code, _ in attempts if (state, code) == ("COMPLETED", "0:0")}
    work = [(index, task_rows(task, len(prompts))) for index, task in enumerate(stage_tasks) if index not in finished]
    remaining = TOTAL_MINUTES * 60 - spent
    reserve = args.reserve_minutes * 60
    phase_rows = sum(task_rows(task, len(prompts)) for task in stage_tasks)
    available = phase_time_budget(remaining, phase_rows, phase_spent, reserve)
    groups = time_groups(work, rate, available)
    rate_basis, task_rates = "pooled", {}
    if groups is None and matched_rates:
        task_rates = {
            index: matched_rates.get(stage_tasks[index][:2], matched_rates.get((stage_tasks[index][0], "all"), rate))
            for index, _ in work
        }
        groups = time_groups(work, rate, available, task_rates)
        rate_basis = "matching_model_or_cell"
    print(json.dumps({"formal_phase": args.phase, "historical_gpu_seconds": initial, "total_spent_gpu_seconds": spent,
                      "remaining_gpu_seconds": remaining, "current_phase_gpu_seconds": phase_spent,
                      "reserved_gpu_seconds": reserve, "available_gpu_seconds_for_phase": available,
                      "observed_images_per_gpu_second": rate, "forecast_rate_basis": rate_basis,
                      "forecast_gpu_seconds_for_remaining_tasks": sum(rows / task_rates.get(index, rate)
                                                                      for index, rows in work),
                      "plan_sha256": None if args.phase == "1" else pipeline.plan_identity(root, args.phase),
                      "array_groups": groups}, indent=2))


if __name__ == "__main__":
    main()
