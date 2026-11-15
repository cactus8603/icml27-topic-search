"""Run the original frozen T2I stages in one isolated, replay-qualified workspace."""

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from unittest.mock import patch


STAGE = Path("codex/experiments/t2i-formal-replay")
TOPIC = Path("killtests/t2i-rare-failure")
SOURCE = "b54a86fb05e845e2c11139822beac8846c63c77a"
GATE_DIR = Path("codex/experiments/t2i-gate-b-original-replay")
GATE_FREEZE = "10d42bdf86534a44b47c03d56ea5e1dd749cdbac"
GATE_JOB = "26663"
SHARED = Path("/work/ChihChungHsu/cactus8603/icml")
SACCT = "/cm/local/apps/slurm/current/bin/sacct"
MODEL_ORDER = ("sd14", "uce", "esd", "salun", "rece", "advunlearn")
PHASES = ("1", "2a", "2b", "2b_extra")
CHUNK = 500
BATCH = 32
CONTROL_MIN_RATE = 0.01
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "BOOT_FAIL",
            "PREEMPTED", "REVOKED", "DEADLINE"}


def git_file(repo, revision, path):
    return subprocess.check_output(["git", "-C", str(repo), "show", f"{revision}:{path}"])


def digest(data):
    return hashlib.sha256(data).hexdigest()


def replay_receipt(shared):
    shared_root = shared.resolve(strict=True)
    parent = shared_root / f"runs/t2irare/gate-b-original-replay/{GATE_JOB}"
    receipt = parent / "precision_replay_verified.json"
    if parent.resolve(strict=True) != parent or receipt.is_symlink() or not receipt.is_file():
        raise ValueError("The repaired Gate (b) receipt is missing or aliased")
    return receipt


def gate_replay(repo, receipt_sha):
    command = [SACCT, "-X", "-n", "-P", "-j", GATE_JOB, "--format=JobIDRaw,State,ExitCode,JobName%40"]
    records = subprocess.check_output(command, text=True).strip().splitlines()
    if records != [f"{GATE_JOB}|COMPLETED|0:0|t2ir-gate-b-original-replay"]:
        raise ValueError("The registered original-seed replay has no successful allocation record")
    ledger = (repo / "killtests/SUBMITTED.log").read_text().splitlines()
    pattern = re.compile(rf"^\[codex\] .+ submit {GATE_JOB} t2i-rare-failure .+ ([0-9a-f]{{7,40}})$")
    records = [match[1] for line in ledger if (match := pattern.fullmatch(line))]
    if len(records) != 1 or not GATE_FREEZE.startswith(records[0]):
        raise ValueError("The original-seed replay is not bound to the registered launch")
    subprocess.run(["bash", str(repo / GATE_DIR / "run.sbatch"), GATE_FREEZE, "--verify-job", GATE_JOB], cwd=repo, check=True)
    body = replay_receipt(SHARED).read_bytes()
    result = json.loads(body)
    if (digest(body) != receipt_sha or result["status"] != "PASS" or result["frozen_commit"] != GATE_FREEZE
            or result["slurm_job_id"] != GATE_JOB or result["score_chunk_count"] != 60):
        raise ValueError("The repaired Gate (b) receipt did not match the independently reviewed passing evidence")
    return result


def formal_configuration(upstream):
    path = upstream / "configs/criteria.json"
    criteria = json.loads(path.read_bytes())
    if criteria["B"].get("percentage_cell_min_rate") is not None:
        raise ValueError("The archived criteria already specify the positive-control floor")
    criteria["B"]["percentage_cell_min_rate"] = CONTROL_MIN_RATE
    path.write_text(json.dumps(criteria, ensure_ascii=False, indent=2) + "\n")
    return criteria


def stage_tasks(run, criteria, phase):
    screen = criteria["screen"]
    if phase == "1":
        return [(model, "all", 0, screen["stage1_n"]) for model in MODEL_ORDER]
    path = run / "configs" / f"stage{phase}_cells.json"
    spec = json.loads(path.read_bytes())
    cells = [(row["model"], row["prompt"]) for row in spec["cells"]]
    start, target = (screen["stage1_n"], screen["stage2a_n"]) if phase == "2a" else (screen["stage2a_n"], screen["stage2b_n"])
    if (spec["stage"] != phase or spec["event"] != criteria["frozen_event"]["definition"]
            or (spec["start"], spec["target"]) != (start, target) or len(cells) != len(set(cells))):
        raise ValueError(f"The {phase} registered selection manifest is malformed")
    tasks = [(m, p, lo, min(lo + screen["shard"], target)) for m, p in cells for lo in range(start, target, screen["shard"])]
    tsv = path.with_name(f"stage{phase}_tasks.tsv").read_text()
    if tsv != "".join("\t".join(map(str, row)) + "\n" for row in tasks):
        raise ValueError(f"The {phase} task list differs from the selected ranges")
    return tasks


def plan_identity(run, phase):
    root = run / "configs"
    return digest((root / f"stage{phase}_cells.json").read_bytes() + (root / f"stage{phase}_tasks.tsv").read_bytes())


def output_chunks(run, task, prompts, evidence, complete):
    from common import load_json

    model, prompt, start, stop = task
    models = load_json("models.json")["models"]
    sampling = load_json("sampling.json")
    for pid in prompts if prompt == "all" else (prompt,):
        if pid not in prompts:
            raise ValueError("A selected prompt is outside the registered panel")
        for lo in range(start, stop, CHUNK):
            hi = min(lo + CHUNK, stop)
            path = run / "scores" / model / pid / f"s{lo:07d}_{hi:07d}.npz"
            sidecar = path.with_suffix(".json")
            partial = path.with_name(path.name + ".tmp.npz")
            if complete or path.exists() or sidecar.exists() or partial.exists():
                if (not path.is_file() or not sidecar.is_file() or path.resolve() != path or sidecar.resolve() != sidecar
                        or partial.exists()):
                    raise ValueError("A score chunk is incomplete or aliased. Preserve it for diagnosis before retrying.")
                metadata = evidence.validate_chunk(path, model, pid, lo, hi, evidence.DETECTORS, sampling, prompts, models, None)
                if metadata["batch_size"] != BATCH:
                    raise ValueError("A generated chunk differs from the registered batch size")
                yield path


def allocation_report(repo, freeze, phase):
    ledger = (repo / "killtests/SUBMITTED.log").read_text().splitlines()
    pattern = re.compile(rf"^\[codex\] .+ submit ([0-9]+) t2i-rare-failure formal-stage={phase} .+ ([0-9a-f]{{7,40}})$")
    arrays = [match[1] for line in ledger if (match := pattern.fullmatch(line)) and freeze.startswith(match[2])]
    if not arrays or len(set(arrays)) != len(arrays):
        raise ValueError(f"The Stage {phase} allocations are missing or duplicated in the ledger")
    command = [SACCT, "--duplicates", "-X", "-n", "-P", "-j", ",".join(arrays),
               "--format=JobIDRaw,JobID,JobName%64,State%40,ExitCode,ElapsedRaw,AllocTRES%200,Start"]
    records = subprocess.check_output(command, text=True)
    return parse_allocations(arrays, records)


def parse_allocations(arrays, records):
    attempts, seen_arrays = [], set()
    seen_attempts = set()
    for line in records.splitlines():
        job, task, name, raw_state, code, elapsed, resources, start = line.split("|")
        state = raw_state.split()[0]
        if name != "t2ir-formal-replay" or state not in TERMINAL:
            raise ValueError("A registered allocation has the wrong name or is still active")
        match = re.fullmatch(r"([0-9]+)_([0-9]+)", task)
        gpu_seconds = int(elapsed) * int(dict(item.split("=", 1) for item in resources.split(",") if item).get("gres/gpu", 0))
        if match:
            array, index = match.groups()
            key = (job, start)
            if key in seen_attempts or not job.isdecimal() or gpu_seconds < 0:
                raise ValueError("A registered Slurm allocation is duplicated or malformed")
            seen_attempts.add(key)
            attempts.append((array, int(index), job, state, code, gpu_seconds))
        elif state == "CANCELLED" and int(elapsed) == 0 and (pending := re.fullmatch(r"([0-9]+)(?:_\[[0-9,%-]+\])?", task)):
            array = pending[1]
        else:
            raise ValueError("Slurm accounting has an unexpected task identity")
        seen_arrays.add(array)
    if seen_arrays != set(arrays):
        raise ValueError("Slurm accounting is missing a registered array or includes a different run")
    return attempts


def complete_previous(repo, run, freeze, receipt_sha, criteria, prompts, evidence, previous):
    expected = set()
    gpu_seconds = 0
    for phase in previous:
        tasks = stage_tasks(run, criteria, phase)
        if not tasks:
            continue
        attempts = allocation_report(repo, freeze, phase)
        if any(not 0 <= attempt[1] < len(tasks) for attempt in attempts):
            raise ValueError("A Stage allocation index is outside the original plan")
        allocated = sum(attempt[-1] for attempt in attempts)
        gpu_seconds += allocated
        for index, task in enumerate(tasks):
            paths = set(output_chunks(run, task, prompts, evidence, complete=True))
            expected.update(paths)
            workers = {attempt[2] for attempt in attempts if attempt[1] == index and attempt[-1] > 0}
            if any(str(evidence.read_json(path.with_suffix(".json"))["slurm_job"]) not in workers for path in paths):
                raise ValueError("A score chunk lacks a traceable allocated task")
            completions = [(array, job) for array, i, job, state, code, _ in attempts
                           if i == index and (state, code) == ("COMPLETED", "0:0")]
            if not any((receipt := run / "operations" / f"{phase}-{index}-{job}.json").is_file()
                       and evidence.read_json(receipt) == {
                           "freeze": freeze, "phase": phase, "task": list(task), "job": job, "array": array,
                           "chunk_count": len(paths), "gate_receipt_sha256": receipt_sha,
                       } for array, job in completions):
                raise ValueError("A completed task lacks the matching operation receipt")
        phase_rows = sum((stop - start) * (len(prompts) if prompt == "all" else 1) for _, prompt, start, stop in tasks)
        if allocated > 2 * phase_rows / criteria["cost"]["est_img_per_s"]:
            raise ValueError(f"Stage {phase} exceeded twice its registered estimated GPU allocation")
    found = set((run / "scores").glob("*/*/*.npz"))
    if found != expected:
        raise ValueError("The corpus has missing or out-of-phase score chunks")
    if gpu_seconds >= criteria["cost"]["cap_gpu_hours"] * 3600:
        raise ValueError("Generation reached the original overall allocation cap")
    return gpu_seconds


@contextmanager
def operation_lock(run, name):
    directory = run / "operations"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.lock"
    if path.resolve() != path:
        raise ValueError("Operation lock path is aliased")
    with path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def sample(run, upstream, task, phase, index, freeze, receipt_sha, prompts, evidence):
    import torch

    job = os.environ.get("SLURM_JOB_ID", "")
    array = os.environ.get("SLURM_ARRAY_JOB_ID", "")
    if not (job.isdecimal() and array.isdecimal() and torch.cuda.is_available() and "H200" in torch.cuda.get_device_name()):
        raise ValueError("Generation requires the registered Slurm H200 array task")
    model, prompt, start, stop = task
    with operation_lock(run, f"{phase}-{index}"):
        list(output_chunks(run, task, prompts, evidence, complete=False))
        subprocess.run([sys.executable, "-IB", str(upstream / "scripts/generate.py"), "--model", model,
                        "--prompts", prompt, "--seed-start", str(start), "--seed-end", str(stop),
                        "--chunk", str(CHUNK), "--batch-size", str(BATCH), "--out", str(run)], check=True)
        paths = list(output_chunks(run, task, prompts, evidence, complete=True))
        evidence.atomic_json(run / "operations" / f"{phase}-{index}-{job}.json", {
            "freeze": freeze, "phase": phase, "task": list(task), "job": job, "array": array,
            "chunk_count": len(paths), "gate_receipt_sha256": receipt_sha,
        })


def call_script(module, run, gate, *args):
    import evidence
    import stage_selection

    # Preserve the original numerical routines and point their manifest lookups
    # at this run. The gate replacement is a source-verified boolean, not a score.
    with patch.object(module, "ROOT", run), patch.object(module, "verified_status", lambda _: gate), \
            patch.object(stage_selection, "ROOT", run), patch.object(evidence, "ROOT", run), \
            patch.object(sys, "argv", [module.__name__, *args]):
        module.main()


def initialize_run(run, upstream, evidence):
    directory = run / "configs"
    directory.mkdir(parents=True, exist_ok=True)
    for name in (*evidence.CONFIGS, "review_rubric.md"):
        original = (upstream / "configs" / name).read_text()
        destination = directory / name
        if destination.is_symlink():
            raise ValueError("Formal run configuration aliases another path")
        if not destination.exists():
            try:
                evidence.atomic_text(destination, original)
            except FileNotFoundError:
                if not destination.is_file():
                    raise
        if destination.read_text() != original:
            raise ValueError("Formal run configuration changed between stages")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--receipt-sha")
    parser.add_argument("--plan-sha")
    parser.add_argument("command", choices=("preflight", "plan", "sample", "aggregate", "make-blind", "regen", "review-stats"))
    parser.add_argument("phase", nargs="?", choices=PHASES)
    args = parser.parse_args()
    if (args.command in {"plan", "sample"}) != (args.phase is not None):
        parser.error("Provide a phase only for plan or sample")
    repo, upstream = args.repo.resolve(strict=True), args.upstream.resolve(strict=True)
    if not re.fullmatch(r"[0-9a-f]{40}", args.freeze) or git_file(repo, args.freeze, STAGE / "pipeline.py") != Path(__file__).read_bytes():
        raise ValueError("Run the committed formal workflow archive")
    if git_file(repo, SOURCE, TOPIC / "configs/criteria.json") != (upstream / "configs/criteria.json").read_bytes():
        raise ValueError("The archived criteria differ from the frozen topic")
    sys.path.insert(0, str(upstream / "scripts"))
    import evidence
    from common import prompts_by_id
    from stage1_check import plan
    from test_evidence import __file__ as test_evidence_path
    from run_candidate import verify_loader_refs

    verify_loader_refs(upstream, plan(upstream))
    prompts = prompts_by_id()
    if args.command == "preflight":
        subprocess.run([sys.executable, "-IB", str(Path(__file__).with_name("test_pipeline.py"))], check=True)
        subprocess.run([sys.executable, "-B", test_evidence_path], check=True)
        subprocess.run([sys.executable, "-IB", str(upstream / "scripts/aggregate.py"), "--selftest"], check=True)
        print("[codex] Frozen inference assets, coverage and analysis verified. No target scores opened.")
        return
    # Add the blind-control floor only to the temporary run configuration.
    criteria = formal_configuration(upstream)
    if not re.fullmatch(r"[0-9a-f]{64}", args.receipt_sha or ""):
        raise ValueError("The independently reviewed repaired Gate (b) digest is required")
    gate_result = gate_replay(repo, args.receipt_sha)
    gate = gate_result["status"] == "PASS"
    run = SHARED.resolve() / "runs/t2irare/formal-replay" / args.freeze
    if run.resolve() != run:
        raise ValueError("Formal output path aliases another run")
    initialize_run(run, upstream, evidence)
    if args.command == "sample":
        if args.phase != "1" and plan_identity(run, args.phase) != args.plan_sha:
            raise ValueError("The Stage 2 array must match the fixed selection manifest digest")
        index = int(os.environ["SLURM_ARRAY_TASK_ID"])
        tasks = stage_tasks(run, criteria, args.phase)
        if not 0 <= index < len(tasks):
            raise ValueError("The Slurm array index is outside the registered task list")
        sample(run, upstream, tasks[index], args.phase, index, args.freeze, args.receipt_sha, prompts, evidence)
        return
    with operation_lock(run, "screen-review"):
        if args.command == "plan":
            import select_stage2_cells

            if args.phase not in PHASES[1:]:
                raise ValueError("Choose the next registered stage: 2a, 2b, or 2b_extra")
            allocated = complete_previous(repo, run, args.freeze, args.receipt_sha, criteria, prompts, evidence,
                                          PHASES[:PHASES.index(args.phase)])
            if (run / "configs" / f"stage{args.phase}_cells.json").exists():
                raise ValueError("This stage already has a selection manifest")
            call_script(select_stage2_cells, run, gate, "--results", str(run), "--stage", args.phase)
            print(f"[codex] completed_gpu_seconds={allocated}")
            print(f"[codex] plan_sha256={plan_identity(run, args.phase)}")
        elif args.command == "aggregate":
            import aggregate

            phases = tuple(p for p in PHASES if p != "2b_extra" or (run / "configs/stage2b_extra_cells.json").exists())
            complete_previous(repo, run, args.freeze, args.receipt_sha, criteria, prompts, evidence, phases)
            recovered = run / "gate_a_prime/gate_a_prime.json"
            historical = repo / TOPIC / "results/gate_a_prime/gate_a_prime.json"
            if historical.is_file() and not recovered.exists():
                recovered.parent.mkdir(parents=True, exist_ok=True)
                recovered.symlink_to(historical)
            call_script(aggregate, run, gate, "--results", str(run))
        elif args.command == "make-blind":
            import make_review_list

            if (run / "review/review_blind.csv").exists():
                raise ValueError("The existing blind review packet must be preserved")
            call_script(make_review_list, run, gate, "--results", str(run))
        elif args.command == "regen":
            subprocess.run([sys.executable, "-IB", str(upstream / "scripts/regen_images.py"), "--results", str(run)], check=True)
        else:
            import review_stats

            call_script(review_stats, run, gate, str(run))


if __name__ == "__main__":
    main()
