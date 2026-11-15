"""Validate a complete activated candidate Stage 1 and reproduce the frozen 2a list."""

import argparse
from collections import defaultdict
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import re
import subprocess
import sys


HERE = Path(__file__).resolve().parent
TOPIC = Path("killtests/t2i-rare-failure")
STAGE = Path("codex/experiments/t2i-stage1-screen")
PRODUCER = "a5d3fb86d489dbdc2b7209179879ba3b1e7ecedb"
UPSTREAM = "b54a86fb05e845e2c11139822beac8846c63c77a"
PLAN_COMMIT = "0d86f961a09954cec06434999e63b136bf0a83b1"
SACCT = "/cm/local/apps/slurm/current/bin/sacct"
JOB_NAME = "t2ir-s1-candidate"
SCOPE = "candidate_stage1_2a_screen_only"
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "BOOT_FAIL",
            "PREEMPTED", "REVOKED", "DEADLINE"}

producer = spec_from_file_location("stage1_candidate", HERE.parent / "t2i-stage1-candidate/run.py")
launcher = module_from_spec(producer)
producer.loader.exec_module(launcher)


def git_file(repo, revision, path):
    return subprocess.check_output(["git", "-C", str(repo), "show", f"{revision}:{path}"])


def verify_sources(repo, upstream):
    files = ("configs/criteria.json", "configs/detectors.json", "configs/models.json", "configs/prompts.json",
             "configs/sampling.json", "scripts/common.py", "scripts/evidence.py", "scripts/select_stage2_cells.py",
             "scripts/stage_selection.py", "scripts/verify_gate_b.py")
    for name in files:
        if (upstream / name).read_bytes() != git_file(repo, UPSTREAM, TOPIC / name):
            raise ValueError(f"Archived Stage 1 source differs: {name}")
    for folder, commit, names in (
        ("t2i-stage1-candidate", PRODUCER, ("run.py",)),
        ("t2i-detector-probe", PLAN_COMMIT, ("probe.py", "stage1_check.py")),
        ("t2i-gate-repair", UPSTREAM, ("review_evidence.py",)),
    ):
        for name in names:
            path = Path(folder) / name
            if (HERE.parent / path).read_bytes() != git_file(repo, commit, STAGE.parent / path):
                raise ValueError(f"Archived candidate source differs: {path}")


def registered_arrays(submissions, freeze):
    pattern = re.compile(r"^\[(?:codex|claude)\] \d{4}-\d\d-\d\d \d\d:\d\d submit ([0-9]+) t2i-rare-failure .+ ([0-9a-f]{7,40})$")
    arrays = []
    for line in submissions.splitlines():
        match = pattern.fullmatch(line)
        if match and freeze.startswith(match[2]):
            arrays.append(match[1])
    if not arrays or len(arrays) != len(set(arrays)):
        raise ValueError("Stage 1 array submissions are missing or duplicated in the shared ledger")
    return arrays


def accounting_rows(payload, arrays):
    attempts, episodes, seen_arrays = defaultdict(list), set(), set()
    for line in payload.splitlines():
        task, array_index, name, state, exit_code, elapsed, tres, start = line.split("|")
        if name != JOB_NAME:
            raise ValueError("A registered Stage 1 submission has the wrong allocation name")
        state = state.split()[0]
        elapsed = int(elapsed)
        fields = dict(entry.split("=", 1) for entry in tres.split(",") if entry)
        gpus = int(fields.get("gres/gpu", "0"))
        match = re.fullmatch(r"([0-9]+)_([0-5])", array_index)
        cancelled_pending = re.fullmatch(r"([0-9]+)(?:_\[[0-9,%-]+\])?", array_index)
        if (not match and cancelled_pending and cancelled_pending[1] in arrays
                and state == "CANCELLED" and elapsed == 0 and gpus == 0):
            seen_arrays.add(cancelled_pending[1])
            continue
        if not match or match[1] not in arrays or not re.fullmatch(r"[0-9]+", task) or state not in TERMINAL:
            raise ValueError("Stage 1 accounting contains an unbound or non-terminal array task")
        array, index = match.groups()
        seen_arrays.add(array)
        if elapsed < 0 or (gpus == 0 and (elapsed != 0 or state == "COMPLETED")) or gpus not in (0, 1):
            raise ValueError("Invalid Stage 1 GPU allocation or elapsed time")
        episode = (task, start)
        if episode in episodes:
            raise ValueError("Stage 1 accounting episodes are ambiguous")
        episodes.add(episode)
        attempts[task].append({"array": array, "model": launcher.MODELS[int(index)], "state": state,
                               "exit_code": exit_code, "gpu_seconds": gpus * elapsed})
    if seen_arrays != set(arrays) or any(len({(x["array"], x["model"]) for x in rows}) != 1 for rows in attempts.values()):
        raise ValueError("Stage 1 accounting is missing an array or task identity is inconsistent")
    return attempts


def collect_accounting(repo, freeze):
    arrays = registered_arrays((repo / "killtests/SUBMITTED.log").read_text(), freeze)
    command = [SACCT, "--duplicates", "-X", "-n", "-P", "-j", ",".join(arrays),
               "--format=JobIDRaw,JobID,JobName%64,State%40,ExitCode,ElapsedRaw,AllocTRES%200,Start"]
    return arrays, accounting_rows(subprocess.check_output(command, text=True), arrays)


def completion_receipts(paths, attempts, freeze, config, revisions):
    models, jobs, prerequisites = set(), set(), []
    for path in paths:
        if path.resolve() != path:
            raise ValueError("Candidate completion receipt aliases a different run")
        row = launcher.review.parse_json(path.read_bytes())
        model, job = row["model"], row["slurm_job_id"]
        if (path.name != f"{model}-{job}.json" or job in jobs or model not in launcher.MODELS
                or row["scope"] != launcher.SCOPE or row["launch_commit"] != freeze or row["score_chunks"] != 56
                or row["config_sha256"] != config or row["proxy_receipt"] != launcher.PROXY_CANDIDATE
                or row["resolved_hf_revisions"] != revisions):
            raise ValueError("Candidate model completion receipt differs from the activated run")
        if not any(record["state"] == "COMPLETED" and record["exit_code"] == "0:0"
                   and record["array"] == row["slurm_array_job_id"] and record["model"] == model
                   for record in attempts.get(job, ())):
            raise ValueError("Candidate model receipt lacks a successful allocation")
        prerequisite = row["prerequisites"]
        if (not re.fullmatch(r"[0-9a-f]{64}", prerequisite["gate_a_sha256"])
                or prerequisite["bench"]["pass"] is not True):
            raise ValueError("Candidate producer prerequisites are incomplete")
        models.add(model)
        jobs.add(job)
        prerequisites.append(prerequisite)
    if models != set(launcher.MODELS) or any(row != prerequisites[0] for row in prerequisites[1:]):
        raise ValueError("All six model completions with identical prerequisites are required")
    return sorted(jobs, key=int)


def summarize_cost(attempts, chunk_seconds, artifact_bytes):
    totals = {model: 0 for model in launcher.MODELS}
    for rows in attempts.values():
        for row in rows:
            totals[row["model"]] += row["gpu_seconds"]
    allocated = sum(totals.values())
    return {"allocated_gpu_seconds": allocated, "allocated_gpu_hours": allocated / 3600,
            "allocated_gpu_seconds_by_model": totals,
            "retained_chunk_seconds": chunk_seconds, "score_and_sidecar_bytes": artifact_bytes,
            "allocation_episodes": sum(map(len, attempts.values()))}


def select(scores, stage_selection):
    return [{"model": model, "prompt": prompt} for model, prompt in stage_selection(scores, "2a")]


def audit(root, freeze, arrays, attempts, evidence, criteria, prompts, stage_selection):
    from probe import CLIP_REPO, CLIP_REV, FAL_REPO, FAL_REV
    from stage1_check import REVISIONS

    revisions = REVISIONS | {CLIP_REPO: CLIP_REV, FAL_REPO: FAL_REV}
    config = evidence.config_sha()
    receipts = sorted((root / "operations").glob("*.json"))
    jobs = completion_receipts(receipts, attempts, freeze, config, revisions)
    stop = criteria["screen"]["stage1_n"]
    panel = {"prompt_ids": list(prompts), "seed_range_per_cell": [0, stop], "default_batch_size": 32}
    registered = evidence.registered_cells()
    for model in launcher.MODELS:
        if launcher.verify_chunks(root, panel, model, evidence, complete=True) != 56:
            raise ValueError("Candidate model does not contain all 56 planned chunks")
    chunks = sorted((root / "scores").glob("*/*/*.npz"))
    sidecars = sorted((root / "scores").glob("*/*/*.json"))
    if len(chunks) != 336 or set(sidecars) != {path.with_suffix(".json") for path in chunks}:
        raise ValueError("Expected exactly 336 score/metadata pairs")
    elapsed = 0.0
    for sidecar in sidecars:
        meta = evidence.read_json(sidecar)
        producer_id, model = meta["slurm_job"], sidecar.parent.parent.name
        if not any(row["model"] == model and row["gpu_seconds"] > 0 for row in attempts.get(producer_id, ())):
            raise ValueError("Score chunk producer is not tied to the registered model allocation")
        seconds = meta["seconds"]
        if type(seconds) not in (int, float) or not 0 < seconds < float("inf"):
            raise ValueError("Score chunk duration is invalid")
        elapsed += seconds
    scores, payload_sha = evidence.load_scores(root, expected_cells=registered)
    if len(scores) != 168000 or len(registered) != 84 or not 0 <= scores.seed.min() <= scores.seed.max() < stop:
        raise ValueError("Candidate scores differ from the full Stage 1 panel")
    cost = summarize_cost(attempts, elapsed, sum(p.stat().st_size for p in (*chunks, *sidecars)))
    screen = criteria["screen"]
    return {"scope": SCOPE, "producer_freeze": freeze, "source_sha256": payload_sha, "config_sha256": config,
            "array_jobs": arrays, "completion_jobs": jobs, "rows": len(scores), "chunks": len(chunks), "cost": cost,
            "screen_event": criteria["frozen_event"]["definition"], "screen_cells": select(scores, stage_selection),
            "candidate_next_seed_range": [stop, screen["stage2a_n"]]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=HERE.parents[2])
    parser.add_argument("--upstream", type=Path)
    parser.add_argument("--analysis-freeze")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--activation-freeze")
    args = parser.parse_args()
    repo = args.repo.resolve()
    upstream = (args.upstream or repo / TOPIC).resolve()
    verify_sources(repo, upstream)
    sys.path.insert(0, str(upstream / "scripts"))
    import evidence
    from common import load_json, prompts_by_id
    from stage_selection import stage_candidates

    criteria, prompts = load_json("criteria.json"), prompts_by_id()
    if (len(evidence.registered_cells()) != 84 or criteria["screen"]["stage1_n"] != 2000
            or launcher.CHUNK != 500 or len(launcher.MODELS) != 6):
        raise ValueError("Source does not contain the registered 168,000-row Stage 1 panel")
    if args.check_only:
        print(json.dumps({"scope": SCOPE, "expected_rows": 168000, "expected_chunks": 336,
                          "config_sha256": evidence.config_sha(), "verified_source": UPSTREAM}, sort_keys=True))
        return
    freeze = args.activation_freeze
    if not all(re.fullmatch(r"[0-9a-f]{40}", value or "") for value in (freeze, args.analysis_freeze)):
        parser.error("Use full analysis and activation freezes")
    if git_file(repo, args.analysis_freeze, STAGE / "screen.py") != Path(__file__).read_bytes():
        raise ValueError("Running audit source differs from its analysis freeze")
    for name in ("run.py", "run.sbatch"):
        file = launcher.STAGE / name
        if git_file(repo, freeze, file) != git_file(repo, PRODUCER, file):
            raise ValueError("Activation commit changed the frozen candidate producer")
    launcher.activation(git_file(repo, freeze, launcher.STAGE / launcher.ACTIVATION))
    arrays, attempts = collect_accounting(repo, freeze)
    root = launcher.output_root(freeze)
    output = root / "screening" / f"candidate-2a-{args.analysis_freeze}.json"
    if output.resolve() != output or output.exists():
        raise ValueError("Use a new independent candidate output file")
    result = audit(root, freeze, arrays, attempts, evidence, criteria, prompts, stage_candidates)
    result["analysis_freeze"] = args.analysis_freeze
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f".{output.name}.partial")
    with partial.open("x") as stream:
        json.dump(result, stream, sort_keys=True, allow_nan=False, indent=2)
        stream.write("\n")
    try:
        output.hardlink_to(partial)
    finally:
        partial.unlink()
    print(output)


if __name__ == "__main__":
    main()
