"""Launch only the fixed Stage 1 panel after a committed user activation."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
STAGE = Path("codex/experiments/t2i-stage1-candidate")
TOPIC = Path("killtests/t2i-rare-failure")
SHARED = Path("/work/ChihChungHsu/cactus8603/icml")
ACTIVATION = "activation.json"
CHUNK = 500
SCOPE = "candidate_stage1_only"

sys.path.insert(0, str(HERE.parent / "t2i-detector-probe"))
from probe import CLIP_REPO, FAL_REPO
from stage1_check import MODELS, PROXY_CANDIDATE, plan

sys.path.insert(0, str(HERE.parent / "t2i-gate-repair"))
import review_evidence as review


def activation(body):
    record = review.parse_json(body)
    key = "user_decision_ref"
    if (not isinstance(record, dict) or set(record) != {"schema", "scope", "adopted_proxy", key}
            or type(record["schema"]) is not int or record["schema"] != 1
            or record["scope"] != SCOPE or record["adopted_proxy"] != PROXY_CANDIDATE
            or not isinstance(record[key], str) or not record[key].strip()):
        raise ValueError("A Stage 1 user decision bound to the exact proxy receipt is required")
    return record


def read_activation(repo, freeze, path):
    if not path.exists():
        return None
    body = path.read_bytes()
    if not freeze:
        raise ValueError("The user decision must be committed in the supplied launch freeze")
    from_git = subprocess.check_output(["git", "-C", str(repo), "show", f"{freeze}:{STAGE / ACTIVATION}"])
    if body != from_git:
        raise ValueError("Archived user decision differs from the launch commit")
    return activation(body)


def candidate_receipt(repo, verify_job):
    review.REPO = repo
    metadata, gate = review.sources_and_gate()
    observed = review.review(review.read_receipt(review.RECEIPT), metadata, gate)
    identity = {"slurm_job_id": observed["job"], "frozen_commit": observed["freeze"],
                "receipt_sha256": observed["receipt_sha256"]}
    if identity != PROXY_CANDIDATE or observed["status"] != "PASS":
        raise ValueError("The reviewed companion proxy receipt did not validate")
    if verify_job:
        accounting = subprocess.check_output(
            [review.SACCT, "-X", "-n", "-P", "-j", review.JOB, "--format=JobIDRaw,State,ExitCode,JobName"], text=True,
        )
        review.verify_job(accounting, (repo / "killtests/SUBMITTED.log").read_text(),
                          (review.RUNS / f"{review.JOB_NAME}-{review.JOB}.out").read_text(), metadata["prompts"])
    return identity


def array_task(env):
    job, array, index = (env.get(name, "") for name in ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID"))
    if not re.fullmatch(r"[0-9]+", job) or not re.fullmatch(r"[0-9]+", array) or not re.fullmatch(r"[0-5]", index):
        raise ValueError("Use an allocated Stage 1 Slurm array task")
    return job, array, MODELS[int(index)]


def output_root(freeze):
    if not re.fullmatch(r"[0-9a-f]{40}", freeze or ""):
        raise ValueError("Use the full activation commit")
    root = SHARED.resolve() / "runs/t2irare/stage1-candidate" / freeze
    if root.resolve() != root:
        raise ValueError("The Stage 1 output root must not alias another run")
    return root


def verify_loader_refs(upstream, panel):
    from huggingface_hub import constants, hf_hub_download, snapshot_download

    cache = Path(constants.HF_HUB_CACHE).expanduser().absolute()
    if cache.resolve() != (SHARED / "hf_home/hub").resolve():
        raise ValueError("The original loader's effective Hugging Face cache differs from the registered cache")
    revisions = panel["generator_revisions"] | {
        FAL_REPO: panel["detectors"]["falconsai_revision"],
        CLIP_REPO: panel["detectors"]["clip_revision"],
    }
    snapshots = {}
    for repo, revision in revisions.items():
        model_cache = cache / f"models--{repo.replace('/', '--')}"
        ref = (model_cache / "refs/main").read_text(encoding="ascii").strip()
        if ref != revision:
            raise ValueError(f"Default Hugging Face ref differs from the registered revision for {repo}")
        expected = model_cache / "snapshots" / revision
        actual = Path(snapshot_download(repo, local_files_only=True)).absolute()
        if actual != expected or not actual.is_dir() or actual.resolve() != expected.parent.resolve() / revision:
            raise ValueError(f"Default Hugging Face snapshot resolves elsewhere for {repo}")
        snapshots[repo] = expected

    # The unchanged generator resolves each replacement UNet with the single-file API.
    models = json.loads((upstream / "configs/models.json").read_bytes())["models"]
    for spec in models.values():
        if spec["kind"] == "unet_pt":
            repo, name = spec["repo"], spec["file"]
            actual = Path(hf_hub_download(repo, name, local_files_only=True)).absolute()
            if actual != snapshots[repo] / name or not actual.is_file():
                raise ValueError(f"Default Hugging Face checkpoint resolves elsewhere for {name}")
    return revisions


def generator_command(upstream, panel, model, output):
    start, stop = panel["seed_range_per_cell"]
    return [sys.executable, "-IB", str(upstream / "scripts/generate.py"), "--model", model,
            "--prompts", "all", "--seed-start", str(start), "--seed-end", str(stop),
            "--batch-size", str(panel["default_batch_size"]), "--chunk", str(CHUNK), "--out", str(output)]


def setup_prerequisites(repo, evidence):
    source = repo / TOPIC
    gaussian = source / "results/gate_a/gate_a.json"
    if not evidence.gate_a_ok(gaussian):
        raise ValueError("Registered Gate (a) prerequisites did not pass")
    bench = evidence.bench_evidence(source / "results_bench", source / "results_bench2")
    if not bench["pass"]:
        raise ValueError("Registered bench prerequisites did not pass")
    return {"gate_a_sha256": evidence.file_sha(gaussian), "bench": bench}


def verify_chunks(output, panel, model, evidence, complete):
    import common

    prompt_ids = panel["prompt_ids"]
    start, stop = panel["seed_range_per_cell"]
    models = common.load_json("models.json")["models"]
    sampling = common.load_json("sampling.json")
    prompts = common.prompts_by_id()
    model_root = output / "scores" / model
    expected = {(model_root / p / f"s{lo:07d}_{min(lo + CHUNK, stop):07d}.npz"): (p, lo, min(lo + CHUNK, stop))
                for p in prompt_ids for lo in range(start, stop, CHUNK)}
    observed = set(model_root.glob("*/*.npz")) | {p.with_suffix(".npz") for p in model_root.glob("*/*.json")}
    if observed - set(expected) or (complete and observed != set(expected)):
        raise ValueError("Stage 1 chunks do not match the fixed prompt/seed coverage")
    for path in sorted(observed):
        prompt, lo, hi = expected[path]
        sidecar = path.with_suffix(".json")
        if (not path.is_file() or not sidecar.is_file() or path.resolve() != path or sidecar.resolve() != sidecar):
            raise ValueError("Incomplete or aliased Stage 1 chunk. Preserve it for diagnosis before retrying")
        meta = evidence.validate_chunk(path, model, prompt, lo, hi, evidence.DETECTORS, sampling, prompts, models, None)
        if meta["batch_size"] != panel["default_batch_size"]:
            raise ValueError("Stage 1 batch size differs from the frozen plan")
        with np.load(path, allow_pickle=False) as data:
            if (set(data.files) != {"seeds", *evidence.DETECTORS} or data["seeds"].dtype != np.int64
                    or not np.array_equal(data["seeds"], np.arange(lo, hi))):
                raise ValueError("Stage 1 stored seed identities differ")
            for detector in evidence.DETECTORS:
                score = data[detector]
                if (score.shape != (hi - lo,) or score.dtype != np.float32
                        or not np.isfinite(score).all() or np.any((score < 0) | (score > 1))):
                    raise ValueError(f"Invalid Stage 1 score vector for {detector}")
    return len(observed)


def execute(repo, upstream, panel, freeze, record):
    if record is None:
        raise ValueError("Stage 1 is disabled until the user adopts the exact proxy repair")
    job, array, model = array_task(os.environ)
    import torch

    if not torch.cuda.is_available() or "H200" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Candidate Stage 1 requires an allocated H200")
    sys.path.insert(0, str(upstream / "scripts"))
    import evidence

    prerequisites = setup_prerequisites(repo, evidence)
    output = output_root(freeze)
    lock = output / "operations" / f"{model}.lock"
    model_dirs = (output / name / model for name in ("images", "scores"))
    for path in (lock, *(directory / prompt for directory in model_dirs for prompt in panel["prompt_ids"])):
        if path.resolve() != path:
            raise ValueError("A Stage 1 model path aliases another output")
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a") as handler:
        fcntl.flock(handler, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify_chunks(output, panel, model, evidence, complete=False)
        revisions = verify_loader_refs(upstream, panel)
        subprocess.run(generator_command(upstream, panel, model, output), check=True)
        count = verify_chunks(output, panel, model, evidence, complete=True)
        evidence.atomic_json(output / "operations" / f"{model}-{job}.json", {
            "scope": SCOPE, "launch_commit": freeze, "slurm_job_id": job, "slurm_array_job_id": array,
            "model": model, "score_chunks": count, "config_sha256": evidence.config_sha(),
            "proxy_receipt": PROXY_CANDIDATE, "prerequisites": prerequisites, "resolved_hf_revisions": revisions,
        })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=HERE.parents[2])
    parser.add_argument("--upstream", type=Path)
    parser.add_argument("--freeze")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.freeze and not re.fullmatch(r"[0-9a-f]{40}", args.freeze):
        parser.error("Use the full launch commit")
    repo = args.repo.resolve()
    upstream = (args.upstream or repo / TOPIC).resolve()
    record = read_activation(repo, args.freeze, HERE / ACTIVATION)
    if args.execute and record is None:
        parser.error("Candidate Stage 1 is disabled pending the user's explicit adoption of proxy 26516")
    if args.execute:
        executable = subprocess.check_output(["git", "-C", str(repo), "show", f"{args.freeze}:{STAGE / 'run.py'}"])
        if executable != Path(__file__).read_bytes():
            raise ValueError("Running Stage 1 source differs from the launch commit")
    panel = plan(upstream)
    revisions = verify_loader_refs(upstream, panel)
    candidate = candidate_receipt(repo, verify_job=args.execute)
    if args.check_only:
        print(json.dumps({
            "scope": SCOPE, "activation_record_present": record is not None, "proxy_receipt": candidate,
            "model_array": panel["model_array"], "prompt_ids": panel["prompt_ids"],
            "seed_range_per_cell": panel["seed_range_per_cell"], "panel_rows": panel["images"],
            "batch_size": panel["default_batch_size"], "chunk_size": CHUNK,
            "output_root": str(output_root(args.freeze)) if args.freeze else "icml/runs/t2irare/stage1-candidate/<freeze>",
            "source_commit": panel["source_commit"], "source_sha256": panel["source_sha256"],
            "resolved_hf_revisions": revisions,
        }, sort_keys=True))
        return
    execute(repo, upstream, panel, args.freeze, record)


if __name__ == "__main__":
    main()
