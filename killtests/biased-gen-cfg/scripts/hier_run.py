"""Single-attempt output and completion evidence for hierarchical training."""
import hashlib
import json
import os
from pathlib import Path


def start_run(output):
    path = Path(output)
    path.mkdir(parents=True, exist_ok=False)


def complete_run(output, data, epochs):
    path = Path(output)
    if any(not (path / f"ep{epoch}.pt").is_file() for epoch in epochs):
        raise RuntimeError("cannot finalize a run with missing checkpoints")
    with open(data, "rb") as stream:
        data_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    record = {
        "version": 1,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "data_sha256": data_sha,
        "meta_sha256": hashlib.sha256((path / "meta.json").read_bytes()).hexdigest(),
        "epochs": epochs,
    }
    pending = path / "completion.json.pending"
    pending.write_text(json.dumps(record, indent=2) + "\n")
    pending.replace(path / "completion.json")
