"""Bind a Chameleon extraction to its inputs and settings."""
import hashlib
from pathlib import Path


def sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def extraction_fingerprint(cfg, data_dir, sets, names, batch_size, shard_rows, limit=None, tiny=False):
    probe_dir = Path(cfg["paths"]["minimal_repo"]) / "probes"
    return {
        "version": 1,
        "checkpoint": cfg["model"]["hf_id"],
        "revision": cfg["model"]["revision"],
        "data_sha256": {name: sha256(Path(data_dir) / (name + ".jsonl")) for name in sets},
        "probe_sha256": {} if tiny else {name: sha256(probe_dir / f"{name}_weights.pt") for name in names},
        "input_config": {key: cfg[key] for key in ("layers_L8", "conds", "templates", "max_prompt_chars", "max_resp_tokens")},
        "evaded_layer": cfg["model"]["evaded_layer"],
        "extraction": "cuda-bfloat16-eager-leftpad-response-mean-v1" if not tiny else "cpu-smoke-v1",
        "batch_size": batch_size,
        "shard_rows": shard_rows,
        "limit": limit,
        "tiny": tiny,
    }
