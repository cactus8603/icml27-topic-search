"""Replay the registered Gate (b) seeds with the validated CLIP arithmetic repair."""

import argparse
from importlib.util import module_from_spec, spec_from_file_location
import json
import os
from pathlib import Path
import re
import runpy
import sys
from unittest.mock import patch

import numpy as np


GENERATOR_COMMIT = "f54cb310a921966440754d42f65f86d9f565dc62"
GENERATOR_SHA = "e849b0a81dcdb2f95f3bda490f640d42378fc0fe329297470a0eba99e42fc76b"
PRECISION_COMMIT = "2b1c7f64b8b9013a9a32f8353024c1d6f82b6e53"
PRECISION_SHA = "d9271787d542729abbd919be5db8114b1cbcb442069b446a7f8b217ff7752af5"
VERIFIER_SHA = "37fea9639ad87ecca9a6c0691360f8eaca7bd50c4b61674078bd910fa78de771"
SHARED = Path("/work/ChihChungHsu/cactus8603/icml")
OUTPUT = Path("runs/t2irare/gate-b-original-replay")
SCOPE = "original_gate_b_seeds_clip_precision_repair"
SEED_COUNT = 10_000
CHUNK = 500
BATCH = 32


def load_inputs(upstream, original, precision_path):
    spec = spec_from_file_location("frozen_gate_b_precision", precision_path)
    precision = module_from_spec(spec)
    spec.loader.exec_module(precision)
    for file, expected in ((precision_path, PRECISION_SHA), (upstream / "scripts/generate.py", GENERATOR_SHA),
                           (upstream / "scripts/verify_gate_b.py", VERIFIER_SHA)):
        if precision.file_sha(file) != expected:
            raise ValueError(f"Frozen source differs: {file.name}")
    reference, gate, runtime = precision.frozen_inputs(upstream)
    if (gate["model"], gate["detector"], gate["n_per_prompt"], reference.BATCH) != ("sd14", "clip", SEED_COUNT, BATCH):
        raise ValueError("Gate (b) model, detector, seed count, or batch changed")

    import verify_gate_b

    # Only prerequisite data are read from the historical workspace. Config loaders
    # still resolve to their frozen archive through common.ROOT.
    with patch.object(verify_gate_b, "ROOT", original):
        prerequisites = verify_gate_b.prerequisites()
    source = {
        "generator_commit": GENERATOR_COMMIT, "generator_sha256": GENERATOR_SHA,
        "precision_replay_commit": PRECISION_COMMIT, "precision_replay_sha256": PRECISION_SHA,
        "reference_commit": precision.SOURCE_COMMIT, "reference_sha256": precision.SOURCE_HASHES,
        "verifier_sha256": VERIFIER_SHA,
    }
    provenance = {
        "scope": SCOPE, "source": source, "precision": precision.ARM,
        "sd_revision": reference.SD_REV, "clip_revision": reference.CLIP_REV,
        "model": gate["model"], "detector": gate["detector"], "prompts": gate["prompts"],
        "seed_range": [0, SEED_COUNT], "batch_size": BATCH, "chunk_size": CHUNK,
        "versions": {**runtime, "numpy": np.__version__}, "prerequisites": prerequisites,
        "stage1_activated": False,
    }
    return reference, verify_gate_b, provenance


def check_assets(reference):
    from common import base_path

    base, clip = reference.check_assets()
    if Path(base_path()).resolve(strict=True) != Path(base).resolve(strict=True):
        raise ValueError("The original generator's default SD snapshot differs from the pinned revision")
    return clip


def stamp(provenance, freeze):
    return {"replay_freeze": freeze, "clip_precision": provenance["precision"],
            "sd_revision": provenance["sd_revision"], "clip_revision": provenance["clip_revision"],
            "generator_commit": GENERATOR_COMMIT, "generator_sha256": GENERATOR_SHA}


def generate(upstream, reference, provenance, clip_snapshot, output, freeze):
    import detectors
    import torch

    if not torch.cuda.is_available() or "H200" not in torch.cuda.get_device_name(0):
        raise RuntimeError("The original Gate (b) replay requires a Slurm H200 allocation")

    class RepairedClip(detectors.Detectors):
        def __init__(self, device, dtype, which):
            if (device, dtype, tuple(which)) != ("cuda", torch.float16, ("clip",)):
                raise ValueError("Replay accepts only the registered FP16 CLIP-only path")
            super().__init__(device, dtype, which, clip_snapshot=clip_snapshot)
            self.versions = stamp(provenance, freeze)

        def score(self, images):
            with torch.no_grad():
                features = reference.image_features(self, images)
                scale = self.clip.logit_scale.exp()
                original_tf32 = torch.backends.cuda.matmul.allow_tf32
                try:
                    torch.backends.cuda.matmul.allow_tf32 = False
                    clip = reference.probabilities(features, self.tf, scale, accumulate_fp32=True).cpu().numpy()
                finally:
                    torch.backends.cuda.matmul.allow_tf32 = original_tf32
            nan = np.full(len(images), np.nan, np.float32)
            return {"clip": clip, "nudenet": nan, "falconsai": nan}

    script = upstream / "scripts/generate.py"
    args = [str(script), "--model", provenance["model"], "--prompts", *provenance["prompts"],
            "--seed-start", "0", "--seed-end", str(SEED_COUNT), "--batch-size", str(BATCH), "--chunk", str(CHUNK),
            "--detectors", "clip", "--no-save-images", "--out", str(output)]
    with patch.object(detectors, "Detectors", RepairedClip), patch.object(sys, "argv", args):
        runpy.run_path(str(script), run_name="__main__")


def validate_chunk(path, lo, hi, metadata, versions):
    sidecar = path.with_suffix(".json")
    if any(not file.is_file() or file.is_symlink() or file.resolve() != file for file in (path, sidecar)):
        raise ValueError(f"Missing or aliased registered chunk: {path}")
    meta = json.loads(sidecar.read_text())
    if (any(meta.get(name) != value for name, value in metadata.items())
            or meta.get("batch_size") != BATCH or meta.get("images_saved") != 0
            or meta.get("detectors") != ["clip"]):
        raise ValueError(f"Replay metadata differs: {path}")
    for name in ("torch", "diffusers", "transformers"):
        value = meta.get(name)
        if not isinstance(value, str) or value.split("+", 1)[0] != versions[name]:
            raise ValueError(f"Runtime version differs for {name}: {path}")
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {"seeds", "clip", "nudenet", "falconsai"}:
            raise ValueError(f"Score columns differ: {path}")
        seeds = data["seeds"]
        if seeds.dtype != np.int64 or not np.array_equal(seeds, np.arange(lo, hi)):
            raise ValueError(f"Ordered seed range differs: {path}")
        clip = data["clip"]
        if clip.dtype != np.float32 or clip.shape != seeds.shape or not np.isfinite(clip).all() or np.any((clip < 0) | (clip > 1)):
            raise ValueError(f"Invalid FP32 CLIP scores: {path}")
        for name in ("nudenet", "falconsai"):
            if data[name].dtype != np.float32 or data[name].shape != seeds.shape or not np.isnan(data[name]).all():
                raise ValueError(f"Disabled detector column is not an empty placeholder: {path}")


def verify(output, verifier, provenance, freeze, job):
    metadata = {**stamp(provenance, freeze), "slurm_job": job}
    score_root = output / "scores"
    expected = {}
    for prompt in provenance["prompts"]:
        for lo in range(0, SEED_COUNT, CHUNK):
            hi = min(lo + CHUNK, SEED_COUNT)
            path = score_root / provenance["model"] / prompt / f"s{lo:07d}_{hi:07d}.npz"
            expected[path] = lo, hi
    observed = {file for file in score_root.rglob("*") if not file.is_dir() or file.is_symlink()}
    images = output / "images"
    if observed != set(expected) | {path.with_suffix(".json") for path in expected} or images.exists() or images.is_symlink():
        raise ValueError("Replay must contain exactly the registered score chunks and no images")
    for path, (lo, hi) in expected.items():
        validate_chunk(path, lo, hi, metadata, provenance["versions"])
    # The original verifier checks config fields and all score thresholds, and
    # hashes every NPZ and sidecar. The candidate receipt uses a separate name.
    result = verifier.evaluate(output, job)
    return {**provenance, "frozen_commit": freeze, "score_chunk_count": len(expected), **result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--precision", type=Path, required=True)
    parser.add_argument("--freeze", required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--run", action="store_true")
    modes.add_argument("--verify-job", metavar="JOB_ID")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.freeze):
        parser.error("Supply a full freeze commit")
    upstream = args.upstream.resolve(strict=True)
    reference, verifier, provenance = load_inputs(upstream, args.original.resolve(strict=True), args.precision)
    if args.preflight:
        check_assets(reference)
        print("[codex] Gate (a), bench, original replay source and both pinned snapshots verified")
        return

    job = os.environ.get("SLURM_JOB_ID", "") if args.run else args.verify_job
    if not re.fullmatch(r"[0-9]+", job):
        parser.error("Supply a Slurm job identity")
    output = SHARED.resolve(strict=True) / OUTPUT / job
    if output.resolve() != output:
        raise ValueError("Replay output must not alias another run")
    receipt = output / "precision_replay_verified.json"
    if args.run:
        clip_snapshot = check_assets(reference)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.mkdir()
        generate(upstream, reference, provenance, clip_snapshot, output, args.freeze)
        from evidence import atomic_json

        atomic_json(receipt, verify(output, verifier, provenance, args.freeze, job))
        print(f"[codex] Complete candidate receipt saved to {receipt}")
    else:
        if not receipt.is_file() or receipt.is_symlink() or json.loads(receipt.read_text()) != verify(
                output, verifier, provenance, args.freeze, job):
            raise ValueError("Candidate receipt does not match the replay evidence")
        print(f"[codex] Candidate receipt and replay files verified for {job}")


if __name__ == "__main__":
    main()
