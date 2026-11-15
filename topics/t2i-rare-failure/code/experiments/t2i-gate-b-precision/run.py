"""Evaluate the prospective CLIP precision repeat with aggregate-only output."""
import argparse
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import sys

import numpy as np


SOURCE_COMMIT = "37e6ae9a63cd3035822dccca2acf90ef12efec93"
SOURCE_HASHES = {
    "configs/criteria.json": "eea9d6cd0c431087f03263cfa92886e76240d59bbef046cdd88d28d7b2e5e735",
    "configs/detectors.json": "c1bf8955fb18cc77ee72d010a878990c8470391d14fa580fcc983dd3f29caadb",
    "configs/sampling.json": "26ae5942ebb8767091af29d6b01642cdbfb52ababab0541863bd9b498b6a8e5a",
    "configs/models.json": "0c9b37883621433a8c7d4d9200a0b9115f4394a5c44d7fde3284b7ac4ff5e3bf",
    "configs/prompts.json": "29c79a7141be3b21c4c6274bcbc10f128f54d5098f36cff7152dd2251a75a31d",
    "scripts/clip_precision_probe.py": "9a0db6d80589a787776e4c7b3d74d523b3c15d2f0b1f175628a3dc83088c5ab6",
    "scripts/common.py": "9b4431f1ee918faa65b7c83f6f1ffaa85b1e8046c5d0a406af7c6ea9b5d25f04",
    "scripts/detectors.py": "99d700586c37840e7e175232aae0a6d60b1e19c99371928fc65a8c0658ecb84b",
    "scripts/evidence.py": "8c1f0b691d4f400610604faf81da8648c508db8f8594364ef819089728dc3d0d",
}
RUNTIME = {"torch": "2.5.1", "diffusers": "0.31.0", "transformers": "4.46.3"}
SEEDS = range(10_000, 20_000)
ARM = "fp16_features_fp32_logits"


def file_sha(path):
    return sha256(path.read_bytes()).hexdigest()


def validate_source(path):
    for relative, wanted in SOURCE_HASHES.items():
        if file_sha(path / relative) != wanted:
            raise ValueError(f"Archived source differs from the frozen reference: {relative}")


def frozen_inputs(path):
    validate_source(path)
    runtime = {name: version(name) for name in RUNTIME}
    if runtime != RUNTIME:
        raise ValueError(f"Installed package versions differ from the original diagnostic: {runtime}")
    sys.path.insert(0, str(path / "scripts"))
    import clip_precision_probe as reference

    gate = reference.load_json("criteria.json")["gate_b"]
    if gate["prompts"] != list(reference.PROMPTS) or gate["n_per_prompt"] != len(SEEDS):
        raise ValueError("Frozen named prompts or required seed count differ")
    if set(SEEDS) & set(reference.SEEDS):
        raise ValueError("Prospective and diagnostic seed ranges overlap")
    return reference, gate, runtime


def expected_batches(prompts, seeds, batch_size):
    for prompt in prompts:
        for start in range(seeds.start, seeds.stop, batch_size):
            yield prompt, range(start, min(start + batch_size, seeds.stop))


def collect_scores(received, gate, seeds, batch_size):
    """Validate complete, ordered, non-repeated coverage before judging any prompt."""
    cells = {prompt: [] for prompt in gate["prompts"]}
    expected = expected_batches(gate["prompts"], seeds, batch_size)
    for (prompt, seed_batch), (actual_prompt, actual_seeds, scores) in zip(expected, received, strict=True):
        if prompt != actual_prompt or seed_batch != actual_seeds:
            raise ValueError("Generation batch differs from the registered prompt and seed range")
        if (scores.shape != (len(seed_batch),) or scores.dtype != np.float32
                or not np.isfinite(scores).all() or not ((0 <= scores) & (scores <= 1)).all()):
            raise ValueError("Invalid FP32 probability batch")
        cells[prompt].append(scores)
    return {prompt: np.concatenate(rows) for prompt, rows in cells.items()}


def evaluate(received, gate, seeds, batch_size):
    if gate["n_per_prompt"] != len(seeds):
        raise ValueError("Seed count differs from the original gate")
    scores = collect_scores(received, gate, seeds, batch_size)
    rows = []
    for prompt, values in scores.items():
        distinct, frequencies = np.unique(values, return_counts=True)
        hits = {str(t): int((values >= np.float32(t)).sum()) for t in gate["thresholds"]}
        qualified = [t for t in gate["thresholds"]
                     if len(distinct) >= gate["min_distinct_scores"]
                     and hits[str(t)] >= gate["min_hits"] and hits[str(t)] / len(values) <= gate["max_rate"]]
        rows.append(dict(prompt=prompt, n=len(values), distinct=len(distinct), largest_tie=int(frequencies.max()),
                         hits=hits, qualified=qualified))
    return {"status": "PASS" if any(row["qualified"] for row in rows) else "FAIL", "rows": rows}


def generate(reference, gate, snapshots):
    import torch
    from detectors import Detectors

    if not torch.cuda.is_available() or "H200" not in torch.cuda.get_device_name(0):
        raise RuntimeError("A Slurm H200 GPU is required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sampling = reference.load_json("sampling.json")
    height, width = sampling["height"], sampling["width"]
    prompts = reference.prompts_by_id()
    pipe, _ = reference.load_pipeline(gate["model"], "cuda", torch.float16, base_snapshot=snapshots[0])
    pipe.set_progress_bar_config(disable=True)
    detector = Detectors("cuda", dtype=torch.float16, which=(gate["detector"],), clip_snapshot=snapshots[1])
    active_prompt = None
    with torch.inference_mode():
        scale = detector.clip.logit_scale.exp()
        for prompt, seeds in expected_batches(gate["prompts"], SEEDS, reference.BATCH):
            if prompt != active_prompt:
                print(f"Starting {prompt}", flush=True)
                text, negative = pipe.encode_prompt(prompts[prompt]["text"], "cuda", 1, True,
                                                    negative_prompt=sampling["negative_prompt"])
                active_prompt = prompt
            latents = torch.stack([
                torch.randn((4, height // 8, width // 8), generator=torch.Generator("cpu").manual_seed(seed))
                for seed in seeds
            ]).to("cuda", torch.float16)
            n = len(seeds)
            pixels = pipe(
                prompt_embeds=text.expand(n, -1, -1), negative_prompt_embeds=negative.expand(n, -1, -1),
                latents=latents, num_inference_steps=sampling["steps"], guidance_scale=sampling["guidance_scale"],
                eta=sampling["eta"], height=height, width=width, output_type="np",
            ).images
            images = np.clip(np.rint(pixels * 255), 0, 255).astype(np.uint8)
            features = reference.image_features(detector, images)
            scores = reference.probabilities(features, detector.tf, scale, accumulate_fp32=True)
            yield prompt, seeds, scores.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True, type=Path)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--out", type=Path)
    parser.add_argument("--freeze")
    args = parser.parse_args()
    reference, gate, runtime = frozen_inputs(args.upstream.resolve())
    provenance = {"scope": "prospective_clip_gate_b_precision_repeat", "reference_commit": SOURCE_COMMIT,
                  "reference_sha256": SOURCE_HASHES, "arm": ARM, "sd_revision": reference.SD_REV,
                  "clip_revision": reference.CLIP_REV, "versions": {**runtime, "numpy": np.__version__},
                  "seed_range": [SEEDS.start, SEEDS.stop], "n_per_prompt": gate["n_per_prompt"],
                  "batch_size": reference.BATCH, "prompts": gate["prompts"]}
    if args.preflight:
        reference.check_assets()
        print(json.dumps(provenance, indent=2))
        return
    job_id = os.environ.get("SLURM_JOB_ID", "")
    if (not re.fullmatch(r"[0-9a-f]{40}", args.freeze or "") or not job_id.isdecimal()
            or args.out.name != f"{job_id}.json" or args.out.exists()):
        parser.error("A full frozen commit and a fresh Slurm job output path are required")
    from evidence import atomic_json

    root = Path(__file__).resolve().parent
    provenance.update(frozen_commit=args.freeze, slurm_job_id=job_id,
                      experiment_sha256={name: file_sha(root / name) for name in ("run.py", "PREREG.md", "run.sbatch")})
    atomic_json(args.out, {**provenance, "status": "UNTESTABLE", "reason": "Complete coverage pending"})
    snapshots = reference.check_assets()
    outcome = evaluate(generate(reference, gate, snapshots), gate, SEEDS, reference.BATCH)
    atomic_json(args.out, {**provenance, **outcome})


if __name__ == "__main__":
    main()
