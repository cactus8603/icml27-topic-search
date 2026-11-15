"""Paired CLIP precision diagnostic. Write only aggregate results."""

import argparse
import importlib.metadata
import os
from pathlib import Path

import numpy as np

from common import load_json, load_pipeline, prompts_by_id
from evidence import atomic_json, config_sha


PROMPTS = ("E1", "N1", "B1")
SEEDS = range(2000, 2500)
BATCH = 32
SD_REV = "133a221b8aa7292a167afc5127cb63fb5005638b"
CLIP_REV = "32bd64288804d66eefd0ccbe215aa642df71cc41"


def check_assets():
    from huggingface_hub import snapshot_download

    diffusion = Path(snapshot_download(load_json("models.json")["base_repo"], revision=SD_REV,
                                      local_files_only=True))
    clip = Path(snapshot_download(load_json("detectors.json")["clip_zeroshot"]["repo"],
                                 revision=CLIP_REV, local_files_only=True))
    if (diffusion.name, clip.name) != (SD_REV, CLIP_REV):
        raise ValueError("Offline model snapshots differ from the frozen revisions")
    sd_files = ("model_index.json", "text_encoder/model.safetensors", "text_encoder/config.json",
                "unet/diffusion_pytorch_model.safetensors", "unet/config.json",
                "vae/diffusion_pytorch_model.safetensors", "vae/config.json",
                "scheduler/scheduler_config.json", "tokenizer/tokenizer_config.json",
                "tokenizer/merges.txt", "tokenizer/vocab.json")
    clip_files = ("config.json", "model.safetensors", "merges.txt", "vocab.json")
    for folder, names in ((diffusion, sd_files), (clip, clip_files)):
        for name in names:
            if not (folder / name).is_file():
                raise ValueError(f"Required offline model file is missing: {folder / name}")
    return str(diffusion), str(clip)


def probabilities(features, text, scale, accumulate_fp32=False):
    if accumulate_fp32:
        features, text, scale = features.float(), text.float(), scale.float()
    return ((scale * features) @ text.T).float().softmax(-1)[:, 0]


def image_features(detector, images):
    torch = detector.torch
    image = torch.from_numpy(images).to(detector.device).permute(0, 3, 1, 2).float() / 255.0
    image = torch.nn.functional.interpolate(
        image, size=(224, 224), mode="bicubic", antialias=True, align_corners=False
    ).clamp(0, 1)
    image = ((image - detector.cm) / detector.cs).to(detector.dtype)
    features = detector.clip.get_image_features(pixel_values=image)
    return features / features.norm(dim=-1, keepdim=True)


def summarize(original, accumulation, full_precision, thresholds):
    arrays = {
        "fp16": np.asarray(original),
        "fp16_features_fp32_logits": np.asarray(accumulation),
        "fp32": np.asarray(full_precision),
    }
    if arrays["fp16"].ndim != 1 or not len(arrays["fp16"]):
        raise ValueError("Expected nonempty paired score vectors")
    baseline = arrays["fp16"]
    output = {}
    for arm, scores in arrays.items():
        if scores.shape != baseline.shape or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
            raise ValueError(f"Invalid scores for {arm}")
        unique, counts = np.unique(scores, return_counts=True)
        output[arm] = {
            "distinct": len(unique),
            "largest_tie": int(counts.max()),
            "absolute_delta_p95": float(np.quantile(np.abs(scores - baseline), 0.95)),
            "hits": {str(t): int(np.count_nonzero(scores >= t)) for t in thresholds},
            "flips_vs_fp16": {
                str(t): int(np.count_nonzero((baseline >= t) != (scores >= t))) for t in thresholds
            },
        }
    return output


def run(args, diffusion_snapshot, clip_snapshot):
    import torch
    from detectors import Detectors

    if args.out.exists():
        raise FileExistsError("Use a new output filename")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sampling = load_json("sampling.json")
    thresholds = load_json("criteria.json")["gate_b"]["thresholds"]
    height, width = sampling["height"], sampling["width"]
    prompts = prompts_by_id()
    pipe, _ = load_pipeline("sd14", "cuda", torch.float16, base_snapshot=diffusion_snapshot)
    pipe.set_progress_bar_config(disable=True)
    half = Detectors("cuda", dtype=torch.float16, which=("clip",), clip_snapshot=clip_snapshot)
    full = Detectors("cuda", dtype=torch.float32, which=("clip",), clip_snapshot=clip_snapshot)
    report = {
        "scope": "posthoc_paired_precision_diagnostic",
        "cannot_replace_gate_b": True,
        "frozen_commit": args.freeze,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "config_sha256": config_sha(),
        "sd_revision": SD_REV,
        "clip_revision": CLIP_REV,
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "diffusers", "transformers")},
        "cells": {},
    }
    with torch.inference_mode():
        for prompt in PROMPTS:
            pe, ne = pipe.encode_prompt(prompts[prompt]["text"], "cuda", 1, True,
                                        negative_prompt=sampling["negative_prompt"])
            paired = [[], [], []]
            for start in range(SEEDS.start, SEEDS.stop, BATCH):
                seeds = range(start, min(start + BATCH, SEEDS.stop))
                latents = torch.stack([
                    torch.randn((4, height // 8, width // 8), generator=torch.Generator("cpu").manual_seed(seed))
                    for seed in seeds
                ]).to("cuda", torch.float16)
                n = len(seeds)
                pixels = pipe(
                    prompt_embeds=pe.expand(n, -1, -1), negative_prompt_embeds=ne.expand(n, -1, -1),
                    latents=latents, num_inference_steps=sampling["steps"], guidance_scale=sampling["guidance_scale"],
                    eta=sampling["eta"], height=height, width=width, output_type="np",
                ).images
                images = np.clip(np.rint(pixels * 255), 0, 255).astype(np.uint8)
                fp16_features = image_features(half, images)
                scale16 = half.clip.logit_scale.exp()
                original = probabilities(fp16_features, half.tf, scale16)
                accumulation = probabilities(fp16_features, half.tf, scale16, accumulate_fp32=True)
                full_precision = probabilities(image_features(full, images), full.tf, full.clip.logit_scale.exp())
                for arm, scores in zip(paired, (original, accumulation, full_precision), strict=True):
                    arm.append(scores.cpu().numpy())
            report["cells"][prompt] = summarize(*(np.concatenate(arm) for arm in paired), thresholds=thresholds)
    report["n_per_prompt"] = len(SEEDS)
    report["seed_range"] = [SEEDS.start, SEEDS.stop]
    report["batch_size"] = BATCH
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.out, report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-assets", action="store_true")
    parser.add_argument("--freeze")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    assets = check_assets()
    if not args.check_assets:
        if args.out is None or not args.freeze or not os.environ.get("SLURM_JOB_ID"):
            parser.error("Run via the frozen Slurm launcher")
        run(args, *assets)


if __name__ == "__main__":
    main()
