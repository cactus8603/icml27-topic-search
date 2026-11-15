"""Fill a formal blind review packet using the original generation batches."""

import argparse
import csv
import fcntl
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
import shutil
import sys
import tempfile


def original_batch(seed, chunk, batch_size, sample_limit):
    if not 0 <= seed < sample_limit:
        raise ValueError("Review seed is outside the formal sample")
    chunk_start = seed // chunk * chunk
    chunk_stop = min(chunk_start + chunk, sample_limit)
    start = chunk_start + (seed - chunk_start) // batch_size * batch_size
    return chunk_start, chunk_stop, start, min(start + batch_size, chunk_stop)


def review_plan(run, models, prompts, chunk, batch_size, sample_limit):
    copies, batches = {}, defaultdict(list)
    review = run / "review"
    items_body = (review / "review_items.csv").read_bytes()
    items = list(csv.DictReader(items_body.decode().splitlines()))
    fields = ("item", "model", "prompt", "seed")
    with (review / "review_key.csv").open(newline="") as key_file:
        identity = [tuple(row[field] for field in fields) for row in csv.DictReader(key_file)]
    with (review / "review_blind.csv").open(newline="") as blind_file:
        blind_ids = [row["item"] for row in csv.DictReader(blind_file)]
    if (identity != [tuple(row[field] for field in fields) for row in items]
            or blind_ids != [row["item"] for row in items]):
        raise ValueError("Blind item identifiers differ from the frozen selection")
    for row in items:
        model, prompt, seed = row["model"], row["prompt"], int(row["seed"])
        item = int(row["item"])
        if model not in models or prompt not in prompts:
            raise ValueError("Review item is outside the frozen model and prompt panel")
        chunk_lo, chunk_hi, lo, hi = original_batch(seed, chunk, batch_size, sample_limit)
        sidecar = run / "scores" / model / prompt / f"s{chunk_lo:07d}_{chunk_hi:07d}.json"
        metadata = json.loads(sidecar.read_bytes())
        if (metadata["model"], metadata["prompt_id"], metadata["seed_range"], metadata["batch_size"]) != (
            model, prompt, [chunk_lo, chunk_hi], batch_size
        ):
            raise ValueError("Review item does not match its source chunk")
        source = run / "images" / model / prompt
        thumbnail = next((source / f"{seed:07d}_{kind}.jpg" for kind in ("pos", "near")
                          if (source / f"{seed:07d}_{kind}.jpg").is_file()), None)
        if item in copies:
            raise ValueError("Duplicate blind item identifier")
        copies[item] = thumbnail
        if thumbnail is None:
            batches[model, prompt, lo, hi, sidecar].append((item, seed - lo))
    if not copies or set(copies) != set(range(len(copies))):
        raise ValueError("Blind item identifiers must be contiguous from zero")
    return copies, batches, hashlib.sha256(items_body).hexdigest()


def render(batches, destination, prompts, sampling, load_pipeline):
    import diffusers
    import numpy as np
    from PIL import Image
    import torch

    if not torch.cuda.is_available() or "H200" not in torch.cuda.get_device_name():
        raise ValueError("The original formal batches require an H200")
    height, width = sampling["height"], sampling["width"]
    dtype = getattr(torch, sampling["dtype"])
    save = sampling["save_policy"]
    model_settings = {key: sampling[key] for key in ("scheduler", "eta", "steps", "guidance_scale", "height",
                                                     "width", "negative_prompt")}
    model_settings.update(torch=torch.__version__, diffusers=diffusers.__version__, dtype=str(dtype))
    model_now, prompt_now = None, None
    with torch.no_grad():
        for (model, prompt, lo, hi, sidecar), targets in sorted(batches.items()):
            metadata = json.loads(sidecar.read_bytes())
            if model != model_now:
                if model_now is not None:
                    del pipe, positive, negative
                pipe, info = load_pipeline(model, "cuda", dtype)
                pipe.set_progress_bar_config(disable=True)
                model_now, prompt_now = model, None
            text = prompts[prompt]["text"]
            expected = {**model_settings, "load_info": info, "prompt": text}
            if any(metadata[key] != value for key, value in expected.items()):
                raise ValueError("Review replay differs from the source sampling environment")
            if prompt != prompt_now:
                positive, negative = pipe.encode_prompt(text, "cuda", 1, True,
                                                       negative_prompt=sampling["negative_prompt"])
                prompt_now = prompt
            seeds = range(lo, hi)
            latents = torch.stack([torch.randn((4, height // 8, width // 8),
                                              generator=torch.Generator("cpu").manual_seed(seed))
                                   for seed in seeds]).to("cuda", dtype)
            count = len(seeds)
            images = pipe(prompt_embeds=positive.expand(count, -1, -1),
                          negative_prompt_embeds=negative.expand(count, -1, -1), latents=latents,
                          num_inference_steps=sampling["steps"], guidance_scale=sampling["guidance_scale"],
                          eta=sampling["eta"], height=height, width=width, output_type="np").images
            pixels = np.clip(np.rint(images * 255), 0, 255).astype(np.uint8)
            for item, position in targets:
                Image.fromarray(pixels[position]).resize((save["thumb_size"],) * 2, Image.LANCZOS).save(
                    destination / f"{item:04d}.jpg", quality=save["jpeg_quality"])


def write_packet(review, copies, batches, item_sha, score_sha, upstream, sampling, prompts, load_pipeline):
    output = review / "exact_packet"
    if output.exists() or (review / "review_imgs").exists() or (review / "index.html").exists():
        raise ValueError("An existing review packet must be preserved")
    with tempfile.TemporaryDirectory(prefix=".exact-replay-", dir=review) as staging:
        staged = Path(staging) / output.name
        images = staged / "review_imgs"
        images.mkdir(parents=True)
        for item, source in copies.items():
            if source is not None:
                shutil.copyfile(source, images / f"{item:04d}.jpg")
        render(batches, images, prompts, sampling, load_pipeline)
        if hashlib.sha256((review / "review_items.csv").read_bytes()).hexdigest() != item_sha:
            raise ValueError("The blind item list changed during replay")
        workbook = staged / "index.html"
        panels = "\n".join(f'<figure><img src="review_imgs/{item:04d}.jpg" width="256" alt="{item}">'
                           f'<figcaption>{item}</figcaption></figure>' for item in sorted(copies))
        workbook.write_text('<meta charset="utf-8"><title>文生圖盲標</title>'
                            '<style>figure{display:inline-block;margin:4px}</style>'
                            '<p>依評分準則檢視圖片，將標籤填入 review_blind.csv。</p>\n' + panels,
                            encoding="utf-8")
        receipt = staged / "exact_replay.json"
        receipt.write_text(json.dumps({
            "review_items_sha256": item_sha,
            "score_sha256": score_sha,
            "job": os.environ.get("SLURM_JOB_ID"),
            "sampling_source": upstream,
            "original_thumbnails": sum(source is not None for source in copies.values()),
            "replayed_items": sum(map(len, batches.values())),
            "replayed_batches": len(batches),
            "generated_images": sum(hi - lo for _, _, lo, hi, _ in batches),
        }, indent=2) + "\n")
        staged.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--source", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(args.upstream / "scripts"))
    from common import load_json, load_pipeline, prompts_by_id
    from pipeline import BATCH, CHUNK
    from run_candidate import verify_loader_refs
    from stage1_check import plan

    verify_loader_refs(args.upstream, plan(args.upstream))
    if args.check_only:
        print("Frozen sampling assets verified. No formal scores or review items opened.")
        return

    run = args.run.resolve(strict=True)
    review = run / "review"
    criteria = load_json("criteria.json")["screen"]
    for key in ("stage1_n", "stage2a_n", "stage2b_n", "shard"):
        if criteria[key] % CHUNK:
            raise ValueError("Formal shard boundaries do not align with the source chunks")
    for config in ("sampling.json", "models.json", "prompts.json"):
        if (run / "configs" / config).read_bytes() != (args.upstream / "configs" / config).read_bytes():
            raise ValueError("Replay configuration differs from the formal sample")
    models, prompts = load_json("models.json")["models"], prompts_by_id()
    with (run / "operations/screen-review.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        formal = json.loads((run / "three_items.json").read_bytes())
        if not formal.get("final_coverage"):
            raise ValueError("Review requires the validated final sample")
        if not (review / "review_blind.csv").is_file():
            raise ValueError("Run the frozen blind selection before replay")
        copies, batches, item_sha = review_plan(run, models, prompts, CHUNK, BATCH, criteria["stage2b_n"])
        print(f"review_items={len(copies)} replayed_batches={len(batches)} "
              f"generated_images={sum(hi - lo for _, _, lo, hi, _ in batches)} items_sha256={item_sha}")
        if not args.plan:
            write_packet(review, copies, batches, item_sha, formal["score_sha256"], args.source,
                         load_json("sampling.json"), prompts, load_pipeline)


if __name__ == "__main__":
    main()
