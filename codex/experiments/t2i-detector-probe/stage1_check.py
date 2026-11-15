"""Verify the archived Stage 1 plan and offline assets without reading scores."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe import HERE, REFERENCE, TOPIC, file_sha, preflight, snapshot


STAGE1_HASHES = {
    "configs/criteria.json": "eea9d6cd0c431087f03263cfa92886e76240d59bbef046cdd88d28d7b2e5e735",
    "configs/models.json": "0c9b37883621433a8c7d4d9200a0b9115f4394a5c44d7fde3284b7ac4ff5e3bf",
    "configs/prompts.json": "29c79a7141be3b21c4c6274bcbc10f128f54d5098f36cff7152dd2251a75a31d",
    "scripts/generate.py": "a4485c0ffe3ccfa4b9bc6e7d27800e822f85032dfa72fdd023b0f49add3fd040",
    "scripts/verify_gate_b.py": "37fea9639ad87ecca9a6c0691360f8eaca7bd50c4b61674078bd910fa78de771",
    "slurm/stage1.sbatch": "d9fe2f280e512074a0bdbf25602749e3cfaee50999d2b16bb362e01f8f4a4869",
}
MODELS = ("sd14", "uce", "esd", "salun", "rece", "advunlearn")
BASE_FILES = (
    "model_index.json", "text_encoder/model.safetensors", "text_encoder/config.json",
    "unet/diffusion_pytorch_model.safetensors", "unet/config.json", "vae/diffusion_pytorch_model.safetensors",
    "vae/config.json", "scheduler/scheduler_config.json", "tokenizer/tokenizer_config.json",
    "tokenizer/merges.txt", "tokenizer/vocab.json",
)
REVISIONS = {
    "CompVis/stable-diffusion-v1-4": "133a221b8aa7292a167afc5127cb63fb5005638b",
    "siyich/ijd-unlearned-checkpoints": "f53b1b10cccf51b5dc7a64f3433163c7289dc25e",
    "OPTML-Group/AdvUnlearn": "bca3b27922b1e70cfe5b858e35beccc2fbc3b5ee",
}
PROXY_CANDIDATE = {
    "slurm_job_id": "26516", "frozen_commit": "2b1c7f64b8b9013a9a32f8353024c1d6f82b6e53",
    "receipt_sha256": "e51ebd48167cbf9d7805c54107ab6b64b4d452e7606e642d2d153a317ed56f26",
}


def settings(path):
    return json.loads(path.read_bytes())


def plan(upstream):
    for name, digest in STAGE1_HASHES.items():
        if file_sha(upstream / name) != digest:
            raise ValueError(f"Registered Stage 1 source differs: {name}")
    _, detector_source = preflight(upstream)
    models = settings(upstream / "configs/models.json")
    prompts = settings(upstream / "configs/prompts.json")["prompts"]
    criteria = settings(upstream / "configs/criteria.json")
    sampling = settings(upstream / "configs/sampling.json")
    if tuple(models["models"]) != MODELS or len(prompts) != 14 or len({p["id"] for p in prompts}) != 14:
        raise ValueError("Registered Stage 1 panel differs")
    registered_seeds = criteria["screen"]["stage1_n"]
    if registered_seeds != 2000:
        raise ValueError("Registered seed range differs")

    base = models["base_repo"]
    snapshot(base, REVISIONS[base], BASE_FILES)
    for spec in models["models"].values():
        if spec["kind"] == "unet_pt":
            snapshot(spec["repo"], REVISIONS[spec["repo"]], (spec["file"],))
        elif spec["kind"] == "text_encoder_bin":
            snapshot(spec["repo"], REVISIONS[spec["repo"]], (f"{spec['subfolder']}/pytorch_model.bin",))

    return {
        "scope": "stage1_offline_plan", "source_commit": REFERENCE, "source_sha256": STAGE1_HASHES,
        "detectors": detector_source, "generator_revisions": REVISIONS, "model_array": MODELS,
        "prompt_ids": [p["id"] for p in prompts], "seed_range_per_cell": [0, registered_seeds],
        "images": len(MODELS) * len(prompts) * registered_seeds, "default_batch_size": 32,
        "sampling": {name: sampling[name] for name in ("scheduler", "steps", "eta", "guidance_scale",
                                                       "height", "width", "dtype")},
        "registered_launcher": str(TOPIC / "slurm/stage1.sbatch"),
        "launch_authorization": "pending_explicit_proxy_activation_in_formal_gate",
        "candidate_proxy_identity": PROXY_CANDIDATE,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", required=True)
    parser.add_argument("--upstream", type=Path, default=HERE.parents[2] / TOPIC)
    args = parser.parse_args()
    print(json.dumps(plan(args.upstream), sort_keys=True))


if __name__ == "__main__":
    main()
