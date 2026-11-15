"""下載第 1 週所需的模型權重與偵測器（只做網路下載，不佔 GPU）。
用法：HF_HOME=/work/ChihChungHsu/cactus8603/icml/hf_home python scripts/download_assets.py [--only base ckpts detectors]
"""
import argparse, os, json
os.environ.setdefault("HF_HOME", "/work/ChihChungHsu/cactus8603/icml/hf_home")
from huggingface_hub import snapshot_download, hf_hub_download

IJD = "siyich/ijd-unlearned-checkpoints"
CKPT_FILES = [
    "nudity/UCE/UCE-Nudity-Diffusers-UNet.pt",
    "nudity/ESD/ESD-Nudity-Diffusers-UNet-noxattn.pt",
    "nudity/Salun/Salun-Nudity-Diffusers-UNet.pt",
    "rece/nudity_ep2.pt",
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=["base", "ckpts", "detectors"])
    a = ap.parse_args()
    out = {}
    if "base" in a.only:
        out["base"] = snapshot_download(
            "CompVis/stable-diffusion-v1-4",
            allow_patterns=["model_index.json", "scheduler/*", "tokenizer/*", "text_encoder/config.json",
                            "text_encoder/model.safetensors", "unet/config.json",
                            "unet/diffusion_pytorch_model.safetensors", "vae/config.json",
                            "vae/diffusion_pytorch_model.safetensors"])
    if "ckpts" in a.only:
        for f in CKPT_FILES:
            out[f] = hf_hub_download(IJD, f)
        out["advunlearn"] = snapshot_download("OPTML-Group/AdvUnlearn", allow_patterns=["nudity_unlearned/*"])
    if "detectors" in a.only:
        out["falconsai"] = snapshot_download("Falconsai/nsfw_image_detection", allow_patterns=["*.json", "*.safetensors"])
        out["clip"] = snapshot_download("openai/clip-vit-large-patch14", allow_patterns=["*.json", "*.safetensors", "merges.txt", "vocab.json"])
    print(json.dumps(out, indent=1))

if __name__ == "__main__":
    main()
