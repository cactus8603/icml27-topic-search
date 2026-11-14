"""共用工具：設定讀取、模型載入、路徑。"""
import json, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", "/work/ChihChungHsu/cactus8603/icml/hf_home")


def load_json(name):
    with open(ROOT / "configs" / name, encoding="utf-8") as f:
        return json.load(f)


def prompts_by_id():
    return {p["id"]: p for p in load_json("prompts.json")["prompts"]}


def _unwrap_state_dict(sd):
    """容許 {'state_dict': ...}、'unet.' / 'module.' 前綴。"""
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    out = {}
    for k, v in sd.items():
        for pre in ("module.", "unet."):
            if k.startswith(pre):
                k = k[len(pre):]
        out[k] = v
    return out


def base_path():
    from huggingface_hub import snapshot_download
    return snapshot_download(load_json("models.json")["base_repo"], local_files_only=True,
                             allow_patterns=["model_index.json", "scheduler/*", "tokenizer/*",
                                             "text_encoder/*", "unet/*", "vae/*"])


def load_pipeline(model_key, device="cuda", dtype=None, size_override=None, *, base_snapshot=None):
    """回傳 StableDiffusionPipeline（safety_checker=None，DDIM scheduler）。"""
    import torch
    from diffusers import StableDiffusionPipeline, DDIMScheduler
    from huggingface_hub import hf_hub_download, snapshot_download
    dtype = dtype or torch.float16
    cfg = load_json("models.json")["models"][model_key]
    pipe = StableDiffusionPipeline.from_pretrained(
        base_snapshot or base_path(), torch_dtype=dtype, safety_checker=None, feature_extractor=None,
        requires_safety_checker=False, local_files_only=True, use_safetensors=True)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    info = {"model": model_key, "kind": cfg["kind"]}
    if cfg["kind"] == "unet_pt":
        p = hf_hub_download(cfg["repo"], cfg["file"], local_files_only=True)
        sd = _unwrap_state_dict(torch.load(p, map_location="cpu", weights_only=True))
        res = pipe.unet.load_state_dict({k: v.to(dtype) for k, v in sd.items()}, strict=False)
        info.update(missing=len(res.missing_keys), unexpected=len(res.unexpected_keys))
        if res.missing_keys or res.unexpected_keys:
            raise RuntimeError(f"{model_key}: UNet 權重鍵不吻合 missing={res.missing_keys[:3]} "
                               f"unexpected={res.unexpected_keys[:3]}（先跑 verify_checkpoints.py）")
    elif cfg["kind"] == "text_encoder_bin":
        d = snapshot_download(cfg["repo"], allow_patterns=[cfg["subfolder"] + "/*"], local_files_only=True)
        sd = torch.load(os.path.join(d, cfg["subfolder"], "pytorch_model.bin"), map_location="cpu", weights_only=True)
        res = pipe.text_encoder.load_state_dict({k: v.to(dtype) for k, v in sd.items()}, strict=False)
        bad_missing = [k for k in res.missing_keys if "position_ids" not in k]
        bad_unexp = [k for k in res.unexpected_keys if "position_ids" not in k]
        info.update(missing=len(bad_missing), unexpected=len(bad_unexp))
        if bad_missing or bad_unexp:
            raise RuntimeError(f"{model_key}: text encoder 鍵不吻合 {bad_missing[:3]} {bad_unexp[:3]}")
    return pipe.to(device), info
