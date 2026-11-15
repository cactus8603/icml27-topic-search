"""CPU 上驗證檢查點格式：載入（weights_only=True，不執行 pickle 程式碼），比對與 base UNet 的鍵/形狀，並統計改了哪些參數。
用法：python verify_checkpoints.py [模型鍵...]   輸出 logs/verify_checkpoints.json
"""
import json, sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load_json, base_path, _unwrap_state_dict
import torch
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download, snapshot_download

cfgs = load_json("models.json")["models"]
keys = sys.argv[1:] or [k for k in cfgs if k != "sd14"]
bp = base_path()
base_unet = load_file(os.path.join(bp, "unet", "diffusion_pytorch_model.safetensors"))
base_te = load_file(os.path.join(bp, "text_encoder", "model.safetensors"))
report = {}
for k in keys:
    c = cfgs[k]; r = {"kind": c["kind"]}
    try:
        if c["kind"] == "unet_pt":
            p = hf_hub_download(c["repo"], c["file"], local_files_only=True)
            raw = torch.load(p, map_location="cpu", weights_only=True)
            r["top_type"] = type(raw).__name__
            sd = _unwrap_state_dict(raw); ref = base_unet
        else:
            d = snapshot_download(c["repo"], allow_patterns=[c["subfolder"] + "/*"], local_files_only=True)
            sd = torch.load(os.path.join(d, c["subfolder"], "pytorch_model.bin"), map_location="cpu", weights_only=True); ref = base_te
        r["n_keys"] = len(sd); r["n_ref"] = len(ref)
        r["missing"] = [x for x in ref if x not in sd and "position_ids" not in x][:5]
        r["unexpected"] = [x for x in sd if x not in ref and "position_ids" not in x][:5]
        r["shape_mismatch"] = [x for x in sd if x in ref and tuple(sd[x].shape) != tuple(ref[x].shape)][:5]
        changed = [x for x in sd if x in ref and sd[x].shape == ref[x].shape and not torch.equal(sd[x].float(), ref[x].float())]
        r["n_changed_params"] = len(changed)
        r["changed_examples"] = changed[:6]
        r["dtype"] = str(next(iter(sd.values())).dtype)
        r["ok"] = not (r["missing"] or r["unexpected"] or r["shape_mismatch"])
    except Exception as e:
        r["error"] = repr(e)[:500]; r["ok"] = False
    report[k] = r; print(k, json.dumps(r, ensure_ascii=False), flush=True)
(ROOT / "logs").mkdir(exist_ok=True)
(ROOT / "logs" / "verify_checkpoints.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
