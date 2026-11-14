"""把 base 加高斯雜訊存成「假 RL」模型，只給 smoke test 用。用法：make_noisy_copy.py SRC DST [std]"""
import sys, shutil, os, glob, torch
from safetensors.torch import load_file, save_file
src, dst = sys.argv[1], sys.argv[2]; std = float(sys.argv[3]) if len(sys.argv) > 3 else 0.004
os.makedirs(dst, exist_ok=True)
torch.manual_seed(0)
sd = {}
for f in glob.glob(src + "/*.safetensors"):
    sd.update(load_file(f))
sd = {k: (v + std * torch.randn_like(v.float()).to(v.dtype)) if v.ndim >= 2 else v for k, v in sd.items()}
save_file(sd, dst + "/model.safetensors", metadata={"format": "pt"})
for f in os.listdir(src):
    if not f.endswith(".safetensors") and os.path.isfile(src + "/" + f) and f != "README.md":
        shutil.copy(src + "/" + f, dst)
