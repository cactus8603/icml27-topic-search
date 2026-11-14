import sys, os, shutil, torch, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from safetensors.torch import load_file, save_file
ap = argparse.ArgumentParser(); ap.add_argument("--setting", required=True); ap.add_argument("--tag", default=""); ap.add_argument("--steps", type=int, nargs="+", default=STEPS)
ap.add_argument("--ws", type=float, nargs="+", default=W_NEG + W_POS)
a = ap.parse_args()
base = SETTINGS[a.setting]["base"]
B = load_file(f"{base}/model.safetensors")
for s in a.steps:
    d = f"{RUNS}/{a.setting}/theta_s{a.tag}/step{s}"
    S = load_file(f"{d}/model.safetensors")
    keys = [k for k in B if k in S]; assert len(keys) == len(B), (len(keys), len(B))
    for w in a.ws:
        out = f"{RUNS}/{a.setting}/merged{a.tag}/s{s}_{wtag(w)}"; os.makedirs(out, exist_ok=True)
        # theta_neon = theta_base - w*(theta_s - theta_base) = (1+w) base - w theta_s ; computed in fp32, stored fp16
        M = {k: ((1 + w) * B[k].float() - w * S[k].float()).to(torch.float16).contiguous() for k in B}
        save_file(M, f"{out}/model.safetensors", metadata={"format": "pt"})
        for f in ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]:
            shutil.copy(f"{base}/{f}", f"{out}/{f}")
        # edit dtype in config
        import json; c = json.load(open(f"{out}/config.json")); c["torch_dtype"] = "float16"; json.dump(c, open(f"{out}/config.json", "w"))
        print("merged", s, w, flush=True)
