"""WiSE-FT：權重內插 theta = (1-lam)*theta_base + lam*theta_rl（fp32 內插，存 bf16）。
輸出到 --out（放節點本機暫存，用完由 slurm 腳本刪除，避免佔 6x3GB 磁碟）。tokenizer/config 取自 base。"""
import argparse, glob, os, shutil, sys
import torch
from safetensors.torch import load_file, save_file

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True, help="本機目錄")
ap.add_argument("--rl", required=True, help="本機目錄")
ap.add_argument("--lam", type=float, required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()


def load_dir(d):
    sd = {}
    for f in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
        sd.update(load_file(f))
    return sd


b, r = load_dir(a.base), load_dir(a.rl)
only_b, only_r = set(b) - set(r), set(r) - set(b)
# 綁定 embedding 的模型可能少 lm_head.weight；其他不一致一律報錯
ok_missing = {"lm_head.weight"}
assert only_b <= ok_missing and only_r <= ok_missing, (sorted(only_b)[:5], sorted(only_r)[:5])
# 審查修正 2026-11-14：單側多出的 lm_head 會被丟掉、輸出沿用 base config 的 tie_word_embeddings（lm_head = 內插後的 embed）。
# 這只有在該側 lm_head 與它自己的 embed_tokens 相同時才正確（SimpleRL-Zoo-1.5B 實查：max|diff| = 0）。不相同就停，不默默算錯。
for sd, name in ((r, "rl"), (b, "base")):
    if "lm_head.weight" in sd and "lm_head.weight" in (only_r if name == "rl" else only_b):
        assert torch.equal(sd["lm_head.weight"].float(), sd["model.embed_tokens.weight"].float()), \
            f"{name} 的 lm_head 與 embed_tokens 不同，不能以 tied 方式內插；需改成內插 lm_head 並設 tie_word_embeddings=false"
out = {}
for k in b:
    if k not in r:
        out[k] = b[k].to(torch.bfloat16); continue
    assert b[k].shape == r[k].shape, k
    out[k] = ((1 - a.lam) * b[k].float() + a.lam * r[k].float()).to(torch.bfloat16)
os.makedirs(a.out, exist_ok=True)
save_file(out, os.path.join(a.out, "model.safetensors"), metadata={"format": "pt"})
for f in ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]:
    if os.path.exists(os.path.join(a.base, f)):
        shutil.copy(os.path.join(a.base, f), a.out)
print("wise", a.lam, "->", a.out, len(out), "tensors")
