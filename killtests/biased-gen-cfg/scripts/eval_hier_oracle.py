#!/usr/bin/env python
"""用精確 BP 算條件／無條件／CFG oracle，輸出各 checkpoint 的 oracle 偏差曲線。

輸入集合：每個 run 的自身訓練集（噪聲化）S_j，以及共同測試集 S_test（預設資料區塊 7，與訓練區塊不相交）。
分支 b：cond、uncond、cfg{w}（x0hat 空間 (1+w)p_c - w p_u；w 由 --guidance 指定，w=0 即 cond 不另列）。
輸出 npz（results/hier_oracle_<tag>.npz）：
  per_model（每個 run j、checkpoint e、時間 t、輸入集合 s、分支 b）：
     mse_oracle  NN 與 BP 的 x0hat 均方差      kl_oracle  KL(BP||NN)（僅 cond/uncond）
     ce          以真實 x0 為標籤的 token CE（僅 cond/uncond，其中 BP 的 ce 另存 ce_bp）
  pair（兩個不相交 run i<j、輸入集合 S_i / S_j / S_test）：pair_mse、pair_cos（x0-hat 空間 cosine 距離，保留但判定不得使用）、
     pair_cos_eps（ε 空間 1-cos，DECISIONS_fable 條文 1.1：每個分支先得 p_b（CFG 為 (1+w)p_c - w p_u），再轉
     ε_b=(x_t-sqrt(abar)p_b)/sqrt(1-abar)；float64；逐樣本整段序列展平；以 0.5*||a_hat-b_hat||^2 計算）
  另存 git_hash（含是否有未提交修改）。
偏差間隙 gap = mse_oracle(自身訓練輸入) - mse_oracle(測試輸入)；pair 版本 = pair_cos(S_i) - pair_cos(S_test)。
"""
import argparse, glob, json, os, subprocess, sys
import math
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hier_cond as H

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--runs", nargs="+", required=True, help="train_hier_cond.py 的輸出資料夾")
ap.add_argument("--out", required=True)
ap.add_argument("--test-block", type=int, default=7)
ap.add_argument("--n-eval", type=int, default=1000)
ap.add_argument("--block-size", type=int, default=5000, help="資料區塊大小（固定 5000，不再取 metas[0]['n']）")
ap.add_argument("--class-filter", type=int, default=None, help="每個區塊只取根 == 此值的前 n-eval 筆（正控制，搭配 --n-eval 700）")
ap.add_argument("--t-grid", default="10,25,50,75,100,150,200,300")
ap.add_argument("--guidance", default="1,3")
ap.add_argument("--epochs", default=None, help="逗號分隔；預設取所有 run 共有的 checkpoint")
ap.add_argument("--seed", type=int, default=123)
ap.add_argument("--device", default=None)
args = ap.parse_args()

dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
metas = [json.load(open(os.path.join(r, "meta.json"))) for r in args.runs]
q, depth, roots, leaves, M = H.load_hier(args.data)
n_train = args.block_size
# run 類型（由 meta 判斷；同一次評估內必須同型）：normal / const（K=1 常數條件，條件輸入給 0、條件 oracle = 無條件 BP）/ uncond_only（pdrop=1.0，只輸出 null 分支）
def run_type(m):
    if m.get("const_label") is not None: return "const"
    if m.get("pdrop") == 1.0: return "uncond_only"
    return "normal"
types = {run_type(m) for m in metas}
assert len(types) == 1, f"同一次評估的 run 類型必須一致：{types}"
RUN_TYPE = types.pop()
CONST_LABEL = metas[0].get("const_label")
picks = [m["pick"] for m in metas]
assert args.test_block not in picks, "測試區塊與訓練區塊重疊"
assert len(set(picks)) == len(picks), "run 的資料區塊必須互不相交"
ne = args.n_eval
T = [int(x) for x in args.t_grid.split(",")]
W = [float(x) for x in args.guidance.split(",") if float(x) != 0]
branches = ["cond", "uncond"] + [f"cfg{w:g}" for w in W]
if RUN_TYPE == "uncond_only":
    branches = ["uncond"]
ab = H.make_alpha_bars(metas[0]["t_final"], dev, torch.float64)
Mt = torch.from_numpy(M).to(dev)

def block(p):
    lv, rt = leaves[p * n_train:(p + 1) * n_train], roots[p * n_train:(p + 1) * n_train]
    if args.class_filter is not None:
        keep = (rt == args.class_filter).nonzero().squeeze(1)[:ne]
        lv, rt = lv[keep], rt[keep]
    assert len(lv) >= ne, f"區塊 {p} 只有 {len(lv)} 筆可用，少於 n-eval={ne}"
    return lv[:ne].to(dev), rt[:ne].to(dev)
sets = [block(p) for p in picks] + [block(args.test_block)]   # 最後一個是 test
S = len(sets)

# 固定噪聲輸入與 oracle（跨 checkpoint 不變）
gen = torch.Generator(device=dev).manual_seed(args.seed)
XT, ORA, TOK, RT = [], [], [], []
for (x0, c) in sets:
    oh = F.one_hot(x0, q).double()
    xts, oras = [], []
    for t in T:
        xt = H.noise_batch(oh, torch.full((ne,), t, device=dev), ab, gen)
        pc, pu, _ = H.oracle_cond_uncond(Mt, depth, xt, float(ab[t - 1]), roots=c)
        xts.append(xt.float())
        if RUN_TYPE == "const":
            pc = pu      # K=1：條件不帶資訊，條件 oracle = 無條件 BP
        d = {"cond": pc, "uncond": pu}
        for w in W:
            d[f"cfg{w:g}"] = H.cfg_combine(pc, pu, w)
        oras.append(d)
    XT.append(xts); ORA.append(oras); TOK.append(x0); RT.append(c)

def ce_of(p, x0):
    return -torch.log(p.gather(-1, x0.unsqueeze(-1)).squeeze(-1).clamp_min(1e-12)).mean().item()
ce_bp = np.zeros((S, len(T), 2))
for s in range(S):
    for ti in range(len(T)):
        for bi, b in enumerate(["cond", "uncond"]):
            ce_bp[s, ti, bi] = ce_of(ORA[s][ti][b], TOK[s])
RIN = [torch.full_like(c, CONST_LABEL) if RUN_TYPE == "const" else c for c in RT]   # 模型的條件輸入

ck_sets = [set(int(os.path.basename(f)[2:-3]) for f in glob.glob(os.path.join(r, "ep*.pt"))) for r in args.runs]
epochs = sorted(set.intersection(*ck_sets)) if args.epochs is None else [int(x) for x in args.epochs.split(",")]
assert epochs, "找不到所有 run 共有的 checkpoint"
print("epochs", epochs, flush=True)
R = len(args.runs)
shape_m = (R, len(epochs), len(T), S, len(branches))
mse = np.zeros(shape_m); kl = np.full(shape_m, np.nan); ce = np.full(shape_m, np.nan)
pairs = [(i, j) for i in range(R) for j in range(i + 1, R)]
pair_mse = np.zeros((len(pairs), len(epochs), len(T), 3, len(branches)))
pair_cos = np.zeros_like(pair_mse)
pair_cos_eps = np.zeros_like(pair_mse)
ABAR = [float(ab[t - 1]) for t in T]

def eps_of(xt, p, abar):
    return (xt.double() - math.sqrt(abar) * p) / math.sqrt(1.0 - abar)

def one_minus_cos_eps(a, b):
    """逐樣本展平後的 1-cos，用 0.5*||a_hat-b_hat||^2（單位向量）避免相消誤差；float64。"""
    a = a.flatten(1).double(); b = b.flatten(1).double()
    a = a / a.norm(dim=1, keepdim=True).clamp_min(1e-300); b = b / b.norm(dim=1, keepdim=True).clamp_min(1e-300)
    return 0.5 * ((a - b) ** 2).sum(1)

def git_hash():
    try:
        d = os.path.dirname(os.path.abspath(__file__))
        h = subprocess.check_output(["git", "-C", d, "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", d, "status", "--porcelain", "--", d], text=True).strip()
        return h + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"

models = [H.CondDenoiser(depth and 2 ** depth, q, m["hidden"], m["nhead"], m["layers"]).to(dev).eval() for m in metas]

@torch.no_grad()
def predict(model, xt, t, c):
    out = model(xt, torch.full((xt.shape[0],), t, device=dev), c)
    return torch.softmax(out.float(), -1).double()

for ei, e in enumerate(epochs):
    preds = {}
    for j, (r, mod) in enumerate(zip(args.runs, models)):
        mod.load_state_dict(torch.load(os.path.join(r, f"ep{e}.pt"), map_location=dev))
        for s in range(S):
            for ti, t in enumerate(T):
                null = torch.full((ne,), q, device=dev)
                pc = predict(mod, XT[s][ti], t, RIN[s]); pu = predict(mod, XT[s][ti], t, null)
                d = {"cond": pc, "uncond": pu}
                for w in W:
                    d[f"cfg{w:g}"] = H.cfg_combine(pc, pu, w)
                preds[(j, s, ti)] = d
                for bi, b in enumerate(branches):
                    o = ORA[s][ti][b]
                    mse[j, ei, ti, s, bi] = ((d[b] - o) ** 2).mean().item()
                    if b in ("cond", "uncond") and b in branches:
                        kl[j, ei, ti, s, bi] = (o * (torch.log(o.clamp_min(1e-12)) - torch.log(d[b].clamp_min(1e-12)))).sum(-1).mean().item()
                        ce[j, ei, ti, s, bi] = ce_of(d[b], TOK[s])
    for pi, (i, j) in enumerate(pairs):
        for k, s in enumerate((i, j, S - 1)):
            for ti in range(len(T)):
                for bi, b in enumerate(branches):
                    a, bb = preds[(i, s, ti)][b], preds[(j, s, ti)][b]
                    pair_mse[pi, ei, ti, k, bi] = ((a - bb) ** 2).mean().item()
                    cs = F.cosine_similarity(a.flatten(1), bb.flatten(1), dim=1)
                    pair_cos[pi, ei, ti, k, bi] = (1 - cs).mean().item()
                    xt64 = XT[s][ti]
                    pair_cos_eps[pi, ei, ti, k, bi] = one_minus_cos_eps(
                        eps_of(xt64, a.double(), ABAR[ti]), eps_of(xt64, bb.double(), ABAR[ti])).mean().item()
    print(f"epoch {e} done", flush=True)

os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
np.savez(args.out, epochs=np.array(epochs), t_grid=np.array(T), branches=np.array(branches),
         picks=np.array(picks), pairs=np.array(pairs), mse_oracle=mse, kl_oracle=kl, ce=ce, ce_bp=ce_bp,
         pair_mse=pair_mse, pair_cos=pair_cos, pair_cos_eps=pair_cos_eps, git_hash=git_hash(), test_block=args.test_block, n_eval=ne, guidance=np.array(W))
print("saved", args.out)
