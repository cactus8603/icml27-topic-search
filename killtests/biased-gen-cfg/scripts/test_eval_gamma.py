#!/usr/bin/env python
"""eval_hier_oracle.py --gamma 的合成測試（CPU；隨機初始化的小模型與隨機資料，不碰任何真實結果）。
驗：(1) 加 --gamma 後既有輸出逐位元不變；(2) pair_cos_gamma 形狀／範圍；(3) 與獨立重算的 1-cos(γ_i, γ_j) 一致；(4) 兩個相同模型的距離為 0；(5) 非 normal run 會拒絕。
用法：python test_eval_gamma.py [--tmp DIR]（需 biasgen 環境）"""
import argparse, json, os, subprocess, sys, shutil
import numpy as np, torch
import torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import hier_cond as H
ap = argparse.ArgumentParser(); ap.add_argument("--tmp", default=os.path.join(os.environ.get("TMPDIR", "."), "evalgamma_test")); a = ap.parse_args()
shutil.rmtree(a.tmp, ignore_errors=True); os.makedirs(a.tmp)
q, depth, N, bs, ne = 3, 2, 200, 20, 6
rng = np.random.default_rng(0)
M = rng.dirichlet(np.ones(q * q), size=q).reshape(q, q, q)
leaves = rng.integers(0, q, size=(2 ** depth, N)); roots = rng.integers(0, q, size=N)
d = np.empty(7, dtype=object); d[:] = [q, depth, 1.0, roots, leaves, M, None]
dpath = os.path.join(a.tmp, "data.npy"); np.save(dpath, d, allow_pickle=True)
EP = [10, 20, 30]
def mkrun(name, pick, seed, extra=None):
    r = os.path.join(a.tmp, name); os.makedirs(r)
    meta = dict(pick=pick, hidden=32, nhead=2, layers=1, t_final=500, pdrop=0.1, **(extra or {})); json.dump(meta, open(os.path.join(r, "meta.json"), "w"))
    for e in EP:
        torch.manual_seed(seed * 100 + e); m = H.CondDenoiser(2 ** depth, q, 32, 2, 1); torch.save(m.state_dict(), os.path.join(r, f"ep{e}.pt"))
    return r
r0, r1, r2 = mkrun("p0", 0, 1), mkrun("p1", 1, 2), mkrun("p2", 2, 3)
def ev(runs, out, *extra):
    cmd = [sys.executable, os.path.join(HERE, "eval_hier_oracle.py"), "--data", dpath, "--runs", *runs, "--out", out, "--block-size", str(bs), "--n-eval", str(ne), "--device", "cpu", *extra]
    return subprocess.run(cmd, capture_output=True, text=True)
fails = []
def check(n, c, x=""):
    print(("OK   " if c else "FAIL ") + n, x)
    if not c: fails.append(n)
base, gam = os.path.join(a.tmp, "base.npz"), os.path.join(a.tmp, "gam.npz")
rb = ev([r0, r1, r2], base); rg = ev([r0, r1, r2], gam, "--gamma")
assert rb.returncode == 0 and rg.returncode == 0, (rb.stderr[-800:], rg.stderr[-800:])
zb, zg = np.load(base, allow_pickle=True), np.load(gam, allow_pickle=True)
same = all(np.array_equal(zb[k], zg[k], equal_nan=True) if zb[k].dtype.kind in "fiu" else np.array_equal(zb[k], zg[k]) for k in zb.files)
check("既有輸出逐位元不變（全部鍵）", same and set(zg.files) - set(zb.files) == {"pair_cos_gamma"}, str(set(zg.files) - set(zb.files)))
check("沒有 --gamma 時不輸出 pair_cos_gamma", "pair_cos_gamma" not in zb.files)
g = zg["pair_cos_gamma"]; check("形狀 [pairs,E,T,3,1]", g.shape == zg["pair_cos_eps"].shape[:-1] + (1,), str(g.shape))
check("有限且在 [0,2]", bool(np.isfinite(g).all() and (g >= 0).all() and (g <= 2 + 1e-9).all()))
# 獨立重算（複製 eval 的噪聲產生順序）：pair (0,1)、epoch 20、t=100、輸入集 S_0（k=0）
dev = torch.device("cpu"); ab = H.make_alpha_bars(500, dev, torch.float64); T = [10, 25, 50, 75, 100, 150, 200, 300]
Mt = torch.from_numpy(M)
def block(p): return torch.from_numpy(leaves.T[p * bs:(p + 1) * bs][:ne].copy()), torch.from_numpy(roots[p * bs:(p + 1) * bs][:ne].copy())
sets = [block(0), block(1), block(2), block(7)]
gen = torch.Generator(device=dev).manual_seed(123); XT = []
for x0, c in sets:
    oh = F.one_hot(x0, q).double(); XT.append([H.noise_batch(oh, torch.full((ne,), t), ab, gen).float() for t in T])
def gam_vec(run, e, s, ti):
    m = H.CondDenoiser(2 ** depth, q, 32, 2, 1); m.load_state_dict(torch.load(os.path.join(run, f"ep{e}.pt"))); m.eval()
    with torch.no_grad():
        pc = torch.softmax(m(XT[s][ti], torch.full((ne,), T[ti]), sets[s][1]).float(), -1).double(); pu = torch.softmax(m(XT[s][ti], torch.full((ne,), T[ti]), torch.full((ne,), q)).float(), -1).double()
    return (pc - pu).flatten(1)
gi, gj = gam_vec(r0, 20, 0, 4), gam_vec(r1, 20, 0, 4)
manual = float((1 - F.cosine_similarity(gi, gj, dim=1)).mean())
pi = [tuple(p) for p in zg["pairs"]].index((0, 1)); ei = list(zg["epochs"]).index(20)
check("與獨立重算的 1-cos(γ_i,γ_j) 一致", abs(g[pi, ei, 4, 0, 0] - manual) < 1e-9, f"{g[pi, ei, 4, 0, 0]:.10f} vs {manual:.10f}")
# 兩個相同模型 -> 0
rdup = os.path.join(a.tmp, "p1dup"); shutil.copytree(r1, rdup); m = json.load(open(os.path.join(rdup, "meta.json"))); m["pick"] = 3; json.dump(m, open(os.path.join(rdup, "meta.json"), "w"))
gd = os.path.join(a.tmp, "dup.npz"); assert ev([r1, rdup], gd, "--gamma").returncode == 0
z = np.load(gd, allow_pickle=True); check("相同權重的兩模型：pair_cos_gamma 與 pair_cos_eps 皆為 0", float(np.abs(z["pair_cos_gamma"]).max()) < 1e-12 and float(np.abs(z["pair_cos_eps"]).max()) < 1e-12)
# 非 normal run 拒絕
rk1 = mkrun("k1", 0, 9, dict(const_label=0)); r = ev([rk1], os.path.join(a.tmp, "bad.npz"), "--gamma")
check("const-label run 加 --gamma 會被拒絕", r.returncode != 0 and "只支援" in r.stderr)
print("\nALL PASS" if not fails else f"\nFAILED: {fails}"); sys.exit(1 if fails else 0)
