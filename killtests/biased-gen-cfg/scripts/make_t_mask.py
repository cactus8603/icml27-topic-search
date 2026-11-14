#!/usr/bin/env python
"""依 DECISIONS_fable 條文 1.3：只用 BP oracle（不看任何 checkpoint）決定階層資料主指標的 t 遮罩。
輸入：測試區塊 7 前 1000 筆；CPU torch.Generator seed 20261114；float64；依 t 網格順序加噪。
g_t = mean_{樣本,位置} Σ_符號 (p_c^BP - p_u^BP)^2（真實根標籤）；保留 g_t >= 0.01 * max_t g_t 的 t。
輸出 configs/t_mask_hier.json（內含 g_t、門檻、seed、資料 sha256）。先 commit 再送評估。
"""
import argparse, hashlib, json, os, sys
import torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hier_cond as H

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--data", default=os.environ.get("HIER_DATA", "/work/ChihChungHsu/cactus8603/icml/data/biasgen-hier/hier_q6_l4_s1.0_qeff4_seed0_n40000.npy"))
ap.add_argument("--out", default=os.path.join(HERE, "..", "configs", "t_mask_hier.json"))
ap.add_argument("--t-grid", default="10,25,50,75,100,150,200,300")
ap.add_argument("--test-block", type=int, default=7)
ap.add_argument("--block-size", type=int, default=5000)
ap.add_argument("--n", type=int, default=1000)
ap.add_argument("--seed", type=int, default=20261114)
ap.add_argument("--rel-threshold", type=float, default=0.01)
args = ap.parse_args()
torch.set_num_threads(4)

q, depth, roots, leaves, M = H.load_hier(args.data)
lo = args.test_block * args.block_size
x0 = leaves[lo:lo + args.n]; c = roots[lo:lo + args.n]
assert len(x0) == args.n
T = [int(x) for x in args.t_grid.split(",")]
ab = H.make_alpha_bars(500, "cpu", torch.float64)
Mt = torch.from_numpy(M)
gen = torch.Generator(device="cpu").manual_seed(args.seed)
oh = F.one_hot(x0, q).double()
g = {}
for t in T:
    xt = H.noise_batch(oh, torch.full((args.n,), t), ab, gen)
    pc, pu, _ = H.oracle_cond_uncond(Mt, depth, xt, float(ab[t - 1]), roots=c)
    g[t] = float(((pc - pu) ** 2).sum(-1).mean())
gmax = max(g.values())
keep = [t for t in T if g[t] >= args.rel_threshold * gmax]
sha = hashlib.sha256(open(args.data, "rb").read()).hexdigest()
out = {"t_grid": T, "g_t": {str(t): g[t] for t in T}, "rel_threshold": args.rel_threshold, "g_max": gmax,
       "keep_t": keep, "seed": args.seed, "generator": "torch.Generator(cpu)", "dtype": "float64",
       "test_block": args.test_block, "block_size": args.block_size, "n_samples": args.n,
       "data_file": os.path.basename(args.data), "data_sha256": sha,
       "rule": "keep t with g_t >= rel_threshold * max_t g_t; equal weights; defined by BP oracle only (DECISIONS_fable 1.3)"}
json.dump(out, open(args.out, "w"), indent=1)
print(json.dumps(out, indent=1))
