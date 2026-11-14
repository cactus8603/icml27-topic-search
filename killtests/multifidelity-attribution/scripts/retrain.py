"""LDS 的移除子集重訓。每個模型只存 k_total 個查詢上的 eval 損失（及 2000 筆測試準確率），不存權重。
模型 j>=0：以 removal_mask(j) 的權重重訓；j<0：無移除的重複（雜訊底線，GPU 非確定性）。
可續跑：已存在的輸出會跳過。"""
import argparse, time, os, sys
import numpy as np
import jax
sys.path.insert(0, os.path.dirname(__file__))
from jax_common import setup, weights_for

ap = argparse.ArgumentParser()
ap.add_argument("--cfg", required=True)
ap.add_argument("--task", type=int, default=0)
ap.add_argument("--ntasks", type=int, default=1)
ap.add_argument("--j-start", type=int, default=0)
ap.add_argument("--j-end", type=int, default=None, help="預設 cfg.M")
ap.add_argument("--noise", type=int, default=None, help="本任務額外做的無移除重複次數（預設只有 task 0 做 cfg.noise_repeats 次）")
ap.add_argument("--train-seed", type=int, default=None, help="跨種子評測時用不同的訓練 seed（初始化與資料順序都變）")
ap.add_argument("--outname", default="retrain")
a = ap.parse_args()
cfg, tr, xq, yq, (xte, yte), P = setup(a.cfg)
seed = cfg["train_seed"] if a.train_seed is None else a.train_seed
out = os.path.join(os.path.dirname(P["retrain"]), a.outname)
os.makedirs(out, exist_ok=True)
jend = cfg["M"] if a.j_end is None else a.j_end
nz = (cfg["noise_repeats"] if a.task == 0 else 0) if a.noise is None else a.noise
js = [-(r + 1) for r in range(nz)] + [j for j in range(a.j_start, jend) if j % a.ntasks == a.task]
xt, yt = jax.numpy.asarray(xte[:2000]), jax.numpy.asarray(yte[:2000])
for j in js:
    f = f"{out}/loss_{j}.npy" if j >= 0 else f"{out}/noise_{-j}_task{a.task}.npy"
    if os.path.exists(f):
        continue
    t0 = time.time()
    st = tr.train(weights_for(cfg, j, tr.n), seed)
    ql = np.asarray(tr.query_losses(st, xq, yq), np.float32)
    acc = float((np.asarray(tr.m.apply(st[0], st[2], xt, False)[0]).argmax(1) == np.asarray(yt)).mean())
    np.save(f, np.concatenate([ql, [acc]]).astype(np.float32))  # 最後一個數是準確率
    print(f"j={j} seed={seed} {time.time()-t0:.0f}s mean_qloss={ql.mean():.4f} acc={acc:.4f}", flush=True)
