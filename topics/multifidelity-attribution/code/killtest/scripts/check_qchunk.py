"""一致性檢查：qchunk=1（逐查詢）與 qchunk>1（vmap）的精確 Y 是否一致。需先有 exact_y.py 的輸出 Y_{s}_{e}.npy。
用法：python check_qchunk.py --cfg configs/smoke.json --qstart 0 --qend 3  （GPU 上正式設定可用 --qend 2 抽查）
通過條件：相對誤差 ||Y1-Yc||/||Yc|| < 1e-4（float32 CPU）；GPU 預設精度可放寬到 1e-2 並記錄。"""
import argparse, os, sys, numpy as np, jax
sys.path.insert(0, os.path.dirname(__file__))
from jax_common import setup, weights_for

ap = argparse.ArgumentParser()
ap.add_argument("--cfg", required=True); ap.add_argument("--qstart", type=int, default=0)
ap.add_argument("--qend", type=int, default=3); ap.add_argument("--tol", type=float, default=1e-4)
a = ap.parse_args()
cfg, tr, xq, yq, _, P = setup(a.cfg)
seed = cfg["train_seed"]; w1 = weights_for(cfg, -1, tr.n)
# 審查修正 2026-11-14：找「包含 [qstart,qend) 的已存區塊」再切片（舊版找 Y_{qstart}_{qend}.npy，--qend 2 時檔案不存在）
import glob
cand = [f for f in glob.glob(f"{P['exact']}/Y_*_*.npy")
        if int(os.path.basename(f)[2:-4].split("_")[0]) <= a.qstart and int(os.path.basename(f)[2:-4].split("_")[1]) >= a.qend]
assert cand, f"找不到包含查詢 [{a.qstart},{a.qend}) 的 Y 區塊"
ref = cand[0]; s0 = int(os.path.basename(ref)[2:-4].split("_")[0])
Yc = np.load(ref)[:, a.qstart - s0:a.qend - s0]  # n x Q
final, ck = tr.train(w1, seed, keep_every=cfg["ckpt_every"])
cols = []
for q in range(a.qstart, a.qend):
    Y, _ = tr.metagrad(w1, seed, xq[q:q+1], yq[q:q+1], cfg["ckpt_every"], (final, ck))
    cols.append(Y.T)
Y1 = np.concatenate(cols, 1)
rel = np.linalg.norm(Y1 - Yc) / np.linalg.norm(Yc)
per = [float(np.linalg.norm(Y1[:, i] - Yc[:, i]) / np.linalg.norm(Yc[:, i])) for i in range(Yc.shape[1])]
print(f"qchunk=1 vs 已存檔(qchunk={cfg['qchunk']}) 相對誤差 {rel:.3e}；逐查詢 {['%.2e' % p for p in per]}；門檻 {a.tol} -> {'PASS' if rel < a.tol else 'FAIL'}")
sys.exit(0 if rel < a.tol else 1)
