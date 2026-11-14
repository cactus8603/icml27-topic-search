"""精確 metagradient Y（MAGIC 類）：對查詢 [qstart,qend) 做反向重播，一次 qchunk 個查詢（vmap）。
存 Y_{qs}_{qe}.npy（n×Q，float32，元素是 dloss_q/dw_i）。--bench：只量吞吐（前向全做、反向 20 步）後外推。"""
import argparse, time, os, sys
import numpy as np
import jax
sys.path.insert(0, os.path.dirname(__file__))
from jax_common import setup, weights_for

ap = argparse.ArgumentParser()
ap.add_argument("--cfg", required=True)
ap.add_argument("--qstart", type=int, default=0)
ap.add_argument("--qend", type=int, default=None)
ap.add_argument("--bench", action="store_true")
a = ap.parse_args()
print('jax.devices():', jax.devices(), flush=True)
if os.environ.get('REQUIRE_GPU') == '1' and jax.devices()[0].platform == 'cpu':
    sys.exit('REQUIRE_GPU=1 但 JAX 只看到 CPU：中止（檢查 jax[cuda13] 與驅動）')
cfg, tr, xq, yq, _, P = setup(a.cfg)
qend = cfg["k_stage0"] if a.qend is None else a.qend
seed = cfg["train_seed"]
w1 = weights_for(cfg, -1, tr.n)
t0 = time.time()
final, ck = tr.train(w1, seed, keep_every=cfg["ckpt_every"])
jax.block_until_ready(final)
t_fwd = time.time() - t0
print(f"forward {t_fwd:.0f}s ({tr.T} steps)", flush=True)
qc = cfg["qchunk"]
if a.bench:
    # 審查修正 2026-11-14：先暖機（JIT 編譯），再以 t(1+nb) − t(1) 取純反向步耗時（兩次都含同樣的段重算，相減抵消），
    # 另以 forward 每步耗時近似段重算成本；舊版把編譯與 49 步段重算都攤進 20 步，外推會系統性高估。
    nb = 20
    jax.block_until_ready(tr.metagrad(w1, seed, xq[:qc], yq[:qc], cfg["ckpt_every"], (final, ck), max_bwd=1)[0])
    t1 = time.time(); tr.metagrad(w1, seed, xq[:qc], yq[:qc], cfg["ckpt_every"], (final, ck), max_bwd=1); ta = time.time() - t1
    t1 = time.time(); tr.metagrad(w1, seed, xq[:qc], yq[:qc], cfg["ckpt_every"], (final, ck), max_bwd=1 + nb); tbb = time.time() - t1
    tb = (tbb - ta) / nb
    tf = t_fwd / tr.T  # 段重算：每個反向步約多一個前向步（上界，含首次編譯）
    per_chunk = (tb + tf) * tr.T / 60 + t_fwd / 60
    print(f"bwd {tb*1000:.0f} ms/step（純反向）+ 段重算約 {tf*1000:.0f} ms/step -> 一個 chunk({qc} 查詢，含一次前向訓練) 約 {per_chunk:.1f} 分鐘；"
          f"{qend-a.qstart} 查詢約 {per_chunk*np.ceil((qend-a.qstart)/qc):.0f} 分鐘", flush=True)
    sys.exit(0)
for s in range(a.qstart, qend, qc):
    e = min(s + qc, qend)
    f = f"{P['exact']}/Y_{s}_{e}.npy"
    if os.path.exists(f):
        continue
    t1 = time.time()
    Y, ql = tr.metagrad(w1, seed, xq[s:e], yq[s:e], cfg["ckpt_every"], (final, ck))
    np.save(f, Y.T.astype(np.float32))
    np.save(f"{P['exact']}/ql_{s}_{e}.npy", ql)
    print(f"queries [{s},{e}) {time.time()-t1:.0f}s |Y| mean abs={np.abs(Y).mean():.3e}", flush=True)
