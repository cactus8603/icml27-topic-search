"""訓練主模型（seed=train_seed，全資料）與 TRAK 集成模型；存最終權重/緩衝（及 TracIn 檢查點）給 PyTorch 便宜方法用，
並存 JAX 側的對照 logits 以驗證匯出保真度。不存大檔：每個 npz 約 40 MB，共約 0.3 GB。"""
import argparse, json, time
import numpy as np
import jax
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from jax_common import setup, weights_for
import mf_core as mc


def flat(state, prefix=""):
    out = {}
    params, _, bufs = state
    for n, d in params.items():
        for k, v in d.items():
            out[f"p.{n}.{k}"] = np.asarray(v)
    for n, d in bufs.items():
        for k, v in d.items():
            out[f"b.{n}.{k}"] = np.asarray(v)
    return out


ap = argparse.ArgumentParser()
ap.add_argument("--cfg", required=True)
a = ap.parse_args()
cfg, tr, xq, yq, (xte, yte), P = setup(a.cfg)
n = tr.n
w1 = weights_for(cfg, -1, n)
seeds = [cfg["train_seed"]] + list(cfg["ensemble_seeds"])
keep = {e * tr.spe: e for e in cfg["tracin_epochs"]}
for s in seeds:
    t0 = time.time()
    state, ck = tr.train(w1, s, keep_every=tr.spe) if s == cfg["train_seed"] else (tr.train(w1, s), {})
    ql = np.asarray(tr.query_losses(state, xq, yq))
    logits = np.asarray(tr.m.apply(state[0], state[2], xq[:64], False)[0])
    acc = float((np.asarray(tr.m.apply(state[0], state[2], jax.numpy.asarray(xte[:2000]), False)[0]).argmax(1) == yte[:2000]).mean())
    np.savez(f"{P['models']}/model_s{s}.npz", ref_logits=logits, query_loss=ql, acc2000=acc, lr_final=tr.lr(tr.T - 1), **flat(state))
    if ck:
        for t, e in keep.items():
            if t in ck:
                np.savez(f"{P['models']}/model_s{s}_ep{e}.npz", lr=tr.lr(max(t - 1, 0)), **flat(ck[t]))
    print(f"seed {s}: {time.time()-t0:.0f}s acc(2000 test)={acc:.4f} mean query loss={ql.mean():.4f}", flush=True)
