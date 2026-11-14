"""MAUVE (x100, mauve-text defaults: gpt2-large, auto buckets, seed 25), distinct-4, unigram entropy for generated continuations."""
import os
for _k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ[_k] = "1"
import sys, glob, time, json, math, argparse, re, collections, numpy as np, multiprocessing as mp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
ap = argparse.ArgumentParser(); ap.add_argument("--setting", required=True); ap.add_argument("--split", required=True)
ap.add_argument("--names", nargs="*", default=None); ap.add_argument("--tag", default=""); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--only-best", default=None, help="json mapping name->config name; restrict to those configs")
ap.add_argument("--procs", type=int, default=12)
a = ap.parse_args()
OUT = f"{RES}/mauve_{a.setting}_{a.split}{a.tag}.json"
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(SETTINGS[a.setting]["base"])
d = np.load(f"{RUNS}/data/lm_{a.split}.npz"); human = d["cont"]
if a.limit: human = human[:a.limit]

def words(t): return t.split()
def div_stats(texts):
    cnt = collections.Counter(); n4 = collections.Counter(); tot4 = 0
    for t in texts:
        w = words(t); cnt.update(w)
        for i in range(len(w) - 3): n4[tuple(w[i:i + 4])] += 1
        tot4 += max(0, len(w) - 3)
    N = sum(cnt.values()); H = -sum(c / N * math.log(c / N) for c in cnt.values())
    return dict(distinct4=len(n4) / max(tot4, 1), unigram_entropy_nats=H)
def mauve_job(args):
    import importlib; cm = importlib.import_module("mauve.compute_mauve")
    key, pf, qf = args
    r = cm.compute_mauve(p_features=pf, q_features=qf, verbose=False)  # defaults
    return key, float(r.mauve) * 100
if __name__ == "__main__":
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    import importlib; cm = importlib.import_module("mauve.compute_mauve")
    htexts = tok.batch_decode(human.tolist())
    pf = cm.get_features_from_input(None, None, htexts, "gpt2-large", 1024, 0, "p", 64)
    res["_human"] = div_stats(htexts); res["_human"]["n"] = len(htexts)
    names = a.names or [n for n, _ in model_list(a.setting)]
    best = json.load(open(a.only_best)) if a.only_best else None
    pool = mp.get_context("fork").Pool(a.procs); pend = []; t0 = time.time(); nf = 0
    def drain(block=False):
        global pend
        keep = []
        for r in pend:
            if r.ready() or block:
                k, m = r.get(); res[k]["mauve"] = m
            else: keep.append(r)
        pend = keep
    for n in names:
        files = sorted(glob.glob(f"{RUNS}/{a.setting}/gen{a.tag}/{n}/{a.split}/*.npy"))
        for f in files:
            b = os.path.basename(f)[:-4]; cfg, seed = b.rsplit("_s", 1)
            if best is not None and best.get(n) != cfg: continue
            key = f"{n}|{cfg}|{seed}"
            if key in res and "mauve" in res[key]: continue
            G = np.load(f)
            if a.limit: G = G[:a.limit]
            texts = tok.batch_decode(G.tolist())
            qf = cm.get_features_from_input(None, None, texts, "gpt2-large", 1024, 0, "q", 64)
            res[key] = div_stats(texts); res[key]["n"] = len(texts); nf += 1
            pend.append(pool.apply_async(mauve_job, ((key, pf, qf),)))
            drain()
            if nf % 20 == 0:
                print(nf, key, f"{time.time()-t0:.0f}s", flush=True); jdump(res, OUT)
    drain(True); pool.close(); jdump(res, OUT); print("done", nf, time.time() - t0)
