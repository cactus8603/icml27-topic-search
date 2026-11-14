"""Full pipeline for one setting. Stages: selfgen, train, merge, devgen, devmauve, nll, select, testgen, testmauve"""
import sys, os, subprocess, time, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
ap = argparse.ArgumentParser(); ap.add_argument("--setting", required=True); ap.add_argument("--stages", nargs="+", required=True)
a = ap.parse_args(); s = a.setting; kind = SETTINGS[s]["kind"]
SC = os.path.dirname(os.path.abspath(__file__)); PY = sys.executable
TF = f"{RES}/timing_{s}.json"; timing = json.load(open(TF)) if os.path.exists(TF) else {}
def run(cmd):
    print("+", cmd, flush=True); r = subprocess.run(cmd, shell=True)
    assert r.returncode == 0, cmd
import re, shutil, glob
def merged(name, path):
    """ensure merged weights exist (merge on demand to save disk); returns True if created here"""
    if name == "base" or os.path.exists(path + "/model.safetensors"): return False
    m = re.match(r"s(\d+)_w([+-][\d.]+)", name); run(f"{PY} {SC}/merge.py --setting {s} --steps {m.group(1)} --ws {float(m.group(2))}"); return True
def cleanup(name, path, created):
    if created and name != "base": shutil.rmtree(path, ignore_errors=True)
def have(n, split, cfgs, seeds, ext):
    return all(os.path.exists(f"{RUNS}/{s}/gen/{n}/{split}/{gname(c)}_s{sd}.{ext}") for c in cfgs for sd in seeds)
EXT = "npy" if kind == "lm" else "npz"
for st in a.stages:
    t0 = time.time()
    if st == "selfgen":
        if kind == "lm": run(f"{PY} {SC}/gen.py --setting {s} --name base --model-dir {SETTINGS[s]['base']} --split selftrain --configs '0.8,0.9,0' --out {RUNS}/{s}/gen_self/selftrain")
        else: run(f"{PY} {SC}/gen.py --setting {s} --name base --model-dir {SETTINGS[s]['base']} --split selftrain --configs default --out {RUNS}/{s}/gen_self/selftrain")
    elif st == "train": run(f"{PY} {SC}/train_selfft.py --setting {s}")
    elif st == "merge": run(f"{PY} {SC}/merge.py --setting {s}")
    elif st == "devgen":
        for n, p in model_list(s):
            if have(n, "dev", grid(), [0], EXT): continue
            cr = merged(n, p); run(f"{PY} {SC}/gen.py --setting {s} --name {n} --model-dir {p} --split dev --configs grid --seeds 0"); cleanup(n, p, cr)
    elif st == "devmauve": run(f"{PY} {SC}/mauve_eval.py --setting {s} --split dev")
    elif st == "nll":
        for n, p in model_list(s):
            cr = merged(n, p); run(f"{PY} {SC}/nll_analysis.py --setting {s} --only {n}"); cleanup(n, p, cr)
    elif st == "select": run(f"{PY} {SC}/select_best.py {s}")
    elif st == "testgen":
        best = json.load(open(f"{RES}/best_{s}.json"))
        for n, p in model_list(s):
            c = best[n]; T, pp, m = c[1:].split("_p")[0], c.split("_p")[1].split("_m")[0], c.split("_m")[1]
            if have(n, "test", [(float(T), float(pp), float(m))], [0, 1, 2], EXT): continue
            cr = merged(n, p); run(f"{PY} {SC}/gen.py --setting {s} --name {n} --model-dir {p} --split test --configs '{T},{pp},{m}' --seeds 0 1 2" + ("" if kind == "lm" else " --save-text")); cleanup(n, p, cr)
    elif st == "testmauve": run(f"{PY} {SC}/mauve_eval.py --setting {s} --split test --only-best {RES}/best_{s}.json")
    else: raise ValueError(st)
    timing[st] = timing.get(st, 0) + time.time() - t0; jdump(timing, TF); print("STAGE", st, "secs", time.time() - t0, flush=True)
