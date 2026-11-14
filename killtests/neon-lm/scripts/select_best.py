import sys, os, glob, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
setting = sys.argv[1]; kind = SETTINGS[setting]["kind"]
names = [n for n, _ in model_list(setting)]
best = {}; table = {}
if kind == "lm":
    R = json.load(open(f"{RES}/mauve_{setting}_dev.json"))
    for n in names:
        rows = {}
        for k, v in R.items():
            if k.startswith(n + "|") and "mauve" in v:
                _, cfg, seed = k.split("|"); rows.setdefault(cfg, []).append(v["mauve"])
        sc = {c: float(np.mean(x)) for c, x in rows.items()}; assert len(sc) == len(grid()), (n, len(sc))
        best[n] = max(sc, key=sc.get); table[n] = sc
else:
    pass1 = {}; pass16 = {}
    for n in names:
        for f in glob.glob(f"{RUNS}/{setting}/gen/{n}/dev/*.npz"):
            cfg = os.path.basename(f)[:-4].rsplit("_s", 1)[0]; z = np.load(f)["corr"]
            pass1.setdefault(n, {}).setdefault(cfg, []).append(z.mean()); pass16.setdefault(n, {}).setdefault(cfg, []).append(z.any(1).mean())
        assert len(pass1[n]) == len(grid()), (n, len(pass1[n]))
    p1 = {n: {c: float(np.mean(v)) for c, v in pass1[n].items()} for n in names}
    p16 = {n: {c: float(np.mean(v)) for c, v in pass16[n].items()} for n in names}
    ref = max(p1["base"].values())  # base's best pass@1 on dev
    for n in names:
        ok = [c for c in p1[n] if p1[n][c] >= ref - 0.01]
        if ok: best[n] = max(ok, key=lambda c: (p16[n][c], p1[n][c]))
        else: best[n] = max(p1[n], key=p1[n].get)  # fallback: constraint unsatisfiable
        table[n] = {c: dict(pass1=p1[n][c], pass16=p16[n][c]) for c in p1[n]}
    best["_ref_pass1_base_best_dev"] = ref
    best["_constraint_unsatisfied"] = [n for n in names if not [c for c in p1[n] if p1[n][c] >= ref - 0.01]]
jdump(best, f"{RES}/best_{setting}.json"); jdump(table, f"{RES}/devgrid_{setting}.json")
print(json.dumps(best, indent=1))
