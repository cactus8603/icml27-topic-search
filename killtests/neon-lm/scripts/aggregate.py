import sys, os, glob, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
out = []; summ = {}
def ms(x): x = np.array(x, float); return float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else 0.0
for s in sys.argv[1:]:
    kind = SETTINGS[s]["kind"]; names = [n for n, _ in model_list(s)]; best = json.load(open(f"{RES}/best_{s}.json")); dg = json.load(open(f"{RES}/devgrid_{s}.json"))
    nl = json.load(open(f"{RES}/nll_analysis_{s}.json")) if os.path.exists(f"{RES}/nll_analysis_{s}.json") else {}
    rows = {}
    if kind == "lm":
        M = json.load(open(f"{RES}/mauve_{s}_test.json"))
        for n in names:
            c = best[n]; recs = [M[f"{n}|{c}|{sd}"] for sd in (0, 1, 2) if f"{n}|{c}|{sd}" in M and "mauve" in M[f"{n}|{c}|{sd}"]]
            mm, sm = ms([r["mauve"] for r in recs]); d4, _ = ms([r["distinct4"] for r in recs]); en, _ = ms([r["unigram_entropy_nats"] for r in recs])
            rows[n] = dict(cfg=c, dev_mauve=dg[n][c], test_mauve=mm, test_mauve_std=sm, n_seeds=len(recs), distinct4=d4, entropy=en,
                           nll=nl.get(n, {}).get("nll", {}).get("1.0"), nll_bestT=min(nl[n]["nll"].items(), key=lambda kv: kv[1]) if n in nl else None,
                           analysis=nl.get(n, {}).get("analysis"))
        key_dev = "dev_mauve"
    else:
        for n in names:
            c = best[n]; z = [np.load(f"{RUNS}/{s}/gen/{n}/test/{c}_s{sd}.npz")["corr"] for sd in (0, 1, 2)]
            p1 = [x.mean() * 100 for x in z]; p16 = [x.any(1).mean() * 100 for x in z]
            m1, s1 = ms(p1); m16, s16 = ms(p16)
            rows[n] = dict(cfg=c, dev_pass1=dg[n][c]["pass1"] * 100, dev_pass16=dg[n][c]["pass16"] * 100, pass1=m1, pass1_std=s1, pass16=m16, pass16_std=s16, n_seeds=3,
                           nll=nl.get(n, {}).get("nll", {}).get("1.0"), analysis=nl.get(n, {}).get("analysis"))
        key_dev = "dev_pass16"
    neon = [n for n in names if n != "base" and not any(n.endswith(wtag(w)) for w in W_POS)]
    pos = [n for n in names if any(n.endswith(wtag(w)) for w in W_POS)]
    ref = rows["base"]
    if kind == "lm":
        prim = max(neon, key=lambda n: rows[n]["dev_mauve"]); topt = max(neon, key=lambda n: rows[n]["test_mauve"])
    else:
        ok = [n for n in neon if best["_ref_pass1_base_best_dev"] is not None and dg[n][best[n]]["pass1"] >= best["_ref_pass1_base_best_dev"] - 0.01]
        prim = max(ok or neon, key=lambda n: rows[n]["dev_pass16"]); topt = max([n for n in neon if rows[n]["pass1"] >= ref["pass1"] - 1.0] or neon, key=lambda n: rows[n]["pass16"])
    summ[s] = dict(rows=rows, primary=prim, best_on_test=topt, neon=neon, pos=pos)
    L = [f"### Setting {s} ({os.path.basename(SETTINGS[s]['base'])})", ""]
    if kind == "lm":
        L += ["| model | cfg(dev-best) | dev MAUVE | test MAUVE (mean±std,3 seeds) | distinct-4 | entropy(nats) | NLL T=1 | NLL best-T |", "|---|---|---|---|---|---|---|---|"]
        for n in names:
            r = rows[n]; bt = f"{r['nll_bestT'][1]:.4f} (T={r['nll_bestT'][0]})" if r["nll_bestT"] else "-"
            L.append(f"| {n} | {r['cfg']} | {r['dev_mauve']:.2f} | {r['test_mauve']:.2f}±{r['test_mauve_std']:.2f} | {r['distinct4']:.4f} | {r['entropy']:.3f} | {r['nll'] if r['nll'] is None else round(r['nll'],4)} | {bt} |")
    else:
        L += ["| model | cfg(dev-selected) | dev pass@1 | dev pass@16 | test pass@1 (mean±std) | test pass@16 (mean±std) | NLL gold-sol T=1 |", "|---|---|---|---|---|---|---|"]
        for n in names:
            r = rows[n]; L.append(f"| {n} | {r['cfg']} | {r['dev_pass1']:.1f} | {r['dev_pass16']:.1f} | {r['pass1']:.2f}±{r['pass1_std']:.2f} | {r['pass16']:.2f}±{r['pass16_std']:.2f} | {r['nll'] if r['nll'] is None else round(r['nll'],4)} |")
    L += ["", f"Primary Neon model (selected by dev): {prim}; best-on-test Neon (optimistic): {topt}", ""]
    # criteria
    P = rows[prim]; 
    if kind == "lm":
        d = P["test_mauve"] - ref["test_mauve"]; thr = P["test_mauve_std"] + ref["test_mauve_std"]
        L.append(f"- Criterion 1 ({prim} vs base): dMAUVE = {d:+.2f}; sum of stds = {thr:.2f}; >=2.0: {d>=2.0}; > sum std: {d>thr}")
        dt = rows[topt]["test_mauve"] - ref["test_mauve"]; thr2 = rows[topt]["test_mauve_std"] + ref["test_mauve_std"]
        L.append(f"  (best-on-test {topt}: dMAUVE = {dt:+.2f}, sum std {thr2:.2f})")
        if ref["nll_bestT"]:
            bb = ref["nll_bestT"]
            L.append(f"- Criterion 2: base NLL at best T ({bb[0]}) = {bb[1]:.4f}; {prim} NLL(T=1) = {P['nll']:.4f} (diff {P['nll']-bb[1]:+.4f}); {prim} NLL best-T = {P['nll_bestT'][1]:.4f} (T={P['nll_bestT'][0]}); min Neon NLL(T=1) over all w>0 models = {min(rows[n]['nll'] for n in neon):.4f}")
    else:
        d = P["pass16"] - ref["pass16"]; thr = P["pass16_std"] + ref["pass16_std"]; d1 = P["pass1"] - ref["pass1"]
        L.append(f"- Criterion 3 ({prim} vs base): dpass@1 = {d1:+.2f} (need >= -1.0: {d1>=-1.0}); dpass@16 = {d:+.2f}; >=2: {d>=2}; > sum std {thr:.2f}: {d>thr}")
        dt = rows[topt]["pass16"] - ref["pass16"]; L.append(f"  (best-on-test under constraint {topt}: dpass@1 = {rows[topt]['pass1']-ref['pass1']:+.2f}, dpass@16 = {dt:+.2f}, sum std {rows[topt]['pass16_std']+ref['pass16_std']:.2f})")
    L.append("- Criterion 4 (positive controls, w<0):")
    for n in pos:
        r = rows[n]; L.append(f"  - {n}: " + (f"test MAUVE {r['test_mauve']:.2f} (d {r['test_mauve']-ref['test_mauve']:+.2f}), NLL {r['nll']:.4f}" if kind == 'lm' else f"pass@1 {r['pass1']:.2f} (d {r['pass1']-ref['pass1']:+.2f}), pass@16 {r['pass16']:.2f} (d {r['pass16']-ref['pass16']:+.2f})"))
    L += ["", "- Criterion 5 (per-position temperature fit; T_i = best global temperature on base logits to match each model):", "", "| model | n_pos | T mean | T std | q05 | q25 | q50 | q75 | q95 | KL raw | KL after fit | frac left |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n in names[1:]:
        an = rows[n]["analysis"]
        if an: L.append(f"| {n} | {an['n_pos']} | {an['T_mean']:.3f} | {an['T_std']:.3f} | " + " | ".join(f"{an['T_q'][k]:.3f}" for k in ["q05","q25","q50","q75","q95"]) + f" | {an['KL_raw_mean']:.4f} | {an['KL_fit_mean']:.4f} | {an['KL_fit_frac_of_raw']:.3f} |")
    L.append(""); out += L
open(f"{RES}/tables.md", "w").write("\n".join(out)); jdump(summ, f"{RES}/summary.json"); print("\n".join(out))
