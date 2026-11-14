"""E2 的 CPU 分析：由 embedding 算各聚合器的 pairwise 正確率 A(k) 與 recall@50，附 bootstrap。
用法：python scripts/e2_analyze.py --model clip_b32 [--with-addon] --out results/e2/analysis_clip_b32.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import EXP, bootstrap_ci, jdump, logmeanexp, params

AGGS = ["holistic", "unit_mean", "unit_min", "llr_sum"]


def unit_tables(z):
    """回傳 cos[c, I] 與 pmi[c, I]，c 為類別索引（依 cat_ids 排序）。"""
    img, unit, tau = z["img"], z["unit"], float(z["tau"])
    cos = unit @ img.T                                    # (80, N)
    pmi = tau * cos - logmeanexp(tau * cos, axis=1)[:, None]
    return cos, pmi


def scores_for(agg, z, tabs, qvec, cat_idx, img_idx):
    """對一個查詢，回傳指定影像 index 陣列上的分數。"""
    if agg == "holistic":
        return z["img"][img_idx] @ qvec
    cos, pmi = tabs
    if agg == "unit_mean":
        return cos[np.ix_(cat_idx, img_idx)].mean(axis=0)
    if agg == "unit_min":
        return cos[np.ix_(cat_idx, img_idx)].min(axis=0)
    if agg == "llr_sum":
        return pmi[np.ix_(cat_idx, img_idx)].sum(axis=0)
    raise ValueError(agg)


def run(z, queries_list, P, n_boot, seed, aggs=AGGS):
    """queries_list: [(name, queries_json)]；name 對應 z['joint_<name>']。"""
    cat_ids = [int(c) for c in z["cat_ids"]]
    cpos = {c: n for n, c in enumerate(cat_ids)}
    tabs = unit_tables(z)
    N = z["img"].shape[0]
    all_idx = np.arange(N)
    res = {a: {} for a in aggs}
    for k in P["k_list"]:
        per = {a: {"win": [], "r": []} for a in aggs}
        for name, qs in queries_list:
            owner = z[f"owner_{name}"]
            rows = {(int(o[0]), int(o[1])): r for r, o in enumerate(owner)}
            for n, q in enumerate(qs["queries"][str(k)]):
                qvec = z[f"joint_{name}"][rows[(k, n)]]
                cidx = [cpos[c] for c in q["cats"]]
                for a in aggs:
                    s_star, s_j = scores_for(a, z, tabs, qvec, cidx, np.array([q["istar"], q["j"]]))
                    per[a]["win"].append(bool(s_star > s_j))
                    full = scores_for(a, z, tabs, qvec, cidx, all_idx)
                    top = np.argsort(-full, kind="stable")[: P["recall_at"]]
                    rel = np.array(q["rel"])
                    hit = np.isin(top, rel).sum()
                    per[a]["r"].append(hit / min(P["recall_at"], len(rel)))
        for a in aggs:
            w = np.array(per[a]["win"], float)
            if len(w) == 0:
                res[a][str(k)] = {"n": 0}
                continue
            err_flags = 1.0 - w                                    # 平手視為錯誤（嚴格大於才算贏）
            ci = bootstrap_ci(err_flags, n_boot, seed + k)
            r = np.array(per[a]["r"])
            res[a][str(k)] = {"n": int(len(w)), "pair_acc": float(w.mean() * 100), "pair_err": ci,
                              "recall_at_50_mean": float(r.mean() * 100),
                              "err_flags": err_flags.astype(int).tolist()}
    return res


def delta_err(res_agg, k_hi, k_lo, n_boot, seed):
    """Δerr = err(k_hi) - err(k_lo)（pp），兩組查詢獨立，各自 bootstrap。"""
    a = np.array(res_agg[str(k_hi)]["err_flags"], float)
    b = np.array(res_agg[str(k_lo)]["err_flags"], float)
    rng = np.random.default_rng(seed)
    ia = rng.integers(0, len(a), size=(n_boot, len(a)))
    ib = rng.integers(0, len(b), size=(n_boot, len(b)))
    d = (a[ia].mean(axis=1) - b[ib].mean(axis=1)) * 100
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"k_hi": k_hi, "k_lo": k_lo, "point": float((a.mean() - b.mean()) * 100), "lo": float(lo), "hi": float(hi)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--with-addon", action="store_true")
    ap.add_argument("--dir", default=str(EXP / "results" / "e2"))
    ap.add_argument("--out")
    args = ap.parse_args()
    p = params()
    P, nb, seed = p["p2"], p["gate_g3"]["n_boot"], p["gate_g3"]["boot_seed"]
    d = Path(args.dir)
    z = np.load(d / f"emb_{args.model}.npz")
    names = ["main"] + (["addon"] if args.with_addon else [])
    ql = [(n, json.loads((d / f"queries_{n}.json").read_text())) for n in names]
    res = run(z, ql, P, nb, seed)
    ns = {k: res["holistic"][str(k)]["n"] for k in P["k_list"]}
    k_ok = [k for k in P["k_list"] if ns[k] >= P["min_n_per_k"]]
    k_top = max(k_ok) if k_ok else None
    out = {"model": args.model, "queries": names, "n_per_k": ns, "k_top": k_top, "per_agg": res}
    if k_top and k_top >= P["min_k_top"]:
        out["delta_holistic"] = delta_err(res["holistic"], k_top, 1, nb, seed + 99)
        errs = [res["holistic"][str(k)]["pair_err"]["point"] for k in P["k_list"] if ns[k] >= P["min_n_per_k"]]
        out["holistic_err_series_pp"] = errs
        out["max_adjacent_drop_pp"] = float(max([0.0] + [errs[i] - errs[i + 1] for i in range(len(errs) - 1)]))
    for a in res:           # 縮小輸出：原始旗標另存
        pass
    jdump(out, args.out or d / f"analysis_{args.model}_{'+'.join(names)}.json")
    print(json.dumps({"n_per_k": ns, "k_top": k_top, "delta": out.get("delta_holistic")}, indent=1))


if __name__ == "__main__":
    main()
