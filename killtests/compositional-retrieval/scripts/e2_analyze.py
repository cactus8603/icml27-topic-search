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
        imgs_k = []   # 每個查詢的 I* 影像 index（群集 bootstrap 的群組；各聚合器共用同一批查詢）
        for name, qs in queries_list:
            owner = z[f"owner_{name}"]
            rows = {(int(o[0]), int(o[1])): r for r, o in enumerate(owner)}
            for n, q in enumerate(qs["queries"][str(k)]):
                qvec = z[f"joint_{name}"][rows[(k, n)]]
                cidx = [cpos[c] for c in q["cats"]]
                imgs_k.append(int(q["istar"]))
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
            ci = bootstrap_ci(err_flags, n_boot, seed + k, groups=imgs_k)   # 以 I* 影像為單位的群集 bootstrap
            r = np.array(per[a]["r"])
            res[a][str(k)] = {"n": int(len(w)), "pair_acc": float(w.mean() * 100), "pair_err": ci,
                              "recall_at_50_mean": float(r.mean() * 100),
                              "err_flags": err_flags.astype(int).tolist(), "istar": imgs_k,
                              "n_clusters": int(len(set(imgs_k)))}
    return res


def _cluster_tables(res_agg, k_hi, k_lo):
    """依 I* 影像建立群集表：回傳 (影像清單, 各影像在 k_hi／k_lo 的錯誤數與查詢數)。同一影像在不同 k 的查詢視為同一群集（跨 k 相依）。"""
    imgs = sorted(set(res_agg[str(k_hi)]["istar"]) | set(res_agg[str(k_lo)]["istar"]))
    pos = {i: n for n, i in enumerate(imgs)}
    out = {}
    for tag, k in (("hi", k_hi), ("lo", k_lo)):
        e = np.zeros(len(imgs)); c = np.zeros(len(imgs))
        for i, f in zip(res_agg[str(k)]["istar"], res_agg[str(k)]["err_flags"]):
            e[pos[i]] += f; c[pos[i]] += 1
        out[tag] = (e, c)
    return imgs, out


def delta_err(res_agg, k_hi, k_lo, n_boot, seed):
    """Δerr = err(k_hi) - err(k_lo)（pp）。群集 bootstrap：以 I* 影像為單位重抽（同一組影像同時決定 k_hi 與 k_lo 的樣本）。"""
    imgs, t = _cluster_tables(res_agg, k_hi, k_lo)
    (eh, ch), (el, cl) = t["hi"], t["lo"]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(imgs), size=(n_boot, len(imgs)))
    with np.errstate(invalid="ignore", divide="ignore"):
        d = (eh[idx].sum(1) / ch[idx].sum(1) - el[idx].sum(1) / cl[idx].sum(1)) * 100
    d = d[np.isfinite(d)]
    lo, hi = np.percentile(d, [2.5, 97.5])
    point = (eh.sum() / ch.sum() - el.sum() / cl.sum()) * 100
    return {"k_hi": k_hi, "k_lo": k_lo, "point": float(point), "lo": float(lo), "hi": float(hi),
            "n_clusters": int(len(imgs)), "cluster": "I* image"}


def dilution_diff(res, a_ref, a_cmp, k_hi, k_lo, n_boot, seed):
    """次要指標（不參與判定）：Δ_{a_ref} − Δ_{a_cmp}（pp），兩聚合器在同一批查詢上配對，同一組群集抽樣。
    例：holistic 與 llr_sum 的稀釋差；> 0 表示 holistic 比逐條件聚合器更被條件數稀釋。"""
    imgs, t1 = _cluster_tables(res[a_ref], k_hi, k_lo)
    _, t2 = _cluster_tables(res[a_cmp], k_hi, k_lo)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(imgs), size=(n_boot, len(imgs)))
    def delta(t, idx=None):
        (eh, ch), (el, cl) = t["hi"], t["lo"]
        if idx is None:
            return eh.sum() / ch.sum() - el.sum() / cl.sum()
        with np.errstate(invalid="ignore", divide="ignore"):
            return eh[idx].sum(1) / ch[idx].sum(1) - el[idx].sum(1) / cl[idx].sum(1)
    d = (delta(t1, idx) - delta(t2, idx)) * 100
    d = d[np.isfinite(d)]
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"ref": a_ref, "cmp": a_cmp, "k_hi": k_hi, "k_lo": k_lo, "point": float((delta(t1) - delta(t2)) * 100),
            "lo": float(lo), "hi": float(hi), "delta_ref_pp": float(delta(t1) * 100), "delta_cmp_pp": float(delta(t2) * 100),
            "note": "次要指標，不參與判定"}


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
        out["secondary_dilution"] = {"holistic_minus_llr_sum": dilution_diff(res, "holistic", "llr_sum", k_top, 1, nb, seed + 98),
                                     "delta_by_agg": {a: delta_err(res[a], k_top, 1, nb, seed + 97) for a in res}}
        out["max_adjacent_drop_pp"] = float(max([0.0] + [errs[i] - errs[i + 1] for i in range(len(errs) - 1)]))
    for a in res:           # 縮小輸出：原始旗標另存
        pass
    jdump(out, args.out or d / f"analysis_{args.model}_{'+'.join(names)}.json")
    print(json.dumps({"n_per_k": ns, "k_top": k_top, "delta": out.get("delta_holistic")}, indent=1))


if __name__ == "__main__":
    main()
