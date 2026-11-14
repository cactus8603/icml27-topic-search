"""E0/E1 的 CPU 分析：由 extract_ht.py 存的 npz 計算 COR／HT（全句，「holistic」）與各聚合器（單元：anchor 與新增細節）的 COR，附 bootstrap。
聚合器（PREREG §3）：
  holistic  整句 cosine（= Half-Truths 官方評分）
  unit_mean 單元 cosine 的平均（A 只有 anchor 單元；A+ 為 [anchor, d+]；A- 為 [anchor, d-]）
  unit_min  單元 cosine 的最小值
  unit_sum  單元 cosine 之和（未校準）
  llr_sum   單元 PMI 之和（PMI 定義見 common.pmi；pool=all_ht_images 為主、same_subset 為探索性）
用法：python scripts/analyze_ht.py --model clip_b32 [--dir results/ht] [--tag ""] [--out ...]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (EXP, SUBSETS, bootstrap_ci, cor_flags, ht_flags, jdump, paired_bootstrap_diff, params, pmi)

AGGS = ["holistic", "unit_mean", "unit_min", "unit_sum", "llr_sum", "llr_sum_samesubset"]


def load(d, model, sub, tag=""):
    return dict(np.load(d / f"{model}_{sub}{tag}.npz", allow_pickle=False))


def pool_all(z_by_sub):
    """所有子集的唯一影像 embedding（依 sha 去重）。不使用任何正確性標籤。"""
    seen, vecs = set(), []
    for sub in SUBSETS:
        if sub not in z_by_sub:
            continue
        z = z_by_sub[sub]
        for sha, v in zip(z["sha"], z["img"]):
            if sha not in seen:
                seen.add(sha)
                vecs.append(v)
    return np.stack(vecs)


def cos(a, b):
    return np.einsum("nd,nd->n", a, b)


def scores(z, pool_all_emb):
    """回傳 {agg: (s_full, s_anchor, s_half)}（逐列分數）。"""
    I, tau = z["img"], float(z["tau"])
    ca, cd_p, cd_m = cos(I, z["anchor"]), cos(I, z["dplus"]), cos(I, z["dminus"])
    cf, ch = cos(I, z["full"]), cos(I, z["half"])
    out = {"holistic": (cf, ca, ch),
           "unit_mean": ((ca + cd_p) / 2, ca, (ca + cd_m) / 2),
           "unit_min": (np.minimum(ca, cd_p), ca, np.minimum(ca, cd_m)),
           "unit_sum": (ca + cd_p, ca, ca + cd_m)}
    for name, pool in [("llr_sum", pool_all_emb), ("llr_sum_samesubset", np.unique(I, axis=0))]:
        pa, pp, pm = pmi(I, z["anchor"], pool, tau), pmi(I, z["dplus"], pool, tau), pmi(I, z["dminus"], pool, tau)
        out[name] = (pa + pp, pa, pa + pm)
    return out


def summarize(flags_cor, flags_ht, ttype, nb, seed):
    res = {}
    for kind in ("all", "entity", "relation"):
        m = np.ones(len(flags_cor), bool) if kind == "all" else (ttype == kind)
        res[kind] = {"cor": bootstrap_ci(flags_cor[m], nb, seed), "ht": bootstrap_ci(flags_ht[m], nb, seed + 1)}
    return res


def by_foil(flags_cor, foil):
    return {str(f): {"n": int((foil == f).sum()), "cor": float(flags_cor[foil == f].mean() * 100)} for f in np.unique(foil)}


def analyze(model, d, tag=""):
    p = params()
    nb, seed = p["gate_g3"]["n_boot"], p["gate_g3"]["boot_seed"]
    z = {s: load(d, model, s, tag) for s in SUBSETS if (d / f"{model}_{s}{tag}.npz").exists()}
    pool = pool_all(z)
    out = {"model": model, "tag": tag, "pool_size_all_ht_images": int(len(pool)), "subsets": {}}
    for sub, zz in z.items():
        sc = scores(zz, pool)
        flags, rec = {}, {}
        for agg, (sf, sa, sh) in sc.items():
            fc, fh = cor_flags(sf, sa, sh), ht_flags(sa, sh)
            flags[agg] = fc
            rec[agg] = summarize(fc, fh, zz["triplet_type"], nb, seed)
            if "foil_type" in zz:
                rec[agg]["by_foil_type"] = by_foil(fc, zz["foil_type"])
            rec[agg]["row_cor_flags"] = fc.astype(int).tolist()
        for agg in AGGS:
            if agg != "holistic":
                rec[agg]["paired_vs_holistic_cor_pp"] = paired_bootstrap_diff(flags[agg], flags["holistic"], nb, seed + 2)
        nonempty = np.array([len(t) > 0 for t in zz["dplus_text"]]) & np.array([len(t) > 0 for t in zz["dminus_text"]])
        cover = {
            "n": int(len(zz["img"])),
            "detail_plus_ok": float(np.mean(zz["ok_plus"])),
            "detail_minus_ok": float(np.mean(zz["ok_minus"])),
            "detail_nonempty_both": float(nonempty.mean()),
            "coverage": float(np.mean(zz["ok_plus"] & zz["ok_minus"] & nonempty)),
        }
        out["subsets"][sub] = {"aggs": rec, "detail_coverage": cover}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dir", default=str(EXP / "results" / "ht"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--out")
    args = ap.parse_args()
    d = Path(args.dir)
    out = analyze(args.model, d, args.tag)
    jdump(out, args.out or d / f"analysis_{args.model}{args.tag}.json")
    for sub, v in out["subsets"].items():
        print(sub, {a: round(v["aggs"][a]["all"]["cor"]["point"], 1) for a in AGGS},
              "coverage", round(v["detail_coverage"]["coverage"], 3))


if __name__ == "__main__":
    main()
