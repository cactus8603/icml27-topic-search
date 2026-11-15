"""第二探針跑前的 CPU 預檢（描述用；用已存的 results/ht/*.npz，不需 GPU）：
 (a) holistic 的成對正確率 P(s(A+)>s(A-)) = P(cos(I,full) > cos(I,half))，兩模型 x 三子集；
 (b) 邊際 PMI（llr_sum，單元 PMI，池 = 1379 張）在「最佳全域門檻 t」下的 COR 上限：
     COR(t) = P(PMI(d+) > t > PMI(d-))（anchor 項抵銷的恆等式；PREREG §3 為 t=0），在資料上掃 t 取最大。只描述。
輸出 results/probe2/precheck.json。
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import EXP, SUBSETS, jdump, pmi
from analyze_ht import pool_all

MODELS = ["clip_b32", "cs_clip"]


def best_threshold_cor(pp, pm):
    """max_t mean[(pp > t) & (t > pm)]（嚴格大於）。候選 t = 所有 pp、pm 值之間的中點與 0。回傳 (best_cor_pct, t)。"""
    v = np.unique(np.r_[pp, pm, 0.0])
    cands = np.r_[v[0] - 1, (v[:-1] + v[1:]) / 2, v[-1] + 1]
    best, bt = -1.0, None
    for t in cands:
        c = float(((pp > t) & (t > pm)).mean() * 100)
        if c > best:
            best, bt = c, float(t)
    return best, bt


def main():
    d = EXP / "results" / "ht"
    out = {"_note": "描述用預檢；不參與任何判定。", "holistic_pairwise_acc_pct": {}, "marginal_pmi_best_threshold": {}}
    for m in MODELS:
        z = {s: dict(np.load(d / f"{m}_{s}.npz", allow_pickle=False)) for s in SUBSETS}
        pool = pool_all(z)
        for s in SUBSETS:
            zz = z[s]
            I, tau = zz["img"], float(zz["tau"])
            cf, ch = np.einsum("nd,nd->n", I, zz["full"]), np.einsum("nd,nd->n", I, zz["half"])
            ca = np.einsum("nd,nd->n", I, zz["anchor"])
            out["holistic_pairwise_acc_pct"][f"{m}/{s}"] = {
                "pairwise_full_gt_half": float((cf > ch).mean() * 100),
                "holistic_cor": float(((cf > ca) & (ca > ch)).mean() * 100), "n": int(len(I))}
            pp, pm = pmi(I, zz["dplus"], pool, tau), pmi(I, zz["dminus"], pool, tau)
            c0 = float(((pp > 0) & (0 > pm)).mean() * 100)
            cb, tb = best_threshold_cor(pp, pm)
            out["marginal_pmi_best_threshold"][f"{m}/{s}"] = {
                "cor_at_t0": c0, "best_cor_upper_bound": cb, "best_t": tb,
                "p_pmi_dplus_gt_dminus": float((pp > pm).mean() * 100), "n": int(len(I))}
    jdump(out, EXP / "results" / "probe2" / "precheck.json")
    print(__import__("json").dumps(out, indent=1))


if __name__ == "__main__":
    main()
