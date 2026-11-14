"""閘門 1（CPU，秒級）：已知答案的合成資料，驗證指標與監測器管線。輸出 results/gate1_synth.json（只有數字，判讀由 aggregate.py 依 PREREG 閾值做）。
(a) 可分資料 AUROC≈1、洗牌標籤≈0.5；(b) 指標與 sklearn 一致；
(c) 植入低維逃避：基線 ENS 在觸發下 TPR 崩，ROB 應救回；(d) 無植入時 ROB−ENS≈0。"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_config, EXP, auroc, tpr_at_fpr
import monitors

cfg = load_config()


def make(rng, n, D, L, w, shift=None):
    """每層：正例 = 噪音 + 2.0*w_l，負例 = 噪音；shift：[L,D] 位移加在正例（逃避）。"""
    def side(lab, sh):
        X = rng.normal(size=(n, L, D))
        if lab:
            X = X + 2.0 * w[None]
            if sh is not None:
                X = X + sh[None]
        return X
    return side


def main():
    out = {}
    rng = np.random.default_rng(0)
    # (a) AUROC
    pos = rng.normal(4, 1, 2000); neg = rng.normal(0, 1, 2000)
    out["synth_auroc"] = auroc(pos, neg)
    allv = np.concatenate([pos, neg]); lab = np.r_[np.ones(2000), np.zeros(2000)]
    out["synth_shuffled_auroc"] = auroc(*(lambda p: (allv[p[:2000]], allv[p[2000:]]))(rng.permutation(4000)))
    # 獨立實作對照：O(n²) 暴力 Mann-Whitney
    brute = float(((pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean()))
    out["auroc_vs_sklearn_absdiff"] = abs(brute - out["synth_auroc"])  # 欄名沿用；實為對暴力解的差
    out["tpr_at_1fpr_analytic"] = dict(measured=tpr_at_fpr(pos, neg, 0.01), expected=float(0.5 * (1 + __import__("math").erf((4 - 2.3263479) / 2 ** 0.5))))
    # (c)(d) 植入逃避
    D, L, k = 48, 3, 2
    sc = dict(cfg); sc["probe"] = dict(cfg["probe"], C=0.5, max_iter=200); sc["robust"] = dict(cfg["robust"], energy=0.9, k_cap=8)
    def trial(planted, seed):
        r = np.random.default_rng(seed)
        Uev = np.linalg.qr(r.normal(size=(D, k)))[0]                      # 逃避子空間（各層共用）
        u = [Uev @ r.normal(size=k) for _ in range(L)]; u = [x / np.linalg.norm(x) for x in u]  # 子空間內的單位方向
        W2 = []
        for l in range(L):                                                # 訊號方向 = 0.4*子空間內 + 0.9*子空間外
            o = r.normal(size=D); o -= Uev @ (Uev.T @ o); o /= np.linalg.norm(o)
            W2.append(0.4 * u[l] + 0.9 * o)
        W2 = np.stack(W2)
        shifts = np.stack([-8.0 * u[l] for l in range(L)]) if planted else None   # 觸發時正例沿 -u 被推走
        def samp(n, lab, sh):
            X = r.normal(size=(n, L, D))
            if lab: X = X + 2.0 * W2[None]
            if lab and sh is not None: X = X + sh[None]
            return X
        ntr, nte = 600, 600
        Xtr = np.concatenate([samp(ntr, 1, None), samp(ntr, 0, None)]); ytr = np.r_[np.ones(ntr), np.zeros(ntr)].astype(int)
        # E 概念位移池：在「別的內容」上量到的觸發位移（同一方向 + 噪音）；負例位移較小
        base = shifts if planted else np.zeros((L, D))
        dp = base[None] + 0.3 * r.normal(size=(400, L, D)); dn = 0.3 * r.normal(size=(400, L, D))
        Xte_pos_clean = samp(nte, 1, None)
        Xte_pos = np.stack([samp(nte, 1, shifts)], 1) if planted else np.stack([Xte_pos_clean], 1)  # [n,1,L,D]
        Xte_neg = samp(nte, 0, None)
        per = []
        for l in range(L):
            res, info = monitors.run_layer(Xtr[:, l], ytr, dp[:, l], dn[:, l], Xte_pos[:, :, l], Xte_neg[:, l], sc, seeds=[0, 1], families=["ENS", "ROB"])
            per.append(res)
        ens = monitors.ensemble(per)
        tp = {f: float(np.mean([tpr_at_fpr(ens[f][0][s].reshape(-1), ens[f][1][s], 0.01) for s in range(2)])) for f in ens}
        per_c = []
        for l in range(L):
            res, _ = monitors.run_layer(Xtr[:, l], ytr, dp[:, l], dn[:, l], np.stack([Xte_pos_clean], 1)[:, :, l], Xte_neg[:, l], sc, seeds=[0, 1], families=["ENS", "ROB"])
            per_c.append(res)
        ensc = monitors.ensemble(per_c)
        tc = {f: float(np.mean([tpr_at_fpr(ensc[f][0][s].reshape(-1), ensc[f][1][s], 0.01) for s in range(2)])) for f in ensc}
        return tp, tc
    pl = [trial(True, s) for s in range(3)]; nl = [trial(False, s) for s in range(3)]
    out["planted_ens_trig_tpr"] = float(np.mean([p[0]["ENS"] for p in pl])); out["planted_rob_trig_tpr"] = float(np.mean([p[0]["ROB"] for p in pl]))
    out["planted_rob_minus_ens"] = out["planted_rob_trig_tpr"] - out["planted_ens_trig_tpr"]
    out["planted_clean_tpr_drop"] = float(np.mean([p[1]["ENS"] - p[1]["ROB"] for p in pl]))
    out["null_rob_minus_ens"] = float(np.mean([p[0]["ROB"] - p[0]["ENS"] for p in nl]))
    os.makedirs(os.path.join(EXP, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(EXP, "results", "gate1_synth.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
