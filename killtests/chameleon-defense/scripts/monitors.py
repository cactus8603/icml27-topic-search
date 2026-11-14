"""監測器家族（全是「讀」激活的探測器，無任何逃避訓練）：
  ENS : 每層一個邏輯迴歸（只用乾淨資料），機率對層平均  — 基線（8 層 ensemble）
  ROB : 同 ENS，但訓練集沿「逃避子空間」擴增（本題主張）
  INV : 把逃避子空間投影掉再訓練（投影不變版）
  RND : 與 ROB 相同但用隨機子空間（對照：好處是否來自子空間本身）
逃避子空間由 defense 側的 benign E 概念估計（觸發 T0 減乾淨，逐層、標準化空間），不碰留出概念。"""
import numpy as np
from common import Scaler, fit_logistic, prob, estimate_subspace, sub_idx

FAMILIES = ["ENS", "ROB", "INV", "RND"]


def _aug(Xs, y, dpos, dneg, U, copies, rng, random_sub=False):
    """Xs 已標準化。dpos/dneg：已標準化的 E 概念逃避位移池（正/負例類別條件）。
    位移先投影到子空間 U（rank-k）再加到訓練樣本（標籤不變）。random_sub=True 時改用同維度隨機子空間，係數取自真位移。"""
    D = Xs.shape[1]
    if random_sub:
        Q, _ = np.linalg.qr(rng.normal(size=(D, U.shape[1])))
        Ueff = Q
    else:
        Ueff = U
    outX, outy = [Xs], [y]
    for _ in range(copies):
        X2 = Xs.copy()
        for c, pool in ((1, dpos), (0, dneg)):
            ii = np.where(y == c)[0]
            if len(ii) == 0 or len(pool) == 0:
                continue
            dj = pool[rng.integers(0, len(pool), len(ii))]
            coef = dj @ U                      # 真位移在 U 上的係數
            X2[ii] += coef @ Ueff.T
        outX.append(X2); outy.append(y)
    return np.concatenate(outX), np.concatenate(outy)


def run_layer(Xtr, ytr, dpos, dneg, Xte_pos, Xte_neg, cfg, seeds, perm=False, families=FAMILIES, Xneg_x=None):
    """單一層：回傳 {family: (pos [S,n_pos,C], neg [S,n_neg])} 與 info。
    Xtr [n,D]（乾淨訓練）；dpos/dneg [m,D] 原始空間的 E 概念位移；Xte_pos [n_pos,C,D]；Xte_neg [n_neg,D]。"""
    pc = cfg["probe"]; rc = cfg["robust"]
    sc = Scaler().fit(Xtr)
    Xs = sc(Xtr)
    dpos_s = dpos / sc.sd; dneg_s = dneg / sc.sd
    U, e = estimate_subspace(np.concatenate([dpos_s, dneg_s]), rc["energy"], rc["k_cap"])
    Tpos = sc(Xte_pos.reshape(-1, Xte_pos.shape[-1])); Tneg = sc(Xte_neg)
    # S4（2026-11-14）：觸發後的負例（Xneg_x [n_neg,E,D]），僅回報 FPR，不進裁定
    Tnx = sc(Xneg_x.reshape(-1, Xneg_x.shape[-1])) if Xneg_x is not None else None
    nx = {f: [] for f in families}
    n_pos, C = Xte_pos.shape[:2]
    out = {f: ([], []) for f in families}
    for s in seeds:
        rng = np.random.default_rng(1000 + s)
        ii = sub_idx(len(Xs), pc["subsample_frac"], s, ytr)
        Xi, yi = Xs[ii], ytr[ii].copy()
        if perm:
            yi = rng.permutation(yi)
        for f in families:
            if f == "ENS":
                clf = fit_logistic(Xi, yi, pc["C"], pc["max_iter"], s); P, N = Tpos, Tneg
            elif f in ("ROB", "RND"):
                Xa, ya = _aug(Xi, yi, dpos_s, dneg_s, U, rc["aug_copies"], rng, random_sub=(f == "RND"))
                clf = fit_logistic(Xa, ya, pc["C"], pc["max_iter"], s); P, N = Tpos, Tneg
            elif f == "INV":
                proj = lambda X: X - (X @ U) @ U.T
                clf = fit_logistic(proj(Xi), yi, pc["C"], pc["max_iter"], s); P, N = proj(Tpos), proj(Tneg)
            out[f][0].append(prob(clf, P).reshape(n_pos, C)); out[f][1].append(prob(clf, N))
            if Tnx is not None:
                nx[f].append(prob(clf, proj(Tnx) if f == "INV" else Tnx).reshape(Xneg_x.shape[:2]))
    res = {f: (np.stack(v[0]), np.stack(v[1])) for f, v in out.items()}
    info = dict(k=int(U.shape[1]), energy_top=[float(x) for x in e[:8]])
    if Tnx is not None: info["negx"] = {f: np.stack(v) for f, v in nx.items()}
    return res, info


def ensemble(per_layer):
    """per_layer: list over layers of {family: (pos,neg)} -> {family: (pos_mean,neg_mean)}。"""
    fams = per_layer[0].keys()
    return {f: (np.mean([pl[f][0] for pl in per_layer], 0), np.mean([pl[f][1] for pl in per_layer], 0)) for f in fams}
