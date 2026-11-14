"""w→∞ 的描述性診斷、oracle 誤差間隙放大探針與第 2 週判讀用語。見 PREREG §8 的修訂。"""
import numpy as np


def gamma_curves(pcg, mask_idx):
    """pcg [pairs,E,T,3(S_i,S_j,test)(,1)]（eval_hier_oracle.py --gamma 的 pair_cos_gamma）
    -> [pairs,E]：與條文 1 同一套：兩份訓練輸入平均、逐 t 除以自身跨 checkpoint 平均（0 或非有限 -> nan）、同一份 t 遮罩等權平均。"""
    a = np.asarray(pcg)
    if a.ndim == 5: a = a[..., 0]
    y = a[:, :, :, :2].mean(axis=3)[:, :, mask_idx]
    m = y.mean(axis=1, keepdims=True); bad = ~np.isfinite(m) | (m == 0)
    with np.errstate(all="ignore"):
        return (y / np.where(bad, np.nan, m)).mean(axis=2)


def h_inf(ep, cond_curves, gamma_curve_arr, pairs, R, tstar, interior, max_invalid, thr, n_boot=2000, seed=0):
    """cond_curves / gamma_curve_arr: [pairs,E]。有效對 = cond 與 γ 兩條主曲線都有內部最低點。
    H∞ = 有效對 log[t*_γ / t*(w=0)] 的等權平均；CI 為 m_i·m_j 加權 bootstrap（只描述）。
    `thr` 是標示極限曲線是否提前達標的參考值。有限 guidance 可有不同最低點。"""
    P = len(pairs); valid, x, info = [], np.full(P, np.nan), []
    for p in range(P):
        oc, rc = interior(ep, cond_curves[p]); og, rg = interior(ep, gamma_curve_arr[p]); v = bool(oc and og)
        info.append({"pair": [int(i) for i in pairs[p]], "valid": v, "reason": {k: r for k, (o, r) in (("cond", (oc, rc)), ("gamma", (og, rg))) if not o}})
        if v: valid.append(p); x[p] = np.log(tstar(ep, gamma_curve_arr[p]) / tstar(ep, cond_curves[p]))
    n_inv = P - len(valid); out = dict(n_pairs=P, n_valid=len(valid), n_invalid=n_inv, max_invalid=int(max_invalid), invalid_pairs=info,
                                       per_pair=[None if not np.isfinite(v) else float(v) for v in x], threshold=float(thr))
    if n_inv > max_invalid or not valid:
        out.update(state="極限曲線未能判讀", h_inf=(float(x[valid].mean()) if valid else None), ci95=None, bootstrap_skipped=None); return out
    h = float(x[valid].mean())
    rng = np.random.default_rng(seed); bs = []; sk = 0; pi, pj = np.asarray(pairs)[:, 0], np.asarray(pairs)[:, 1]
    for _ in range(n_boot):
        m = np.bincount(rng.integers(0, R, R), minlength=R).astype(float); w = m[pi] * m[pj]; den = w[valid].sum()
        if den == 0: sk += 1; continue
        bs.append(float((w[valid] * x[valid]).sum() / den))
    out.update(h_inf=h, ci95=[float(q) for q in np.percentile(bs, [2.5, 97.5])] if bs else None, bootstrap_skipped=sk,
               state="極限提前達標" if h <= thr else "極限提前未達標")
    return out


# ---------------- 條文 8：oracle 偏差間隙放大比 ----------------
def oracle_errors(mse, e_idx, t_idx):
    """取各模型對自身訓練輸入和共用留出輸入的原始 x0 MSE，shape [R,T',B]。"""
    errors = np.asarray(mse, dtype=float)[:, e_idx, :, :, :]
    errors = errors[:, t_idx, :, :]
    own = np.stack([errors[j, :, j, :] for j in range(len(errors))])
    return own, errors[:, :, -1, :]


def gap_terms(mse, e_idx, t_idx, br_cond, br_cfg):
    """正間隙表示自身訓練輸入比留出輸入更接近 BP oracle。
    mse: [R,E,T,S,B]，run j 的自身訓練輸入是第 j 格，留出輸入是第 S−1 格。
    回傳 (g_cfg [R,T'], g_cond [R,T'])，以同一條件分支的留出 MSE 正規化。"""
    own, test = oracle_errors(mse, e_idx, t_idx)
    norm = test[:, :, br_cond]
    norm = np.where(np.isfinite(norm) & (norm > 0), norm, np.nan)
    gaps = (test - own) / norm[:, :, None]
    return gaps[:, :, br_cfg], gaps[:, :, br_cond]


def percentile_interval(samples, level):
    if not len(samples):
        return [None, None]
    tail = 50 * (1 - level)
    return np.percentile(samples, [tail, 100 - tail]).tolist()


def membership_gap(own, test, n_boot, seed, ci_level, min_gap, fail_upper):
    """G3：差值總和除以留出誤差總和，重抽整個模型及其所有 t。"""
    out = {"measure": "CFG3 membership gap proxy", "state": "未能測試", "ci_level": ci_level}
    if not (np.isfinite(own).all() and np.isfinite(test).all()
            and (own >= 0).all() and (test >= 0).all()):
        return {**out, "reason": "CFG3 的原始 x0 MSE 含非有限值或負值"}
    numerator = (test - own).sum(axis=1)
    denominator = test.sum(axis=1)
    total_test = float(denominator.sum())
    if not np.isfinite(total_test) or total_test <= 0:
        return {**out, "reason": "CFG3 留出誤差總和非有限正數"}

    draws = np.random.default_rng(seed).integers(0, len(own), (n_boot, len(own)))
    draw_test = denominator[draws].sum(axis=1)
    draw_gap = numerator[draws].sum(axis=1)
    gap = float(numerator.sum() / total_test)
    out.update(g3=gap, heldout_mse_sum=total_test, raw_gap_sum=float(numerator.sum()))
    if not np.isfinite(gap) or (draw_test <= 0).any() or not np.isfinite(draw_test).all() or not np.isfinite(draw_gap).all():
        return {**out, "reason": "G3 點估計或重抽未定義"}
    lo, hi = percentile_interval(draw_gap / draw_test, ci_level)
    out["ci"] = [lo, hi]
    if lo is None:
        return {**out, "reason": "重抽沒有可用區間"}
    if gap >= min_gap and lo > 0:
        out["state"] = "PASS"
    elif hi < fail_upper:
        out["state"] = "FAIL"
    else:
        out["state"] = "灰區"
    return out


def gamma_ratio(g_cfg, g_cond, runs=None):
    """Γ = Σ_j Σ_t g̃_cfg3 / Σ_j Σ_t g̃_cond；回傳 (Γ, 分母)。僅有限且為正的分母可用。"""
    runs = list(range(g_cfg.shape[0])) if runs is None else list(runs)
    num = float(g_cfg[runs].sum()); den = float(g_cond[runs].sum())
    if not np.isfinite(den):
        return None, None
    if den <= 0 or not np.isfinite(num):
        return None, den
    ratio = num / den
    return (ratio if np.isfinite(ratio) else None), den


def probe_amplification(g6, c6, picks6, g1, c1, picks1, gamma_min, n_boot, seed, skip_max, ci_level=0.95):
    """條文 8 的判定。g*/c* 為 gap_terms 輸出；picks 為各 run 的區塊編號。
    Γ6 用 K=6 全部 run；Γ1 用 K=1 run（只取與 K=6 相同的 pick）。log Γ6 − log Γ1 以區塊為單位配對 bootstrap（只用共同 pick）。
    點估計或重抽的分母與 Γ 須為有限正數。無效重抽比例 > skip_max 或點估計無效時記未能測試。
    PASS：Γ6 >= gamma_min 且 log Γ6 − log Γ1 的區間下界 > 0。"""
    common = [p for p in picks1 if p in picks6]
    i6 = {p: picks6.index(p) for p in common}; i1 = {p: picks1.index(p) for p in common}
    G6, d6 = gamma_ratio(g6, c6)                                         # Γ6：K=6 全部 run
    G6c, d6c = gamma_ratio(g6, c6, [i6[p] for p in common])              # 共同 pick 上的 Γ6（配對差用）
    G1, d1 = gamma_ratio(g1, c1, [i1[p] for p in common])
    out = dict(gamma6=G6, gamma6_common_picks=G6c, gamma1=G1, denominators=dict(k6=d6, k6_common=d6c, k1=d1), common_picks=common)
    if not all(g is not None and g > 0 for g in (G6, G6c, G1)):
        out.update(state="未能測試", reason="點估計的分母與 Γ 須為有限正數"); return out
    rng = np.random.default_rng(seed); ds = []; sk = 0; n = len(common)
    for _ in range(n_boot):
        draw = rng.integers(0, n, n)
        a6, a1 = gamma_ratio(g6, c6, [i6[common[k]] for k in draw])[0], gamma_ratio(g1, c1, [i1[common[k]] for k in draw])[0]
        if a6 is None or a1 is None or a6 <= 0 or a1 <= 0: sk += 1; continue
        ds.append(np.log(a6) - np.log(a1))
    out["bootstrap_skipped_fraction"] = sk / n_boot
    if sk / n_boot > skip_max or not ds:
        out.update(state="未能測試", reason=f"bootstrap 跳過比例 {sk / n_boot:.3f} > {skip_max}"); return out
    lo, hi = percentile_interval(ds, ci_level)
    out.update(diff_log_gamma=float(np.log(G6c) - np.log(G1)), diff_ci=[lo, hi], diff_ci_level=ci_level,
               state="PASS" if (G6 >= gamma_min and lo > 0) else "未通過")
    if ci_level == 0.95:
        out["diff_ci95"] = out["diff_ci"]
    return out


# ---------------- 條文 9：層級用語 ----------------
def tier_text(probe_state):
    """(c) 落在 FAIL 帶時，只判這次測得的 w=3 起點與間隙放大。"""
    amp = {"PASS": "本設定的 oracle 間隙放大獲支持", "未通過": "放大探針未達門檻", "未能測試": "放大探針未能測試"}[probe_state]
    return f"此設定的 w=3 起點結果落在 FAIL 帶（實作／劑量層）。{amp}。"
