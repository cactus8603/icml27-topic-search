"""可達性上界（w→∞ 極限；DECISIONS_fable 條文 7）與 oracle 偏差間隙放大比探針（條文 8）、層級用語（條文 9）的共用函式。
aggregate_week1.py（H∞，探索性）與 aggregate_week2.py（正式）共用，避免兩份實作漂移。
2026-11-14，代理落差稽核後、結果前定案；非探索性（第 1 週的 H∞ 只算探索性）。"""
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
    """條文 7。cond_curves / gamma_curve_arr: [pairs,E]。有效對 = cond 與 γ 兩條主曲線都有內部最低點。
    H∞ = 有效對 log[t*_γ / t*(w=0)] 的等權平均；CI 為 m_i·m_j 加權 bootstrap（只描述）。
    判定只用點估計：無效對數 > max_invalid（⌊P/3⌋）-> 可達性未知；H∞ <= thr（log 0.8）-> 可達；否則 構造不可達。"""
    P = len(pairs); valid, x, info = [], np.full(P, np.nan), []
    for p in range(P):
        oc, rc = interior(ep, cond_curves[p]); og, rg = interior(ep, gamma_curve_arr[p]); v = bool(oc and og)
        info.append({"pair": [int(i) for i in pairs[p]], "valid": v, "reason": {k: r for k, (o, r) in (("cond", (oc, rc)), ("gamma", (og, rg))) if not o}})
        if v: valid.append(p); x[p] = np.log(tstar(ep, gamma_curve_arr[p]) / tstar(ep, cond_curves[p]))
    n_inv = P - len(valid); out = dict(n_pairs=P, n_valid=len(valid), n_invalid=n_inv, max_invalid=int(max_invalid), invalid_pairs=info,
                                       per_pair=[None if not np.isfinite(v) else float(v) for v in x], threshold=float(thr))
    if n_inv > max_invalid or not valid:
        out.update(state="可達性未知", h_inf=(float(x[valid].mean()) if valid else None), ci95=None, bootstrap_skipped=None); return out
    h = float(x[valid].mean())
    rng = np.random.default_rng(seed); bs = []; sk = 0; pi, pj = np.asarray(pairs)[:, 0], np.asarray(pairs)[:, 1]
    for _ in range(n_boot):
        m = np.bincount(rng.integers(0, R, R), minlength=R).astype(float); w = m[pi] * m[pj]; den = w[valid].sum()
        if den == 0: sk += 1; continue
        bs.append(float((w[valid] * x[valid]).sum() / den))
    out.update(h_inf=h, ci95=[float(q) for q in np.percentile(bs, [2.5, 97.5])] if bs else None, bootstrap_skipped=sk,
               state="可達" if h <= thr else "構造不可達")
    return out


# ---------------- 條文 8：oracle 偏差間隙放大比 ----------------
def gap_terms(mse, e_idx, t_idx, br_cond, br_cfg):
    """mse: mse_oracle [R,E,T,S,B]；S 的最後一格是測試輸入，run j 的自身訓練輸入為第 j 格。
    g̃_b(j,t) = [mse(自身訓練輸入) - mse(測試輸入)] / mse(測試輸入, cond)，皆取在 e_idx。回傳 (g_cfg [R,T'], g_cond [R,T'])。"""
    R = mse.shape[0]; S = mse.shape[3]
    own = np.stack([mse[j, e_idx, :, j, :] for j in range(R)])        # [R,T,B]
    test = mse[:, e_idx, :, S - 1, :]                                    # [R,T,B]
    den = test[:, :, br_cond]
    sel = list(t_idx)
    g = lambda b: ((own[:, :, b] - test[:, :, b]) / den)[:, sel]
    return g(br_cfg), g(br_cond)


def gamma_ratio(g_cfg, g_cond, runs=None):
    """Γ = Σ_j Σ_t g̃_cfg3 / Σ_j Σ_t g̃_cond；回傳 (Γ, 分母)。分母 <= 0 -> Γ=None。"""
    runs = list(range(g_cfg.shape[0])) if runs is None else list(runs)
    num = float(g_cfg[runs].sum()); den = float(g_cond[runs].sum())
    return (None if den <= 0 else num / den), den


def probe_amplification(g6, c6, picks6, g1, c1, picks1, gamma_min, n_boot, seed, skip_max, ci_level=0.95):
    """條文 8 的判定。g*/c* 為 gap_terms 輸出；picks 為各 run 的區塊編號。
    Γ6 用 K=6 全部 run；Γ1 用 K=1 run（只取與 K=6 相同的 pick）。log Γ6 − log Γ1 以區塊為單位配對 bootstrap（只用共同 pick）。
    分母 <= 0 的重抽跳過（Γ6 或 Γ1 <= 0 使 log 無定義，亦跳過並計入）；跳過比例 > skip_max，或點估計分母 <= 0（或 Γ1 <= 0）-> 未能測試。
    PASS：Γ6 >= gamma_min 且 log Γ6 − log Γ1 的 95% CI 下界 > 0；其他一律『未通過』（無灰區、不加碼）。"""
    common = [p for p in picks1 if p in picks6]
    i6 = {p: picks6.index(p) for p in common}; i1 = {p: picks1.index(p) for p in common}
    G6, d6 = gamma_ratio(g6, c6)                                         # Γ6：K=6 全部 run
    G6c, d6c = gamma_ratio(g6, c6, [i6[p] for p in common])              # 共同 pick 上的 Γ6（配對差用）
    G1, d1 = gamma_ratio(g1, c1, [i1[p] for p in common])
    out = dict(gamma6=G6, gamma6_common_picks=G6c, gamma1=G1, denominators=dict(k6=d6, k6_common=d6c, k1=d1), common_picks=common)
    if G6 is None or G6c is None or G1 is None or G1 <= 0 or G6c <= 0:
        out.update(state="未能測試", reason="點估計的分母 <= 0，或 Γ1 / 共同 pick 的 Γ6 <= 0（log 無定義）"); return out
    rng = np.random.default_rng(seed); ds = []; sk = 0; n = len(common)
    for _ in range(n_boot):
        draw = rng.integers(0, n, n)
        a6, a1 = gamma_ratio(g6, c6, [i6[common[k]] for k in draw])[0], gamma_ratio(g1, c1, [i1[common[k]] for k in draw])[0]
        if a6 is None or a1 is None or a6 <= 0 or a1 <= 0: sk += 1; continue
        ds.append(np.log(a6) - np.log(a1))
    out["bootstrap_skipped_fraction"] = sk / n_boot
    if sk / n_boot > skip_max or not ds:
        out.update(state="未能測試", reason=f"bootstrap 跳過比例 {sk / n_boot:.3f} > {skip_max}"); return out
    lo, hi = np.percentile(ds, [100 * (1 - ci_level) / 2, 100 * (1 + ci_level) / 2])
    out.update(diff_log_gamma=float(np.log(G6c) - np.log(G1)), diff_ci95=[float(lo), float(hi)],
               state="PASS" if (G6 >= gamma_min and lo > 0) else "未通過")
    return out


# ---------------- 條文 9：層級用語 ----------------
def tier_text(reach_state, probe_state, gate4_ok):
    """(c) 落在 FAIL 帶時的層級用語。reach_state ∈ {可達, 構造不可達, 可達性未知}；probe_state ∈ {PASS, 未通過, 未能測試}；gate4_ok = 閘門 4（正控制）是否通過。"""
    amp = {"PASS": "放大子主張有支持", "未通過": "放大子主張未獲支持", "未能測試": "放大子主張未測"}[probe_state]
    if reach_state == "可達性未知":
        return dict(text=f"起點子預測「未能測試」；放大結果照記（{amp}）", stop=None, may_propose_elimination=False)
    start = {"可達": "起點子預測在 w=3 被否證（實作／劑量層）", "構造不可達": "起點子預測在本設定下對所有 w 被否證（機制層，限本文法與 n/K）"}[reach_state]
    if probe_state == "PASS":
        return dict(text=f"{start}；{amp}", stop="此實作停損：題目縮成「n/K 時鐘＋CFG 放大類別特定的記憶成分」；要不要續做、要不要送 (d)，交使用者決定。報告必須附上 56 號「幅度近乎算術」的保留意見。", may_propose_elimination=False)
    if probe_state == "未通過":
        return dict(text=f"{start}；兩個子主張都未獲支持", stop=None, may_propose_elimination=True,
                    elimination_reason=("可依 57 號規則 2 提請題目淘汰" + ("；閘門 4 也未通過，理由寫成「無可區隔的切入點」" if not gate4_ok else "")))
    return dict(text=f"{start}；只記起點子預測的層級，放大子主張未測", stop=None, may_propose_elimination=False)
