"""機械判讀：依 configs/criteria.json 輸出閘門 0／判準 1 的 PASS／灰區／FAIL／未能測試與全部數字。
用法：aggregate.py [--pairs A B] [--results results] [--selftest]
正式結果目錄約定：results/<pair>/<cond>/graded_verified.json 與取樣／判分憑證，cond ∈
  base, rl（取樣 n=128, test）、rl_greedy, base_greedy（split all, n=1）、rl_replace、wise_0.15 ...（WiSE-FT 各 λ）
判準的所有數字只從 criteria.json 讀；改門檻不必改此檔。
2026-11-14 [claude]（使用者原則變更後、結果前定案，報告 60 通用條文 G 與第 5 節）：整體判定改為現象／訊號／空間三項
（phenomenon_eval／signal_eval／space_eval／r3_status／overall_eval）；gate1 的 verdict 與 R1 FAIL 條件、rl_replace 對 WiSE 的比較只做描述。
2026-11-14（代理落差稽核後、結果前定案）：R1 (ii) 的 FAIL 需排除 +2pp；R2 回應層級混合曲線（只描述）；R3 geo_mix 第二探針（probe2）。
"""
import argparse, json, math, os, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, cfg, load_data
from artifacts import ArtifactError, atomic_json, check_gate_minus1, load_verified, require_hf_pairing


def pass_at_k_vec(c, n, k):
    """無偏估計 1 - C(n-c,k)/C(n,k)，c 為每題答對數的陣列。"""
    c = np.asarray(c)
    out = np.ones(len(c))
    for i, ci in enumerate(c):
        if n - ci < k:
            continue
        # prod_{j=n-ci+1}^{n} (1 - k/j)
        j = np.arange(n - ci + 1, n + 1, dtype=float)
        out[i] = 1.0 - float(np.prod(1.0 - k / j)) if ci > 0 else 0.0
    return out


def per_q(graded, ids, k):
    p1, pk, trunc, ntok, repl = [], [], [], [], []
    for i in ids:
        g = graded[i]
        n = len(g["correct"])
        c = sum(g["correct"])
        p1.append(c / n)
        pk.append(pass_at_k_vec([c], n, k)[0] if n >= k else np.nan)
        trunc.append(1 - sum(g["finished"]) / n)
        ntok.append(sum(g["ntok"]))
        repl.append(sum(g.get("n_repl", [0])))
    return dict(p1=np.array(p1), pk=np.array(pk), trunc=np.array(trunc), ntok=np.array(ntok), repl=np.array(repl))


def interp_frontier(xs, ys, x0):
    """分段線性前緣在 x0 的值（xs 排序；超出範圍取端點並記錄）。"""
    o = np.argsort(xs)
    return float(np.interp(x0, np.asarray(xs)[o], np.asarray(ys)[o]))


def boot_idx(Q, B, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, Q, size=(B, Q))


def mix_curve(c_rl, c_base, n, k, pis):
    """R2（只描述）：回應層級混合的 (pass@1, pass@k) 無偏估計。每次嘗試以機率 π 取 RL 樣本、否則取 base 樣本。
    (1-p)^k 估為 Σ_m Binom(k, m; π) · C(n-c_rl, m)/C(n, m) · C(n-c_base, k-m)/C(n, k-m)。回傳 (p1[], pk[])（對題目平均）。"""
    c_rl, c_base = np.asarray(c_rl, float), np.asarray(c_base, float)
    def hyp(c):  # H[:, m] = C(n-c, m)/C(n, m) = prod_{j<m} (n-c-j)/(n-j)
        H = np.ones((len(c), k + 1))
        for m in range(k):
            H[:, m + 1] = H[:, m] * np.maximum(n - c - m, 0.0) / (n - m)
        return H
    Hr, Hb = hyp(c_rl), hyp(c_base)
    p1, pk = [], []
    for pi in pis:
        allfail = np.zeros(len(c_rl))
        for m in range(k + 1):
            w = math.comb(k, m) * (pi ** m) * ((1 - pi) ** (k - m))
            if w > 0:
                allfail += w * Hr[:, m] * Hb[:, k - m]
        pk.append(float(np.mean(1.0 - allfail)))
        p1.append(float(pi * np.mean(c_rl / n) + (1 - pi) * np.mean(c_base / n)))
    return np.array(p1), np.array(pk)


def wise_pts(pq, ccfg):
    pts = [("0", pq["base"])] + [(str(l), pq.get("wise_%s" % l)) for l in ccfg["wise_lambdas"]] + [("1", pq["rl"])]
    return pts, [n for n, v in pts if v is None]


def wise_margin_boot(rr, pts, idx, alpha):
    """同 pass@1 的前緣差值；未匹配的 bootstrap 差值界於 −100 與 +100 pp。"""
    P1 = np.stack([v["p1"] for _, v in pts], 1)
    PK = np.stack([v["pk"] for _, v in pts], 1)
    p1, xp = P1.mean(0), rr["p1"].mean()
    low, high = p1.min(), p1.max()
    in_range = bool(low <= xp <= high)
    # 範圍外的點差值保留作描述，bootstrap 只對範圍內的重抽樣做內插。
    margin = rr["pk"].mean() - interp_frontier(p1, PK.mean(0), xp)
    bm = np.full(len(idx), np.nan)
    for b, ii in enumerate(idx):
        p1_b, xp_b = P1[ii].mean(0), rr["p1"][ii].mean()
        if p1_b.min() <= xp_b <= p1_b.max():
            bm[b] = rr["pk"][ii].mean() - interp_frontier(p1_b, PK[ii].mean(0), xp_b)
    matched = np.isfinite(bm)
    usable = in_range and bool(matched.any())
    lower_draws = np.where(matched, bm, -1.0)
    upper_draws = np.where(matched, bm, 1.0)
    return {"pt_pp": 100 * margin, "target_p1_pp": 100 * xp, "frontier_p1_range_pp": [100 * low, 100 * high],
            "frontier_clamped": not in_range, "bootstrap_supported_draws": int(matched.sum()),
            "bootstrap_total_draws": len(idx), "usable": usable,
            "lo_pp": 100 * np.quantile(lower_draws, alpha) if usable else None,
            "upper_pp": 100 * np.quantile(upper_draws, 1 - alpha) if usable else None,
            "lo_best_case_pp": 100 * np.quantile(upper_draws, alpha) if usable else None}


def probe2_eval(pq, ccfg, pcfg, ecfg):
    """R3：λ=0.5 幾何混合探針。三項都要成立（單側 α=pcfg.ci_one_sided_alpha），不加碼。"""
    gm, rlh, base = pq[pcfg["cond"]], pq[pcfg["rl_ref_cond"]], pq["base"]
    pts, missing = wise_pts(pq, ccfg)
    if missing:
        return {"verdict": "未能測試（缺 WiSE 點: %s）" % missing}
    Q = len(gm["pk"])
    idx = boot_idx(Q, ecfg["bootstrap_B"], ecfg["bootstrap_seed"])
    a = pcfg["ci_one_sided_alpha"]
    d = gm["pk"] - rlh["pk"]
    c1_pt, c1_lo = 100 * d.mean(), 100 * np.quantile(d[idx].mean(1), a)
    c1 = c1_pt >= pcfg["pk_gain_vs_rl_hf_min_pp"] and c1_lo > 0
    matched = wise_margin_boot(gm, pts, idx, a)
    c2_pt, c2_lo = matched["pt_pp"], matched["lo_pp"]
    if not matched["usable"] or (c2_pt > 0 and c2_lo <= 0 < matched["lo_best_case_pp"]):
        c2 = None
    else:
        c2 = bool(c2_pt > 0 and c2_lo > 0)
    gain_gm = 100 * (gm["p1"].mean() - base["p1"].mean())
    gain_rl = 100 * (rlh["p1"].mean() - base["p1"].mean())
    c3 = gain_gm >= pcfg["retain_p1_frac_of_rl_gain"] * gain_rl
    if c2 is None:
        verdict = "未能測試（WiSE 前緣缺同 pass@1 比較證據）"
    elif c1 and c2 and c3:
        verdict = "通過"
    else:
        verdict = "未通過（不加碼；探針未通過才可提請淘汰）"
    return {"verdict": verdict,
            "cond1_pk_minus_rlhf_pp": c1_pt, "cond1_lo_pp": c1_lo, "cond1_ok": bool(c1),
            "cond2_margin_vs_wise_pp": c2_pt, "cond2_lo_pp": c2_lo, "cond2_ok": c2,
            "cond2_p1_in_wise_range": not matched["frontier_clamped"], "cond2_frontier": matched,
            "cond3_p1_gain_pp": gain_gm, "cond3_rl_p1_gain_pp": gain_rl, "cond3_required_pp": pcfg["retain_p1_frac_of_rl_gain"] * gain_rl, "cond3_ok": bool(c3),
            "alpha_one_sided": a, "geo_pk": 100 * gm["pk"].mean(), "geo_p1": 100 * gm["p1"].mean()}


def gate1(pq, ccfg, ecfg, merged=False):
    # (i) 比較對象：rl_hf（與 rl_replace 同為 HF 引擎）；(ii) 前緣端點 λ=1 用 vLLM 的 rl，rl_hf 作敏感度
    rr, rl, base = pq["rl_replace"], pq["rl"], pq["base"]
    rlh = pq[ccfg.get("rl_compare_cond", "rl_hf")]
    Q = len(rr["pk"])
    idx = boot_idx(Q, ecfg["bootstrap_B"], ecfg["bootstrap_seed"])
    alpha = ccfg["ci_one_sided_alpha"]
    d = rr["pk"] - rlh["pk"]
    bd = d[idx].mean(1)
    res = {"rr_minus_rl_pp": 100 * d.mean(), "rr_minus_rl_lo_pp": 100 * np.quantile(bd, alpha),
           "rr_minus_rl_ci95_pp": [100 * np.quantile(bd, 0.025), 100 * np.quantile(bd, 0.975)],
           "alpha_one_sided": alpha, "rl_compare_cond": ccfg.get("rl_compare_cond", "rl_hf")}
    # 診斷：引擎差（HF − vLLM 的 RL）
    dv = rlh["pk"] - rl["pk"]; bv = dv[idx].mean(1)
    res["diag_rlhf_minus_rlvllm_pk_pp"] = 100 * dv.mean()
    res["diag_rlhf_minus_rlvllm_pk_ci95_pp"] = [100 * np.quantile(bv, 0.025), 100 * np.quantile(bv, 0.975)]
    dv_p1 = rlh["p1"] - rl["p1"]
    res["diag_rlhf_minus_rlvllm_p1_pp"] = 100 * dv_p1.mean()
    res["diag_rlhf_minus_rlvllm_p1_ci95_pp"] = (100 * np.quantile(dv_p1[idx].mean(1), [0.025, 0.975])).tolist()
    res["diag_trunc_pct"] = {"base": 100 * base["trunc"].mean(), "rl_vllm": 100 * rl["trunc"].mean(),
                             "rl_hf": 100 * rlh["trunc"].mean(), "rl_replace": 100 * rr["trunc"].mean()}
    # WiSE-FT 前緣：λ=0(base)、各 λ、λ=1(rl)
    lams = ccfg["wise_lambdas"]
    pts = [("0", base)] + [(str(l), pq.get("wise_%s" % l)) for l in lams] + [("1", rl)]
    missing = [n for n, v in pts if v is None]
    res["wise_missing"] = missing
    res["wise_points"] = {n: [100 * v["p1"].mean(), 100 * v["pk"].mean()] for n, v in pts if v is not None}
    if not missing:
        P1 = np.stack([v["p1"] for _, v in pts], 1)  # Q x m
        PK = np.stack([v["pk"] for _, v in pts], 1)
        x_pt, y_pt = P1.mean(0), PK.mean(0)
        x_rr, y_rr = rr["p1"].mean(), rr["pk"].mean()
        res["rr_p1_in_wise_range"] = bool(x_pt.min() <= x_rr <= x_pt.max())
        margin = y_rr - interp_frontier(x_pt, y_pt, x_rr)
        bm = np.empty(len(idx))
        for b in range(len(idx)):
            ii = idx[b]
            bm[b] = rr["pk"][ii].mean() - interp_frontier(P1[ii].mean(0), PK[ii].mean(0), rr["p1"][ii].mean())
        res["wise_margin_pp"] = 100 * margin
        res["wise_margin_lo_pp"] = 100 * np.quantile(bm, alpha)
        res["wise_margin_ci95_pp"] = [100 * np.quantile(bm, 0.025), 100 * np.quantile(bm, 0.975)]
        res["wise_margin_upper_pp"] = 100 * np.quantile(bm, ccfg["fail_ii_upper_pct"] / 100.0)   # R1：98.75 百分位
        # 敏感度（不參與判定）：λ=1 端點換成 rl_hf
        P1s, PKs = P1.copy(), PK.copy()
        P1s[:, -1], PKs[:, -1] = rlh["p1"], rlh["pk"]
        res["wise_margin_sens_rlhf_endpoint_pp"] = 100 * (y_rr - interp_frontier(P1s.mean(0), PKs.mean(0), x_rr))
    res["rr_p1"], res["rr_pk"] = 100 * rr["p1"].mean(), 100 * rr["pk"].mean()
    res["rl_p1"], res["rl_pk"] = 100 * rl["p1"].mean(), 100 * rl["pk"].mean()
    res["rlhf_p1"], res["rlhf_pk"] = 100 * rlh["p1"].mean(), 100 * rlh["pk"].mean()
    res["base_p1"], res["base_pk"] = 100 * base["p1"].mean(), 100 * base["pk"].mean()
    res["recovery_p1"] = (res["rr_p1"] - res["base_p1"]) / (res["rl_p1"] - res["base_p1"]) if res["rl_p1"] != res["base_p1"] else float("nan")
    res["replace_rate_token_weighted"] = float(rr["repl"].sum() / rr["ntok"].sum())
    # R2（只描述，不參與判定）：回應層級混合曲線與弦
    mc = ccfg.get("mix_curve")
    if mc:
        nS, kS = mc["n"], mc["k"]
        pis = [round(i * mc["pi_step"], 10) for i in range(int(round(1 / mc["pi_step"])) + 1)]
        cmp_pq = pq[mc["compare_cond"]] if mc["compare_cond"] in pq else rl
        p1c, pkc = mix_curve(np.round(cmp_pq["p1"] * nS), np.round(base["p1"] * nS), nS, kS, pis)
        chord = [100 * (base["pk"].mean() + pi * (cmp_pq["pk"].mean() - base["pk"].mean())) for pi in pis]
        res["mix_curve_descriptive"] = {"compare_cond": mc["compare_cond"] if mc["compare_cond"] in pq else "rl",
                                        "points": [{"pi": pi, "p1": 100 * a, "pk": 100 * b, "chord_pk": c, "above_chord_pp": 100 * b - c}
                                                   for pi, a, b, c in zip(pis, p1c, pkc, chord)]}
    res["descriptive_only"] = True   # 2026-11-14 [claude]：不再決定整體判定（見 phenomenon_eval 等）
    # 三態（描述用）
    passed = (res["rr_minus_rl_pp"] >= ccfg["replace_minus_rl_min_pp"] and res["rr_minus_rl_lo_pp"] > 0 and not missing
              and res["wise_margin_pp"] > ccfg["wise_margin_min_pp"] and res["wise_margin_lo_pp"] > 0)
    gmin = ccfg["gray_zone"]["point_diff_min_pp"]
    res["i_pass"] = bool(res["rr_minus_rl_pp"] >= ccfg["replace_minus_rl_min_pp"] and res["rr_minus_rl_lo_pp"] > 0)
    if not missing:
        res["ii_pass"] = bool(res["wise_margin_pp"] > ccfg["wise_margin_min_pp"] and res["wise_margin_lo_pp"] > 0)
        # R1：(ii) 的 FAIL = 點估計 <= 0 且 98.75 百分位 < +2.0pp
        res["ii_fail"] = bool(res["wise_margin_pp"] <= 0 and res["wise_margin_upper_pp"] < ccfg["fail_ii_exclude_pp"])
        res["ii_underpowered"] = bool(res["wise_margin_pp"] <= 0 and not res["ii_fail"])
    if missing:
        res["verdict"] = "未能測試（缺 WiSE 點: %s）" % missing
    elif passed:
        res["verdict"] = "PASS"
    elif res["rr_minus_rl_pp"] < gmin or res["ii_fail"]:
        res["verdict"] = "FAIL"      # 只代表「λ=1 端點」這個實作停損，不等於題目淘汰
    elif res["ii_underpowered"] and merged:
        res["verdict"] = "未能測試（檢定力不足）"   # 已用過唯一一次 dev 加碼，仍排除不了 2pp
    else:
        res["verdict"] = "灰區"      # 含 R1：點估計 <= 0 但 98.75 百分位 >= 2pp
    return res


def gate0(pq, greedy_graded, pair, crit, ecfg, data):
    g0 = crit["gate0"]
    out = {}
    # (a) 重現 RL 已報 pass@1（greedy，對各大型來源）
    src = {r["id"]: r["source"] for r in data}
    rep = {}
    if "rl_greedy" in greedy_graded:
        by = {}
        for i, g in greedy_graded["rl_greedy"].items():
            by.setdefault(src[i], []).append(g["correct"][0])
        for s, v in by.items():
            rep[s] = 100 * float(np.mean(v)) if v else None
    out["rl_greedy_by_source"] = rep
    ref = pair["reported_rl_greedy"]
    diffs = {s: rep[s] - ref[s] for s in g0["reproduce_sources"] if s in rep and s in ref}
    out["reproduce_diffs_pp"] = diffs
    out["reproduce_record_only_diffs_pp"] = {s_: rep[s_] - ref[s_] for s_ in g0.get("reproduce_record_only_sources", []) if s_ in rep and s_ in ref}
    out["reproduce_ok"] = (len(diffs) == len(g0["reproduce_sources"])) and all(abs(v) <= g0["reproduce_tol_pp"] for v in diffs.values())
    # (b) 截斷率
    tb, tr = 100 * pq["base"]["trunc"].mean(), 100 * pq["rl"]["trunc"].mean()
    out["trunc_pct"] = {"base": tb, "rl": tr}
    out["trunc_ok"] = abs(tb - tr) < g0["trunc_rate_diff_max_pp"]
    # (c) 配對資格
    bp1, rp1 = 100 * pq["base"]["p1"].mean(), 100 * pq["rl"]["p1"].mean()
    bpk, rpk = 100 * pq["base"]["pk"].mean(), 100 * pq["rl"]["pk"].mean()
    out.update(base_p1=bp1, rl_p1=rp1, base_pk=bpk, rl_pk=rpk)
    lo, hi = g0["pass_window_pct"]
    win = {"base_pass_at_k_main": bpk, "rl_pass_at_k_main": rpk, "base_p1": bp1, "rl_p1": rp1}
    out["window_ok"] = all(lo <= win[k] <= hi for k in g0["pass_window_applies_to"])
    out["gain_ok"] = (rp1 - bp1) >= g0["pair_rl_p1_gain_min_pp"]
    out["rl_pk_below_base_ok"] = (bpk - rpk) >= g0["pair_rl_pk_below_base_min_pp"]
    out["qualified"] = out["gain_ok"] and out["rl_pk_below_base_ok"] and out["window_ok"]
    if not (out["reproduce_ok"] and out["trunc_ok"]):
        out["verdict"] = "未能測試（重現或截斷率閘門未過：先修程式／模板，限 3 個工作天）"
    elif not out["qualified"]:
        out["verdict"] = "配對不合格（換配對，不算題目死）"
    else:
        out["verdict"] = "配對合格"
    return out


def mix_pq(c_rl, c_base, n, k, pi):
    """R2 混合在單一 π 的逐題 (p1_q, pass@k_q)；mix_curve 的逐題版（bootstrap 用）。"""
    c_rl, c_base = np.asarray(c_rl, float), np.asarray(c_base, float)
    def hyp(c):
        H = np.ones((len(c), k + 1))
        for m in range(k):
            H[:, m + 1] = H[:, m] * np.maximum(n - c - m, 0.0) / (n - m)
        return H
    Hr, Hb = hyp(c_rl), hyp(c_base)
    allfail = np.zeros(len(c_rl))
    for m in range(k + 1):
        w = math.comb(k, m) * (pi ** m) * ((1 - pi) ** (k - m))
        if w > 0:
            allfail += w * Hr[:, m] * Hb[:, k - m]
    return pi * c_rl / n + (1 - pi) * c_base / n, 1.0 - allfail


# ---------------- 報告 60 第 5 節：現象／訊號／空間 ----------------
def phenomenon_eval(summary, crit):
    """現象＝閘門 0(c)。任一組合格 → PASS（主配對＝順序上第一個合格者）；
    每個有資料的組都 (a)(b) 過、且皆不符 (c) → FAIL；其餘（含 (a)(b) 失敗、缺資料）→ 未能測試。"""
    order = crit["phenomenon"]["pair_order"]
    g0 = {p: summary[p]["gate0"] for p in order if p in summary}
    for p in order:
        verdict = g0.get(p, {}).get("verdict", "")
        if verdict == "配對合格":
            return {"verdict": "PASS", "primary_pair": p, "scope": crit["phenomenon"]["scope"]}
        if not verdict.startswith("配對不合格"):
            return {"verdict": "未能測試", "primary_pair": None, "waiting_for_eligibility": p,
                    "note": "前序配對須先確定有資格或確認不合格"}
    ab_ok = [bool(g.get("reproduce_ok") and g.get("trunc_ok")) for g in g0.values() if "reproduce_ok" in g]
    if len(ab_ok) == len(order) and all(ab_ok):
        return {"verdict": "FAIL", "primary_pair": None, "scope": crit["phenomenon"]["scope"],
                "note": "(a)(b) 都過但 A、B 皆不符 (c)；範圍限 %s；配對 C 是否另開由使用者決定" % crit["phenomenon"]["scope"]}
    return {"verdict": "未能測試", "primary_pair": None, "note": "至少一組 (a)(b) 未過或缺資料，且沒有合格配對（修復期 3 個工作天）"}


def signal_eval(pq, crit, merged=False):
    """訊號：量 1 = rl_replace pass@64 − rl_hf ≥ 2pp 且單側 1.25 百分位 > 0；
    量 2 = pass@1 回收率 (rr−base)/(rl_hf−base) ≥ 0.5。PASS＝兩者；FAIL＝量 1 差 < 1pp；其餘灰區。"""
    sc, ecfg = crit["signal"], crit["eval"]
    rr, rlh, base = pq["rl_replace"], pq[sc["q1_cond"]], pq["base"]
    idx = boot_idx(len(rr["pk"]), ecfg["bootstrap_B"], ecfg["bootstrap_seed"])
    d = rr["pk"] - rlh["pk"]
    q1_pt, q1_lo = 100 * d.mean(), 100 * np.quantile(d[idx].mean(1), sc["q1_ci_one_sided_alpha"])
    q1 = bool(q1_pt >= sc["q1_rr_minus_rlhf_pk_min_pp"] and q1_lo > 0)
    den = rlh["p1"].mean() - base["p1"].mean()
    rec = float((rr["p1"].mean() - base["p1"].mean()) / den) if den != 0 else float("nan")
    q2 = bool(rec >= sc["q2_recovery_min"])
    if q1 and q2:
        v = "PASS"
    elif q1_pt < sc["fail_q1_below_pp"]:
        v = "FAIL"
    else:
        v = "灰區（加碼後仍灰區：交使用者）" if merged else "灰區"
    return {"verdict": v, "q1_rr_minus_rlhf_pk_pp": q1_pt, "q1_lo_pp": q1_lo, "q1_ok": q1,
            "q2_recovery_p1": rec, "q2_rlhf_minus_base_p1_pp": 100 * den,
            "q2_denominator_positive": bool(den > 0), "q2_ok": q2, "alpha_one_sided": sc["q1_ci_one_sided_alpha"]}


def space_eval(pq, crit, merged=False):
    """空間：H_a（rl_replace 對 WiSE 前緣）、H_b（R2 混合在 π* 對前緣），擇一（m=2，單側 α=0.00625）。"""
    sc, ecfg, g1 = crit["space"], crit["eval"], crit["gate1"]
    miss = [c for c in ["base", "rl_replace", sc["mix_cond"]] + sc["endpoint_conds"] + ["wise_%s" % l for l in g1["wise_lambdas"]] if c not in pq]
    if miss:
        return {"verdict": "未能測試（缺 %s）" % sorted(set(miss))}
    base, rr = pq["base"], pq["rl_replace"]
    idx = boot_idx(len(rr["pk"]), ecfg["bootstrap_B"], ecfg["bootstrap_seed"])
    a = sc["ci_one_sided_alpha"]
    mid = [("%s" % l, pq["wise_%s" % l]) for l in g1["wise_lambdas"]]
    def frontier_gap(target):
        """λ=1 端點用 rl 與 rl_hf 各算一次，取點估計較小者。"""
        comparisons = {}
        for ec in sc["endpoint_conds"]:
            pts = [("0", base)] + mid + [("1", pq[ec])]
            comparisons[ec] = wise_margin_boot(target, pts, idx, a)
        endpoint = min(comparisons, key=lambda ec: comparisons[ec]["pt_pp"])
        matched = comparisons[endpoint]
        in_range = {ec: not stats["frontier_clamped"] for ec, stats in comparisons.items()}
        return {**matched, "endpoint": endpoint, "endpoints_in_range": in_range,
                "usable": matched["usable"] and all(in_range.values())}
    Ha = frontier_gap(rr)
    # H_b：π*（mix pass@1 最接近 rl_replace pass@1；同距取小）
    nS, kS = g1["mix_curve"]["n"], g1["mix_curve"]["k"]
    c_rl, c_b = np.round(pq[sc["mix_cond"]]["p1"] * nS), np.round(base["p1"] * nS)
    step = sc["mix_pi_step"]
    pis = [round(i * step, 10) for i in range(int(round(1 / step)) + 1)]
    p1c, _ = mix_curve(c_rl, c_b, nS, kS, pis)
    dist = np.abs(p1c - rr["p1"].mean())
    pi_star = pis[int(np.argmin(dist + 1e-12 * np.arange(len(pis))))]   # 同距取較小 π
    mp1, mpk = mix_pq(c_rl, c_b, nS, kS, pi_star)
    Hb = frontier_gap({"p1": mp1, "pk": mpk})
    Hb["pi_star"] = pi_star
    paths = {"H_a": Ha, "H_b": Hb}
    ok = [n for n, h in paths.items() if h["usable"] and h["pt_pp"] >= sc["pass_min_pp"] and h["lo_pp"] > 0]
    fail = all(h["usable"] and h["pt_pp"] <= 0 and h["upper_pp"] < sc["fail_exclude_pp"] for h in paths.values())
    if ok:
        v = "PASS"
    elif fail:
        v = "FAIL"
    elif any(not h["usable"] for h in paths.values()):
        v = "未能測試（WiSE 前緣缺同 pass@1 比較證據）"
    elif merged:
        v = "未能測試（檢定力不足）"
    else:
        v = "灰區"
    return {"verdict": v, "paths_passing": ok, "H_a": Ha, "H_b": Hb, "alpha_one_sided": a, "m_paths": sc["m_paths"]}


def r3_status(primary, sig, spc, PQ, crit, artifact_issue=None):
    """R3＝空間的第三個上限候選。觸發：主配對訊號 PASS 而空間 FAIL。"""
    pcfg = crit["probe2"]
    if not (primary and sig.get("verdict") == "PASS" and spc.get("verdict") == "FAIL"):
        return {"triggered": False, "result": {"verdict": "未觸發"}}
    if artifact_issue is not None:
        return {"triggered": True, "pair": primary, "result": {"verdict": "未能測試（R3 取樣憑證不相容）", "reason": artifact_issue}}
    pq = PQ[primary]
    if pcfg["cond"] in pq and all(c in pq for c in ("base", "rl", "rl_hf")):
        return {"triggered": True, "pair": primary, "result": probe2_eval(pq, crit["gate1"], pcfg, crit["eval"])}
    return {"triggered": True, "pair": primary, "result": {"verdict": "待執行（觸發；缺 %s 取樣結果，見 slurm/06_geo_mix.sbatch）" % pcfg["cond"]}}


def overall_eval(phen, sig, spc, r3):
    """G1：三項都 PASS → 進候選池；任一 FAIL → 不進池（寫明哪一項）；其餘灰區／未能測試／待執行。"""
    space_v = spc.get("verdict", "未評估")
    space_note = None
    if r3.get("triggered"):
        rv = r3["result"]["verdict"]
        if rv == "通過":
            space_v, space_note = "未能測試（R3 多重性待定案）", "R3 通過為描述，第三條擇一路徑的顯著水準尚待定案"
        elif rv.startswith("未通過"):
            space_v, space_note = "FAIL", "R3 未通過"
        elif rv.startswith("未能測試"):
            space_v, space_note = rv, r3["result"].get("reason", rv)
        else:
            space_v, space_note = "待執行 R3", "訊號 PASS 而空間 FAIL，須先做 R3"
    items = {"現象": phen.get("verdict", "未評估"), "訊號": sig.get("verdict", "未評估"), "空間": space_v}
    fails = [k for k, v in items.items() if v == "FAIL"]
    if fails:
        ov = "FAIL：在本 PREREG 範圍內不進候選池（%s FAIL；範圍限 1.5B、3000 token、本批配對）" % "、".join(fails)
    elif all(v == "PASS" for v in items.values()):
        ov = "PASS：進候選池"
    elif any(v.startswith("待執行") for v in items.values()):
        ov = "待 R3（其餘項目未 FAIL）"
    elif any(v.startswith("未能測試") for v in items.values()):
        ov = "未能測試（%s）" % "、".join(k for k, v in items.items() if v.startswith("未能測試"))
    else:
        ov = "灰區（只准條文指定的一次 dev 400 加碼，否則交給使用者）"
    return {"items": items, "space_note": space_note, "overall": ov}


def serializable(value):
    if isinstance(value, dict):
        return {k: serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    if isinstance(value, np.generic):
        return serializable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_summary(path, summary):
    serial = serializable(summary)
    atomic_json(path, serial)
    print(json.dumps(serial, indent=1, ensure_ascii=False, allow_nan=False))


def evaluate(results, pairs, gate, crit, pc, data):
    """Gate 0 reads only reproduction/sample evidence; later files require a qualified primary pair."""
    ecfg = crit["eval"]
    test_ids = [r["id"] for r in data if r["split"] == "test"]
    assert len(test_ids) >= ecfg["test_questions_min"], "test 題數不足"
    if len(set(r["id"] for r in data)) != len(data):
        raise ArtifactError("凍結資料含重複題目 ID")
    order = crit["phenomenon"]["pair_order"]
    selected = set(order) if gate != "0" or not pairs else set(pairs)
    summary, PQ, receipts = {}, {}, {}

    def load_condition(p, c):
        try:
            return load_verified(results / p / c, p, c, pc, ecfg, data, receipts[p])
        except (ArtifactError, OSError, ValueError, KeyError, TypeError) as exc:
            summary[p]["artifact_issues"][c] = str(exc)
            return None

    for p in order:
        if p not in selected:
            continue
        s = summary[p] = {"pair": pc["pairs"][p]["name"], "artifact_issues": {}}
        try:
            receipts[p] = check_gate_minus1(results, p, pc)
        except (ArtifactError, OSError, ValueError, KeyError, TypeError) as exc:
            s["gate_minus1"] = {"verdict": "未能測試", "reason": str(exc)}
            s["gate0"] = {"verdict": "未能測試（缺相容閘門 −1 憑證）"}
            continue
        s["gate_minus1"] = {"verdict": "PASS", **receipts[p]}
        pq, greedy = {}, {}
        PQ[p] = pq
        for c in ("base", "rl", "base_greedy", "rl_greedy"):
            g = load_condition(p, c)
            if g is not None:
                if c.endswith("_greedy"):
                    greedy[c] = g
                else:
                    pq[c] = per_q(g, test_ids, ecfg["k_main"])
        if set(pq) == {"base", "rl"} and set(greedy) == {"base_greedy", "rl_greedy"}:
            s["gate0"] = gate0(pq, greedy, pc["pairs"][p], crit, ecfg, data)
        else:
            s["gate0"] = {"verdict": "未能測試（四個取樣與 greedy 條件須全部驗證）"}
    if gate == "0":
        return {"phase": "gate0", "partial": True, "pairs": summary}

    phen = phenomenon_eval(summary, crit)
    prim = phen.get("primary_pair")
    sig = spc = {"verdict": "未評估（現象未 PASS）"}
    r3_issue = None
    if prim:
        pq = PQ[prim]
        later = ["rl_replace", "rl_hf"] + ["wise_%s" % l for l in crit["gate1"]["wise_lambdas"]]
        for c in later:
            g = load_condition(prim, c)
            if g is not None:
                pq[c] = per_q(g, test_ids, ecfg["k_main"])
        if {"rl_replace", "rl_hf"} <= pq.keys():
            try:
                require_hf_pairing(results, prim, pc, ecfg, data, receipts[prim])
            except (OSError, ValueError, KeyError, TypeError) as exc:
                summary[prim]["artifact_issues"]["HF_pairing"] = str(exc)
                pq.pop("rl_hf")
        need_s = ("base", "rl_replace", crit["signal"]["q1_cond"])
        sig = signal_eval(pq, crit) if all(c in pq for c in need_s) else {"verdict": "未能測試（缺 %s）" % [c for c in need_s if c not in pq]}
        spc = space_eval(pq, crit)
        if all(c in pq for c in ("base", "rl", "rl_replace", crit["gate1"].get("rl_compare_cond", "rl_hf"))):
            summary[prim]["gate1"] = gate1(pq, crit["gate1"], ecfg)
        else:
            summary[prim]["gate1"] = {"verdict": "未執行（缺正式取樣憑證）"}
        if sig.get("verdict") == "PASS" and spc.get("verdict") == "FAIL":
            probe = crit["probe2"]
            c = probe["cond"]
            if (results / prim / c).exists():
                g = load_condition(prim, c)
                if g is None:
                    r3_issue = summary[prim]["artifact_issues"][c]
                else:
                    try:
                        require_hf_pairing(results, prim, pc, ecfg, data, receipts[prim], reference=probe["rl_ref_cond"], candidate=c)
                    except (OSError, ValueError, KeyError, TypeError) as exc:
                        r3_issue = summary[prim]["artifact_issues"][c] = str(exc)
                    else:
                        pq[c] = per_q(g, test_ids, ecfg["k_main"])
    r3 = r3_status(prim, sig, spc, PQ, crit, artifact_issue=r3_issue)
    summary["phenomenon"], summary["signal"], summary["space"], summary["r3"] = phen, sig, spc, r3
    ov = overall_eval(phen, sig, spc, r3)
    summary["overall_three_items"] = ov
    overall = ov["overall"]
    summary["overall"] = overall
    summary["probe2"] = r3     # 相容舊鍵名
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="*", choices=["A", "B"], help="只篩選 gate 0 報告的配對")
    ap.add_argument("--results", type=Path, default=Path(ROOT) / "results")
    ap.add_argument("--gate", choices=["0", "1", "all"], default="all")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.pairs is not None and a.gate != "0":
        ap.error("--pairs 只適用於 --gate 0；正式判定一律檢查凍結的配對順序")
    crit, pc = cfg("criteria.json"), cfg("pairs.json")
    summary = evaluate(a.results, a.pairs, a.gate, crit, pc, load_data("all"))
    a.results.mkdir(parents=True, exist_ok=True)
    if a.gate == "0":
        suffix = "_" + "_".join(sorted(set(a.pairs))) if a.pairs else ""
        write_summary(a.results / f"aggregate_gate0{suffix}.json", summary)
    else:
        write_summary(a.results / "aggregate.json", summary)
        print("\n== 現象／訊號／空間：", summary["overall_three_items"]["items"], "\n== OVERALL:", summary["overall"])


def selftest():
    # 無偏估計式對照暴力組合
    from math import comb
    for n, c, k in [(128, 0, 64), (128, 5, 64), (128, 70, 64), (128, 128, 64), (10, 3, 4)]:
        ref = 0.0 if c == 0 else (1.0 if n - c < k else 1 - comb(n - c, k) / comb(n, k))
        assert abs(pass_at_k_vec([c], n, k)[0] - ref) < 1e-9, (n, c, k)
    print("pass@k estimator OK")
    # 合成資料：rr 明顯高於 rl，應 PASS／或 FAIL 依設定
    rng = np.random.default_rng(0)
    Q, n = 1000, 128
    skill = rng.beta(0.6, 0.9, Q)
    def mk(shift):
        s = np.clip(skill + shift, 0, 1)
        c = rng.binomial(n, s)
        return {"p1": c / n, "pk": pass_at_k_vec(c, n, 64), "trunc": np.zeros(Q), "ntok": np.full(Q, 1000), "repl": np.full(Q, 40)}
    crit = cfg("criteria.json")
    pq = {"base": mk(0.0), "rl": mk(0.0), "rl_replace": mk(0.0), "rl_hf": mk(0.0)}
    for i, l in enumerate(crit["gate1"]["wise_lambdas"]):
        pq["wise_%s" % l] = mk(0.0)
    r = gate1(pq, crit["gate1"], crit["eval"]); print("null scenario verdict:", r["verdict"], round(r["rr_minus_rl_pp"], 2)); assert r["verdict"] == "FAIL"
    assert crit["gate1"]["ci_one_sided_alpha"] == 0.0125 and crit["gate0"]["reproduce_sources"] == ["math500", "minerva"]
    # rr 明顯高於 rl_hf：應 PASS 的路徑（rl_hf 仍 = rl 水準）
    pq2 = dict(pq); pq2["rl_replace"] = mk(0.08)
    r2 = gate1(pq2, crit["gate1"], crit["eval"]); print("positive scenario:", r2["verdict"], round(r2["rr_minus_rl_pp"], 2), r2["rl_compare_cond"])
    assert r2["rr_minus_rl_pp"] > 2 and "diag_rlhf_minus_rlvllm_pk_pp" in r2
    selftest_2026_11_14(crit)
    print("selftest OK")


def selftest_2026_11_14(crit):
    """R1–R3 的新增分支（合成資料）。rr／geo_mix 取為 WiSE 某節點加固定位移，使前緣差值恰等於位移（無內插誤差）。"""
    rng = np.random.default_rng(1)
    g1c = dict(crit["gate1"])
    ecfg = dict(crit["eval"]); ecfg["bootstrap_B"] = 2000
    # ---- R2：混合曲線 ----
    n, k, Q = 128, 64, 300
    c_b, c_r = rng.binomial(n, rng.beta(0.5, 1.0, Q)), rng.binomial(n, rng.beta(1.0, 0.8, Q))
    p1, pk = mix_curve(c_r, c_b, n, k, [0.0, 0.5, 1.0])
    assert abs(pk[0] - pass_at_k_vec(c_b, n, k).mean()) < 1e-9 and abs(pk[2] - pass_at_k_vec(c_r, n, k).mean()) < 1e-9
    assert abs(p1[0] - c_b.mean() / n) < 1e-12 and abs(p1[2] - c_r.mean() / n) < 1e-12
    # 無偏性：真實 p 固定，蒙地卡羅平均的 (1-pk) 應等於 (π(1-p_r)+(1-π)(1-p_b))^k
    pr, pb, pi_, K, N, reps = 0.05, 0.10, 0.4, 8, 32, 6000
    cr_ = rng.binomial(N, pr, reps); cb_ = rng.binomial(N, pb, reps)
    _, pkk = mix_curve(cr_, cb_, N, K, [pi_])
    true_pass = 1 - (pi_ * (1 - pr) + (1 - pi_) * (1 - pb)) ** K
    assert abs(pkk[0] - true_pass) < 0.01 and 0.3 < true_pass < 0.7, (pkk[0], true_pass)
    print("R2 mix_curve OK (端點等於 pass@k；無偏性 %.4f vs %.4f)" % (pkk[0], true_pass))
    # ---- 合成 pq：每題 p1、pk；WiSE 節點 λ：p1 = a + λ(r−a)，pk = 1−(1−p1)^k 的凹函數 ----
    def build(Q, noise_sd, shift_rr, rr_node="0.6", rl_pk_penalty=0.15):
        a = rng.uniform(0.02, 0.30, Q); r = np.clip(a + rng.uniform(0.0, 0.5, Q), 0, 1)
        f = lambda p: 1 - (1 - p) ** 8
        mkq = lambda p1_, pk_: {"p1": p1_, "pk": pk_, "trunc": np.zeros(Q), "ntok": np.full(Q, 1000), "repl": np.full(Q, 40)}
        pq = {"base": mkq(a, f(a)), "rl": mkq(r, np.clip(f(r) - rl_pk_penalty, 0, 1))}
        pq["rl_hf"] = mkq(pq["rl"]["p1"].copy(), pq["rl"]["pk"].copy())
        for l in g1c["wise_lambdas"]:
            q = a + l * (r - a); pq["wise_%s" % l] = mkq(q, f(q))
        node = pq["wise_%s" % rr_node]
        nz = rng.normal(0, noise_sd, Q); nz -= nz.mean()
        pq["rl_replace"] = mkq(node["p1"].copy(), node["pk"] + shift_rr + nz)
        return pq, mkq, f, a, r
    # 1) (i) 成立、(ii) 明確 FAIL：位移 −3pp，Q 大、雜訊小 → 98.75 百分位 < 2pp
    pq, mkq, f, a, r = build(1000, 0.05, -0.03)
    r1 = gate1(pq, g1c, ecfg)
    assert r1["i_pass"] and r1["wise_margin_pp"] <= 0 and r1["wise_margin_upper_pp"] < 2.0 and r1["ii_fail"] and r1["verdict"] == "FAIL", r1
    # 2) 點估計 <= 0 但檢定力不足：Q 小、雜訊大 → 上界 >= 2pp → 灰區；合併（merged）後記「未能測試（檢定力不足）」
    pq2, *_ = build(60, 0.30, -0.005)
    r2 = gate1(pq2, g1c, ecfg)
    assert r2["wise_margin_pp"] <= 0 and r2["wise_margin_upper_pp"] >= 2.0 and not r2["ii_fail"] and r2["ii_underpowered"], r2
    assert r2["verdict"] == "灰區", r2["verdict"]
    r2m = gate1(pq2, g1c, ecfg, merged=True)
    assert r2m["verdict"] == "未能測試（檢定力不足）", r2m["verdict"]
    # 3) (ii) 通過 → PASS（位移 +3pp）；(i) 不成立時（rl_hf 與 rr 同高）仍判 FAIL
    pq3, *_ = build(1000, 0.05, +0.03)
    r3 = gate1(pq3, g1c, ecfg)
    assert r3["verdict"] == "PASS" and r3["ii_pass"] and not r3["ii_fail"], r3
    pq4 = dict(pq3); pq4["rl_hf"] = dict(pq3["rl_replace"])
    assert gate1(pq4, g1c, ecfg)["verdict"] == "FAIL"
    mc_pts = r1["mix_curve_descriptive"]["points"]
    assert len(mc_pts) == 21 and mc_pts[0]["pi"] == 0.0 and abs(mc_pts[-1]["pi"] - 1.0) < 1e-12 and all(math.isfinite(q["pk"]) for q in mc_pts)
    print("R1 分支 OK：FAIL(明確)／灰區(檢定力不足)／merged→未能測試／PASS／(i) 不成立→FAIL")
    # ---- R3：probe2 ----
    pcfg = dict(crit["probe2"])
    def with_geo(pq, node, shift, pk_over=None):
        q = dict(pq); nd = pq["wise_%s" % node]
        q["geo_mix"] = {"p1": nd["p1"].copy(), "pk": (nd["pk"] + shift) if pk_over is None else pk_over,
                        "trunc": np.zeros(len(nd["p1"])), "ntok": np.full(len(nd["p1"]), 1000), "repl": np.full(len(nd["p1"]), 40)}
        return q
    base_pq = pq                                  # 情境 1：(i) 成立、(ii) FAIL
    ok_ = probe2_eval(with_geo(base_pq, "0.6", 0.05), g1c, pcfg, ecfg)
    assert ok_["verdict"] == "通過" and ok_["cond1_ok"] and ok_["cond2_ok"] and ok_["cond3_ok"], ok_
    no2 = probe2_eval(with_geo(base_pq, "0.6", 0.0), g1c, pcfg, ecfg)         # 沒高過前緣
    assert no2["verdict"].startswith("未通過") and not no2["cond2_ok"] and no2["cond1_ok"], no2
    no3 = probe2_eval(with_geo(base_pq, "0.3", 0.05), g1c, pcfg, ecfg)         # pass@1 保留不足（節點 0.3 < 0.5×RL 增益）
    assert no3["verdict"].startswith("未通過") and not no3["cond3_ok"] and no3["cond2_ok"], no3
    no1 = probe2_eval(with_geo(base_pq, "0.6", 0.0, pk_over=base_pq["rl_hf"]["pk"] + 0.01), g1c, pcfg, ecfg)  # pass@64 只比 rl_hf 高 1pp
    assert no1["verdict"].startswith("未通過") and not no1["cond1_ok"], no1
    # 觸發：訊號 PASS 而空間 FAIL（主配對）
    PQs = {"A": with_geo(base_pq, "0.6", 0.05)}
    crit_ = crit
    tr = r3_status("A", {"verdict": "PASS"}, {"verdict": "FAIL"}, PQs, crit_)
    assert tr["triggered"] and tr["pair"] == "A" and tr["result"]["verdict"] == "通過"
    assert overall_eval({"verdict": "PASS"}, {"verdict": "PASS"}, {"verdict": "FAIL"}, tr)["overall"].startswith("未能測試")
    tr_no = r3_status("A", {"verdict": "PASS"}, {"verdict": "FAIL"}, {"A": base_pq}, crit_)   # 尚無 geo_mix
    assert tr_no["triggered"] and tr_no["result"]["verdict"].startswith("待執行")
    assert overall_eval({"verdict": "PASS"}, {"verdict": "PASS"}, {"verdict": "FAIL"}, tr_no)["overall"].startswith("待 R3")
    fail_r3 = {"triggered": True, "result": {"verdict": "未通過（x）"}}
    assert overall_eval({"verdict": "PASS"}, {"verdict": "PASS"}, {"verdict": "FAIL"}, fail_r3)["overall"].startswith("FAIL")
    for sg, sp in [("灰區", "FAIL"), ("FAIL", "FAIL"), ("PASS", "PASS"), ("PASS", "灰區")]:
        assert not r3_status("A", {"verdict": sg}, {"verdict": sp}, PQs, crit_)["triggered"]
    assert not r3_status(None, {"verdict": "PASS"}, {"verdict": "FAIL"}, PQs, crit_)["triggered"]
    print("R3 probe2 OK：通過／cond2 失敗／cond3 失敗／cond1 失敗／觸發條件（訊號 PASS 且空間 FAIL）")
    selftest_three_items(crit)


def selftest_three_items(crit):
    """報告 60 第 5 節：現象／訊號／空間各自的 PASS、FAIL、灰區（合成資料）與整體規則。"""
    rng = np.random.default_rng(7)
    ecfg = dict(crit["eval"]); ecfg["bootstrap_B"] = 2000
    crit = dict(crit); crit["eval"] = ecfg
    n = crit["gate1"]["mix_curve"]["n"]; kk = crit["gate1"]["mix_curve"]["k"]
    # ---- 現象 ----
    g_ok = {"reproduce_ok": True, "trunc_ok": True, "verdict": "配對合格"}
    g_unq = {"reproduce_ok": True, "trunc_ok": True, "verdict": "配對不合格（換配對，不算題目死）"}
    g_bad = {"reproduce_ok": False, "trunc_ok": True, "verdict": "未能測試（x）"}
    S = lambda a, b: {"A": {"gate0": a}, "B": {"gate0": b}}
    assert phenomenon_eval(S(g_unq, g_ok), crit) == {"verdict": "PASS", "primary_pair": "B", "scope": "1.5B, 3000 token"}
    assert phenomenon_eval(S(g_ok, g_unq), crit)["primary_pair"] == "A"
    assert phenomenon_eval(S(g_unq, g_unq), crit)["verdict"] == "FAIL"
    assert phenomenon_eval(S(g_unq, g_bad), crit)["verdict"] == "未能測試"
    assert phenomenon_eval({"A": {"gate0": g_unq}}, crit)["verdict"] == "未能測試"      # B 缺資料
    assert phenomenon_eval({"B": {"gate0": g_ok}}, crit)["waiting_for_eligibility"] == "A"
    assert phenomenon_eval(S(g_bad, g_ok), crit)["waiting_for_eligibility"] == "A"
    # ---- 合成逐題資料（與計數一致）----
    def mkq(c, pen=0.0):
        c = np.asarray(c, float)
        return {"p1": c / n, "pk": np.clip(pass_at_k_vec(c, n, kk) - pen, 0, 1), "trunc": np.zeros(len(c)), "ntok": np.full(len(c), 1000), "repl": np.full(len(c), 40)}
    def world(Q, noise_sd, shift, rr_node="0.6", wise_pen=0.0):
        a = rng.uniform(0.02, 0.30, Q); r = np.clip(a + rng.uniform(0.0, 0.5, Q), 0, 1)
        pq = {"base": mkq(np.round(a * n)), "rl": mkq(np.round(r * n))}
        pq["rl_hf"] = dict(pq["rl"])
        for l in crit["gate1"]["wise_lambdas"]:
            pq["wise_%s" % l] = mkq(np.round((a + l * (r - a)) * n), pen=wise_pen)
        node = pq["wise_%s" % rr_node]
        nz = rng.normal(0, noise_sd, Q); nz -= nz.mean()
        pq["rl_replace"] = {**node, "pk": node["pk"] + shift + nz}
        return pq
    # ---- 訊號 ----
    pqs = world(1000, 0.02, 0.0)
    def with_rr(pq, rec, delta, noise=0.0):
        q = dict(pq); b, r = pq["base"], pq["rl_hf"]
        nz = rng.normal(0, noise, len(b["p1"])); nz -= nz.mean()
        q["rl_replace"] = {**pq["rl_replace"], "p1": b["p1"] + rec * (r["p1"] - b["p1"]), "pk": r["pk"] + delta + nz}
        return q
    sp = signal_eval(with_rr(pqs, 0.8, 0.04, 0.02), crit)
    assert sp["verdict"] == "PASS" and sp["q1_ok"] and sp["q2_ok"], sp
    sf = signal_eval(with_rr(pqs, 0.8, 0.002, 0.02), crit)                       # 量 1 < 1pp
    assert sf["verdict"] == "FAIL", sf
    sg1 = signal_eval(with_rr(pqs, 0.8, 0.015, 0.02), crit)                      # 1pp <= 差 < 2pp
    assert sg1["verdict"] == "灰區" and not sg1["q1_ok"], sg1
    sg2 = signal_eval(with_rr(pqs, 0.3, 0.04, 0.02), crit)                       # 量 1 成立、回收率不足
    assert sg2["verdict"] == "灰區" and sg2["q1_ok"] and not sg2["q2_ok"], sg2
    sg3 = signal_eval(with_rr(world(60, 0.02, 0.0), 0.8, 0.025, 0.30), crit)     # 差 >= 2pp 但下界 <= 0
    assert sg3["verdict"] == "灰區" and not sg3["q1_ok"], sg3
    assert signal_eval(with_rr(pqs, 0.8, 0.015, 0.02), crit, merged=True)["verdict"].startswith("灰區（加碼後")
    print("訊號 OK：PASS／FAIL／灰區（差在 1–2pp、回收率不足、下界不過）")
    # ---- 空間 ----
    sp_pass = space_eval(world(1000, 0.05, +0.03), crit)                          # H_a 通過
    assert sp_pass["verdict"] == "PASS" and "H_a" in sp_pass["paths_passing"], sp_pass
    sp_hb = space_eval(world(1000, 0.05, 0.0, wise_pen=0.05), crit)               # WiSE 被壓低 5pp：H_a≈0、H_b>2
    assert sp_hb["verdict"] == "PASS" and sp_hb["paths_passing"] == ["H_b"] and sp_hb["H_a"]["pt_pp"] < 2.0, sp_hb
    sp_fail = space_eval(world(1000, 0.05, -0.03), crit)                          # H_a < 0；H_b≈0
    assert sp_fail["verdict"] == "FAIL", sp_fail
    sp_gray = space_eval(world(60, 0.30, -0.005), crit)                           # 檢定力不足
    assert sp_gray["verdict"] == "灰區", sp_gray
    assert space_eval(world(60, 0.30, -0.005), crit, merged=True)["verdict"].startswith("未能測試（檢定力不足）")
    assert space_eval({"base": pqs["base"]}, crit)["verdict"].startswith("未能測試")
    # π* 選取：同距取較小
    p1c = np.array([0.1, 0.2, 0.3]); pis_ = [0.0, 0.5, 1.0]; tgt = 0.25
    assert pis_[int(np.argmin(np.abs(p1c - tgt) + 1e-12 * np.arange(3)))] == 0.5
    print("空間 OK：PASS(H_a)／PASS(僅 H_b)／FAIL／灰區／合併後未能測試；α=%.5f" % sp_pass["alpha_one_sided"])
    # ---- 整體 ----
    P, F, G = {"verdict": "PASS"}, {"verdict": "FAIL"}, {"verdict": "灰區"}
    nt = {"triggered": False, "result": {"verdict": "未觸發"}}
    assert overall_eval(P, P, P, nt)["overall"].startswith("PASS")
    assert overall_eval(P, F, P, nt)["overall"].startswith("FAIL") and "訊號" in overall_eval(P, F, P, nt)["overall"]
    assert overall_eval(P, P, G, nt)["overall"].startswith("灰區")
    assert overall_eval({"verdict": "未能測試"}, {"verdict": "未評估"}, {"verdict": "未評估"}, nt)["overall"].startswith("未能測試")
    print("整體 OK")


if __name__ == "__main__":
    main()
