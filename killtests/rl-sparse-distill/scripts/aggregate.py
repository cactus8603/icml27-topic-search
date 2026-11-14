"""機械判讀：依 configs/criteria.json 輸出閘門 0／判準 1 的 PASS／灰區／FAIL／未能測試與全部數字。
用法：aggregate.py [--pairs A B] [--results results] [--selftest]
結果目錄約定：results/<pair>/<cond>/graded.json，cond ∈
  base, rl（取樣 n=128, test）、rl_greedy, base_greedy（split all, n=1）、rl_replace、wise_0.15 ...（WiSE-FT 各 λ）
判準的所有數字只從 criteria.json 讀；改門檻不必改此檔。
2026-11-14（代理落差稽核後、結果前定案）：R1 (ii) 的 FAIL 需排除 +2pp；R2 回應層級混合曲線（只描述）；R3 geo_mix 第二探針（probe2）。
"""
import argparse, json, math, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, cfg, load_data


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


def load_graded(path):
    with open(path) as f:
        return json.load(f)


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
    """分段線性前緣在 x0 的值（xs 排序；超出範圍取端點，對 rl_replace 不利的保守方向由 flag 回報）。"""
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


def wise_margin_boot(rr, pts, idx):
    """rr 對 WiSE 前緣的差值：點估計與 bootstrap 分布（與 gate1 (ii) 同一算式）。"""
    P1 = np.stack([v["p1"] for _, v in pts], 1)
    PK = np.stack([v["pk"] for _, v in pts], 1)
    margin = rr["pk"].mean() - interp_frontier(P1.mean(0), PK.mean(0), rr["p1"].mean())
    bm = np.empty(len(idx))
    for b in range(len(idx)):
        ii = idx[b]
        bm[b] = rr["pk"][ii].mean() - interp_frontier(P1[ii].mean(0), PK[ii].mean(0), rr["p1"][ii].mean())
    return margin, bm


def probe2_eval(pq, ccfg, pcfg, ecfg):
    """R3：λ=0.5 幾何混合探針。三項都要成立（單側 α=pcfg.ci_one_sided_alpha）；只有通過／未通過，不加碼。"""
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
    mg, bm = wise_margin_boot(gm, pts, idx)
    c2_pt, c2_lo = 100 * mg, 100 * np.quantile(bm, a)
    c2 = c2_pt > 0 and c2_lo > 0
    gain_gm = 100 * (gm["p1"].mean() - base["p1"].mean())
    gain_rl = 100 * (rlh["p1"].mean() - base["p1"].mean())
    c3 = gain_gm >= pcfg["retain_p1_frac_of_rl_gain"] * gain_rl
    P1 = np.stack([v["p1"] for _, v in pts], 1).mean(0)
    return {"verdict": "通過" if (c1 and c2 and c3) else "未通過（只有通過或未通過，不加碼；探針未通過才可提請淘汰）",
            "cond1_pk_minus_rlhf_pp": c1_pt, "cond1_lo_pp": c1_lo, "cond1_ok": bool(c1),
            "cond2_margin_vs_wise_pp": c2_pt, "cond2_lo_pp": c2_lo, "cond2_ok": bool(c2),
            "cond2_p1_in_wise_range": bool(P1.min() <= gm["p1"].mean() <= P1.max()),
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
    res["diag_rlhf_minus_rlvllm_p1_pp"] = 100 * (rlh["p1"].mean() - rl["p1"].mean())
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
    # 三態
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


def probe2_status(summary, PQ, crit, pc):
    """R3 觸發：合格配對的 (i) 成立而 (ii) FAIL；只做一個配對，A 符合就做 A，否則做 B。"""
    pcfg = crit["probe2"]
    for p in pcfg["pair_order"]:
        s = summary.get(p)
        if s and s["gate0"].get("verdict") == "配對合格" and s["gate1"].get("i_pass") and s["gate1"].get("ii_fail"):
            pq = PQ[p]
            if pcfg["cond"] in pq and all(c in pq for c in ("base", "rl", "rl_hf")):
                return {"triggered": True, "pair": p, "result": probe2_eval(pq, crit["gate1"], pcfg, crit["eval"])}
            return {"triggered": True, "pair": p, "result": {"verdict": "待執行（觸發；缺 %s 取樣結果，見 slurm/06_geo_mix.sbatch）" % pcfg["cond"]}}
    return {"triggered": False, "result": {"verdict": "未觸發"}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="*", default=["A", "B"])
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    ap.add_argument("--gate", choices=["0", "1", "all"], default="all")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    crit, pc = cfg("criteria.json"), cfg("pairs.json")
    ecfg = crit["eval"]
    data = load_data("all")
    test_ids = [r["id"] for r in data if r["split"] == "test"]
    assert len(test_ids) >= ecfg["test_questions_min"], "test 題數不足"
    summary, PQ = {}, {}
    for p in a.pairs:
        d = os.path.join(a.results, p)
        pq, greedy = {}, {}
        conds = ["base", "rl", "rl_replace", "rl_hf", crit["probe2"]["cond"]] + ["wise_%s" % l for l in crit["gate1"]["wise_lambdas"]]
        for c in conds:
            f = os.path.join(d, c, "graded.json")
            if os.path.exists(f):
                g = load_graded(f)
                miss = [i for i in test_ids if i not in g]
                if miss:
                    print("[%s/%s] 缺 %d 題，略過" % (p, c, len(miss))); continue
                bad = [i for i in test_ids if len(g[i]["correct"]) < ecfg["n_samples"]]
                if bad:
                    print("[%s/%s] %d 題樣本數 < %d，略過" % (p, c, len(bad), ecfg["n_samples"])); continue
                pq[c] = per_q(g, test_ids, ecfg["k_main"])
        for c in ["rl_greedy", "base_greedy"]:
            f = os.path.join(d, c, "graded.json")
            if os.path.exists(f):
                greedy[c] = load_graded(f)
        PQ[p] = pq
        s = {"pair": pc["pairs"][p]["name"]}
        if "base" in pq and "rl" in pq:
            s["gate0"] = gate0(pq, greedy, pc["pairs"][p], crit, ecfg, data)
        else:
            s["gate0"] = {"verdict": "未能測試（缺 base 或 rl 取樣結果）"}
        if a.gate in ("1", "all") and all(c in pq for c in ("base", "rl", "rl_replace", crit["gate1"].get("rl_compare_cond", "rl_hf"))):
            s["gate1"] = gate1(pq, crit["gate1"], ecfg)
        else:
            s["gate1"] = {"verdict": "未執行（缺 rl_replace、rl_hf 或 base/rl）"}
        summary[p] = s
    # 整體：至少一個「配對合格」且判準 1 PASS 才 PASS；有灰區無 PASS → 灰區；合格配對全 FAIL → FAIL；無合格配對 → 未能測試
    qual = [p for p, s in summary.items() if s["gate0"].get("verdict") == "配對合格"]
    v1 = {p: summary[p]["gate1"].get("verdict") for p in qual}
    if not qual:
        overall = "未能測試（沒有合格配對：換配對，見 pairs.json reserve）"
    elif any(v == "PASS" for v in v1.values()):
        overall = "PASS"
    elif any(v == "灰區" for v in v1.values()):
        overall = "灰區（只准一次事先定好的加碼）"
    elif all(v == "FAIL" for v in v1.values()):
        overall = "FAIL（停 λ=1 端點路線，不做蒸餾；不等於題目淘汰：若 (i) 成立而 (ii) FAIL 須先做 R3 探針）"
    else:
        overall = "未完成（合格配對尚缺判準 1 結果）"
    summary["overall"] = overall
    summary["probe2"] = probe2_status(summary, PQ, crit, pc)
    os.makedirs(a.results, exist_ok=True)
    with open(os.path.join(a.results, "aggregate.json"), "w") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False, default=float)
    print(json.dumps(summary, indent=1, ensure_ascii=False, default=float))
    print("\n== OVERALL:", overall)


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
    # 觸發與配對選擇
    summ = {"A": {"gate0": {"verdict": "配對合格"}, "gate1": {"i_pass": True, "ii_fail": True}},
            "B": {"gate0": {"verdict": "配對合格"}, "gate1": {"i_pass": True, "ii_fail": True}}}
    pqs = {"A": with_geo(base_pq, "0.6", 0.05), "B": base_pq}
    tr = probe2_status(summ, pqs, crit, None)
    assert tr["triggered"] and tr["pair"] == "A" and tr["result"]["verdict"] == "通過"
    summ["A"]["gate1"] = {"i_pass": True, "ii_fail": False}                    # A 不符合 → 做 B（尚無 geo_mix → 待執行）
    tr = probe2_status(summ, pqs, crit, None)
    assert tr["triggered"] and tr["pair"] == "B" and tr["result"]["verdict"].startswith("待執行")
    summ["B"]["gate1"] = {"i_pass": False, "ii_fail": True}                    # (i) 不成立 → 不觸發
    summ["A"]["gate0"] = {"verdict": "配對不合格"}
    assert not probe2_status(summ, pqs, crit, None)["triggered"]
    print("R3 probe2 OK：通過／cond2 失敗／cond3 失敗／cond1 失敗／觸發與配對選擇")


if __name__ == "__main__":
    main()
