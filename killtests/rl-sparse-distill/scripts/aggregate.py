"""機械判讀：依 configs/criteria.json 輸出閘門 0／判準 1 的 PASS／灰區／FAIL／未能測試與全部數字。
用法：aggregate.py [--pairs A B] [--results results] [--selftest]
結果目錄約定：results/<pair>/<cond>/graded.json，cond ∈
  base, rl（取樣 n=128, test）、rl_greedy, base_greedy（split all, n=1）、rl_replace、wise_0.15 ...（WiSE-FT 各 λ）
判準的所有數字只從 criteria.json 讀；改門檻不必改此檔。
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


def gate1(pq, ccfg, ecfg):
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
    # 三態
    passed = (res["rr_minus_rl_pp"] >= ccfg["replace_minus_rl_min_pp"] and res["rr_minus_rl_lo_pp"] > 0 and not missing
              and res["wise_margin_pp"] > ccfg["wise_margin_min_pp"] and res["wise_margin_lo_pp"] > 0)
    gmin = ccfg["gray_zone"]["point_diff_min_pp"]
    if missing:
        res["verdict"] = "未能測試（缺 WiSE 點: %s）" % missing
    elif passed:
        res["verdict"] = "PASS"
    elif res["rr_minus_rl_pp"] < gmin or res["wise_margin_pp"] <= 0:
        res["verdict"] = "FAIL"
    else:
        res["verdict"] = "灰區"
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
    summary = {}
    for p in a.pairs:
        d = os.path.join(a.results, p)
        pq, greedy = {}, {}
        conds = ["base", "rl", "rl_replace", "rl_hf"] + ["wise_%s" % l for l in crit["gate1"]["wise_lambdas"]]
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
        overall = "FAIL（停：不做蒸餾）"
    else:
        overall = "未完成（合格配對尚缺判準 1 結果）"
    summary["overall"] = overall
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
    print("selftest OK")


if __name__ == "__main__":
    main()
