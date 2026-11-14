"""機械輸出各閘門與主判準結果（PREREG.md 的判讀規則），並列出原始數字。不含任何人工判斷。
用法：python scripts/aggregate.py [--results results] [--out results/aggregate]
輸出：results/aggregate.json 與 results/aggregate.md（繁體中文）。
閘門失敗 -> 該題記為「未能測試」，不輸出 PASS／FAIL。
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import EXP, SUBSETS, params

PASS, GRAY, FAIL, UNTESTABLE = "PASS", "灰區", "FAIL", "未能測試"


# ------------------------------------------------------------------ 純判讀函式（unit_tests.py 會測）

def decide_p1(ci, thr, addon=None, addon_margin=5.0):
    """ci: {'point','lo','hi'}（CLIP B/32 + LLR 在 COCO(Qwen) 的 COR，pp；lo/hi 為 97.5% 單尾對應的 2.5/97.5 百分位）。
    addon: None 或 paired 差 {'point','lo'}（CS-CLIP+LLR 減 CS-CLIP holistic 的 COR，pp）。
    回傳 (state, 說明)。"""
    if ci["lo"] >= thr:
        return PASS, "下界 >= 門檻"
    if ci["hi"] < thr:
        return FAIL, "上界 < 門檻（明確失敗，不給 CS-CLIP 救援）"
    # 灰區：信賴區間跨越門檻，唯一加碼 = CS-CLIP 救援
    if addon is None:
        return GRAY, "信賴區間跨越門檻；加碼資料不足"
    rescued = addon["point"] >= addon_margin and addon["lo"] > 0
    if rescued:
        return PASS, "灰區 -> 加碼（CS-CLIP+LLR 比 CS-CLIP 高 >= 5pp 且配對下界 > 0）通過"
    if ci["point"] < thr:
        return FAIL, "灰區 -> 加碼未通過，且點估計 < 門檻"
    return GRAY, "灰區 -> 加碼未通過，但點估計 >= 門檻；未解，交使用者"


def decide_p2(delta, max_drop, min_delta, max_adjacent_drop):
    """delta: {'point','lo','hi'}（Δerr = err(k_top) - err(1)，pp）；max_drop: 相鄰 k 的最大下降（pp）。"""
    if delta["lo"] >= min_delta and max_drop <= max_adjacent_drop:
        return PASS, "Δ 下界 >= 門檻且相鄰下降在容許範圍內"
    if delta["hi"] < min_delta:
        return FAIL, "Δ 上界 < 門檻"
    return GRAY, "Δ 信賴區間跨越門檻，或單調性違反"


# ------------------------------------------------------------------ 讀檔

def _load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def gates(R, P):
    out = {}
    tol = P["repro_tolerance_pp"]
    # G1 程式正確性
    g1u = _load(R / "gates" / "g1_unit.json")
    g1c = _load(R / "gates" / "g1_crosscheck.json")
    ht = {m: _load(R / "ht" / f"analysis_{m}.json") for m in ("clip_b32", "cs_clip")}
    cov = {}
    for m, a in ht.items():
        if a:
            for s, v in a["subsets"].items():
                cov[f"{m}/{s}"] = v["detail_coverage"]["coverage"]
    g1 = {"unit_tests_pass": bool(g1u and g1u.get("all_pass")),
          "crosscheck_max_abs_diff": g1c.get("max_abs_diff") if g1c else None,
          "crosscheck_pass": bool(g1c and g1c["max_abs_diff"] <= P["gate_g1"]["max_abs_diff_vs_repo_script"]),
          "detail_coverage": cov,
          "coverage_pass": bool(cov) and all(v >= P["gate_g1"]["min_detail_coverage"] for v in cov.values())}
    g1["pass"] = g1["unit_tests_pass"] and g1["crosscheck_pass"] and g1["coverage_pass"]
    out["G1_程式正確性"] = g1
    # G2 重現
    g2 = {"cs_clip_cor": {}, "clip_cor_diag": {}, "ht_acc_diag": {}}
    ok = bool(ht["cs_clip"])
    for s in SUBSETS:
        if ht["cs_clip"] and s in ht["cs_clip"]["subsets"]:
            c = ht["cs_clip"]["subsets"][s]["aggs"]["holistic"]["all"]
            v = c["cor"]["point"]
            d = {"measured": v, "paper": P["paper_cor"][s], "diff": v - P["paper_cor"][s],
                 "within_tol": abs(v - P["paper_cor"][s]) <= tol}
            g2["cs_clip_cor"][s] = d
            ok &= d["within_tol"]
            g2["ht_acc_diag"][f"cs_clip/{s}"] = {"measured": c["ht"]["point"], "readme": P["readme_ht_acc_cs_clip"][s]}
        else:
            ok = False
        if ht["clip_b32"] and s in ht["clip_b32"]["subsets"]:
            c = ht["clip_b32"]["subsets"][s]["aggs"]["holistic"]["all"]
            g2["clip_cor_diag"][s] = {"measured": c["cor"]["point"], "paper": P["paper_clip_cor"][s],
                                      "within_tol": abs(c["cor"]["point"] - P["paper_clip_cor"][s]) <= tol}
            g2["ht_acc_diag"][f"clip_b32/{s}"] = {"measured": c["ht"]["point"], "readme": P["readme_ht_acc_clip"][s]}
    g2["pass"] = bool(ok)
    g2["說明"] = "阻擋條件 = CS-CLIP 三組 COR 皆在 paper ±3pp；CLIP COR 與 HT 準確率僅為診斷（不阻擋，但偏離須在報告中解釋）。"
    out["G2_重現"] = g2
    # G3 雜訊底線
    g3 = {"ci_halfwidth_pp": {}, "rerun": _load(R / "gates" / "g3_rerun.json")}
    okw = True
    for m, a in ht.items():
        if a:
            for s, v in a["subsets"].items():
                c = v["aggs"]["holistic"]["all"]["cor"]
                hw = (c["hi"] - c["lo"]) / 2
                g3["ci_halfwidth_pp"][f"{m}/{s}"] = hw
                okw &= hw <= P["gate_g3"]["max_ci_halfwidth_pp"]
    rr = g3["rerun"]
    g3["rerun_pass"] = bool(rr and rr["max_abs_diff"] <= P["gate_g3"]["max_rerun_abs_diff"])
    g3["pass"] = bool(g3["ci_halfwidth_pp"]) and bool(okw) and g3["rerun_pass"]
    out["G3_雜訊底線"] = g3
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(EXP / "results"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    R = Path(args.results)
    p = params()
    G = gates(R, p)
    gate_ok = all(G[k]["pass"] for k in G)
    res = {"gates": G, "gate_all_pass": gate_ok, "primaries": {}}

    # ---- P1
    a_clip = _load(R / "ht" / "analysis_clip_b32.json")
    a_cs = _load(R / "ht" / "analysis_cs_clip.json")
    P1 = p["p1"]
    p1 = {"state": UNTESTABLE, "note": "資料不足"}
    if a_clip and "coco_qwen" in a_clip["subsets"]:
        ci = a_clip["subsets"]["coco_qwen"]["aggs"]["llr_sum"]["all"]["cor"]
        addon = None
        if a_cs and "coco_qwen" in a_cs["subsets"]:
            addon = a_cs["subsets"]["coco_qwen"]["aggs"]["llr_sum"]["paired_vs_holistic_cor_pp"]
        st, note = decide_p1(ci, P1["threshold_cor_pp"], addon, P1["addon_margin_pp"])
        p1 = {"metric": "CLIP B/32 + LLR 加總，COCO(Qwen) COR (%)", "ci": {k: ci[k] for k in ("point", "lo", "hi", "n")},
              "threshold": P1["threshold_cor_pp"], "addon_cs_clip_paired_diff": addon,
              "state_raw": st, "state": st if gate_ok else UNTESTABLE, "note": note + ("" if gate_ok else "；但閘門未全過 -> 未能測試")}
    res["primaries"]["P1_E1_LLR聚合"] = p1

    # ---- P2
    P2 = p["p2"]
    p2 = {"state": UNTESTABLE, "note": "資料不足"}
    m = _load(R / "e2" / "analysis_clip_b32_main.json")
    if m and "delta_holistic" in m:
        st, note = decide_p2(m["delta_holistic"], m["max_adjacent_drop_pp"], P2["min_delta_pp"], P2["max_adjacent_drop_pp"])
        used = "main"
        if st == GRAY:
            ma = _load(R / "e2" / "analysis_clip_b32_main+addon.json")
            if ma and "delta_holistic" in ma:
                st, note = decide_p2(ma["delta_holistic"], ma["max_adjacent_drop_pp"], P2["min_delta_pp"], P2["max_adjacent_drop_pp"])
                note = "主資料為灰區 -> 加碼（第二組 seed 的查詢合併）後：" + note
                m, used = ma, "main+addon"
        p2 = {"metric": "holistic 錯誤率 Δ = err(k_top) - err(k=1) (pp)", "used": used, "k_top": m["k_top"],
              "n_per_k": m["n_per_k"], "delta": m["delta_holistic"], "err_series_pp": m["holistic_err_series_pp"],
              "max_adjacent_drop_pp": m["max_adjacent_drop_pp"], "state_raw": st,
              "state": st if gate_ok else UNTESTABLE, "note": note + ("" if gate_ok else "；但閘門未全過 -> 未能測試")}
    elif m:
        p2 = {"state": UNTESTABLE, "note": f"k_top={m['k_top']} 小於最低 {P2['min_k_top']} 或樣本不足", "n_per_k": m["n_per_k"]}
    res["primaries"]["P2_E2_條件數稀釋"] = p2

    # ---- 次要（描述用，不參與判準）
    sec = {}
    for model, a in (("clip_b32", a_clip), ("cs_clip", a_cs)):
        if a:
            sec[model] = {s: {agg: v["aggs"][agg]["all"]["cor"]["point"] for agg in v["aggs"]} for s, v in a["subsets"].items()}
    res["secondary_cor_by_aggregator"] = sec
    res["secondary_e2"] = {}
    for model in ("clip_b32", "cs_clip"):
        e = _load(R / "e2" / f"analysis_{model}_main+addon.json") or _load(R / "e2" / f"analysis_{model}_main.json")
        if e:
            res["secondary_e2"][model] = {a: {k: {"n": v.get("n"), "pair_acc": v.get("pair_acc"),
                                                   "recall_at_50": v.get("recall_at_50_mean")} for k, v in d.items()}
                                          for a, d in e["per_agg"].items()}
    res["reranker"] = {"state": "未測", "note": _load(R / "reranker_status.json") or "CORE reranker 未取得可用程式（見 README），未評估；90% 窄化旗標無法判定。"}
    states = [v["state"] for v in res["primaries"].values()]
    # PREREG §5「任一主指標 FAIL → 停損」優先於另一主指標的「未能測試」（閘門未過時 state 已全被覆寫為未能測試，不受影響）
    # 審查修正 2026-11-14：原本 UNTESTABLE 先判，會讓「P1 FAIL + P2 資料不足」顯示為未能測試。
    if FAIL in states:
        overall = "停損（至少一個主指標 FAIL）"
    elif UNTESTABLE in states:
        overall = UNTESTABLE
    elif all(s == PASS for s in states):
        overall = "繼續（兩個主指標皆 PASS）"
    else:
        overall = "灰區（交使用者定案）"
    res["overall"] = overall

    out = Path(args.out) if args.out else R / "aggregate"
    out.parent.mkdir(parents=True, exist_ok=True)
    Path(str(out) + ".json").write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n")
    Path(str(out) + ".md").write_text(render(res))
    print(render(res))


def render(res):
    L = ["# 機械彙整結果（由 aggregate.py 產生，未含人工判斷）", ""]
    L.append(f"整體：**{res['overall']}**；閘門全過：{res['gate_all_pass']}")
    L += ["", "## 閘門"]
    for k, v in res["gates"].items():
        L.append(f"- {k}：{'通過' if v['pass'] else '未通過'}")
        L.append("  ```\n  " + json.dumps({a: b for a, b in v.items() if a != 'pass'}, ensure_ascii=False, indent=1).replace("\n", "\n  ") + "\n  ```")
    L += ["", "## 主判準"]
    for k, v in res["primaries"].items():
        L.append(f"- {k}：**{v['state']}**")
        L.append("  ```\n  " + json.dumps(v, ensure_ascii=False, indent=1).replace("\n", "\n  ") + "\n  ```")
    L += ["", "## 次要（描述用）", "```", json.dumps(res["secondary_cor_by_aggregator"], ensure_ascii=False, indent=1), "```",
          "", f"reranker：{res['reranker']}", ""]
    return "\n".join(L)


if __name__ == "__main__":
    main()
