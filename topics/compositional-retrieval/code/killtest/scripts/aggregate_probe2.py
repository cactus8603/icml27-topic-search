"""第二探針的機械判定（PREREG_probe2.md）。不含人工判斷。須等 PREREG_probe2.md 與本檔 commit 凍結後才可對真實資料執行。
讀 results/probe2/analysis_cs_clip.json（主）、analysis_clip_b32.json（描述）、precheck.json（描述）。
閘門任一失敗 -> 「未能測試」（不輸出 PASS／FAIL）。
用法：python scripts/aggregate_probe2.py [--results results/probe2] [--out results/probe2/aggregate_probe2]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import EXP

PASS, GRAY, FAIL, UNTESTABLE = "PASS", "灰區", "FAIL", "未能測試"
REJECT_TEXT = ("可淘汰。理由：免訓練、固定索引的雙塔證據聚合，邊際和條件兩種形式都贏不了 CS-CLIP；"
               "剩下的路線已被 CS-CLIP 和 CORE 佔住，沒有可區隔的切入點。")


def decide_primary(diff, min_delta):
    """diff: 配對差 {'point','lo_1s','hi_1s'}（pp，單尾 alpha=0.025）。
    PASS：point >= min_delta 且 lo_1s > 0。FAIL：hi_1s < min_delta。其餘灰區（不加碼，交使用者）。"""
    if diff["point"] >= min_delta and diff["lo_1s"] > 0:
        return PASS, "點估計 >= 5pp 且單尾下界 > 0"
    if diff["hi_1s"] < min_delta:
        return FAIL, "單尾上界 < 5pp"
    return GRAY, "信賴區間跨越 5pp 門檻；灰區不加碼，交使用者"


def eval_gates(pos_cor, neg_cor, pairwise_acc, holistic_cor, g):
    """回傳 (all_pass, dict)。pos_cor 為 None（控制組 embedding 缺）-> 該閘門視為未通過。
    成對正確率門檻 = holistic_cor + margin（嚴格大於才通過）。"""
    thr = holistic_cor + g["pairwise_margin_over_holistic_cor_pp"]
    d = {"pos_control": {"value": pos_cor, "min": g["pos_control_min_cor_pp"],
                         "pass": pos_cor is not None and pos_cor >= g["pos_control_min_cor_pp"]},
         "neg_control": {"value": neg_cor, "max": g["neg_control_max_cor_pp"],
                         "pass": neg_cor is not None and neg_cor <= g["neg_control_max_cor_pp"]},
         "pairwise_upper_bound": {"value": pairwise_acc, "threshold_gt": thr,
                                  "pass": pairwise_acc is not None and pairwise_acc > thr}}
    return all(v["pass"] for v in d.values()), d


def decide(analysis, p):
    """analysis: analysis_cs_clip.json 的內容。回傳 dict（state、note、gates、primary）。"""
    pr, g = p["primary"], p["gates"]
    s = analysis["subsets"].get(pr["subset"]) if analysis else None
    if not s:
        return {"state": UNTESTABLE, "note": "缺主指標資料"}
    ok, gd = eval_gates(s.get("pos_control_cor"), s.get("neg_control_cor"), s["pairwise_acc_stilde"],
                        s["cor_holistic"]["point"], g)
    diff = s["paired_stilde_minus_holistic_pp"]
    raw, note = decide_primary(diff, pr["min_delta_pp"])
    res = {"gates": gd, "gates_pass": ok, "primary_diff": diff, "state_raw": raw, "note_raw": note,
           "cor_stilde": s["cor_stilde"]["point"], "cor_holistic": s["cor_holistic"]["point"]}
    if not ok:
        failed = [k for k, v in gd.items() if not v["pass"]]
        extra = "（構造不可行：PASS 不可能）" if "pairwise_upper_bound" in failed else ""
        res.update(state=UNTESTABLE, note="構造閘門失敗：" + "、".join(failed) + extra + "；記為未能測試，不得據以淘汰")
    else:
        res.update(state=raw, note=note)
        if raw == FAIL:
            res["rejection_text"] = REJECT_TEXT
    return res


def describe(a, p):
    """描述項（不參與判定）。a 可為 None。"""
    if not a:
        return None
    sub = p["primary"]["subset"]
    out = {}
    for k, v in a["subsets"].items():
        out[k] = {"cor_stilde": v["cor_stilde"]["point"], "cor_holistic": v["cor_holistic"]["point"],
                  "diff_pp": v["paired_stilde_minus_holistic_pp"]["point"], "pairwise_acc_stilde": v["pairwise_acc_stilde"],
                  "pos_control_cor": v.get("pos_control_cor"), "neg_control_cor": v.get("neg_control_cor")}
        if "coco5000_pool" in v:
            out[k]["coco5000_pool"] = {"cor_stilde": v["coco5000_pool"]["cor_stilde"]["point"],
                                       "diff_pp": v["coco5000_pool"]["paired_vs_holistic_pp"]["point"]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(EXP / "results" / "probe2"))
    ap.add_argument("--out")
    a = ap.parse_args()
    R = Path(a.results)
    p = json.loads((EXP / "configs" / "params_probe2.json").read_text())
    ld = lambda f: json.loads((R / f).read_text()) if (R / f).exists() else None
    cs, clip = ld("analysis_cs_clip.json"), ld("analysis_clip_b32.json")
    res = {"primary": decide(cs, p),
           "descriptive": {"cs_clip": describe(cs, p), "clip_b32_same_metric": describe(clip, p),
                           "approach_B_vlm_llr": "未做", "precheck": ld("precheck.json")}}
    res["state"] = res["primary"]["state"]
    out = Path(a.out) if a.out else R / "aggregate_probe2"
    out.parent.mkdir(parents=True, exist_ok=True)
    Path(str(out) + ".json").write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n")
    md = ["# 第二探針機械判定（aggregate_probe2.py 產生，無人工判斷）", "", f"結果：**{res['state']}**", "",
          "```", json.dumps(res["primary"], ensure_ascii=False, indent=1), "```", "",
          "## 描述項（不參與判定）", "```", json.dumps(res["descriptive"], ensure_ascii=False, indent=1), "```", ""]
    Path(str(out) + ".md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
