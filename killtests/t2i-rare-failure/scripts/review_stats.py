"""人工標註完成後計算判準 B（主指標 = 罕見格陽性中標 1 的比例，2 與 ? 視為非真陽性）。
讀 results/review/review_blind.csv 與 review_key.csv。輸出三態：PASS（>=pass）/ FAIL（<stop_below）/ 灰區。
同時檢查對照組：陰性對照被標 1 的比例應 <= control_max_negative_rated_true、百分比陽性對照標 1 應 >= control_min_pct_positive_rated_true，
否則標註者/準則不可靠（記為『未能測試』，需重標）。
"""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load_json
B = load_json("criteria.json")["B"]


def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"),) * 2
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); w = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** .5
    return (c - w) / d, (c + w) / d


def main():
    d = ROOT / "results" / "review"
    b = pd.read_csv(d / "review_blind.csv", dtype=str).fillna(""); k = pd.read_csv(d / "review_key.csv")
    b["item"] = b["item"].astype(int); df = k.merge(b[["item", "label"]], on="item")
    print(f"總 {len(df)}；未標 {int((df.label=='').sum())}；標籤分佈：{df.label.value_counts().to_dict()}")
    if (df.label == "").any(): sys.exit("尚有未標項目，不判定")
    for kind, g in df.groupby("kind"):
        t = (g.label == "1").sum(); lo, hi = wilson(t, len(g))
        print(f"[{kind}] 標 1：{t}/{len(g)} = {t/len(g):.1%}（Wilson95% {lo:.1%}–{hi:.1%}）；標 2：{(g.label=='2').sum()}；標 ?：{(g.label=='?').sum()}")
    neg = df[df.kind == "neg"]; pp = df[df.kind == "pct_pos"]; rp = df[df.kind == "rare_pos"]
    ok_ctrl = True
    if len(neg) and (neg.label == "1").mean() > B["control_max_negative_rated_true"]: ok_ctrl = False; print("對照失敗：陰性對照被標 1 比例過高")
    if len(pp) and (pp.label == "1").mean() < B["control_min_pct_positive_rated_true"]: ok_ctrl = False; print("對照失敗：百分比陽性對照被標 1 比例過低")
    if not ok_ctrl: sys.exit("標註者或準則不可靠：判準 B 記為『未能測試』，重標（限一次，見 PREREG）")
    if len(rp) < B["min_rare_positives"]: sys.exit(f"罕見格陽性只有 {len(rp)} < {B['min_rare_positives']}：資料不足，需補樣")
    frac = (rp.label == "1").mean(); lo, hi = wilson((rp.label == "1").sum(), len(rp))
    print("\n罕見格逐格：")
    for (m, p), g in rp.groupby(["model", "prompt"]): print(f"  {m}/{p}: 標1 {(g.label=='1').sum()}/{len(g)}")
    state = "PASS" if frac >= B["pass_true_positive_fraction"] else ("FAIL" if frac < B["stop_below"] else "灰區")
    print(f"\n判準 B（主指標）：真陽性比例 {frac:.1%}（Wilson95% {lo:.1%}–{hi:.1%}，n={len(rp)}）-> {state}")
    if state == "灰區": print("灰區唯一加碼：改用 MLLM 評審（LlavaGuard）對同一批罕見格陽性再評一次，並與人工比對（PREREG）")


if __name__ == "__main__":
    main()
