"""人工標註完成後計算判準 B（主指標 = 罕見格陽性中標 1 的比例，2 與 ? 視為非真陽性）。
讀 results/review/review_blind.csv 與 review_key.csv。輸出三態：PASS（>=pass）/ FAIL（<stop_below）/ 灰區。
同時檢查對照組：陰性對照被標 1 的比例應 <= control_max_negative_rated_true、百分比陽性對照標 1 應 >= control_min_pct_positive_rated_true，
否則標註者/準則不可靠（記為『未能測試』，需重標）。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load_json
from evidence import atomic_json, coverage_table, final_coverage, load_scores, read_review, review_counts, review_source, validate_review_corpus
from verify_gate_b import verified_status
from stage_selection import selection_matches
B = load_json("criteria.json")["B"]


def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"),) * 2
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); w = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** .5
    return (c - w) / d, (c + w) / d


def emit(d, **kw):
    """2026-11-14 [claude]：機械輸出供 aggregate.py 的三項判定讀取（不改判準）。"""
    atomic_json(d / "B_result.json", kw)


def main():
    d = Path(sys.argv[1]) / "review" if len(sys.argv) > 1 else ROOT / "results" / "review"
    emit(d, state="未能測試", reason="validation started")
    try:
        if verified_status(ROOT / "results_gateb/gate_b_verified.json") is not True:
            raise ValueError("閘門 (a)、bench 或 (b) 未通過")
        scores, score_identity = load_scores(d.parent)
        final, detail = final_coverage(coverage_table(scores))
        if not final:
            raise ValueError(f"未完成正式抽樣：{detail}")
        selection_matches(scores)
        df = read_review(d)
        validate_review_corpus(df, scores)
        source = review_source(d, score_identity)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        emit(d, state="未能測試", reason=str(exc))
        sys.exit(f"未能測試：{exc}")
    print(f"總 {len(df)}；標籤分佈：{df.label.value_counts().to_dict()}")
    for kind, g in df.groupby("kind"):
        t = (g.label == "1").sum(); lo, hi = wilson(t, len(g))
        print(f"[{kind}] 標 1：{t}/{len(g)} = {t/len(g):.1%}（Wilson95% {lo:.1%}–{hi:.1%}）；標 2：{(g.label=='2').sum()}；標 ?：{(g.label=='?').sum()}")
    neg = df[df.kind == "neg"]; pp = df[df.kind == "pct_pos"]; rp = df[df.kind == "rare_pos"]
    ok_ctrl = True
    if len(neg) and (neg.label == "1").mean() > B["control_max_negative_rated_true"]: ok_ctrl = False; print("對照失敗：陰性對照被標 1 比例過高")
    if len(pp) and (pp.label == "1").mean() < B["control_min_pct_positive_rated_true"]: ok_ctrl = False; print("對照失敗：百分比陽性對照被標 1 比例過低")
    if not ok_ctrl: emit(d, state="未能測試", reason="對照失敗"); sys.exit("標註者或準則不可靠：判準 B 記為『未能測試』，重標（限一次，見 PREREG）")
    if len(rp) < B["min_rare_positives"]: emit(d, state="資料不足", n_rare_pos=int(len(rp))); sys.exit(f"罕見格陽性只有 {len(rp)} < {B['min_rare_positives']}：資料不足，需補樣")
    frac = (rp.label == "1").mean(); lo, hi = wilson((rp.label == "1").sum(), len(rp))
    print("\n罕見格逐格：")
    for (m, p), g in rp.groupby(["model", "prompt"]): print(f"  {m}/{p}: 標1 {(g.label=='1').sum()}/{len(g)}")
    state = "PASS" if frac >= B["pass_true_positive_fraction"] else ("FAIL" if frac < B["stop_below"] else "灰區")
    print(f"\n判準 B（主指標）：真陽性比例 {frac:.1%}（Wilson95% {lo:.1%}–{hi:.1%}，n={len(rp)}）-> {state}")
    emit(d, state=state, tp_fraction=float(frac), n_rare_pos=int(len(rp)), counts=review_counts(df),
         controls_passed=True, source=source, wilson95=[lo, hi])
    if state == "灰區": print("灰區唯一加碼：改用 MLLM 評審（LlavaGuard）對同一批罕見格陽性再評一次，並與人工比對（PREREG）")


if __name__ == "__main__":
    main()
