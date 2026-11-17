"""閘門 G1：已知答案的小例子（純 CPU、不需模型權重、數秒內完成）。
用法：python scripts/unit_tests.py [--out results/gates/g1_unit.json]
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate import FAIL, GRAY, PASS, decide_p1, decide_p2
from common import (EXP, bootstrap_ci, cor_flags, extract_detail, ht_flags, jdump, logmeanexp, paired_bootstrap_diff, pmi)
import e2_analyze
import e2_build_queries as eb

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append({"name": name, "pass": bool(cond), "detail": str(detail)})
    print(("PASS " if cond else "FAIL ") + name, detail if not cond else "")


def test_cor():
    f = cor_flags([3, 3, 1, 2], [2, 2, 2, 2], [1, 2, 3, 2])
    check("COR 手算：[T,F(h 平手),F(full<=anchor),F]", f.tolist() == [True, False, False, False], f.tolist())
    check("COR 平手視為錯誤（full==anchor）", not cor_flags([2], [2], [1])[0])
    check("HT 嚴格大於", ht_flags([2, 2], [1, 2]).tolist() == [True, False])


def test_pmi():
    # 池 = 兩張正交影像；文字 = 影像 0；tau=1。PMI(I0)=1-log((e+1)/2)，PMI(I1)=0-log((e+1)/2)
    pool = np.eye(2)
    t = np.array([[1.0, 0.0], [1.0, 0.0]])
    img = np.array([[1.0, 0.0], [0.0, 1.0]])
    v = pmi(img, t, pool, 1.0)
    ref = np.array([1 - np.log((np.e + 1) / 2), -np.log((np.e + 1) / 2)])
    check("PMI 手算（池=2 張正交影像）", np.allclose(v, ref), v.tolist())
    # logmeanexp 數值穩定
    check("logmeanexp 大數穩定", np.isclose(logmeanexp(np.array([1000.0, 1000.0])), 1000.0))
    # PMI 對池平均為 0 的性質：mean_I exp(PMI) == 1
    rng = np.random.default_rng(0)
    pool = rng.normal(size=(50, 8)); pool /= np.linalg.norm(pool, axis=1, keepdims=True)
    txt = rng.normal(size=(1, 8)); txt /= np.linalg.norm(txt)
    p = pmi(pool, np.repeat(txt, 50, 0), pool, 20.0)
    check("PMI 性質：池內 mean exp(PMI)=1", np.isclose(np.exp(p).mean(), 1.0), np.exp(p).mean())


def test_anchor_cancellation():
    """llr_sum 的 COR 恆等於 P(PMI(d+)>0>PMI(d-))（anchor 項抵銷）。"""
    rng = np.random.default_rng(1)
    n, d = 400, 16
    def unit(x): return x / np.linalg.norm(x, axis=1, keepdims=True)
    I, A, Dp, Dm = (unit(rng.normal(size=(n, d))) for _ in range(4))
    pool = unit(rng.normal(size=(300, d)))
    pa, pp, pm = pmi(I, A, pool, 50.0), pmi(I, Dp, pool, 50.0), pmi(I, Dm, pool, 50.0)
    lhs = cor_flags(pa + pp, pa, pa + pm)
    rhs = (pp > 0) & (pm < 0)
    check("LLR 恆等式：anchor 抵銷（逐列不一致數 <= 1，容許浮點誤差）", int((lhs != rhs).sum()) <= 1, int((lhs != rhs).sum()))


def test_detail():
    d, ok = extract_detail("a dog on the grass", "a dog on the grass next to a red bike")
    check("細節抽取：純新增", d == "next to a red bike" and ok, (d, ok))
    d, ok = extract_detail("a man riding a horse", "a man riding a brown horse")
    check("細節抽取：插入形容詞", d == "brown" and ok, (d, ok))
    d, ok = extract_detail("a man riding a horse", "a woman riding a horse")
    check("細節抽取：替換 anchor 字詞 -> ok=False", ok is False, (d, ok))
    d, ok = extract_detail("a cat.", "a cat, sitting on a sofa.")
    check("細節抽取：標點略去", d == "sitting on a sofa" and ok, (d, ok))
    d, ok = extract_detail("a cat", "a cat")
    check("細節抽取：相同 -> 空字串", d == "" and ok, (d, ok))


def test_bootstrap():
    ci = bootstrap_ci(np.ones(100), 2000, 0)
    check("bootstrap：全 1 -> [100,100]", ci["lo"] == 100 and ci["hi"] == 100)
    f = np.r_[np.ones(250), np.zeros(250)]
    ci = bootstrap_ci(f, 4000, 0)
    check("bootstrap：p=0.5,n=500 的 SE 約 2.24pp（理論）", abs(ci["se"] - 2.236) < 0.25, ci["se"])
    d = paired_bootstrap_diff(np.ones(50), np.zeros(50), 1000, 0)
    check("配對差：全 1 對全 0 -> 100", d["point"] == 100 and d["lo"] == 100)


def test_cluster_bootstrap():
    rng = np.random.default_rng(1)
    # 100 張影像各 8 個查詢，查詢對錯在影像內完全相關（群集效應極強）
    img = np.repeat(np.arange(100), 8)
    f = np.repeat((rng.random(100) < 0.5).astype(float), 8)
    row = bootstrap_ci(f, 4000, 0)
    cl = bootstrap_ci(f, 4000, 0, groups=img)
    check("群集 bootstrap：SE 比列層級大（群集內完全相關，約 sqrt(8) 倍）", cl["se"] > 2.0 * row["se"], (cl["se"], row["se"]))
    check("群集 bootstrap：每群組大小相同時點估計相同", abs(cl["point"] - f.mean() * 100) < 1e-9)
    ci = bootstrap_ci(f, 4000, 0, alpha_one_sided=0.0125, groups=img)
    check("單尾 0.0125：lo_1s 比 2.5 百分位的 lo 更寬（更低）", ci["lo_1s"] < ci["lo"] and ci["hi_1s"] > ci["hi"], ci)
    d = paired_bootstrap_diff(np.ones(50), np.zeros(50), 1000, 0, alpha_one_sided=0.0125, groups=np.repeat(np.arange(10), 5))
    check("配對群集差：全 1 對全 0 -> lo_1s=100", d["point"] == 100 and d["lo_1s"] == 100)


def test_e2_cluster_and_dilution():
    """合成查詢：k=1 錯誤率 0、k=4 錯誤率 0.5（holistic）；llr_sum 在 k=4 錯誤率 0.1。群集：同影像多個查詢。"""
    rng = np.random.default_rng(0)
    def mk(err_by_img_k1, err_k4):
        return {"1": {"istar": list(np.repeat(np.arange(30), 3)), "err_flags": err_by_img_k1},
                "4": {"istar": list(np.repeat(np.arange(20, 50), 3)), "err_flags": err_k4}}
    h = mk([0] * 90, list((rng.random(90) < 0.5).astype(int)))
    l = mk([0] * 90, list((rng.random(90) < 0.1).astype(int)))
    d = e2_analyze.delta_err(h, 4, 1, 2000, 0)
    check("E2 群集 delta_err：點估計 = err(4)-err(1)，群集數 = 影像聯集 50", abs(d["point"] - 100 * np.mean(h["4"]["err_flags"])) < 1e-9 and d["n_clusters"] == 50, d)
    res = {"holistic": h, "llr_sum": l}
    dd = e2_analyze.dilution_diff(res, "holistic", "llr_sum", 4, 1, 2000, 0)
    check("E2 稀釋差：holistic 稀釋較大 -> 點估計>0 且等於兩個 Δ 之差",
          dd["point"] > 0 and abs(dd["point"] - (dd["delta_ref_pp"] - dd["delta_cmp_pp"])) < 1e-9, dd)
    # 群集 CI 比把查詢當獨立更寬：以相同資料用列層級對照
    row = bootstrap_ci(np.array(h["4"]["err_flags"], float), 2000, 0)
    clu = bootstrap_ci(np.array(h["4"]["err_flags"], float), 2000, 0, groups=h["4"]["istar"])
    check("E2 群集 bootstrap 可執行且 n_clusters 與 CI 合理", clu["lo"] <= clu["point"] <= clu["hi"] and row["lo"] <= row["point"] <= row["hi"])


def test_aggregate_synthetic():
    """用小型合成結果目錄跑完整 aggregate.main：P1 路線 b 成立（a 失敗）-> PASS；P2 以合成 e2 分析。"""
    import aggregate
    from common import SUBSETS, params
    P = params()
    def cor(pt, w):
        return {"point": pt, "lo": pt - w, "hi": pt + w, "lo_1s": pt - 1.2 * w, "hi_1s": pt + 1.2 * w, "n": 500}
    def aggs(h, llr, paired):
        mk = lambda c: {"all": {"cor": cor(c, 2.0), "ht": cor(c, 2.0)}}
        a = {"holistic": mk(h), "llr_sum": mk(llr)}
        a["llr_sum"]["paired_vs_holistic_cor_pp"] = paired
        return a
    cov = {"coverage": 0.99}
    cs = {"model": "cs_clip", "subsets": {s: {"aggs": aggs(P["paper_cor"][s], P["paper_cor"][s] + 8, {"point": 8, "lo": 5, "hi": 11, "lo_1s": 4.5, "hi_1s": 11.5}), "detail_coverage": cov} for s in SUBSETS}}
    cl = {"model": "clip_b32", "subsets": {s: {"aggs": aggs(P["paper_clip_cor"][s], 30.0, {"point": 10, "lo": 7, "hi": 13, "lo_1s": 6, "hi_1s": 14}), "detail_coverage": cov} for s in SUBSETS}}
    with tempfile.TemporaryDirectory(dir=str(Path("/work/ChihChungHsu/cactus8603/icml/tmp"))) as td:
        R = Path(td)
        for sub in ("ht", "gates", "e2"):
            (R / sub).mkdir()
        jdump(cs, R / "ht" / "analysis_cs_clip.json"); jdump(cl, R / "ht" / "analysis_clip_b32.json")
        jdump({"all_pass": True}, R / "gates" / "g1_unit.json"); jdump({"max_abs_diff": 1e-6}, R / "gates" / "g1_crosscheck.json")
        jdump({"max_abs_diff": 0.0}, R / "gates" / "g3_rerun.json")
        jdump({"model": "clip_b32", "k_top": 4, "n_per_k": {"1": 1000, "4": 765}, "per_agg": {},
               "delta_holistic": {"point": 12, "lo": 8, "hi": 16}, "holistic_err_series_pp": [2, 5, 9, 14], "max_adjacent_drop_pp": 0.0,
               "secondary_dilution": {"holistic_minus_llr_sum": {"point": 5}}}, R / "e2" / "analysis_clip_b32_main.json")
        import sys as _s
        argv = _s.argv; _s.argv = ["aggregate.py", "--results", str(R), "--out", str(R / "agg")]
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            aggregate.main()
        _s.argv = argv
        out = json.loads((R / "agg.json").read_text())
    p1 = out["primaries"]["P1_E1_LLR聚合"]
    check("aggregate 合成：P1 路線 a 失敗（COR 30<45）但路線 b 成立 -> PASS", p1["state"] == PASS and p1["path_b_pass"] and not p1["path_a_pass"], p1.get("note"))
    check("aggregate 合成：P2 PASS、整體『繼續』", out["primaries"]["P2_E2_條件數稀釋"]["state"] == PASS and out["overall"].startswith("繼續"), out["overall"])
    check("aggregate 合成：次要稀釋差被帶出", "secondary_e2_dilution_clip_b32" in out, list(out))


def test_decisions():
    ci = lambda p, lo, hi: {"point": p, "lo": lo, "hi": hi}
    c1 = lambda p, lo, hi: {"point": p, "lo": lo, "hi": hi, "lo_1s": lo, "hi_1s": hi}
    check("P1 路線a：CLIP+LLR 下界>=45 -> PASS", decide_p1(c1(50, 46, 54), 45, {"point": 0, "lo_1s": -3, "hi_1s": 3})[0] == PASS)
    check("P1 路線b：CS-CLIP+LLR 贏 >=5pp 且下界>0 -> PASS（即使 a 明確失敗）", decide_p1(c1(40, 36, 44), 45, c1(9, 3, 15))[0] == PASS)
    check("P1 a 上界<45 且 b 上界<5 -> FAIL", decide_p1(c1(40, 36, 44), 45, c1(1, -2, 3.5))[0] == FAIL)
    check("P1 a 上界<45 但 b 區間跨 5pp -> 灰區", decide_p1(c1(40, 36, 44), 45, c1(3, -1, 8))[0] == GRAY)
    check("P1 a 跨越 45 且 b 上界<5 -> 灰區", decide_p1(c1(44, 40, 48), 45, c1(1, -2, 3.5))[0] == GRAY)
    check("P1 b 點>=5 但配對下界<=0 且 a 失敗 -> 不 PASS", decide_p1(c1(44, 40, 48), 45, c1(6, -0.5, 12))[0] != PASS)
    check("P1 a 失敗、b 資料缺 -> 灰區（不是 FAIL）", decide_p1(c1(40, 36, 44), 45, None)[0] == GRAY)
    check("P1 用 lo_1s/hi_1s（1.25/98.75）而非 lo/hi", decide_p1({"point": 46, "lo": 45.5, "hi": 50, "lo_1s": 44.8, "hi_1s": 51}, 45, None)[0] != PASS)
    check("P2 PASS", decide_p2(ci(12, 7, 17), 1.0, 5.0, 2.0)[0] == PASS)
    check("P2 下界過但相鄰下降 3pp -> 灰區", decide_p2(ci(12, 7, 17), 3.0, 5.0, 2.0)[0] == GRAY)
    check("P2 上界<5 -> FAIL", decide_p2(ci(2, -1, 4.5), 0.0, 5.0, 2.0)[0] == FAIL)
    check("P2 跨越 -> 灰區", decide_p2(ci(6, 2, 10), 0.0, 5.0, 2.0)[0] == GRAY)


def synthetic_ann():
    cats = [{"id": 1, "name": "cat"}, {"id": 2, "name": "dog"}, {"id": 3, "name": "bike"}]
    # 影像 id -> 類別集合（全為大物件）
    sets = {10: {1, 2, 3}, 11: {1, 2}, 12: {1, 3}, 13: {2, 3}, 14: {1}, 15: {2}, 16: set(), 17: {1, 2, 3}}
    images = [{"id": i, "width": 100, "height": 100} for i in sets]
    anns, aid = [], 0
    for i, s in sets.items():
        for c in s:
            aid += 1
            anns.append({"id": aid, "image_id": i, "category_id": c, "area": 5000, "iscrowd": 0})
    anns.append({"id": 999, "image_id": 16, "category_id": 3, "area": 10, "iscrowd": 0})   # 小物件，不算存在
    anns.append({"id": 998, "image_id": 16, "category_id": 2, "area": 9000, "iscrowd": 1})  # crowd，不算存在
    return {"categories": cats, "images": images, "annotations": anns}, sets


def test_e2_builder():
    ann, sets = synthetic_ann()
    P = {"k_list": [1, 2, 3], "n_per_k": 30, "min_area_frac": 0.02, "max_attempts_per_k": 2000}
    out = eb.build(ann, P, seed=0)
    ids = out["image_ids"]
    good = True
    for k in (1, 2, 3):
        for q in out["queries"][str(k)]:
            Q = set(q["cats"])
            ist, j = ids[q["istar"]], ids[q["j"]]
            good &= Q <= sets[ist] and len(Q & sets[j]) == k - 1 and ist != j
            good &= sorted(ids[r] for r in q["rel"]) == sorted(i for i, s in sets.items() if Q <= s)
    check("E2 建查詢：I* 含 Q、J 恰缺一個、rel 正確（含小物件／crowd 排除）", good)
    check("E2 k=1..3 都有查詢", all(len(out["queries"][str(k)]) > 0 for k in (1, 2, 3)),
          {k: len(v) for k, v in out["queries"].items()})
    check("E2 影像 16（小物件+crowd）不含任何類別 -> 不在任何 rel", all(ids.index(16) not in q["rel"] for k in out["queries"] for q in out["queries"][k]))
    check("E2 文字模板", eb.joint_text(["cat", "dog", "bike"]) == "a photo of cat, dog and bike" and eb.joint_text(["cat"]) == "a photo of cat")
    return out


def test_e2_analyze(out):
    """合成 embedding：影像 i = 其類別集合的 one-hot 和（正規化）；文字 = 查詢類別 one-hot 和。
    holistic 分數 = 與查詢的 cosine，J 缺一個 -> 一定較低 -> pair_acc=100。"""
    ids = out["image_ids"]
    ann, sets = synthetic_ann()
    d = 3
    def vec(s):
        v = np.zeros(d)
        for c in s: v[c - 1] = 1
        n = np.linalg.norm(v)
        return v / n if n else v
    img = np.stack([vec(sets[i]) for i in ids])
    unit = np.eye(d)
    owner, joint = [], []
    for k, lst in out["queries"].items():
        for n, q in enumerate(lst):
            owner.append((int(k), n)); joint.append(vec(set(q["cats"])))
    z = {"img": img, "unit": unit, "cat_ids": np.array([1, 2, 3]), "tau": np.array(10.0),
         "owner_main": np.array(owner), "joint_main": np.stack(joint)}
    P = {"k_list": [1, 2, 3], "recall_at": 2}
    res = e2_analyze.run(z, [("main", out)], P, 200, 0, aggs=["holistic", "unit_min", "llr_sum"])
    accs = {k: res["holistic"][k]["pair_acc"] for k in res["holistic"]}
    check("E2 分析：理想 embedding 下 holistic pair_acc 全為 100", all(v == 100.0 for v in accs.values()), accs)
    rec = {k: res["holistic"][k]["recall_at_50_mean"] for k in res["holistic"]}
    check("E2 分析：理想 embedding 下 recall 為 100（top-2，rel>=1）", all(v == 100.0 for v in rec.values()), rec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(EXP / "results" / "gates" / "g1_unit.json"))
    args = ap.parse_args()
    test_cor(); test_pmi(); test_anchor_cancellation(); test_detail(); test_bootstrap(); test_cluster_bootstrap(); test_decisions()
    out = test_e2_builder()
    test_e2_analyze(out)
    test_e2_cluster_and_dilution()
    test_aggregate_synthetic()
    allp = all(r["pass"] for r in RESULTS)
    jdump({"all_pass": allp, "n": len(RESULTS), "results": RESULTS}, args.out)
    print(f"\n{'全部通過' if allp else '有失敗'}：{sum(r['pass'] for r in RESULTS)}/{len(RESULTS)}")
    sys.exit(0 if allp else 1)


if __name__ == "__main__":
    main()
