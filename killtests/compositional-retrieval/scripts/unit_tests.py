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


def test_decisions():
    ci = lambda p, lo, hi: {"point": p, "lo": lo, "hi": hi}
    check("P1 下界>=45 -> PASS", decide_p1(ci(50, 46, 54), 45)[0] == PASS)
    check("P1 上界<45 -> FAIL（不救援）", decide_p1(ci(40, 36, 44), 45, {"point": 9, "lo": 3})[0] == FAIL)
    check("P1 跨越門檻、無加碼資料 -> 灰區", decide_p1(ci(44, 40, 48), 45)[0] == GRAY)
    check("P1 灰區+加碼通過 -> PASS", decide_p1(ci(44, 40, 48), 45, {"point": 6, "lo": 1})[0] == PASS)
    check("P1 灰區+加碼不過+點<門檻 -> FAIL", decide_p1(ci(44, 40, 48), 45, {"point": 2, "lo": -1})[0] == FAIL)
    check("P1 灰區+加碼不過+點>=門檻 -> 灰區（未解）", decide_p1(ci(46, 42, 50), 45, {"point": 2, "lo": -1})[0] == GRAY)
    check("P1 加碼點>=5 但配對下界<=0 -> 不算通過", decide_p1(ci(44, 40, 48), 45, {"point": 6, "lo": -0.5})[0] != PASS)
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
    test_cor(); test_pmi(); test_anchor_cancellation(); test_detail(); test_bootstrap(); test_decisions()
    out = test_e2_builder()
    test_e2_analyze(out)
    allp = all(r["pass"] for r in RESULTS)
    jdump({"all_pass": allp, "n": len(RESULTS), "results": RESULTS}, args.out)
    print(f"\n{'全部通過' if allp else '有失敗'}：{sum(r['pass'] for r in RESULTS)}/{len(RESULTS)}")
    sys.exit(0 if allp else 1)


if __name__ == "__main__":
    main()
