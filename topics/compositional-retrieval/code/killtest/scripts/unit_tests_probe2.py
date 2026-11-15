"""第二探針的單元測試（純 CPU、不需模型）：s~ 手算、控制組函式、aggregate_probe2 的 PASS／FAIL／灰區與三個閘門失敗。
用法：python scripts/unit_tests_probe2.py"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate_probe2 import FAIL, GRAY, PASS, UNTESTABLE, decide, decide_primary, eval_gates
from probe2_condpmi import neg_control_perm, pairwise_acc, pos_control_details, replace_detail, stilde

R = []


def check(name, cond, detail=""):
    R.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name, "" if cond else detail)


def test_stilde():
    # 池 = 3 個單位向量 e0,e1,e2；tau=2；文字 = e0；影像 = e0、e1。
    pool = np.eye(3)
    txt = np.array([[1.0, 0, 0], [1.0, 0, 0]])
    img = np.array([[1.0, 0, 0], [0, 1.0, 0]])
    ref_log = np.log((np.exp(2) + 2) / 3)
    ref = np.array([2 - ref_log, 0 - ref_log])
    v = stilde(img, txt, pool, 2.0)
    check("s~ 手算（池 3 張正交影像，tau=2）", np.allclose(v, ref), v.tolist())
    # 池只有影像自己時 s~ = 0（tau*s - log exp(tau*s)）
    v1 = stilde(img[:1], txt[:1], img[:1], 5.0)
    check("s~ 池只含該影像時為 0", np.allclose(v1, 0), v1.tolist())
    # 文字不同時各用各的 log-mean-exp
    txt2 = np.array([[0, 1.0, 0], [1.0, 0, 0]])
    v2 = stilde(img, txt2, pool, 1.0)
    lme = np.log((np.e + 2) / 3)
    check("s~ 逐列文字各自正規化", np.allclose(v2, [0 - lme, 0 - lme]), v2.tolist())


def test_pairwise():
    check("成對正確率：平手算錯", pairwise_acc([2, 2, 3, 1], [1, 2, 1, 2]) == 50.0)


def test_controls():
    a, e = "a man on a bench", "a man on a bench near a red car"
    check("replace_detail 替換單一新增段", replace_detail(a, e, "with a dog") == "a man on a bench with a dog",
          replace_detail(a, e, "with a dog"))
    e2 = "a man , old , on a bench"
    check("replace_detail 無新增段 -> 接句尾", replace_detail(a, a, "x y") == "a man on a bench x y")
    r = replace_detail("a cat sits", "a big cat sits quietly", "tiny")
    check("replace_detail 多段只在第一段放入", r == "a tiny cat sits", r)
    dp = ["red car", "dog", "red car", "tree", "dog"]
    d1 = pos_control_details(dp, 20261115, 0)
    d2 = pos_control_details(dp, 20261115, 0)
    check("正控制：同 seed 可重現", d1 == d2)
    check("正控制：不得與本列 d+ 相同", all(x != y for x, y in zip(d1, dp)), d1)
    check("正控制：來自別列（取值屬於其他列）", all(x in dp for x in d1))
    perm = neg_control_perm(50, 20261116, 1)
    check("負控制排列：是排列且無不動點", sorted(perm.tolist()) == list(range(50)) and not (perm == np.arange(50)).any())
    check("負控制排列：同 seed 可重現", (perm == neg_control_perm(50, 20261116, 1)).all())
    try:
        pos_control_details(["a", "a"], 1, 0)
        check("正控制無候選時報錯", False)
    except ValueError:
        check("正控制無候選時報錯", True)


P = json.loads((Path(__file__).resolve().parents[1] / "configs" / "params_probe2.json").read_text())


def fake(diff_point, lo, hi, pos=80.0, neg=10.0, pair=90.0, hol=58.0, stc=64.0):
    return {"subsets": {"coco_qwen": {
        "paired_stilde_minus_holistic_pp": {"point": diff_point, "lo_1s": lo, "hi_1s": hi},
        "pos_control_cor": pos, "neg_control_cor": neg, "pairwise_acc_stilde": pair,
        "cor_holistic": {"point": hol}, "cor_stilde": {"point": stc}}}}


def test_decide():
    check("主指標 PASS", decide_primary({"point": 6, "lo_1s": 1, "hi_1s": 11}, 5)[0] == PASS)
    check("主指標 PASS 邊界 point=5 lo>0", decide_primary({"point": 5, "lo_1s": 0.1, "hi_1s": 9}, 5)[0] == PASS)
    check("點估計夠但下界 =0 -> 灰區", decide_primary({"point": 7, "lo_1s": 0.0, "hi_1s": 14}, 5)[0] == GRAY)
    check("主指標 FAIL（上界<5）", decide_primary({"point": -29, "lo_1s": -35, "hi_1s": -23}, 5)[0] == FAIL)
    check("上界剛好 =5 -> 灰區", decide_primary({"point": 1, "lo_1s": -3, "hi_1s": 5.0}, 5)[0] == GRAY)
    check("灰區（區間跨 5）", decide_primary({"point": 3, "lo_1s": -2, "hi_1s": 8}, 5)[0] == GRAY)
    # 整合
    r = decide(fake(6, 1, 11), P)
    check("整合 PASS", r["state"] == PASS and r["gates_pass"], r)
    r = decide(fake(-20, -26, -14), P)
    check("整合 FAIL 並附淘汰理由", r["state"] == FAIL and "rejection_text" in r, r)
    r = decide(fake(3, -2, 8), P)
    check("整合灰區", r["state"] == GRAY, r)
    r = decide(fake(6, 1, 11, pos=69.9), P)
    check("正控制 <70 -> 未能測試（即使主指標 PASS）", r["state"] == UNTESTABLE and "pos_control" in r["note"], r)
    r = decide(fake(-20, -26, -14, pos=69.9), P)
    check("正控制失敗且主指標 FAIL -> 仍為未能測試、無淘汰理由", r["state"] == UNTESTABLE and "rejection_text" not in r, r)
    r = decide(fake(6, 1, 11, neg=20.1), P)
    check("負控制 >20 -> 未能測試", r["state"] == UNTESTABLE and "neg_control" in r["note"], r)
    r = decide(fake(6, 1, 11, pair=63.0, hol=58.0), P)
    check("成對正確率 63.0 <= 58+5 -> 構造不可行", r["state"] == UNTESTABLE and "構造不可行" in r["note"], r)
    r = decide(fake(6, 1, 11, pair=63.01, hol=58.0), P)
    check("成對正確率 63.01 > 63 -> 閘門通過", r["state"] == PASS, r)
    r = decide(fake(6, 1, 11, pos=None), P)
    check("正控制資料缺 -> 未能測試", r["state"] == UNTESTABLE, r)
    ok, _ = eval_gates(70.0, 15.0, 63.2, 58.15, P["gates"])
    check("閘門邊界：pos=70、neg=20 皆通過；pair 63.2 > 63.15", ok)
    ok, _ = eval_gates(70.0, 15.0, 63.15, 58.15, P["gates"])
    check("閘門邊界：pair = 63.15 不通過（嚴格大於）", not ok)


if __name__ == "__main__":
    test_stilde(); test_pairwise(); test_controls(); test_decide()
    print(f"{sum(R)}/{len(R)} passed")
    sys.exit(0 if all(R) else 1)
