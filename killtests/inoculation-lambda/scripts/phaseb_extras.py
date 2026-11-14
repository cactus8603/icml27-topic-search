#!/usr/bin/env python
"""Phase B 的現象、訊號、空間與 L1–L3，依 PREREG.md §3–§3a。

輸入（Phase B 訓練與 Λ 批次腳本須產出）：
  --leak     leakage.csv     task,arm,family,seed,leak,lam,base_pred,id_ok  （同 aggregate --phase b；
                              leak 是 [0, 1] 比例，seed 是訓練 seed，每 arm 含 seed 0 與 1）
  --lamseeds lambda_seeds.csv task,arm,lora_seed,lam                        （LoRA 初始化 seed 0,1,2 各一列）
  --horizon  horizon.csv     task,arm,lam_0,lam_1p25,lam_2p5,lam_5,lam_id,dtau5,dtau_final
                              lam_k = 在該 arm seed 0 的 k% 步 checkpoint 上算的 Λ；lam_id = ID 資料的 Λ；
                              dtau5 = 5% 步 τ − step 0 τ；dtau_final = 最終 τ − step 0 τ
  --gates    gates.json      gate1、gate2、gate3 各自為 "PASS" 等正式判定，須全過
  --comparators comparators.csv task,arm,grant_cosine,wichers_elicitation,tan_loss_drop,phm_persona_distance
                              每個有防禦 arm 一列，值越大表示預測漏出越多；空欄表示方法明確不適用
  --addon-from phaseb_extras.json  唯一一次訊號灰區加碼的初判結果，與後續完整輸入表併用
用法：phaseb_extras.py --leak ... --lamseeds ... --gates ... --horizon ... --comparators ... --outdir ...
      phaseb_extras.py --selftest
"""
import argparse
import csv
import itertools
import json
import math
import os
import sys

import numpy as np
import yaml
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import aggregate as AG  # noqa: E402

PASS = "PASS"
FAIL = "FAIL"
GRAY = "GRAY"
UNTESTABLE = "UNTESTABLE"


def sb(r, k):
    """Spearman-Brown：單一重複的信度 r 校正到 k 個重複。"""
    if r is None or math.isnan(r):
        return float("nan")
    denominator = 1 + (k - 1) * r
    return k * r / denominator if denominator > 0 else 0.0


def rho(a, b):
    if len(a) < 3 or len(set(a)) < 2 or len(set(b)) < 2 or not np.isfinite([a, b]).all():
        return float("nan")
    return float(stats.spearmanr(a, b)[0])


def zpool(vals, weights=None):
    if weights is None:
        weights = [1] * len(vals)
    present = [(v, w) for v, w in zip(vals, weights, strict=True) if v is not None and math.isfinite(v)]
    if not present:
        return float("nan")
    vals, weights = zip(*present)
    z = [np.arctanh(np.clip(v, -0.999999, 0.999999)) for v in vals]
    return float(np.tanh(np.average(z, weights=weights)))


def guarded_rho(arms, predictor="lam"):
    """每個任務至少四個 arm，全部任務都要有可定義的相關。"""
    per_task = {task: rho([a[predictor] for a in rows.values()], [a["leak"] for a in rows.values()])
                if len(rows) >= 4 else float("nan") for task, rows in arms.items()}
    if not per_task or any(not math.isfinite(v) for v in per_task.values()):
        return None, {t: v if math.isfinite(v) else None for t, v in per_task.items()}
    return zpool(list(per_task.values()), [len(rows) - 3 for rows in arms.values()]), per_task


def bootstrap_correlations(arms, predictors, c):
    """配對重抽任務內的 arm，跨任務以 n−3 加權；常數重抽排除。"""
    rng = np.random.default_rng(c["bootstrap_seed"])
    boots = [[] for _ in predictors]
    weights = []
    for rows in arms.values():
        if len(rows) < 4:
            return np.empty((len(predictors), 0))
        values = list(rows.values())
        idx = rng.integers(0, len(rows), (c["bootstrap_n"], len(rows)))
        arrays = [np.asarray([a[name] for a in values]) for name in ("leak", *predictors)]
        ranks = [stats.rankdata(values[idx], axis=1) for values in arrays]
        norms = []
        for rank in ranks:
            rank -= rank.mean(axis=1, keepdims=True)
            norms.append(np.linalg.norm(rank, axis=1))
        for b, rank, norm in zip(boots, ranks[1:], norms[1:], strict=True):
            divisor = norm * norms[0]
            corr = np.divide((rank * ranks[0]).sum(axis=1), divisor,
                             out=np.full(len(idx), np.nan), where=divisor > 0)
            b.append(np.arctanh(np.clip(corr, -0.999999, 0.999999)))
        weights.append(len(rows) - 3)
    combined = np.array([np.tanh(np.average(b, axis=0, weights=weights)) for b in boots])
    return combined[:, np.all(np.isfinite(combined), axis=0)]


def interval(boots, level):
    if not len(boots):
        return [None, None]
    q = (1 - level) / 2
    return np.quantile(boots, [q, 1 - q]).tolist()


def signal_stats(arms, crit):
    x, c = crit["phase_b_extras"], crit["primary"]
    point, per_task = guarded_rho(arms)
    families = sorted({a["family"] for rows in arms.values() for a in rows.values()})
    drop = {}
    for family in families:
        remaining = {t: {k: a for k, a in rows.items() if a["family"] != family} for t, rows in arms.items()}
        drop[family] = guarded_rho(remaining)[0]
    boots = bootstrap_correlations(arms, ["lam"], c)[0] if point is not None else []
    ci = interval(boots, x["signal"]["ci_level"])
    old_ci = interval(boots, c["ci_level"])
    tasks_positive = sum(r is not None and r > 0 for r in per_task.values())
    testable = point is not None and ci[0] is not None and all(v is not None for v in drop.values())
    strong = (testable and bool(drop) and point >= c["pooled_rho_pass"] and old_ci[0] >= c["ci_lower_pass"]
              and tasks_positive >= c["min_tasks_positive"] and min(drop.values()) >= c["drop_family_min"])
    return dict(pooled_rho=point, ci=ci, ci_level=x["signal"]["ci_level"], per_task_rho=per_task,
                tasks_positive=tasks_positive, drop_family=drop, testable=testable,
                strong_signal_descriptive=bool(strong), strong_signal_ci=old_ci,
                bootstrap_valid_replicates=len(boots), n_arms={t: len(rows) for t, rows in arms.items()})


def signal_pass(m, rules):
    return (m["testable"] and m["pooled_rho"] >= rules["rho_min"] and m["ci"][0] > rules["ci_lower_gt"]
            and m["tasks_positive"] >= rules["min_tasks_positive"]
            and all(v > rules["drop_family_gt"] for v in m["drop_family"].values()))


def canonical_lambda_rows(leak_rows, lamseed_rows, x):
    """Derive every primary Λ from the exact three registered LoRA seeds."""
    keys = {(r["task"], r["arm"]) for r in leak_rows if r["arm"] != "none"}
    required = set(x["ceiling_lambda_seeds"])
    per_arm = {key: {} for key in keys}
    for r in lamseed_rows:
        key = r["task"], r["arm"]
        if key not in keys:
            continue
        seed = int(r["lora_seed"])
        value = float(r["lam"])
        if seed not in required or seed in per_arm[key] or not math.isfinite(value):
            raise ValueError(f"lambda_seeds.csv 的 LoRA seed 或 Λ 無效：{key}/{seed}")
        per_arm[key][seed] = value
    missing = sorted(key for key, values in per_arm.items() if set(values) != required)
    if missing:
        return None, missing
    expected = {key: math.fsum(values.values()) / len(required) for key, values in per_arm.items()}
    mismatched = sorted({(r["task"], r["arm"]) for r in leak_rows if r["arm"] != "none"
                         and not math.isclose(float(r["lam"]), expected[r["task"], r["arm"]], **x["lambda_consistency"])})
    if mismatched:
        return None, mismatched
    return [dict(r, lam=expected[r["task"], r["arm"]]) if r["arm"] != "none" else r for r in leak_rows], []


def phenomenon(arms, gates, reliability, rules):
    range_pp = {t: float(100 * (max(a["leak"] for a in rows.values()) - min(a["leak"] for a in rows.values())))
                for t, rows in arms.items()}
    tasks_with_range = sum(v + 1e-12 >= rules["leak_range_min_pp"] for v in range_pp.values())
    pending = [gate for gate in ("gate1", "gate2", "gate3") if gates.get(gate) != PASS]
    if pending:
        status, reason = UNTESTABLE, "前置閘門未全通過：" + "、".join(pending)
    elif not reliability["sufficient"]:
        status, reason = UNTESTABLE, "信度上限不足或未能估計"
    elif tasks_with_range < rules["min_tasks_with_range"]:
        status, reason = FAIL, "漏出全距達標的任務不足"
    else:
        status, reason = PASS, "前置閘門、信度上限及漏出全距達標"
    return dict(status=status, reason=reason, leak_range_pp=range_pp, tasks_with_range=tasks_with_range,
                prerequisite_gates={k: gates.get(k) for k in ("gate1", "gate2", "gate3")})


def headroom(arms, comparator_rows, reliability, rules, c):
    expected = {(t, arm) for t, rows in arms.items() for arm in rows}
    comparator = {}
    for row in comparator_rows:
        key = row["task"], row["arm"]
        if key in comparator or key not in expected:
            raise ValueError(f"comparators.csv 重複或意外 arm：{key}")
        if not all(name in row and row[name] is not None for name in rules["predictors"]):
            return dict(status=UNTESTABLE, reason="comparators.csv 缺少必算既有預測量欄位")
        vals = {name: None if not row[name].strip() else float(row[name]) for name in rules["predictors"]}
        if any(v is not None and not math.isfinite(v) for v in vals.values()):
            raise ValueError(f"comparators.csv 有非有限預測量：{key}")
        comparator[key] = vals
    if expected != set(comparator):
        return dict(status=UNTESTABLE, reason="comparators.csv 缺少防禦 arm 資料（未知覆蓋不等於無法適用）",
                    missing_arms=sorted(expected - set(comparator)))
    per_predictor, matched = {}, {}
    for name in rules["predictors"]:
        common = {t: {arm: dict(a, existing=comparator[t, arm][name]) for arm, a in rows.items()
                      if comparator[t, arm][name] is not None} for t, rows in arms.items()}
        counts = {t: len(rows) for t, rows in common.items()}
        sufficiently_covered = all(n >= rules["min_covered_arms_per_task"] for n in counts.values())
        pooled, per = guarded_rho(common, "existing") if all(n >= rules["min_arms_for_rho"] for n in counts.values()) else (None, {})
        per_predictor[name] = dict(pooled_rho=pooled, per_task_rho=per, n_covered_arms=counts,
                                   sufficiently_covered=sufficiently_covered)
        matched[name] = common
    eligible = {name: p["pooled_rho"] for name, p in per_predictor.items() if p["pooled_rho"] is not None}
    strongest = max(eligible, key=eligible.get) if eligible else None
    highest = eligible.get(strongest)
    ceiling_value = reliability["C"]
    gap = ceiling_value - highest if highest is not None and math.isfinite(ceiling_value) else None
    covered = [p for p in per_predictor.values() if p["sufficiently_covered"]]
    if not reliability["sufficient"]:
        status, reason = UNTESTABLE, "信度上限不足或未能估計"
    elif any(p["pooled_rho"] is not None and p["pooled_rho"] >= ceiling_value - rules["fail_gap_max"] for p in covered):
        status, reason = FAIL, "覆蓋足夠的既有預測量已接近信度上限"
    elif any(p["pooled_rho"] is None for p in covered):
        status, reason = UNTESTABLE, "覆蓋足夠的既有預測量之相關未能估計"
    elif not covered or (gap is not None and gap >= rules["pass_gap_min"]):
        status, reason = PASS, "既有量覆蓋不足或與信度上限有差距"
    else:
        status, reason = GRAY, "信度上限與既有預測量的差距落在灰區"

    comparisons = {}
    for name in ("grant_cosine", strongest):
        if name not in eligible or name in comparisons:
            continue
        common = matched[name]
        lam_rho, _ = guarded_rho(common)
        eligible_rho = eligible[name]
        if lam_rho is None:
            continue
        bs_lam, bs_existing = bootstrap_correlations(common, ["lam", "existing"], c)
        ci = interval(bs_lam - bs_existing, c["baseline_delta_ci_level"])
        comparisons[name] = dict(delta_rho=lam_rho - eligible_rho, ci=ci, ci_level=c["baseline_delta_ci_level"])
    return dict(status=status, reason=reason, rho_exist=highest, best_predictor=strongest, C_minus_rho_exist=gap,
                predictors=per_predictor, descriptive_matched_deltas=comparisons,
                note="覆蓋例外使空間近乎由構造保證；相對既有量的差值只描述")


# ---------------------------------------------------------------- L1
def ceiling(leak_rows, lamseed_rows, x):
    """每任務 C_task = sqrt(max(r_L,0)·max(r_Λ,0))；三任務 Fisher z 等權合併成 C。只用有防禦 arm。"""
    ls, ks = x["ceiling_leak_seeds"], x["ceiling_lambda_seeds"]
    tasks = sorted({r["task"] for r in leak_rows if r["arm"] != "none"})
    per, c_task = {}, []
    for t in tasks:
        L = {}
        for r in leak_rows:
            if r["task"] == t and r["arm"] != "none":
                L.setdefault(r["arm"], {})[int(r["seed"])] = float(r["leak"])
        arms = sorted(a for a, d in L.items() if all(s in d for s in ls))
        r_l1 = rho([L[a][ls[0]] for a in arms], [L[a][ls[1]] for a in arms])
        r_l = sb(r_l1, len(ls))
        M = {}
        for r in lamseed_rows:
            if r["task"] == t and r["arm"] in L:
                seed = int(r["lora_seed"])
                values = M.setdefault(r["arm"], {})
                if seed in values:
                    raise ValueError(f"lambda_seeds.csv 重複 LoRA seed：{t}/{r['arm']}/{seed}")
                values[seed] = float(r["lam"])
        arms_m = sorted(a for a, d in M.items() if all(s in d for s in ks))
        pair = [rho([M[a][i] for a in arms_m], [M[a][j] for a in arms_m]) for i, j in itertools.combinations(ks, 2)]
        r_lam1 = float(np.mean(pair))
        r_lam = sb(r_lam1, len(ks))
        if len(arms) != len(L) or len(arms_m) != len(L) or not (math.isfinite(r_l) and math.isfinite(r_lam)):
            ct = float("nan")
        else:
            ct = math.sqrt(max(r_l, 0.0) * max(r_lam, 0.0))
        per[t] = dict(n_arms_leak=len(arms), n_arms_lambda=len(arms_m), r_L_single=r_l1, r_L=r_l,
                      r_Lambda_single=r_lam1, r_Lambda=r_lam, C_task=ct)
        c_task.append(ct)
    C = zpool(c_task) if c_task and not any(math.isnan(v) for v in c_task) else float("nan")
    return dict(C=C, per_task=per, ceiling_min=x["ceiling_min"],
                sufficient=bool((not math.isnan(C)) and C >= x["ceiling_min"]))


# ---------------------------------------------------------------- L2
def horizon(hz_rows, leak_rows, x):
    """S5 = 各任務 Spearman(Λ_0, Δτ_5%) 的 Fisher z 合併（權重 n_arm−3，同主判準）。另輸出只描述的分解。"""
    by = {}
    for r in hz_rows:
        if r["arm"] != "none":
            key = r["task"], r["arm"]
            if key in by:
                raise ValueError(f"horizon.csv 重複 arm：{key}")
            by[key] = r
    leak = AG.table_by_task(leak_rows)
    s5, ns, d_final, d_leak = [], [], [], []
    per = {}
    for t, arms in sorted(leak.items()):
        rs = [by[(t, arm)] for arm in sorted(arms) if (t, arm) in by]
        lam = [float(arms[r["arm"]]["lam"]) for r in rs]
        d5, dfinal = column(rs, "dtau5"), column(rs, "dtau_final")
        r5 = rho(lam, d5) if len(rs) == len(arms) else float("nan")
        rf = rho(lam, dfinal)
        lk = [leak.get(t, {}).get(r["arm"], {}).get("leak") for r in rs]
        rl = rho([delta for delta, v in zip(dfinal, lk) if v is not None], [v for v in lk if v is not None])
        per[t] = dict(n=len(rs), rho_lam_dtau5=r5, rho_lam_dtau_final=rf, rho_dtau_final_leak=rl)
        s5.append(r5); ns.append(len(rs)); d_final.append(rf); d_leak.append(rl)
    w = [max(n - 3, 1) for n in ns]
    S5 = zpool(s5, w) if s5 and all(math.isfinite(v) for v in s5) else float("nan")
    ok = S5 >= x["horizon_s5_min"] if math.isfinite(S5) else None
    complete = ok is not None and all(math.isfinite(r) for r in (*d_final, *d_leak))
    return dict(S5=S5, short_horizon_holds=ok, complete_for_probe=bool(complete), threshold=x["horizon_s5_min"], per_task=per,
                descriptive_pooled=dict(rho_lam_dtau_final=zpool(d_final, w), rho_dtau_final_leak=zpool(d_leak, w)),
                note="S5 不擋主判準；descriptive_pooled 只描述")


def column(rows, key):
    return [float(r[key]) if r.get(key) not in (None, "") else float("nan") for r in rows]


# ---------------------------------------------------------------- L3
def probe(leak_rows, hz_rows, version, x, crit):
    """唯一的第二探針：用主判準同一組 PASS 條件判一次；灰區記未通過，不加碼。"""
    hz = {(r["task"], r["arm"]): r for r in hz_rows}
    rows, excluded = [], []
    for r in leak_rows:
        if r["arm"] == "none":
            continue
        h = hz.get((r["task"], r["arm"]))
        if h is None:
            return dict(status="UNTESTABLE", reason="horizon.csv 缺 arm %s/%s" % (r["task"], r["arm"]))
        r2 = dict(r)
        required = ("lam_id",) if version == "ratio" else ("lam_0", "lam_1p25", "lam_2p5", "lam_5")
        values = [float(h[k]) for k in required if h.get(k) not in (None, "")]
        if len(values) != len(required) or not all(math.isfinite(v) for v in values):
            return dict(status=UNTESTABLE, version=version, reason=f"horizon.csv 缺可用探針資料：{r['task']}/{r['arm']}")
        if version == "ratio":
            lid = values[0]
            if lid <= 0:
                excluded.append((r["task"], r["arm"]))
                continue
            r2["lam"] = float(r["lam"]) / lid
        else:
            r2["lam"] = float(np.mean(values))
        rows.append(r2)
    excl = sorted(set(excluded))
    if len(excl) > x["probe_max_excluded_arms"]:
        return dict(status=UNTESTABLE, version=version, excluded=excl,
                    reason="Λ_ID <= 0 的 arm 超過 %d 個" % x["probe_max_excluded_arms"])
    arms = AG.table_by_task(rows)
    res = signal_stats(arms, crit)
    st = (PASS if signal_pass(res, x["signal"]) else "NOT_PASS") if res["testable"] else UNTESTABLE
    return dict(status=st, version=version, excluded=excl, **res, gpu_hour_cap=x["probe_gpu_hour_cap"],
                note="探針灰區記未通過，不加碼；缺資料或無法估計時記未能測試")


# ---------------------------------------------------------------- 整合
def index_leak_rows(rows):
    arms = {}
    for r in rows:
        if r["arm"] == "none":
            continue
        task, arm, seed = r["task"], r["arm"], int(r["seed"])
        if not task or not arm or not r["family"]:
            raise ValueError("leakage.csv 缺 task、arm 或 family")
        if not 0 <= float(r["leak"]) <= 1 or not math.isfinite(float(r["lam"])):
            raise ValueError(f"leakage.csv 非比例或 Λ 非有限：{task}/{arm}/{seed}")
        entry = arms.setdefault(task, {}).setdefault(arm, (r["family"], set()))
        if entry[0] != r["family"] or seed in entry[1]:
            raise ValueError(f"leakage.csv 重複 seed 或 family 不一致：{task}/{arm}/{seed}")
        entry[1].add(seed)
    return arms


def check_design(rows, crit, initial):
    x, c = crit["phase_b_extras"], crit["primary"]
    arms = index_leak_rows(rows)
    seeds = set(x["ceiling_leak_seeds"])
    if initial is None:
        if len(arms) != x["expected_tasks"] or any(len(a) != c["n_defended_arms"] for a in arms.values()):
            return None
        if not all(s == seeds for a in arms.values() for _, s in a.values()):
            return None
        return 0, arms

    if (initial.get("criteria_version") != x["decision_version"] or initial.get("overall") != GRAY
            or initial.get("signal", {}).get("status") != GRAY
            or initial.get("gray_addon_used") is not False or initial.get("gray_addon") != c["gray_extra"]):
        raise ValueError("加碼須引用同版判準的初判灰區結果")
    prior = initial.get("original_design", {})
    if set(prior) != set(arms) or any(len(a) != c["n_defended_arms"] for a in prior.values()):
        raise ValueError("初判與加碼的原定任務、arm 不一致")
    for task, original in prior.items():
        if not all(arm in arms[task] and arms[task][arm][0] == family for arm, family in original.items()):
            raise ValueError(f"加碼缺原定 arm 或防禦家族已變更：{task}")
    if sum(len(arms[task]) - len(original) for task, original in prior.items()) != c["gray_extra"]["arms"]:
        raise ValueError("加碼須新增原定數量的任務／arm 配對")
    observed_seed_sets = {frozenset(s) for a in arms.values() for _, s in a.values()}
    if len(observed_seed_sets) != 1:
        raise ValueError("加碼所有 arm 須使用相同的 seed 集合")
    after, = observed_seed_sets
    if not seeds <= after or len(after - seeds) != c["gray_extra"]["seeds"]:
        raise ValueError("加碼須保留原 seed 並加入原定數量的新 seed")
    return 1, {task: {arm: arms[task][arm] for arm in original} for task, original in prior.items()}


def run(leak_rows, lamseed_rows, hz_rows, gates, comparator_rows, crit, initial=None):
    x = crit["phase_b_extras"]
    rules = x["signal"]
    out = dict(criteria_version=x["decision_version"], scope="Phase B 三任務與凍結的防禦 arm")
    design = check_design(leak_rows, crit, initial)
    if design is None:
        reason = "需要 3 個任務各 13 個有防禦 arm，每 arm 訓練 seed 為 0 與 1"
        return dict(**out, overall=UNTESTABLE, reason=reason)
    addon_used, original = design
    out["gray_addon_used"] = bool(addon_used)
    out["original_design"] = {task: {arm: entry[0] for arm, entry in a.items()} for task, a in original.items()}
    leak_rows, lam_issues = canonical_lambda_rows(leak_rows, lamseed_rows, x)
    if leak_rows is None:
        reason = "缺完整 LoRA seed，或輸入 Λ 與凍結的三 seed 平均不符"
        return dict(**out, overall=UNTESTABLE, reason=reason, lambda_issues=lam_issues)
    original_keys = {(task, arm) for task, a in original.items() for arm in a}
    original_leak_rows = [r for r in leak_rows if (r["task"], r["arm"]) in original_keys]
    original_arms = AG.table_by_task(original_leak_rows)
    arms = AG.table_by_task(leak_rows)
    out["L1_ceiling"] = c = ceiling(original_leak_rows, lamseed_rows, x)
    out["L2_horizon"] = h = horizon(hz_rows, leak_rows, x)
    out["phenomenon"] = p = phenomenon(original_arms, gates, c, x["phenomenon"])
    out["headroom"] = headroom(original_arms, comparator_rows, c, x["headroom"], crit["primary"])
    raw = signal_stats(arms, crit)
    l3 = dict(status="NOT_TRIGGERED", reason="原始 Λ 未通過且前置閘門、信度上限達標時，才計算 L3")
    if p["status"] == UNTESTABLE:
        raw_status = status = UNTESTABLE
    elif signal_pass(raw, rules):
        raw_status = status = PASS
    else:
        raw_status = "NOT_PASS" if raw["testable"] else UNTESTABLE
        if addon_used:
            l3 = initial["L3_probe"]
        elif not h["complete_for_probe"]:
            l3 = dict(status=UNTESTABLE, reason="缺完整且可計算的 L2 地平線統計")
        else:
            version = "ratio" if h["short_horizon_holds"] else "kstep"
            l3 = probe(leak_rows, hz_rows, version, x, crit)
        if l3["status"] == PASS:
            status = PASS
        elif not raw["testable"]:
            status = UNTESTABLE
        elif (raw["pooled_rho"] < rules["rho_fail_below"] and raw["ci"][1] < rules["ci_upper_fail_below"]):
            status = FAIL if l3["status"] == "NOT_PASS" else UNTESTABLE
        else:
            status = GRAY
    out["signal"] = dict(status=status, raw_status=raw_status, raw=raw)
    out["L3_probe"] = l3
    statuses = {name: out[name]["status"] for name in ("phenomenon", "signal", "headroom")}
    if p["status"] == UNTESTABLE:
        out["overall"] = UNTESTABLE
    elif FAIL in statuses.values():
        out["overall"] = FAIL
    elif all(st == PASS for st in statuses.values()):
        out["overall"] = PASS
    elif UNTESTABLE in statuses.values():
        out["overall"] = UNTESTABLE
    else:
        out["overall"] = GRAY
    out["failed_items"] = [name for name, st in statuses.items() if st == FAIL]
    out["gray_addon"] = crit["primary"]["gray_extra"] if out["overall"] == GRAY and status == GRAY and not addon_used else None
    return out


def readcsv(p):
    with open(p, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


# ---------------------------------------------------------------- selftest（合成資料）
def synth():
    leak_rows, ls_rows, hz_rows, cmp_rows = [], [], [], []
    for ti in range(3):
        t = f"T{ti}"
        for ai in range(13):
            y = ai / 12
            a = f"a{ai}"
            for s in (0, 1):
                leak_rows.append(dict(task=t, arm=a, family=f"f{ai % 3}", seed=s, leak=y, lam=y, id_ok="True"))
            for s in (0, 1, 2):
                ls_rows.append(dict(task=t, arm=a, lora_seed=s, lam=y))
            hz_rows.append(dict(task=t, arm=a, lam_0=y, lam_1p25=y, lam_2p5=y, lam_5=y,
                                lam_id=1.0, dtau5=y, dtau_final=y))
            cmp_rows.append(dict(task=t, arm=a, grant_cosine="", wichers_elicitation="", tan_loss_drop="",
                                 phm_persona_distance=""))
        leak_rows.append(dict(task=t, arm="none", family="none", seed=0, leak=1.0, lam=1.0, id_ok="True"))
    return leak_rows, ls_rows, hz_rows, cmp_rows


def selftest():
    crit = AG.load_criteria()
    crit["primary"]["bootstrap_n"] = 250
    x = crit["phase_b_extras"]
    good = {f"gate{i}": PASS for i in range(1, 4)}
    L, S, H, C = synth()

    def seed_rows(leak):
        return [dict(task=r["task"], arm=r["arm"], lora_seed=seed, lam=r["lam"])
                for r in leak if r["arm"] != "none" and r["seed"] == 0 for seed in (0, 1, 2)]

    def assess(leak=L, lamseeds=None, hz=H, comparators=C, gates=good):
        lamseeds = seed_rows(leak) if lamseeds is None else lamseeds
        return run(leak, lamseeds, hz, gates, comparators, crit)

    assert math.isclose(sb(0.5, 2), 2 / 3) and math.isclose(sb(0.5, 3), 0.75) and sb(-0.75, 3) == 0
    o = assess()
    assert o["overall"] == PASS and o["L3_probe"]["status"] == "NOT_TRIGGERED", o
    assert o["signal"]["raw"]["strong_signal_descriptive"] and o["signal"]["raw"]["ci_level"] == 0.975
    o = assess(hz=[])
    assert o["overall"] == PASS and o["L2_horizon"]["short_horizon_holds"] is None
    assert assess(leak=L[2:])["overall"] == UNTESTABLE
    assert assess(gates=dict(good, gate2=FAIL))["overall"] == UNTESTABLE
    assert assess(lamseeds=S[1:])["overall"] == UNTESTABLE
    low = [dict(r, lam=3 * r["lam"] if r["lora_seed"] != 2 else -3 * r["lam"]) for r in S]
    assert assess(lamseeds=low)["phenomenon"]["status"] == UNTESTABLE
    assert assess(lamseeds=[dict(r, lam=-r["lam"]) for r in S])["overall"] == UNTESTABLE
    assert assess(lamseeds=[dict(r, lam=r["lam"] + 1e-10) for r in S])["overall"] == PASS
    narrow = [dict(r, leak=r["leak"] * 0.10) if r["task"] != "T0" else r for r in L]
    o = assess(leak=narrow)
    assert o["phenomenon"]["status"] == FAIL and o["phenomenon"]["tasks_with_range"] == 1
    assert assess(comparators=C[:-1])["headroom"]["status"] == UNTESTABLE
    strong_baseline = [dict(r, grant_cosine=str(int(r["arm"][1:]))) for r in C]
    o = assess(comparators=strong_baseline)
    assert o["overall"] == FAIL and o["failed_items"] == ["headroom"], o
    assert set(o["headroom"]["descriptive_matched_deltas"]) == {"grant_cosine"}
    support9 = [dict(r, grant_cosine="") if int(r["arm"][1:]) >= 9 else r for r in strong_baseline]
    assert assess(comparators=support9)["headroom"]["status"] == PASS
    constant_baseline = [dict(r, grant_cosine="1") for r in C]
    assert assess(comparators=constant_baseline)["headroom"]["status"] == UNTESTABLE
    reversed_lam = [dict(r, lam=-r["lam"]) for r in L]
    reversed_hz = [dict(r, lam_0=-r["lam_0"], dtau5=-r["dtau5"], dtau_final=-r["dtau_final"]) for r in H]
    o = assess(leak=reversed_lam, hz=reversed_hz)
    assert o["L3_probe"]["version"] == "ratio" and o["signal"]["status"] == FAIL, o
    o = assess(leak=reversed_lam, hz=[])
    assert o["overall"] == UNTESTABLE and o["L3_probe"]["status"] == UNTESTABLE
    o = assess(leak=reversed_lam, hz=[dict(r, dtau_final="") if i == 0 else r for i, r in enumerate(reversed_hz)])
    assert o["L2_horizon"]["short_horizon_holds"] and not o["L2_horizon"]["complete_for_probe"]
    assert o["L3_probe"]["status"] == UNTESTABLE
    for n, expected in ((3, "NOT_PASS"), (4, UNTESTABLE)):
        hz = [dict(r, lam_id=0.0) if i < n else r for i, r in enumerate(reversed_hz)]
        o = assess(leak=reversed_lam, hz=hz)
        assert o["L3_probe"]["status"] == expected and len(o["L3_probe"]["excluded"]) == n, o
    hz = [dict(r, lam_0=-r["lam_0"]) for r in H]
    o = assess(leak=reversed_lam, hz=hz)
    assert o["L3_probe"]["version"] == "kstep" and o["L3_probe"]["status"] == PASS and o["overall"] == PASS, o
    collapse_family = [dict(r, lam=r["lam"] if r["family"] == "f0" else 0.0) for r in L]
    hz = [dict(r, dtau5=-r["dtau5"]) for r in H]
    o = assess(leak=collapse_family, hz=hz)
    assert o["signal"]["raw"]["drop_family"]["f0"] is None and o["signal"]["raw_status"] == UNTESTABLE
    assert o["L3_probe"]["version"] == "kstep" and o["L3_probe"]["status"] == PASS and o["overall"] == PASS, o
    positive_leak = [dict(r, leak=(1 + r["leak"]) / 2) for r in L]
    ratio_lam = [dict(r, lam=1 - r["leak"] / 2) for r in positive_leak]
    hz = [dict(r, dtau5=-r["dtau5"],
               lam_id=(1 - (1 + r["lam_0"]) / 4) / ((1 + r["lam_0"]) / 2)) for r in H]
    o = assess(leak=ratio_lam, hz=hz)
    assert o["L3_probe"]["version"] == "ratio" and o["L3_probe"]["status"] == PASS and o["overall"] == PASS, o
    boundary = dict(testable=True, pooled_rho=0.35, ci=[0.01, 0.8], tasks_positive=2, drop_family={"pps": 0.01})
    assert signal_pass(boundary, x["signal"])
    assert not signal_pass(dict(boundary, ci=[0.0, 0.8]), x["signal"])
    assert not signal_pass(dict(boundary, drop_family={"pps": 0.0}), x["signal"])
    almost_reversed = [12, 10, 11, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    almost_positive = [1, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    mixed = [dict(r, lam=(almost_reversed if r["task"] == "T0" else almost_positive)[int(r["arm"][1:])])
             if r["arm"] != "none" else r for r in L]
    lam_by_arm = {(r["task"], r["arm"]): r["lam"] for r in mixed}
    mixed_hz = [dict(r, lam_0=lam_by_arm[r["task"], r["arm"]], dtau5=lam_by_arm[r["task"], r["arm"]]) for r in H]
    initial = assess(leak=mixed, hz=mixed_hz)
    assert initial["overall"] == GRAY and initial["gray_addon"] and initial["L3_probe"]["status"] == "NOT_PASS", initial
    more = mixed + [dict(r, seed=2) for r in mixed if r["arm"] != "none" and r["seed"] == 0]
    more.extend(dict(task=f"T{t}", arm=f"x{i}", family="f0", seed=s, leak=i / 2, lam=i / 2, id_ok="True")
                for t in range(3) for i in range(2) for s in range(3))
    rerun = run(more, seed_rows(more), mixed_hz, good, C, crit, initial)
    assert rerun["gray_addon_used"] and rerun["gray_addon"] is None and rerun["headroom"]["status"] == PASS, rerun
    assert rerun["signal"]["raw"]["n_arms"] == {"T0": 15, "T1": 15, "T2": 15}
    for bad, prior in ((more[:-1], initial), (more, rerun)):
        try:
            run(bad, seed_rows(bad), mixed_hz, good, C, crit, prior)
        except ValueError:
            pass
        else:
            raise AssertionError("未拒絕不完整或重複灰區加碼")
    print("phaseb_extras selftest: OK (現象、訊號、空間及 L1–L3)")


def safe_json(obj):
    if isinstance(obj, dict):
        return {k: safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (tuple, list)):
        return [safe_json(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--leak"); ap.add_argument("--lamseeds"); ap.add_argument("--horizon")
    ap.add_argument("--gates"); ap.add_argument("--comparators"); ap.add_argument("--outdir")
    ap.add_argument("--addon-from")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    for required in ("leak", "lamseeds", "gates", "outdir"):
        if not getattr(a, required):
            ap.error(f"--{required} 為必要參數")
    with open(a.gates, encoding="utf-8") as handle:
        gates = json.load(handle)
    initial = None
    if a.addon_from:
        if os.path.abspath(a.addon_from) == os.path.abspath(os.path.join(a.outdir, "phaseb_extras.json")):
            ap.error("加碼輸出須使用獨立目錄，保留初判檔")
        with open(a.addon_from, encoding="utf-8") as handle:
            initial = json.load(handle)
    o = run(readcsv(a.leak), readcsv(a.lamseeds), readcsv(a.horizon) if a.horizon else [], gates,
            readcsv(a.comparators) if a.comparators else [], AG.load_criteria(), initial)
    payload = json.dumps(safe_json(o), indent=2, ensure_ascii=False, allow_nan=False)
    os.makedirs(a.outdir, exist_ok=True)
    with open(os.path.join(a.outdir, "phaseb_extras.json"), "w", encoding="utf-8") as handle:
        handle.write(payload + "\n")
    print(payload)
    print(f"RESULT_PHASE_B={o['overall']}")


if __name__ == "__main__":
    main()
