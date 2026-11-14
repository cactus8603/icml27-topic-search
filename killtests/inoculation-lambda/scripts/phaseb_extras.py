#!/usr/bin/env python
"""Phase B 的 L1–L3（2026-11-14 代理落差稽核後、結果前定案；見 PREREG.md §3a）。

不改 aggregate.py 的主判準邏輯：本腳本先讀（或呼叫）aggregate.phase_b 的原判定，再依 L1 決定
FAIL 是否改記「未能測試（信度上限不足）」，並做 L2 地平線歸因與 L3 唯一的第二探針。

輸入（皆 CSV；Phase B 訓練與 Λ 批次腳本須產出這些欄位）：
  --leak     leakage.csv     task,arm,family,seed,leak,lam,base_pred,id_ok  （同 aggregate --phase b；
                              seed = 訓練 seed，每 arm 至少含 seed 0 與 1）
  --lamseeds lambda_seeds.csv task,arm,lora_seed,lam                        （LoRA 初始化 seed 0,1,2 各一列）
  --horizon  horizon.csv     task,arm,lam_0,lam_1p25,lam_2p5,lam_5,lam_id,dtau5,dtau_final
                              lam_k = 在該 arm seed 0 的 k% 步 checkpoint 上算的 Λ；lam_id = ID 資料的 Λ；
                              dtau5 = 5% 步 τ − step 0 τ；dtau_final = 最終 τ − step 0 τ
  --primary  aggregate_phaseB.json（aggregate.py --phase b 已產生）
用法：phaseb_extras.py --leak ... --lamseeds ... --horizon ... --primary ... --outdir ...
      phaseb_extras.py --selftest
"""
import argparse
import csv
import itertools
import json
import math
import os
import sys
import tempfile

import numpy as np
import yaml
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import aggregate as AG  # noqa: E402  只重用 phase_b／table_by_task／fisher_pool，不改其邏輯


def sb(r, k):
    """Spearman-Brown：單一重複的信度 r 校正到 k 個重複。"""
    if r is None or math.isnan(r):
        return float("nan")
    return k * r / (1 + (k - 1) * r)


def rho(a, b):
    if len(a) < 3:
        return float("nan")
    return float(stats.spearmanr(a, b)[0])


def zpool(vals, weights=None):
    vals = [v for v in vals if v is not None and not math.isnan(v)]
    if not vals:
        return float("nan")
    z = [np.arctanh(np.clip(v, -0.999999, 0.999999)) for v in vals]
    return float(np.tanh(np.average(z, weights=weights)))


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
                M.setdefault(r["arm"], {})[int(r["lora_seed"])] = float(r["lam"])
        arms_m = sorted(a for a, d in M.items() if all(s in d for s in ks))
        pair = [rho([M[a][i] for a in arms_m], [M[a][j] for a in arms_m]) for i, j in itertools.combinations(ks, 2)]
        pair = [p for p in pair if not math.isnan(p)]
        r_lam1 = float(np.mean(pair)) if pair else float("nan")
        r_lam = sb(r_lam1, len(ks))
        if math.isnan(r_l) or math.isnan(r_lam):
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
        by.setdefault(r["task"], []).append(r)
    leak = AG.table_by_task(leak_rows)
    s5, ns, d_final, d_leak = [], [], [], []
    per = {}
    for t, rs in sorted(by.items()):
        rs = [r for r in rs if r["arm"] != "none"]
        lam = [float(r["lam_0"]) for r in rs]
        r5 = rho(lam, [float(r["dtau5"]) for r in rs])
        rf = rho(lam, [float(r["dtau_final"]) for r in rs])
        lk = [leak.get(t, {}).get(r["arm"], {}).get("leak") for r in rs]
        rl = rho([float(r["dtau_final"]) for r, v in zip(rs, lk) if v is not None], [v for v in lk if v is not None])
        per[t] = dict(n=len(rs), rho_lam_dtau5=r5, rho_lam_dtau_final=rf, rho_dtau_final_leak=rl)
        s5.append(r5); ns.append(len(rs)); d_final.append(rf); d_leak.append(rl)
    w = [max(n - 3, 1) for n in ns]
    S5 = zpool(s5, w)
    ok = bool((not math.isnan(S5)) and S5 >= x["horizon_s5_min"])
    return dict(S5=S5, short_horizon_holds=ok, threshold=x["horizon_s5_min"], per_task=per,
                descriptive_pooled=dict(rho_lam_dtau_final=zpool(d_final, w), rho_dtau_final_leak=zpool(d_leak, w)),
                note="S5 不擋主判準；descriptive_pooled 只描述")


# ---------------------------------------------------------------- L3
def probe(leak_rows, hz_rows, version, x, crit, outdir):
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
        if version == "ratio":
            lid = float(h["lam_id"])
            if lid <= 0:
                excluded.append((r["task"], r["arm"]))
                continue
            r2["lam"] = float(r["lam"]) / lid
        else:
            r2["lam"] = float(np.mean([float(h[k]) for k in ("lam_0", "lam_1p25", "lam_2p5", "lam_5")]))
        r2["base_pred"] = ""      # 探針不判基線
        rows.append(r2)
    excl = sorted(set(excluded))
    if len(excl) > x["probe_max_excluded_arms"]:
        return dict(status="UNTESTABLE", version=version, excluded=excl,
                    reason="Λ_ID <= 0 的 arm 超過 %d 個" % x["probe_max_excluded_arms"])
    pdir = os.path.join(outdir, "probe_l3")
    os.makedirs(pdir, exist_ok=True)
    tpath = os.path.join(pdir, "leakage_probe.csv")
    with open(tpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    ns = argparse.Namespace(table=tpath)
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):      # 不重印整份 JSON
        AG.phase_b(ns, crit)
    res = json.load(open(os.path.join(pdir, "aggregate_phaseB.json")))
    st = res["status"]
    return dict(status="PASS" if st == "PASS" else "NOT_PASS", underlying=st, version=version, excluded=excl,
                pooled_rho=res["pooled_rho"], ci=res["ci"], per_task_rho=res["per_task_rho"],
                drop_family=res["drop_family"], gpu_hour_cap=x["probe_gpu_hour_cap"],
                note="灰區也記未通過，不加碼；探針也未通過才可提請淘汰")


# ---------------------------------------------------------------- 整合
def run(leak_rows, lamseed_rows, hz_rows, primary_status, crit, outdir):
    x = crit["phase_b_extras"]
    out = dict(criteria_version=crit["version"], primary_status_original=primary_status)
    out["L1_ceiling"] = c = ceiling(leak_rows, lamseed_rows, x)
    out["L2_horizon"] = h = horizon(hz_rows, leak_rows, x)
    final = primary_status
    if primary_status == "FAIL" and not c["sufficient"]:
        final = "未能測試（信度上限不足）"
    out["primary_status_final"] = final
    if primary_status == "FAIL" and c["sufficient"]:
        version = "ratio" if h["short_horizon_holds"] else "kstep"
        out["L3_probe"] = probe(leak_rows, hz_rows, version, x, crit, outdir)
    else:
        out["L3_probe"] = dict(status="NOT_TRIGGERED",
                               reason="觸發條件：主判準 FAIL 且 C >= %.2f" % x["ceiling_min"])
    os.makedirs(outdir, exist_ok=True)
    json.dump(out, open(os.path.join(outdir, "phaseb_extras.json"), "w"), indent=1, default=str)
    return out


def readcsv(p):
    return list(csv.DictReader(open(p)))


# ---------------------------------------------------------------- selftest（合成資料）
def synth(rng, n_tasks=3, n_arms=13, leak_noise=0.1, lam_noise=0.1, rel_lam=True, ratio_signal=False):
    """合成：leak 與 lam 共享潛變數 z。lam 可加噪；回傳三張表。"""
    leak_rows, ls_rows, hz_rows = [], [], []
    for ti in range(n_tasks):
        t = f"T{ti}"
        for ai in range(n_arms):
            z = rng.normal()
            a = f"a{ai}"
            lam_id = float(rng.uniform(0.5, 2.0))
            lam_true = z * lam_id if ratio_signal else z
            for s in (0, 1):
                leak_rows.append(dict(task=t, arm=a, family=f"f{ai % 3}", seed=s, leak=z + leak_noise * rng.normal(),
                                      lam=lam_true, base_pred="", id_ok="True"))
            for s in (0, 1, 2):
                ls_rows.append(dict(task=t, arm=a, lora_seed=s, lam=lam_true + lam_noise * rng.normal()))
            hz_rows.append(dict(task=t, arm=a, lam_0=lam_true, lam_1p25=lam_true + 0.05 * rng.normal(),
                                lam_2p5=lam_true + 0.05 * rng.normal(), lam_5=lam_true + 0.05 * rng.normal(),
                                lam_id=lam_id, dtau5=lam_true + 0.05 * rng.normal(), dtau_final=lam_true + 0.3 * rng.normal()))
        leak_rows.append(dict(task=t, arm="none", family="none", seed=0, leak=5.0, lam=1.0, base_pred="", id_ok="True"))
    return leak_rows, ls_rows, hz_rows


def selftest():
    crit = yaml.safe_load(open(os.path.join(ROOT, "configs", "criteria.yaml")))
    crit["primary"]["bootstrap_n"] = 200
    x = crit["phase_b_extras"]
    # 1. Spearman-Brown
    assert abs(sb(0.5, 2) - 2 / 3) < 1e-12 and abs(sb(0.5, 3) - 0.75) < 1e-12 and sb(0.0, 3) == 0.0
    tmp = tempfile.mkdtemp(dir=os.environ.get("TMPDIR"))
    rng = np.random.default_rng(1)
    # 2. 高信度、Λ 與漏出強相關 → C ≥ 0.75，主判準 PASS，L3 不觸發
    L, S, H = synth(rng)
    c = ceiling(L, S, x)
    assert c["sufficient"] and c["C"] > 0.9, c["C"]
    o = run(L, S, H, "PASS", crit, os.path.join(tmp, "s_pass"))
    assert o["primary_status_final"] == "PASS" and o["L3_probe"]["status"] == "NOT_TRIGGERED"
    # 3. 信度不足：Λ 對 LoRA seed 幾乎是純雜訊、漏出 seed 間也無關 → C < 0.75
    L, S, H = synth(rng, leak_noise=3.0, lam_noise=3.0)
    c = ceiling(L, S, x)
    assert not c["sufficient"], c["C"]
    o = run(L, S, H, "FAIL", crit, os.path.join(tmp, "s_lowc"))
    assert o["primary_status_final"] == "未能測試（信度上限不足）" and o["L3_probe"]["status"] == "NOT_TRIGGERED"
    o = run(L, S, H, "PASS", crit, os.path.join(tmp, "s_lowc_pass"))      # PASS／灰區照原規則
    assert o["primary_status_final"] == "PASS"
    o = run(L, S, H, "GRAY", crit, os.path.join(tmp, "s_lowc_gray"))
    assert o["primary_status_final"] == "GRAY"
    # 4. 信度足夠但主判準 FAIL（以原判定為 FAIL 輸入）：短地平線成立 → 比值版，且比值版通過
    L, S, H = synth(rng, ratio_signal=True)      # leak ~ z；原 lam = z*lam_id；比值 lam/lam_id = z → 通過
    c = ceiling(L, S, x)
    assert c["sufficient"]
    h = horizon(H, L, x)
    assert h["short_horizon_holds"], h["S5"]
    o = run(L, S, H, "FAIL", crit, os.path.join(tmp, "s_ratio"))
    assert o["primary_status_final"] == "FAIL" and o["L3_probe"]["version"] == "ratio", o["L3_probe"]
    assert o["L3_probe"]["status"] == "PASS", o["L3_probe"]
    # 5. 比值版：Λ_ID ≤ 0 的 arm 排除；排除 > 3 個 → 未能測試
    H2 = [dict(r) for r in H]
    for r in H2[:3]:
        r["lam_id"] = -1.0
    o = run(L, S, H2, "FAIL", crit, os.path.join(tmp, "s_ex3"))
    assert o["L3_probe"]["status"] in ("PASS", "NOT_PASS") and len(o["L3_probe"]["excluded"]) == 3
    H3 = [dict(r) for r in H]
    for r in H3[:4]:
        r["lam_id"] = 0.0
    o = run(L, S, H3, "FAIL", crit, os.path.join(tmp, "s_ex4"))
    assert o["L3_probe"]["status"] == "UNTESTABLE" and len(o["L3_probe"]["excluded"]) == 4
    # 6. 短地平線不成立 → k 步版（四時點等權平均）；構造使 k 步版 Λ 通過
    H4 = []
    for r in H:
        r = dict(r)
        r["dtau5"] = rng.normal()                         # Δτ5 與 Λ 無關 → S5 低
        H4.append(r)
    h = horizon(H4, L, x)
    assert not h["short_horizon_holds"]
    o = run(L, S, H4, "FAIL", crit, os.path.join(tmp, "s_kstep"))
    assert o["L3_probe"]["version"] == "kstep"
    # k 步版的數值：確認 lam 欄確實被換成四時點平均
    pt = readcsv(os.path.join(tmp, "s_kstep", "probe_l3", "leakage_probe.csv"))
    hzd = {(r["task"], r["arm"]): r for r in H4}
    r0 = pt[0]; hh = hzd[(r0["task"], r0["arm"])]
    assert abs(float(r0["lam"]) - np.mean([hh[k] for k in ("lam_0", "lam_1p25", "lam_2p5", "lam_5")])) < 1e-9
    # 7. k 步版未通過：Λ 在各時點都是雜訊
    H5 = []
    for r in H4:
        r = dict(r)
        for k in ("lam_0", "lam_1p25", "lam_2p5", "lam_5"):
            r[k] = rng.normal()
        H5.append(r)
    L5 = [dict(r, lam=rng.normal()) for r in L]      # 主判準也 FAIL 的情境
    o = run(L5, S, H5, "FAIL", crit, os.path.join(tmp, "s_kfail"))
    if o["L1_ceiling"]["sufficient"]:
        assert o["L3_probe"]["status"] == "NOT_PASS", o["L3_probe"]
    print("phaseb_extras selftest: OK (L1/L2/L3 全部分支)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--leak"); ap.add_argument("--lamseeds"); ap.add_argument("--horizon")
    ap.add_argument("--primary"); ap.add_argument("--outdir")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    crit = yaml.safe_load(open(os.path.join(ROOT, "configs", "criteria.yaml")))
    st = json.load(open(a.primary))["status"]
    o = run(readcsv(a.leak), readcsv(a.lamseeds), readcsv(a.horizon), st, crit, a.outdir)
    print(json.dumps(o, indent=1, default=str))
    print("PRIMARY_FINAL=%s  L3=%s" % (o["primary_status_final"], o["L3_probe"]["status"]))


if __name__ == "__main__":
    main()
