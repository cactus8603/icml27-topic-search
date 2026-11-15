#!/usr/bin/env python
"""機械判讀（不含人工判斷）。所有門檻從 configs/criteria.yaml 讀。

phase a（本次交付，零訓練）：
    閘門 1（有限差分）→ PASS/FAIL；
    對照兩篇論文已報告數字的 Spearman（僅「參考」，不停損、不入判準）。
phase b（訓練階段，尚未產生資料；程式已備，供審查）：
    主判準：有防禦 13 個 arm、各任務內 Spearman → Fisher z 合併 → 依任務分層 bootstrap，
    輸出 PASS / GRAY / FAIL 與「贏過基線」判定。輸入 CSV 欄位見 PREREG.md。

用法：
    aggregate.py --phase a [--results-dir results] [--include-smoke]
    aggregate.py --phase b --table results/phaseB/leakage.csv
"""
import argparse
import csv
import glob
import json
import math
import os
import sys

import numpy as np
import yaml
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PHASE_A_RESULTS = "/work/ChihChungHsu/cactus8603/icml/runs/inoculation-lambda/phasea_legacy/manual/results"


def load_criteria():
    return yaml.safe_load(open(os.path.join(ROOT, "configs", "criteria.yaml")))


def spearman(a, b):
    if len(a) < 3:
        return float("nan")
    return float(stats.spearmanr(a, b)[0])


def read_summary(path):
    with open(path) as f:
        return list(csv.DictReader(f))


FD_TRAITS = ("evil", "sycophantic", "hallucinating", "pirate", "spanish", "allcaps", "optimistic")


def gate1_inputs(rdir, config, origin_job=None):
    """Resolve and validate the seven complete, same-job FD inputs."""
    values = []
    jobs = set()
    for trait in FD_TRAITS:
        name = f"grant_{trait}_qwen7b"
        with open(os.path.join(ROOT, "configs", name + ".yaml")) as stream:
            expected = yaml.safe_load(stream)
        expected["dtype"] = config["fd_dtype"]
        expected["n_folds"] = 1
        expected["lora"]["seeds"] = expected["lora"]["seeds"][:config["fd_n_lora_seeds"]]
        expected["microbatch"] = min(expected.get("microbatch", 4), 2)
        folder = name + "_fd"
        with open(os.path.join(rdir, folder, "lambda_raw.json")) as stream:
            data = json.load(stream)
        job = data["job_id"]
        jobs.add(job)
        recorded = data["config"]
        for context, planned in zip(recorded["contexts"], expected["contexts"]):
            if planned.get("prefix_file"):
                context.pop("prefix", None)
        if not job or (origin_job and job != origin_job) or recorded != expected or data["model"] != expected["model"]:
            raise ValueError(f"{folder} does not match the registered FD job and configuration")
        keys = {(seed, 0, defense["name"], tau, config["fd_ctx"], group["name"])
                for seed in expected["lora"]["seeds"] for defense in expected["defenses"]
                for tau in expected["tau_sets"] for group in expected["groups"]}
        actual = set()
        for row in data["rows"]:
            if row["ctx"] != config["fd_ctx"]:
                if "dtau_pred" in row:
                    raise ValueError(f"unexpected finite difference context in {folder}")
                continue
            key = tuple(row[field] for field in ("seed", "fold", "defense", "tau_set", "ctx", "group"))
            if key in actual or not all(np.isfinite(row[f]) for f in ("dtau_pred", "dtau_true", "dtau_central")):
                raise ValueError(f"duplicate or invalid finite difference row in {folder}")
            if (row["n_train_sequences"] != expected["train"]["n"]
                    or row["n_train_items"] != expected["train"]["n"] * row["n_prefixes"]
                    or row["n_tau_items"] != expected["tau_sets"][row["tau_set"]]["n"]
                    or row["n_train_tokens"] <= 0 or row["n_tau_tokens"] <= 0):
                raise ValueError(f"incomplete retained training or evaluation coverage in {folder}")
            actual.add(key)
        if actual != keys:
            raise ValueError(f"{folder} does not have the registered defense/seed/trait coverage")
        values.append((folder, data["rows"]))
    if len(jobs) != 1:
        raise ValueError("formal FD inputs combine more than one job")
    return values


# ----------------------------------------------------------------------------
# phase a
# ----------------------------------------------------------------------------
def gate1(rdir, crit, include_smoke):
    g = crit["gate1"]
    if include_smoke:
        files = sorted(glob.glob(os.path.join(rdir, "*_fd_smoke", "lambda_raw.json")))
        bundles = [(os.path.basename(os.path.dirname(f)), json.load(open(f))["rows"]) for f in files]
    else:
        origin_job = os.environ.get("SLURM_JOB_ID") if os.environ.get("SLURM_JOB_NAME") == "lam_gate1_fd" else None
        try:
            bundles = gate1_inputs(rdir, g, origin_job)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return dict(status="UNTESTABLE", reason=str(exc), n=0)
    pred, true, tags = [], [], []
    per_cfg = {}
    for cfgname, rows in bundles:
        for r in rows:
            if "dtau_pred" not in r:
                continue
            t = r["dtau_central"] if g["use_central"] else r["dtau_true"]
            pred.append(r["dtau_pred"]); true.append(t); tags.append((cfgname, r["defense"]))
            per_cfg.setdefault(cfgname, ([], []))
            per_cfg[cfgname][0].append(r["dtau_pred"]); per_cfg[cfgname][1].append(t)
    if len(pred) < 3:
        return dict(status="NOT_RUN", n=len(pred))
    if not np.isfinite(pred).all() or not np.isfinite(true).all() or not np.ptp(pred) or not np.ptp(true):
        return dict(status="UNTESTABLE", reason="nonfinite or constant finite differences", n=len(pred))
    pooled = float(np.corrcoef(pred, true)[0, 1])
    pc = {k: float(np.corrcoef(v[0], v[1])[0, 1]) if len(v[0]) > 2 else float("nan") for k, v in per_cfg.items()}
    slope = float(np.polyfit(pred, true, 1)[0])
    ok = pooled >= g["pearson_min"]
    return dict(status="PASS" if ok else "FAIL", pearson_pooled=pooled, per_config=pc, slope_true_on_pred=slope,
                n=len(pred), threshold=g["pearson_min"],
                descriptive=gate1_descriptive(per_cfg, pred))


def gate1_descriptive(per_cfg, pred):
    """2026-11-14 新增（L0 條件 2）：逐 trait 的 Pearson、斜率與預測 |Δτ| 最小值。只描述，不參與判定。"""
    def pear(x, y):
        if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
            return float("nan")
        return float(np.corrcoef(x, y)[0, 1])
    def slope(x, y):
        if len(x) < 2 or np.std(x) == 0:
            return float("nan")
        return float(np.polyfit(x, y, 1)[0])
    per_trait = {k: dict(n=len(v[0]), pearson=pear(v[0], v[1]), slope_true_on_pred=slope(v[0], v[1]),
                         min_abs_pred_dtau=float(np.min(np.abs(v[0]))) if len(v[0]) else float("nan"))
                 for k, v in per_cfg.items()}
    return dict(per_trait=per_trait, min_abs_pred_dtau_all=float(np.min(np.abs(pred))),
                note="只描述，不參與閘門 1 判定（PASS/FAIL 只看 pooled Pearson）")


def _lam(rows, defense, tau_set, group, ctx="MEAN"):
    for r in rows:
        if r["defense"] == defense and r["tau_set"] == tau_set and r["group"] == group and r["ctx"] == ctx:
            return float(r["lam"])
    return None


def ref_grant(rdir, ref):
    out = {}
    # G1：Table 13，7 trait × {IP, PPS}，以「保留比例」比較（Λ_d/Λ_none vs expr_d/expr_none）
    xs, ys, used, skipped = [], [], [], []
    for t, info in ref["grant_table13"]["traits"].items():
        p = os.path.join(rdir, f"grant_{t}_qwen7b", "lambda_summary.csv")
        if not os.path.exists(p):
            continue
        rows = read_summary(p)
        l0 = _lam(rows, "none", "trait_heldout", "all")
        for kind in ("ip", "pps"):
            ld = _lam(rows, info[f"{kind}_defense"], "trait_heldout", "all")
            if l0 is None or ld is None or l0 <= 0:
                skipped.append((t, kind)); continue
            xs.append(ld / l0); ys.append(info[kind] / info["none"]); used.append((t, kind))
    out["G1_table13_retained_fraction"] = dict(rho=spearman(xs, ys), n=len(xs), skipped_lambda_none_nonpositive=skipped)
    # G2：Table 10 prevention（sycophancy）
    p = os.path.join(rdir, "grant_sycophantic_qwen7b", "lambda_summary.csv")
    if os.path.exists(p):
        rows = read_summary(p)
        arms = ref["grant_table10_prevention"]["arms"]
        lam, ex, g20, sl, se, sg = [], [], [], [], [], []
        for a, v in arms.items():
            l = _lam(rows, a, "trait_heldout", "all")
            if l is None:
                continue
            lam.append(l); ex.append(v["expr"])
            if a.startswith("single"):
                sl.append(l); se.append(v["expr"]); sg.append(v["g20"])
        out["G2_table10_all_arms"] = dict(rho=spearman(lam, ex), n=len(lam))
        out["G2_table10_nine_singles"] = dict(rho=spearman(sl, se), n=len(sl),
                                              baseline_g20_rho=spearman(sg, se),
                                              baseline_paper_reported=ref["grant_table10_prevention"]["paper_reported_rho_g20_vs_expr_nine_singles"])
    return out


def ref_pg(rdir, ref):
    pts_b, pts_l, pts_r, dir_ok, n_cell = [], [], [], [], 0
    depth_rho = []
    for c in ref["pg_fig5_slopes"]["cells"]:
        p = os.path.join(rdir, f"pg_{c['model']}_id{c['id']}", "lambda_summary.csv")
        if not os.path.exists(p):
            continue
        rows = read_summary(p)
        groups = sorted({r["group"] for r in rows if r["group"].startswith("chunk_")})
        if not groups:
            continue
        def R(g):
            lo = _lam(rows, "none", f"OOD_{c['ood']}", g, "plain")
            li = _lam(rows, "none", f"ID_{c['id']}_heldout", g, "plain")
            return lo, li
        gb, gt = groups[0], groups[-1]
        (ob, ib), (ot, it) = R(gb), R(gt)
        if None in (ob, ib, ot, it) or ib == 0 or it == 0:
            continue
        n_cell += 1
        for beta, lo, li in ((c["bottom"], ob, ib), (c["top"], ot, it)):
            pts_b.append(beta); pts_l.append(lo); pts_r.append(lo / li)
        dir_ok.append(((ob / ib) > (ot / it)) == (c["bottom"] > c["top"]))
        rr = [R(g)[0] / R(g)[1] for g in groups if None not in R(g) and R(g)[1] != 0]
        if len(rr) >= 4:
            depth_rho.append(spearman(rr, list(range(len(rr)))))
    return dict(A_ratio_pooled_bottom_top=dict(rho=spearman(pts_r, pts_b), n_points=len(pts_b), n_cells=n_cell),
                A_rawLambdaOOD_pooled=dict(rho=spearman(pts_l, pts_b)),
                direction_agreement=dict(k=int(sum(dir_ok)), n=len(dir_ok)),
                within_cell_depth_rho_median=(float(np.median(depth_rho)) if depth_rho else None),
                within_cell_depth_rho_note="論文：49/50 條曲線斜率對深度為負秩相關；此處期望為負")


def phase_a(args, crit):
    rdir = os.path.join(ROOT, args.results_dir or PHASE_A_RESULTS)
    ref = json.load(open(os.path.join(ROOT, "configs", "reference.json")))
    g1 = gate1(rdir, crit, args.include_smoke)
    refs = {"grant": ref_grant(rdir, ref), "pg_chunks": ref_pg(rdir, ref)}
    cut = crit["reference"]
    def lab(r):
        if r is None or (isinstance(r, float) and math.isnan(r)):
            return "N/A"
        return "支持" if r >= cut["rho_supportive"] else ("弱" if r < cut["rho_weak"] else "中間")
    flat = []
    for k, v in refs["grant"].items():
        flat.append((f"grant/{k}", v.get("rho")))
    flat.append(("pg/A_ratio_pooled", refs["pg_chunks"]["A_ratio_pooled_bottom_top"]["rho"]))
    res = dict(criteria_version=crit["version"], gate1=g1, reference_only=refs)
    output_name = "aggregate_phaseA_smoke.json" if args.include_smoke else "aggregate_phaseA.json"
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, output_name), "x", encoding="utf-8") as report:
        json.dump(res, report, indent=1, default=str)
    print("=" * 70)
    print(f"[criteria {crit['version']}] 閘門 1（有限差分 Pearson ≥ {crit['gate1']['pearson_min']}）：{g1['status']}"
          + (f"  pooled r={g1['pearson_pooled']:.3f}  n={g1['n']}  斜率(實測/預測)={g1['slope_true_on_pred']:.2f}" if "pearson_pooled" in g1 else ""))
    if "reason" in g1:
        print(g1["reason"])
    if "descriptive" in g1:
        d = g1["descriptive"]
        print("閘門 1 逐 trait（只描述）：min |預測Δτ| 全體 = %.3g" % d["min_abs_pred_dtau_all"])
        for k, v in sorted(d["per_trait"].items()):
            print(f"  {k:34s} n={v['n']:2d}  pearson={v['pearson']:7.3f}  斜率={v['slope_true_on_pred']:7.3f}  min|pred|={v['min_abs_pred_dtau']:.3g}")
    print("-" * 70)
    print("參考（對照論文已報數字；不停損、不入判準）：")
    for k, r in flat:
        print(f"  {k:48s} Spearman = {('%.3f' % r) if r is not None and not math.isnan(r) else 'n/a':>7s}  [{lab(r)}]")
    print(f"  pg 方向一致 {refs['pg_chunks']['direction_agreement']}  深度 ρ 中位 {refs['pg_chunks']['within_cell_depth_rho_median']}")
    print("=" * 70)
    print(f"RESULT_GATE1={g1['status']}")
    sys.exit(0 if g1["status"] == "PASS" or (args.include_smoke and g1["status"] == "NOT_RUN") else 1)


# ----------------------------------------------------------------------------
# phase b（主判準）
# ----------------------------------------------------------------------------
def fisher_pool(rhos, ns):
    z = [np.arctanh(np.clip(r, -0.999999, 0.999999)) for r in rhos]
    w = [max(n - 3, 1) for n in ns]
    return float(np.tanh(np.average(z, weights=w)))


def table_by_task(rows, fam_exclude=()):
    """rows -> {task: {arm: dict(lam, leak, base, family, id_ok)}}，arm 內對 seed 取平均；排除 none。"""
    acc = {}
    for r in rows:
        if r["arm"] == "none" or r["family"] in fam_exclude:
            continue
        a = acc.setdefault(r["task"], {}).setdefault(r["arm"], dict(family=r["family"], leak=[], lam=[], base=[], id_ok=[]))
        a["leak"].append(float(r["leak"])); a["lam"].append(float(r["lam"]))
        if r.get("base_pred", "") not in ("", None):
            a["base"].append(float(r["base_pred"]))
        a["id_ok"].append(str(r.get("id_ok", "True")).lower() in ("true", "1"))
    out = {}
    for t, arms in acc.items():
        out[t] = {k: dict(family=v["family"], leak=np.mean(v["leak"]), lam=np.mean(v["lam"]),
                          base=(np.mean(v["base"]) if v["base"] else None), id_ok=all(v["id_ok"])) for k, v in arms.items()}
    return out


def pooled_rho(tasks, key="lam"):
    rhos, ns = [], []
    for t, arms in tasks.items():
        sub = [a for a in arms.values() if a[key] is not None]
        if len(sub) < 4:
            continue
        rhos.append(spearman([a[key] for a in sub], [a["leak"] for a in sub])); ns.append(len(sub))
    return (fisher_pool(rhos, ns) if rhos else float("nan")), rhos


def phase_b(args, crit):
    c = crit["primary"]
    rows = list(csv.DictReader(open(args.table)))
    tasks = table_by_task(rows)
    rho, per_task = pooled_rho(tasks)
    rng = np.random.default_rng(c["bootstrap_seed"])
    def boot_once(tasks_):
        bt = {}
        for t, arms in tasks_.items():
            keys = list(arms)
            idx = rng.integers(0, len(keys), len(keys))
            bt[t] = {f"{keys[i]}#{j}": arms[keys[i]] for j, i in enumerate(idx)}
        return bt
    boots, dboots = [], []
    for _ in range(c["bootstrap_n"]):
        bt = boot_once(tasks)
        r1, _ = pooled_rho(bt)
        boots.append(r1)
        # 基線：只在共同有定義（base 不為 None）的 arm 上配對
        shared = {t: {k: a for k, a in arms.items() if a["base"] is not None} for t, arms in bt.items()}
        shared = {t: a for t, a in shared.items() if a}
        if shared:
            ra, _ = pooled_rho(shared, "lam"); rb, _ = pooled_rho(shared, "base")
            dboots.append(ra - rb)
    boots = np.array([b for b in boots if not math.isnan(b)])
    lo_q = (1 - c["ci_level"]) / 2
    ci = (float(np.quantile(boots, lo_q)), float(np.quantile(boots, 1 - lo_q))) if len(boots) else (float("nan"),) * 2
    # 拿掉任一家族
    fams = sorted({a["family"] for arms in tasks.values() for a in arms.values()})
    drop = {f: pooled_rho(table_by_task(rows, fam_exclude=(f,)))[0] for f in fams}
    min_drop = min(drop.values()) if drop else float("nan")
    tasks_pos = sum(1 for r in per_task if r > c["per_task_floor"])
    passed = (rho >= c["pooled_rho_pass"] and ci[0] >= c["ci_lower_pass"] and tasks_pos >= c["min_tasks_positive"]
              and min_drop >= c["drop_family_min"])
    status = "PASS" if passed else ("GRAY" if rho >= c["gray_lo"] else "FAIL")
    # 基線
    q = c["baseline_delta_ci_level"]
    base = None
    if dboots:
        d = np.array(dboots)
        lo, hi = float(np.quantile(d, (1 - q) / 2)), float(np.quantile(d, 1 - (1 - q) / 2))
        base = dict(delta_ci=(lo, hi), beats_baseline=bool(lo > 0), n_boot=len(d), baseline=c["baseline"])
    flagged = sorted((t, k) for t, arms in tasks.items() for k, a in arms.items() if not a["id_ok"])
    rho_no_flag, _ = pooled_rho({t: {k: a for k, a in arms.items() if a["id_ok"]} for t, arms in tasks.items()})
    res = dict(criteria_version=crit["version"], status=status, pooled_rho=rho, ci=ci, per_task_rho=per_task,
               tasks_positive=tasks_pos, drop_family=drop, baseline=base, id_match_flagged=flagged,
               pooled_rho_excluding_flagged=rho_no_flag, n_arms={t: len(a) for t, a in tasks.items()})
    outp = os.path.join(os.path.dirname(args.table), "aggregate_phaseB.json")
    json.dump(res, open(outp, "w"), indent=1, default=str)
    print(json.dumps(res, indent=1, default=str))
    print(f"RESULT_PRIMARY={status}")
    if status == "GRAY":
        print("灰區：僅准一次事先定好的加碼 = 加 1 個 seed 與 6 個 arm（見 PREREG.md），之後以同一腳本重判，不得再加碼。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["a", "b"], required=True)
    ap.add_argument("--results-dir", help="Phase A 原始結果目錄")
    ap.add_argument("--output-dir", default=PHASE_A_RESULTS, help="Phase A 新報表目錄")
    ap.add_argument("--include-smoke", action="store_true")
    ap.add_argument("--table", default=None)
    args = ap.parse_args()
    crit = load_criteria()
    (phase_a if args.phase == "a" else phase_b)(args, crit)


if __name__ == "__main__":
    main()
