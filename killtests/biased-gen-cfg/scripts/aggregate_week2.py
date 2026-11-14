#!/usr/bin/env python
"""第 2 週機械判讀（DECISIONS_fable 條文 2、3、4、5；判準全部來自 configs/criteria_week2.json）。
輸入（皆為 eval_hier_oracle.py 的 npz，同一個判定內的所有 run 必須一次評完）：
  --k6      K=6 條件模型 6 個 run（pick 0–5）的一般評估（分支含 cond、cfg3），用於 m6 與 (b) 閘門
  --k1      K=1 常數條件 run（N 個，pick 0–N-1）的一般評估（--n-eval 1000，條件輸入 0），用於 m1
  --k6c0    K=6 模型在類別 0 前 700 筆（--class-filter 0 --n-eval 700）的評估，取 cond 分支 -> t*_c（只用 pick 0–N-1，--k6c0 的 run 須為這 N 個）
  --u833c0  類別 0 專用無條件模型（pdrop=1.0，batch 85）同上評估，取 uncond 分支 -> t*_u833
  --k1c0    K=1 模型在類別 0 前 700 筆的評估，取 cond 分支 -> t*_u5000
  --h       第 1 週 3 個階層 run 的實測平均 GPU 小時（決定 N；只用於一致性檢查與標示低檢定力）
  --gpu-used / --gpu-projected-d   條文 6 停點 2：已用 GPU 小時與依 (d) 實測吞吐推估的用量；合計 > 115 -> 輸出「先停下回報」
2026-11-14 新增（代理落差稽核後、結果前定案）：條文 7 可達性 H∞（--k6 的 npz 須含 pair_cos_gamma）、條文 8 第二探針（(c) FAIL 帶才觸發）、條文 9 層級用語、條文 6 停點。
各輸出段缺檔時該段記「未能測試（缺檔）」。輸出 results/week2_verdict.json。
"""
import argparse, json, os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
import reach
ap = argparse.ArgumentParser()
ap.add_argument("--criteria", default=os.path.join(ROOT, "configs", "criteria_week2.json"))
for k in ("k6", "k1", "k6c0", "u833c0", "k1c0"):
    ap.add_argument("--" + k, default=None)
ap.add_argument("--h", type=float, default=None)
ap.add_argument("--gpu-used", type=float, default=None); ap.add_argument("--gpu-projected-d", type=float, default=None)
ap.add_argument("--out", default=os.path.join(ROOT, "results", "week2_verdict.json"))
args = ap.parse_args()
crit = json.load(open(args.criteria))
K = crit["smooth_window"]; RISE = crit["interior_min_rise"]; NB = crit["n_boot"]
verdict = {"criteria_file": os.path.abspath(args.criteria), "sections": {}}


def smooth(y, k):
    y = np.asarray(y, float); h = k // 2
    return np.array([y[max(0, i - h): i + h + 1].mean() for i in range(len(y))])


def tstar(ep, y, k=K):
    y = smooth(y, k)
    return float("nan") if (len(y) == 0 or not np.all(np.isfinite(y))) else float(ep[int(np.argmin(y))])


def interior_info(ep, y, k=K, rise=RISE):
    y = smooth(y, k)
    if len(y) == 0 or not np.all(np.isfinite(y)): return False, "nonfinite"
    i = int(np.argmin(y))
    if i == 0: return False, "first"
    if i == len(y) - 1: return False, "last"
    if y[i:].max() < rise * y[i]: return False, "insufficient_rise"
    return True, None


def load_mask():
    p = crit["t_mask_file"]; p = p if os.path.isabs(p) else os.path.join(ROOT, p)
    return json.load(open(p))["keep_t"]


KEEP = load_mask()


class Run:
    """一份 npz：每個分支、每對的主曲線（ε 空間、逐 t 正規化、遮罩等權平均；條文 1）。"""
    def __init__(self, path):
        d = np.load(path, allow_pickle=True)
        assert "pair_cos_eps" in d.files, f"{path} 缺 pair_cos_eps"
        self.path = path; self.ep = d["epochs"]; self.br = list(d["branches"]); self.pairs = d["pairs"]; self.picks = [int(x) for x in d["picks"]]
        self.R = len(self.picks); self.P = len(self.pairs)
        tg = [int(x) for x in d["t_grid"]]; assert all(t in tg for t in KEEP), (KEEP, tg)
        idx = [tg.index(t) for t in KEEP]
        pce = d["pair_cos_eps"]; self.curve = {}
        for bi, b in enumerate(self.br):
            y = pce[:, :, :, :2, bi].mean(axis=3)[:, :, idx]
            m = y.mean(axis=1, keepdims=True); bad = ~np.isfinite(m) | (m == 0)
            with np.errstate(all="ignore"):
                self.curve[b] = (y / np.where(bad, np.nan, m)).mean(axis=2)
        self.git = str(d["git_hash"]) if "git_hash" in d.files else None
        self.tgrid = tg; self.mse = d["mse_oracle"] if "mse_oracle" in d.files else None; self.ce = d["ce"] if "ce" in d.files else None; self.t_idx = idx
        self.gamma = reach.gamma_curves(d["pair_cos_gamma"], idx) if "pair_cos_gamma" in d.files else None

    def ts(self, b, p): return tstar(self.ep, self.curve[b][p])
    def inter(self, b, p): return interior_info(self.ep, self.curve[b][p])


def weights(rng, R, pairs):
    m = np.bincount(rng.integers(0, R, R), minlength=R).astype(float)
    return m, m[pairs[:, 0]] * m[pairs[:, 1]]


def max_invalid(P): return P // crit["max_invalid_pair_divisor"]   # ⌊P/3⌋（2026-11-14 修正：原 floor(P*0.3333) 在 P=15 給 4，應為 5；與 PREREG／條文 2 及 aggregate_week1 一致）


def main_metric(run, label):
    """m = log[t*(cfg3)/t*(cond)]，只用兩條曲線都有內部最低點的對；回傳 dict（含每對無效原因）。"""
    nb, db = crit["main_branches"]["numerator"], crit["main_branches"]["denominator"]
    x = np.full(run.P, np.nan); valid = []; info = []
    for p in range(run.P):
        rs = {b: run.inter(b, p) for b in (nb, db)}; v = all(o for o, _ in rs.values())
        info.append({"pair": [int(i) for i in run.pairs[p]], "valid": v, "reason": {b: r for b, (o, r) in rs.items() if not o}})
        if v: valid.append(p); x[p] = np.log(run.ts(nb, p) / run.ts(db, p))
    n_inv = run.P - len(valid)
    return {"label": label, "x": x, "valid": valid, "n_pairs": run.P, "n_invalid": n_inv, "max_invalid": max_invalid(run.P),
            "testable": n_inv <= max_invalid(run.P) and len(valid) > 0, "pairs_info": info}


def wmean(x, w, valid):
    den = w[valid].sum()
    return None if den == 0 else float((w[valid] * x[valid]).sum() / den)


def pub(mm, ci=None, skipped=None):
    d = {k: v for k, v in mm.items() if k not in ("x", "valid")}
    d["per_pair"] = [None if not np.isfinite(v) else float(v) for v in mm["x"]]
    d["mean"] = float(mm["x"][mm["valid"]].mean()) if mm["valid"] else None
    if ci is not None: d["ci95"] = ci; d["bootstrap_skipped"] = skipped
    return d


def boot_single(run, mm):
    rng = np.random.default_rng(crit["bootstrap_seed"]); bs = []; sk = 0
    for _ in range(NB):
        _, w = weights(rng, run.R, run.pairs); v = wmean(mm["x"], w, mm["valid"])
        if v is None: sk += 1
        else: bs.append(v)
    return ([float(q) for q in np.percentile(bs, [2.5, 97.5])] if bs else [None, None]), sk


S = verdict["sections"]
need = lambda k: getattr(args, k) is not None and os.path.exists(getattr(args, k))

# ---------------- K=6 主指標 m6 與 (b) 閘門（第 2 週 15 對中 >= 10） ----------------
r6 = m6 = ci6 = None
if need("k6"):
    r6 = Run(args.k6); m6 = main_metric(r6, "K=6 m6")
    n_cond = sum(1 for p in range(r6.P) if r6.inter("cond", p)[0]); req = crit["c"]["b_gate_min_cond_interior"].get(str(r6.P), int(np.ceil(2 * r6.P / 3)))
    S["b_gate"] = {"cond_interior_pairs": n_cond, "required": req, "result": "PASS" if n_cond >= req else "未能測試（20000 epochs 內條件分支未進入偏差期）"}
    if m6["testable"]:
        ci6, sk6 = boot_single(r6, m6)
        S["m6"] = pub(m6, ci6, sk6)
    else:
        S["m6"] = {**pub(m6), "status": "未能測試（無偏差期或設限；不得判 PASS 或 FAIL）"}
else:
    S["m6"] = "未能測試（缺 --k6）"

# ---------------- 負控制（條文 3） ----------------
r1 = m1 = None; neg = {"status": "未能測試（缺 --k1）"}; delta = None
if need("k1"):
    r1 = Run(args.k1); m1 = main_metric(r1, "K=1 m1")
    nc = crit["negative_control"]
    neg = {"n_runs": r1.R, "low_power": r1.R <= nc["low_power_if_n_runs"], "expected_n_runs_from_h": None}
    if args.h is not None:
        neg["h_gpu_hours"] = args.h
        neg["expected_n_runs_from_h"] = nc["n_runs_if_h_le_threshold"] if args.h <= nc["h_gpu_hours_threshold"] else nc["n_runs_if_h_gt_threshold"]
        neg["n_runs_matches_rule"] = neg["expected_n_runs_from_h"] == r1.R
    if not m1["testable"]:
        neg.update(pub(m1)); neg["status"] = "未能測試（K=1 無效對過多）：(c) 加註後交人工判讀"
    else:
        ci1, sk1 = boot_single(r1, m1); neg.update(pub(m1, ci1, sk1))
        passed = bool(ci1[0] is not None and ci1[0] > -nc["null_band"] and ci1[1] < nc["null_band"])
        neg["status"] = "通過（m1 的 95% CI 落在 ±log1.1 內）" if passed else "未通過：(c) 的 PASS 改用 Δ = m6 - m1"
        neg["passed"] = passed
        # Δ 配對 bootstrap：K=6 只取與 K=1 相同的 pick
        if r6 is not None and m6["testable"]:
            common = [p for p in r1.picks if p in r6.picks]
            sel6 = [p for p in range(r6.P) if all(int(i) in common for i in r6.pairs[p])]
            v6 = [p for p in sel6 if p in m6["valid"]]
            # 以 pick 對應 K=1 的 pair 索引
            key1 = {tuple(sorted((r1.picks[int(i)], r1.picks[int(j)]))): p for p, (i, j) in enumerate(r1.pairs)}
            pk6 = [tuple(sorted((r6.picks[int(i)], r6.picks[int(j)]))) for i, j in r6.pairs]
            pairs_c = [(pk6.index(k), key1[k]) for k in key1 if k in pk6]
            vp = [(a, b) for a, b in pairs_c if a in m6["valid"] and b in m1["valid"]]
            if vp:
                x6 = np.array([m6["x"][a] for a, _ in vp]); x1 = np.array([m1["x"][b] for _, b in vp])
                ids = [(tuple(sorted(pk6[a]))) for a, _ in vp]   # pick 對（兩邊相同）
                pa = np.array([[k_[0], k_[1]] for k_ in ids])
                rng = np.random.default_rng(crit["bootstrap_seed"]); ds = []; skd = 0
                all_picks = sorted(common)
                for _ in range(NB):
                    cnt = {p_: 0 for p_ in all_picks}
                    for d_ in rng.integers(0, len(all_picks), len(all_picks)): cnt[all_picks[d_]] += 1
                    w = np.array([cnt[a] * cnt[b] for a, b in pa], float)
                    if w.sum() == 0: skd += 1; continue
                    ds.append(float((w * x6).sum() / w.sum() - (w * x1).sum() / w.sum()))
                dmean = float(x6.mean() - x1.mean()); dci = [float(q) for q in np.percentile(ds, [2.5, 97.5])] if ds else [None, None]
                delta = {"mean": dmean, "ci95": dci, "n_common_valid_pairs": len(vp), "bootstrap_skipped": skd,
                         "pass": bool(dmean <= np.log(1 - crit["c"]["min_early_fraction"]) and dci[1] is not None and dci[1] < 0)}
S["negative_control"] = neg

# ---------------- (c) 綜合 ----------------
if m6 is not None and m6["testable"]:
    c = {"m6_mean": float(m6["x"][m6["valid"]].mean()), "m6_ci95": ci6}
    lo, hi = ci6
    fail_band = bool(lo > -np.log(1 + crit["c"]["fail_band"]) and hi < np.log(1 + crit["c"]["fail_band"]))
    raw_pass = bool(c["m6_mean"] <= np.log(1 - crit["c"]["min_early_fraction"]) and hi < 0)
    c["fail_band_met"] = fail_band
    if "passed" not in neg:
        c["status"] = "負控制未能測試：(c) 加註後交人工判讀"; c["pass"] = None
    elif neg["passed"]:
        c["status"] = "PASS" if raw_pass else ("FAIL 帶" if fail_band else "其他組合，人工判讀"); c["pass"] = raw_pass
    else:
        c["delta"] = delta
        if delta and delta["pass"]: c["status"] = "PASS（以 Δ 判定）"; c["pass"] = True
        else: c["status"] = "FAIL 帶（看 m6）" if fail_band else "其他組合，人工判讀（負控制未過且 Δ 不滿足）"; c["pass"] = False if fail_band else None
    S["c_main"] = c
else:
    S["c_main"] = {"status": "未能測試（m6 無法計算或無效對過多）", "pass": None}

# ---------------- 正控制（條文 4） ----------------
pcn = crit["positive_control"]
if all(need(k) for k in ("k6c0", "u833c0", "k1c0")):
    rc, ru, r5 = Run(args.k6c0), Run(args.u833c0), Run(args.k1c0)
    ok_shape = rc.picks == ru.picks == r5.picks
    if not ok_shape:
        S["positive_control"] = {"status": f"未能測試（三組 pick 不一致：{rc.picks} {ru.picks} {r5.picks}）"}
    else:
        grp = {"c": (rc, "cond"), "u833": (ru, "uncond"), "u5000": (r5, "cond")}
        inv = {}; valid = None
        for g, (r, b) in grp.items():
            vs = [p for p in range(r.P) if r.inter(b, p)[0]]
            inv[g] = {"n_invalid": r.P - len(vs), "reasons": [{"pair": [int(i) for i in r.pairs[p]], "reason": r.inter(b, p)[1]} for p in range(r.P) if not r.inter(b, p)[0]]}
            valid = set(vs) if valid is None else valid & set(vs)
        P = rc.P; too_many = any(v["n_invalid"] > max_invalid(P) for v in inv.values())
        if too_many or not valid:
            S["positive_control"] = {"status": "未能測試（任一組無效對數 > floor(P/3)）", "invalid": inv}
        else:
            vl = sorted(valid)
            lt = {g: np.array([np.log(r.ts(b, p)) for p in range(P)]) for g, (r, b) in grp.items()}
            def stat(w):
                L = {g: float((w[vl] * lt[g][vl]).sum() / w[vl].sum()) for g in lt}
                return L, (L["c"] - L["u5000"]) / (L["u833"] - L["u5000"]) if L["u833"] != L["u5000"] else None
            L0, lam0 = stat(np.ones(P))
            rng = np.random.default_rng(crit["bootstrap_seed"]); lams = []; skp = 0
            for _ in range(NB):
                _, w = weights(rng, rc.R, rc.pairs)
                if w[vl].sum() == 0: skp += 1; continue
                _, lm = stat(w)
                if lm is None: skp += 1; continue
                lams.append(lm)
            lci = [float(q) for q in np.percentile(lams, [2.5, 97.5])] if lams else [None, None]
            sep = L0["u5000"] - L0["u833"]
            disc = bool(sep >= pcn["discrimination_min"])
            d = {"L": L0, "lambda": lam0, "lambda_ci95": lci, "bootstrap_skipped": skp, "separation_Lu5000_minus_Lu833": sep,
                 "discriminating": disc, "abs_Lc_minus_Lu833": abs(L0["c"] - L0["u833"]), "n_valid_pairs": len(vl), "invalid": inv}
            if not disc: d["status"] = "控制無分辨力（未能測試）"
            elif lci[0] is not None and lci[0] >= pcn["lambda_ci_lower_min"] and d["abs_Lc_minus_Lu833"] <= pcn["abs_Lc_minus_Lu833_max"]:
                d["status"] = "通過"
            else: d["status"] = "n/K 時鐘前提不成立（題目層級發現；(c)(d) 照常判讀，但論文不得主張 n/K 時鐘，交使用者決定）"
            S["positive_control"] = d
else:
    S["positive_control"] = {"status": "未能測試（缺 --k6c0/--u833c0/--k1c0）"}

# ---------------- 條文 7：可達性（w→∞ 極限）；K=6 的 npz 須含 pair_cos_gamma ----------------
rcr = crit["reachability"]; RS = None
if r6 is not None and r6.gamma is not None:
    RS = reach.h_inf(r6.ep, np.stack([r6.curve["cond"][p] for p in range(r6.P)]), r6.gamma, r6.pairs, r6.R, lambda e_, y: tstar(e_, y), lambda e_, y: interior_info(e_, y),
                     max_invalid(r6.P), rcr["reachable_max_log_ratio"], rcr["n_boot"], rcr["bootstrap_seed"])
    S["reachability"] = RS
else:
    S["reachability"] = {"state": "可達性未知", "note": "K=6 npz 缺 pair_cos_gamma（須以 eval_hier_oracle.py --gamma 重評）或缺 --k6"}
c_main = S.get("c_main", {}); cst = str(c_main.get("status", ""))
c_fail_band = cst.startswith("FAIL 帶")
if cst.startswith("PASS") and S["reachability"]["state"] == "構造不可達":
    c_main["note"] = "與上界矛盾：(c) PASS 照記，但條文 7 判為構造不可達；列入人工審查"
    verdict.setdefault("manual_review", []).append("(c) PASS 與可達性上界（構造不可達）矛盾")

# ---------------- 條文 8：唯一的第二探針（oracle 偏差間隙放大比；(c) FAIL 帶才觸發；0 GPU） ----------------
pb = crit["second_probe_gap_amplification"]
if not c_fail_band:
    S["second_probe"] = {"state": "未觸發", "note": "(c) 未落在 FAIL 帶"}
elif r6 is None or r1 is None or r6.mse is None or r1.mse is None or r6.ce is None:
    S["second_probe"] = {"state": "未能測試", "reason": "缺 --k6 或 --k1，或 npz 缺 mse_oracle／ce"}
else:
    ce6 = r6.ce[:, :, :, -1, r6.br.index("cond")].mean(axis=(0, 2)); ys = smooth(ce6, K)
    ei = int(np.argmin(ys)) if np.all(np.isfinite(ys)) else None
    if ei is None: S["second_probe"] = {"state": "未能測試", "reason": "K=6 測試輸入 CE 曲線含非有限值"}
    elif ei == 0: S["second_probe"] = {"state": "未能測試", "reason": "e† 落在第一格", "e_dagger": int(r6.ep[0])}
    elif int(r6.ep[ei]) not in [int(x) for x in r1.ep]: S["second_probe"] = {"state": "未能測試", "reason": "K=1 沒有 e† 對應的 checkpoint", "e_dagger": int(r6.ep[ei])}
    else:
        e1 = [int(x) for x in r1.ep].index(int(r6.ep[ei])); bc6, bf6 = r6.br.index("cond"), r6.br.index("cfg3"); bc1, bf1 = r1.br.index("cond"), r1.br.index("cfg3")
        tsel6, tsel1 = list(r6.t_idx), list(r1.t_idx)   # 凍結的 t 遮罩（條文 1）
        g6, c6 = reach.gap_terms(r6.mse, ei, tsel6, bc6, bf6); g1, c1 = reach.gap_terms(r1.mse, e1, tsel1, bc1, bf1)
        out = reach.probe_amplification(g6, c6, r6.picks, g1, c1, r1.picks, pb["pass"]["gamma6_min"], pb["n_boot"], pb["bootstrap_seed"], pb["max_skipped_fraction"])
        # 描述：全 t 網格版（不進判定）
        a6, a_c6 = reach.gap_terms(r6.mse, ei, list(range(len(r6.tgrid))), bc6, bf6); a1, a_c1 = reach.gap_terms(r1.mse, e1, list(range(len(r1.tgrid))), bc1, bf1)
        out["gamma6_all_t_descriptive"] = reach.gamma_ratio(a6, a_c6)[0]; out["gamma1_all_t_descriptive"] = reach.gamma_ratio(a1, a_c1, [r1.picks.index(p) for p in out["common_picks"]])[0]
        out.update(e_dagger=int(r6.ep[ei]), e_dagger_is_last_grid=bool(ei == len(r6.ep) - 1), t_used=KEEP, note="探針 PASS 不等於題目 PASS")
        S["second_probe"] = out

# ---------------- 條文 9：(c) 落在 FAIL 帶時的層級用語 ----------------
if c_fail_band:
    probe_state = S["second_probe"]["state"]
    g4ok = S["positive_control"].get("status") == "通過" if isinstance(S.get("positive_control"), dict) else False
    S["tier"] = {"reachability": S["reachability"]["state"], "second_probe": probe_state, "gate4_passed": g4ok, **reach.tier_text(S["reachability"]["state"], probe_state, g4ok)}
    S["stop_point_1"] = "(c) 落在 FAIL 帶：不自動送 (d)。已做條文 8 第二探針（0 GPU）；連同條文 7 與閘門 4 的結果回報使用者，由使用者決定要不要送 (d)。"
else:
    S["tier"] = {"note": "(c) 未落在 FAIL 帶；不適用條文 9 層級用語"}

# ---------------- 條文 6 停點 2 ----------------
if args.gpu_used is not None and args.gpu_projected_d is not None:
    tot = args.gpu_used + args.gpu_projected_d
    S["stop_point_2"] = {"gpu_used": args.gpu_used, "gpu_projected_d": args.gpu_projected_d, "total": tot, "cap": crit["stop_points"]["gpu_hours_cap"],
                         "action": "先停下回報使用者，不自行刪減" if tot > crit["stop_points"]["gpu_hours_cap"] else "未超過上限"}

verdict["git_hash_k6"] = r6.git if r6 else None
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
json.dump(verdict, open(args.out, "w"), indent=1, ensure_ascii=False, default=lambda o: None if o is None else str(o))
print(json.dumps(verdict, indent=1, ensure_ascii=False, default=str))
