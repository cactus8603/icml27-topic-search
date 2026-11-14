#!/usr/bin/env python
"""第 2 週機械判讀（報告 60 §1，判準來自 configs/criteria_week2.json）。
輸入（皆為 eval_hier_oracle.py 的 npz，同一個判定內的所有 run 必須一次評完）：
  --k6      K=6 條件模型 6 個 run（pick 0–5）的一般評估（分支含 cond、cfg3），用於 m6 與 (b) 閘門
  --k1      K=1 常數條件 run（N 個，pick 0–N-1）的一般評估（--n-eval 1000，條件輸入 0），用於 m1
  --k6c0    K=6 模型在類別 0 前 700 筆（--class-filter 0 --n-eval 700）的評估，取 cond 分支 -> t*_c（只用 pick 0–N-1，--k6c0 的 run 須為這 N 個）
  --u833c0  類別 0 專用無條件模型（pdrop=1.0，batch 85）同上評估，取 uncond 分支 -> t*_u833
  --k1c0    K=1 模型在類別 0 前 700 筆的評估，取 cond 分支 -> t*_u5000
  --h       第 1 週 3 個階層 run 的實測平均 GPU 小時（決定 N；只用於一致性檢查與標示低檢定力）
  --week1-verdict  第 1 週機械判讀 JSON，用於現象閘門（預設 results/week1_verdict.json）
  --gpu-used / --gpu-projected-d   入池後 CIFAR (d) 的預算判讀
H∞、舊版 (c) 及層級用語為描述性資訊；oracle 間隙放大探針一律計算可用部分。
各輸出段缺檔時該段記「未能測試（缺檔）」。輸出 results/week2_verdict.json。
"""
import argparse, json, os
from itertools import combinations
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
import reach
from schedule import WEEK2_T_GRID, week2_epochs
from week2_identity import expected_eval, manifest_error
ap = argparse.ArgumentParser()
ap.add_argument("--criteria", default=os.path.join(ROOT, "configs", "criteria_week2.json"))
for k in ("k6", "k1", "k6c0", "u833c0", "k1c0"):
    ap.add_argument("--" + k, default=None)
ap.add_argument("--h", type=float, default=None)
ap.add_argument("--week1-verdict", default=os.path.join(ROOT, "results", "week1_verdict.json"))
ap.add_argument("--gpu-used", type=float, default=None); ap.add_argument("--gpu-projected-d", type=float, default=None)
ap.add_argument("--out", default=os.path.join(ROOT, "results", "week2_verdict.json"))
args = ap.parse_args()
crit = json.load(open(args.criteria))
screen = crit["candidate_screen"]
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
    with open(p) as fd:
        return json.load(fd)


MASK = load_mask()
KEEP = MASK["keep_t"]


class Run:
    """一份 npz：每個分支、每對的主曲線（ε 空間、逐 t 正規化、遮罩等權平均；條文 1）。"""
    def __init__(self, path):
        d = np.load(path, allow_pickle=True)
        assert "pair_cos_eps" in d.files, f"{path} 缺 pair_cos_eps"
        self.path = path; self.ep = d["epochs"]; self.br = list(d["branches"]); self.pairs = d["pairs"]; self.picks = [int(x) for x in d["picks"]]
        self.R = len(self.picks); self.P = len(self.pairs)
        tg = [int(x) for x in d["t_grid"]]
        self.complete_grid = self.ep.tolist() == week2_epochs() and tuple(tg) == WEEK2_T_GRID
        try:
            self.manifest = json.loads(d["week2_manifest"].item()) if "week2_manifest" in d.files else None
        except (ValueError, TypeError):
            self.manifest = None
        self.n_eval = d["n_eval"].item() if "n_eval" in d.files and d["n_eval"].shape == () else None
        self.test_block = d["test_block"].item() if "test_block" in d.files and d["test_block"].shape == () else None
        self.guidance = d["guidance"].tolist() if "guidance" in d.files else None
        idx = [tg.index(t) for t in KEEP if t in tg]
        full_mask = len(idx) == len(KEEP)
        pce = d["pair_cos_eps"]; self.curve = {}
        for bi, b in enumerate(self.br):
            if not full_mask:
                self.curve[b] = np.full((self.P, len(self.ep)), np.nan)
                continue
            y = pce[:, :, :, :2, bi].mean(axis=3)[:, :, idx]
            m = y.mean(axis=1, keepdims=True); bad = ~np.isfinite(m) | (m == 0)
            with np.errstate(all="ignore"):
                self.curve[b] = (y / np.where(bad, np.nan, m)).mean(axis=2)
        self.git = str(d["git_hash"]) if "git_hash" in d.files else None
        self.tgrid = tg; self.mse = d["mse_oracle"] if "mse_oracle" in d.files else None; self.ce = d["ce"] if "ce" in d.files else None; self.t_idx = idx
        self.gamma = reach.gamma_curves(d["pair_cos_gamma"], idx) if "pair_cos_gamma" in d.files and full_mask else None

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
        rs = {b: run.inter(b, p) if b in run.curve else (False, "missing_branch") for b in (nb, db)}
        v = all(o for o, _ in rs.values())
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


def boot_single(run, mm, ci_level=0.95):
    rng = np.random.default_rng(crit["bootstrap_seed"]); bs = []; sk = 0
    for _ in range(NB):
        _, w = weights(rng, run.R, run.pairs); v = wmean(mm["x"], w, mm["valid"])
        if v is None: sk += 1
        else: bs.append(v)
    return reach.percentile_interval(bs, ci_level), sk


def expected_panel(run, sizes):
    return (run is not None and run.R in sizes and sorted(run.picks) == list(range(run.R))
            and run.pairs.tolist() == [list(pair) for pair in combinations(range(run.R), 2)])


def formal_panel_error(run, sizes, role):
    if not expected_panel(run, sizes):
        return "run 或配對未形成完整區塊"
    if not run.complete_grid:
        return "缺凍結的完整 checkpoint 或 8 個 t"
    error = manifest_error(run.manifest, role, run.picks, MASK["data_sha256"])
    if error:
        return error
    expected = expected_eval(role)
    if run.n_eval != expected["n_eval"] or run.test_block != expected["test_block"] or run.guidance != expected["guidance"]:
        return "NPZ 樣本數、測試區塊或 guidance 與 manifest 不符"
    branches = ["uncond"] if expected["run_type"] == "uncond_only" else ["cond", "uncond", *[f"cfg{w:g}" for w in run.guidance]]
    if run.br != branches:
        return "NPZ 分支與 manifest 不符"
    return None


def same_sources(left, right):
    a = {r["training"]["pick"]: r["meta_sha256"] for r in left.manifest["runs"]}
    return all(a.get(r["training"]["pick"]) == r["meta_sha256"] for r in right.manifest["runs"])


def pick_pairs(run):
    return [tuple(sorted((run.picks[i], run.picks[j]))) for i, j in run.pairs]


def matched_delta(r6, m6, r1, m1):
    # 配對索引依各 NPZ 的模型排列而定。配對差以共享區塊的 pick 對齊。
    pair_to_index = {pair: i for i, pair in enumerate(pick_pairs(r6))}
    x = np.array([m6["x"][pair_to_index[pair]] for pair in pick_pairs(r1)]) - m1["x"]
    valid = np.flatnonzero(np.isfinite(x))
    required = r1.P - max_invalid(r1.P)
    result = {"n_common_valid_pairs": len(valid), "n_common_pairs": r1.P, "required_valid_pairs": required}
    if len(valid) < required:
        return {**result, "reason": "共同有效配對不足"}
    ci, skipped = boot_single(r1, {"x": x, "valid": valid}, screen["signal_ci_level"])
    result.update(mean=float(x[valid].mean()), ci=ci, bootstrap_skipped=skipped)
    if ci[0] is None:
        result["reason"] = "共同有效配對足夠，bootstrap 無有效重抽"
    return result


S = verdict["sections"]
need = lambda k: getattr(args, k) is not None and os.path.exists(getattr(args, k))

# ---------------- K=6 主指標 m6 與 (b) 閘門（第 2 週 15 對中 >= 10） ----------------
r6 = m6 = ci6 = None
panel6_error = "缺 --k6"
if need("k6"):
    r6 = Run(args.k6)
    panel6_error = formal_panel_error(r6, (6,), "k6")
    m6 = main_metric(r6, "K=6 m6")
    n_cond = sum(1 for p in range(r6.P) if r6.inter("cond", p)[0]) if "cond" in r6.curve else 0
    req = crit["c"]["b_gate_min_cond_interior"]["15"]
    b_status = f"未能測試（K=6：{panel6_error}）" if panel6_error else (
        "PASS" if n_cond >= req else "未能測試（條件分支有內部最低點的配對不足）")
    S["b_gate"] = {"cond_interior_pairs": n_cond, "required": req, "result": b_status}
    if m6["testable"]:
        ci6, sk6 = boot_single(r6, m6)
        S["m6"] = pub(m6, ci6, sk6)
    else:
        S["m6"] = {**pub(m6), "status": "未能測試（無偏差期或設限；不得判 PASS 或 FAIL）"}
else:
    S["m6"] = "未能測試（缺 --k6）"
    S["b_gate"] = {"result": "未能測試（缺 --k6）"}
panel6_ok = panel6_error is None

# ---------------- 負控制（條文 3） ----------------
r1 = m1 = None; neg = {"status": "未能測試（缺 --k1）"}; delta = None
panel1_error = "缺 --k1"
nc = crit["negative_control"]
if args.h is None:
    control_sizes = (nc["n_runs_if_h_le_threshold"], nc["n_runs_if_h_gt_threshold"])
else:
    size_key = "n_runs_if_h_le_threshold" if args.h <= nc["h_gpu_hours_threshold"] else "n_runs_if_h_gt_threshold"
    control_sizes = (nc[size_key],)
if need("k1"):
    r1 = Run(args.k1); m1 = main_metric(r1, "K=1 m1")
    neg = {"n_runs": r1.R, "low_power": r1.R <= nc["low_power_if_n_runs"], "expected_n_runs_from_h": None}
    if args.h is not None:
        neg["h_gpu_hours"] = args.h
        neg["expected_n_runs_from_h"] = control_sizes[0]
        neg["n_runs_matches_rule"] = r1.R in control_sizes
    panel1_error = formal_panel_error(r1, control_sizes, "k1")
    neg["valid_panel"] = panel1_error is None
    if not m1["testable"]:
        neg.update(pub(m1)); neg["status"] = "未能測試（K=1 無效對過多）：(c) 加註後交人工判讀"
    else:
        ci1, sk1 = boot_single(r1, m1); neg.update(pub(m1, ci1, sk1))
        if panel1_error:
            neg["status"] = f"未能測試（K=1：{panel1_error}）"
        elif ci1[0] is None:
            neg["status"] = "未能測試（K=1 bootstrap 無有效重抽）"
        else:
            passed = bool(ci1[0] > -nc["null_band"] and ci1[1] < nc["null_band"])
            neg["status"] = "通過（m1 的 95% CI 落在 ±log1.1 內）" if passed else "未通過（S-a 改用 Δ = m6 - m1）"
            neg["passed"] = passed
            if panel6_ok and m6["testable"] and not passed:
                delta = matched_delta(r6, m6, r1, m1)
panel1_ok = panel1_error is None
S["negative_control"] = neg

# ---------------- (c) 綜合 ----------------
if m6 is not None and m6["testable"]:
    c = {"m6_mean": float(m6["x"][m6["valid"]].mean()), "m6_ci95": ci6}
    lo, hi = ci6
    fail_band = bool(lo is not None and lo > -np.log(1 + crit["c"]["fail_band"]) and hi < np.log(1 + crit["c"]["fail_band"]))
    raw_pass = bool(hi is not None and c["m6_mean"] <= np.log(1 - crit["c"]["min_early_fraction"]) and hi < 0)
    c["fail_band_met"] = fail_band
    if not panel1_ok or "passed" not in neg:
        c["status"] = "負控制未能測試：(c) 加註後交人工判讀"; c["pass"] = None
    elif neg["passed"]:
        c["status"] = "PASS" if raw_pass else ("FAIL 帶" if fail_band else "其他組合，人工判讀"); c["pass"] = raw_pass
    else:
        c["delta"] = delta
        if delta and "reason" not in delta and delta["mean"] <= np.log(1 - crit["c"]["min_early_fraction"]) and delta["ci"][1] < 0:
            c["status"] = "PASS（以 Δ 判定）"; c["pass"] = True
        else: c["status"] = "FAIL 帶（看 m6）" if fail_band else "其他組合，人工判讀（負控制未過且 Δ 不滿足）"; c["pass"] = False if fail_band else None
    S["c_main"] = c
else:
    S["c_main"] = {"status": "未能測試（m6 無法計算或無效對過多）", "pass": None}
S["c_main"]["use"] = "descriptive_only"

# ---------------- 正控制（條文 4） ----------------
pcn = crit["positive_control"]
if all(need(k) for k in ("k6c0", "u833c0", "k1c0")):
    rc, ru, r5 = Run(args.k6c0), Run(args.u833c0), Run(args.k1c0)
    panels = dict(k6c0=rc, u833c0=ru, k1c0=r5)
    errors = {role: formal_panel_error(run, control_sizes, role) for role, run in panels.items()}
    if not any(errors.values()):
        if panel6_ok and not same_sources(r6, rc):
            errors["k6c0"] = "正控制與 K=6 未使用相同訓練 run"
        if panel1_ok and not same_sources(r1, r5):
            errors["k1c0"] = "正控制與 K=1 未使用相同訓練 run"
    if rc.picks != ru.picks or rc.picks != r5.picks or any(errors.values()):
        S["positive_control"] = {"status": "未能測試（正控制輸入或配對不符）", "input_errors": errors,
                                 "picks": [r.picks for r in (rc, ru, r5)]}
    else:
        grp = {"c": (rc, "cond"), "u833": (ru, "uncond"), "u5000": (r5, "cond")}
        inv = {}; valid = {}
        for g, (r, b) in grp.items():
            vs = [p for p in range(r.P) if r.inter(b, p)[0]]
            inv[g] = {"n_invalid": r.P - len(vs), "reasons": [{"pair": [int(i) for i in r.pairs[p]], "reason": r.inter(b, p)[1]} for p in range(r.P) if not r.inter(b, p)[0]]}
            valid[g] = vs
        P = rc.P; too_many = any(v["n_invalid"] > max_invalid(P) for v in inv.values())
        if too_many:
            S["positive_control"] = {"status": "未能測試（任一組無效對數 > floor(P/3)）", "invalid": inv}
        else:
            lt = {g: np.array([np.log(r.ts(b, p)) for p in range(P)]) for g, (r, b) in grp.items()}
            def stat(w):
                L = {g: wmean(lt[g], w, valid[g]) for g in lt}
                if any(v is None for v in L.values()) or L["u833"] == L["u5000"]:
                    return L, None
                return L, (L["c"] - L["u5000"]) / (L["u833"] - L["u5000"])
            L0, lam0 = stat(np.ones(P))
            rng = np.random.default_rng(crit["bootstrap_seed"]); lams = []; skp = 0
            for _ in range(NB):
                _, w = weights(rng, rc.R, rc.pairs)
                _, lm = stat(w)
                if lm is None: skp += 1; continue
                lams.append(lm)
            lci = [float(q) for q in np.percentile(lams, [2.5, 97.5])] if lams else [None, None]
            sep = L0["u5000"] - L0["u833"]
            disc = bool(sep >= pcn["discrimination_min"])
            d = {"L": L0, "lambda": lam0, "lambda_ci95": lci, "bootstrap_skipped": skp, "separation_Lu5000_minus_Lu833": sep,
                 "discriminating": disc, "abs_Lc_minus_Lu833": abs(L0["c"] - L0["u833"]),
                 "n_valid_pairs": {g: len(vs) for g, vs in valid.items()}, "invalid": inv}
            if not disc: d["status"] = "控制無分辨力（未能測試）"
            elif lci[0] is None: d["status"] = "未能測試（正控制 bootstrap 無有效重抽）"
            elif lci[0] >= pcn["lambda_ci_lower_min"] and d["abs_Lc_minus_Lu833"] <= pcn["abs_Lc_minus_Lu833_max"]:
                d["status"] = "通過"
            else: d["status"] = "n/K 時鐘前提不成立（本正控制未達門檻）"
            S["positive_control"] = d
else:
    S["positive_control"] = {"status": "未能測試（缺 --k6c0/--u833c0/--k1c0）"}

# ---------------- 條文 7：w→∞ 極限診斷 ----------------
limit_config = crit["asymptotic_onset"]
if r6 is not None and r6.gamma is not None and "cond" in r6.curve:
    S["asymptotic_onset"] = reach.h_inf(r6.ep, r6.curve["cond"], r6.gamma, r6.pairs, r6.R, tstar, interior_info,
                                       max_invalid(r6.P), limit_config["reference_log_ratio"], limit_config["n_boot"], limit_config["bootstrap_seed"])
else:
    S["asymptotic_onset"] = {"state": "極限曲線未能判讀", "note": "缺 --k6 或 pair_cos_gamma"}

def untestable(reason):
    return {"state": "未能測試", "reason": reason}


def phenotype():
    try:
        with open(args.week1_verdict) as fh:
            w1 = json.load(fh)
        w1_state = w1.get("overall")
        g2 = w1.get("gates", {}).get("G2_celeba_reproduction")
    except (OSError, ValueError, AttributeError):
        return untestable("第 1 週機械判讀缺失或無法讀取")
    details = {"week1_result": w1_state, "week1_gate2": g2, "week2_b_gate": S["b_gate"]}
    if g2 != "PASS":
        return {**details, **untestable("第 1 週閘門 2 未通過")}
    if S["b_gate"]["result"] != "PASS":
        return {**details, **untestable("第 2 週 (b) 閘門未通過")}
    return {**details, "state": "PASS"}


def oracle_checkpoint():
    if not panel6_ok:
        return None, f"K=6：{panel6_error}"
    if r6.ce is None:
        return None, "K=6 缺 ce"
    ce = r6.ce[:, :, :, -1, r6.br.index("cond")].mean(axis=(0, 2))
    if not len(ce) or not np.isfinite(ce).all():
        return None, "K=6 測試輸入 CE 曲線含非有限值或為空"
    ei = int(np.argmin(smooth(ce, K)))
    return (None, "e† 落在第一格") if ei == 0 else (ei, None)


def oracle_terms(run, epoch_idx, mask):
    return reach.gap_terms(run.mse, epoch_idx, mask, run.br.index("cond"), run.br.index("cfg3"))


def amplification(ei, checkpoint_error):
    if checkpoint_error or r6.mse is None:
        return untestable(checkpoint_error or "K=6 缺 mse_oracle")
    epoch = int(r6.ep[ei])
    g6, c6 = oracle_terms(r6, ei, r6.t_idx)
    a6, ac6 = oracle_terms(r6, ei, list(range(len(r6.tgrid))))
    gamma6, denom6 = reach.gamma_ratio(g6, c6)
    details = {"gamma6": gamma6, "denominators": {"k6": denom6}, "e_dagger": epoch,
               "e_dagger_is_last_grid": ei == len(r6.ep) - 1, "t_used": KEEP,
               "gamma6_all_t_descriptive": reach.gamma_ratio(a6, ac6)[0]}
    if not panel1_ok or r1.mse is None:
        return {**details, **untestable(f"K=1：{panel1_error or '缺 mse_oracle'}")}
    if epoch not in r1.ep:
        return {**details, **untestable("K=1 沒有 e† 對應的 checkpoint")}
    e1 = int(np.flatnonzero(r1.ep == epoch)[0])
    g1, c1 = oracle_terms(r1, e1, r1.t_idx)
    a1, ac1 = oracle_terms(r1, e1, list(range(len(r1.tgrid))))
    pb = crit["second_probe_gap_amplification"]
    result = reach.probe_amplification(g6, c6, r6.picks, g1, c1, r1.picks, pb["pass"]["gamma6_min"],
                                      pb["n_boot"], pb["bootstrap_seed"], pb["max_skipped_fraction"], screen["signal_ci_level"])
    details["gamma1_all_t_descriptive"] = reach.gamma_ratio(a1, ac1)[0]
    return {**details, **result}


def headroom(ei, checkpoint_error):
    if checkpoint_error or r6.mse is None:
        return {"measure": "CFG3 membership gap proxy", **untestable(checkpoint_error or "K=6 缺 mse_oracle")}
    own, test = reach.oracle_errors(r6.mse, ei, r6.t_idx)
    j = r6.br.index("cfg3")
    result = reach.membership_gap(own[:, :, j], test[:, :, j], screen["headroom_n_boot"], screen["headroom_seed"],
                                  screen["headroom_ci_level"], screen["headroom_min"], screen["headroom_upper_fail"])
    return {**result, "e_dagger": int(r6.ep[ei]), "t_used": KEEP}


def equivalent(ci):
    bound = screen["signal_equivalence_bound"]
    return ci[0] is not None and -bound < ci[0] and ci[1] < bound


def onset():
    details = {"ci_level": screen["signal_ci_level"]}
    if not panel6_ok or not m6["testable"]:
        return {**details, **untestable(f"K=6：{panel6_error or '主曲線有效配對不足'}")}
    m6_ci, skipped = boot_single(r6, m6, details["ci_level"])
    details.update(m6_ci=m6_ci, m6_bootstrap_skipped=skipped, m6_equivalent=equivalent(m6_ci))
    if m6_ci[0] is None:
        return {**details, **untestable("m6 bootstrap 無有效重抽")}
    if not panel1_ok or "passed" not in neg:
        return {**details, **untestable("K=1 負控制未能測試")}
    if neg["passed"]:
        mean = S["m6"]["mean"]
        ci = m6_ci
        details.update(metric="m6", onset_equivalent=details["m6_equivalent"])
    else:
        if "reason" in delta:
            return {**details, "delta": delta, **untestable(delta["reason"])}
        mean, ci = delta["mean"], delta["ci"]
        details.update(metric="matched_delta", delta=delta, onset_equivalent=equivalent(ci))
    passed = mean <= np.log(1 - crit["c"]["min_early_fraction"]) and ci[1] < 0
    return {**details, "mean": mean, "ci": ci, "state": "PASS" if passed else "未通過"}


def signal(sa, sb, nk_passed):
    details = {"S_a": sa, "S_b": sb, "n_over_k_control_passed": nk_passed}
    passed = [name for name in ("S_a", "S_b") if details[name]["state"] == "PASS"]
    if passed:
        return {**details, "state": "PASS", "passed_routes": passed}
    if nk_passed:
        return {**details, "state": "灰區", "reason": "只有 n/K 時鐘正控制通過"}
    if any(route["state"] == "未能測試" for route in (sa, sb)):
        return {**details, **untestable("兩條路徑沒有可確認的 PASS，且有路徑未能測試")}
    amp_weak = (sb["gamma6"] < screen["amplification_fail_below"]
                or sb["diff_ci"][1] <= screen["amplification_log_contrast_upper_fail"])
    weak = sa["m6_equivalent"] and sa["onset_equivalent"] and amp_weak
    return {**details, "state": "FAIL" if weak else "灰區"}


def candidate_status(phen, sig, gap):
    states = {axis: result["state"] for axis, result in (("phenomenon", phen), ("signal", sig), ("headroom", gap))}
    if phen["state"] != "PASS" or gap["state"] == "未能測試":
        return {**untestable("現象或空間無法判讀"), "axes": states}
    failed = [axis for axis, state in states.items() if state == "FAIL"]
    if failed:
        return {"state": "FAIL", "reason": "在本 PREREG 範圍內不進候選池", "failed_axes": failed, "axes": states}
    if set(states.values()) == {"PASS"}:
        return {"state": "PASS", "reason": "符合候選池條件", "axes": states}
    if "未能測試" in states.values():
        return {**untestable("有判準未能測試"), "axes": states}
    return {"state": "灰區", "reason": "交給使用者", "axes": states}


def cifar_action(candidate):
    out = {"phase": "入池後", "cap": crit["stop_points"]["gpu_hours_cap"]}
    if candidate["state"] != "PASS":
        return {**out, "action": "候選池判準未通過，暫不做 CIFAR (d)"}
    hours = (args.gpu_used, args.gpu_projected_d)
    if not all(h is not None and np.isfinite(h) and h >= 0 for h in hours):
        return {**out, "action": "使用者決定入池後，先確認已用與推估 GPU 小時"}
    total = sum(hours)
    action = "超過上限，先停下回報使用者" if total > out["cap"] else "未超過上限，使用者決定入池後可做 CIFAR (d)"
    return {**out, "gpu_used": hours[0], "gpu_projected_d": hours[1], "total": total, "action": action}


ei, checkpoint_error = oracle_checkpoint()
phen = phenotype()
sb = S["second_probe"] = amplification(ei, checkpoint_error)
nk_passed = S["positive_control"].get("status") == "通過"
sig = signal(onset(), sb, nk_passed)
gap = headroom(ei, checkpoint_error)
candidate = candidate_status(phen, sig, gap)
S["report60"] = {"phenomenon": phen, "signal": sig, "headroom": gap, "candidate_pool": candidate, "cifar_d": cifar_action(candidate)}

if S["c_main"]["status"].startswith("FAIL 帶"):
    S["tier"] = {"second_probe": sb["state"], "gate4_passed": nk_passed, "text": reach.tier_text(sb["state"]), "use": "descriptive_only"}

verdict["git_hash_k6"] = r6.git if r6 else None
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
json.dump(verdict, open(args.out, "w"), indent=1, ensure_ascii=False, default=lambda o: None if o is None else str(o))
print(json.dumps(verdict, indent=1, ensure_ascii=False, default=str))
