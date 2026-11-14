#!/usr/bin/env python
"""機械判讀第 1 週結果（判準全部來自 configs/criteria_week1.json，與 PREREG.md 一致）。
用法：python aggregate_week1.py [--celeba results/celeba_eval.npz] [--hier results/hier_oracle_w1.npz]
                                [--bp results/bp_unit_test.json] [--out results/week1_verdict.json]
輸出三態：PASS / FAIL / 未能測試（閘門失敗；不等於題目不成立）。
"""
import argparse, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
ap = argparse.ArgumentParser()
ap.add_argument("--criteria", default=os.path.join(ROOT, "configs", "criteria_week1.json"))
ap.add_argument("--celeba", default=os.path.join(ROOT, "results", "celeba_eval.npz"))
ap.add_argument("--hier", default=os.path.join(ROOT, "results", "hier_oracle_w1.npz"))
ap.add_argument("--bp", default=os.path.join(ROOT, "results", "bp_unit_test.json"))
ap.add_argument("--out", default=os.path.join(ROOT, "results", "week1_verdict.json"))
ap.add_argument("--n-boot", type=int, default=2000)
args = ap.parse_args()
crit = json.load(open(args.criteria))
# 條文 1.5：x0-hat 空間 pair_cos 留在 npz，但本檔不得用它算任何 t* 或判定（只使用 pair_cos_eps）。
verdict = {"criteria_file": os.path.abspath(args.criteria), "gates": {}, "info": {}}


def smooth(y, k):
    """固定平滑：在 checkpoint 索引上做置中移動平均（視窗 k，邊緣縮小視窗）。"""
    y = np.asarray(y, float); h = k // 2
    return np.array([y[max(0, i - h): i + h + 1].mean() for i in range(len(y))])


def tstar(epochs, dist, k):
    """平滑後最低點 epoch。曲線含 NaN/inf 時回傳 nan（np.argmin 會把 NaN 當最小值，不能直接用）。"""
    y = smooth(dist, k)
    if len(y) == 0 or not np.all(np.isfinite(y)):
        return float("nan")
    return float(epochs[int(np.argmin(y))])


def interior_info(epochs, dist, k, rise):
    """條文 2 的「內部最低點」：平滑後 argmin 不在第一/最後一格，且 argmin 之後的最大值 >= rise × 最低值。
    回傳 (ok, reason)；reason ∈ {None, 'nonfinite', 'first', 'last', 'insufficient_rise'}。"""
    y = smooth(dist, k)
    if len(y) == 0 or not np.all(np.isfinite(y)):
        return False, "nonfinite"
    i = int(np.argmin(y))
    if i == 0:
        return False, "first"
    if i == len(y) - 1:
        return False, "last"
    if y[i:].max() < rise * y[i]:
        return False, "insufficient_rise"
    return True, None


def at_edge(epochs, e):
    """t* 落在 checkpoint 網格第一或最後一格（PREREG 閘門 3：要標記）。"""
    return "nan" if not np.isfinite(e) else ("first" if e == epochs[0] else ("last" if e == epochs[-1] else None))


# ---------------- 閘門 1：BP 單元測試 ----------------
if os.path.exists(args.bp):
    bp = json.load(open(args.bp))
    verdict["gates"]["G1_bp_unit_tests"] = "PASS" if bp["all_pass"] else "FAIL"
else:
    verdict["gates"]["G1_bp_unit_tests"] = "未能測試（找不到 bp_unit_test.json）"

# ---------------- 閘門 2：(a) CelebA 重現（條文 2、5） ----------------
cc = crit["celeba"]
RISE = crit["interior_min_rise"]


def celeba_models(ep, loss, sim, sel, ks):
    """對給定的模型子集 ks（用其餘 ks 計 e_bias）回傳每個模型的 dict。"""
    out = {}
    for i in ks:
        others = [j for j in ks if j != i]
        if not others:
            continue
        dist = 1 - sim[i][others][:, :, sel].mean(axis=(0, 2))
        eb = tstar(ep, dist, cc["smooth_window"]); el = tstar(ep, loss[i], cc["smooth_window"])
        ok_in, why = interior_info(ep, dist, cc["smooth_window"], RISE)
        loss_edge = at_edge(ep, el)
        valid = bool(ok_in and np.isfinite(el) and loss_edge != "first"
                     and (loss_edge != "last" or cc["loss_last_edge_is_right_censored_valid"]))
        out[i] = dict(eb=eb, el=el, bias_internal=ok_in, bias_reason=why, loss_edge=loss_edge, valid=valid,
                      diff=float(np.log(el / eb)) if (valid and np.isfinite(eb) and eb > 0) else float("nan"))
    return out


if os.path.exists(args.celeba):
    d = np.load(args.celeba)
    ep, loss, sim = d["epochs"], d["test_loss"], d["sim_train"]  # sim[a,b,e,t]
    times = list(d["times"]); sel = [times.index(t) for t in cc["similarity_times"] if t in times]
    missing_t = [t for t in cc["similarity_times"] if t not in times]
    M = loss.shape[0]
    mods = celeba_models(ep, loss, sim, sel, list(range(M)))
    valid_ids = [i for i in range(M) if mods[i]["valid"] and np.isfinite(mods[i]["diff"])]
    diffs = np.array([mods[i]["diff"] for i in valid_ids])
    n_valid, n_invalid = len(valid_ids), M - len(valid_ids)
    finite_ok = (not missing_t) and bool(np.all(np.isfinite(loss))) and bool(np.all(np.isfinite(sim[:, :, :, sel])))
    mean = float(diffs.mean()) if n_valid else float("nan")
    sem = float(diffs.std(ddof=1) / np.sqrt(n_valid)) if n_valid > 1 else float("nan")
    testable = finite_ok and M >= cc["min_models"] and n_valid >= cc["min_valid_models"] and n_invalid <= cc["max_invalid_models"]
    ok = bool(testable and mean > cc["sem_multiple"] * sem and (mean > 0 or not cc["require_positive_gap"]))
    # 建議 1：模型層級 bootstrap（只描述，不進判定）：重抽 M 個模型；對抽到的相異模型只用同次抽到的其他相異模型重算 e_bias，依抽中次數加權
    rng = np.random.default_rng(cc["model_bootstrap_descriptive"]["seed"]); bs, skipped = [], 0
    for _ in range(cc["model_bootstrap_descriptive"]["n_boot"]):
        draw = rng.integers(0, M, M); cnt = np.bincount(draw, minlength=M); ks = [i for i in range(M) if cnt[i] > 0]
        if len(ks) < 2:
            skipped += 1; continue
        r = celeba_models(ep, loss, sim, sel, ks)
        w = [(cnt[i], r[i]["diff"]) for i in ks if r[i]["valid"] and np.isfinite(r[i]["diff"])]
        if not w:
            skipped += 1; continue
        bs.append(sum(c_ * x for c_, x in w) / sum(c_ for c_, _ in w))
    bci = [float(x) for x in np.percentile(bs, [2.5, 97.5])] if bs else [float("nan")] * 2
    verdict["info"]["celeba"] = {
        "bias_min_epoch": [mods[i]["eb"] for i in range(M)], "test_loss_min_epoch": [mods[i]["el"] for i in range(M)],
        "model_valid": [mods[i]["valid"] for i in range(M)],
        "bias_internal_min": [mods[i]["bias_internal"] for i in range(M)], "bias_invalid_reason": [mods[i]["bias_reason"] for i in range(M)],
        "test_loss_min_edge": [mods[i]["loss_edge"] for i in range(M)],
        "right_censored_models": [i for i in range(M) if mods[i]["loss_edge"] == "last"],
        "log_ratio_per_valid_model": diffs.tolist(), "log_ratio_mean": mean, "sem": sem,
        "threshold": float(cc["sem_multiple"] * sem), "n_models": M, "n_valid_models": n_valid, "n_invalid_models": n_invalid,
        "test_loss_min_at_grid_end": bool(any(mods[i]["loss_edge"] == "last" for i in range(M))),
        "log_ratio_is_lower_bound_if_censored": True,
        "curves_finite": finite_ok, "missing_similarity_times": missing_t,
        "model_bootstrap_descriptive": {"mean_ci95": bci, "n_used": len(bs), "n_skipped": skipped, "used_in_verdict": False},
    }
    if not testable:
        verdict["gates"]["G2_celeba_reproduction"] = ("未能測試（(a) 有效模型不足或曲線無內部最低點／設限；"
                                                     "依 PREREG §4 診斷順序處理，非題目不成立）")
    else:
        verdict["gates"]["G2_celeba_reproduction"] = "PASS" if ok else "未能測試（重現未通過；記為可行性問題，非題目不成立；依 PREREG §4 診斷順序）"
else:
    verdict["gates"]["G2_celeba_reproduction"] = "未能測試（找不到 celeba_eval.npz）"

# ---------------- (b)(c) 階層資料（條文 1、2、5）。第 1 週僅為管線檢查 + 探索性主指標 ----------------
ch = crit["hier"]


def curve_pairs(pce, ep, mask_idx):
    """pce[pair,e,t,k(S_i,S_j,test),b] -> 逐 t 正規化並遮罩等權平均後的主曲線 dict[b] -> (pairs, E)；
    某 (對,分支,t) 的跨 ckpt 平均為 0 或非有限 -> 該對該分支整條記 nan。"""
    out = {}
    for b in range(pce.shape[-1]):
        y = pce[:, :, :, :2, b].mean(axis=3)[:, :, mask_idx]               # (pairs, E, T_keep)：兩份訓練輸入平均
        m = y.mean(axis=1, keepdims=True)                                    # 跨 checkpoint 平均
        bad = ~np.isfinite(m) | (m == 0)
        with np.errstate(all="ignore"):
            yn = y / np.where(bad, np.nan, m)
        out[b] = yn.mean(axis=2)                                             # (pairs, E)
    return out


if os.path.exists(args.hier):
    h = np.load(args.hier, allow_pickle=True)
    br = list(h["branches"]); ep = h["epochs"]; R = h["mse_oracle"].shape[0]
    assert "pair_cos_eps" in h.files, "npz 缺 pair_cos_eps（需用條文 1.1 的 eval_hier_oracle.py 重算）"
    finite = all(np.isfinite(h[k]).all() for k in ("mse_oracle", "pair_mse", "pair_cos_eps")) and np.isfinite(h["ce"][..., :2]).all() and np.isfinite(h["kl_oracle"][..., :2]).all()
    have = all(b in br for b in ch["required_branches"])
    ok_b = finite and have and R >= ch["min_runs"] and len(ep) >= 10
    verdict["gates"]["G3_hier_oracle_curves"] = "PASS" if ok_b else "FAIL（曲線含 NaN/缺分支/run 數不足）"
    pairs = h["pairs"]; P = len(pairs)
    mpath = crit["t_mask_file"] if os.path.isabs(crit["t_mask_file"]) else os.path.join(ROOT, crit["t_mask_file"])
    tm = json.load(open(mpath)); tg = [int(x) for x in h["t_grid"]]
    keep_t = [int(x) for x in tm["keep_t"]]
    assert all(t in tg for t in keep_t), f"t 遮罩 {keep_t} 不在 npz 的 t 網格 {tg}"
    mask_idx = [tg.index(t) for t in keep_t]
    ib = {b: br.index(b) for b in br}
    k = ch["smooth_window"]
    curves = curve_pairs(h["pair_cos_eps"], ep, mask_idx)                   # curves[b][p] = 主曲線
    def tstar_pb(b, p):
        return tstar(ep, curves[ib[b]][p], k)
    def interior_pb(b, p):
        return interior_info(ep, curves[ib[b]][p], k, RISE)
    base, hi = ch["main_numerator_branch"], ch["main_denominator_branch"]

    # ---- (b) 閘門：cond 主曲線有內部最低點的對數 ----
    cond_int = [interior_pb(hi, p) for p in range(P)]
    n_cond_int = sum(1 for o, _ in cond_int if o)
    need_b = ch["b_gate_min_cond_interior"].get(str(P), int(np.ceil(2 * P / 3)))
    b_gate = "PASS" if n_cond_int >= need_b else "未能測試（20000 epochs 內條件分支未進入偏差期）"
    verdict["gates"]["G3b_hier_cond_interior_min"] = b_gate

    # ---- (c) 主指標：有效對 = cond 與 cfg3 都有內部最低點 ----
    inv_info, valid_p = [], []
    for p in range(P):
        rs = {b: interior_pb(b, p) for b in (base, hi)}
        v = all(o for o, _ in rs.values())
        if v: valid_p.append(p)
        inv_info.append({"pair": [int(x) for x in pairs[p]], "valid": v,
                         "reason": {b: r for b, (o, r) in rs.items() if not o}})
    n_invalid = P - len(valid_p)
    max_inv = P // ch["max_invalid_pair_divisor"]
    c_testable = n_invalid <= max_inv and len(valid_p) > 0
    lr_all = np.array([np.log(tstar_pb(base, p) / tstar_pb(hi, p)) if p in valid_p else np.nan for p in range(P)])
    lo_ = hi_ = np.nan; skipped = 0; c_pass = c_fail = None; mean_c = np.nan
    if c_testable:
        mean_c = float(lr_all[valid_p].mean())
        rng = np.random.default_rng(0); boots = []
        pi_, pj_ = pairs[:, 0], pairs[:, 1]
        for _ in range(args.n_boot):
            m = np.bincount(rng.integers(0, R, R), minlength=R).astype(float)
            wgt = m[pi_] * m[pj_]
            den = wgt[valid_p].sum()
            if den == 0:
                skipped += 1; continue
            boots.append(float((wgt[valid_p] * lr_all[valid_p]).sum() / den))
        lo_, hi_ = np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan)
        c_pass = bool(mean_c <= np.log(1 - ch["min_early_fraction"]) and hi_ < 0)
        c_fail = bool(lo_ > -np.log(1 + ch["fail_band"]) and hi_ < np.log(1 + ch["fail_band"]))
    # 次要：正規化幅度 = test-loss 最低點 epoch 的主曲線值 / 曲線最低值（對有效對平均後的主曲線；只描述）
    ce_c = h["ce"][:, :, :, -1, 0].mean(axis=(0, 2))
    e_loss = tstar(ep, ce_c, k)
    mag = {}
    for b in br:
        mc = curves[ib[b]]; cur = smooth(np.nanmean(mc, axis=0), k)
        mag[b] = float(cur[int(np.where(ep == e_loss)[0][0])] / cur.min()) if (np.isfinite(e_loss) and np.all(np.isfinite(cur))) else float("nan")
    tstar_edges = {b: [at_edge(ep, tstar_pb(b, p)) for p in range(P)] for b in br}
    verdict["info"]["hier"] = {
        "n_runs": R, "n_pairs": P, "t_mask_keep": keep_t, "t_mask_g_t": tm["g_t"], "distance_space": crit["distance_space"],
        "b_gate": {"cond_interior_pairs": n_cond_int, "required": need_b,
                   "per_pair": [{"pair": [int(x) for x in pairs[p]], "interior": o, "reason": r} for p, (o, r) in enumerate(cond_int)]},
        "tstar_per_branch_mean_over_pairs": {b: float(np.mean([tstar_pb(b, p) for p in range(P)])) for b in br},
        "main_metric_log_tstar_ratio": {"mean": mean_c, "ci95": [float(lo_), float(hi_)], "per_pair": [None if not np.isfinite(x) else float(x) for x in lr_all],
                                       "numerator": base, "denominator": hi, "n_valid_pairs": len(valid_p), "n_invalid_pairs": n_invalid,
                                       "max_invalid_allowed": max_inv, "bootstrap_skipped": skipped, "bootstrap_weight": crit["bootstrap_weight"]},
        "invalid_pairs": inv_info,
        "c_status": "可判讀" if c_testable else "未能測試（無偏差期或設限；不得判 PASS 或 FAIL）",
        "c_pass_criteria_met": c_pass, "c_fail_band_met": c_fail,
        "status_note": "第 1 週只有 3 個 run：此主指標僅為探索性，不得用來調整第 2 週任何設定（條文 6）" if R < ch["min_runs_main"] else "樣本數足夠，可作 (c) 正式判讀",
        "test_loss_min_epoch": e_loss, "test_loss_min_edge": at_edge(ep, e_loss), "normalized_magnitude_at_loss_min": mag,
        "tstar_edge_per_pair": tstar_edges,
        "main_metric_censored": bool(n_invalid > 0),
        "git_hash_of_eval": str(h["git_hash"]) if "git_hash" in h.files else None,
    }
else:
    verdict["gates"]["G3_hier_oracle_curves"] = "未能測試（找不到 hier_oracle npz）"

g = verdict["gates"]
if all(v == "PASS" for v in g.values()):
    overall = "WEEK1: PASS（管線與重現閘門通過；主假設檢定留到第 2 週 (c)(d)）"
elif any(str(v).startswith("FAIL") for v in g.values()):
    overall = "WEEK1: FAIL（程式／管線問題，先修；3 個工作天內修不好則記可行性停損）"
else:
    overall = "WEEK1: 未能測試"
verdict["overall"] = overall
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
json.dump(verdict, open(args.out, "w"), indent=1, ensure_ascii=False)
print(json.dumps(verdict, indent=1, ensure_ascii=False))
print("\n" + overall)
