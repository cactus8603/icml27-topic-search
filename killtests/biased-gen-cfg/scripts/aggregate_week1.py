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
verdict = {"criteria_file": os.path.abspath(args.criteria), "gates": {}, "info": {}}


def smooth(y, k):
    """固定平滑：在 checkpoint 索引上做置中移動平均（視窗 k，邊緣縮小視窗）。"""
    y = np.asarray(y, float); h = k // 2
    return np.array([y[max(0, i - h): i + h + 1].mean() for i in range(len(y))])


def tstar(epochs, dist, k):
    return float(epochs[int(np.argmin(smooth(dist, k)))])


# ---------------- 閘門 1：BP 單元測試 ----------------
if os.path.exists(args.bp):
    bp = json.load(open(args.bp))
    verdict["gates"]["G1_bp_unit_tests"] = "PASS" if bp["all_pass"] else "FAIL"
else:
    verdict["gates"]["G1_bp_unit_tests"] = "未能測試（找不到 bp_unit_test.json）"

# ---------------- 閘門 2：(a) CelebA 重現 ----------------
cc = crit["celeba"]
if os.path.exists(args.celeba):
    d = np.load(args.celeba)
    ep, loss, sim = d["epochs"], d["test_loss"], d["sim_train"]  # sim[a,b,e,t]
    times = list(d["times"]); sel = [times.index(t) for t in cc["similarity_times"] if t in times]
    M = loss.shape[0]
    diffs, e_b, e_l = [], [], []
    for i in range(M):
        others = [j for j in range(M) if j != i]
        dist = 1 - sim[i][others][:, :, sel].mean(axis=(0, 2))       # 輸入來自 i 的訓練集，與其餘模型的 ε 預測距離
        eb = tstar(ep, dist, cc["smooth_window"]); el = tstar(ep, loss[i], cc["smooth_window"])
        e_b.append(eb); e_l.append(el); diffs.append(np.log(el / eb))
    diffs = np.array(diffs)
    mean, sem = diffs.mean(), diffs.std(ddof=1) / np.sqrt(M)
    ok = (M >= cc["min_models"]) and mean > cc["sem_multiple"] * sem and (mean > 0 or not cc["require_positive_gap"])
    verdict["info"]["celeba"] = {"bias_min_epoch": e_b, "test_loss_min_epoch": e_l, "log_ratio_mean": float(mean),
                                 "sem": float(sem), "threshold": float(cc["sem_multiple"] * sem),
                                 "n_models": M, "test_loss_min_at_grid_end": bool(any(e == ep[-1] for e in e_l))}
    verdict["gates"]["G2_celeba_reproduction"] = "PASS" if ok else "未能測試（重現未通過；記為可行性問題，非題目不成立）"
else:
    verdict["gates"]["G2_celeba_reproduction"] = "未能測試（找不到 celeba_eval.npz）"

# ---------------- (b) 階層資料 oracle 曲線與 t*(w)（第 1 週僅為管線檢查 + 探索性主指標） ----------------
ch = crit["hier"]
if os.path.exists(args.hier):
    h = np.load(args.hier, allow_pickle=True)
    br = list(h["branches"]); ep = h["epochs"]; R = h["mse_oracle"].shape[0]
    finite = all(np.isfinite(h[k]).all() for k in ("mse_oracle", "pair_mse", "pair_cos")) and np.isfinite(h["ce"][..., :2]).all() and np.isfinite(h["kl_oracle"][..., :2]).all()
    have = all(b in br for b in ch["required_branches"])
    ok_b = finite and have and R >= ch["min_runs"] and len(ep) >= 10
    verdict["gates"]["G3_hier_oracle_curves"] = "PASS" if ok_b else "FAIL（曲線含 NaN/缺分支/run 數不足）"
    pc, pairs = h["pair_cos"], h["pairs"]   # pc[pair,e,t,k(S_i,S_j,test),b]
    ib = {b: br.index(b) for b in br}
    k = ch["smooth_window"]
    def pair_t(b, p):
        return tstar(ep, pc[p][:, :, :2, ib[b]].mean(axis=(1, 2)), k)   # 兩份訓練輸入、全部 t 平均
    base, hi = ch["main_numerator_branch"], ch["main_denominator_branch"]
    lr_pair = np.array([np.log(pair_t(base, p) / pair_t(hi, p)) for p in range(len(pairs))])  # log[t*(w=3)/t*(w=0)]
    rng = np.random.default_rng(0); boots = []
    for _ in range(args.n_boot):
        mod = rng.integers(0, R, R)
        sel = [p for p, (i, j) in enumerate(pairs) if i in mod and j in mod]
        if sel:
            boots.append(lr_pair[sel].mean())
    lo_, hi_ = np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan)
    c_pass = bool(lr_pair.mean() <= np.log(1 - ch["min_early_fraction"]) and hi_ < 0)
    c_fail = bool(lo_ > -np.log(1 + ch["fail_band"]) and hi_ < np.log(1 + ch["fail_band"]))
    # 次要：正規化幅度 = test-loss 最低點 epoch 的距離 / 曲線最低距離
    ce_c = h["ce"][:, :, :, -1, 0].mean(axis=(0, 2))           # 共同測試集、條件分支、跨 run 與 t 平均
    e_loss = tstar(ep, ce_c, k); ie = int(np.where(ep == e_loss)[0][0])
    mag = {}
    for b in br:
        curve = smooth(pc[:, :, :, :2, ib[b]].mean(axis=(0, 2, 3)), k)
        mag[b] = float(curve[ie] / curve.min())
    verdict["info"]["hier"] = {
        "n_runs": R, "tstar_per_branch_mean_over_pairs": {b: float(np.mean([pair_t(b, p) for p in range(len(pairs))])) for b in br},
        "main_metric_log_tstar_ratio": {"mean": float(lr_pair.mean()), "ci95": [float(lo_), float(hi_)], "per_pair": lr_pair.tolist(),
                                       "numerator": base, "denominator": hi},
        "c_pass_criteria_met": c_pass, "c_fail_band_met": c_fail,
        "status_note": "第 1 週只有 3 個 run：此主指標僅為探索性，正式 (c) 需 6 份不相交資料（見 PREREG）" if R < ch["min_runs_main"] else "樣本數足夠，可作 (c) 正式判讀",
        "test_loss_min_epoch": e_loss, "normalized_magnitude_at_loss_min": mag,
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
