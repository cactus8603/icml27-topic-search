"""機械式判讀：依 PREREG.md 與 configs/criteria.json 輸出閘門狀態與 階段0 / 階段1 的 STOP/CONTINUE/PASS/GREY/FAIL。
用法：python aggregate.py --cfg configs/stage0.json --stage 0 [--M 200] [--smoke]
只做讀檔與數值計算（CPU）。結論由 Opus 對照 PREREG 審，不由本腳本或實作者定案。"""
import argparse, glob, json, os, sys
import numpy as np
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, os.path.dirname(__file__))
import hybrid as H
from common import cfg_paths, removal_mask_np, RES

ap = argparse.ArgumentParser()
ap.add_argument("--cfg", required=True)
ap.add_argument("--stage", type=int, default=0)
ap.add_argument("--M", type=int, default=None, help="使用的重訓模型數（灰區加碼後設 400）")
ap.add_argument("--smoke", action="store_true", help="smoke 模式：不強制閘門，只驗證流程")
a = ap.parse_args()
cfg = json.load(open(a.cfg)); CR = json.load(open(os.path.join(os.path.dirname(__file__), "..", "configs", "criteria.json")))
P = cfg_paths(cfg); M = a.M or cfg["M"]
k = cfg["k_stage0"] if a.stage == 0 else cfg["k_total"]
rng = np.random.RandomState(CR["bootstrap"]["seed"])
out = dict(stage=a.stage, k=k, M=M, tag=cfg["tag"])
# 審查加入 2026-11-14：記錄程式版本與判準檔雜湊（通用規則「git hash 全留」；確認判讀時的 criteria.json 即凍結版）
import hashlib, subprocess
_here = os.path.dirname(os.path.abspath(__file__))
try:
    out["git_head"] = subprocess.run(["git", "-C", _here, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() or None
    out["git_dirty"] = bool(subprocess.run(["git", "-C", _here, "status", "--porcelain", "--", ".."], capture_output=True, text=True).stdout.strip())
except Exception as e_:  # git 不可用不影響判讀
    out["git_head"] = f"unavailable: {e_}"
out["criteria_sha256"] = hashlib.sha256(open(os.path.join(_here, "..", "configs", "criteria.json"), "rb").read()).hexdigest()


def lds_q(S, masks, L):
    """逐查詢 LDS：Spearman(預測 Δloss = -masks@S, 實測重訓查詢損失)。S: n×k，masks: M×n，L: M×k。"""
    pred = -(masks.astype(np.float32) @ S)
    rp, rl = rankdata(pred, axis=0), rankdata(L, axis=0)
    rp -= rp.mean(0); rl -= rl.mean(0)
    return (rp * rl).sum(0) / (np.sqrt((rp ** 2).sum(0) * (rl ** 2).sum(0)) + 1e-30)


def boot_ci(x, n=CR["bootstrap"]["n"]):
    idx = rng.randint(0, len(x), (n, len(x)))
    m = np.nanmean(x[idx], 1)
    lo = (1 - CR["bootstrap"]["ci"]) / 2
    return float(np.nanquantile(m, lo)), float(np.nanquantile(m, 1 - lo))


def paired(dx):
    return dict(mean=float(np.nanmean(dx)), ci=boot_ci(dx))


# ------------------------------------------------------------ 讀資料
def load_retrain(dirname, Mu):
    Ls, js = [], []
    for j in range(Mu):
        f = f"{dirname}/loss_{j}.npy"
        if os.path.exists(f):
            Ls.append(np.load(f)); js.append(j)
    return js, (np.stack(Ls) if Ls else None)


rd = os.path.join(os.path.dirname(P["retrain"]), "retrain")
js, Lall = load_retrain(rd, M)
if Lall is None or len(js) < M:
    print(f"重訓不完整：{0 if Lall is None else len(js)}/{M}"); 
    if not a.smoke: sys.exit(2)
if Lall is None: sys.exit(2)
L, acc = Lall[:, :k], Lall[:, -1]
os.makedirs(P["res"], exist_ok=True)
np.save(f"{P['res']}/retrain_query_losses_k{k}_M{M}.npy", np.concatenate([L, acc[:, None]], 1))  # 原始數字（小檔）留存
n = cfg["n_train"] or 50000
masks = np.stack([removal_mask_np(cfg, j, n) for j in js])
noise = [np.load(f)[:cfg["k_total"]][:k] for f in sorted(glob.glob(f"{rd}/noise_*_task*.npy"))]
# 精確 Y：只接受由 0 起、首尾相接、互不重疊的 Y_{s}_{e}.npy 區塊（審查修正 2026-11-14：
# 舊版以 glob 全收，若目錄裡多了 Y_0_2.npy 之類的抽查檔，欄位會被靜默錯接）
chunks = {}
for f in glob.glob(f"{P['exact']}/Y_*_*.npy"):
    s_, e_ = map(int, os.path.basename(f)[2:-4].split("_"))
    chunks.setdefault(s_, []).append((e_, f))
cols, pos = [], 0
while pos < k:
    if pos not in chunks or len(chunks[pos]) != 1:
        print(f"精確 Y 不完整或區塊重疊（位置 {pos}：{chunks.get(pos)}）"); sys.exit(2)
    e_, f = chunks[pos][0]
    Yc_ = np.load(f)
    assert Yc_.shape[1] == e_ - pos, f"{f} 欄數 {Yc_.shape[1]} 與檔名不符"
    cols.append(Yc_); pos = e_
Y = np.concatenate(cols, 1)[:, :k]
cheap, missing = {}, []
for nm in CR["cheap_variants"]:
    f = f"{P['cheap']}/{nm}.npy"
    if os.path.exists(f):
        cheap[nm] = np.load(f)[:, :k]
    else:
        missing.append(nm)
if missing:  # PREREG 固定 7 個變體；缺任何一個都不判（避免在子集合上選「最佳」）
    print("便宜方法不完整：", missing)
    if not a.smoke: sys.exit(2)
GV = np.load(f"{P['cheap']}/gram_query_grads.npy")[:k, :k]

# ------------------------------------------------------------ 閘門
G = {}
gd = f"{RES}/gates"
f = f"{gd}/hybrid_identities.json"
G["hybrid_identities"] = bool(json.load(open(f))["pass"]) if os.path.exists(f) else None
# 數值閘門依 criteria.json 重新判定（不信任子腳本寫的 pass 旗標；審查修正 2026-11-14）
f = f"{gd}/mlp_magic.json"
if os.path.exists(f):
    gm = json.load(open(f)); cm = CR["gates"]["mlp"]
    G["mlp_magic"] = bool(np.isfinite(gm["fd_rel_err_max"]) and gm["fd_rel_err_max"] < cm["fd_rel_err_max"]
                          and np.isfinite(gm["loo_pearson_mean"]) and gm["loo_pearson_mean"] > cm["loo_pearson_min"])
else:
    G["mlp_magic"] = None
f = f"{P['res']}/export_fidelity.json"
if os.path.exists(f):
    d_ = json.load(open(f))["max_abs_logit_diff"]
    G["export_fidelity"] = bool(np.isfinite(d_) and d_ < CR["gates"]["export_fidelity_max_abs_logit_diff"])
else:
    G["export_fidelity"] = None
lds_magic_q = lds_q(Y, masks, L)
G["cifar_magic_lds"] = float(np.nanmean(lds_magic_q))
G["cifar_magic_repro"] = bool(G["cifar_magic_lds"] >= CR["gates"]["cifar_magic_lds_min"])
if noise:
    ns = np.std(np.stack(noise), 0, ddof=1) if len(noise) > 1 else np.full(k, np.nan)
    sig = L.std(0, ddof=1)
    G["noise_std_over_signal_std"] = float(np.nanmean(ns / sig))
    G["noise_ok"] = bool(G["noise_std_over_signal_std"] <= CR["gates"]["noise_floor"]["max_noise_std_vs_signal_std"])
else:
    G["noise_ok"] = None
out["gates"] = G
gate_keys = ["hybrid_identities", "mlp_magic", "export_fidelity", "cifar_magic_repro", "noise_ok"]
bad = [g for g in gate_keys if G.get(g) is not True]
out["gate_status"] = "ALL_PASS" if not bad else "UNTESTABLE(未能測試): " + ",".join(bad)
print(json.dumps(out["gates"], indent=1)); print("閘門：", out["gate_status"])
if bad and not a.smoke:
    out["verdict"] = "UNTESTABLE"
    json.dump(out, open(f"{P['res']}/verdict_stage{a.stage}.json", "w"), indent=1); print("結論：未能測試（前三關失敗不判題目）"); sys.exit(3)

# ------------------------------------------------------------ 便宜方法 LDS
cl = {nm: lds_q(S, masks, L) for nm, S in cheap.items()}
out["cheap_lds"] = {nm: float(np.nanmean(v)) for nm, v in cl.items()}
out["magic_lds"] = G["cifar_magic_lds"]


def invalid(why):
    """程式錯誤（NaN/inf 等）：記為 INVALID，與「題目不成立」分開，不給 STOP/CONTINUE/PASS/FAIL。"""
    out["verdict"] = f"INVALID(程式錯誤，非題目結論): {why}"
    json.dump(out, open(f"{P['res']}/verdict_stage{a.stage}.json", "w"), indent=1, default=float)
    print("結論：", out["verdict"]); sys.exit(4)


nonfinite = [nm for nm, S in cheap.items() if not np.all(np.isfinite(S))] + \
            [nm for nm, v in out["cheap_lds"].items() if not np.isfinite(v)]
if not np.all(np.isfinite(Y)) or not np.all(np.isfinite(L)) or not np.all(np.isfinite(GV)):
    nonfinite.append("Y/L/G_V")
if nonfinite:
    invalid("非有限值：" + ",".join(sorted(set(nonfinite))))
best = max(out["cheap_lds"], key=out["cheap_lds"].get)
if a.stage == 1:  # 階段 1 沿用階段 0 事先選定的最佳便宜方法，不重選
    v0 = json.load(open(f"{P['res']}/verdict_stage0.json"))
    if not str(v0.get("verdict", "")).startswith("CONTINUE") and not a.smoke:
        print("階段 0 結論不是 CONTINUE，不得進行階段 1 判讀：", v0.get("verdict")); sys.exit(2)
    best = v0["best_cheap"]
out["best_cheap"] = best; C = cheap[best]
B = max(1, int(round(CR["B_frac"] * k))); out["B"] = B
Ug = H.select_U(GV, B, "gram"); Ur = H.select_U(GV, B, "random", seed=0)
hyb, alpha = H.hybrid(Y, C, Ug); sp = H.spell(Y, Ug)
out["alpha"] = alpha; out["kappa"] = H.kappa(Y, Ug); out["kappa_res"] = H.kappa_res(Y, C, Ug, alpha)
out["kappa_res_over_kappa"] = out["kappa_res"] / out["kappa"]
out["kappa_random_U"] = H.kappa(Y, Ur)
hl, sl = lds_q(hyb, masks, L), lds_q(sp, masks, L)
out["hybrid_lds"], out["spell_lds"] = float(np.nanmean(hl)), float(np.nanmean(sl))
hr, _ = H.hybrid(Y, C, Ur); out["hybrid_randomU_lds"] = float(np.nanmean(lds_q(hr, masks, L)))
out["spell_randomU_lds"] = float(np.nanmean(lds_q(H.spell(Y, Ur), masks, L)))
out["gain"] = out["hybrid_lds"] - max(out["spell_lds"], out["cheap_lds"][best])
out["per_method_kappa_res_over_kappa"] = {}
for nm, S in cheap.items():
    _, al = H.hybrid(Y, S, Ug); out["per_method_kappa_res_over_kappa"][nm] = H.kappa_res(Y, S, Ug, al) / out["kappa"]
dec = [out["alpha"], out["kappa"], out["kappa_res"], out["kappa_res_over_kappa"], out["hybrid_lds"], out["spell_lds"], out["gain"]]
if not all(np.isfinite(dec)):
    invalid("判讀量含 NaN/inf（α、κ、κ_res、LDS 或 gain）")

# ------------------------------------------------------------ 階段 0 判讀
S0 = CR["stage0"]
if a.stage == 0:
    # 僅描述、不參與判讀（審查加入 2026-11-14）：k=50 下增益的配對 bootstrap CI 與最佳便宜方法 LDS 的折半信度
    out["descriptive_stage0"] = dict(
        hybrid_minus_spell=paired(hl - sl), hybrid_minus_cheap=paired(hl - cl[best]),
        split_half_r_best_cheap=float(spearmanr(lds_q(C, masks[0::2], L[0::2]), lds_q(C, masks[1::2], L[1::2]), nan_policy="omit")[0]),
        split_half_r_magic=float(spearmanr(lds_q(Y, masks[0::2], L[0::2]), lds_q(Y, masks[1::2], L[1::2]), nan_policy="omit")[0]))
    if out["cheap_lds"][best] >= S0["best_cheap_lds_stop_at_or_above"]:
        v = "STOP(最佳便宜方法 LDS≥0.8)"
    elif out["kappa_res_over_kappa"] < S0["kappa_res_over_kappa_stop_below"]:
        v = "STOP(κ_res/κ<0.5)"
    elif out["gain"] < S0["hybrid_gain_stop_below"]:
        v = "STOP(混合增益<0.04)"
    else:
        v = "CONTINUE(進階段 1：k=200）"
    out["verdict"] = v
else:
    S1 = CR["stage1"]
    dh_s, dh_c = hl - sl, hl - cl[best]
    ps, pc = paired(dh_s), paired(dh_c)
    out["hybrid_minus_spell"], out["hybrid_minus_cheap"] = ps, pc
    g = min(ps["mean"], pc["mean"]); lo = min(ps["ci"][0], pc["ci"][0]); out["gain_min"], out["ci_lower_min"] = g, lo
    # 跨種子（hybrid 與最佳便宜方法，在 train_seed=1 的重訓上）
    cs_dir = os.path.join(os.path.dirname(P["retrain"]), "retrain_cs")
    jcs, Lcs = load_retrain(cs_dir, 10 ** 6)
    if Lcs is None:
        out["crossseed"] = None
    else:
        mcs = np.stack([removal_mask_np(cfg, j, n) for j in jcs])
        d = float(np.nanmean(lds_q(hyb, mcs, Lcs[:, :k])) - np.nanmean(lds_q(C, mcs, Lcs[:, :k])))
        out["crossseed"] = dict(n=len(jcs), hybrid_minus_cheap=d, ok=bool(d >= S1["crossseed_hybrid_minus_cheap_min"]))
    # 可靠度
    cq = cl[best]
    half = [lds_q(C, masks[i::2], L[i::2]) for i in (0, 1)]
    r = float(spearmanr(half[0], half[1], nan_policy="omit")[0]); sb = 2 * r / (1 + r)
    rho = H.reliability(Y, C, Ug)
    qloss = np.load(f"{P['models']}/model_s{cfg['train_seed']}.npz")["query_loss"][:k]
    sp_rho = float(spearmanr(rho, cq, nan_policy="omit")[0])
    sp_triv = {"query_loss": float(spearmanr(qloss, cq, nan_policy="omit")[0]), "norm_C": float(spearmanr(np.linalg.norm(C, axis=0), cq, nan_policy="omit")[0])}
    R = S1["reliability"]
    out["reliability"] = dict(split_half_SB=sb, spearman_rho_hat=sp_rho, trivial=sp_triv,
                              pass_=bool(sb >= R["split_half_min"] and sp_rho >= R["spearman_min"] and sp_rho > max(abs(v) for v in sp_triv.values())),
                              grey=bool(sb >= R["split_half_min"] and R["spearman_grey_low"] <= sp_rho < R["spearman_min"]))
    if not all(np.isfinite([g, lo, sb, sp_rho] + list(sp_triv.values()))) or \
            (out["crossseed"] is not None and not np.isfinite(out["crossseed"]["hybrid_minus_cheap"])):
        invalid("階段 1 判讀量含 NaN/inf（增益、CI、折半信度、可靠度 Spearman 或跨種子差）")
    cs_ok = out["crossseed"] is not None and out["crossseed"]["ok"]
    cs_missing = "未完成(缺跨種子重訓)"
    if g >= S1["gain_pass"] and lo > S1["ci_lower_gt"]:
        v = "PASS" if cs_ok else ("FAIL(跨種子輸給便宜方法)" if out["crossseed"] else cs_missing)
    elif g >= S1["gain_pass"]:
        v = "GREY(增益≥0.08 但 CI 下界≤0.03)"
    elif g >= S1["gain_grey_low"]:
        if out["reliability"]["pass_"]:
            v = "PASS(僅可靠度路線)" if cs_ok else ("FAIL(跨種子)" if out["crossseed"] else cs_missing)
        else:
            v = "GREY(可靠度灰區)" if out["reliability"]["grey"] else "FAIL"
    else:
        v = "FAIL"
    out["verdict"] = v
json.dump(out, open(f"{P['res']}/verdict_stage{a.stage}.json", "w"), indent=1, default=float)
print(json.dumps({k_: v_ for k_, v_ in out.items() if k_ != "gates"}, indent=1, default=float))
print("\n結論：", out["verdict"], "\n（機械輸出；最終判讀由 Opus 對照 PREREG.md 審）")
