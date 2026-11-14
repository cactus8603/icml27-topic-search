"""依 PREREG.md 與 criteria.json 判讀階段 0 的現象、訊號、空間。
用法：python aggregate.py --cfg configs/stage0.json [--crossseed] [--smoke]
階段 1 與舊版混合比較僅供描述。"""
import argparse, glob, json, os, sys
import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(__file__))
import hybrid as H
from common import cfg_paths, removal_mask_np, RES
from stage0_criteria import column_cosine, descriptive_spearman, grade_crossseed, issue_stage0_decision, lds_q, stable_mean

ap = argparse.ArgumentParser()
ap.add_argument("--cfg", required=True)
ap.add_argument("--stage", type=int, choices=(0, 1), default=0)
ap.add_argument("--addon", action="store_true", help="在已有 k=400 資料時產生描述性輸出；僅 --stage 1")
ap.add_argument("--crossseed", action="store_true", help="三項全通過後，使用同一批查詢與 seed 5 的 100 次重訓補算 H_x")
ap.add_argument("--smoke", action="store_true", help="smoke 模式：不強制閘門，只驗證流程")
a = ap.parse_args()
cfg = json.load(open(a.cfg)); CR = json.load(open(os.path.join(os.path.dirname(__file__), "..", "configs", "criteria.json")))
P = cfg_paths(cfg); M = cfg["M"]
QC = CR["queries"]  # 查詢數事先固定於 criteria.json；正式 cfg（tag=stage0）必須一致，其他 cfg（smoke、假輸入測試）不檢查
if cfg["tag"] == "stage0":
    assert (cfg["k_stage0"], cfg["k_stage1"], cfg["k_addon"], cfg["k_total"], cfg["query_seed"]) == \
        (QC["k_stage0"], QC["k_stage1"], QC["k_addon"], QC["k_addon"], QC["query_seed"]), "cfg 的查詢設定與 criteria.json 不一致"
if a.addon and a.stage != 1:
    sys.exit("--addon 只能與 --stage 1 並用")
if a.crossseed and a.stage != 0:
    sys.exit("--crossseed 只能與 --stage 0 並用")
if cfg["tag"] == "stage0" and cfg["crossseed_train_seed"] != CR["stage0"]["crossseed"]["train_seed"]:
    sys.exit("cfg 的跨種子設定與 criteria.json 不一致")
k = cfg["k_stage0"] if a.stage == 0 else (cfg["k_addon"] if a.addon else cfg["k_stage1"])
VNAME = f"verdict_stage{a.stage}" + ("_addon" if a.addon else "_crossseed" if a.crossseed else "")
rng = np.random.RandomState(CR["bootstrap"]["seed"])
out = dict(stage=a.stage, addon=a.addon, k=k, M=M, tag=cfg["tag"])
# 審查加入 2026-11-14：記錄程式版本與判準檔雜湊（通用規則「git hash 全留」；確認判讀時的 criteria.json 即凍結版）
import hashlib, subprocess
_here = os.path.dirname(os.path.abspath(__file__))
try:
    out["git_head"] = subprocess.run(["git", "-C", _here, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() or None
    out["git_dirty"] = bool(subprocess.run(["git", "-C", _here, "status", "--porcelain", "--", ".."], capture_output=True, text=True).stdout.strip())
except Exception as e_:  # git 不可用不影響判讀
    out["git_head"] = f"unavailable: {e_}"
out["criteria_sha256"] = hashlib.sha256(open(os.path.join(_here, "..", "configs", "criteria.json"), "rb").read()).hexdigest()
out["cfg_sha256"] = hashlib.sha256(open(a.cfg, "rb").read()).hexdigest()
out["matmul_precision"] = cfg.get("matmul_precision", "default")
out["scope"] = "CIFAR-10／ResNet-9、1% 移除、固定 G_V 子空間及 7 個便宜方法"


def save_verdict():
    path = f"{P['res']}/{VNAME}.json"
    temp = f"{path}.{os.getpid()}.tmp"
    with open(temp, "w") as stream:
        json.dump(nullable(out), stream, indent=1, allow_nan=False)
    os.replace(temp, path)


def finite_float(value):
    return float(value) if np.isfinite(value) else None


def complete_mean(values):
    return stable_mean(values) if np.isfinite(values).all() else float("nan")


def nullable(value):
    if isinstance(value, dict):
        return {key: nullable(val) for key, val in value.items()}
    if isinstance(value, (tuple, list)):
        return [nullable(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    return finite_float(value) if isinstance(value, float) else value


def boot_ci(x, n=CR["bootstrap"]["n"]):
    idx = rng.randint(0, len(x), (n, len(x)))
    m = np.nanmean(x[idx], 1)
    lo = (1 - CR["bootstrap"]["ci"]) / 2
    return float(np.nanquantile(m, lo)), float(np.nanquantile(m, 1 - lo))


def paired(dx):
    if not np.isfinite(dx).all():
        return dict(mean=None, ci=None, undefined_queries=int((~np.isfinite(dx)).sum()))
    return dict(mean=stable_mean(dx), ci=boot_ci(dx))


def synthetic_gain(Y, C_target_lds, Ug, spell_lds, masks, L, seed, tolerance):
    """Report 59 M2 diagnostic. Report calibration failure for this fixed noise draw."""
    scale = np.std(Y, axis=0, keepdims=True)
    E = np.random.default_rng(seed).standard_normal(Y.shape).astype(Y.dtype) * scale
    removed = -masks.astype(np.float32)
    pred_y, pred_e = removed @ Y, removed @ E
    target_rank = rankdata(L, axis=0)
    target_rank -= target_rank.mean(axis=0)
    target_norm = np.linalg.norm(target_rank, axis=0)

    def evaluate(sigma):
        pred_rank = rankdata(pred_y + sigma * pred_e, axis=0)
        pred_rank -= pred_rank.mean(axis=0)
        numer = (pred_rank * target_rank).sum(axis=0)
        denom = np.linalg.norm(pred_rank, axis=0) * target_norm + 1e-30
        return float(np.mean(numer / denom))

    lo, hi = 0.0, 1.0
    score = evaluate(hi)
    while score > C_target_lds + tolerance and hi < 65536:
        lo, hi = hi, hi * 2
        score = evaluate(hi)
    best_sigma, best_score = min(((0.0, evaluate(0.0)), (hi, score)), key=lambda p: abs(p[1] - C_target_lds))
    for _ in range(35):
        if abs(best_score - C_target_lds) <= tolerance:
            break
        sigma = (lo + hi) / 2
        score = evaluate(sigma)
        if abs(score - C_target_lds) < abs(best_score - C_target_lds):
            best_sigma, best_score = sigma, score
        if score > C_target_lds:
            lo = sigma
        else:
            hi = sigma
    Cs = Y + best_sigma * E
    exact_score = float(lds_q(Cs, masks, L).mean())
    result = dict(sigma=best_sigma, cheap_synthetic_lds=exact_score, cheap_real_lds=C_target_lds,
                  calibrated=abs(exact_score - C_target_lds) <= tolerance,
                  noise_family="固定的逐查詢尺度 iid 常態；僅描述此控制")
    if result["calibrated"]:
        Hs, _ = H.hybrid(Y, Cs, Ug)
        result["gain"] = float(lds_q(Hs, masks, L).mean()) - max(spell_lds, exact_score)
    return result


def descriptive_report59():
    legacy = CR["stage0"]["descriptive_legacy"]
    shrink = CR["exploratory_not_for_verdict"]["alpha_q_shrink_weight"]
    partitions = (np.arange(k), np.arange(k)[::2], np.arange(k)[1::2])
    v2 = []
    for q in partitions:
        y, c, l = Y[:, q], C[:, q], L[:, q]
        b = max(1, int(round(CR["B_frac"] * len(q))))
        uc = H.select_U(c.T @ c, b)
        y_projected, c_projected = H.spell(y, uc), H.spell(c, uc)
        scalar = H.fit_alpha(y, c, uc)
        den = (c_projected * c_projected).sum(axis=0)
        fit = np.divide((y_projected * c_projected).sum(axis=0), den, out=np.zeros_like(den), where=den > 0)
        hyb_v2 = y_projected + ((1 - shrink) * fit + shrink * scalar)[None] * (c - c_projected)
        gv_subset = H.select_U(GV[np.ix_(q, q)], b)
        lds_v2, lc, lsv, lsc = (float(lds_q(z, masks, l).mean()) for z in (hyb_v2, c, H.spell(y, gv_subset), y_projected))
        v2.append(dict(k=len(q), B=b, hybrid_lds=lds_v2, cheap_lds=lc, spell_GV_lds=lsv,
                       spell_C_gram_lds=lsc, gain=lds_v2 - max(lc, lsv, lsc)))
    result = dict(v2={"all": v2[0], "even": v2[1], "odd": v2[2]})
    ratio = out["kappa_res_over_kappa"]
    if ratio is not None and np.isfinite(ratio) and ratio < legacy["kappa_ratio_threshold"]:
        result["legacy_description"] = "G_V 的前 B 個方向捕捉到的殘差能量偏低（限固定 U）"
    if out["gain"] < legacy["gain_threshold"]:
        yr, cr = Y - sp, C - H.spell(C, Ug)
        den = (cr * cr).sum(axis=0)
        a_oracle = np.divide((cr * yr).sum(axis=0), den, out=np.zeros_like(den), where=den > 0)
        oracle_lds = float(lds_q(sp + a_oracle[None] * cr, masks, L).mean())
        oracle_gain = oracle_lds - max(out["spell_lds"], out["cheap_lds"][best])
        syn = synthetic_gain(Y, out["cheap_lds"][best], Ug, out["spell_lds"], masks, L,
                             CR["stage0"]["negative_control"]["seed"], legacy["synthetic_lds_tolerance"])
        if not syn["calibrated"]:
            reading = "合成控制未校準"
        elif syn["gain"] < legacy["gain_threshold"]:
            reading = "此合成雜訊控制的增益低於 0.04"
        elif oracle_gain < legacy["gain_threshold"]:
            reading = "固定 U 和 C 下，補空間的逐查詢原地擬合增益低於 0.04"
        else:
            reading = "固定 U 和 C 的逐查詢原地擬合有可見增益"
        result["M2"] = dict(synthetic=syn, oracle_alpha_in_sample_lds=oracle_lds,
                            oracle_alpha_in_sample_gain=oracle_gain, reading=reading)
    return result


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
if Lall.ndim != 2 or Lall.shape[1] != cfg["k_total"] + 1 or not np.isfinite(Lall).all():
    sys.exit("重訓查詢損失／準確率欄數不符或含非有限值")
L, acc = Lall[:, :k], Lall[:, -1]
os.makedirs(P["res"], exist_ok=True)
np.save(f"{P['res']}/retrain_query_losses_k{k}_M{M}.npy", np.concatenate([L, acc[:, None]], 1))  # 原始數字（小檔）留存
n = cfg["n_train"] or 50000
masks = np.stack([removal_mask_np(cfg, j, n) for j in js])
noise_paths = [f"{rd}/noise_{i}_task0.npy" for i in range(1, cfg["noise_repeats"] + 1)]
noise = [np.load(f) for f in noise_paths if os.path.isfile(f)]
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
    G["export_fidelity_max_abs_logit_diff"] = finite_float(d_)
else:
    G["export_fidelity"] = None
lds_magic_q = lds_q(Y, masks, L)
G["cifar_magic_lds"] = finite_float(complete_mean(lds_magic_q))
G["cifar_magic_repro"] = bool(G["cifar_magic_lds"] is not None and G["cifar_magic_lds"] >= CR["gates"]["cifar_magic_lds_min"])
if G["cifar_magic_lds"] is None:
    G["cifar_magic_undefined_queries"] = np.flatnonzero(~np.isfinite(lds_magic_q)).tolist()
if len(noise) == cfg["noise_repeats"] and len(noise) > 1 and \
        all(x.shape == (cfg["k_total"] + 1,) and np.isfinite(x).all() for x in noise):
    ns = np.std(np.stack(noise)[:, :k], 0, ddof=1)
    sig = L.std(0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.divide(ns, sig)
    G["noise_std_over_signal_std"] = stable_mean(ratio) if np.isfinite(ratio).all() else None
    G["noise_ok"] = bool(G["noise_std_over_signal_std"] is not None and G["noise_std_over_signal_std"] <= CR["gates"]["noise_floor"]["max_noise_std_vs_signal_std"])
else:
    G["noise_ok"] = None
out["gates"] = G
gate_keys = ["hybrid_identities", "mlp_magic", "export_fidelity", "cifar_magic_repro", "noise_ok"]
bad = [g for g in gate_keys if G.get(g) is not True]
out["gate_status"] = "ALL_PASS" if not bad else "UNTESTABLE(未能測試): " + ",".join(bad)
print(json.dumps(out["gates"], indent=1)); print("閘門：", out["gate_status"])
if bad and not a.smoke:
    out["verdict"] = "UNTESTABLE"
    save_verdict()
    print("結論：未能測試（前三關失敗不判題目）"); sys.exit(3)

# ------------------------------------------------------------ 便宜方法 LDS
out["magic_lds"] = G["cifar_magic_lds"]


def invalid(why):
    """程式錯誤（NaN/inf 等）：記為 INVALID，與「題目不成立」分開，不給 STOP/CONTINUE/PASS/FAIL。"""
    out["verdict"] = f"INVALID(程式錯誤，非題目結論): {why}"
    save_verdict()
    print("結論：", out["verdict"]); sys.exit(4)


def untestable(why, undefined):
    out["verdict"] = f"UNTESTABLE({why})"
    out["undefined_queries"] = undefined
    save_verdict()
    print("結論：", out["verdict"]); sys.exit(3)


cheap = {}
missing = []
for nm in CR["cheap_variants"]:
    f = f"{P['cheap']}/{nm}.npy"
    if os.path.exists(f):
        cheap[nm] = np.load(f)[:, :k]
    else:
        missing.append(nm)
if missing:
    print("便宜方法不完整：", missing); sys.exit(2)
GV = np.load(f"{P['cheap']}/gram_query_grads.npy")[:k, :k]
nonfinite = [nm for nm, S in cheap.items() if S.shape != (n, k) or not np.all(np.isfinite(S))]
if not np.all(np.isfinite(Y)) or not np.all(np.isfinite(L)) or not np.all(np.isfinite(GV)):
    nonfinite.append("Y/L/G_V")
if nonfinite:
    invalid("非有限值：" + ",".join(sorted(set(nonfinite))))
cl = {nm: lds_q(S, masks, L) for nm, S in cheap.items()}
undefined_cheap = {nm: np.flatnonzero(~np.isfinite(vals)).tolist() for nm, vals in cl.items() if not np.isfinite(vals).all()}
if undefined_cheap and a.stage == 0:
    untestable("便宜方法 LDS 有零秩變異", undefined_cheap)
out["cheap_lds"] = {nm: stable_mean(vals) if np.isfinite(vals).all() else None for nm, vals in cl.items()}
if a.stage == 1:  # 階段 1 沿用階段 0 事先選定的最佳便宜方法，不重選
    with open(f"{P['res']}/verdict_stage0.json") as fh:
        best = json.load(fh)["best_cheap"]
else:
    best = max(out["cheap_lds"], key=out["cheap_lds"].get)
out["best_cheap"] = best; C = cheap[best]
B = max(1, int(round(CR["B_frac"] * k))); out["B"] = B
Ug = H.select_U(GV, B, "gram"); Ur = H.select_U(GV, B, "random", seed=0)
hyb, alpha = H.hybrid(Y, C, Ug); sp = H.spell(Y, Ug)
with np.errstate(divide="ignore", invalid="ignore"):
    out["alpha"] = alpha; out["kappa"] = finite_float(H.kappa(Y, Ug)); out["kappa_res"] = finite_float(H.kappa_res(Y, C, Ug, alpha))
    out["kappa_res_over_kappa"] = out["kappa_res"] / out["kappa"] if out["kappa"] and out["kappa_res"] is not None else None
    out["kappa_random_U"] = finite_float(H.kappa(Y, Ur))
hl, sl = lds_q(hyb, masks, L), lds_q(sp, masks, L)
if a.stage == 0 and not np.isfinite(sl).all():
    untestable("Spell LDS 有零秩變異", {"spell": np.flatnonzero(~np.isfinite(sl)).tolist()})
out["hybrid_lds"], out["spell_lds"] = complete_mean(hl), complete_mean(sl)
hr, _ = H.hybrid(Y, C, Ur); out["hybrid_randomU_lds"] = complete_mean(lds_q(hr, masks, L))
out["spell_randomU_lds"] = complete_mean(lds_q(H.spell(Y, Ur), masks, L))
cheap_lds_best = out["cheap_lds"][best]
out["gain"] = out["hybrid_lds"] - max(out["spell_lds"], cheap_lds_best) if cheap_lds_best is not None and np.isfinite(hl).all() else float("nan")
out["per_method_kappa_res_over_kappa"] = {}
for nm, S in cheap.items():
    al = H.fit_alpha(Y, S, Ug)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["per_method_kappa_res_over_kappa"][nm] = finite_float(H.kappa_res(Y, S, Ug, al) / out["kappa"]) if out["kappa"] else None

# ------------------------------------------------------------ 探索性／次要報告（審查後定案 2026-11-14；不參與任何判讀）
def exploratory():
    ex = {}
    # S3：逐查詢 α_q（量測子空間內各查詢自己的最小平方係數），向純量 α 收縮，收縮權重事先固定為 0.5（不調）
    Pg = Ug @ Ug.T; YP, CP = Y @ Pg, C @ Pg  # 逐查詢（欄）在量測子空間內的投影
    den = (CP * CP).sum(0)
    aq = np.where(den > 0, (YP * CP).sum(0) / np.where(den > 0, den, 1), 0.0)
    for w in (0.0, 0.5):
        al_q = (1 - w) * aq + w * alpha
        Yh = YP + al_q[None, :] * (C - CP)
        lq = lds_q(Yh, masks, L)
        value = stable_mean(lq) if np.isfinite(lq).all() else float("nan")
        ex[f"alpha_q_shrink{w}"] = dict(lds=value, gain=value - max(out["spell_lds"], cheap_lds_best) if cheap_lds_best is not None else None)
    ex["alpha_q_stats"] = dict(mean=float(aq.mean()), std=float(aq.std()), min=float(aq.min()), max=float(aq.max()))
    # S4：查詢對半分（偶數／奇數索引），各自獨立估 U、α，各自算增益
    halves = []
    for h in (np.arange(k)[0::2], np.arange(k)[1::2]):
        Bh = max(1, int(round(CR["B_frac"] * len(h)))); Uh = H.select_U(GV[np.ix_(h, h)], Bh, "gram")
        Yh_, Ch_ = Y[:, h], C[:, h]
        hy, ah = H.hybrid(Yh_, Ch_, Uh); spx = H.spell(Yh_, Uh)
        lh, ls_, lc = lds_q(hy, masks, L[:, h]), lds_q(spx, masks, L[:, h]), lds_q(Ch_, masks, L[:, h])
        mixed_score, spell_score, cheap_score = map(complete_mean, (lh, ls_, lc))
        g = mixed_score - max(spell_score, cheap_score) if np.isfinite([mixed_score, spell_score, cheap_score]).all() else float("nan")
        halves.append(dict(k=len(h), B=Bh, alpha=ah, hybrid_lds=mixed_score, spell_lds=spell_score,
                           cheap_lds=cheap_score, gain=g))
    left, right = (part["gain"] for part in halves)
    ex["split_queries"] = dict(halves=halves, same_sign=bool(left * right > 0) if np.isfinite([left, right]).all() else None,
                               gain_diff=left - right)
    return ex


out["exploratory_not_for_verdict"] = exploratory()

# ------------------------------------------------------------ 階段 0 判讀
if a.stage == 0:
    S0 = CR["stage0"]
    control = S0["negative_control"]
    perm = np.random.default_rng(control["seed"]).permutation(n)
    C_perm = C[perm]
    H_perm, _ = H.hybrid(Y, C_perm, Ug)
    lds_C_perm = complete_mean(lds_q(C_perm, masks, L))
    lds_H_perm = complete_mean(lds_q(H_perm, masks, L))
    perm_gain = lds_H_perm - max(out["spell_lds"], lds_C_perm) if np.isfinite(lds_C_perm) else float("nan")
    Y_perp = Y - sp
    rho_q = column_cosine(Y_perp, C - H.spell(C, Ug))
    perm_q = column_cosine(Y_perp, C_perm - H.spell(C_perm, Ug))
    decision, v = issue_stage0_decision(rho_q, perm_q, lds_magic_q, sl, cl[best], perm_gain, S0, CR["bootstrap"])
    decision["signal"].setdefault("control", {}).update(dict(permuted_cheap_lds=lds_C_perm, permuted_hybrid_lds=lds_H_perm))
    out["phenomenon_signal_headroom"] = decision
    if v.startswith("INVALID") and not a.smoke:
        invalid("置換負控制未通過；未能測試")
    if v.startswith("UNTESTABLE") and not a.smoke:
        out["verdict"] = v
        save_verdict()
        print("結論：", v); sys.exit(3)
    if a.crossseed and v.startswith("NEED_CROSSSEED"):
        cs = S0["crossseed"]
        count = cs["retrain_count"]
        cs_dir = os.path.join(os.path.dirname(P["retrain"]), "retrain_cs")
        jcs, Lcs = load_retrain(cs_dir, count)
        if len(jcs) != count:
            print(f"跨種子重訓不完整：{len(jcs)}/{count}"); sys.exit(2)
        if Lcs.shape != (count, cfg["k_total"] + 1) or not np.isfinite(Lcs).all():
            invalid("跨種子重訓查詢數不符或含非有限值")
        mcs = np.stack([removal_mask_np(cfg, j, n) for j in jcs])
        exact_cs, cheap_cs = lds_q(Y, mcs, Lcs[:, :k]), lds_q(C, mcs, Lcs[:, :k])
        undefined = {name: np.flatnonzero(~np.isfinite(vals)).tolist() for name, vals in (("exact", exact_cs), ("cheap", cheap_cs)) if not np.isfinite(vals).all()}
        if undefined:
            untestable("跨種子 LDS 有零秩變異", undefined)
        result = grade_crossseed(exact_cs, cheap_cs, cs["headroom_pass_min"])
        decision["crossseed"] = dict(result, n=count, train_seed=cs["train_seed"])
        v = "PASS(符合候選池條件)" if result["status"] == "PASS" else "GREY(跨種子 H_x 未達門檻)"
    elif a.crossseed:
        print("階段 0 三項須全通過才能判讀跨種子 H_x：", v); sys.exit(2)
    # 僅描述、不參與判讀（審查加入 2026-11-14）：k=50 下增益的配對 bootstrap CI 與最佳便宜方法 LDS 的折半信度
    out["descriptive_stage0"] = dict(
        hybrid_minus_spell=paired(hl - sl), hybrid_minus_cheap=paired(hl - cl[best]),
        split_half_r_best_cheap=descriptive_spearman(lds_q(C, masks[0::2], L[0::2]), lds_q(C, masks[1::2], L[1::2])),
        split_half_r_magic=descriptive_spearman(lds_q(Y, masks[0::2], L[0::2]), lds_q(Y, masks[1::2], L[1::2])))
    out["exploratory_not_for_verdict"]["report59"] = descriptive_report59()
    out["interpretation_note"] = "ρ⊥ 衡量固定補空間的幾何對齊。H 是完整精確 Y 的 LDS 參考差，無法保證 10 個方向能兌現。混合估計器與基線的比較僅供描述。"
    out["verdict"] = v
else:
    S1 = CR["stage1"]
    dh_s, dh_c = hl - sl, hl - cl[best]
    ps, pc = paired(dh_s), paired(dh_c)
    out["hybrid_minus_spell"], out["hybrid_minus_cheap"] = ps, pc
    g = min(ps["mean"], pc["mean"]) if ps["mean"] is not None and pc["mean"] is not None else None
    lo = min(ps["ci"][0], pc["ci"][0]) if ps["ci"] is not None and pc["ci"] is not None else None
    out["gain_min"], out["ci_lower_min"] = g, lo
    # 舊階段 1 的描述性跨種子比較。
    cs_dir = os.path.join(os.path.dirname(P["retrain"]), "retrain_cs")
    cs_count = CR["stage0"]["crossseed"]["retrain_count"]
    jcs, Lcs = load_retrain(cs_dir, cs_count)
    if len(jcs) != cs_count:
        out["crossseed"] = None
    else:
        mcs = np.stack([removal_mask_np(cfg, j, n) for j in jcs])
        d = complete_mean(lds_q(hyb, mcs, Lcs[:, :k]) - lds_q(C, mcs, Lcs[:, :k]))
        out["crossseed"] = dict(n=len(jcs), hybrid_minus_cheap=finite_float(d), ok=bool(d >= S1["crossseed_hybrid_minus_cheap_min"]))
    # 可靠度
    cq = cl[best]
    half = [lds_q(C, masks[i::2], L[i::2]) for i in (0, 1)]
    r = descriptive_spearman(*half)
    sb = 2 * r / (1 + r) if r is not None and r > -1 else None
    rho = H.reliability(Y, C, Ug)
    qloss = np.load(f"{P['models']}/model_s{cfg['train_seed']}.npz")["query_loss"][:k]
    sp_rho = descriptive_spearman(rho, cq)
    sp_triv = {"query_loss": descriptive_spearman(qloss, cq), "norm_C": descriptive_spearman(np.linalg.norm(C, axis=0), cq)}
    R = S1["reliability"]
    reliability_defined = all(x is not None for x in (sb, sp_rho, *sp_triv.values()))
    out["reliability"] = dict(split_half_SB=sb, spearman_rho_hat=sp_rho, trivial=sp_triv,
                              pass_=bool(sb >= R["split_half_min"] and sp_rho >= R["spearman_min"] and sp_rho > max(abs(v) for v in sp_triv.values())) if reliability_defined else None,
                              grey=bool(sb >= R["split_half_min"] and R["spearman_grey_low"] <= sp_rho < R["spearman_min"]) if reliability_defined else None,
                              note=None if reliability_defined else "至少一個選用的等級相關無法計算")
    cs = out["crossseed"]
    cs_missing = "未完成(缺跨種子重訓)"
    # 審查後定案（2026-11-14，M2）：跨種子差 < 0 直接 FAIL，優先於灰區與 PASS；增益 < 0.04 也是 FAIL，不需跨種子
    if g is None or lo is None:
        v = "未能判讀（增益未定義）"
    elif g < S1["gain_grey_low"]:
        v = "FAIL"
    elif cs is None or cs["hybrid_minus_cheap"] is None:
        v = cs_missing
    elif cs["hybrid_minus_cheap"] < S1["crossseed_hybrid_minus_cheap_min"]:
        v = "FAIL(跨種子差<0，優先於灰區)"
    elif g >= S1["gain_pass"] and lo > S1["ci_lower_gt"]:
        v = "PASS"
    elif g >= S1["gain_pass"]:
        v = "GREY(增益≥0.08 但 CI 下界≤0.03)"
    elif not reliability_defined:
        v = "未能判讀（可靠度未定義）"
    elif out["reliability"]["pass_"]:
        v = "PASS(僅可靠度路線)"
    elif out["reliability"]["grey"]:
        v = "GREY(可靠度灰區)"
    else:
        v = "FAIL"
    # 審查後定案（2026-11-14，M3）：加碼（k=400）後仍為灰區記為 FAIL
    if a.addon and v.startswith("GREY"):
        v = "FAIL(加碼後仍為灰區)"
    out["exploratory_not_for_verdict"]["legacy_stage1_label"] = v
    v = "DESCRIPTIVE(階段 1 僅供描述)"
    out["verdict"] = v
save_verdict()
print(json.dumps({k_: v_ for k_, v_ in out.items() if k_ != "gates"}, indent=1, default=float))
print("\n結論：", out["verdict"], "\n（機械輸出；最終判讀由 Opus 對照 PREREG.md 審）")
