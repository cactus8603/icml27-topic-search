"""罕見事件估計器（numpy，與生成模型無關）：事件 {score(z) >= t}，z ~ N(0, I_d)。
  - subset_simulation：Au-Beck 子集模擬，pCN MCMC（保持標準高斯不變），自適應中間門檻（條件機率 p0）。
  - ce_is_mean_shift：交叉熵法的均值平移重要性抽樣（對角單位共變異的高斯提議，只估均值）。
score_fn：輸入 (n, d) 回傳 (n,) 分數；設計成之後可換成「雜訊 -> SD 生成 -> CLIP 代理分數」。
回傳 (估計值, 模型呼叫次數(score_fn 被評估的樣本總數), 診斷資訊)。
"""
import numpy as np


def subset_simulation(score_fn, d, t, N=1000, p0=0.1, mcmc_steps=1, beta0=0.5, rng=None, max_levels=30):
    rng = rng or np.random.default_rng()
    z = rng.standard_normal((N, d)); s = score_fn(z); calls = N
    n_seed = int(N * p0); p = 1.0; levels = 0; beta = beta0; accs = []
    while levels < max_levels:
        thr = np.sort(s)[::-1][n_seed - 1]  # 第 n_seed 大 = 目前層門檻
        if thr >= t:
            return p * float((s >= t).mean()), calls, dict(levels=levels, acc=float(np.mean(accs)) if accs else None)
        p *= n_seed / N; levels += 1
        idx = np.argsort(s)[::-1][:n_seed]
        zc, sc = z[idx], s[idx]
        per = N // n_seed
        Z, S = [zc], [sc]
        cur_z, cur_s = zc, sc
        for _ in range(per - 1):
            for _ in range(mcmc_steps):
                prop = np.sqrt(1 - beta ** 2) * cur_z + beta * rng.standard_normal(cur_z.shape)
                ps = score_fn(prop); calls += len(prop)
                ok = ps >= thr
                cur_z = np.where(ok[:, None], prop, cur_z); cur_s = np.where(ok, ps, cur_s)
                a = ok.mean(); accs.append(a)
                beta = float(np.clip(beta * np.exp(a - 0.3), 0.05, 1.0))  # 簡單自適應：目標接受率約 0.3
            Z.append(cur_z); S.append(cur_s)
        z = np.concatenate(Z)[:N]; s = np.concatenate(S)[:N]
    return float("nan"), calls, dict(levels=levels, failed=True)


def ce_is_mean_shift(score_fn, d, t, N=1000, rho=0.1, max_iter=30, N_final=None, rng=None):
    """CE 多階段：每階段取前 rho 分位的菁英，以加權（對 N(0,I) 的概似比）平均更新提議均值；最後用固定提議做 IS。"""
    rng = rng or np.random.default_rng()
    mu = np.zeros(d); calls = 0; N_final = N_final or N
    for it in range(max_iter):
        z = rng.standard_normal((N, d)) + mu; s = score_fn(z); calls += N
        thr = min(np.sort(s)[::-1][int(N * rho) - 1], t)
        el = s >= thr
        logw = -(z[el] * mu).sum(1) + 0.5 * mu @ mu  # log N(z;0,I) - log N(z;mu,I)，z 為提議樣本
        w = np.exp(logw - logw.max())
        mu = (w[:, None] * z[el]).sum(0) / w.sum()
        if not np.all(np.isfinite(mu)): return float("nan"), calls, dict(failed="mu_nonfinite")
        if thr >= t: break
    z = rng.standard_normal((N_final, d)) + mu; s = score_fn(z); calls += N_final
    logw = -(z * mu).sum(1) + 0.5 * mu @ mu
    with np.errstate(over="ignore", under="ignore"):
        est = float(np.mean((s >= t) * np.exp(np.clip(logw, -700, 700))))  # 高維下權重退化時此值失真，由閘門 (a) 偵測
    return est, calls, dict(iters=it + 1, mu_norm=float(np.linalg.norm(mu)), hits=int((s >= t).sum()))


def subset_simulation_v2(score_fn, d, t, N=1000, p0=0.1, mcmc_steps=4, beta0=0.5, rng=None, max_levels=30):
    """補充閘門 (a') 用版本（2026-11-14 審查後、結果前定案）。與 subset_simulation 的差異只有兩點：
      1. pCN 步長 beta 在同一階段內固定，只在階段之間依「上一階段的平均接受率」調整（beta *= exp(acc - 0.3)，夾在 [0.05, 1]）；
      2. mcmc_steps 為每條鏈在兩次記錄之間走的 MCMC 步數（凍結 4，範圍 3-5）。
    診斷另回 delta2：Au-Beck 變異的『獨立近似』（各階段 (1-p_i)/(p_i N)，不含鏈相關項，會低估），僅供 (a') 回報覆蓋率用。
    嚴禁以任何高斯問題上量到的係數校正對 SD 的估計值。"""
    rng = rng or np.random.default_rng()
    z = rng.standard_normal((N, d)); s = score_fn(z); calls = N
    n_seed = int(N * p0); p = 1.0; levels = 0; beta = beta0; accs = []; delta2 = 0.0
    while levels < max_levels:
        thr = np.sort(s)[::-1][n_seed - 1]
        if thr >= t:
            pl = float((s >= t).mean())
            if pl > 0: delta2 += (1 - pl) / (pl * N)
            return p * pl, calls, dict(levels=levels, acc=float(np.mean(accs)) if accs else None, delta2=float(delta2))
        p *= n_seed / N; levels += 1; delta2 += (1 - p0) / (p0 * N)
        idx = np.argsort(s)[::-1][:n_seed]
        cur_z, cur_s = z[idx], s[idx]
        per = N // n_seed
        Z, S = [cur_z], [cur_s]; lvl_acc = []
        for _ in range(per - 1):
            for _ in range(mcmc_steps):
                prop = np.sqrt(1 - beta ** 2) * cur_z + beta * rng.standard_normal(cur_z.shape)
                ps = score_fn(prop); calls += len(prop)
                ok = ps >= thr
                cur_z = np.where(ok[:, None], prop, cur_z); cur_s = np.where(ok, ps, cur_s)
                lvl_acc.append(float(ok.mean()))
            Z.append(cur_z); S.append(cur_s)
        a = float(np.mean(lvl_acc)); accs += lvl_acc
        beta = float(np.clip(beta * np.exp(a - 0.3), 0.05, 1.0))  # 只在階段之間調整
        z = np.concatenate(Z)[:N]; s = np.concatenate(S)[:N]
    return float("nan"), calls, dict(levels=levels, failed=True)
