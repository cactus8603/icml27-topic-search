"""判準 E / D 的計算函式（第 2 週用，純 numpy；2026-11-14 審查後、結果前定案）。
- 相對 RMSE 換算暴力抽樣等效張數（不用 CV：估計器有偏時 CV 會高估加速）：
    rrmse = sqrt(mean((est/p_ref - 1)^2))；n_brute_equiv = (1 - p_ref) / (p_ref * rrmse^2)
- 成本 = 單次估計的模型呼叫數（含被拒的 MCMC 提議）+ dev 格調參總呼叫數 / dev_divisor（20）。
- 加速倍數 = n_brute_equiv / 成本；報告時附 bootstrap 區間（對重複次數重抽）。
- 覆蓋率（只回報、不進判定）：名目 95% 區間包含參考值的比例；兩種區間並列：
    'replicate'：est * exp(+-1.96 * sd(log est))，sd 取自同格全部重複（est>0 者）；
    'own_variance'：est * (1 +- 1.96 * sqrt(delta2))，delta2 = 估計器自回的變異（獨立近似，會低估）。
"""
import numpy as np


def rrmse(ests, p_ref):
    e = np.asarray(ests, float)
    return float(np.sqrt(np.mean((e / p_ref - 1.0) ** 2)))  # 含 NaN 時回 NaN，不靜默忽略


def brute_equiv(ests, p_ref):
    r = rrmse(ests, p_ref)
    return float((1 - p_ref) / (p_ref * r ** 2)) if r > 0 and np.isfinite(r) else float("nan")


def speedup(ests, calls, p_ref, dev_calls_total=0.0, dev_divisor=20, n_boot=2000, seed=0):
    ests = np.asarray(ests, float); calls = np.asarray(calls, float)
    def one(e, c):
        return brute_equiv(e, p_ref) / (c.mean() + dev_calls_total / dev_divisor)
    pt = one(ests, calls)
    rng = np.random.default_rng(seed); n = len(ests)
    bs = np.array([one(ests[i], calls[i]) for i in (rng.integers(0, n, n) for _ in range(n_boot))])
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return dict(speedup=float(pt), ci95=[float(lo), float(hi)], rrmse=rrmse(ests, p_ref),
                cost_per_estimate=float(calls.mean() + dev_calls_total / dev_divisor))


def coverage(ests, p_ref, delta2=None, z=1.96):
    e = np.asarray(ests, float); out = {}
    ok = e > 0
    if ok.sum() >= 2:
        s = np.std(np.log(e[ok]), ddof=1)
        out["replicate"] = float(np.mean((e[ok] * np.exp(-z * s) <= p_ref) & (p_ref <= e[ok] * np.exp(z * s))))
    if delta2 is not None:
        d = np.sqrt(np.asarray(delta2, float))
        out["own_variance"] = float(np.mean((e * (1 - z * d) <= p_ref) & (p_ref <= e * (1 + z * d))))
    return out
