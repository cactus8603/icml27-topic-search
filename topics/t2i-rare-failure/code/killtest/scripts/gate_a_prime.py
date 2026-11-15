"""補充閘門 (a')（第 2 週估計之前必須通過；純 CPU；2026-11-14 審查後、結果前定案，判準讀 criteria.json 的 gate_a_prime）。
估計器 = estimators.subset_simulation_v2（步長只在階段之間調、mcmc_steps 凍結 4、N=1000、p0=0.1）。
問題：(1) d=16384 高斯半空間，p = 1e-3、1e-4、1e-5；(2) 多峰：k=3 個互相正交半空間的聯集 {max_i w_i·z >= t}，p=1e-4，
真值 = 1-(1-Phi_bar(t))^3（解析）。每個問題 100 次重複。通過 = 全部問題 |mean(est)/p - 1| < 10% 且無 NaN。
另回報（不進判定）：相對 RMSE、CV、均值標準誤與 z、覆蓋率（replicate / own_variance 兩種區間）。
禁止：用此處量到的偏差係數校正對 SD 的估計值。
2026-11-14 [claude]（報告 60 第 3 節，結果前定案）：輸出新增加速欄位（只改輸出，不改估計器）：
  n_eq = (1-p)/(p*rrmse^2)（暴力抽樣等效張數）；加速(p) = n_eq / 呼叫數（呼叫數 = 每次估計的平均模型呼叫數，含被拒提議）。
  判準 B 的空間項（代理量）用 gauss_1e-5、gauss_1e-4 兩列的加速；此為代理，見 PREREG 第 13 節。
用法：python gate_a_prime.py [--procs 8] [--reps 100] [--out results/gate_a_prime]   結束碼 0 = 通過
"""
import argparse, json, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load_json
from evidence import atomic_json, atomic_text, formal_a_prime_config
from estimators import subset_simulation_v2
from e_metric import rrmse, coverage, brute_equiv

G = load_json("criteria.json")["gate_a_prime"]
_W = None


def _problem(d, kind, k):
    rng = np.random.default_rng(0)
    q, _ = np.linalg.qr(rng.standard_normal((d, k if kind == "multimodal" else 1)))
    return q.T  # (k, d) 正交單位列


def _run(args):
    name, d, k, kind, t, N, steps, seed, rep = args
    W = _problem(d, kind, k)
    score = (lambda z: (z @ W.T).max(1)) if kind == "multimodal" else (lambda z: z @ W[0])
    rng = np.random.default_rng([seed, sum(map(ord, name)), rep])
    e, c, diag = subset_simulation_v2(score, d, t, N=N, p0=G["p0"], mcmc_steps=steps, rng=rng)
    return name, rep, e, c, diag.get("delta2", float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=G["reps"]); ap.add_argument("--N", type=int, default=G["N_per_level"])
    ap.add_argument("--d", type=int, default=G["d"]); ap.add_argument("--seed", type=int, default=G["rng_seed"])
    ap.add_argument("--mcmc-steps", type=int, default=G["mcmc_steps"]); ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--out")
    ap.add_argument("--smoke", action="store_true", help="僅供程式測試；結果不得當閘門 (a') 結果")
    a = ap.parse_args()
    formal_out = (ROOT / "results/gate_a_prime").resolve()
    a.out = str(Path(a.out).resolve() if a.out else (
        ROOT / "results_smoke/gate_a_prime" if a.smoke else formal_out))
    if a.smoke and Path(a.out) == formal_out:
        sys.exit("smoke 不得覆寫正式 (a') 閘門")
    if not a.smoke and any(getattr(a, key) != val for key, val in formal_a_prime_config().items()):
        sys.exit("正式 (a') 閘門必須使用預先登記的參數")
    if not (G["mcmc_steps_allowed_range"][0] <= a.mcmc_steps <= G["mcmc_steps_allowed_range"][1]):
        sys.exit("mcmc_steps 必須在登記範圍內")
    d_out = Path(a.out)
    atomic_json(d_out / "gate_a_prime.json", {"status": "UNTESTABLE", "smoke": a.smoke, "reason": "validation started"})
    probs = []
    for ep in G["gaussian_log10_p"]:
        p = 10.0 ** ep; probs.append((f"gauss_1e{ep}", p, float(stats.norm.isf(p)), "gauss", 1))
    k = G["multimodal"]["k_halfspaces"]
    for ep in G["multimodal"]["log10_p"]:
        p = 10.0 ** ep; qq = -np.expm1(np.log1p(-p) / k)  # 單一半空間機率 q：1-(1-q)^k = p
        probs.append((f"multimodal{k}_1e{ep}", p, float(stats.norm.isf(qq)), "multimodal", k))
    jobs = [(n, a.d, kk, kind, t, a.N, a.mcmc_steps, a.seed, r) for n, p, t, kind, kk in probs for r in range(a.reps)]
    t0 = time.time()
    if a.procs > 1:
        with Pool(a.procs) as P: res = P.map(_run, jobs, chunksize=1)
    else:
        res = [_run(j) for j in jobs]
    out = {"config": vars(a), "criteria": G, "rows": [], "smoke": a.smoke}
    lines = ["# 補充閘門 (a')：subset_simulation_v2（階段間調步長、mcmc_steps=%d）\n" % a.mcmc_steps,
             f"d={a.d}，N={a.N}，重複 {a.reps}，門檻 |偏差| < {G['max_abs_rel_bias']:.0%}（含 NaN 即失敗）。{'**smoke 測試，非正式結果**' if a.smoke else ''}\n",
             "| 問題 | p 真值 | 均值/真值 | 偏差 | 均值SE | z | rRMSE | CV | 覆蓋(replicate) | 覆蓋(own_var) | 呼叫數 | n_eq | 加速 | 通過 |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    ok_all = True
    for n, p, t, kind, kk in probs:
        rr = sorted([r for r in res if r[0] == n], key=lambda r: r[1])
        e = np.array([r[2] for r in rr]); c = np.array([r[3] for r in rr]); d2 = np.array([r[4] for r in rr])
        m = float(np.mean(e)); bias = m / p - 1; se = float(np.std(e, ddof=1) / np.sqrt(len(e)) / p)
        cv = float(np.std(e, ddof=1) / m) if m > 0 else float("inf")
        valid = np.isfinite(e).all() and ((0 <= e) & (e <= 1)).all() and np.isfinite(c).all() and (c > 0).all()
        ok = bool(valid and abs(bias) < G["max_abs_rel_bias"]); ok_all &= ok
        cov = coverage(e, p, d2)
        n_eq = brute_equiv(e, p); speed = n_eq / float(c.mean()) if np.isfinite(n_eq) else float("nan")   # 加速(p) = n_eq / 呼叫數
        z = bias / se if se > 0 else float("nan")
        out["rows"].append(dict(problem=n, p=p, t=t, mean_ratio=m / p, bias=bias, se=se, z=z,
                                rrmse=rrmse(e, p), cv=cv, coverage=cov, calls_mean=float(c.mean()), n_eq=n_eq, speedup=speed, ok=ok, ests=e.tolist(), calls=c.tolist()))
        lines.append(f"| {n} | {p:.0e} | {m/p:.3f} | {bias:+.1%} | {se:.1%} | {z:+.1f} | {rrmse(e,p):.2f} | {cv:.2f} | "
                     f"{cov.get('replicate', float('nan')):.2f} | {cov.get('own_variance', float('nan')):.2f} | {c.mean():.0f} | {n_eq:.3g} | {speed:.3g} | {'通過' if ok else '失敗'} |")
        print(lines[-1], flush=True)
    out["status"] = "COMPLETE"
    out["gate_a_prime_pass"] = bool(ok_all); out["seconds"] = round(time.time() - t0, 1)
    lines.append(f"\n**閘門 (a')：{'通過' if ok_all else '未通過'}**。不得以此處偏差係數校正 SD 估計值。")
    def nullable(value):
        if isinstance(value, dict):
            return {key: nullable(val) for key, val in value.items()}
        if isinstance(value, list):
            return [nullable(item) for item in value]
        if isinstance(value, np.generic):
            value = value.item()
        return None if isinstance(value, float) and not np.isfinite(value) else value

    atomic_text(d_out / "gate_a_prime.md", "\n".join(lines) + "\n")
    atomic_json(d_out / "gate_a_prime.json", nullable(out))
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
