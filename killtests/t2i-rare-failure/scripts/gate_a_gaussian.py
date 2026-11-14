"""閘門 (a)：同維度（d=4x64x64=16384）高斯半空間 {w·z >= t} 的解析尾機率驗證，純 CPU。
真值 = 1 - Phi(t)，目標機率 1e-3..1e-6，每個機率 20 次重複，判準：|mean(est)/p - 1| < 20%（相對偏差）。
另報變異係數與每次模型呼叫數（對照暴力抽樣：達到同變異係數所需張數 ~ (1-p)/(p*CV^2)）。
輸出 results/gate_a/gate_a.json 與 results/gate_a/gate_a.md，結束碼 0 = 至少一個估計器通過。
用法：python gate_a_gaussian.py [--reps 20] [--N 1000]
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load_json
from estimators import subset_simulation, ce_is_mean_shift

G = load_json("criteria.json")["gate_a"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=G["reps"])
    ap.add_argument("--N", type=int, default=G["N_per_level"])
    ap.add_argument("--d", type=int, default=G["d"])
    ap.add_argument("--seed", type=int, default=G["rng_seed"])
    ap.add_argument("--mcmc-steps", type=int, default=G["mcmc_steps"])
    a = ap.parse_args()
    d = a.d; w = np.random.default_rng(0).standard_normal(d); w /= np.linalg.norm(w)
    score = lambda z: z @ w
    out = {"config": vars(a), "rows": []}
    lines = ["# 閘門 (a)：高斯半空間解析尾機率\n", f"d={d}，每階段 N={a.N}，重複 {a.reps}，偏差門檻 {G['max_rel_bias']:.0%}\n",
             "| 估計器 | p 真值 | 估計均值/真值 | 相對偏差 | CV | 平均呼叫數 | 暴力等效張數(同CV) | 通過 |", "|---|---|---|---|---|---|---|---|"]
    verdict = {}
    for name, fn in (("subset_sim", lambda p_t, rng: subset_simulation(score, d, p_t, N=a.N, mcmc_steps=a.mcmc_steps, rng=rng)),
                     ("ce_is", lambda p_t, rng: ce_is_mean_shift(score, d, p_t, N=a.N, rng=rng))):
        ok_all = True
        for ep in G["log10_p"]:
            p = 10.0 ** ep; t = stats.norm.isf(p)
            ests, calls = [], []
            t0 = time.time()
            for r in range(a.reps):
                rng = np.random.default_rng([a.seed, abs(ep), r, 1 if name == "ce_is" else 0])
                e, c, _ = fn(t, rng); ests.append(e); calls.append(c)
            ests = np.array(ests); m = np.nanmean(ests); bias = m / p - 1
            cv = float(np.nanstd(ests, ddof=1) / m) if m > 0 else float("inf")
            ok = bool(abs(bias) < G["max_rel_bias"]) and not np.isnan(ests).any()
            ok_all &= ok
            brute = (1 - p) / (p * cv ** 2) if cv > 0 and np.isfinite(cv) else float("nan")
            out["rows"].append(dict(estimator=name, p=p, t=float(t), mean_ratio=float(m / p), bias=float(bias), cv=cv,
                                    calls_mean=float(np.mean(calls)), brute_equiv=float(brute), ok=ok, ests=ests.tolist(),
                                    seconds=round(time.time() - t0, 1)))
            lines.append(f"| {name} | 1e{ep} | {m/p:.3f} | {bias:+.1%} | {cv:.2f} | {np.mean(calls):.0f} | {brute:.2e} | {'通過' if ok else '失敗'} |")
            print(lines[-1], flush=True)
        verdict[name] = ok_all
    out["verdict"] = verdict
    out["gate_a_pass"] = any(verdict.values())
    lines.append(f"\n各估計器全部機率皆通過：{verdict}。**閘門 (a)：{'通過' if out['gate_a_pass'] else '未通過（記為未能測試，限 3 個工作天修）'}**"
                 "（PREREG：至少一個估計器全數通過即可，之後只能用通過者）")
    d_out = ROOT / "results" / "gate_a"; d_out.mkdir(parents=True, exist_ok=True)
    (d_out / "gate_a.json").write_text(json.dumps(out, indent=1)); (d_out / "gate_a.md").write_text("\n".join(lines) + "\n")
    sys.exit(0 if out["gate_a_pass"] else 1)


if __name__ == "__main__":
    main()
