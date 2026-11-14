#!/usr/bin/env python
"""aggregate_week2.py 的合成資料測試。用法：python test_aggregate_week2.py [--tmp DIR]"""
import argparse, json, os, subprocess, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from schedule import ckpt_schedule
ap = argparse.ArgumentParser(); ap.add_argument("--tmp", default=os.path.join(os.environ.get("TMPDIR", "."), "agg2_test")); a = ap.parse_args()
os.makedirs(a.tmp, exist_ok=True)
EPH = np.array(ckpt_schedule(30, 20000)); E = len(EPH)
TG = [10, 25, 50, 75, 100, 150, 200, 300]; SC = np.array([1e-4, 3e-3, 1e-2, 5e-3, 2e-3, 8e-4, 3e-4, 4e-5])
rng = np.random.default_rng(3)
JIT = rng.normal(0, 0.06, 6)   # 每個區塊（模型）共同的 t* 偏移，使 CI 有寬度

def curve(kind, ts):
    le = np.log(EPH / ts)
    return {"v": 0.2 + 0.1 * np.abs(le), "down": np.linspace(2, 1, E)}[kind]

def mk(name, picks, tmap, kinds=None, branches=("cond", "uncond", "cfg1", "cfg3")):
    """tmap[branch] = 基準 t*；pair 的 t* = base*exp(jit_i+jit_j)；kinds[(branch)] = 'down' 使該分支無內部最低點。"""
    pairs = [(i, j) for i in range(len(picks)) for j in range(i + 1, len(picks))]
    pc = np.zeros((len(pairs), E, 8, 3, len(branches)))
    for p, (i, j) in enumerate(pairs):
        for b, br in enumerate(branches):
            ts = tmap[br] * np.exp(JIT[picks[i]] + JIT[picks[j]]); kd = (kinds or {}).get(br, "v")
            kd = kd(p) if callable(kd) else kd
            pc[p, :, :, :, b] = curve(kd, ts)[:, None, None] * SC[None, :, None]
    path = os.path.join(a.tmp, name + ".npz")
    np.savez(path, epochs=EPH, t_grid=np.array(TG), branches=np.array(branches), picks=np.array(picks), pairs=np.array(pairs), pair_cos_eps=pc, git_hash="t")
    return path

def agg(**kw):
    cmd = [sys.executable, os.path.join(HERE, "aggregate_week2.py"), "--out", os.path.join(a.tmp, "v.json")]
    for k, v in kw.items(): cmd += ["--" + k.replace("_", ""), str(v)] if k != "h" else ["--h", str(v)]
    r = subprocess.run(cmd, capture_output=True, text=True); assert r.returncode == 0, r.stderr[-1500:]
    return json.load(open(os.path.join(a.tmp, "v.json")))["sections"]

fails = []
def check(n, c, x=""):
    print(("OK   " if c else "FAIL ") + n, x)
    if not c: fails.append(n)

P6 = list(range(6)); P4 = list(range(4))
k6 = mk("k6", P6, {"cond": 1000, "uncond": 1000, "cfg1": 800, "cfg3": 500})
# 情境 1：負控制通過（m1≈0），m6 PASS
k1 = mk("k1_ok", P4, {"cond": 1000, "cfg3": 1000}, branches=("cond", "cfg3"))
s = agg(k6=k6, k1=k1, h=2.5)
check("neg ctrl passed -> c PASS", s["negative_control"]["passed"] is True and s["c_main"]["status"] == "PASS", s["c_main"]["status"])
check("N=4 matches h>2 rule, low power", s["negative_control"]["n_runs_matches_rule"] and s["negative_control"]["low_power"])
check("b gate pass", s["b_gate"]["result"] == "PASS")
# 情境 2：負控制未過（m1 與 m6 同為 -0.69）→ Δ≈0 → 不得 PASS，人工判讀
k1b = mk("k1_bad", P4, {"cond": 1000, "cfg3": 500}, branches=("cond", "cfg3"))
s = agg(k6=k6, k1=k1b)
check("neg ctrl fail + delta~0 -> no PASS", s["negative_control"]["passed"] is False and "其他組合" in s["c_main"]["status"] and s["c_main"]["pass"] is None, str(s["c_main"]["status"]))
# 情境 3：負控制未過但 Δ 夠大
k1c = mk("k1_mid", P4, {"cond": 1000, "cfg3": 820}, branches=("cond", "cfg3"))   # m1≈-0.2（超出 ±0.0953）
s = agg(k6=k6, k1=k1c)
d = s["c_main"].get("delta")
check("neg ctrl fail + large delta -> PASS via delta", d and d["pass"] and s["c_main"]["status"].startswith("PASS（以 Δ"), str(d))
# 情境 4：K=1 無效對 > floor(6/3)=2
k1d = mk("k1_inv", P4, {"cond": 1000, "cfg3": 1000}, kinds={"cfg3": lambda p: "down" if p < 3 else "v"}, branches=("cond", "cfg3"))
s = agg(k6=k6, k1=k1d)
check("K=1 invalid>2 -> 負控制未能測試, (c) 交人工", s["negative_control"]["status"].startswith("未能測試") and s["c_main"]["pass"] is None, s["c_main"]["status"])
# 情境 5：K=6 無效對過多 -> (c) 未能測試
k6bad = mk("k6bad", P6, {"cond": 1000, "uncond": 1000, "cfg1": 800, "cfg3": 500}, kinds={"cfg3": lambda p: "down" if p < 6 else "v"})
s = agg(k6=k6bad, k1=k1)
check("K=6 invalid 6/15 -> 未能測試", s["c_main"]["status"].startswith("未能測試"))
# 正控制
def pcs(tc, tu833, tu5000, **kw):
    return dict(k6c0=mk("pc_c", P4, {"cond": tc}, branches=("cond",)), u833c0=mk("pc_u833", P4, {"uncond": tu833}, branches=("uncond",)),
                k1c0=mk("pc_u5000", P4, {"cond": tu5000}, branches=("cond",)))
s = agg(**pcs(350, 300, 1000))["positive_control"]
check("pos ctrl pass", s["status"] == "通過", f"λ={s['lambda']:.2f} ci={s['lambda_ci95']} sep={s['separation_Lu5000_minus_Lu833']:.2f}")
s = agg(**pcs(900, 800, 1000))["positive_control"]
check("pos ctrl no discrimination", s["status"].startswith("控制無分辨力"), s["status"])
s = agg(**pcs(900, 300, 1000))["positive_control"]
check("pos ctrl premise fails", s["status"].startswith("n/K 時鐘前提不成立"), f"λ={s['lambda']:.2f}")
s = agg(**pcs(300, 1000, 300))["positive_control"]   # u833 比 u5000 晚 -> 方向錯
check("pos ctrl wrong direction -> 無分辨力", s["status"].startswith("控制無分辨力"))
s = agg(k6c0=mk("pc_c", P4, {"cond": 400}, branches=("cond",)), u833c0=mk("pc_u833", P4, {"uncond": 300}, kinds={"uncond": lambda p: "down" if p < 3 else "v"}, branches=("uncond",)),
        k1c0=mk("pc_u5000", P4, {"cond": 1000}, branches=("cond",)))["positive_control"]
check("pos ctrl invalid>floor(P/3) -> 未能測試", s["status"].startswith("未能測試"))
print("\nALL PASS" if not fails else f"\nFAILED: {fails}"); sys.exit(1 if fails else 0)
