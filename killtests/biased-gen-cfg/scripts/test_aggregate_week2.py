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

def mk(name, picks, tmap, kinds=None, branches=("cond", "uncond", "cfg1", "cfg3"), gamma_t=None, gamma_kind="v", gaps=None, e_dag=None):
    """gamma_t：γ 曲線的基準 t*（寫入 pair_cos_gamma）；gaps=(g_cond, g_cfg3, jitter)：e† 處 oracle 偏差間隙；e_dag：e† 的 checkpoint 索引（CE 曲線最低點）。"""
    """tmap[branch] = 基準 t*；pair 的 t* = base*exp(jit_i+jit_j)；kinds[(branch)] = 'down' 使該分支無內部最低點。"""
    pairs = [(i, j) for i in range(len(picks)) for j in range(i + 1, len(picks))]
    pc = np.zeros((len(pairs), E, 8, 3, len(branches)))
    for p, (i, j) in enumerate(pairs):
        for b, br in enumerate(branches):
            ts = tmap[br] * np.exp(JIT[picks[i]] + JIT[picks[j]]); kd = (kinds or {}).get(br, "v")
            kd = kd(p) if callable(kd) else kd
            pc[p, :, :, :, b] = curve(kd, ts)[:, None, None] * SC[None, :, None]
    path = os.path.join(a.tmp, name + ".npz"); extra = {}
    if gamma_t is not None:
        pg = np.zeros((len(pairs), E, 8, 3, 1))
        for p, (i, j) in enumerate(pairs):
            ts = gamma_t * np.exp(JIT[picks[i]] + JIT[picks[j]]); kd = gamma_kind(p) if callable(gamma_kind) else gamma_kind
            pg[p, :, :, :, 0] = curve(kd, ts)[:, None, None] * SC[None, :, None]
        extra["pair_cos_gamma"] = pg
    if gaps is not None:
        R = len(picks); B = len(branches); gc, gf, jit = gaps
        mse = np.ones((R, E, 8, R + 1, B)); ce = np.ones((R, E, 8, R + 1, B))
        for j in range(R):   # 只有 e† 處有間隙；其他 epoch 間隙為 0（若誤用會得到分母 0）
            mse[j, e_dag, :, j, branches.index("cond")] += gc * (1 + jit * (picks[j] - 2.5))
            if "cfg3" in branches: mse[j, e_dag, :, j, branches.index("cfg3")] += gf * (1 + jit * (picks[j] - 2.5))
        ce[...] = 1 + 0.1 * np.abs(np.arange(E) - e_dag)[None, :, None, None, None]
        extra.update(mse_oracle=mse, ce=ce)
    np.savez(path, epochs=EPH, t_grid=np.array(TG), branches=np.array(branches), picks=np.array(picks), pairs=np.array(pairs), pair_cos_eps=pc, git_hash="t", **extra)
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

# ================= 2026-11-14 條文 6–9（可達性、第二探針、層級用語、停點）=================
LAST = E - 1; ti = lambda S_: S_["tier"]
def agg2(**kw):
    cmd = [sys.executable, os.path.join(HERE, "aggregate_week2.py"), "--out", os.path.join(a.tmp, "v2.json")]
    for k, v in kw.items(): cmd += ["--" + k.replace("_", "-"), str(v)]
    r = subprocess.run(cmd, capture_output=True, text=True); assert r.returncode == 0, r.stderr[-1500:]
    return json.load(open(os.path.join(a.tmp, "v2.json")))
K1OK = mk("k1_ok2", P4, {"cond": 1000, "cfg3": 1000}, branches=("cond", "cfg3"), gaps=(1.0, 1.5, 0.0), e_dag=LAST)   # m1≈0（負控制通過）、Γ1 = 1.5
def k6f(gt, gk="v", gaps=(1.0, 10.0, 0.02), e=LAST, name="k6f"):   # (c) FAIL 帶：cfg3 與 cond 同 t*
    return mk(name, P6, {"cond": 1000, "uncond": 1000, "cfg1": 1000, "cfg3": 1000}, gamma_t=gt, gamma_kind=gk, gaps=gaps, e_dag=e)
# A 可達 + 探針 PASS
V = agg2(k6=k6f(500), k1=K1OK); S_ = V["sections"]
check("(c) FAIL 帶且觸發探針", S_["c_main"]["status"] == "FAIL 帶" and S_["second_probe"]["state"] == "PASS", f"G6={S_['second_probe'].get('gamma6'):.2f} G1={S_['second_probe'].get('gamma1'):.2f} ci={S_['second_probe'].get('diff_ci95')}")
check("條文 7：γ 提早一半 -> 可達", S_["reachability"]["state"] == "可達" and S_["reachability"]["h_inf"] < -0.2231, f"H={S_['reachability']['h_inf']:.3f} ci={S_['reachability']['ci95']}")
t = S_["tier"]; check("條文 9 列 1：可達+PASS", "w=3 被否證（實作／劑量層）" in t["text"] and "放大子主張有支持" in t["text"] and "此實作停損" in t["stop"] and "56 號" in t["stop"] and not t["may_propose_elimination"], t["text"])
check("停點 1 文字", "不自動送 (d)" in S_["stop_point_1"])
# B 構造不可達 + PASS
S_ = agg2(k6=k6f(950, name="k6f_b"), k1=K1OK)["sections"]
check("條文 9 列 2：構造不可達+PASS", S_["reachability"]["state"] == "構造不可達" and "對所有 w 被否證（機制層，限本文法與 n/K）" in S_["tier"]["text"] and "放大子主張有支持" in S_["tier"]["text"], S_["tier"]["text"])
# C 可達性未知（γ 無效對 > floor(15/3)）+ PASS
S_ = agg2(k6=k6f(500, gk=lambda p: "down" if p < 6 else "v", name="k6f_c"), k1=K1OK)["sections"]
check("條文 9 列 3：可達性未知（任一）", S_["reachability"]["state"] == "可達性未知" and "起點子預測「未能測試」" in S_["tier"]["text"] and "放大結果照記" in S_["tier"]["text"] and not S_["tier"]["may_propose_elimination"], S_["tier"]["text"])
check("無效對恰為 floor(P/3)=5 仍可判", agg2(k6=k6f(500, gk=lambda p: "down" if p < 5 else "v", name="k6f_c5"), k1=K1OK)["sections"]["reachability"]["state"] == "可達")
# D 探針未通過：Γ6 < 4（Γ6=3）
S_ = agg2(k6=k6f(500, gaps=(1.0, 3.0, 0.02), name="k6f_d"), k1=K1OK)["sections"]
check("條文 9 列 4：可達+未通過 -> 兩個子主張都未獲支持，可提請淘汰，閘門 4 缺 -> 無可區隔", S_["second_probe"]["state"] == "未通過" and "兩個子主張都未獲支持" in S_["tier"]["text"] and S_["tier"]["may_propose_elimination"] and "無可區隔的切入點" in S_["tier"]["elimination_reason"], str(S_["tier"]))
# D2 Γ6 >= 4 但與 K=1 無差別 -> 未通過
K1same = mk("k1_same", P4, {"cond": 1000, "cfg3": 1000}, branches=("cond", "cfg3"), gaps=(1.0, 10.0, 0.02), e_dag=LAST)
S_ = agg2(k6=k6f(950, name="k6f_d2"), k1=K1same)["sections"]
check("Γ6=10=Γ1：CI 含 0 -> 未通過", S_["second_probe"]["state"] == "未通過" and S_["second_probe"]["gamma6"] >= 4 and S_["second_probe"]["diff_ci95"][0] <= 0, str(S_["second_probe"].get("diff_ci95")))
# D3 閘門 4 通過時淘汰理由不含「無可區隔」
V = agg2(k6=k6f(500, gaps=(1.0, 3.0, 0.02), name="k6f_d3"), k1=K1OK, **pcs(350, 300, 1000)); S_ = V["sections"]
check("閘門 4 通過 -> 淘汰理由不寫無可區隔", S_["positive_control"]["status"] == "通過" and S_["tier"]["may_propose_elimination"] and "無可區隔" not in S_["tier"]["elimination_reason"])
# E 未能測試：e† 落在第一格 / 分母 <= 0 / K=1 缺
S_ = agg2(k6=k6f(500, e=0, name="k6f_e0"), k1=mk("k1_e0", P4, {"cond": 1000, "cfg3": 1000}, branches=("cond", "cfg3"), gaps=(1.0, 1.5, 0.0), e_dag=0))["sections"]
check("e† 落在第一格 -> 未能測試；層級『放大子主張未測』", S_["second_probe"]["state"] == "未能測試" and "第一格" in S_["second_probe"]["reason"] and "放大子主張未測" in S_["tier"]["text"], S_["tier"]["text"])
S_ = agg2(k6=k6f(500, gaps=(-1.0, 10.0, 0.0), name="k6f_neg"), k1=K1OK)["sections"]
check("點估計分母 <= 0 -> 未能測試", S_["second_probe"]["state"] == "未能測試", str(S_["second_probe"].get("reason")))
S_ = agg2(k6=k6f(500, name="k6f_nok1"))["sections"]
check("缺 K=1 -> (c) 不判 FAIL 帶，探針未觸發（不是 PASS）", S_["second_probe"]["state"] == "未觸發" and "負控制未能測試" in S_["c_main"]["status"])
check("max_invalid = floor(P/3)：P=15->5, P=6->2, P=3->1", all(P_ // 3 == v for P_, v in ((15, 5), (6, 2), (3, 1))))
# e† 只用 K=6 的 CE：K=1 的 CE 最低點不同也不影響（K=1 用同一個 e†）
K1other = mk("k1_other", P4, {"cond": 1000, "cfg3": 1000}, branches=("cond", "cfg3"), gaps=(1.0, 1.5, 0.0), e_dag=LAST)
z = dict(np.load(K1other)); z["ce"] = np.ones_like(z["ce"]) * (1 + 0.1 * np.abs(np.arange(E) - 3))[None, :, None, None, None]; np.savez(K1other, **z)
check("K=1 的 CE 曲線不影響 e†", agg2(k6=k6f(500, name="k6f_f"), k1=K1other)["sections"]["second_probe"]["state"] == "PASS")
# F (c) PASS 但構造不可達 -> 加註「與上界矛盾」；探針不觸發
V = agg2(k6=mk("k6_pass_unreach", P6, {"cond": 1000, "uncond": 1000, "cfg1": 800, "cfg3": 500}, gamma_t=950), k1=K1OK); S_ = V["sections"]
check("(c) PASS + 構造不可達 -> 與上界矛盾、列入人工審查", S_["c_main"]["status"] == "PASS" and "與上界矛盾" in S_["c_main"]["note"] and V.get("manual_review") and S_["second_probe"]["state"] == "未觸發" and "不適用" in S_["tier"]["note"], str(V.get("manual_review")))
check("可達性只描述：PASS 不被改寫", S_["c_main"]["pass"] is True)
# G 停點 2
V = agg2(k6=k6f(500, name="k6f_g"), k1=K1OK, gpu_used=70, gpu_projected_d=50)
check("停點 2：70+50 > 115 -> 停下回報", "先停下回報" in V["sections"]["stop_point_2"]["action"])
V = agg2(k6=k6f(500, name="k6f_g2"), k1=K1OK, gpu_used=70, gpu_projected_d=45)
check("停點 2：70+45 = 115 -> 未超過", "未超過" in V["sections"]["stop_point_2"]["action"])
# H K=6 npz 沒有 γ -> 可達性未知（不 crash）
S_ = agg2(k6=mk("k6_nogamma", P6, {"cond": 1000, "uncond": 1000, "cfg1": 1000, "cfg3": 1000}, gaps=(1.0, 10.0, 0.02), e_dag=LAST), k1=K1OK)["sections"]
check("缺 γ -> 可達性未知、層級用語仍可輸出", S_["reachability"]["state"] == "可達性未知" and "未能測試" in S_["tier"]["text"])
# t 遮罩與全 t 網格：全 t 版只描述
S_ = agg2(k6=k6f(500, name="k6f_t"), k1=K1OK)["sections"]["second_probe"]
check("全 t 網格版 Γ 只描述且存在", S_["gamma6_all_t_descriptive"] is not None and S_["t_used"] == [50, 75, 100, 150, 200, 300])


# reach.probe_amplification 的跳過規則（直接測函式）
import reach
def terms(c0, nrun=6, nt=6, cfg=10.0):
    c = np.ones((nrun, nt)); c[0] = c0; g = np.full((nrun, nt), cfg); return g, c
g6, c6 = terms(-2.0); g1, c1 = terms(1.0, cfg=1.5)
o = reach.probe_amplification(g6, c6, list(range(6)), g1[:4], c1[:4], list(range(4)), 4.0, 2000, 0, 0.05)
check("分母 <= 0 的重抽比例 > 5% -> 未能測試", o["state"] == "未能測試" and o.get("bootstrap_skipped_fraction", 0) > 0.05, str(o.get("bootstrap_skipped_fraction")))
g6, c6 = terms(-0.5)
o = reach.probe_amplification(g6, c6, list(range(6)), g1[:4], c1[:4], list(range(4)), 4.0, 2000, 0, 0.05)
check("跳過比例 <= 5% -> 照判（回報跳過比例）", o["state"] in ("PASS", "未通過") and 0 < o["bootstrap_skipped_fraction"] <= 0.05, f"{o['state']} skipped={o.get('bootstrap_skipped_fraction')}")
g6, c6 = terms(-2.0, nrun=6); c6[1:] = 0.4
o = reach.probe_amplification(g6, c6, list(range(6)), g1[:4], c1[:4], list(range(4)), 4.0, 2000, 0, 0.05)
check("點估計分母 <= 0 -> 未能測試", o["state"] == "未能測試" and o["denominators"]["k6"] <= 0, str(o["denominators"]))
g6, c6 = terms(1.0); g1b, c1b = terms(1.0, cfg=-1.0)
o = reach.probe_amplification(g6, c6, list(range(6)), g1b[:4], c1b[:4], list(range(4)), 4.0, 2000, 0, 0.05)
check("Γ1 <= 0（log 無定義）-> 未能測試", o["state"] == "未能測試")
# 條文 9 表（直接測 tier_text 全 3x3 組合）
for rs in ("可達", "構造不可達", "可達性未知"):
    for ps in ("PASS", "未通過", "未能測試"):
        tt = reach.tier_text(rs, ps, True)
        okk = (tt["may_propose_elimination"] == (rs != "可達性未知" and ps == "未通過")) and ((tt["stop"] is not None) == (rs != "可達性未知" and ps == "PASS"))
        check(f"tier_text({rs},{ps})", okk, tt["text"])

print("\nALL PASS" if not fails else f"\nFAILED: {fails}"); sys.exit(1 if fails else 0)
