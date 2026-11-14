#!/usr/bin/env python
"""aggregate_week1.py 的合成資料測試（DECISIONS_fable 條文 1、2、5）。用法：python test_aggregate_week1.py [--tmp DIR]"""
import argparse, json, os, subprocess, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
from schedule import ckpt_schedule
ap = argparse.ArgumentParser(); ap.add_argument("--tmp", default=os.path.join(os.environ.get("TMPDIR", "."), "agg_test")); a = ap.parse_args()
os.makedirs(a.tmp, exist_ok=True)
EPH = np.array(ckpt_schedule(30, 20000)); E = len(EPH)
TG = [10, 25, 50, 75, 100, 150, 200, 300]
SCALE = np.array([1e-4, 3e-3, 1e-2, 5e-3, 2e-3, 8e-4, 3e-4, 4e-5])   # 各 t 量級差很大（模擬 ε 空間 1-cos）
rng0 = np.random.default_rng(1)

def vcurve(tstar, kind="v"):
    le = np.log(EPH / tstar)
    if kind == "v": return 0.2 + 0.1 * np.abs(le)                 # 內部最低點、回升明顯
    if kind == "flatrise": return 1 + 0.02 * np.abs(le)             # 最低點在內部但回升 < 10%
    raise ValueError

def mk_curve(kind, tstar):
    if kind == "down": return np.linspace(2, 1, E)        # 最低點在最後一格
    if kind == "up": return np.linspace(1, 2, E)          # 最低點在第一格
    return vcurve(tstar, kind)

def hier_npz(path, R, spec):
    """spec(p, branch) -> (kind, tstar)"""
    pairs = [(i, j) for i in range(R) for j in range(i + 1, R)]; P = len(pairs)
    br = ["cond", "uncond", "cfg1", "cfg3"]
    pc = np.zeros((P, E, 8, 3, 4))
    for p in range(P):
        for b, name in enumerate(br):
            kind, ts = spec(p, name)
            cv = mk_curve(kind, ts)
            noise = 1 + 0.003 * rng0.standard_normal((E, 8, 3))
            pc[p, :, :, :, b] = cv[:, None, None] * SCALE[None, :, None] * noise
    np.savez(path, epochs=EPH, t_grid=np.array(TG), branches=np.array(br), picks=np.arange(R), pairs=np.array(pairs),
             mse_oracle=np.ones((R, E, 8, 4, 4)), kl_oracle=np.ones((R, E, 8, 4, 4)), ce=np.ones((R, E, 8, 4, 4)), ce_bp=np.ones((4, 8, 2)),
             pair_mse=np.ones((P, E, 8, 3, 4)), pair_cos=np.full((P, E, 8, 3, 4), np.nan),   # x0-hat pair_cos 故意放 NaN：aggregate 不得使用
             pair_cos_eps=pc, git_hash="test", guidance=np.array([1., 3.]))

def run(args):
    r = subprocess.run([sys.executable, os.path.join(HERE, "aggregate_week1.py"), "--out", os.path.join(a.tmp, "v.json")] + args,
                       capture_output=True, text=True, cwd=a.tmp)
    assert r.returncode == 0, r.stderr[-2000:]
    return json.load(open(os.path.join(a.tmp, "v.json")))

NOBP = ["--bp", os.path.join(ROOT, "results", "bp_unit_test.json"), "--celeba", os.path.join(a.tmp, "none.npz")]
fails = []
def check(name, cond, extra=""):
    print(("OK   " if cond else "FAIL ") + name, extra)
    if not cond: fails.append(name)

def H(R, spec, name):
    pth = os.path.join(a.tmp, name + ".npz"); hier_npz(pth, R, spec); return run(NOBP + ["--hier", pth])

# ---- (c) PASS：cfg3 比 cond 早約 50%（15 對）
v = H(6, lambda p, b: ("v", 500 if b == "cfg3" else 1000), "pass6")
i = v["info"]["hier"]; m = i["main_metric_log_tstar_ratio"]
check("c PASS (15 pairs)", i["c_pass_criteria_met"] is True and i["c_fail_band_met"] is False and m["n_invalid_pairs"] == 0, f"mean={m['mean']:.3f} ci={m['ci95']}")
check("t mask keeps 50..300", i["t_mask_keep"] == [50, 75, 100, 150, 200, 300])
check("b gate pass 15", v["gates"]["G3b_hier_cond_interior_min"] == "PASS")
# ---- (c) FAIL 帶：兩者相同
v = H(6, lambda p, b: ("v", 1000), "fail6"); i = v["info"]["hier"]
check("c FAIL band", i["c_pass_criteria_met"] is False and i["c_fail_band_met"] is True, str(i["main_metric_log_tstar_ratio"]["ci95"]))
# ---- 有內部最低點 vs 無（cfg3 單調下降：最後一格）；5 對無效（等於上限）→ 仍可判，剔除無效對
bad = {0, 3, 7, 11, 13}
v = H(6, lambda p, b: ("down", 0) if (b == "cfg3" and p in bad) else ("v", 500 if b == "cfg3" else 1000), "inv5")
i = v["info"]["hier"]; m = i["main_metric_log_tstar_ratio"]
check("5/15 invalid -> testable, PASS using valid pairs", i["c_status"] == "可判讀" and m["n_invalid_pairs"] == 5 and i["c_pass_criteria_met"] is True)
check("invalid reason recorded (last, cfg3)", all(i["invalid_pairs"][p]["reason"] == {"cfg3": "last"} for p in bad))
# ---- 無效對過多（6/15 > 5）→ 未能測試，不得 PASS/FAIL
v = H(6, lambda p, b: ("down", 0) if (b == "cfg3" and p < 6) else ("v", 500 if b == "cfg3" else 1000), "inv6")
i = v["info"]["hier"]
check("6/15 invalid -> 未能測試", i["c_status"].startswith("未能測試") and i["c_pass_criteria_met"] is None and i["c_fail_band_met"] is None)
# ---- 回升不足 / first
v = H(6, lambda p, b: ("flatrise", 1000) if (b == "cond" and p < 7) else (("up", 0) if (b == "cond" and p == 7) else ("v", 1000)), "rise")
i = v["info"]["hier"]; rs = {tuple(x["reason"].items()) for x in i["invalid_pairs"] if not x["valid"]}
check("reasons insufficient_rise and first", rs == {(("cond", "insufficient_rise"),), (("cond", "first"),)}, str(rs))
check("(b) gate: 8/15 cond interior <10 -> 未能測試", v["gates"]["G3b_hier_cond_interior_min"].startswith("未能測試"), v["gates"]["G3b_hier_cond_interior_min"])
# ---- (b) 閘門 3 對：2/3 通過，1/3 未能測試
v = H(3, lambda p, b: ("down", 0) if (b == "cond" and p == 0) else ("v", 1000), "b2of3")
check("(b) gate 2/3 pass", v["gates"]["G3b_hier_cond_interior_min"] == "PASS")
v = H(3, lambda p, b: ("v", 1000) if (b == "cond" and p == 0) else (("down", 0) if b == "cond" else ("v", 1000)), "b1of3")
check("(b) gate 1/3 -> 未能測試 (not FAIL)", v["gates"]["G3b_hier_cond_interior_min"].startswith("未能測試") and not v["overall"].startswith("WEEK1: FAIL"), v["overall"])
v = H(3, lambda p, b: ("down", 0) if (b == "cfg3" and p == 0) else ("v", 500 if b == "cfg3" else 1000), "inv1of3")
check("P=3: 1 invalid == floor(3/3) -> (c) 可判讀", v["info"]["hier"]["c_status"] == "可判讀")
v = H(3, lambda p, b: ("down", 0) if (b == "cfg3" and p < 2) else ("v", 500 if b == "cfg3" else 1000), "inv2of3")
check("P=3: 2 invalid > floor(3/3)=1 -> (c) 未能測試", v["info"]["hier"]["c_status"].startswith("未能測試"))
v = H(3, lambda p, b: ("v", 500 if b == "cfg3" else 1000), "pass3")
check("P=3 exploratory note", "探索性" in v["info"]["hier"]["status_note"])
# ---- 逐 t 正規化：x0-hat pair_cos 為 NaN 不影響（上面皆通過即證明）；bootstrap 回報跳過次數
check("bootstrap skipped reported", "bootstrap_skipped" in m)

# ---- (a) CelebA
EC = np.array(ckpt_schedule(10, 5000)); EN = len(EC); TC = [50, 100, 150]
def celeba(M, bias_t, loss_t, loss_kind="v"):
    loss = np.zeros((M, EN)); sim = np.zeros((M, M, EN, 3))
    for i in range(M):
        l = np.log(EC / loss_t[i])
        loss[i] = {"v": 0.1 + 0.05 * np.abs(l), "down": np.linspace(2, 1, EN), "up": np.linspace(1, 2, EN)}[loss_kind if isinstance(loss_kind, str) else loss_kind[i]]
        for j in range(M):
            bt = bias_t[i]
            if bt is None: d = np.linspace(1, 2, EN) * 0.01          # 單調：最低點在第一格
            else: d = 0.01 * (0.2 + 0.1 * np.abs(np.log(EC / bt)))
            sim[i, j] = (1 - d)[:, None]
    pth = os.path.join(a.tmp, "ce.npz"); np.savez(pth, epochs=EC, times=np.array(TC), test_loss=loss, sim_train=sim, sim_test=sim); return pth
def runa(pth): return run(["--bp", os.path.join(ROOT, "results", "bp_unit_test.json"), "--celeba", pth, "--hier", os.path.join(a.tmp, "none.npz")])
v = runa(celeba(6, [100] * 6, [400] * 6)); c = v["info"]["celeba"]
check("(a) all valid, log ratio = log 4", c["n_valid_models"] == 6 and abs(c["log_ratio_mean"] - np.log(4)) < 0.3, f"{c['log_ratio_mean']:.3f}")
lt = [400 * (1 + 0.05 * k) for k in range(6)]; bt = [100 * (1 + 0.04 * (k % 3)) for k in range(6)]
v = runa(celeba(6, bt, lt)); check("(a) PASS", v["gates"]["G2_celeba_reproduction"] == "PASS", str(v["info"]["celeba"]["log_ratio_mean"]))
# 右設限：1 個模型 test loss 最低點在最後一格 → 仍有效
v = runa(celeba(6, bt, lt, ["v"] * 5 + ["down"])); c = v["info"]["celeba"]
check("(a) right-censored still valid, PASS", c["n_valid_models"] == 6 and c["right_censored_models"] == [5] and v["gates"]["G2_celeba_reproduction"] == "PASS")
# 1 個無效（偏差曲線無內部最低點）→ 有效 5，仍判定
bt1 = list(bt); bt1[2] = None
v = runa(celeba(6, bt1, lt)); c = v["info"]["celeba"]
check("(a) 1 invalid model -> valid=5 -> still judged", c["n_valid_models"] == 5 and c["n_invalid_models"] == 1 and v["gates"]["G2_celeba_reproduction"] == "PASS", str(c["bias_invalid_reason"]))
bt2 = list(bt1); bt2[4] = None
v = runa(celeba(6, bt2, lt)); check("(a) 2 invalid -> 未能測試", v["gates"]["G2_celeba_reproduction"].startswith("未能測試") and v["overall"] == "WEEK1: 未能測試")
v = runa(celeba(6, bt, lt, ["v"] * 4 + ["up", "up"])); check("(a) test loss min first-edge x2 -> 未能測試", v["gates"]["G2_celeba_reproduction"].startswith("未能測試"))
v = runa(celeba(6, bt, lt)); cb = v["info"]["celeba"]["model_bootstrap_descriptive"]
check("(a) descriptive model bootstrap reported", cb["n_used"] > 1900 and cb["n_skipped"] + cb["n_used"] == 2000, str(cb))
print("\nALL PASS" if not fails else f"\nFAILED: {fails}"); sys.exit(1 if fails else 0)
