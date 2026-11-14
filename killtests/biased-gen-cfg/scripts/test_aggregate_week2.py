#!/usr/bin/env python
"""aggregate_week2.py 的合成資料測試。用法：python test_aggregate_week2.py [--tmp DIR]"""
import argparse, hashlib, json, os, subprocess, sys
from pathlib import Path
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from schedule import ckpt_schedule
from week2_identity import expected_eval, expected_training, is_legacy, read_runs, sha256
ap = argparse.ArgumentParser(); ap.add_argument("--tmp", default=os.path.join(os.environ.get("TMPDIR", "."), "agg2_test")); a = ap.parse_args()
os.makedirs(a.tmp, exist_ok=True)
EPH = np.array(ckpt_schedule(30, 20000)); E = len(EPH)
TG = [10, 25, 50, 75, 100, 150, 200, 300]; SC = np.array([1e-4, 3e-3, 1e-2, 5e-3, 2e-3, 8e-4, 3e-4, 4e-5])
rng = np.random.default_rng(3)
JIT = rng.normal(0, 0.06, 6)   # 每個區塊（模型）共同的 t* 偏移，使 CI 有寬度
DATA_SHA = json.loads((Path(HERE).parent / "configs/t_mask_hier.json").read_text())["data_sha256"]

def curve(kind, ts):
    le = np.log(EPH / ts)
    return {"v": 0.2 + 0.1 * np.abs(le), "down": np.linspace(2, 1, E)}[kind]

def mk(name, picks, tmap, kinds=None, branches=("cond", "uncond", "cfg1", "cfg3"), gamma_t=None, gamma_kind="v", gaps=None, e_dag=None, role=None):
    """gamma_t：γ 曲線的基準 t*（寫入 pair_cos_gamma）；gaps=(g_cond, g_cfg3, jitter)：e† 處 oracle 偏差間隙；e_dag：e† 的 checkpoint 索引（CE 曲線最低點）。"""
    """tmap[branch] = 基準 t*；pair 的 t* = base*exp(jit_i+jit_j)；kinds[(branch)] = 'down' 使該分支無內部最低點。"""
    if role is None:
        role = {("cond", "cfg3"): "k1", ("cond",): "k6c0", ("uncond",): "u833c0"}.get(branches, "k6")
    ev = expected_eval(role)
    tmap = dict(tmap)
    branches = ["uncond"] if ev["run_type"] == "uncond_only" else ["cond", "uncond", *[f"cfg{w:g}" for w in ev["guidance"]]]
    for br in branches:
        tmap.setdefault(br, next(iter(tmap.values())))
    pairs = [(i, j) for i in range(len(picks)) for j in range(i + 1, len(picks))]
    pc = np.zeros((len(pairs), E, 8, 3, len(branches)))
    for p, (i, j) in enumerate(pairs):
        for b, br in enumerate(branches):
            ts = tmap[br] * np.exp(JIT[picks[i]] + JIT[picks[j]]); kd = (kinds or {}).get(br, "v")
            kd = kd(p) if callable(kd) else kd
            pc[p, :, :, :, b] = curve(kd, ts)[:, None, None] * SC[None, :, None]
    path = os.path.join(a.tmp, name + ".npz")
    runs = []
    for pick in picks:
        kind = ev["run_type"]
        train = expected_training(kind, pick)
        if kind in ("const", "uncond_only"):
            train.update(block_size=5000, n_c=5000 if kind == "const" else 850, steps_per_epoch=10)
        meta_sha = hashlib.sha256(f"{kind}:{pick}".encode()).hexdigest()
        receipt = {"version": 1, "data_sha256": DATA_SHA, "meta_sha256": meta_sha, "epochs": EPH.tolist()}
        runs.append({"training": train, "meta_sha256": meta_sha, "completion": None if is_legacy(kind, pick) else receipt})
    manifest = {"version": 1, "evaluation": ev, "data_sha256": DATA_SHA, "runs": runs}
    extra = {"week2_manifest": np.array(json.dumps(manifest)), "n_eval": np.array(ev["n_eval"]),
             "test_block": np.array(ev["test_block"]), "guidance": np.array(ev["guidance"])}
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
            scale = 0.05 * (1 + jit * (picks[j] - 2.5))
            mse[j, e_dag, :, j, branches.index("cond")] -= gc * scale
            if "cfg3" in branches:
                mse[j, e_dag, :, j, branches.index("cfg3")] -= gf * scale
        ce[...] = 1 + 0.1 * np.abs(np.arange(E) - e_dag)[None, :, None, None, None]
        extra.update(mse_oracle=mse, ce=ce)
    np.savez(path, epochs=EPH, t_grid=np.array(TG), branches=np.array(branches), picks=np.array(picks), pairs=np.array(pairs), pair_cos_eps=pc, git_hash="t", **extra)
    return path

W1 = os.path.join(a.tmp, "w1.json")
with open(W1, "w") as fh:
    json.dump({"overall": "WEEK1: PASS（合成 fixture）", "gates": {"G2_celeba_reproduction": "PASS"}}, fh)


def agg(**kw):
    kw.setdefault("week1_verdict", W1)
    outfile = os.path.join(a.tmp, "v.json")
    cmd = [sys.executable, os.path.join(HERE, "aggregate_week2.py"), "--out", outfile]
    for k, v in kw.items():
        cmd += ["--" + k.replace("_", "-"), str(v)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-1500:]
    with open(outfile) as fh:
        return json.load(fh)["sections"]


def rewrite(source, name, mutate):
    with np.load(source) as source_npz:
        data = dict(source_npz)
    mutate(data)
    path = os.path.join(a.tmp, name + ".npz")
    np.savez(path, **data)
    return path


def change_manifest(data, mutate):
    manifest = json.loads(data["week2_manifest"].item())
    mutate(manifest)
    data["week2_manifest"] = np.array(json.dumps(manifest))

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
d = s["report60"]["signal"]["S_a"]
check("neg ctrl fail + large delta -> PASS via delta", d["state"] == "PASS" and d["metric"] == "matched_delta", d)
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
                k1c0=mk("pc_u5000", P4, {"cond": tu5000}, branches=("cond",), role="k1c0"))
s = agg(**pcs(350, 300, 1000))["positive_control"]
check("pos ctrl pass", s["status"] == "通過", f"λ={s['lambda']:.2f} ci={s['lambda_ci95']} sep={s['separation_Lu5000_minus_Lu833']:.2f}")
s = agg(**pcs(900, 800, 1000))["positive_control"]
check("pos ctrl no discrimination", s["status"].startswith("控制無分辨力"), s["status"])
s = agg(**pcs(900, 300, 1000))["positive_control"]
check("pos ctrl premise fails", s["status"].startswith("n/K 時鐘前提不成立"), f"λ={s['lambda']:.2f}")
s = agg(**pcs(300, 1000, 300))["positive_control"]   # u833 比 u5000 晚 -> 方向錯
check("pos ctrl wrong direction -> 無分辨力", s["status"].startswith("控制無分辨力"))
s = agg(k6c0=mk("pc_c", P4, {"cond": 400}, branches=("cond",)), u833c0=mk("pc_u833", P4, {"uncond": 300}, kinds={"uncond": lambda p: "down" if p < 3 else "v"}, branches=("uncond",)),
        k1c0=mk("pc_u5000", P4, {"cond": 1000}, branches=("cond",), role="k1c0"))["positive_control"]
check("pos ctrl invalid>floor(P/3) -> 未能測試", s["status"].startswith("未能測試"))
disjoint = []
for group, branch, onset, start in (("k6c0", "cond", 350, 0), ("u833c0", "uncond", 300, 5), ("k1c0", "cond", 1000, 10)):
    kinds = {branch: lambda p, start=start: "down" if start <= p < start + 5 else "v"}
    disjoint.append(mk("disjoint_" + group, P6, {branch: onset}, kinds=kinds, branches=(branch,), role=group))
s = agg(k6c0=disjoint[0], u833c0=disjoint[1], k1c0=disjoint[2])["positive_control"]
check("正控制各用自身的 10/15 有效對，共用 bootstrap 區塊",
      s["status"] == "通過" and set(s["n_valid_pairs"].values()) == {10} and s["bootstrap_skipped"] > 0)

# 報告 60：現象、兩條訊號路徑、G3、入池與 CIFAR 預算。
LAST = E - 1
K1OK = mk("k1_ok2", P4, {"cond": 1000, "cfg3": 1000}, branches=("cond", "cfg3"), gaps=(1.0, 1.5, 0.0), e_dag=LAST)


def k6f(name, *, cfg_t=1000, gt=500, gk="v", gaps=(1.0, 10.0, 0.02), e=LAST, kinds=None):
    return mk(name, P6, {"cond": 1000, "uncond": 1000, "cfg1": 1000, "cfg3": cfg_t}, kinds=kinds,
              gamma_t=gt, gamma_kind=gk, gaps=gaps, e_dag=e)


def k1f(name, *, cfg_t=1000, gaps=(1.0, 1.5, 0.0), e=LAST, kinds=None):
    return mk(name, P4, {"cond": 1000, "cfg3": cfg_t}, kinds=kinds,
              branches=("cond", "cfg3"), gaps=gaps, e_dag=e)


AMP = k6f("amp")
sections = agg(k6=AMP, k1=K1OK)
formal = sections["report60"]
probe = sections["second_probe"]
check("S-b PASS 即使 (c) 落在舊 FAIL 帶", sections["c_main"]["use"] == "descriptive_only"
      and sections["c_main"]["status"] == "FAIL 帶" and formal["signal"]["passed_routes"] == ["S_b"])
check("Γ6=10，Γ1=1.5，使用 97.5% 區間", np.isclose(probe["gamma6"], 10.0)
      and np.isclose(probe["gamma1"], 1.5) and probe["diff_ci_level"] == 0.975)
check("空間用 raw CFG3 G3，正值代表自身訓練誤差更低", formal["headroom"]["measure"] == "CFG3 membership gap proxy"
      and np.isclose(formal["headroom"]["g3"], 0.5) and formal["headroom"]["state"] == "PASS")
check("三軸 PASS 進候選池", formal["candidate_pool"]["state"] == "PASS")
check("H∞ 提前達標仍為描述", sections["asymptotic_onset"]["state"] == "極限提前達標" and "停損" not in sections["tier"]["text"])
check("預算缺失時先確認", "先確認" in formal["cifar_d"]["action"])
check("70+50 > 115 先停下回報", "先停下回報" in agg(k6=AMP, k1=K1OK, gpu_used=70, gpu_projected_d=50)["report60"]["cifar_d"]["action"])
check("70+45 = 115 仍須先入池", "使用者決定入池後可做" in agg(k6=AMP, k1=K1OK, gpu_used=70, gpu_projected_d=45)["report60"]["cifar_d"]["action"])

for name, gamma_t, gamma_kind, state in (
    ("late_limit", 950, "v", "極限提前未達標"),
    ("bad_limit", 500, lambda p: "down" if p < 6 else "v", "極限曲線未能判讀"),
    ("five_invalid", 500, lambda p: "down" if p < 5 else "v", "極限提前達標"),
    ("missing_limit", None, "v", "極限曲線未能判讀"),
):
    data = agg(k6=k6f(name, gt=gamma_t, gk=gamma_kind), k1=K1OK)
    check(name + " does not change pool outcome", data["asymptotic_onset"]["state"] == state
          and data["report60"]["candidate_pool"]["state"] == "PASS")

onset_only = agg(k6=k6f("onset_only", cfg_t=500, gaps=(1, 3, 0)), k1=K1OK)["report60"]
check("S-a 獨立通過，同時固定計算 Γ", onset_only["signal"]["passed_routes"] == ["S_a"]
      and onset_only["signal"]["S_a"]["ci_level"] == 0.975
      and np.isclose(onset_only["signal"]["S_b"]["gamma6"], 3))
check("S-a PASS 時，S-b 未能測試也可通過訊號", agg(k6=k6f("pass_untested", cfg_t=500, gaps=(-1, 10, 0)),
      k1=K1OK)["report60"]["signal"]["passed_routes"] == ["S_a"])
overlap = agg(k6=k6f("overlap"), k1=k1f("near_gamma", gaps=(1, 9, 0)))["report60"]["signal"]
check("S-b PASS 優先於重疊的聯合弱訊號門檻", overlap["passed_routes"] == ["S_b"] and overlap["S_a"]["m6_equivalent"]
      and overlap["S_b"]["diff_ci"][1] <= 0.2231)

weak = k6f("weak_signal", gaps=(2.0, 3.0, 0.0))
formal = agg(k6=weak, k1=K1OK, gpu_used=1, gpu_projected_d=1)["report60"]
check("兩條路徑均有效且符合弱區間 -> 訊號 FAIL", formal["signal"]["state"] == "FAIL"
      and formal["candidate_pool"]["failed_axes"] == ["signal"] and "暫不做" in formal["cifar_d"]["action"])
formal = agg(k6=weak, k1=K1OK, **pcs(350, 300, 1000))["report60"]
check("沒有 S-a 或 S-b PASS、只有 n/K 通過 -> 灰區", formal["signal"]["state"] == "灰區"
      and formal["candidate_pool"]["state"] == "灰區")
formal = agg(k6=weak, k1=k1f("non_null_delta", cfg_t=500))["report60"]
check("K=1 非真空，原 m6 等價但 Δ 不等價 -> 灰區", formal["signal"]["state"] == "灰區"
      and formal["signal"]["S_a"]["m6_equivalent"] and not formal["signal"]["S_a"]["onset_equivalent"])

bad_mask = {"cfg3": lambda p: "down" if p in (0, 1) else "v"}
other_mask = {"cfg3": lambda p: "down" if p in (2, 3) else "v"}
few_pairs = agg(k6=k6f("pairs_k6", gaps=(2, 3, 0), kinds=bad_mask), k1=k1f("pairs_k1", cfg_t=500, kinds=other_mask))["report60"]["signal"]
check("兩組各有足夠有效對，但共同有效對不足 -> 單列原因", few_pairs["S_a"]["delta"]["n_common_valid_pairs"] == 2
      and few_pairs["S_a"]["delta"]["required_valid_pairs"] == 4 and few_pairs["S_a"]["reason"] == "共同有效配對不足"
      and few_pairs["state"] == "未能測試")

first = agg(k6=k6f("first_cp", e=0), k1=K1OK)["report60"]
check("e† 落在第一格 -> S-b 與 G3 均未能測試", all("第一格" in r["reason"] for r in (first["signal"]["S_b"], first["headroom"])))
neg_g = agg(k6=k6f("negative_denominator", gaps=(-1, 10, 0)), k1=K1OK)["report60"]
check("Γ 分母非正 -> S-b 未能測試，G3 仍可判", neg_g["signal"]["S_b"]["state"] == "未能測試" and neg_g["headroom"]["state"] == "PASS")
missing = agg(k6=AMP)["report60"]
check("K=1 缺失照報 Γ6 和 G3，訊號未能測試", np.isclose(missing["signal"]["S_b"]["gamma6"], 10)
      and missing["signal"]["state"] == "未能測試" and missing["headroom"]["state"] == "PASS")
check("缺 Week1 結果，現象未能測試", agg(k6=AMP, k1=K1OK, week1_verdict=os.path.join(a.tmp, "absent.json"))["report60"]["phenomenon"]["state"] == "未能測試")
w1bad = os.path.join(a.tmp, "w1bad.json")
with open(w1bad, "w") as fh:
    json.dump({"overall": "WEEK1: 未能測試（探索性階層未過）", "gates": {"G2_celeba_reproduction": "PASS"}}, fh)
check("第 1 週 G2 PASS、探索性整體未能測試，現象可由第 2 週通過",
      agg(k6=AMP, k1=K1OK, week1_verdict=w1bad)["report60"]["phenomenon"]["state"] == "PASS")
with open(w1bad, "w") as fh:
    json.dump({"overall": "WEEK1: 未能測試", "gates": {"G2_celeba_reproduction": "未能測試"}}, fh)
check("第 1 週 G2 未能測試，現象未能測試",
      agg(k6=AMP, k1=K1OK, week1_verdict=w1bad)["report60"]["phenomenon"]["state"] == "未能測試")
short = mk("short_panel", P4, {"cond": 1000, "cfg3": 500}, branches=("cond", "cfg3"))
check("4-run 6-pair panel 不能通過 6-run 15-pair (b) 閘門", agg(k6=short, k1=K1OK)["report60"]["phenomenon"]["state"] == "未能測試")
perm = mk("perm_panel", [5, 3, 4, 1, 0, 2], {"cond": 1000, "cfg3": 500}, branches=("cond", "cfg3"), role="k6")
perm_delta = agg(k6=perm, k1=k1c)["report60"]
check("同一組 pick 換序仍有效，配對 Δ 依 pick ID 對齊", perm_delta["phenomenon"]["state"] == "PASS"
      and perm_delta["signal"]["S_a"]["state"] == "PASS" and np.isclose(perm_delta["signal"]["S_a"]["mean"], d["mean"]))
limited = k6f("few_cond_onsets", kinds={"cond": lambda p: "down" if p < 6 else "v"})
check("(b) 9/15 未能測試", agg(k6=limited, k1=K1OK)["report60"]["phenomenon"]["state"] == "未能測試")
limited = k6f("ten_cond_onsets", kinds={"cond": lambda p: "down" if p < 5 else "v"})
check("(b) 10/15 通過", agg(k6=limited, k1=K1OK)["report60"]["phenomenon"]["state"] == "PASS")

for label, axis, drop in (("terminal", 1, LAST), ("interior", 1, 25), ("unmasked_t", 2, 0), ("retained_t", 2, 2)):
    with np.load(AMP) as f:
        z = dict(f)
    grid = "epochs" if axis == 1 else "t_grid"
    z[grid] = np.delete(z[grid], drop)
    for key in ("pair_cos_eps", "pair_cos_gamma", "mse_oracle", "ce"):
        z[key] = np.delete(z[key], drop, axis=axis)
    path = os.path.join(a.tmp, "incomplete_" + label + ".npz")
    np.savez(path, **z)
    incomplete = agg(k6=path, k1=K1OK)["report60"]
    check("不完整網格禁止正式判讀 " + label, all(incomplete[k]["state"] == "未能測試"
          for k in ("phenomenon", "signal", "headroom", "candidate_pool")))

# 在載入資料或提交 CUDA 計算前，正式 evaluator 要點名少哪份 checkpoint。
run_dir = os.path.join(a.tmp, "fake_run")
os.makedirs(run_dir, exist_ok=True)
for epoch in EPH:
    if epoch != EPH[25] and epoch != EPH[-1]:
        open(os.path.join(run_dir, f"ep{epoch}.pt"), "wb").close()
cmd = [sys.executable, os.path.join(HERE, "eval_hier_oracle.py"), "--require-full-week2", "--runs", run_dir,
       "--data", os.path.join(a.tmp, "intentionally_absent.npy"), "--out", os.path.join(a.tmp, "no_output.npz")]
evaluation = subprocess.run(cmd, text=True, capture_output=True)
check("正式 evaluator 逐 run 預檢並回報缺少的中段及末段 checkpoint", evaluation.returncode != 0
      and f"[{EPH[25]}, {EPH[-1]}]" in evaluation.stderr and "缺少必要 checkpoint" in evaluation.stderr)

q = agg(k6=AMP, k1=AMP)["report60"]
check("K=6 NPZ 不能冒用 K=1", q["signal"]["state"] == "未能測試" and q["headroom"]["state"] == "PASS")
for key, value in (("n_eval", 1), ("test_block", 6)):
    path = rewrite(AMP, f"bad_{key}", lambda data: data.__setitem__(key, np.array(value)))
    q = agg(k6=path, k1=K1OK)["report60"]
    check(f"拒絕 NPZ 錯誤 {key}", q["candidate_pool"]["state"] == "未能測試")
legacy = rewrite(AMP, "no_manifest", lambda data: data.pop("week2_manifest"))
check("未標明正式評估角色的歷史 NPZ 不得作第 2 週判讀",
      agg(k6=legacy, k1=K1OK)["report60"]["candidate_pool"]["state"] == "未能測試")

for failure, edit in (
    ("data", lambda m: m.__setitem__("data_sha256", "0" * 64)),
    ("train_t_final", lambda m: m["runs"][2]["training"].__setitem__("t_final", 700)),
    ("receipt", lambda m: m["runs"][3].__setitem__("completion", None)),
):
    path = rewrite(AMP, "manifest_" + failure, lambda z: change_manifest(z, edit))
    check(f"拒絕正式來源不符 {failure}", agg(k6=path, k1=K1OK)["report60"]["candidate_pool"]["state"] == "未能測試")

pc = pcs(350, 300, 1000)
pc["k1c0"] = K1OK
q = agg(k6=AMP, k1=K1OK, **pc)["positive_control"]
check("未過濾的 K=1 NPZ 不得充當類別 0 正控制", q["status"].startswith("未能測試") and "評估角色" in q["input_errors"]["k1c0"])

root = Path(a.tmp) / "source_run"
root.mkdir(exist_ok=True)
datafile = Path(a.tmp) / "fake_data.npy"
datafile.write_bytes(b"synthetic fixture")
for pick in (0, 3):
    model_meta = {**expected_training("normal", pick), "out": str(root), "data": str(datafile)}
    (root / "meta.json").write_text(json.dumps(model_meta))
    if pick == 0:
        check("舊 pick 0 完整正式設定時可沿用", read_runs([root], [model_meta], "normal", datafile, DATA_SHA)[0]["completion"] is None)
    else:
        try:
            read_runs([root], [model_meta], "normal", datafile, DATA_SHA)
        except ValueError as exc:
            check("新 pick 3 缺 completion.json 時拒絕", "completion.json" in str(exc))
        else:
            check("新 pick 3 缺 completion.json 時拒絕", False)
        receipt = {"version": 1, "data_sha256": DATA_SHA, "meta_sha256": sha256(root / "meta.json"), "epochs": EPH.tolist()}
        (root / "completion.json").write_text(json.dumps(receipt))
        check("新 pick 3 完成標記吻合，來源檢查通過",
              read_runs([root], [model_meta], "normal", datafile, DATA_SHA)[0]["completion"] == receipt)

q = agg(k6=k6f("low_headroom", gaps=(1, 0.5, 0)), k1=K1OK)["report60"]
check("G3 95% 上界低於 .05 -> 空間 FAIL", q["headroom"]["state"] == "FAIL" and "headroom" in q["candidate_pool"]["failed_axes"])
q = agg(k6=k6f("gray_headroom", gaps=(1, 1.5, 0)), k1=K1OK)["report60"]
check("G3=.075 -> 空間灰區", q["headroom"]["state"] == "灰區")

K1other = k1f("k1_other")
with np.load(K1other) as f:
    z = dict(f)
z["ce"] = np.ones_like(z["ce"]) * (1 + 0.1 * np.abs(np.arange(E) - 3))[None, :, None, None, None]
np.savez(K1other, **z)
probe = agg(k6=AMP, k1=K1other)["second_probe"]
check("e† 只使用 K=6 CE；全 t Γ 仍輸出作描述", probe["state"] == "PASS" and probe["e_dagger"] == EPH[LAST]
      and probe["gamma6_all_t_descriptive"] is not None and probe["t_used"] == [50, 75, 100, 150, 200, 300])

# reach.probe_amplification 的跳過規則（直接測函式）
import reach
mse_own = np.array([[0., 0.], [8., 8.], [8., 8.], [8., 8.], [8., 8.], [8., 8.]])
mse_test = np.array([[1., 1.], [10., 10.], [10., 10.], [10., 10.], [10., 10.], [10., 10.]])
gap = reach.membership_gap(mse_own, mse_test, 2000, 0, .95, .10, .05)
check("G3 是原始誤差總和之比，非 run 比值平均", np.isclose(gap["g3"], 11 / 51) and not np.isclose(gap["g3"], .20 + .8 / 6))
check("G3 負誤差不具物理意義", reach.membership_gap(-mse_test, mse_test, 2000, 0, .95, .10, .05)["state"] == "未能測試")


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
g6, c6 = terms(1.0); c6[0, 0] = np.nan
o = reach.probe_amplification(g6, c6, list(range(6)), g1[:4], c1[:4], list(range(4)), 4.0, 2000, 0, 0.05)
check("分母含非有限值時回報未能測試", o["state"] == "未能測試" and o["denominators"]["k6"] is None)

# 實際 cosine 反例：w=3 的兩分支在中段抵銷，極限與 w=0 的最低點相同。
epochs = np.array([100, 110, 121, 133, 146, 160, 176, 193, 212])
cond_y = np.array([6, 6, 6, 6, 6, 0.2, 0.2, 9, 10])
gamma_y = np.array([2, -2, -2, -2, 2, 0.1, 0.1, 3, 4])

def onset(x, y):
    distance = 1 - x / np.hypot(x, y)
    distance /= distance.mean()
    smooth = np.array([distance[max(i - 1, 0):i + 2].mean() for i in range(len(epochs))])
    return int(epochs[smooth.argmin()])

check("有限 w=3 的最低點可早於 w→∞", onset(2, cond_y) == onset(1, gamma_y) == 160 and onset(5, cond_y + 3 * gamma_y) == 121)

print("\nALL PASS" if not fails else f"\nFAILED: {fails}"); sys.exit(1 if fails else 0)
