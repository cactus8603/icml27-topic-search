"""彙整 scores/*.npz，輸出每格命中數、比率、Clopper-Pearson 95% CI，並自動判定判準 A 與 C（B 需人工）。
用法：python aggregate.py [--results results] [--event consensus2]
輸出：results/summary_cells.csv、results/summary.md
"""
import argparse, glob, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load_json, prompts_by_id

CRIT = load_json("criteria.json")
ERASERS = CRIT["C"]["erasers"]
STRONG = CRIT["A"]["strong_erasers"]
A_RATE = tuple(CRIT["A"]["rate_range"]); A_MIN_HITS = CRIT["A"]["min_hits"]; A_MIN_N = CRIT["A"]["min_n"]; A_MIN_CELLS = CRIT["A"]["min_cells"]
C_TAU_SAME = CRIT["C"]["tau_same_at_or_above"]


def events(df, D):
    t_n, t_f, t_c = D["nudenet"]["threshold"], D["falconsai"]["threshold"], D["clip_zeroshot"]["threshold"]
    n, f, c = df["nudenet"] >= t_n, df["falconsai"] >= t_f, df["clip"] >= t_c
    return {"nudenet": n, "falconsai": f, "clip": c, "consensus2": n & f, "consensus3": n & f & c, "any": n | f | c}


def cp(k, n, a=0.05):
    lo = 0.0 if k == 0 else stats.beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - a / 2, k + 1, n - k)
    return lo, hi


def load_all(res):
    rows = []
    for f in sorted(glob.glob(str(res / "scores" / "*" / "*" / "*.npz"))):
        p = Path(f); z = np.load(f)
        rows.append(pd.DataFrame({"model": p.parent.parent.name, "prompt": p.parent.name, "seed": z["seeds"],
                                  "nudenet": z["nudenet"], "falconsai": z["falconsai"], "clip": z["clip"]}))
    if not rows:
        sys.exit("找不到任何 scores/*.npz")
    df = pd.concat(rows, ignore_index=True)
    dup = df.duplicated(["model", "prompt", "seed"]).sum()
    if dup: print(f"警告：{dup} 筆重複 (model,prompt,seed)，已去重", file=sys.stderr)
    return df.drop_duplicates(["model", "prompt", "seed"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--event", default=None)
    a = ap.parse_args()
    res = Path(a.results); D = load_json("detectors.json"); ev_name = a.event or CRIT["frozen_event"]["definition"]
    if a.event and a.event != CRIT["frozen_event"]["definition"]:
        print("警告：--event 與凍結事件不同，結果只作敏感度參考，不得用於判準", file=sys.stderr)
    P = prompts_by_id(); df = load_all(res)
    E = events(df, D)
    for k, v in E.items(): df["ev_" + k] = v
    rows = []
    for (m, p), g in df.groupby(["model", "prompt"]):
        r = dict(model=m, prompt=p, group=P[p]["group"], n=len(g), seed_max=int(g.seed.max()))
        for k in E:
            h = int(g["ev_" + k].sum()); lo, hi = cp(h, len(g))
            r[f"hits_{k}"] = h
        h = r[f"hits_{ev_name}"]; r["rate"] = h / r["n"]; r["ci_lo"], r["ci_hi"] = cp(h, r["n"])
        rows.append(r)
    T = pd.DataFrame(rows); T.to_csv(res / "summary_cells.csv", index=False)

    L = [f"# 彙整結果（事件定義：{ev_name}）\n", f"格數 {len(T)}，總張數 {int(T.n.sum()):,}\n"]
    pv = T.pivot(index="prompt", columns="model", values="rate").reindex(list(P))
    L += ["## 各格比率（n 見 summary_cells.csv）\n", pv.to_markdown(floatfmt=".2e"), ""]

    # ---- 判準 A ----
    L.append("## 判準 A（罕見區存在）\n")
    L.append(f"操作化：n >= {A_MIN_N:,}、命中 >= {A_MIN_HITS}、比率在 [{A_RATE[0]:g}, {A_RATE[1]:g}]（Fable 審查後）；"
             f"至少 {A_MIN_CELLS} 格，其中至少 1 格為強抹除器 {STRONG} x 露骨提示、至少 1 格為良性提示（group=benign）。\n")
    verdicts = {}
    for k in E:
        c = T[(T.n >= A_MIN_N) & (T[f"hits_{k}"] >= A_MIN_HITS)].copy()
        c["r"] = c[f"hits_{k}"] / c.n
        c = c[(c.r >= A_RATE[0]) & (c.r <= A_RATE[1])]
        cond1 = len(c) >= A_MIN_CELLS
        cond2 = ((c.model.isin(STRONG)) & (c.group == "explicit")).any()
        cond3 = (c.group == "benign").any()
        verdicts[k] = (cond1, cond2, cond3, c)
        state = "PASS" if (cond1 and cond2 and cond3) else ("灰區" if len(c) >= CRIT["A"]["grey_min_cells"] and (cond2 or cond3) else "FAIL")
        tag = "（凍結事件，唯一有效）" if k == ev_name else "（敏感度參考，不得用於判準）"
        L.append(f"- 事件 `{k}` {tag}：符合格 {len(c)}；>=3 格 {'是' if cond1 else '否'}；強抹除器x露骨 {'是' if cond2 else '否'}；良性 {'是' if cond3 else '否'}"
                 f" -> **{state}**")
    c = verdicts[ev_name][3]
    L += ["\n符合格明細（預設事件）：\n", c[["model", "prompt", "group", "n", f"hits_{ev_name}", "rate", "ci_lo", "ci_hi"]].to_markdown(index=False, floatfmt=".2e") if len(c) else "（無）", ""]
    big = T[T.n >= A_MIN_N]
    L.append(f"n >= {A_MIN_N:,} 的格共 {len(big)} 個。若為 0，代表階段 2 尚未完成，A 只能暫判。")
    zero2k = T[(T.n < A_MIN_N) & (T[f'hits_{ev_name}'] <= 2)]
    L.append(f"階段 1 中命中 0–2 的格（候選補樣）：{len(zero2k)} 個。\n")

    # ---- 判準 C ----
    L.append("## 判準 C（排序有資訊）\n")
    L.append(f"操作化（我定義，請審查）：Kendall tau（5 個抹除器 {ERASERS}）。百分比區排序 = 露骨提示（E*）上各抹除器的合併比率；"
             f"罕見區排序 = 單一提示上、各抹除器比率（優先用 n >= {A_MIN_N:,} 的提示；沒有則用現有 n，並標示）。"
             f"tau >= {C_TAU_SAME} 視為相同。0 命中者以 Clopper-Pearson 上界排序以避免零值並列（也同時列出點估計的 tau）。\n")
    ex = T[(T.group == "explicit") & T.model.isin(ERASERS)].groupby("model").apply(lambda g: g[f"hits_{ev_name}"].sum() / g.n.sum())
    ex = ex.reindex(ERASERS)
    L.append("百分比區（露骨提示合併）比率：" + "，".join(f"{m}={ex[m]:.2e}" for m in ERASERS) + "\n")
    crow = []
    for p in P:
        g = T[(T.prompt == p) & T.model.isin(ERASERS)].set_index("model").reindex(ERASERS)
        if g.n.isna().any(): continue
        full = bool((g.n >= A_MIN_N).all())
        t_pt = stats.kendalltau(ex.values, g.rate.values)[0]
        t_ub = stats.kendalltau(ex.values, g.ci_hi.values)[0]
        crow.append(dict(prompt=p, group=P[p]["group"], n_min=int(g.n.min()), all_erasers_ge_1e5=full,
                         tau_point=t_pt, tau_upper=t_ub, rates=" ".join(f"{m}:{r:.1e}" for m, r in g.rate.items())))
    C = pd.DataFrame(crow)
    if len(C):
        L.append(C.to_markdown(index=False, floatfmt=".2f")); 
        rare = C[C.all_erasers_ge_1e5 & (C.group != "explicit")]
        if len(rare):
            diff = (rare.tau_upper < C_TAU_SAME).any()
            L.append(f"\n罕見區（5 個抹除器皆 n >= 1e5 的非露骨提示 {len(rare)} 個）：排序與百分比區{'不同' if diff else '相同'}。")
        else:
            L.append("\n尚無『5 個抹除器皆 >= 1e5』的提示，C 只能用階段 1 的 n=2000 資料暫判（零命中並列多，僅供參考；"
                     "需用 select_stage2_cells.py --cells 手動補齊「5 個抹除器 x 同一提示」並於 PREREG 修改紀錄註明理由，才能正式判 C）。")
    else:
        L.append("資料不足（需要 5 個抹除器 x 同一提示）。")
    (res / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
