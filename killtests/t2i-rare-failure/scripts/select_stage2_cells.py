"""分三段篩選（Fable 審查）：階段 1（每格 2,000 張）-> 2a（候選格補到 2x10^4）-> 2b（最有希望的 6 格補到 10^5）。
只讀 results/summary_cells.csv（由 aggregate.py 產生，需用凍結事件定義），不碰 GPU。所有數字讀 configs/criteria.json。
  --stage 2a：候選 = n==stage1_n 且 hits<=2 的格；依 hits 大到小（同分：tb_s 大者優先，再用 rng 20261114），先保證覆蓋（至少 1 個 強抹除器x露骨、1 個良性），總數上限 stage2a_max_cells。
  --stage 2b：從 n>=stage2a_n 的格，依「比率最接近 2.2e-4」（對數距離）挑 stage2b_n_cells 格（命中>=stage2b_min_hits_in_2a），先保證覆蓋；
              並以固定 rng 指派 dev（調超參數）/ test（只測一次）/ reserve。
  --cells model:prompt ... 手動指定（需在 PREREG 記錄理由）。
輸出：configs/stage2{a,b}_tasks.tsv（model prompt start end）、configs/stage2{a,b}_cells.json。
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load_json

C = load_json("criteria.json"); S = C["screen"]; STRONG = set(C["A"]["strong_erasers"])


def order_cells(c, h, stage):
    """機械排序（2026-11-14 審查後、結果前定案）。隨機鍵：候選格先依 (model,prompt) 字母排序，再用 rng(tiebreak_rng_seed) 依序抽 uniform。
    2a：h 大者優先；h 相同比 tb_s（min(NudeNet/0.6, Falconsai/0.5) 在前 stage1_n 張的第 10 高值）大者優先；仍相同用隨機鍵。
    2b / 2b_extra：|log(rate) - log(stage2b_target_rate)| 小者優先；仍相同用隨機鍵（比率 = h/n）。"""
    c = c.sort_values(["model", "prompt"]).copy()
    c["_u"] = np.random.default_rng(S["tiebreak_rng_seed"]).random(len(c))
    if stage == "2a":
        if "tb_s" not in c or c["tb_s"].isna().any(): sys.exit("tb_s 有 NaN（階段 1 資料不足）；先重跑 aggregate.py")
        return c.sort_values([h, "tb_s", "_u"], ascending=[False, False, True], kind="stable")
    c["_dist"] = np.abs(np.log(c[h] / c["n"]) - np.log(S["stage2b_target_rate"]))
    return c.sort_values(["_dist", "_u"], ascending=[True, True], kind="stable")


def pick(c, h, k, stage):
    c = order_cells(c, h, stage)
    sel = []
    for mask in (c.model.isin(STRONG) & (c.group == "explicit"), c.group == "benign"):
        x = c[mask]
        if len(x): sel.append((x.iloc[0].model, x.iloc[0].prompt))
    for _, r in c.iterrows():
        if len(sel) >= k: break
        if (r.model, r.prompt) not in sel: sel.append((r.model, r.prompt))
    return sel[:k] if len(sel) > k else sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["2a", "2b", "2b_extra"], required=True,
                    help="2b_extra = 判準 A 灰區唯一加碼：同一規則下排在 2b 六格之後的 3 格補到 10^5（PREREG 第 5 節）")
    ap.add_argument("--cells", nargs="*")
    ap.add_argument("--results", default=str(ROOT / "results"))
    a = ap.parse_args()
    ev = C["frozen_event"]["definition"]; h = f"hits_{ev}"
    T = pd.read_csv(Path(a.results) / "summary_cells.csv")
    # 2026-11-14 審查：前一階段未完成（缺格、n 不足或 seed 有缺口）時不得篩選，否則候選格會被靜默排除
    if a.stage == "2a":
        n_exp = len(load_json("models.json")["models"]) * len(load_json("prompts.json")["prompts"])
        bad = T[(T.n < S["stage1_n"]) | (~T.contiguous)] if "contiguous" in T else T[T.n < S["stage1_n"]]
        if len(T) != n_exp or len(bad):
            sys.exit(f"階段 1 未完成：{len(T)}/{n_exp} 格，n 不足或不連續 {len(bad)} 格；先補跑 stage1 再篩選")
    elif a.stage == "2b":
        prev = json.loads((ROOT / "configs" / "stage2a_cells.json").read_text())["cells"] if not a.cells else []
        TT = T.set_index(["model", "prompt"])
        miss = [(c["model"], c["prompt"]) for c in prev
                if (c["model"], c["prompt"]) not in TT.index or TT.loc[(c["model"], c["prompt"]), "n"] < S["stage2a_n"]
                or ("contiguous" in TT and not TT.loc[(c["model"], c["prompt"]), "contiguous"])]
        if miss: sys.exit(f"階段 2a 未完成的格：{miss}；先補跑 stage2a 再篩選")
    if a.stage == "2a":
        start, target, k = S["stage1_n"], S["stage2a_n"], S["stage2a_max_cells"]
        c = T[(T.n == start) & (T[h] <= S["stage2a_max_hits_in_stage1"])]
    else:
        start, target, k = S["stage2a_n"], S["stage2b_n"], S["stage2b_n_cells"]
        c = T[(T.n >= start) & (T.n < target) & (T[h] >= S["stage2b_min_hits_in_2a"])]
        if a.stage == "2b_extra":  # 已補到 10^5 的 2b 格自然被 n < target 排除；依同一規則（先保證覆蓋、再依 h）取 3 格
            done = {(x["model"], x["prompt"]) for x in json.loads((ROOT / "configs" / "stage2b_cells.json").read_text())["cells"]}
            c = c[[(m, p) not in done for m, p in zip(c.model, c.prompt)]]; k = 3
    c = c[c.model != "__none__"]
    sel = [tuple(x.split(":")) for x in a.cells] if a.cells else pick(c, h, k, a.stage)
    meta = {"stage": a.stage, "event": ev, "cells": [{"model": m, "prompt": p} for m, p in sel], "start": start, "target": target}
    if a.stage == "2b":
        rng = np.random.default_rng(S["split_rng_seed"]); order = rng.permutation(len(sel))
        role = ["reserve"] * len(sel)
        for rank, i in enumerate(order):
            role[i] = "dev" if rank < S["dev_cells"] else ("test" if rank < S["dev_cells"] + S["test_cells"] else "reserve")
        for cell, r in zip(meta["cells"], role): cell["role"] = r
    tasks = [(m, p, s, min(s + S["shard"], target)) for m, p in sel for s in range(start, target, S["shard"])]
    (ROOT / "configs" / f"stage{a.stage}_cells.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    with open(ROOT / "configs" / f"stage{a.stage}_tasks.tsv", "w") as f:
        for t in tasks: f.write("\t".join(map(str, t)) + "\n")
    n_img = sum(t[3] - t[2] for t in tasks); rate = C["cost"]["est_img_per_s"]
    print(f"階段 {a.stage}：{len(sel)} 格 {sel}\n任務 {len(tasks)}（--array=0-{len(tasks)-1}），{n_img:,} 張，推估 {n_img/rate/3600:.1f} GPU 小時（{rate} 張/秒，未實測）")
    if a.stage == "2b": print("角色：", [(c['model'], c['prompt'], c['role']) for c in meta["cells"]])


if __name__ == "__main__":
    main()
