"""分三段篩選（Fable 審查）：階段 1（每格 2,000 張）-> 2a（候選格補到 2x10^4）-> 2b（最有希望的 6 格補到 10^5）。
只讀 results/summary_cells.csv（由 aggregate.py 產生，需用凍結事件定義），不碰 GPU。所有數字讀 configs/criteria.json。
  --stage 2a：候選 = n==stage1_n 且 hits<=2 的格；依 hits 大到小，先保證覆蓋（至少 1 個 強抹除器x露骨、1 個良性），總數上限 stage2a_max_cells。
  --stage 2b：從 n>=stage2a_n 的格，依 2a 命中數挑 stage2b_n_cells 格（命中>=stage2b_min_hits_in_2a），先保證覆蓋；
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


def pick(c, h, k):
    c = c.sort_values([h, "model", "prompt"], ascending=[False, True, True])
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
    ap.add_argument("--stage", choices=["2a", "2b"], required=True)
    ap.add_argument("--cells", nargs="*")
    ap.add_argument("--results", default=str(ROOT / "results"))
    a = ap.parse_args()
    ev = C["frozen_event"]["definition"]; h = f"hits_{ev}"
    T = pd.read_csv(Path(a.results) / "summary_cells.csv")
    if a.stage == "2a":
        start, target, k = S["stage1_n"], S["stage2a_n"], S["stage2a_max_cells"]
        c = T[(T.n == start) & (T[h] <= S["stage2a_max_hits_in_stage1"])]
    else:
        start, target, k = S["stage2a_n"], S["stage2b_n"], S["stage2b_n_cells"]
        c = T[(T.n >= start) & (T.n < target) & (T[h] >= S["stage2b_min_hits_in_2a"])]
    c = c[c.model != "__none__"]
    sel = [tuple(x.split(":")) for x in a.cells] if a.cells else pick(c, h, k)
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
