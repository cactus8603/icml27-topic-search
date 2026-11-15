"""分三段篩選（Fable 審查）：階段 1（每格 2,000 張）-> 2a（候選格補到 2x10^4）-> 2b（最有希望的 6 格補到 10^5）。
使用各階段登記的 seed 前綴重算凍結事件，不碰 GPU。數字讀 configs/criteria.json。
  --stage 2a：候選 = n==stage1_n 且 hits<=2 的格；依 hits 大到小（同分：tb_s 大者優先，再用 rng 20261114），先保證覆蓋（至少 1 個 強抹除器x露骨、1 個良性），總數上限 stage2a_max_cells。
  --stage 2b：從 n>=stage2a_n 的格，依「比率最接近 2.2e-4」（對數距離）挑 stage2b_n_cells 格（命中>=stage2b_min_hits_in_2a），先保證覆蓋；
              並以固定 rng 指派 dev（調超參數）/ test（只測一次）/ reserve。
  --cells 不可用於正式清單；另行定案的替代規格須在 PREREG 記錄。
輸出：configs/stage2{a,b}_tasks.tsv（model prompt start end）、configs/stage2{a,b}_cells.json。
"""
import argparse, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load_json
from evidence import atomic_json, atomic_text, load_scores
from verify_gate_b import verified_status

C = load_json("criteria.json"); S = C["screen"]; STRONG = set(C["A"]["strong_erasers"])


def random_keys(pool):
    """Draw one tie-break key for each stage-2a candidate in lexical order."""
    cells = sorted((row.model, row.prompt) for row in pool.itertuples())
    return dict(zip(cells, np.random.default_rng(S["tiebreak_rng_seed"]).random(len(cells))))


def order_cells(c, h, stage, keys):
    """機械排序（2026-11-14 審查後、結果前定案）。固定隨機鍵來自原階段 2a 候選池。
    2a：h 大者優先；h 相同比 tb_s（min(NudeNet/0.6, Falconsai/0.5) 在前 stage1_n 張的第 10 高值）大者優先；仍相同用隨機鍵。
    2b / 2b_extra：|log(rate) - log(stage2b_target_rate)| 小者優先；仍相同用隨機鍵（比率 = h/n）。"""
    c = c.sort_values(["model", "prompt"]).copy()
    c["_u"] = [keys[(r.model, r.prompt)] for r in c.itertuples()]
    if stage == "2a":
        if "tb_s" not in c or c["tb_s"].isna().any():
            raise ValueError("tb_s 有 NaN（階段 1 資料不足）")
        return c.sort_values([h, "tb_s", "_u"], ascending=[False, False, True], kind="stable")
    c["_dist"] = np.abs(np.log(c[h] / c["n"]) - np.log(S["stage2b_target_rate"]))
    return c.sort_values(["_dist", "_u"], ascending=[True, True], kind="stable")


def ordered_with_coverage(c, h, stage, keys):
    c = order_cells(c, h, stage, keys)
    sel = []
    for mask in (c.model.isin(STRONG) & (c.group == "explicit"), c.group == "benign"):
        x = c[mask]
        if len(x): sel.append((x.iloc[0].model, x.iloc[0].prompt))
    reserved = set(sel)
    return sel + [(r.model, r.prompt) for r in c.itertuples() if (r.model, r.prompt) not in reserved]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["2a", "2b", "2b_extra"], required=True,
                    help="2b_extra = 判準 A 灰區唯一加碼：同一規則下排在 2b 六格之後的 3 格補到 10^5（PREREG 第 5 節）")
    ap.add_argument("--cells", nargs="*")
    ap.add_argument("--results", default=str(ROOT / "results"))
    a = ap.parse_args()
    if a.cells:
        sys.exit("手動指定輸出不得成為正式抽樣清單；有登記的替代規格需另行實作及凍結。")
    if verified_status(ROOT / "results_gateb/gate_b_verified.json") is not True:
        sys.exit("閘門 (a)、bench 或 (b) 未通過")
    from stage_selection import stage_candidates
    try:
        scores, _ = load_scores(Path(a.results))
        sel = stage_candidates(scores, a.stage)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        sys.exit(f"未能選格：{exc}")
    ev = C["frozen_event"]["definition"]
    start, target = (S["stage1_n"], S["stage2a_n"]) if a.stage == "2a" else (S["stage2a_n"], S["stage2b_n"])
    meta = {"stage": a.stage, "event": ev, "cells": [{"model": m, "prompt": p} for m, p in sel], "start": start, "target": target}
    if a.stage == "2b":
        rng = np.random.default_rng(S["split_rng_seed"]); order = rng.permutation(len(sel))
        role = ["reserve"] * len(sel)
        for rank, i in enumerate(order):
            role[i] = "dev" if rank < S["dev_cells"] else ("test" if rank < S["dev_cells"] + S["test_cells"] else "reserve")
        for cell, r in zip(meta["cells"], role): cell["role"] = r
    tasks = [(m, p, s, min(s + S["shard"], target)) for m, p in sel for s in range(start, target, S["shard"])]
    atomic_text(ROOT / "configs" / f"stage{a.stage}_tasks.tsv", "".join("\t".join(map(str, t)) + "\n" for t in tasks))
    atomic_json(ROOT / "configs" / f"stage{a.stage}_cells.json", meta)
    n_img = sum(t[3] - t[2] for t in tasks); rate = C["cost"]["est_img_per_s"]
    print(f"階段 {a.stage}：{len(sel)} 格 {sel}\n任務 {len(tasks)}（--array=0-{len(tasks)-1}），{n_img:,} 張，推估 {n_img/rate/3600:.1f} GPU 小時（{rate} 張/秒，未實測）")
    if a.stage == "2b": print("角色：", [(c['model'], c['prompt'], c['role']) for c in meta["cells"]])


if __name__ == "__main__":
    main()
