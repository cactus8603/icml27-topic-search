"""重現/雜訊底線檢查：比對兩個結果資料夾中相同 (model,prompt,seed) 的分數（例如不同 batch size 的重跑）。
用法：python compare_runs.py results_bench results_bench2    輸出最大絕對差、事件翻轉比例；輸出到 stdout 與 results_bench2/compare.md
通過條件（criteria.json 的 noise_floor）：凍結事件翻轉比例 <= max_event_flip_frac。
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_json
from aggregate import load_all, events
C = load_json("criteria.json")["noise_floor"]; D = load_json("detectors.json"); EV = load_json("criteria.json")["frozen_event"]["definition"]
a, b = load_all(Path(sys.argv[1])), load_all(Path(sys.argv[2]))
m = a.merge(b, on=["model", "prompt", "seed"], suffixes=("_a", "_b"))
if not len(m): sys.exit("沒有共同的 (model,prompt,seed)")
ea = events(a, D)[EV]; eb = events(b, D)[EV]
a = a.assign(ev=ea.values); b = b.assign(ev=eb.values)
mm = a.merge(b, on=["model", "prompt", "seed"], suffixes=("_a", "_b"))
L = [f"共同樣本 {len(mm)}"]
for k in ("nudenet", "falconsai", "clip"):
    d = np.abs(mm[k + "_a"] - mm[k + "_b"]); L.append(f"{k}: 最大絕對差 {np.nanmax(d):.4f}，平均 {np.nanmean(d):.5f}")
flip = float((mm.ev_a != mm.ev_b).mean()); L.append(f"凍結事件翻轉比例 {flip:.4%}（門檻 {C['max_event_flip_frac']:.2%}）-> {'通過' if flip <= C['max_event_flip_frac'] else '未通過'}")
print("\n".join(L)); (Path(sys.argv[2]) / "compare.md").write_text("\n".join(L) + "\n")
sys.exit(0 if flip <= C["max_event_flip_frac"] else 1)
