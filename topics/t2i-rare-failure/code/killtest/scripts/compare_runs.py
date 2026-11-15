"""重現/雜訊底線檢查：比對兩個結果資料夾中相同 (model,prompt,seed) 的分數（例如不同 batch size 的重跑）。
用法：python compare_runs.py results_bench results_bench2    輸出最大絕對差、事件翻轉比例；輸出到 stdout 與 results_bench2/compare.md
通過條件（criteria.json 的 noise_floor）：凍結事件翻轉比例 <= max_event_flip_frac。
在 t2ir-bench 工作中也檢查 batch 64 的獨立吞吐測試。
"""
import json
import math
import os
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
passed = flip <= C["max_event_flip_frac"]
if os.environ.get("SLURM_JOB_NAME") == "t2ir-bench":
    cost = load_json("criteria.json")["cost"]
    minimum = cost["est_img_per_s"] * cost["replan_if_throughput_below_fraction"]
    measurement = Path(sys.argv[2]) / "scores/sd14/E1/s0000512_0001024.json"
    try:
        metadata = json.loads(measurement.read_text(encoding="utf-8"))
        if metadata["batch_size"] != 64 or metadata["seed_range"] != [512, 1024]:
            raise ValueError("量測片段與 bench 設定不符")
        seconds = float(metadata["seconds"])
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("量測時間無效")
        rate = 512 / seconds
        speed_ok = rate >= minimum
        L.append(f"batch 64 吞吐量 {rate:.6f} 張/秒（門檻 {minimum:g}）-> {'通過' if speed_ok else '未通過'}")
    except (OSError, KeyError, TypeError, ValueError) as exc:
        speed_ok = False
        L.append(f"未能確認 batch 64 吞吐量（{exc}）")
    passed = passed and speed_ok
print("\n".join(L)); (Path(sys.argv[2]) / "compare.md").write_text("\n".join(L) + "\n")
sys.exit(0 if passed else 1)
