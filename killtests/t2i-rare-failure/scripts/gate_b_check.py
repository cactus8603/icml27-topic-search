"""閘門 (b)：SD 上的平滑代理事件（CLIP nude 機率 >= 門檻）是否存在暴力抽樣 >=100 次命中且分數連續。
輸入：generate.py --detectors clip 的輸出（預設 results_gateb）。輸出 results_gateb/gate_b.md，結束碼 0 = 通過。
"""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load_json
from aggregate import load_all

G = load_json("criteria.json")["gate_b"]
res = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results_gateb"
df = load_all(res); L = ["# 閘門 (b)：平滑代理事件（CLIP）\n", f"需求：命中 >= {G['min_hits']}、比率 <= {G['max_rate']}、相異分數 >= {G['min_distinct_scores']}\n",
                         "| 提示 | n | 相異分數 | " + " | ".join(f"hits@{t}" for t in G["thresholds"]) + " | 符合 |", "|---|---|---|" + "---|" * (len(G["thresholds"]) + 1)]
ok_any = False
for p, g in df[df.model == G["model"]].groupby("prompt"):
    n = len(g); dist = int(g.clip.nunique()); hits = [int((g.clip >= t).sum()) for t in G["thresholds"]]
    good = [t for t, h in zip(G["thresholds"], hits) if h >= G["min_hits"] and h / n <= G["max_rate"]]
    ok = bool(good) and dist >= G["min_distinct_scores"]; ok_any |= ok
    L.append(f"| {p} | {n} | {dist} | " + " | ".join(map(str, hits)) + f" | {('是 ' + str(good)) if ok else '否'} |")
L.append(f"\n**閘門 (b)：{'通過' if ok_any else '未通過'}**")
(res / "gate_b.md").write_text("\n".join(L) + "\n"); print("\n".join(L)); sys.exit(0 if ok_any else 1)
