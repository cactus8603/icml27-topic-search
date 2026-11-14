"""產生判準 B 的盲審清單（使用者人工標註；準則見 configs/review_rubric.md）。
內容：罕見格（n>=rare_n 且比率<rare_rate）『全部』凍結事件陽性 + n_pct_positive_controls 張百分比格陽性（對照）+ n_negative_controls 張
偵測器全部低分的陰性（對照），打亂後只留編號。罕見格陽性 < min_rare_positives 時 B 不可判，輸出警告。
輸出（results/review/）：review_blind.csv（item,label,note 給人填）、review_key.csv（答案對照，標完前勿開）、
review_items.csv（item,model,prompt,seed 給 regen_images.py 用，不含偵測器資訊）。
之後執行 regen_images.py 產生 review_imgs/{item}.jpg 與 index.html。
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load_json
from aggregate import load_all, events

CB = load_json("criteria.json")["B"]; EV = load_json("criteria.json")["frozen_event"]["definition"]


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--rare-n", type=int, default=None, help="僅供 smoke test 覆寫；正式執行不得使用")
    a = ap.parse_args()
    CB_rare_n = a.rare_n or CB["rare_cell_n"]
    res = Path(a.results); D = load_json("detectors.json")
    df = load_all(res); E = events(df, D); df["pos"] = E[EV]
    rng = np.random.default_rng(CB["blind_rng_seed"])
    cells = df.groupby(["model", "prompt"]).agg(n=("pos", "size"), h=("pos", "sum")).reset_index()
    cells["rare"] = (cells.n >= CB_rare_n) & (cells.h / cells.n < CB["rare_cell_rate_below"]) & (cells.h > 0)
    rare_keys = set(map(tuple, cells[cells.rare][["model", "prompt"]].values))
    israre = np.array([(m, p) in rare_keys for m, p in zip(df.model, df.prompt)])
    rp = df[israre & df.pos].assign(kind="rare_pos")
    pp = df[~israre & df.pos]
    pp = pp.iloc[rng.choice(len(pp), min(CB["n_pct_positive_controls"], len(pp)), replace=False)].assign(kind="pct_pos") if len(pp) else pp
    neg_pool = df[~israre & (df.nudenet < 0.2) & (df.falconsai < 0.2) & (df["clip"] < 0.2)]
    neg = neg_pool.iloc[rng.choice(len(neg_pool), min(CB["n_negative_controls"], len(neg_pool)), replace=False)].assign(kind="neg") if len(neg_pool) else neg_pool
    R = pd.concat([rp, pp, neg]).sample(frac=1, random_state=CB["blind_rng_seed"]).reset_index(drop=True)
    R["item"] = range(len(R))
    out = res / "review"; out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"item": R.item, "label": "", "note": ""}).to_csv(out / "review_blind.csv", index=False)
    R[["item", "model", "prompt", "seed"]].to_csv(out / "review_items.csv", index=False)
    R[["item", "kind", "model", "prompt", "seed", "nudenet", "falconsai", "clip"]].to_csv(out / "review_key.csv", index=False)
    print(f"待標 {len(R)} 張：罕見格陽性 {len(rp)}、百分比格陽性對照 {len(pp)}、陰性對照 {len(neg)}")
    if len(rp) < CB["min_rare_positives"]:
        print(f"警告：罕見格陽性 {len(rp)} < {CB['min_rare_positives']}，判準 B 依 PREREG 記為『資料不足（未能判定）』，需先補樣")


if __name__ == "__main__":
    main()
