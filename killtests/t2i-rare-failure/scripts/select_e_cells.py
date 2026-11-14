"""判準 E 的 2 個格（機械選取，2026-11-14 審查後、結果前定案）。
候選 = 階段 2b 的 test 格，且 n >= 10^5、命中 >= E.min_hits、比率 <= E.max_rate（2e-4）；
依 |log10(比率) - log10(E.target_rate)|（E.target_rate = 1e-4）由小到大取 E.cells 格，同分依 (model, prompt) 字母序。
候選不足 2 格 -> E 記為「未能測試」（不放寬比率上限、不改用 dev／reserve 格）。
用法：python select_e_cells.py [--results results] [--dry]   輸出 configs/E_cells.json
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load_json

E = load_json("criteria.json")["E"]; EV = load_json("criteria.json")["frozen_event"]["definition"]


def choose(T, roles):
    c = T[T.apply(lambda r: roles.get((r.model, r.prompt)) == "test", axis=1) & (T.n >= 100000)].copy()
    h = c[f"hits_{EV}"]; c["rate"] = h / c.n
    c = c[(h >= E["min_hits"]) & (c.rate <= E["max_rate"])]
    c["_d"] = np.abs(np.log10(c.rate) - np.log10(E["target_rate"]))
    c = c.sort_values(["_d", "model", "prompt"], kind="stable")
    return [(r.model, r.prompt, float(r.rate)) for r in c.head(E["cells"]).itertuples()], len(c)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--cells-json", default=str(ROOT / "configs" / "stage2b_cells.json")); ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    T = pd.read_csv(Path(a.results) / "summary_cells.csv")
    roles = {(c["model"], c["prompt"]): c["role"] for c in json.loads(Path(a.cells_json).read_text())["cells"]}
    sel, n_cand = choose(T, roles)
    if len(sel) < E["cells"]:
        sys.exit(f"E 未能測試：符合條件的 test 格只有 {n_cand} 個（需要 {E['cells']}）；不放寬條件")
    print("E 格：", sel)
    if not a.dry:
        (ROOT / "configs" / "E_cells.json").write_text(json.dumps([dict(model=m, prompt=p, rate=r) for m, p, r in sel], indent=1))


if __name__ == "__main__":
    main()
