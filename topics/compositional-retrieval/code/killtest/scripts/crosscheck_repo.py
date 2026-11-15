"""閘門 G1 的交叉核對：以官方 repo 的 scripts/evaluate_half_truth.py 的逐列 cosine，對照本實驗 extract_ht.py 存的 npz。
OpenAI CLIP 在官方腳本走 openai-clip 套件（下載 ViT-B-32.pt），本實驗走 open_clip(pretrained=openai)；此核對同時確認兩者等價。
用法：python scripts/crosscheck_repo.py --model clip_b32 --subset coco_qwen --npz results/ht/clip_b32_coco_qwen.npz [--limit N] [--device cuda]
需先用相同 --limit／--tag 跑過 extract_ht.py。輸出 results/gates/g1_crosscheck_<model>_<subset>.json；
scripts/aggregate.py 讀取 results/gates/g1_crosscheck.json（由 --merge 合併各檔，取最大差）。
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import EXTRA_SITE, EXP, ICML, cs_clip_ckpt, ht_parquet, jdump

REPO = ICML / "ext" / "CS-CLIP"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["clip_b32", "cs_clip"])
    ap.add_argument("--subset", default="coco_qwen")
    ap.add_argument("--npz")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--merge", action="store_true", help="合併 results/gates/g1_crosscheck_*_*.json 為 g1_crosscheck.json")
    ap.add_argument("--gates-dir", default=str(EXP / "results" / "gates"))
    ap.add_argument("--scratch", default=str(ICML / "tmp" / "crosscheck"))
    args = ap.parse_args()
    gd = Path(args.gates_dir)
    if args.merge:
        parts = [json.loads(p.read_text()) for p in sorted(gd.glob("g1_crosscheck_*_*.json"))]
        jdump({"max_abs_diff": max(p["max_abs_diff"] for p in parts), "parts": parts}, gd / "g1_crosscheck.json")
        print("merged", len(parts), "max", max(p["max_abs_diff"] for p in parts))
        return
    scratch = Path(args.scratch); scratch.mkdir(parents=True, exist_ok=True)
    src = ht_parquet(args.subset)
    data = src
    if args.limit:
        import pyarrow.parquet as pq
        t = pq.read_table(src).slice(0, args.limit)
        data = scratch / f"{args.subset}_{args.limit}.parquet"
        pq.write_table(t, data)
    out_json = scratch / f"repo_{args.model}_{args.subset}.json"
    cmd = [sys.executable, str(REPO / "scripts" / "evaluate_half_truth.py"), "--subset", args.subset, "--data", str(data),
           "--device", args.device, "--output", str(out_json)]
    if args.model == "cs_clip":
        cmd += ["--checkpoint", cs_clip_ckpt()]
    env = dict(os.environ, PYTHONPATH=str(EXTRA_SITE) + os.pathsep + str(EXP / "scripts_shim"), HOME=str(scratch / "home"), TMPDIR=str(ICML / "tmp"))
    (scratch / "home").mkdir(exist_ok=True)
    subprocess.run(cmd, check=True, env=env, cwd=str(scratch))
    repo = {r["id"]: r for r in json.loads(out_json.read_text())["scores"]}
    z = np.load(args.npz)
    mine = {
        "s_anchor": np.einsum("nd,nd->n", z["img"], z["anchor"]),
        "s_half_truth": np.einsum("nd,nd->n", z["img"], z["half"]),
        "s_full_truth": np.einsum("nd,nd->n", z["img"], z["full"]),
    }
    diffs = []
    for key, arr in mine.items():
        ref = np.array([repo[i][key] for i in z["ids"]])
        diffs.append(float(np.abs(ref - arr).max()))
    res = {"model": args.model, "subset": args.subset, "n": int(len(z["ids"])), "max_abs_diff": max(diffs),
           "per_key": dict(zip(mine, diffs)), "device": args.device}
    jdump(res, gd / f"g1_crosscheck_{args.model}_{args.subset}.json")
    print(res)


if __name__ == "__main__":
    main()
