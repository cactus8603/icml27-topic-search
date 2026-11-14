"""E0/E1 的 GPU 步驟：對 Half-Truths 各子集算 embedding（影像、anchor/full/half 全句、兩個新增細節片段），存 npz。
之後所有指標（COR、HT、LLR、bootstrap）都用 analyze_ht.py 在 CPU 上由 npz 重算。
用法：python scripts/extract_ht.py --model clip_b32|cs_clip [--subsets ...] [--limit N] [--device cuda] [--out-dir ...]
"""
import argparse
import io
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (EXP, SUBSETS, encode_images, encode_texts, extract_detail, jdump, load_ht, load_model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["clip_b32", "cs_clip"])
    ap.add_argument("--subsets", nargs="+", default=SUBSETS)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out-dir", default=str(EXP / "results" / "ht"))
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    from PIL import Image
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model, preprocess, tok = load_model(args.model, args.device)
    tau = float(model.logit_scale.exp().item())
    for sub in args.subsets:
        t0 = time.time()
        rows = load_ht(sub, args.limit)
        openers = [(lambda r=r: Image.open(io.BytesIO(r["image"]["bytes"]))) for r in rows]
        img = encode_images(model, preprocess, openers, args.device, bs=args.batch_size)
        anchor, full, half = ([r[k] for r in rows] for k in ("anchor", "full_truth", "half_truth"))
        dplus, ok_p = zip(*[extract_detail(r["anchor"], r["full_truth"]) for r in rows])
        dminus, ok_m = zip(*[extract_detail(r["anchor"], r["half_truth"]) for r in rows])
        arr = {"img": img, "tau": np.array(tau)}
        for name, texts in [("anchor", anchor), ("full", full), ("half", half),
                            ("dplus", list(dplus)), ("dminus", list(dminus))]:
            arr[name] = encode_texts(model, tok, texts, args.device)
        arr["ids"] = np.array([r["id"] for r in rows])
        arr["sha"] = np.array([r["image_sha256"] for r in rows])
        arr["triplet_type"] = np.array([r["triplet_type"] for r in rows])
        arr["foil_type"] = np.array([r["foil_type"] for r in rows])
        arr["dplus_text"] = np.array(dplus)
        arr["dminus_text"] = np.array(dminus)
        arr["ok_plus"] = np.array(ok_p)
        arr["ok_minus"] = np.array(ok_m)
        np.savez_compressed(out_dir / f"{args.model}_{sub}{args.tag}.npz", **arr)
        jdump({"model": args.model, "subset": sub, "n": len(rows), "tau": tau, "seconds": time.time() - t0,
               "device": args.device}, out_dir / f"{args.model}_{sub}{args.tag}.meta.json")
        print(args.model, sub, len(rows), f"{time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
