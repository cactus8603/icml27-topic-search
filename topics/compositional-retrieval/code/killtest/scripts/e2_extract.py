"""E2 的 GPU 步驟：對 COCO val2017 全部影像與查詢文字做 embedding，存成 npz。
需先有 results/e2/queries_main.json（及可選 queries_addon.json）。
用法：python scripts/e2_extract.py --model clip_b32|cs_clip [--device cuda] [--limit-images N]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import COCO_DIR, EXP, encode_images, encode_texts, jdump, load_model
from e2_build_queries import joint_text, unit_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--queries", nargs="+", default=["main", "addon"])
    ap.add_argument("--limit-images", type=int)
    ap.add_argument("--out-dir", default=str(EXP / "results" / "e2"))
    ap.add_argument("--image-dir", default=str(COCO_DIR / "val2017"))
    args = ap.parse_args()
    from PIL import Image
    out_dir = Path(args.out_dir)
    t0 = time.time()
    model, preprocess, tok = load_model(args.model, args.device)
    tau = float(model.logit_scale.exp().item())
    qsets = {}
    for q in args.queries:
        p = out_dir / f"queries_{q}.json"
        if p.exists():
            qsets[q] = json.loads(p.read_text())
    if not qsets:
        raise SystemExit("找不到 queries_*.json")
    first = next(iter(qsets.values()))
    image_ids = first["image_ids"][: args.limit_images]
    cats = {int(k): v for k, v in first["categories"].items()}
    cat_ids = sorted(cats)
    openers = [(lambda i=i: Image.open(Path(args.image_dir) / f"{i:012d}.jpg")) for i in image_ids]
    img = encode_images(model, preprocess, openers, args.device)
    t_img = time.time() - t0
    unit = encode_texts(model, tok, [unit_text(cats[c]) for c in cat_ids], args.device)
    arrays = {"img": img, "unit": unit, "cat_ids": np.array(cat_ids), "tau": np.array(tau),
              "image_ids": np.array(image_ids)}
    for name, qs in qsets.items():
        texts, owner = [], []
        for k, lst in qs["queries"].items():
            for n, q in enumerate(lst):
                names = [cats[c] for c in q["cats"]]
                texts.append(joint_text(names))
                owner.append((int(k), n))
        arrays[f"joint_{name}"] = encode_texts(model, tok, texts, args.device)
        arrays[f"owner_{name}"] = np.array(owner)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"emb_{args.model}.npz", **arrays)
    jdump({"model": args.model, "n_images": len(image_ids), "tau": tau, "seconds_total": time.time() - t0,
           "seconds_images": t_img, "device": args.device}, out_dir / f"emb_{args.model}.meta.json")
    print("done", args.model, img.shape, f"{time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
