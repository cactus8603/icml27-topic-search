"""第二探針的 GPU 步驟（只做 embedding，不做任何判定）：
 (1) 正控制句（half 句中 d- 換成別列 d+）的文字 embedding -> results/probe2/ctrl_<model>_<subset>.npz
 (2) COCO val2017 全部 5000 張影像 embedding（敏感度池）-> results/probe2/pool_coco5000_<model>.npz
用法：python scripts/probe2_embed.py --model clip_b32|cs_clip [--device cuda] [--limit N]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import COCO_DIR, EXP, SUBSETS, encode_images, encode_texts, jdump, load_ht, load_model
from probe2_condpmi import p2params, pos_control_details, replace_detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["clip_b32", "cs_clip"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out-dir", default=str(EXP / "results" / "probe2"))
    ap.add_argument("--ht-dir", default=str(EXP / "results" / "ht"))
    ap.add_argument("--skip-coco", action="store_true")
    a = ap.parse_args()
    from PIL import Image
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = p2params()
    model, preprocess, tok = load_model(a.model, a.device)
    for i, sub in enumerate(SUBSETS):
        rows = load_ht(sub, a.limit)
        z = np.load(Path(a.ht_dir) / f"{a.model}_{sub}.npz", allow_pickle=False)
        n = len(rows)
        assert list(z["ids"][:n]) == [r["id"] for r in rows], "npz 與 parquet 列順序不一致"
        dplus = list(z["dplus_text"][:n])
        det = pos_control_details(dplus, p["seeds"]["pos_control"], i)
        texts = [replace_detail(r["anchor"], r["half_truth"], d) for r, d in zip(rows, det)]
        emb = encode_texts(model, tok, texts, a.device)
        np.savez_compressed(out / f"ctrl_{a.model}_{sub}.npz", ctrl_pos=emb, ctrl_pos_text=np.array(texts),
                            ctrl_pos_detail=np.array(det), ids=z["ids"][:n])
        print(a.model, sub, n, "ctrl ok; e.g.:", texts[0], flush=True)
    if not a.skip_coco:
        d = COCO_DIR / "val2017"
        files = sorted(d.glob("*.jpg"))[: a.limit]
        openers = [(lambda f=f: Image.open(f)) for f in files]
        img = encode_images(model, preprocess, openers, a.device)
        np.savez_compressed(out / f"pool_coco5000_{a.model}.npz", img=img, files=np.array([f.name for f in files]))
        print(a.model, "coco pool", img.shape, flush=True)
    jdump({"model": a.model, "device": a.device}, out / f"embed_{a.model}.meta.json")


if __name__ == "__main__":
    main()
