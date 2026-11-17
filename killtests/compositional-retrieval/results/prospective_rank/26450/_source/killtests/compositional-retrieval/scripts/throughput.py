"""10 分鐘吞吐實測：量測影像編碼（COCO val2017 jpg 與 Half-Truths 影像）與文字編碼的每秒處理量，
並推估完整 E0–E2 的 GPU 時間。用法：python scripts/throughput.py --device cuda --seconds 120
輸出 results/throughput.json。
"""
import argparse
import io
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import COCO_DIR, EXP, encode_images, encode_texts, jdump, load_ht, load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seconds", type=float, default=120.0, help="每個模型每個項目最多量測秒數")
    ap.add_argument("--out", default=str(EXP / "results" / "throughput.json"))
    args = ap.parse_args()
    import torch
    from PIL import Image
    res = {"device": args.device, "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None}
    rows = load_ht("coco_qwen")
    coco = sorted((COCO_DIR / "val2017").glob("*.jpg"))
    for name in ("clip_b32", "cs_clip"):
        t0 = time.time()
        model, pre, tok = load_model(name, args.device)
        res[f"{name}_load_s"] = time.time() - t0
        # 影像：Half-Truths（bytes）
        n, t0 = 0, time.time()
        while time.time() - t0 < args.seconds / 2 and n < 4 * len(rows):
            ops = [(lambda r=r: Image.open(io.BytesIO(r["image"]["bytes"]))) for r in rows]
            encode_images(model, pre, ops, args.device, bs=64)
            n += len(ops)
        res[f"{name}_ht_img_per_s"] = n / (time.time() - t0)
        # 影像：COCO val2017（檔案）
        if coco:
            n, t0 = 0, time.time()
            while time.time() - t0 < args.seconds / 2 and n < 2 * len(coco):
                ops = [(lambda p=p: Image.open(p)) for p in coco[:2500]]
                encode_images(model, pre, ops, args.device, bs=64)
                n += len(ops)
            res[f"{name}_coco_img_per_s"] = n / (time.time() - t0)
        texts = [r["half_truth"] for r in rows] * 10
        t0 = time.time()
        encode_texts(model, tok, texts, args.device)
        res[f"{name}_txt_per_s"] = len(texts) / (time.time() - t0)
        del model
        torch.cuda.empty_cache() if args.device == "cuda" else None
    # 推估：HT 3 子集共 1437 列 × 1 張影像 + 5 組文字；E2 5000 影像 + 約 6000×2 查詢文字；兩模型；重跑一次
    est = 0.0
    for name in ("clip_b32", "cs_clip"):
        est += 1437 / res[f"{name}_ht_img_per_s"] * 2          # 含重跑
        est += 1437 * 5 / res[f"{name}_txt_per_s"] * 2
        if f"{name}_coco_img_per_s" in res:
            est += 5000 / res[f"{name}_coco_img_per_s"]
            est += 12000 / res[f"{name}_txt_per_s"]
        est += res[f"{name}_load_s"] * 3
    res["estimated_total_gpu_hours_E0_E2"] = est / 3600
    jdump(res, args.out)
    print(res)


if __name__ == "__main__":
    main()
