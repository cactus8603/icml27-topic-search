#!/usr/bin/env python
"""把 HF 的 nielsr/CelebA-faces（202,599 張 img_align_celeba，檔名排序 = 原始編號順序）
轉成原作 repo 要用的 CelebA32.pt（灰階 32x32，與 preprocess_celeba.py 同一組 transform）。

原作用「前 index*n 張當訓練、最後 2048 張當測試」，所以只需要前 NHEAD 張與最後 NTAIL 張。
輸出檔列為 [前 NHEAD 張 | 最後 NTAIL 張]，原作的 full_tensor[-2048:] 與 full_tensor[i*n:(i+1)*n]
語意不變（只要 i*n+n <= NHEAD）。

用法：python prep_celeba.py --out-dir $CELEBA_DATA_ROOT [--nhead 16384] [--ntail 2048]
"""
import argparse, io, os, sys
import numpy as np
import torch
import torchvision.transforms as T
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image

REPO = "nielsr/CelebA-faces"
TOTAL = 202599
SHARDS = 3


def tfm_fn(size):
    # 與 wdmdm/Experiments/src/Utils/preprocess_celeba.py 相同：ToTensor -> Resize -> CenterCrop -> Grayscale
    return T.Compose([T.ToTensor(), T.Resize(size), T.CenterCrop(size), T.Grayscale(1)])


def rows_of_shard(path):
    pf = pq.ParquetFile(path)
    for rg in range(pf.num_row_groups):
        tbl = pf.read_row_group(rg, columns=["image"]).to_pylist()
        for r in tbl:
            yield r["image"]["path"], r["image"]["bytes"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--nhead", type=int, default=16384)
    ap.add_argument("--ntail", type=int, default=2048)
    ap.add_argument("--size", type=int, default=32)
    ap.add_argument("--max-rows-per-shard", type=int, default=None, help="smoke test 用")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    tfm = tfm_fn(args.size)

    # 先掃每個 shard 的列數，決定需要哪些列
    paths = [hf_hub_download(REPO, f"data/train-{i:05d}-of-{SHARDS:05d}.parquet", repo_type="dataset")
             for i in range(SHARDS) if i in (0, SHARDS - 1) or args.nhead > 0]
    counts = [pq.ParquetFile(p).metadata.num_rows for p in paths]
    print("shard rows:", counts)
    # 只下載了 shard 0 與最後一個 shard；若 nhead 超過 shard 0 的列數需要中間 shard
    if args.nhead > counts[0]:
        raise SystemExit("nhead 超過 shard 0，請改下載全部 shard")

    head, tail_buf = [], []
    names_head = []
    for k, (name, b) in enumerate(rows_of_shard(paths[0])):
        if k >= args.nhead or (args.max_rows_per_shard and k >= args.max_rows_per_shard):
            break
        names_head.append(name)
        head.append(tfm(Image.open(io.BytesIO(b)).convert("RGB")))
    assert names_head[0].endswith("000001.jpg"), f"第一張不是 000001.jpg: {names_head[0]}"
    for a, b_ in zip(names_head, names_head[1:]):
        assert int(os.path.splitext(os.path.basename(b_))[0]) == int(os.path.splitext(os.path.basename(a))[0]) + 1, "檔名不連續"

    last_path = paths[-1]
    n_last = pq.ParquetFile(last_path).metadata.num_rows
    start_tail = n_last - args.ntail
    if args.max_rows_per_shard:
        start_tail = max(0, args.max_rows_per_shard - args.ntail) if False else start_tail
    names_tail = []
    for k, (name, b) in enumerate(rows_of_shard(last_path)):
        if k < start_tail:
            continue
        names_tail.append(name)
        tail_buf.append(tfm(Image.open(io.BytesIO(b)).convert("RGB")))
    assert names_tail[-1].endswith(f"{TOTAL:06d}.jpg"), f"最後一張不是 {TOTAL:06d}.jpg: {names_tail[-1]}"

    full = torch.stack(head + tail_buf, 0)
    out = os.path.join(args.out_dir, f"CelebA{args.size}.pt")
    torch.save(full, out)
    print("saved", out, tuple(full.shape), full.dtype, "range", float(full.min()), float(full.max()))
    print("first:", names_head[0], "head_last:", names_head[-1], "tail_first:", names_tail[0], "tail_last:", names_tail[-1])


if __name__ == "__main__":
    main()
