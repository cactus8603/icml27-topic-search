"""E2：由 COCO val2017 instances 標註建立 k=1..6 的多條件查詢（CPU，不用 GPU）。

條件 = COCO 類別（影像中至少一個非 crowd 實例、面積 >= min_area_frac 才算「存在」）。
每個查詢：Q（k 個類別）、正例影像 I*（含 Q 全部）、混淆影像 J（恰含 Q 中 k-1 個、缺 1 個）。
另存 rel：含 Q 全部類別的影像（供 recall@50 使用）。
全程只用 rng(seed)，不看任何模型分數。
用法：python scripts/e2_build_queries.py --seed-key seed|addon_seed --out results/e2/queries_main.json
"""
import argparse
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import COCO_DIR, params, jdump

NAMES_OVERRIDE = {}  # 不改名，直接使用 COCO 類別名


def joint_text(cats):
    names = [c for c in cats]
    if len(names) == 1:
        body = names[0]
    elif len(names) == 2:
        body = f"{names[0]} and {names[1]}"
    else:
        body = ", ".join(names[:-1]) + f" and {names[-1]}"
    return f"a photo of {body}"


def unit_text(cat):
    return f"a photo of {cat}"


def load_annotations(zip_path=None, ann_json=None):
    if ann_json:
        return json.loads(Path(ann_json).read_text())
    with zipfile.ZipFile(zip_path or COCO_DIR / "annotations_trainval2017.zip") as z:
        with z.open("annotations/instances_val2017.json") as f:
            return json.load(f)


def image_category_sets(ann, min_area_frac):
    cats = {c["id"]: c["name"] for c in ann["categories"]}
    imgs = {i["id"]: i for i in ann["images"]}
    present = defaultdict(set)
    for a in ann["annotations"]:
        if a.get("iscrowd", 0):
            continue
        im = imgs[a["image_id"]]
        if a["area"] / (im["width"] * im["height"]) >= min_area_frac:
            present[a["image_id"]].add(a["category_id"])
    image_ids = sorted(imgs)  # 含無類別影像，全部當作檢索池
    return cats, image_ids, {i: frozenset(present.get(i, ())) for i in image_ids}


def build(ann, P, seed):
    cats, image_ids, C = image_category_sets(ann, P["min_area_frac"])
    pos = {i: n for n, i in enumerate(image_ids)}
    rng = np.random.default_rng(seed)
    out = {"image_ids": image_ids, "categories": {str(k): v for k, v in cats.items()}, "queries": {}, "seed": seed}
    for k in P["k_list"]:
        eligible = [i for i in image_ids if len(C[i]) >= k]
        qs, seen, attempts = [], set(), 0
        while len(qs) < P["n_per_k"] and attempts < P["max_attempts_per_k"] and eligible:
            attempts += 1
            istar = eligible[rng.integers(len(eligible))]
            Q = tuple(sorted(rng.choice(sorted(C[istar]), size=k, replace=False).tolist()))
            if (istar, Q) in seen:
                continue
            Qs = set(Q)
            conf = [j for j in image_ids if j != istar and len(Qs & C[j]) == k - 1]
            if not conf:
                continue
            seen.add((istar, Q))
            j = conf[rng.integers(len(conf))]
            rel = [pos[i] for i in image_ids if Qs <= C[i]]
            qs.append({"istar": pos[istar], "j": pos[j], "cats": list(Q),
                       "cat_names": [cats[c] for c in Q], "rel": rel})
        out["queries"][str(k)] = qs
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-key", default="seed", choices=["seed", "addon_seed"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--ann-json", help="測試用：直接提供標註 JSON")
    ap.add_argument("--n-per-k", type=int)
    args = ap.parse_args()
    P = dict(params()["p2"])
    if args.n_per_k:
        P["n_per_k"] = args.n_per_k
    ann = load_annotations(ann_json=args.ann_json)
    res = build(ann, P, P[args.seed_key])
    jdump(res, args.out)
    print({k: len(v) for k, v in res["queries"].items()})


if __name__ == "__main__":
    main()
