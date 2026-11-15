"""One GPU pass over the exposed 8,191-image pool; every later analysis runs on CPU.

Writes to --out (must not exist):
  vectors.npz   CORE-EMBED and CS-CLIP unit vectors for all images, full queries and single-condition texts
  cross.npz     CORE-RERANKER log-odds for each query x CORE-EMBED top-depth candidate (aligned with core_order)
  panel.json    ids, queries (text, key, relevant, group), condition texts, image categories (COCO, >=2% area)
  meta.json     versions, device, timings, peak memory, bundle commits, replay check against 26601/26806

The full queries, images and top-50 cross pairs are encoded in exactly the 26806 order and batch sizes,
so the CORE baseline must replay the published 54.53 / 74.02 / 73.77 (31 queries without a hit).
"""

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
from time import monotonic

import numpy as np


BUNDLE = Path(os.environ.get("DEV_BUNDLE", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(BUNDLE / "codex" / "experiments" / "retrieval-recall-budget"))
import score  # noqa: E402  (frozen 26806 stage; loads the anchor-rerank evaluator as score.original)
import prepare as anchor_prepare  # noqa: E402  (imported by the evaluator; its prior.prior is the COCO reader)
from e2_build_queries import joint_text  # noqa: E402

original = score.original
K = original.K
DEPTH = 200


def package_version(name):
    try:
        return version(name)
    except Exception:  # open_clip lives in an extra site directory without metadata on some hosts
        return None


def condition_texts(panel):
    names = {int(cat): name for cat, name in panel["names"].items()}
    per_query = [[joint_text([names[cat]]) for cat in q["key"]] for q in panel["queries"]]
    unique = sorted({text for pair in per_query for text in pair})
    position = {text: i for i, text in enumerate(unique)}
    return unique, [[position[text] for text in pair] for pair in per_query]


def image_categories(ids):
    """COCO categories present per image, with the panel's own rule (non-crowd, area >= 2%)."""
    coco = anchor_prepare.prior.prior
    wanted = set(ids)
    shapes = {row["id"]: [row["width"], row["height"]] for row in coco.coco_rows("images") if row["id"] in wanted}
    if set(shapes) != wanted:
        raise ValueError("Some pool images are missing from the COCO annotations")
    _, found = coco.categories({"pool": shapes})
    return {image_id: sorted(found[image_id]) for image_id in ids}


def check_relevance(panel, categories):
    for q in panel["queries"]:
        expected = sorted(i for i in panel["ids"] if set(q["key"]) <= set(categories[i]))
        if expected != q["relevant"]:
            raise ValueError(f"Category rule does not reproduce the frozen relevance for {q['key']}")


def encode(name, model_class, snapshot, config, images, groups):
    import torch

    torch.cuda.reset_peak_memory_stats()
    started = monotonic()
    encoder = model_class(name, snapshot, config, "cuda")
    image_vectors = original.unit_scores(encoder, images, config["batch_size"]["embed_image"], images=True)
    texts = [original.unit_scores(encoder, group, config["batch_size"]["embed_text"]) for group in groups]
    timing = score.timeline(torch, started, len(images), "indexed_images")
    del encoder
    torch.cuda.empty_cache()
    return image_vectors, texts, timing


def run(out, depth):
    import torch

    panel, config, snapshots, context = score.preflight()
    ids, queries = panel["ids"], panel["queries"]
    captions = [q["text"] for q in queries]
    conditions, condition_index = condition_texts(panel)
    images = [lambda key=key: original.rgb(original.image_path(key)) for key in ids]
    print(f"Reading COCO categories for {len(ids)} pool images", flush=True)
    categories = image_categories(ids)
    check_relevance(panel, categories)

    vectors, timings = {}, {}
    for name, model in (("core_embed", original.Encoder), ("cs_clip", original.ClipEncoder)):
        print(f"Encoding pool, {len(captions)} queries and {len(conditions)} condition texts with {name}", flush=True)
        image_vectors, (full, single), timings[name] = encode(
            name, model, snapshots[name], config, images, (captions, conditions))
        vectors |= {f"{name}_image": image_vectors, f"{name}_full": full, f"{name}_condition": single}

    core = original.ranks(vectors["core_embed_full"] @ vectors["core_embed_image"].T)
    top = core[:, :depth]
    batch = config["batch_size"]["cross_pair"]
    torch.cuda.reset_peak_memory_stats()
    started = monotonic()
    scorer = original.CrossScorer("core_reranker", snapshots["core_reranker"], config, "cuda")
    print(f"Scoring {top[:, :K].size} top-{K} pairs in the 26806 order", flush=True)
    first = original.cross_scores(scorer, [c for c in captions for _ in range(K)],
                                  [images[i] for i in top[:, :K].flat], batch).reshape(len(queries), K)
    extra = top[:, K:]
    print(f"Scoring {extra.size} pairs at CORE ranks {K + 1}-{depth}", flush=True)
    rest = original.cross_scores(scorer, [c for c in captions for _ in range(depth - K)],
                                 [images[i] for i in extra.flat], batch).reshape(extra.shape)
    timings["core_reranker"] = score.timeline(torch, started, top.size, "pairs")
    cross = np.concatenate((first, rest), axis=1)

    inverse = {key: col for col, key in enumerate(ids)}
    relevance = np.zeros(core.shape, dtype=bool)
    for row, q in enumerate(queries):
        relevance[row, [inverse[key] for key in q["relevant"]]] = True
    base = score.aggregate(score.query_metrics(score.rerank_shortlist(core, top[:, :K], first), relevance, K),
                           relevance.sum(axis=1))
    try:
        score.check_replay(base)
        replay = {"state": "PASS", "core": base}
    except ValueError as error:
        replay = {"state": "MISMATCH", "core": base, "error": str(error)}

    np.savez(out / "vectors.npz", **vectors)
    np.savez(out / "cross.npz", core_order=core.astype(np.int32), cross=cross)
    keep = ("text", "key", "relevant", "group", "anchor", "shuffled_anchor")
    (out / "panel.json").write_text(json.dumps({
        "ids": ids, "queries": [{k: q[k] for k in keep if k in q} for q in queries],
        "condition_texts": conditions, "condition_index": condition_index,
        "image_categories": {str(k): v for k, v in categories.items()}}, sort_keys=True))
    meta = context | {
        "depth": depth, "top_k": K, "timings": timings, "replay": replay,
        "device": torch.cuda.get_device_name(0), "bf16": torch.cuda.is_bf16_supported(),
        "versions": {name: package_version(name) for name in ("numpy", "torch", "transformers", "open_clip_torch")},
        "bundle": {key: os.environ.get(f"DEV_{key.upper()}") for key in ("source", "freeze", "commit", "dirty")},
        "scope": "exposed 8,191-image pool; development only, not formal evidence"}
    (out / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True, default=str))
    print(json.dumps({"replay": replay["state"], **base}), flush=True)
    if replay["state"] != "PASS":
        sys.exit("Baseline did not replay the published 26806 CORE numbers; fix parity before analysing")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true", help="CPU: verify panel, models and source receipt")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--depth", type=int, default=DEPTH)
    args = parser.parse_args()
    if args.preflight:
        panel, _, _, context = score.preflight()
        check_relevance(panel, image_categories(panel["ids"]))
        conditions, _ = condition_texts(panel)
        print(json.dumps(context | {"relevance_rebuilt_from_coco": True, "condition_texts": len(conditions)},
                         sort_keys=True, default=str))
        return
    if args.out is None or not K < args.depth <= 8191:
        parser.error("--run needs --out and K < --depth <= pool size")
    args.out.mkdir(parents=True, exist_ok=False)
    run(args.out, args.depth)


if __name__ == "__main__":
    main()
