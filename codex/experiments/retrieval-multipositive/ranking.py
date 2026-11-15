"""Compare four fixed retrieval methods on complete, balanced spatial groups."""

import argparse
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "retrieval-anchor-rerank"))
from evaluate import MODEL_CONFIG_SHA, arms, models, ranks, summarize
from headroom import cross_scores, rgb
from headroom_models import ClipEncoder, CrossScorer, unit_scores


PLAN_SHA = "33363955507ec1c7cad716c3294f41c7892b12d944760631ce42fb85894295bb"
WIDTH = 8
GROUPS = 123
DIRECTIONS = ("left", "right")
SOURCE = "2daedf522c6d64b860277657d3067b61da3a0f2b"


def load_plan(path):
    data = path.read_bytes()
    if sha256(data).hexdigest() != PLAN_SHA:
        raise ValueError("Candidate or relevance manifest differs from the fixed annotation panel")
    return json.loads(data)


def layout(groups, images):
    image_ids, captions, anchors, sham, cluster, relevance = [], [], [], [], [], []
    if len(groups) != GROUPS or len({tuple(group["key"]) for group in groups}) != len(groups):
        raise ValueError("Expected all fixed, unique groups")
    for index, group in enumerate(groups):
        ids = group["images"]
        if len(ids) != WIDTH or ids != sorted(set(ids)):
            raise ValueError("All eight images must appear in their fixed tie-breaking order")
        if [query["direction"] for query in group["queries"]] != list(DIRECTIONS):
            raise ValueError("Each group needs both fixed query directions")
        left, right = [set(query["relevant"]) for query in group["queries"]]
        if len(left) != WIDTH // 2 or len(right) != WIDTH // 2 or left & right or (left | right) != set(ids):
            raise ValueError("The two queries must have complementary groups of four positive images")
        for query, positives in zip(group["queries"], (left, right), strict=True):
            image_ids.append(ids)
            captions.append(query["text"])
            anchors.append(group["anchor"])
            sham.append(group["shuffled_anchor"])
            cluster.append(index)
            relevance.append([image_id in positives for image_id in ids])
    unique = {image_id for group in image_ids for image_id in group}
    if len(unique) != GROUPS * WIDTH or {str(image_id) for image_id in unique} != images.keys():
        raise ValueError("Each JPEG must come from exactly one group in the fixed manifest")
    return np.asarray(image_ids), captions, anchors, sham, np.asarray(cluster), np.asarray(relevance)


def analyze(cross, full, real, shuffled, relevance, cluster):
    if len(cluster) % 2 or not np.array_equal(cluster[::2], cluster[1::2]):
        raise ValueError("Each cluster needs two consecutive mirror queries")
    methods = arms(cross, full, real, shuffled, top_k=WIDTH)
    result = summarize({name: ranks(scores) for name, scores in methods.items()}, relevance, cluster, top_k=WIDTH)
    oracle = result.pop("fixed_top50_oracle_map_pp")
    if not np.isclose(oracle, 100):
        raise ValueError("Label oracle differs from perfect ranking of four known positives per query")
    result.pop("recall_at_50_pp")
    result.pop("queries_without_top50_hit")
    real_a = real[::2]
    if not np.array_equal(real_a, real[1::2]):
        raise ValueError("The anchor must be unchanged when the spatial query is flipped")
    true = np.where(relevance[::2], full[::2], full[1::2])
    foil = np.where(relevance[::2], full[1::2], full[::2])
    result["same_image_cs_clip"] = {"n_images": int(true.size),
                                    "correct_full_gt_foil_pp": float(100 * (true > foil).mean()),
                                    "strict_full_gt_anchor_gt_foil_pp": float(100 * ((true > real_a) & (real_a > foil)).mean())}
    result["fixed_group_label_oracle_map_pp"] = float(oracle)
    result["gap_to_oracle_pp"] = {arm: float(oracle - value) for arm, value in result["map_pp"].items()}
    result["candidates_per_query"] = WIDTH
    result["positives_per_query"] = WIDTH // 2
    return result


def run(panel, assets, config, snapshots):
    import torch

    if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Formal scoring requires one compatible GPU")
    ids, captions, anchors, sham, cluster, relevance = layout(panel["groups"], panel["images"])
    distinct = np.unique(ids)
    images = {int(key): (lambda image_id=int(key): rgb(assets / "images" / f"{image_id:012d}.jpg"))
              for key in distinct}
    batch = config["batch_size"]
    print(f"[codex] Scoring {ids.size} CORE cross-encoder pairs", flush=True)
    reranker = CrossScorer("core_reranker", snapshots["core_reranker"], config, "cuda")
    cross = cross_scores(reranker, [caption for caption in captions for _ in range(WIDTH)],
                         [images[int(key)] for key in ids.flat], batch["cross_pair"]).reshape(ids.shape)
    del reranker
    torch.cuda.empty_cache()

    print(f"[codex] Encoding {len(distinct)} images with fixed CS-CLIP", flush=True)
    clip = ClipEncoder("cs_clip", snapshots["cs_clip"], config, "cuda")
    texts = sorted(set(captions + anchors + sham))
    text_positions = {caption: i for i, caption in enumerate(texts)}
    image_vectors = unit_scores(clip, [images[int(i)] for i in distinct], batch["embed_image"], images=True)
    text_vectors = unit_scores(clip, texts, batch["embed_text"])
    del clip
    torch.cuda.empty_cache()
    dot = text_vectors @ image_vectors.T
    image_pos = np.searchsorted(distinct, ids)
    full, real, shuffled = [dot[np.asarray([text_positions[text] for text in values])[:, None], image_pos]
                            .astype(np.float64) for values in (captions, anchors, sham)]
    return analyze(cross, full, real, shuffled, relevance, cluster)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--receipt-sha256", required=True)
    parser.add_argument("--freeze")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    panel = load_plan(args.plan)
    receipt = args.assets / "image-receipt.json"
    if sha256(receipt.read_bytes()).hexdigest() != args.receipt_sha256:
        raise ValueError("Image verification receipt differs from the frozen bytes")
    layout(panel["groups"], panel["images"])
    config, snapshots = models()
    context = {"plan_sha256": PLAN_SHA, "receipt_sha256": args.receipt_sha256,
               "ranking_method_source": SOURCE, "model_config_sha256": MODEL_CONFIG_SHA,
               "model_revisions": {name: snapshots[name].name for name in ("cs_clip", "core_reranker")},
               "coverage": panel["coverage"]}
    if args.check_only:
        if args.freeze or args.output:
            parser.error("Check-only does not perform model scoring")
        print(json.dumps(context, sort_keys=True))
        return
    job = os.environ.get("SLURM_JOB_ID", "")
    if not re.fullmatch(r"[1-9][0-9]*", job) or not re.fullmatch(r"[0-9a-f]{40}", args.freeze or "") or not args.output:
        parser.error("Formal scoring requires the Slurm job, full frozen commit and new output directory")
    args.output.mkdir(parents=True)
    result = run(panel, args.assets, config, snapshots)
    report = context | result | {"freeze": args.freeze, "job": job, "scope": "fixed COCO spatial groups all eight ranked",
                                "runtime": {name: version(name) for name in ("numpy", "torch", "transformers")}}
    payload = (json.dumps(report, sort_keys=True, allow_nan=False) + "\n").encode()
    with (args.output / "summary.json").open("xb") as handle:
        handle.write(payload)
    print(f"[codex] Multipositive ranking summary SHA-256 {sha256(payload).hexdigest()}", flush=True)


if __name__ == "__main__":
    main()
