"""Evaluate a fixed-index, two-object conjunction rule on independent COCO pools."""

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare import DATA, HERE, PARTS, QUERY_MINIMUM, file_sha, image_path, stats
from e2_build_queries import joint_text
from headroom import CONFIG, model_dir, rgb
from headroom_models import Encoder, unit_scores


MODEL_CONFIG_SHA256 = "862591c60c5609f16727d902993df0cba9cab96ded72611d69d263a02fe4ad1d"
ORACLE_K = (10, 50, 200)
GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
ARMS = ("core_full", "component_mean", "component_min")
PRIMARY = ARMS[2]
MIN_GAIN_PP = 2.0
SEED = 2026111531


def ranking(score):
    return np.argsort(-score, axis=1, kind="stable")


def percentile(scores):
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite embedding scores")
    order = np.argsort(scores, axis=1, kind="stable")
    sorted_scores = np.take_along_axis(scores, order, axis=1)
    count = scores.shape[1]
    result = np.zeros_like(scores, dtype=float)
    for ranks, positions, row in zip(result, order, sorted_scores, strict=True):
        boundaries = np.r_[0, 1 + np.flatnonzero(row[:-1] != row[1:]), count]
        midpoints = (boundaries[:-1] + boundaries[1:] + 1) / (2 * count)
        ranks[positions] = np.repeat(midpoints, np.diff(boundaries))
    return result


def average_precision(relevance):
    positives = relevance.sum(axis=1)
    if not np.all(positives > 0):
        raise ValueError("Every evaluated query requires a positive")
    at_rank = relevance.cumsum(axis=1) / np.arange(1, relevance.shape[1] + 1)
    return (at_rank * relevance).sum(axis=1) / positives


def sorted_labels(score, labels):
    return np.take_along_axis(labels, ranking(score), axis=1)


def bound(relevance, top_k):
    # Exact upper bound with the original top-k membership and original tail order.
    available = relevance[:, :top_k].sum(axis=1)
    best_prefix = np.arange(top_k)[None, :] < available[:, None]
    reconstructed = np.concatenate((best_prefix, relevance[:, top_k:]), axis=1)
    return average_precision(reconstructed), available / relevance.sum(axis=1)


def scores_for(encoder, split, names, model_config):
    ids = split["ids"]
    images = [lambda image_id=image_id: rgb(image_path(DATA, image_id)) for image_id in ids]
    batch = model_config["batch_size"]
    visual = unit_scores(encoder, images, batch["embed_image"], images=True)
    keys = [tuple(q["key"]) for q in split["queries"]]
    captions = [joint_text([names[str(cat)] for cat in pair]) for pair in keys]
    cats = sorted({cat for key in keys for cat in key})
    single = [joint_text([names[str(cat)]]) for cat in cats]
    offsets = {cat: i for i, cat in enumerate(cats)}
    text = unit_scores(encoder, captions + single, batch["embed_text"])
    raw, per_cat = np.split(text @ visual.T, [len(keys)])
    singles = percentile(per_cat)
    first = singles[[offsets[key[0]] for key in keys]]
    second = singles[[offsets[key[1]] for key in keys]]
    full = percentile(raw)
    if not np.array_equal(ranking(raw), ranking(full)):
        raise ValueError("Percentile conversion changed the baseline ranking")
    labels = np.zeros_like(raw, dtype=bool)
    locations = {image_id: index for index, image_id in enumerate(ids)}
    for row, q in enumerate(split["queries"]):
        labels[row, [locations[image_id] for image_id in q["relevant"]]] = True
    return (full, (first + second) / 2, np.minimum(first, second)), labels


def mix(full, components, weight):
    return (1 - weight) * full + weight * components


def choose(arms, labels):
    full = arms[0]
    grids = {}
    for name, component in zip(ARMS[1:], arms[1:], strict=True):
        values = [float(100 * average_precision(sorted_labels(mix(full, component, w), labels)).mean()) for w in GRID]
        best = max(range(len(GRID)), key=lambda j: (values[j], -GRID[j]))
        grids[name] = {"selected_weight": GRID[best], "map_pp_by_weight": dict(zip(GRID, values, strict=True))}
    return grids


def paired_test(ap_by_arm, query_groups):
    _, indexes = np.unique(query_groups, return_inverse=True)
    counts = np.bincount(indexes)
    draws = np.random.default_rng(SEED).integers(0, len(counts), (10000, len(counts)))
    sample_sizes = counts[draws].sum(axis=1)
    comparisons = {}
    for name in ARMS[:2]:
        change = 100 * (ap_by_arm[PRIMARY] - ap_by_arm[name])
        sums = np.bincount(indexes, weights=change)
        intervals = np.quantile(sums[draws].sum(axis=1) / sample_sizes, [0.0125, 0.9875])
        comparisons[name] = {"diff_pp": float(change.mean()), "lo_pp": float(intervals[0]), "hi_pp": float(intervals[1])}
    if all(x["diff_pp"] >= MIN_GAIN_PP and x["lo_pp"] > 0 for x in comparisons.values()):
        state = "PASS"
    elif any(x["hi_pp"] < MIN_GAIN_PP for x in comparisons.values()):
        state = "FAIL"
    else:
        state = "灰區"
    return {"state": state, "comparisons_vs_component_min": comparisons, "clusters": len(counts)}


def evaluate(arms, labels, grids, groups):
    full = arms[0]
    order = {"core_full": sorted_labels(full, labels)}
    for name, component in zip(ARMS[1:], arms[1:], strict=True):
        order[name] = sorted_labels(mix(full, component, grids[name]["selected_weight"]), labels)
    ap = {name: average_precision(rel) for name, rel in order.items()}
    values = {name: {"map_pp": float(100 * score.mean()),
                     "recall_at_50_pp": float(100 * (order[name][:, :50].sum(axis=1) / labels.sum(axis=1)).mean())}
              for name, score in ap.items()}
    limits = {}
    for k in ORACLE_K:
        ceiling, recall = bound(order["core_full"], k)
        limits[k] = {"map_ceiling_pp": float(100 * ceiling.mean()),
                     "reordering_headroom_pp": float(100 * (ceiling - ap["core_full"]).mean()),
                     "missed_positives_fraction_pp": float(100 * (1 - recall).mean()),
                     "queries_without_any_shortlisted_positive": int((recall == 0).sum())}
    subgroup = {"single_positive": labels.sum(axis=1) == 1, "multi_positive": labels.sum(axis=1) > 1}
    return paired_test(ap, groups) | {"arms": values, "top_k_core_oracle": limits,
                                    "subgroups": {tag: {name: float(100 * score[rows].mean()) if rows.any() else None
                                                        for name, score in ap.items()} for tag, rows in subgroup.items()}}


def preflight():
    panel = json.loads((HERE / "panel.json").read_text())
    manifest_path = DATA / "manifest.json"
    if file_sha(manifest_path) != panel["manifest_sha256"] or file_sha(CONFIG) != MODEL_CONFIG_SHA256:
        raise ValueError("Expected the frozen panel and model configuration")
    config = json.loads(CONFIG.read_text())
    snapshot = model_dir(config["models"]["core_embed"])
    manifest = json.loads(manifest_path.read_text())
    previous_ids, previous_keys, previous_shas = set(), set(), set()
    if set(manifest["parts"]) != set(PARTS):
        raise ValueError("Unexpected image partitions")
    for part, subset in manifest["parts"].items():
        ids = subset["ids"]
        id_set = set(ids)
        image_shas = subset["jpg_sha256"]
        sha_set = set(image_shas)
        keys = {tuple(q["key"]) for q in subset["queries"]}
        groups = {q["group"] for q in subset["queries"]}
        floor_queries, floor_groups = QUERY_MINIMUM[part]
        if (len(ids) != len(id_set) or ids != sorted(ids) or len(ids) != len(sha_set)
                or not PARTS[part] * .95 <= len(ids) <= PARTS[part] or len(keys) != len(subset["queries"])
                or id_set & previous_ids or keys & previous_keys or sha_set & previous_shas
                or len(keys) < floor_queries or len(groups) < floor_groups):
            raise ValueError(f"Invalid image or query partition for {part}")
        for q in subset["queries"]:
            if not set(q["relevant"]) <= id_set or q["group"] not in q["relevant"] or len(q["key"]) != 2:
                raise ValueError("Invalid relevance for the fixed candidate pool")
        for image_id, sha in zip(ids, image_shas, strict=True):
            if file_sha(image_path(DATA, image_id)) != sha:
                raise ValueError(f"JPEG identity check failed for {image_id}")
        previous_ids |= id_set
        previous_keys |= keys
        previous_shas |= sha_set
    return manifest, config, snapshot, panel


def write(path, values):
    with path.open("x") as out:
        json.dump(values, out, sort_keys=True, allow_nan=False)
        out.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--output", type=Path)
    parser.add_argument("--freeze")
    args = parser.parse_args()
    manifest, config, snapshot, panel = preflight()
    context = {"panel_sha256": panel["manifest_sha256"], "coverage": stats(manifest),
               "core_revision": snapshot.name, "model_config_sha256": MODEL_CONFIG_SHA256}
    if args.preflight:
        print(json.dumps(context, sort_keys=True))
        return
    job = os.environ.get("SLURM_JOB_ID", "")
    expected = DATA.parent.parent / "runs" / "retrieval-conjunction" / job
    if not job.isascii() or not job.isdecimal() or args.output != expected or not args.freeze:
        parser.error("Use a new recorded Slurm job and frozen commit")
    args.output.mkdir(parents=True, exist_ok=False)
    provenance = context | {"freeze": args.freeze, "slurm_job_id": job,
                            "environment": {name: version(name) for name in
                                            ("torch", "numpy", "transformers")}}
    model = Encoder("core_embed", snapshot, config, "cuda")
    names = manifest["names"]
    print("Scoring development pool", flush=True)
    dev_scores, dev_labels = scores_for(model, manifest["parts"]["dev"], names, config)
    grids = choose(dev_scores, dev_labels)
    write(args.output / "dev_weights.json", provenance | {"dev_grids": grids})
    print("Development weights recorded; scoring independent test pool", flush=True)
    test = manifest["parts"]["test"]
    arms, labels = scores_for(model, test, names, config)
    result = provenance | {"dev_grids": grids, "test": evaluate(arms, labels, grids, [q["group"] for q in test["queries"]])}
    pending = args.output / "summary.partial"
    write(pending, result)
    pending.replace(args.output / "summary.json")


if __name__ == "__main__":
    main()
