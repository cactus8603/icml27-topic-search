"""Fit the frozen CS-CLIP query rule on DEV, then score TEST once."""
import argparse
from importlib.metadata import version
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import COCO_DIR, jdump
from headroom import CONFIG as MODEL_CONFIG, model_dir, rgb
from headroom_models import ClipEncoder, Encoder, unit_scores
from prospective_manifest import CONFIG, MANIFEST, file_sha, verify_manifest


BASE = "cs_clip_plain"
ADJUSTED = "cs_clip_increment"
SHUFFLE = "cs_clip_anchor_shuffle"
CORE = "core_embed_plain"


def design_inputs():
    config = json.loads(CONFIG.read_text())
    manifest = json.loads(MANIFEST.read_text())
    verify_manifest(manifest, config)
    if file_sha(MODEL_CONFIG) != config["model_config_sha256"]:
        raise ValueError("The frozen checkpoint configuration digest does not match")
    return config, manifest, json.loads(MODEL_CONFIG.read_text())


def image_path(image_id):
    return COCO_DIR / "val2017" / f"{image_id:012d}.jpg"


def preflight(manifest, model_config):
    for arm in ("cs_clip", "core_embed"):
        print(arm, model_dir(model_config["models"][arm]), flush=True)
    for name, split in manifest["splits"].items():
        for image_id, expected in zip(split["candidate_ids"], split["candidate_image_sha256"]):
            if file_sha(image_path(image_id)) != expected:
                raise ValueError(f"Candidate JPEG digest mismatch {name}/{image_id}")
        print(name, len(split["candidate_ids"]), "fixed images verified", flush=True)


def split_scores(split, encoder, model_config, need_controls):
    queries = split["queries"]
    open_images = [lambda i=i: rgb(image_path(i)) for i in split["candidate_ids"]]
    images = unit_scores(encoder, open_images, model_config["batch_size"]["embed_image"], images=True)
    fields = ("text", "foil_text", "anchor", "shuffled_anchor") if need_controls else ("text", "foil_text")
    texts = sorted({q[field] for q in queries for field in fields if q[field] is not None})
    positions = {caption: i for i, caption in enumerate(texts)}
    vectors = unit_scores(encoder, texts, model_config["batch_size"]["embed_text"])
    cosine = vectors @ images.T
    if cosine.shape != (len(texts), len(open_images)) or not np.isfinite(cosine).all():
        raise ValueError("Invalid image-by-text scores")
    return {field: cosine[[positions[q[field] or q["text"]] for q in queries]].astype(np.float64) for field in fields}


def ranks(scores):
    # Candidate IDs in the manifest are ascending. This specifies all exact ties.
    return np.argsort(-scores, axis=1, kind="stable")


def labels(split):
    positions = {i: offset for offset, i in enumerate(split["candidate_ids"])}
    positive_by_query = np.zeros((len(split["queries"]), len(positions)), dtype=bool)
    for row, q in enumerate(split["queries"]):
        positive_by_query[row, [positions[i] for i in q["relevant_ids"]]] = True
    return positive_by_query


def retrieval_metrics(scores, positive, recall_at):
    relevance = np.take_along_axis(positive, ranks(scores), axis=1)
    n_positive = relevance.sum(axis=1)
    if np.any(n_positive == 0):
        raise ValueError("Every query must have an annotated positive")
    ran = np.arange(1, relevance.shape[1] + 1, dtype=float)
    prec = relevance.cumsum(axis=1) / ran
    top = relevance[:, :recall_at].sum(axis=1)
    return {"ap": (prec * relevance).sum(axis=1) / n_positive,
            "recall": top / n_positive, "success": top > 0}


def increment(score, anchor, weight):
    return score + weight * (score - anchor)


def arm_scores(cs, core, weight):
    raw = cs["text"]
    zero = increment(raw, cs["anchor"], 0)
    # A query-only term must preserve every rank. Use the anchor row mean as a nonzero control.
    constant = raw - cs["anchor"].mean(axis=1, keepdims=True)
    original_rank = ranks(raw)
    if not (np.array_equal(raw, zero) and np.array_equal(original_rank, ranks(constant))):
        raise ValueError("No-op rank controls do not match the plain baseline")
    return {BASE: raw, ADJUSTED: increment(raw, cs["anchor"], weight),
            SHUFFLE: increment(raw, cs["shuffled_anchor"], weight), CORE: core["text"]}


def select_on_dev(split, cs, config):
    primary = np.array([q["k"] == config["primary_k"] for q in split["queries"]])
    positive = labels(split)[primary]
    raw, anchor = cs["text"][primary], cs["anchor"][primary]
    values = [(weight, float(retrieval_metrics(increment(raw, anchor, weight), positive, config["recall_at"])["ap"].mean()))
              for weight in config["lambda_grid"]]
    best, _ = max(values, key=lambda v: (v[1], -v[0]))
    return best, {str(weight): 100 * m for weight, m in values}


def bootstrap_comparisons(ap_by_arm, groups, config):
    # Queries sharing the chosen sentinel positive image use one bootstrap group.
    # Both comparisons share the same draws. The metric is macro averaged by query.
    _, group = np.unique(groups, return_inverse=True)
    count = np.bincount(group)
    differences = {ref: ap_by_arm[ADJUSTED] - ap_by_arm[ref] for ref in (BASE, SHUFFLE)}
    idx = np.random.default_rng(config["bootstrap_seed"]).integers(0, len(count), (config["bootstrap_draws"], len(count)))
    den = count[idx].sum(axis=1)
    alpha = (1 - config["comparison_ci_level"]) / 2
    results = {}
    for name, diff in differences.items():
        group_sum = np.bincount(group, weights=diff)
        draws = 100 * group_sum[idx].sum(axis=1) / den
        low, high = np.quantile(draws, [alpha, 1 - alpha])
        results[name] = {"difference_pp": 100 * float(diff.mean()), "lo_pp": float(low), "hi_pp": float(high),
                         "n_queries": len(group), "n_clusters": len(count)}
    return results


def decision(results, min_gain):
    if all(x["difference_pp"] >= min_gain and x["lo_pp"] > 0 for x in results.values()):
        return "PASS"
    if any(x["hi_pp"] < min_gain for x in results.values()):
        return "FAIL"
    return "灰區"


def materialize_stats(split, cs, core, weight, config):
    arm = arm_scores(cs, core, weight)
    positive = labels(split)
    gathered = {name: retrieval_metrics(v, positive, config["recall_at"]) for name, v in arm.items()}
    positions = {i: j for j, i in enumerate(split["candidate_ids"])}
    out = {}
    for k in (config["primary_k"], config["secondary_k"]):
        rows = np.array([j for j, q in enumerate(split["queries"]) if q["k"] == k], dtype=int)
        if not len(rows):
            out[str(k)] = {"n_queries": 0}
            continue
        pos_ids = [split["queries"][j]["paired_positive_id"] for j in rows]
        pos = [positions[i] for i in pos_ids]
        neg = [positions[split["queries"][j]["paired_hard_foil_id"]] for j in rows]
        has_text_foil = np.array([split["queries"][j]["foil_text"] is not None for j in rows], dtype=bool)
        text_pair = {}
        for name, vals in ((BASE, cs), (CORE, core)):
            delta = vals["text"][rows, pos] - vals["foil_text"][rows, pos]
            text_pair[name] = float((delta[has_text_foil] > 0).mean() * 100) if has_text_foil.any() else None
        metrics = {}
        for name, vv in gathered.items():
            metrics[name] = {"map_pp": float(100 * vv["ap"][rows].mean()),
                             f"recall_at_{config['recall_at']}_pp": float(100 * vv["recall"][rows].mean()),
                             f"success_at_{config['recall_at']}_pp": float(100 * vv["success"][rows].mean()),
                             "hard_pair_image_accuracy_pp": float(100 * (arm[name][rows, pos] > arm[name][rows, neg]).mean())}
        result = {"n_queries": len(rows), "n_sentinel_positive_images": len(set(pos_ids)),
                  "annotated_relevant_images_per_query_median": float(np.median(positive[rows].sum(axis=1))),
                  "full_gt_foil_on_same_positive": {"n_queries": int(has_text_foil.sum()), "pass_percent": text_pair}, "arms": metrics}
        if k == config["primary_k"]:
            comparisons = bootstrap_comparisons({name: v["ap"][rows] for name, v in gathered.items()}, pos_ids, config)
            per_query = [{"key": split["queries"][row]["key"], "paired_positive_id": positive_id,
                          "ap_pp": {name: 100 * float(v["ap"][row]) for name, v in gathered.items()}}
                         for row, positive_id in zip(rows, pos_ids)]
            result.update(comparisons=comparisons, state=decision(comparisons, config["min_map_gain_pp"]),
                          per_query=per_query)
        out[str(k)] = result
    return out


def run(args, config, manifest, model_config):
    out = args.out_dir
    provenance = {"manifest_file_sha256": file_sha(MANIFEST), "criteria_file_sha256": file_sha(CONFIG),
                  "model_config_sha256": file_sha(MODEL_CONFIG), "selected_encoder": "cs_clip",
                  "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "device": args.device,
                  "package_versions": {name: version(name) for name in ("torch", "torchvision", "transformers", "numpy", "open-clip-torch")},
                  "frozen_commit": (out / "frozen_commit.txt").read_text().strip() if (out / "frozen_commit.txt").is_file() else None}
    cs = ClipEncoder("cs_clip", model_dir(model_config["models"]["cs_clip"]), model_config, args.device)
    print("Encoding DEV with CS-CLIP", flush=True)
    ds = split_scores(manifest["splits"]["dev"], cs, model_config, need_controls=True)
    weight, grid = select_on_dev(manifest["splits"]["dev"], ds, config)
    jdump({**provenance, "selected_lambda": weight, "dev_cs_clip_map_pp_by_lambda": grid}, out / "dev_calibration.json")
    del ds
    print("DEV weight frozen; encoding TEST with CS-CLIP", flush=True)
    split = manifest["splits"]["test"]
    cs_test = split_scores(split, cs, model_config, need_controls=True)
    del cs
    import torch
    if args.device != "cpu":
        torch.cuda.empty_cache()
    print("Encoding TEST CORE-EMBED reference", flush=True)
    core = Encoder("core_embed", model_dir(model_config["models"]["core_embed"]), model_config, args.device)
    core_test = split_scores(split, core, model_config, need_controls=False)
    results = materialize_stats(split, cs_test, core_test, weight, config)
    jdump({**provenance, "selected_lambda": weight, "state": results[str(config["primary_k"])]["state"],
           "outcome_scope": "frozen incremental ranking rule on new COCO conjunctions",
           "independent_from_headroom_eligibility": True, "test": results}, out / "verdict_prospective_rank.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--out-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config, manifest, model_config = design_inputs()
    if args.out_dir is None:
        preflight(manifest, model_config)
        return
    if not args.out_dir.is_dir() or any(p.name not in ("_source", "frozen_commit.txt") for p in args.out_dir.iterdir()):
        parser.error("Pass a fresh job directory, containing only frozen source provenance")
    try:
        preflight(manifest, model_config)
        run(args, config, manifest, model_config)
    except (FileNotFoundError, ValueError) as exc:
        jdump({"state": "未能測試", "reason": str(exc), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
               "manifest_file_sha256": file_sha(MANIFEST)}, args.out_dir / "verdict_prospective_rank.json")
        raise


if __name__ == "__main__":
    main()
