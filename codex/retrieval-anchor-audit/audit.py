"""CPU-only Half-Truths anchor diagnosis on the frozen legacy embeddings."""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from pyarrow import parquet as pq

ROOT = Path(__file__).resolve().parents[2]
TOPIC = ROOT / "killtests/compositional-retrieval"
sys.path.insert(0, str(TOPIC / "scripts"))
from analyze_ht import pool_all, scores as stage1_scores
from common import SUBSETS, cor_flags, paired_bootstrap_diff
from probe2_condpmi import stilde


MODELS = ("clip_b32", "cs_clip")
FIELDS = ("full", "anchor", "half")


class Access:
    def __init__(self):
        self.reads = []

    def record(self, path):
        self.reads.append({"file": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
                           "time_taipei": datetime.now(ZoneInfo("Asia/Taipei")).isoformat(timespec="seconds")})
        return path

    def json(self, path):
        return json.loads(self.record(path).read_text())

    def npz(self, path):
        with np.load(self.record(path), allow_pickle=False) as source:
            return dict(source)

    def parquet_columns(self, path):
        return pq.read_table(self.record(path), columns=["id", "image_sha256", "triplet_type", "foil_type"]).to_pydict()


def flags(score):
    full, anchor, half = score.T
    return cor_flags(full, anchor, half)


def pct(value, n):
    return 100.0 * float(value) / n if n else None


def decomposition(score):
    if score.ndim != 2 or score.shape[1] != 3 or not np.isfinite(score).all():
        raise ValueError("expected three finite scores per row")
    full, anchor, half = score.T
    high = anchor >= full
    low = half >= anchor
    pair = full > half
    correct = flags(score)
    events = {
        "cor": correct,
        "full_gt_half": pair,
        "full_lt_half": full < half,
        "anchor_ge_full": high,
        "half_ge_anchor": low,
        "both_anchor_failures": high & low,
        "only_anchor_ge_full": high & ~low,
        "only_half_ge_anchor": low & ~high,
        "anchor_eq_full": anchor == full,
        "half_eq_anchor": half == anchor,
        "full_eq_half": full == half,
        "any_adjacent_tie": (anchor == full) | (half == anchor),
    }
    total_counts = {k: int(v.sum()) for k, v in events.items()}
    conditional_counts = {k: int((v & pair).sum()) for k, v in events.items() if k not in ("full_gt_half", "full_lt_half", "full_eq_half")}
    n, pair_n, cor_n = len(score), total_counts["full_gt_half"], total_counts["cor"]
    gap = pair_n - cor_n
    if np.any(correct & ~pair) or conditional_counts["both_anchor_failures"] or gap != conditional_counts["anchor_ge_full"] + conditional_counts["half_ge_anchor"]:
        raise AssertionError("COR, endpoint order, and conditional failure partition must agree")
    return {"n": n, "count": total_counts, "percent": {k: pct(v, n) for k, v in total_counts.items()},
            "conditional_on_full_gt_half": {"n": pair_n, "count": conditional_counts,
                                            "percent": {k: pct(v, pair_n) for k, v in conditional_counts.items()}},
            "per_row_anchor_oracle_cor_pp": pct(pair_n, n), "per_row_anchor_oracle_gain_pp": pct(gap, n),
            "recoverable_rows_under_per_row_oracle": gap}


def margins(score):
    full, anchor, half = score.astype(np.float64).T
    return half - anchor, full - anchor


def valid_with_offset(score, offset):
    lower, upper = margins(score)
    return (lower < offset) & (offset < upper)


def best_offset(score):
    """Maximize the number of strict open intervals containing one offset."""
    lower, upper = margins(score)
    possible = lower < upper
    border = np.unique(np.r_[lower[possible], upper[possible]])
    candidates = np.r_[0.0, (border[:-1] + border[1:]) / 2]
    counts = ((lower[:, None] < candidates) & (candidates < upper[:, None])).sum(axis=0)
    # Use the evaluated midpoint nearest zero among equal counts. Break sign ties by smaller offset.
    winner = np.lexsort((candidates, np.abs(candidates), -counts))[0]
    return {"offset": float(candidates[winner]), "cor_count": int(counts[winner]), "cor_pp": pct(counts[winner], len(score))}


def offset_diagnostics(scores, aligned_sha, nb, seed):
    out = {}
    for test in SUBSETS:
        train = [sub for sub in SUBSETS if sub != test]
        masks = {sub: ~np.isin(aligned_sha[sub], aligned_sha[test]) for sub in train}
        fit_rows = np.concatenate([scores[sub][masks[sub]] for sub in train])
        fit = best_offset(fit_rows)
        test_scores = scores[test]
        corrected = valid_with_offset(test_scores, fit["offset"])
        original = flags(test_scores)
        out[test] = {"training_rows_by_split": {sub: int(masks[sub].sum()) for sub in train},
                     "drop_overlapping_image_rows_by_split": {sub: int((~masks[sub]).sum()) for sub in train},
                     "train_fit": fit, "held_out_cor_pp": pct(corrected.sum(), len(original)),
                     "held_out_net_change_pp": paired_bootstrap_diff(corrected, original, nb, seed, groups=aligned_sha[test]),
                     "held_out_rescued_rows": int((corrected & ~original).sum()),
                     "held_out_broken_rows": int((original & ~corrected).sum()),
                     "same_split_single_offset_ceiling_descriptive": best_offset(test_scores)}
    return out


def check_equivalent(actual, expected, description):
    if abs(actual - expected) > 1e-7:
        raise ValueError(f"frozen reference mismatch ({description}): {actual} != {expected}")


def run():
    io = Access()
    assets = io.json(TOPIC / "configs/assets_lock.json")["kbora/Half-Truths"]
    criteria = io.json(TOPIC / "configs/params.json")["gate_g3"]
    frozen_dir = TOPIC / "results/headroom/26393"
    frozen = {name: io.json(frozen_dir / f"{name}.json") for name in ("cs_clip", "core_embed", "siglip2", "clic", "core_reranker")}
    dataset = {sub: io.parquet_columns(Path(assets["snapshot"]) / "data" / f"{sub}.parquet") for sub in SUBSETS}
    output = {"dataset_revision": assets["sha"], "frozen_headroom_verdict": io.json(frozen_dir / "verdict_headroom.json"),
              "decomposition": {}, "anchor_offset_diagnostic": {}, "summary_only_other_models": {}, "verification": {}}
    for name in ("core_embed", "siglip2", "clic", "core_reranker"):
        ref = frozen[name]
        if ref["dataset_revision"] != assets["sha"] or ref["attempt_id"] != "26393":
            raise ValueError(f"wrong dataset or attempt for {name}")
        output["summary_only_other_models"][name] = {sub: {
            "n": stat["cor"]["n"], "cor_pp": stat["cor"]["point"], "full_gt_half_pp": stat["pairwise_accuracy_pp"],
            "per_row_anchor_oracle_gain_pp": stat["pairwise_accuracy_pp"] - stat["cor"]["point"],
            "adjacent_tie_count": stat["ties"],
        } for sub, stat in ref["half_truths"].items()}
    for model in MODELS:
        old = io.json(TOPIC / "results/ht" / f"analysis_{model}.json")
        probe2 = io.json(TOPIC / "results/probe2" / f"analysis_{model}.json")
        z = {sub: io.npz(TOPIC / "results/ht" / f"{model}_{sub}.npz") for sub in SUBSETS}
        coco_pool = io.npz(TOPIC / "results/probe2" / f"pool_coco5000_{model}.npz")["img"]
        pool = pool_all(z)
        if len(pool) != probe2["pool_size"] or len(coco_pool) != probe2["pool_coco5000_size"]:
            raise ValueError("wrong PMI pool size")
        outputs = output["decomposition"][model] = {}
        holistic_scores = {}
        shas = {}
        for sub, arr in z.items():
            ref = dataset[sub]
            columns = {"ids": "id", "sha": "image_sha256", "foil_type": "foil_type", "triplet_type": "triplet_type"}
            for col, target in columns.items():
                if not np.array_equal(arr[col], ref[target]):
                    raise ValueError(f"{model}/{sub} row order or labels mismatch ({col})")
            shas[sub] = arr["sha"]
            old_agg = old["subsets"][sub]["aggs"]
            decomposition_sub = outputs[sub] = {}
            stage1 = stage1_scores(arr, pool)
            vectors = {"holistic": stage1["holistic"], "marginal_pmi": stage1["llr_sum"]}
            for method, image_pool in (("whole_caption_pmi_ht", pool), ("whole_caption_pmi_coco5000", coco_pool)):
                vectors[method] = tuple(stilde(arr["img"], arr[k], image_pool, float(arr["tau"])) for k in FIELDS)
            for method, v in vectors.items():
                s = np.asarray(v).T
                if method == "holistic":
                    holistic_scores[sub] = s
                nref = len(arr["ids"])
                if s.shape != (nref, len(FIELDS)):
                    raise ValueError(f"{model}/{sub}/{method}: invalid shape")
                agg = decomposition_sub[method] = decomposition(s)
                for group in ("triplet_type", "foil_type"):
                    agg[f"by_{group}"] = {str(g): decomposition(s[arr[group] == g]) for g in np.unique(arr[group])}
                if method in ("holistic", "marginal_pmi"):
                    key = "holistic" if method == "holistic" else "llr_sum"
                    if not np.array_equal(flags(s), old_agg[key]["row_cor_flags"]):
                        raise ValueError(f"{model}/{sub}/{method}: frozen row COR flags differ")
                if method != "marginal_pmi":
                    p2 = probe2["subsets"][sub]
                    if method == "holistic":
                        cor, pair = p2["cor_holistic"]["point"], p2["pairwise_acc_holistic"]
                    elif method == "whole_caption_pmi_ht":
                        cor, pair = p2["cor_stilde"]["point"], p2["pairwise_acc_stilde"]
                    else:
                        cor, pair = p2["coco5000_pool"]["cor_stilde"]["point"], p2["coco5000_pool"]["pairwise_acc"]
                    check_equivalent(agg["percent"]["cor"], cor, f"{model}/{sub}/{method}/COR")
                    check_equivalent(agg["percent"]["full_gt_half"], pair, f"{model}/{sub}/{method}/full>half")
                if model == "cs_clip" and method == "holistic":
                    record = frozen[model]["half_truths"][sub]
                    check_equivalent(agg["percent"]["cor"], record["cor"]["point"], f"26393/{sub}/COR")
                    check_equivalent(agg["percent"]["full_gt_half"], record["pairwise_accuracy_pp"], f"26393/{sub}/full>half")
            output["verification"][f"{model}/{sub}"] = {"aligned_cols": columns, "n": nref, "row_cor_flags": "exact match for holistic and marginal PMI", "split_metrics": "match frozen Probe 2 and CS-CLIP 26393 where applicable"}
        # Holistic diagnostics only. HT-pool PMI uses held-out images in its unsupervised normalization.
        output["anchor_offset_diagnostic"][model] = offset_diagnostics(holistic_scores, shas, criteria["n_boot"], criteria["boot_seed"])
    output["reads"] = io.reads
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "results/anchor_audit.json")
    args = parser.parse_args()
    data = run()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
