import argparse
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from scipy.stats import rankdata


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXPERIMENT = ROOT / "killtests/multifidelity-attribution"
CAL = slice(0, 50)
TEST = slice(50, 400)
FIT_MASK = slice(0, 100)
TEST_MASK = slice(100, 200)
BUDGET = 10
REFERENCE = 0.04
CHEAP_NAMES = ("grad_dot", "tracin", "trak1", "trak5", "ekfac_auto", "ekfac_1e-8", "ekfac_1e-3")


def now():
    return datetime.now(ZoneInfo("Asia/Taipei")).isoformat(timespec="seconds")


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def checked(data, shape, label):
    require(data.shape == shape, f"{label} wrong shape {data.shape}; expected {shape}")
    require(np.isfinite(data).all(), f"{label} has nonfinite values")
    return data


def read_array(path, shape, mmap=False):
    return checked(np.load(path, mmap_mode="r" if mmap else None), shape, path)


def lds(predicted, losses):
    rp = rankdata(predicted, axis=0).astype(np.float64)
    rt = rankdata(losses, axis=0).astype(np.float64)
    rp -= rp.mean(axis=0)
    rt -= rt.mean(axis=0)
    den = np.linalg.norm(rp, axis=0) * np.linalg.norm(rt, axis=0)
    require(np.all(den > 0), f"undefined LDS in query columns {np.flatnonzero(den == 0).tolist()}")
    return (rp * rt).sum(axis=0) / den


def ten_directions(gram, k, budget):
    values, vectors = np.linalg.eigh(gram[:k, :k])
    eig = values[-budget:][::-1]
    u = vectors[:, -budget:][:, ::-1]
    require(eig[-1] > 0 and eig[-1] > values[-budget - 1], "ambiguous or nonpositive budget cutoff")
    pivot = np.argmax(np.abs(u), axis=0)
    u = u * np.sign(u[pivot, np.arange(budget)])
    coords = (u.T @ gram[:k, :]) / eig[:, None]
    checks = {
        "U_orthogonal_max_abs_error": float(np.max(np.abs(u.T @ u - np.eye(budget)))),
        "D_calibration_matches_U_transpose_max_abs_error": float(np.max(np.abs(coords[:, :k] - u.T))),
        "lambda_11": float(values[-budget - 1]),
    }
    require(max(checks["U_orthogonal_max_abs_error"], checks["D_calibration_matches_U_transpose_max_abs_error"]) <= 1e-10,
            "Gram extension does not reproduce calibration projection")
    return u, eig, coords, checks


def fit_alpha(measured, cheap_measured):
    numerator = float(np.sum(measured * cheap_measured))
    denominator = float(np.square(cheap_measured).sum())
    require(denominator > 0 and np.isfinite(numerator), "invalid direction fit")
    alpha = numerator / denominator
    error = measured - alpha * cheap_measured
    first_order = abs(float(np.sum(error * cheap_measured))) / (np.linalg.norm(measured) * np.linalg.norm(cheap_measured))
    require(np.isfinite(first_order) and first_order < 1e-12, "alpha fit violates least squares condition")
    return alpha, {"alpha_denominator": denominator, "alpha_normalized_normal_equation_error": float(first_order)}


def predictions_on_masks(removed, measured, cheap, cheap_measured, coords, alpha):
    spell = (removed @ measured) @ coords
    residual = alpha * (cheap - (removed @ cheap_measured) @ coords)
    return spell, residual, spell + residual


def bootstrap(query_score, cheap_names):
    rng = np.random.default_rng(20261115)
    q = len(query_score["hybrid"])
    index = rng.integers(0, q, size=(10_000, q))
    resampled = {key: values[index].mean(axis=1) for key, values in query_score.items()}
    mean = {key: float(values.mean()) for key, values in query_score.items()}
    candidate = resampled["hybrid"]
    competitors = ("spell", *cheap_names)
    best = max(competitors, key=mean.__getitem__)
    best_cheap = max(cheap_names, key=mean.__getitem__)

    def difference(reference, value):
        return {"point": mean["hybrid"] - reference, "ci95": np.percentile(candidate - value, (2.5, 97.5)).tolist()}

    return {
        "lds": mean,
        "best_baseline": best,
        "best_cheap": best_cheap,
        "gain_vs_best": difference(mean[best], np.maximum.reduce([resampled[key] for key in competitors])),
        "gain_vs_best_cheap": difference(mean[best_cheap], np.maximum.reduce([resampled[key] for key in cheap_names])),
        "gain_vs_spell": difference(mean["spell"], resampled["spell"]),
        "gain_vs_selected_cheap": difference(mean["selected_cheap"], resampled["selected_cheap"]),
    }


def metric_slice(pred, losses, query, masks, bootstrap_ci=False):
    observed = losses[masks, query]
    valid = np.ptp(observed, axis=0) > 0
    require(np.any(valid), "all target queries have constant retrain loss")
    excluded = (query.start + np.flatnonzero(~valid)).tolist()
    score = {key: lds(value[masks, query][:, valid], observed[:, valid]) for key, value in pred.items()}
    if bootstrap_ci:
        result = bootstrap(score, CHEAP_NAMES)
    else:
        means = {key: float(value.mean()) for key, value in score.items()}
        best = max(("spell", *CHEAP_NAMES), key=means.__getitem__)
        result = {"lds": means, "best_baseline": best, "gain_vs_best": means["hybrid"] - means[best]}
    return result | {"tested_query_count": int(valid.sum()), "constant_target_query_positions": excluded,
                     "strict_complete_query_status": "UNTESTABLE_CONSTANT_TARGET" if excluded else "SCORED"}


def scale(array):
    return {
        "rms": float(np.sqrt(np.square(array).mean())),
        "max_abs": float(np.max(np.abs(array))),
        "median_per_column_sd": float(np.median(array.std(axis=0, ddof=1))),
    }


def load(cfg, criteria, run):
    n, total, k, m = cfg["n_train"] or 50_000, cfg["k_total"], cfg["k_stage0"], cfg["M"]
    require((n, total, k, m, cfg["qchunk"]) == (50_000, 400, 50, 200, 10), "unexpected Stage 0 dimensions")
    require((cfg["query_seed"], cfg["mask_seed"], cfg["p_drop"], cfg["train_seed"], cfg["matmul_precision"]) ==
            (20261114, 1234, 0.01, 0, "default"), "unexpected frozen trajectory or RNG settings")
    require(tuple(criteria["cheap_variants"]) == CHEAP_NAMES, "Stage 0 method identities changed")
    paths = (
        [run / "retrain" / f"loss_{i}.npy" for i in range(m)]
        + [run / "retrain" / f"noise_{i}_task0.npy" for i in range(1, 5)]
        + [run / "exact" / f"{kind}_{i}_{i + 10}.npy" for kind in ("Y", "ql") for i in range(0, k, 10)]
        + [run / "cheap" / f"{name}.npy" for name in (*CHEAP_NAMES, "gram_query_grads")]
    )
    sizes = {path: (path.stat().st_size, path.stat().st_mtime_ns) for path in paths}
    observed = {p.name for p in (run / "exact").glob("Y_*.npy")}
    require(observed == {f"Y_{i}_{i + 10}.npy" for i in range(0, k, 10)}, "unexpected exact query chunks")
    losses = np.stack([read_array(run / "retrain" / f"loss_{i}.npy", (total + 1,)) for i in range(m)])
    results = EXPERIMENT / "results/stage0"
    snapshot = results / "retrain_query_losses_k50_M200.npy"
    require(np.array_equal(read_array(snapshot, (m, k + 1)), losses[:, [*range(k), -1]]), "frozen loss rows or query positions disagree")
    mask = np.stack([np.random.RandomState(cfg["mask_seed"] * 100003 + i).random_sample(n) < cfg["p_drop"] for i in range(m)])
    counts = mask.sum(axis=1)
    packed = np.packbits(mask, axis=1)
    require(np.unique(packed, axis=0).shape[0] == m and np.all(counts > 0), "degenerate removal masks")
    query_indices = np.random.RandomState(cfg["query_seed"]).permutation(10_000)[:total]
    model_path = run / "models/model_s0.npz"
    model_hash = sha256(model_path)
    with np.load(model_path) as model:
        model_loss = checked(model["query_loss"], (total,), "seed 0 checkpoint query loss")
        original_logits = checked(model["ref_logits"], (64, 10), "seed 0 checkpoint logits")
    original = np.concatenate([read_array(run / "exact" / f"ql_{i}_{i + 10}.npy", (10,)) for i in range(0, k, 10)])
    baselines = {"exact": original, "checkpoint": model_loss[CAL]}
    for i in range(1, 5):
        baselines[f"noise_{i}"] = read_array(run / "retrain" / f"noise_{i}_task0.npy", (total + 1,))[CAL]
    exact_rank = rankdata(original)
    require(np.unique(exact_rank).size == k, "ties in exact baseline identity check")
    require(all(np.array_equal(rankdata(values), exact_rank) for values in baselines.values()),
            "seed 0 exact, checkpoint and no-removal baseline loss rank/order differ")
    baseline_loss_diffs = {}
    for key, values in baselines.items():
        error = np.abs(values - model_loss[CAL])
        baseline_loss_diffs[key] = {
            "max_abs": float(error.max()), "median_abs": float(np.median(error)),
            "rms": float(np.sqrt(np.mean(error**2))),
        }
    export_path = results / "export_fidelity.json"
    highest_path = EXPERIMENT / "logs/diag-fidelity-26325/jax_logits.npz"
    export = json.loads(export_path.read_text())
    require(export["reference"]["checkpoint_sha256"] == model_hash, "cheap checkpoint hash differs")
    require(export["max_abs_logit_diff"] < criteria["gates"]["export_fidelity_max_abs_logit_diff"], "export gate failed")
    with np.load(highest_path) as reference:
        require(reference["checkpoint_sha256"].item() == model_hash, "precision reference checkpoint hash differs")
        require(np.array_equal(reference["query_indices"], query_indices[:64]), "precision reference query indices differ")
        require(np.array_equal(reference["saved"], original_logits), "precision reference saved logits differ")
    gram = read_array(run / "cheap/gram_query_grads.npy", (total, total)).astype(np.float64)
    gram_symmetry = float(np.max(np.abs(gram - gram.T)) / np.max(np.abs(gram)))
    require(gram_symmetry < 1e-6 and np.all(np.diag(gram) > 0), "invalid query-gradient Gram")
    audit = {
        "query_indices": query_indices.tolist(),
        "calibration_baseline_query_rank_agreement": True,
        "calibration_baseline_loss_differences_from_checkpoint": baseline_loss_diffs,
        "query_order_check": "seed-defined 400 positions; saved checkpoint fidelity manifest explicitly certifies first 64",
        "exact_and_retrain_query_order": "shared query_indices source, exact filename chunk positions, and calibration rank check; no per-array query ID manifest",
        "cheap_export_max_abs_logit_difference": export["max_abs_logit_diff"],
        "frozen_first50_retrain_snapshot_exact_match": True,
        "sha256_packed_removal_masks": hashlib.sha256(packed.tobytes()).hexdigest(),
        "distinct_removal_masks": m,
        "removal_count_min_mean_max": [int(counts.min()), float(counts.mean()), int(counts.max())],
        "gram_relative_max_asymmetry": gram_symmetry,
        "sha256_original_frozen_stage0_verdict": sha256(results / "verdict_stage0.json"),
        "additional_file_sha256": {str(p): sha256(p) for p in (snapshot, export_path, highest_path)},
        "checkpoint_sha256": model_hash,
    }
    return n, k, total, losses[:, :total], -mask.astype(np.float32), (gram + gram.T) / 2, audit, sizes


def evaluate(cfg_path, criteria_path):
    begin = now()
    plan = {
        "protocol_sha256": sha256(HERE / "PROTOCOL.md"),
        "script_sha256": sha256(Path(__file__)),
        "input_read_started_at": begin,
        "repository_head": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "config_sha256": sha256(cfg_path),
        "original_criteria_sha256": sha256(criteria_path),
        "numpy_version": np.__version__,
    }
    cfg, criteria = json.loads(cfg_path.read_text()), json.loads(criteria_path.read_text())
    run = Path(cfg["run_dir"])
    n, k, total, losses, removed, gram, quality, sizes = load(cfg, criteria, run)
    raw_predictions = {}
    for name in CHEAP_NAMES:
        cheap = read_array(run / "cheap" / f"{name}.npy", (n, total), mmap=True)
        raw_predictions[name] = (removed @ cheap).astype(np.float64)
    fit = {key: float(lds(values[FIT_MASK, CAL], losses[FIT_MASK, CAL]).mean()) for key, values in raw_predictions.items()}
    selected = max(CHEAP_NAMES, key=fit.__getitem__)
    u, eig, coords, invariants = ten_directions(gram, k, BUDGET)
    measured = np.zeros((n, BUDGET), dtype=np.float64)
    for i in range(0, k, cfg["qchunk"]):
        e = i + cfg["qchunk"]
        block = read_array(run / "exact" / f"Y_{i}_{e}.npy", (n, e - i), mmap=True)
        measured += block @ u[i:e]
    c_cal = read_array(run / "cheap" / f"{selected}.npy", (n, total), mmap=True)[:, CAL]
    cheap_measured = c_cal @ u
    alpha, fit_checks = fit_alpha(measured, cheap_measured)
    pred_c = raw_predictions[selected]
    spell, residual, hybrid = predictions_on_masks(removed, measured, pred_c, cheap_measured, coords, alpha)
    # Algebra on calibration scores, independent of the float32 mask aggregates.
    check_residual = c_cal.astype(np.float64) - cheap_measured @ coords[:, CAL]
    projection_error = float(np.linalg.norm(check_residual @ u) / np.linalg.norm(cheap_measured))
    require(projection_error < 1e-10, "cheap residual retains a measured calibration component")
    invariants.update(fit_checks, residual_calibration_projected_relative_norm=projection_error)
    quality["algebra"] = invariants
    capture = np.sum(eig[:, None] * coords**2, axis=0) / np.diag(gram)
    require(float(np.max(capture)) <= 1.00001, "gradient projection captures more than total energy")
    prediction = dict(raw_predictions, selected_cheap=pred_c, spell=spell, alpha_cheap=alpha * pred_c,
                      residual_only=residual, hybrid=hybrid)
    named_slices = {
        "primary_unseen_queries_unseen_masks": (TEST, TEST_MASK),
        "unseen_queries_selection_masks": (TEST, FIT_MASK),
        "calibration_queries_unseen_masks": (CAL, TEST_MASK),
        "calibration_queries_selection_masks": (CAL, FIT_MASK),
        "primary_queries_masks_100_149": (TEST, slice(100, 150)),
        "primary_queries_masks_150_199": (TEST, slice(150, 200)),
    }
    assessments = {
        name: metric_slice(prediction, losses, queries, masks, bootstrap_ci=name.startswith("primary_unseen"))
        for name, (queries, masks) in named_slices.items()
    }
    for row in assessments.values():
        row["constant_target_query_cifar_indices"] = [quality["query_indices"][q] for q in row["constant_target_query_positions"]]
    primary = assessments["primary_unseen_queries_unseen_masks"]
    gain, (lo, hi) = primary["gain_vs_best"]["point"], primary["gain_vs_best"]["ci95"]
    assessment = "inconclusive"
    if gain >= REFERENCE and lo > 0:
        assessment = "supports_practical_gain_in_this_setting"
    elif hi < REFERENCE:
        assessment = "below_practical_reference_in_this_setting"
    hashes = {str(p.relative_to(run)): sha256(p) for p in sizes}
    require(all((p.stat().st_size, p.stat().st_mtime_ns) == signature for p, signature in sizes.items()), "input file changed during evaluation")
    return {
        "method_and_timing": plan | {"completed_at": now()},
        "inputs": quality | {"producer_jobs": {"seed0_and_ensemble_checkpoints": 26324, "exact_chunks": 26327,
                                                "retrain": 26326, "cheap_and_gram": 26374, "precision_reference": 26353},
                              "file_sha256": hashes},
        "budget": BUDGET,
        "fit": {"query_positions_inclusive": [0, 49], "cheap_selection_mask_rows_inclusive": [0, 99],
                 "alpha_fit_removal_mask_rows": [], "alpha_fit_all_training_rows": True, "cheap_calibration_lds": fit,
                 "selected_cheap": selected, "alpha": alpha, "gram_eigenvalues": eig.tolist()},
        "test": {"query_positions_inclusive": [50, 399], "mask_rows_inclusive": [100, 199],
                 "primary_absolute_scales": {key: scale(values[TEST_MASK, TEST]) for key, values in prediction.items()},
                 "measured_A_scale": scale(measured),
                 "test_gradient_energy_capture_quantiles": np.percentile(capture[TEST], (0, 25, 50, 75, 100)).tolist(),
                 "loss_scale": scale(losses[TEST_MASK, TEST])},
        "assessments": assessments,
        "practical_reference": REFERENCE,
        "strict_primary_assessment": primary["strict_complete_query_status"],
        "primary_assessment_on_scorable_queries": assessment,
        "status": "exploratory_internal_holdout_same_training_trajectory",
    }


def self_test():
    rng = np.random.default_rng(11)
    gradients = rng.normal(size=(12, 5))
    gram = gradients @ gradients.T
    u, _, d, _ = ten_directions(gram, 7, 3)
    cheap = rng.normal(size=(35, 12))
    measured_c = cheap[:, :7] @ u
    a = 2.5 * measured_c
    alpha, _ = fit_alpha(a, measured_c)
    remove = -rng.binomial(1, 0.5, (20, 35))
    cheap_p = remove @ cheap
    s, r, hybrid = predictions_on_masks(remove, a, cheap_p, measured_c, d, alpha)
    np.testing.assert_allclose(s[:, :7], remove @ a @ u.T, atol=1e-12)
    np.testing.assert_allclose(s + r, hybrid, atol=1e-12)
    np.testing.assert_allclose(hybrid, 2.5 * cheap_p, atol=1e-12)
    np.testing.assert_allclose(lds(hybrid, cheap_p), 1, atol=1e-12)
    np.testing.assert_allclose(lds(-hybrid, cheap_p), -1, atol=1e-12)
    losses = cheap_p.copy()
    losses[:, 1] = 0.0
    controls = {name: cheap_p for name in CHEAP_NAMES}
    metric = metric_slice(dict(controls, spell=s, hybrid=hybrid), losses, slice(0, 4), slice(0, 20))
    assert metric["tested_query_count"] == 3 and metric["constant_target_query_positions"] == [1]
    assert metric["strict_complete_query_status"] == "UNTESTABLE_CONSTANT_TARGET" and metric["lds"]["hybrid"] > 0.999
    print("Synthetic algebra, scaling, LDS and constant-target checks pass")


def main():
    ap = argparse.ArgumentParser(description="Fixed-ten-direction internal CPU holdout")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--output", type=Path, default=HERE / "results.json")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    require(not args.output.exists(), f"use a new output path; existing result: {args.output}")
    result = evaluate(EXPERIMENT / "configs/stage0.json", EXPERIMENT / "configs/criteria.json")
    with args.output.open("x") as file:
        json.dump(result, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\n")
    print(result["strict_primary_assessment"], result["primary_assessment_on_scorable_queries"],
          result["assessments"]["primary_unseen_queries_unseen_masks"]["gain_vs_best"])


if __name__ == "__main__":
    main()
