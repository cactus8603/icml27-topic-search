"""Mechanically report phenomenon, incremental signal, and headroom."""

import argparse
import hashlib
import json
import random
import statistics
from itertools import combinations
from pathlib import Path
from statistics import NormalDist

from pilot import CONFIG_PATH, HERE


def wilson(events, count, confidence):
    if not count:
        return None
    p = events / count
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    z_squared = z * z
    correction = 1 + z_squared / count
    center = (p + z_squared / (2 * count)) / correction
    half_width = z * (p * (1 - p) / count + z_squared / (4 * count**2)) ** 0.5 / correction
    return [max(0.0, center - half_width), min(1.0, center + half_width)]


def summarize_rate(events, count, confidence):
    return {
        "events": events,
        "prompts": count,
        "estimate": (events / count) if count else None,
        "interval": wilson(events, count, confidence),
    }


def bootstrap_mean(values, settings):
    if not values:
        return {"prompts": 0, "estimate": None, "interval": None}
    rng = random.Random(settings["bootstrap_seed"])
    repeats = settings["bootstrap_resamples"]
    iterations = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(repeats))
    tail = (1 - settings["confidence_level"]) / 2
    return {
        "prompts": len(values),
        "estimate": statistics.mean(values),
        "interval": [iterations[int(tail * repeats)], iterations[int((1 - tail) * repeats)]],
    }


def gate(metric, floor, enough=True):
    interval = metric["interval"]
    status = "inconclusive"
    if enough and interval is not None:
        if interval[0] >= floor:
            status = "supported"
        elif interval[1] < floor:
            status = "below_floor"
    return {**metric, "floor": floor, "status": status, "minimum_coverage_met": enough}


def score_comparison(winner, loser, key):
    if winner[key] == loser[key]:
        return 0.5
    return float(winner[key] > loser[key])


def candidate_features(candidate, baseline_positions):
    positions = candidate["positions"]
    swap_distance = 0
    if set(positions) != baseline_positions:
        (removed,) = baseline_positions.difference(positions)
        (added,) = set(positions).difference(baseline_positions)
        swap_distance = abs(added - removed)
    return (
        statistics.mean(candidate["confidence"]),
        min(candidate["confidence"]),
        positions[0],
        statistics.mean(positions),
        positions[-1],
        swap_distance,
        int(swap_distance > 0),
        candidate["additive_score"],
        candidate["joint_score"],
    )


def discordant_pairs(candidates):
    original_positions = set(candidates[0]["positions"])
    features = [candidate_features(candidate, original_positions) for candidate in candidates]
    pairs = []
    for i, j in combinations(range(len(candidates)), 2):
        if candidates[i]["correct"] != candidates[j]["correct"]:
            # Candidate order is fixed before scoring answers.
            difference = tuple(a - b for a, b in zip(features[i], features[j], strict=True))
            pairs.append((difference, int(candidates[i]["correct"])))
    return pairs


def prompt_fold(doc_id, settings):
    key = f"{settings['signal_fold_seed']}:{doc_id}".encode()
    return int.from_bytes(hashlib.sha256(key).digest(), "big") % settings["signal_folds"]


def fit_pairwise_logistic(train_x, train_y, weights, test_x, ridge):
    import numpy as np
    from scipy.optimize import minimize
    from scipy.special import expit

    # No intercept or centering. Reversing a pair must reverse its log odds.
    weights = weights / weights.sum()
    scale = np.sqrt(weights @ (train_x**2))
    scale[scale < 1e-8] = 1.0
    train_x = train_x / scale

    def loss(beta):
        logits = train_x @ beta
        logistic_loss = weights @ (np.logaddexp(0, logits) - train_y * logits)
        penalty = (ridge / 2) * (beta @ beta)
        gradient = train_x.T @ (weights * (expit(logits) - train_y)) + ridge * beta
        return logistic_loss + penalty, gradient

    fit = minimize(loss, np.zeros(train_x.shape[1]), jac=True, method="L-BFGS-B")
    if not fit.success:
        raise RuntimeError(f"Signal predictor fit failed: {fit.message}")
    return (test_x / scale) @ fit.x


def held_out_concordance(pairs_by_prompt, settings):
    import numpy as np

    by_fold = [[] for _ in range(settings["signal_folds"])]
    for doc_id, pairs in pairs_by_prompt.items():
        by_fold[prompt_fold(doc_id, settings)].append(pairs)
    fold_counts = [len(prompts) for prompts in by_fold]
    if sum(count > 0 for count in fold_counts) < 2:
        return [], fold_counts

    concordance = []
    for fold, held_out in enumerate(by_fold):
        if not held_out:
            continue
        train = [pairs for i, prompts in enumerate(by_fold) if i != fold for pairs in prompts]
        x, y = zip(*(pair for pairs in train for pair in pairs), strict=True)
        weights = [1 / len(pairs) for pairs in train for _ in pairs]
        train_x, train_y, weights = np.asarray(x), np.asarray(y), np.asarray(weights)
        test_x, test_y = zip(*(pair for pairs in held_out for pair in pairs), strict=True)
        test_x, test_y = np.asarray(test_x), np.asarray(test_y)
        signed_truth = 2 * test_y - 1
        correct = []
        for count in (train_x.shape[1] - 1, train_x.shape[1]):
            predictions = fit_pairwise_logistic(
                train_x[:, :count], train_y, weights, test_x[:, :count],
                settings["signal_ridge"],
            )
            margins = signed_truth * predictions
            correct.append(np.where(np.abs(margins) <= 1e-12, 0.5, margins > 0))
        first = 0
        for pairs in held_out:
            last = first + len(pairs)
            concordance.append(tuple(float(scores[first:last].mean()) for scores in correct))
            first = last
    return concordance, fold_counts


def select(candidates, key):
    return max(range(len(candidates)), key=lambda index: candidates[index][key])


def summarize(rows, config):
    settings = config["judgment"]
    threshold = settings["threshold_margin"]
    confidence = settings["confidence_level"]
    total = len(rows)
    eligible_prompts = 0
    baselines = 0
    prompt_flips = 0
    baseline_position_flips = 0
    baseline_position_count = 0
    baseline_argmax_overturns = 0
    residuals = []
    raw_concordance = []
    pairs_by_prompt = {}
    rescues = 0
    correct_by_arm = dict(a_confidence=0, b_matched_swap=0, c_exact_singletons=0, d_exact_joint=0)
    method_difference = []
    rerank_disagreements = 0
    first_anchor_counts = {str(step): 0 for step in config["decoder"]["anchors"]}
    eligibility_by_anchor = {str(step): 0 for step in config["decoder"]["anchors"]}
    for row in rows:
        eligible = [a for a in row["anchors"] if a["ineligible_reason"] is None]
        baseline_correct = row["baseline_correct"]
        baselines += baseline_correct
        rescues += (not baseline_correct) and any(
            c["correct"] for a in eligible for c in a["candidates"]
        )
        if not eligible:
            for arm in correct_by_arm:
                correct_by_arm[arm] += baseline_correct
            method_difference.append(0)
            continue

        eligible_prompts += 1
        has_flip = False
        signal_pairs = []
        raw_pairs = []
        for anchor in eligible:
            eligibility_by_anchor[str(anchor["step"])] += 1
            anchor_candidates = anchor["candidates"]
            baseline_candidate = anchor_candidates[0]
            for additive, direct in zip(
                baseline_candidate["additive_margins"], baseline_candidate["joint_margins"],
                strict=True,
            ):
                changed = (additive >= threshold) != (direct >= threshold)
                has_flip |= changed
                baseline_position_flips += changed
                baseline_position_count += 1
                baseline_argmax_overturns += direct < 0
                residuals.append(abs(direct - additive))
            # Labels adjudicate scored pairs after all candidates have shared the same continuation.
            signal_pairs.extend(discordant_pairs(anchor_candidates))
            for one, other in combinations(anchor_candidates, 2):
                if one["correct"] == other["correct"]:
                    continue
                winner, loser = (one, other) if one["correct"] else (other, one)
                direct = score_comparison(winner, loser, "joint_score")
                additive = score_comparison(winner, loser, "additive_score")
                raw_pairs.append((direct, additive))
        prompt_flips += has_flip
        if signal_pairs:
            pairs_by_prompt[row["doc_id"]] = signal_pairs
            raw_concordance.append(
                tuple(statistics.mean(col) for col in zip(*raw_pairs, strict=True))
            )

        first_anchor = eligible[0]
        first_anchor_counts[str(first_anchor["step"])] += 1
        candidates = first_anchor["candidates"]
        c = select(candidates, "additive_score")
        d = select(candidates, "joint_score")
        rerank_disagreements += c != d
        for arm, index in zip(correct_by_arm, (0, 1, c, d), strict=True):
            correct_by_arm[arm] += candidates[index]["correct"]
        method_difference.append(int(candidates[d]["correct"]) - int(candidates[c]["correct"]))

    eligible_enough = eligible_prompts >= settings["min_eligible_prompts"]
    concordance, fold_counts = held_out_concordance(pairs_by_prompt, settings)
    paired_signal = [full - control for control, full in concordance]
    return {
        "prompt_count": total,
        "eligible_prompt_count": eligible_prompts,
        "eligible_anchors_by_step": eligibility_by_anchor,
        "first_eligible_anchors_by_step": first_anchor_counts,
        "phenomenon": {
            **gate(
                summarize_rate(prompt_flips, eligible_prompts, confidence),
                settings["phenomenon_min_prompt_flip_rate"], eligible_enough,
            ),
            "baseline_position_flip_rate": (
                baseline_position_flips / baseline_position_count if baseline_position_count else None
            ),
            "baseline_position_argmax_overturn_rate": (
                baseline_argmax_overturns / baseline_position_count if baseline_position_count else None
            ),
            "median_baseline_position_absolute_residual": (
                statistics.median(residuals) if residuals else None
            ),
        },
        "signal": {
            **gate(
                bootstrap_mean(paired_signal, settings),
                settings["signal_min_concordance_improvement"],
                eligible_enough and len(paired_signal) >= settings["signal_min_discordant_prompts"],
            ),
            "discordant_candidate_pairs": sum(map(len, pairs_by_prompt.values())),
            "discordant_prompts": len(pairs_by_prompt),
            "discordant_prompts_by_fold": fold_counts,
            "controls_concordance": (
                statistics.mean(control for control, _ in concordance) if concordance else None
            ),
            "controls_plus_joint_concordance": (
                statistics.mean(full for _, full in concordance) if concordance else None
            ),
        },
        "headroom": {
            **gate(
                summarize_rate(rescues, total, confidence),
                settings["headroom_min_rescued_rate"], eligible_enough,
            ),
            "baseline_accuracy": baselines / total,
            "observed_pool_oracle_accuracy": (baselines + rescues) / total,
        },
        "descriptive_first_eligible_anchor": {
            "correct_by_arm": correct_by_arm,
            "accuracy_by_arm": {arm: n / total for arm, n in correct_by_arm.items()},
            "d_vs_c_paired_accuracy": bootstrap_mean(method_difference, settings),
            "c_d_different_selections": rerank_disagreements,
        },
        "descriptive_unadjusted_concordance": {
            "joint": statistics.mean(d for d, _ in raw_concordance) if raw_concordance else None,
            "additive": statistics.mean(a for _, a in raw_concordance) if raw_concordance else None,
        },
    }


def load_complete(input_path, config, config_digest):
    with input_path.open() as handle:
        metadata = json.loads(handle.readline())
        rows = [json.loads(line) for line in handle]
    if (
        metadata.get("type") != "metadata"
        or metadata.get("mode") != "main"
        or metadata.get("config_sha256") != config_digest
        or metadata.get("data_sha256") != config["main"]["sha256"]
        or metadata.get("revision") != config["checkpoint"]["revision"]
    ):
        raise ValueError("Run metadata does not match this frozen main configuration.")

    policy = config["decoder"]
    indices = metadata["doc_ids"]
    if (
        len(rows) != config["main"]["sample_size"]
        or len(set(indices)) != len(rows)
        or [row["doc_id"] for row in rows] != indices
    ):
        raise ValueError("Run is incomplete or its prompt IDs differ from metadata.")
    step_count = policy["generation_length"] // policy["quota"]
    for row in rows:
        if row.get("type") != "prompt":
            raise ValueError("Unexpected main artifact row.")
        if [a["step"] for a in row["anchors"]] != policy["anchors"]:
            raise ValueError("Recorded anchors differ from the frozen configuration.")
        for anchor in row["anchors"]:
            if anchor["ineligible_reason"] is not None:
                continue
            candidates = anchor["candidates"]
            if len(candidates) != policy["candidate_count"]:
                raise ValueError("Incomplete candidate pool.")
            if len({tuple(c["positions"]) for c in candidates}) != len(candidates):
                raise ValueError("Candidate commitments must be unique.")
            for candidate in candidates:
                if len(candidate["positions"]) != policy["quota"]:
                    raise ValueError("Candidate quotas differ.")
                if candidate["continuation_forwards"] != step_count - anchor["step"] - 1:
                    raise ValueError("Continuation budget differs.")
            if candidates[0]["correct"] != row["baseline_correct"]:
                raise ValueError("Baseline did not retain its original continuation.")
    return metadata, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result_dir = (HERE / "results").resolve()
    for path in (args.input, args.output):
        if path.resolve().parent != result_dir or path.suffix not in {".jsonl", ".json"}:
            raise ValueError("Use complete JSONL input and JSON output directly under results/.")
    config_bytes = CONFIG_PATH.read_bytes()
    config = json.loads(config_bytes)
    config_digest = hashlib.sha256(config_bytes).hexdigest()
    metadata, rows = load_complete(args.input, config, config_digest)
    verdict = summarize(rows, config)
    with args.output.open("x") as handle:
        json.dump({"metadata": metadata, **verdict}, handle, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
