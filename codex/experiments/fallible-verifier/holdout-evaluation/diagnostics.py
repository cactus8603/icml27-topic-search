"""Round 2 phenomenon and conditional signal on sealed binary measurements."""

from collections import defaultdict
import math

import numpy as np

from evaluation import coverage, reference


def quality_stats(values):
    (negative, strong_fa, shared, unknown, unknown_accept, weak_na, strong_na, size, base, augmented,
     weak_fa, weak_negative_observed, strong_negative_observed, strong_fa_weak_observed,
     signal_rows, signal_strong_na) = values

    def ratio(num, den):
        return num / den if den else np.nan

    return {
        "weak_fa": ratio(weak_fa, negative),
        "strong_fa": ratio(strong_fa, negative),
        "shared_fa": ratio(shared, strong_fa),
        "strong_fa_unknown_lower": ratio(strong_fa, negative + unknown) if negative else np.nan,
        "shared_fa_unknown_lower": ratio(shared, strong_fa + unknown_accept) if strong_fa else np.nan,
        "weak_fa_among_observed": ratio(weak_fa, weak_negative_observed),
        "strong_fa_among_observed": ratio(strong_fa, strong_negative_observed),
        "shared_fa_when_weak_observed": ratio(shared, strong_fa_weak_observed),
        "weak_unavailable_rate": ratio(weak_na, size),
        "strong_unavailable_rate": ratio(strong_na, size),
        "gold_parse_coverage": 1 - ratio(unknown, size),
        "weak_solver_brier": ratio(base, signal_rows),
        "weak_solver_strong_brier": ratio(augmented, signal_rows),
        "matched_brier_gain": 1 - ratio(augmented, base),
        "signal_strong_unavailable_rate": ratio(signal_strong_na, signal_rows),
    }


def measure_binary_signal(train, test, *, smoothing, bootstrap):
    """Fit the paired Brier models on development data, then score one locked split."""
    if train.name != "train" or test.name != "test":
        raise ValueError("Signal requires separately bound development and locked test splits")
    if train.binding != test.binding or train.question_ids & test.question_ids:
        raise ValueError("Train and test measurement binding or question isolation differs")
    draws, seed, tail = (bootstrap[key] for key in ("draws", "seed", "tail"))
    if (not math.isfinite(smoothing) or smoothing <= 0 or type(draws) is not int or draws <= 0
            or type(seed) is not int or seed < 0 or not 0 < tail < 0.5):
        raise ValueError("Supply an explicit smoothing and paired bootstrap plan")

    scores = reference.strong_signal(train.rows, test.rows, smoothing)
    quality = reference.quality_metrics(test.rows, scores)
    observed = defaultdict(lambda: np.zeros(6))
    for row in test.rows:
        wrong = row["gold"] is False
        weak_observed = row["weak_status"] == "observed"
        strong_observed = row["strong_status"] == "observed"
        matched = row["id"] in scores
        observed[row["question_id"]] += [
            wrong and weak_observed and row["weak"] == reference.ACCEPT,
            wrong and weak_observed,
            wrong and strong_observed,
            wrong and weak_observed and row["strong"] == reference.ACCEPT,
            matched,
            matched and not strong_observed,
        ]

    groups, weights = reference.bootstrap(test.rows, draws, seed)
    matrix = np.asarray([np.concatenate((quality[q], observed[q])) for q in groups])
    totals = matrix.sum(axis=0)
    return {
        "development_coverage": coverage(train), "test_coverage": coverage(test),
        "bootstrap": {"draws": draws, "seed": seed, "tail": tail},
        "smoothing": smoothing,
        "denominators": {
            "known_wrong": int(totals[0]), "known_wrong_weak_observed": int(totals[11]),
            "known_wrong_strong_observed": int(totals[12]),
            "strong_false_accepts": int(totals[1]), "strong_false_accepts_weak_observed": int(totals[13]),
            "brier_candidates": int(totals[14]), "brier_strong_unavailable": int(totals[15]),
        },
        "diagnostics": reference.paired_intervals(quality_stats(totals),
                                                  [quality_stats(row) for row in weights @ matrix], tail),
    }


def assess_binary_signal(report, criteria):
    """Use independently declared floors and only the Round 2 primary definitions."""
    test, train = report["test_coverage"], report["development_coverage"]
    unknown = test["gold"].get("unknown", 0)
    limits = {key: criteria[key] for key in ("min_gold_parse_coverage", "min_fa", "min_shared_fa", "min_brier_gain")}
    min_class = criteria["min_test_each_truth"]
    if any(not 0 <= value <= 1 for value in limits.values()) or type(min_class) is not int or min_class <= 0:
        raise ValueError("All coverage and signal floors must be provided explicitly")
    ready = (
        min(1 - unknown / test["candidates"], 1 - train["gold"].get("unknown", 0) / train["candidates"])
        >= limits["min_gold_parse_coverage"]
        and min(test["gold"].get("correct", 0), test["gold"].get("wrong", 0)) >= min_class
    )
    values = report["diagnostics"]
    draws = report["bootstrap"]["draws"]
    return {
        "coverage_sufficient": ready,
        "strong_false_accept": reference.classify(values["strong_fa_unknown_lower"], limits["min_fa"],
                                                   True, ready, draws, allow_opposite=unknown == 0),
        "shared_false_accept": reference.classify(values["shared_fa_unknown_lower"], limits["min_shared_fa"],
                                                   True, ready, draws, allow_opposite=unknown == 0),
        "conditional_signal_on_parsed_gold": reference.classify(values["matched_brier_gain"],
                                                                  limits["min_brier_gain"], True, ready, draws),
    }
