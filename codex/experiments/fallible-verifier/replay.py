"""Produce descriptive CPU replay results from a frozen configuration."""

import argparse
from collections import Counter, defaultdict
import hashlib
from itertools import product
import json
from pathlib import Path
import random
import subprocess
import sys

import numpy as np

from data import fetch, load_pairs, synthetic
from policy import (
    deploy, fit_policy, full_distribution, probe_indices, sparse_distribution,
    split_by_question, stratified_audits,
)


def add_groups(destination, statistics, scale=1.0):
    for group, values in statistics.items():
        if group not in destination:
            destination[group] = np.zeros(len(values))
        destination[group] += scale * np.asarray(values)


def metrics(values):
    cost, fa, fr, strong, gold, negatives, positives = np.moveaxis(np.asarray(values), -1, 0)
    count = negatives + positives
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.stack([cost / count, fa / negatives, fr / positives,
                         strong / count, gold / count], axis=-1)


def summarize(values, boot, warm_cost):
    point = metrics(values.sum(axis=0))
    sampled = metrics(boot @ values)
    names = ("deployment_cost", "false_accept", "false_reject", "strong_rate", "gold_rate")
    result = {}
    for index, name in enumerate(names):
        result[name] = {
            "estimate": float(point[index]),
            "question_bootstrap_95": np.nanpercentile(sampled[:, index], [2.5, 97.5]).tolist(),
        }
    result["warm_cost_per_evaluation_candidate"] = warm_cost
    cost = result["deployment_cost"]
    result["replay_cost_with_warmup"] = {
        "estimate": cost["estimate"] + warm_cost,
        "question_bootstrap_95": [bound + warm_cost for bound in cost["question_bootstrap_95"]],
    }
    return result


def bootstrap(groups, draws, seed):
    rng = np.random.default_rng(seed)
    blocks = defaultdict(list)
    for i, (_, dataset) in enumerate(groups):
        blocks[dataset].append(i)
    counts = np.zeros((draws, len(groups)), dtype=int)
    for indices in blocks.values():
        weights = np.ones(len(indices)) / len(indices)
        counts[:, indices] = rng.multinomial(len(indices), weights, size=draws)
    return counts


def fit_signal_predictors(train, strength):
    """Share the cheap prior. Estimate extra signal only within comparable strata."""
    counts = Counter()
    positives = Counter()
    for row in train:
        keys = ((row.weak,), (row.weak, row.dataset),
                (row.weak, row.dataset, row.solver_model),
                (row.weak, row.dataset, row.solver_model, row.strong))
        counts.update(keys)
        for key in keys:
            positives[key] += row.gold

    def predict(row):
        weak = (row.weak,)
        source = (*weak, row.dataset)
        stratum = (*source, row.solver_model)
        p_weak = (positives[weak] + 0.5) / (counts[weak] + 1)
        p_source = (positives[source] + strength * p_weak) / (counts[source] + strength)
        n = counts[stratum]
        p_cheap = (positives[stratum] + strength * p_source) / (n + strength)

        # Without both strong outcomes in this same cheap stratum, the training
        # data does not separate the strong verdict from the cheap features.
        overlap = min(counts[(*stratum, 0)], counts[(*stratum, 1)])
        if not overlap:
            return p_cheap, p_cheap
        conditional = (*stratum, row.strong)
        cell_rate = positives[conditional] / counts[conditional]
        p_cell = (n * cell_rate + strength * p_source) / (n + strength)
        weight = overlap / (overlap + strength)
        return p_cheap, p_cheap + weight * (p_cell - p_cheap)

    return predict


def diagnostics(train, test, output, strength):
    joint = full_distribution(train)
    n_train = len(train)
    predict = fit_signal_predictors(train, strength)
    for row in test:
        weak_mass = joint[row.weak].sum(axis=0)
        strong_mass = joint[row.weak, row.strong]
        p_weak = (weak_mass[1] * n_train + 0.5) / (weak_mass.sum() * n_train + 1)
        p_joint = (strong_mass[1] * n_train + 0.5) / (strong_mass.sum() * n_train + 1)
        p_cheap, p_augmented = predict(row)
        totals = output.setdefault(row.group, np.zeros(13))
        totals[:5] += [int(row.gold == 0),
                       int(row.gold == 0 and row.strong == 1),
                       int(row.gold == 0 and row.weak == row.strong == 1),
                       int(row.gold == 1), int(row.gold == 1 and row.strong == 0)]
        start = 5 + 4 * row.weak
        totals[start:start + 4] += [(row.gold - p_cheap) ** 2, (row.gold - p_augmented) ** 2,
                                   (row.gold - p_weak) ** 2, (row.gold - p_joint) ** 2]


def summarize_diagnostics(points, boot):
    def scores(values):
        negative, false_accept, joint, positive, false_reject = values[..., :5].T
        base0, aug0, weak0, ws0, base1, aug1, weak1, ws1 = values[..., 5:].T
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.array([false_accept / negative, joint / false_accept,
                             false_reject / positive,
                             1 - (aug0 + aug1) / (base0 + base1), 1 - aug0 / base0, 1 - aug1 / base1,
                             1 - (ws0 + ws1) / (weak0 + weak1), 1 - ws0 / weak0, 1 - ws1 / weak1]).T

    names = ("strong_false_accept", "share_of_strong_false_accepts_also_weak_accepts",
             "strong_false_reject", "matched_relative_brier_gain",
             "matched_relative_brier_gain_when_weak_rejects",
             "matched_relative_brier_gain_when_weak_accepts", "weak_only_relative_brier_gain",
             "weak_only_relative_brier_gain_when_weak_rejects",
             "weak_only_relative_brier_gain_when_weak_accepts")
    point = scores(points.sum(axis=0))
    bounds = np.nanpercentile(scores(boot @ points), [2.5, 97.5], axis=0)
    return {name: {"estimate": float(point[i]), "question_bootstrap_95": bounds[:, i].tolist()}
            for i, name in enumerate(names)}


def audit_variants(train, config, fraction, seed, fold):
    weak = [row.weak for row in train]
    count = min(len(train), max(1, round(len(train) * fraction)))
    probe_ids = probe_indices(weak, config["strong_probe_fraction"], random.Random((seed << 12) + fold))
    observed_strong = {i: train[i].strong for i in probe_ids}
    uniform = sorted(random.Random((seed << 12) + fold + 100).sample(range(len(train)), count))
    stratified = stratified_audits(weak, observed_strong, count, config["stratified_uniform_fraction"],
                                   random.Random((seed << 12) + fold + 200))
    for label, probes, audit in (
        ("gold_only_uniform", {}, uniform),
        ("fixed_cascade_uniform", observed_strong, uniform),
        ("fixed_cascade_stratified", observed_strong, stratified),
        ("oracle_extra_gold_same_probes", observed_strong, range(len(train))),
    ):
        observed_gold = {i: train[i].gold for i in audit}
        yield (label, sparse_distribution(weak, probes, observed_gold, config["smoothing_strength"]),
               bool(probes), len(probes), len(observed_gold))


def replay(rows, config):
    assignment = split_by_question(rows, config["folds"], random.Random(config["split_seed"]))
    group_datasets = defaultdict(set)
    for row in rows:
        group_datasets[row.group].add(row.dataset)
    groups = [(group, min(datasets)) for group, datasets in sorted(group_datasets.items())]
    boot = bootstrap(groups, config["bootstrap_draws"], config["bootstrap_seed"])
    cells = list(product(config["audit_fractions"], config["strong_costs"], config["train_risk_targets"]))
    labels = ("gold_only_uniform", "fixed_cascade_uniform", "fixed_cascade_stratified",
              "oracle_extra_gold_same_probes", "oracle_all_gold_no_strong", "oracle_all_gold_optional_strong")
    outcomes = {(cell, label): {} for cell in cells for label in labels}
    warm = defaultdict(float)
    warm_counts = defaultdict(lambda: np.zeros(2))
    diag = {}
    gold_cost = config["gold_cost"]

    for fold in range(config["folds"]):
        train = [row for row in rows if assignment[row.group] != fold]
        test = [row for row in rows if assignment[row.group] == fold]
        diagnostics(train, test, diag, config["smoothing_strength"])
        full = full_distribution(train)
        subsets = {fraction: [tuple(audit_variants(train, config, fraction, seed, fold))
                              for seed in config["audit_seeds"]] for fraction in config["audit_fractions"]}
        for cell in cells:
            fraction, strong_cost, target = cell
            for label, use_strong in zip(labels[-2:], (False, True)):
                policy = fit_policy(full, use_strong, target, strong_cost, gold_cost)
                add_groups(outcomes[cell, label], deploy(policy, test, strong_cost, gold_cost))
                warm[cell, label] += len(train) * (gold_cost + use_strong * strong_cost)
                warm_counts[cell, label] += [use_strong * len(train), len(train)]
            for variants in subsets[fraction]:
                scale = 1 / len(subsets[fraction])
                for label, distribution, cascade, probes, gold_labels in variants:
                    policy = fit_policy(distribution, cascade, target, strong_cost, gold_cost)
                    values = deploy(policy, test, strong_cost, gold_cost)
                    add_groups(outcomes[cell, label], values, scale)
                    warm[cell, label] += scale * (gold_labels * gold_cost + probes * strong_cost)
                    warm_counts[cell, label] += scale * np.asarray([probes, gold_labels])

    diag_result = summarize_diagnostics(np.array([diag[group] for group, _ in groups]), boot)
    cells_result = []
    for cell in cells:
        sums = {label: np.array([outcomes[cell, label][group] for group, _ in groups]) for label in labels}
        summary = {label: summarize(values, boot, warm[cell, label] / len(rows)) for label, values in sums.items()}
        for label, result in summary.items():
            probes, audits = warm_counts[cell, label] / len(rows)
            result["warm_strong_probes_per_evaluation_candidate"] = float(probes)
            result["warm_gold_audits_per_evaluation_candidate"] = float(audits)
        comparisons = {}
        for reference, candidate in ((labels[0], labels[1]), (labels[1], labels[2]),
                                     (labels[2], labels[3]), (labels[4], labels[5])):
            candidate_boot = metrics(boot @ sums[candidate])
            reference_boot = metrics(boot @ sums[reference])
            candidate_point = metrics(sums[candidate].sum(axis=0))
            reference_point = metrics(sums[reference].sum(axis=0))
            difference = candidate_boot - reference_boot
            point = candidate_point - reference_point
            comparison = np.nanpercentile(difference, [2.5, 97.5], axis=0)
            comparison_result = {
                metric: {"estimate": point[i], "question_bootstrap_95": comparison[:, i].tolist()}
                for i, metric in enumerate(("deployment_cost", "false_accept", "false_reject"))
            }
            warm_delta = (warm[cell, candidate] - warm[cell, reference]) / len(rows)
            comparison_result["replay_cost_with_warmup"] = {
                "estimate": point[0] + warm_delta,
                "question_bootstrap_95": (comparison[:, 0] + warm_delta).tolist(),
            }
            if reference_point[0] > 0:
                with np.errstate(invalid="ignore", divide="ignore"):
                    ratios = np.divide(candidate_boot[:, 0], reference_boot[:, 0],
                                       out=np.full(len(boot), np.nan), where=reference_boot[:, 0] > 0)
                comparison_result["relative_deployment_cost_saving"] = {
                    "estimate": 1 - candidate_point[0] / reference_point[0],
                    "question_bootstrap_95": np.nanpercentile(1 - ratios, [2.5, 97.5]).tolist(),
                }
            comparisons[f"{candidate}_minus_{reference}"] = comparison_result
        cells_result.append(dict(zip(("audit_fraction", "strong_cost", "train_risk_target"), cell))
                            | {"policies": summary, "paired_differences": comparisons})
    return {"questions": len(groups), "candidates": len(rows), "diagnostics": diag_result,
            "cells": cells_result, "triage": triage(diag_result, cells_result, config)}


def triage(diagnostics_result, cells, config):
    limits = config["triage"]
    primary = next(cell for cell in cells if all(cell[key] == value for key, value in config["primary"].items()))
    policies = primary["policies"]
    base = policies["oracle_all_gold_no_strong"]
    optional = policies["oracle_all_gold_optional_strong"]
    risk = ("false_accept", "false_reject")
    base_cost = base["deployment_cost"]["estimate"]
    saving = 1 - optional["deployment_cost"]["estimate"] / base_cost if base_cost > 0 else None
    phenomenon = (
        diagnostics_result["strong_false_accept"]["question_bootstrap_95"][0] >= limits["min_strong_false_accept"]
        and diagnostics_result["share_of_strong_false_accepts_also_weak_accepts"]
        ["question_bootstrap_95"][0] >= limits["min_joint_fraction"]
    )
    signal = diagnostics_result["matched_relative_brier_gain"]["question_bootstrap_95"][0] >= limits[
        "min_brier_relative_gain"
    ]
    headroom = (
        saving is not None and saving >= limits["min_oracle_cost_saving"]
        and all(optional[key]["estimate"] <= limits["max_oracle_risk"] for key in risk)
        and all(optional[key]["estimate"] - base[key]["estimate"] <= limits["max_oracle_risk_deterioration"]
                for key in risk)
    )
    return {"phenomenon": bool(phenomenon), "accessible_signal": bool(signal),
            "full_training_gold_headroom": bool(headroom), "full_training_gold_relative_saving": saving,
            "note": "Descriptive criteria only. Sparse-policy comparisons do not decide project viability."}


def finite(value):
    if isinstance(value, dict):
        return {key: finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def checked_commit():
    def git(*args):
        return subprocess.check_output(["git", *args], text=True, cwd=Path(__file__).resolve().parent).strip()

    commit = git("rev-parse", "HEAD")
    if git("status", "--porcelain", "--", "."):
        raise RuntimeError("commit and freeze this experiment directory before a public-data replay")
    return commit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--synthetic", action="store_true")
    mode.add_argument("--fetch", action="store_true")
    mode.add_argument("--public", action="store_true")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config_path = Path(__file__).with_name("config.json")
    config = json.loads(config_path.read_text())
    if args.fetch or args.public:
        if args.data_dir is None:
            parser.error("--data-dir is required")
    if args.fetch:
        fetch(config, args.data_dir)
        print("[codex] Verified all four pinned dataset shards.")
        return
    if args.public:
        if args.output is None:
            parser.error("--public requires --output in the experiment results directory")
        result_root = Path(__file__).resolve().parent / "results"
        if not args.output.resolve().is_relative_to(result_root):
            parser.error("--output must be in this experiment's results/ directory")
        if args.output.exists():
            parser.error("refusing to overwrite an existing replay artifact")
        commit = checked_commit()
        pairs = load_pairs(config, args.data_dir)
    else:
        if args.output is not None:
            parser.error("--synthetic prints smoke status and does not write output")
        commit = None
        pairs = synthetic(config)

    result = {"commit": commit, "input": "synthetic" if args.synthetic else "public",
              "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
              "dataset_revision": config["dataset_revision"], "models": {}}
    for model, (rows, missing) in pairs.items():
        result["models"][model] = replay(rows, config) | {"incomplete_verdict_pairs": missing}
    if args.synthetic:
        print("[codex] Synthetic grouped replay succeeded.", file=sys.stderr)
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pending = args.output.with_name(f".{args.output.name}.pending")
    with pending.open("x") as output:
        try:
            json.dump(finite(result), output, indent=2, sort_keys=True, allow_nan=False)
            output.write("\n")
        except BaseException:
            pending.unlink(missing_ok=True)
            raise
    try:
        args.output.hardlink_to(pending)
    finally:
        pending.unlink()
    print(f"[codex] Wrote replay result to {args.output}.")


if __name__ == "__main__":
    main()
