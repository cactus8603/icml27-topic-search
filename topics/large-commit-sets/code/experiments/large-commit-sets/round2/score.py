"""Train matched selectors on GSM8K train and score the locked test prompts."""

import argparse
import json
import math
import os
import statistics
import time
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from design import HERE, counts_and_path, doubleton_rows, joint_rows, load_design, results_directory
from design import pilot, sha
from judge import candidate_features, gate, summarize_rate


SELECTORS = ("confidence", "round1_joint", "shared", "doubleton", "joint", "pool_oracle")


def load_artifact(path, mode, provenance, indices, settings):
    counts_and_path(path, mode)
    with path.open() as file:
        metadata = json.loads(file.readline())
        rows = [json.loads(line) for line in file]
    expected = {"type": "metadata", "mode": mode, "doc_ids": indices, **provenance}
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Incompatible {mode} metadata: {path}")
    if [row["doc_id"] for row in rows] != indices:
        raise ValueError(f"Missing, reordered or repeated {mode} rows.")
    for row in rows:
        verify_row(row, settings)
    return metadata, rows


def verify_row(row, settings):
    steps = settings["generation_length"] // settings["quota"]
    if row.get("type") != "prompt" or type(row["baseline_correct"]) is not bool:
        raise ValueError("Invalid prompt record.")
    anchor = row["anchor"]
    if anchor is None:
        if list(map(int, row["eligibility"])) != settings["anchors"] or any(
            value is None for value in row["eligibility"].values()
        ) or row["forwards"] != steps:
            raise ValueError("Unexplained baseline fallback.")
        return
    step = anchor["step"]
    if step not in settings["anchors"]:
        raise ValueError("Unexpected anchor.")
    prefixes = settings["anchors"][: settings["anchors"].index(step) + 1]
    if list(map(int, row["eligibility"])) != prefixes or list(row["eligibility"].values()).count(None) != 1:
        raise ValueError("Not the first eligible anchor.")
    if row["eligibility"].get(str(step)) is not None:
        raise ValueError("Observed anchor was marked ineligible.")
    candidates = anchor["candidates"]
    if len(candidates) != settings["candidate_count"]:
        raise ValueError("Inconsistent candidate count.")
    pools = [tuple(candidate["positions"]) for candidate in candidates]
    quota = settings["quota"]
    if len(set(pools)) != len(pools) or any(tuple(sorted(set(pool))) != pool or len(pool) != quota for pool in pools):
        raise ValueError("Invalid candidate pools.")
    if any(len(set(pools[0]) - set(pool)) != 1 for pool in pools[1:]):
        raise ValueError("Alternative candidates must swap exactly one token.")
    joint_budget = len(joint_rows(pools))
    expected_doubletons = [list(pair) for pair in doubleton_rows(pools, joint_budget)]
    if (
        anchor["joint_forwards"] != joint_budget
        or anchor["doubleton_forwards"] != joint_budget
        or anchor["doubleton_reveals"] != expected_doubletons
        or anchor["singleton_forwards"] != len(set().union(*pools))
    ):
        raise ValueError("Inconsistent arm probe budgets.")
    if row["forwards"] != steps + anchor["singleton_forwards"] + joint_budget * 2 + (
        len(candidates) - 1
    ) * (steps - step - 1):
        raise ValueError("Recorded total forward count is inconsistent.")
    if candidates[0]["correct"] != row["baseline_correct"]:
        raise ValueError("Candidate zero must retain original confidence rollout.")
    for candidate in candidates:
        if candidate["continuation_forwards"] != steps - step - 1:
            raise ValueError("Unmatched candidate continuation.")
        if type(candidate["correct"]) is not bool:
            raise ValueError("Unscored candidate.")
        for key in (
            "confidence", "additive_margins", "doubleton_margins", "joint_margins",
            "fixed_alt_residual", "dynamic_alt_switched", "doubleton_coverage",
        ):
            if len(candidate[key]) != quota or not all(math.isfinite(x) for x in candidate[key]):
                raise ValueError(f"Incomplete or invalid candidate field: {key}")
        for prefix in ("additive", "doubleton", "joint"):
            if not math.isclose(candidate[f"{prefix}_score"], pilot.set_score(candidate[f"{prefix}_margins"])):
                raise ValueError("Recorded candidate score differs from its margins.")


def design_features(candidates, arm):
    baseline_positions = set(candidates[0]["positions"])
    base = [candidate_features(candidate, baseline_positions)[:-1] for candidate in candidates]
    if arm == "shared":
        return np.asarray(base)
    score_name = f"{arm}_score"
    return np.asarray([raw + (c[score_name],) for raw, c in zip(base, candidates, strict=True)])


def discordant(candidates):
    return [
        (i, j, int(candidates[i]["correct"]))
        for i, j in combinations(range(len(candidates)), 2)
        if candidates[i]["correct"] != candidates[j]["correct"]
    ]


def fit(train_rows, arm, ridge):
    train_x, labels, weights = [], [], []
    for row in train_rows:
        if not row["anchor"]:
            continue
        candidates = row["anchor"]["candidates"]
        design = design_features(candidates, arm)
        pairs = discordant(candidates)
        for i, j, label in pairs:
            train_x.append(design[i] - design[j])
            labels.append(label)
            weights.append(1 / len(pairs))
    x, y, w = np.asarray(train_x), np.asarray(labels), np.asarray(weights)
    w /= w.sum()
    scale = np.sqrt(w @ (x**2))
    scale[scale < 1e-8] = 1
    x /= scale

    def objective(beta):
        logits = x @ beta
        loss = w @ (np.logaddexp(0, logits) - y * logits) + (ridge / 2) * (beta @ beta)
        gradient = x.T @ (w * (expit(logits) - y)) + ridge * beta
        return loss, gradient

    learned = minimize(objective, np.zeros(x.shape[1]), jac=True, method="L-BFGS-B")
    if not learned.success:
        raise RuntimeError(f"{arm} fit did not converge: {learned.message}")
    return learned.x / scale


def paired_bootstrap(values, config):
    if not values:
        return {"prompts": 0, "estimate": None, "interval": None}
    array = np.asarray(values)
    rng = np.random.default_rng(config["bootstrap_seed"])
    repeats = config["bootstrap_resamples"]
    resampled = []
    for start in range(0, repeats, 250):
        count = min(250, repeats - start)
        indices = rng.integers(0, len(array), size=(count, len(array)))
        resampled.extend(array[indices].mean(axis=1))
    alpha = 1 - config["confidence_level"]
    return {
        "prompts": len(array),
        "estimate": float(array.mean()),
        "interval": np.quantile(resampled, [alpha / 2, 1 - alpha / 2]).tolist(),
    }


def evaluate(train_rows, test_rows, config):
    if not test_rows:
        raise ValueError("No locked test prompts to score.")
    train_discordant = sum(
        bool(row["anchor"] and discordant(row["anchor"]["candidates"])) for row in train_rows
    )
    fitting_enabled = train_discordant >= config["minimum_train_discordant"]
    arms = ("shared", "doubleton", "joint")
    betas = {arm: fit(train_rows, arm, config["ridge"]) for arm in arms} if fitting_enabled else {}
    eligible = 0
    discordant_test_prompts = 0
    switched = 0
    flipped_positions = 0
    fixed_residual_on_flip = 0
    total_positions = 0
    number_flips = 0
    fixed_mechanism_prompts = 0
    outcomes, ranking = [], {arm: [] for arm in arms}
    timings = {arm: [] for arm in SELECTORS if arm != "pool_oracle"}
    nfes = {arm: [] for arm in timings}
    anchor_counts = Counter()
    for row in test_rows:
        trace = row["anchor"]
        answer = int(row["baseline_correct"])
        scores = {arm: answer for arm in SELECTORS}
        if trace is None:
            for arm in timings:
                timings[arm].append(row["baseline_seconds"])
                nfes[arm].append(row["forwards"])
            outcomes.append(scores)
            continue
        candidates = trace["candidates"]
        eligible += 1
        anchor_counts[str(trace["step"])] += 1
        original = candidates[0]
        flags = [
            (a >= config["margin_threshold"]) != (j >= config["margin_threshold"])
            for a, j in zip(original["additive_margins"], original["joint_margins"], strict=True)
        ]
        number_flips += any(flags)
        fixed_mechanism_prompts += any(
            flag and abs(delta) >= config["fixed_margin_residual_abs_threshold"]
            for flag, delta in zip(flags, original["fixed_alt_residual"], strict=True)
        )
        total_positions += len(flags)
        flipped_positions += sum(flags)
        switched += sum(
            flag and change for flag, change in zip(flags, original["dynamic_alt_switched"], strict=True)
        )
        fixed_residual_on_flip += sum(
            flag and abs(delta) >= config["fixed_margin_residual_abs_threshold"]
            for flag, delta in zip(flags, original["fixed_alt_residual"], strict=True)
        )
        started = time.monotonic()
        indices = {"confidence": 0, "round1_joint": max(
            range(len(candidates)), key=lambda i: candidates[i]["joint_score"]
        )}
        ranking_seconds = {"confidence": 0.0, "round1_joint": time.monotonic() - started}
        pairs = discordant(candidates)
        discordant_test_prompts += bool(pairs)
        for arm, beta in betas.items():
            started = time.monotonic()
            logits = design_features(candidates, arm) @ beta
            indices[arm] = int(np.argmax(logits))
            ranking_seconds[arm] = time.monotonic() - started
            if pairs:
                pair_correct = []
                for i, j, label in pairs:
                    signed = (logits[i] - logits[j]) * (2 * label - 1)
                    pair_correct.append(0.5 if abs(signed) <= 1e-12 else int(signed > 0))
                ranking[arm].append(statistics.mean(pair_correct))
        for arm, index in indices.items():
            candidate = candidates[index]
            extra = None if arm == "shared" else ("joint" if arm == "round1_joint" else arm)
            probes = int(arm in arms)
            extra_time = trace[f"{extra}_seconds"] if extra in ("joint", "doubleton") else 0
            plan_time = trace[f"{extra}_plan_seconds"] if extra in ("joint", "doubleton") else 0
            extra_nfe = trace[f"{extra}_forwards"] if extra in ("joint", "doubleton") else 0
            timings[arm].append(
                trace["prefix_seconds"] + candidate["continuation_seconds"]
                + probes * trace["singleton_seconds"] + extra_time + plan_time + ranking_seconds[arm]
            )
            rollout_nfe = trace["step"] + 1 + candidate["continuation_forwards"]
            nfes[arm].append(rollout_nfe + probes * trace["singleton_forwards"] + extra_nfe)
            scores[arm] = int(candidate["correct"])
        scores["pool_oracle"] = int(any(c["correct"] for c in candidates))
        outcomes.append(scores)

    count = len(outcomes)
    confidence = config["confidence_level"]
    covered = eligible >= config["minimum_eligible"]
    if fitting_enabled and nfes["joint"] != nfes["doubleton"]:
        raise RuntimeError("Learned alternatives did not spend equal scored full forwards.")
    signal_delta = [j - d for j, d in zip(ranking["joint"], ranking["doubleton"], strict=True)]

    def gap(ref):
        rescues = sum(row["pool_oracle"] - row[ref] for row in outcomes)
        if rescues < 0 or any(row["pool_oracle"] < row[ref] for row in outcomes):
            raise ValueError("Observed pool oracle must dominate every candidate policy.")
        return summarize_rate(rescues, count, confidence)

    def paired(left, right):
        deltas = [row[left] - row[right] for row in outcomes]
        return {
            **paired_bootstrap(deltas, config),
            "left_correct_right_wrong": deltas.count(1),
            "left_wrong_right_correct": deltas.count(-1),
        }

    evaluated = SELECTORS if fitting_enabled else ("confidence", "round1_joint", "pool_oracle")
    means = {name: statistics.mean(row[name] for row in outcomes) for name in evaluated}
    added_signal = paired_bootstrap(signal_delta, config)
    return {
        "test_prompts": count,
        "eligible_prompts": eligible,
        "anchors": dict(anchor_counts),
        "phenomenon": {
            **gate(summarize_rate(number_flips, eligible, confidence), config["phenomenon_prompt_rate_floor"], covered),
            "prompt_with_flip_and_fixed_alt_residual": summarize_rate(fixed_mechanism_prompts, eligible, confidence),
            "baseline_positions": total_positions,
            "positions_flipped": flipped_positions,
            "flips_with_changed_competitor": switched,
            "flips_with_fixed_alt_residual": fixed_residual_on_flip,
        },
        "signal": {
            **gate(
                added_signal, config["signal_concordance_gain_floor"],
                covered and fitting_enabled and discordant_test_prompts >= config["minimum_test_discordant"],
            ),
            "train_discordant_prompts": train_discordant,
            "test_discordant_prompts": discordant_test_prompts,
            "held_out_concordance": {
                arm: statistics.mean(values) if values else None for arm, values in ranking.items()
            },
        },
        "headroom": {
            **gate(
                gap("doubleton") if fitting_enabled else summarize_rate(0, 0, confidence),
                config["headroom_floor"], covered and fitting_enabled,
            ),
            "pool_oracle_vs_confidence": gap("confidence"),
            "pool_oracle_vs_current_round1_joint": gap("round1_joint"),
            "pool_oracle_vs_learned_joint": gap("joint") if fitting_enabled else None,
        },
        "descriptive": {
            "accuracy_by_arm": means,
            "learned_joint_minus_doubleton": paired("joint", "doubleton") if fitting_enabled else None,
            "learned_joint_minus_confidence": paired("joint", "confidence") if fitting_enabled else None,
            "doubleton_minus_confidence": paired("doubleton", "confidence") if fitting_enabled else None,
            "round1_joint_minus_confidence": paired("round1_joint", "confidence"),
            "selector_accuracy_gain_target": config["descriptive_selector_accuracy_gain_target"],
            "forward_count_mean": {arm: statistics.mean(nfes[arm]) for arm in nfes if nfes[arm]},
            "reconstructed_wall_seconds_mean": {
                arm: statistics.mean(timings[arm]) for arm in timings if timings[arm]
            },
            "probe_seconds_joint_minus_doubleton": paired_bootstrap(
                [r["anchor"]["joint_seconds"] - r["anchor"]["doubleton_seconds"] for r in test_rows if r["anchor"]],
                config,
            ),
            "fit_coefficients": {arm: beta.tolist() for arm, beta in betas.items()},
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--test-data", type=Path, required=True)
    parser.add_argument("--parquet-python", type=Path, required=True)
    parser.add_argument("--train-artifact", type=Path, required=True)
    parser.add_argument("--test-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config, round1, _, splits, provenance = load_design(
        args.train_data, args.test_data, args.parquet_python
    )
    provenance["artifacts_sha256"] = sha(HERE / "artifacts.json")
    inputs = {
        "train": load_artifact(args.train_artifact, "train", provenance, splits["train"], round1["decoder"]),
        "test": load_artifact(args.test_artifact, "test", provenance, splits["test"], round1["decoder"]),
    }
    output = args.output
    if output.resolve().parent != results_directory() or output.suffix != ".json":
        raise ValueError("The summary needs a fresh JSON file directly under results/.")
    if output.exists() or output.with_suffix(".partial").exists():
        raise FileExistsError(output)
    result = evaluate(inputs["train"][1], inputs["test"][1], config)
    result["metadata"] = {
        **provenance,
        "train_job_id": inputs["train"][0]["job_id"],
        "test_job_id": inputs["test"][0]["job_id"],
        "train_doc_ids": len(splits["train"]),
        "test_doc_ids": len(splits["test"]),
    }
    partial = output.with_suffix(".partial")
    with partial.open("x") as file:
        json.dump(result, file, indent=2, allow_nan=False)
        file.write("\n")
    os.link(partial, output)
    partial.unlink()


if __name__ == "__main__":
    main()
