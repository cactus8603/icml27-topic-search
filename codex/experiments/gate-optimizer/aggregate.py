import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np

from data import write_json

DETECTORS = ("write_q95", "qk_max_last_query", "activation_max", "grad_norm", "out_update_rms")
STRONG = ("stableadam", "muon", "sandwich")


def roc_scores(scores, targets):
    scores = np.asarray(scores)
    targets = np.asarray(targets, dtype=bool)
    positive = scores[targets]
    negative = scores[~targets]
    if positive.size == 0 or negative.size == 0:
        return None
    difference = positive[:, None] - negative
    return float(((difference > 0) + 0.5 * (difference == 0)).mean())


def roc(rows, metric, threshold):
    return roc_scores([r[metric] for r in rows], [r["delta_nll"] >= threshold for r in rows])


def sigmoid(z):
    return np.exp(-np.logaddexp(0, -z))


def ridge_logistic(train, labels, test, penalty):
    # Standardize only on training seeds. Keep the intercept unpenalized.
    center, scale = train.mean(axis=0), train.std(axis=0)
    scale[scale < 1e-12] = 1.0
    train = np.column_stack((np.ones(len(train)), (train - center) / scale))
    test = np.column_stack((np.ones(len(test)), (test - center) / scale))
    ridge = np.full(train.shape[1], penalty)
    ridge[0] = 0
    beta = np.zeros(train.shape[1])

    def loss(beta_value):
        logits = train @ beta_value
        return (np.logaddexp(0, logits) - labels * logits).mean() + 0.5 * (ridge * beta_value**2).sum()

    for _ in range(80):
        p = sigmoid(train @ beta)
        g = train.T @ (p - labels) / len(train) + ridge * beta
        if np.max(np.abs(g)) < 1e-9:
            return test @ beta
        hessian = train.T @ (train * (p * (1 - p))[:, None]) / len(train) + np.diag(ridge)
        update = np.linalg.solve(hessian, g)
        current = loss(beta)
        rate = 1.0
        while rate > 1e-7 and loss(beta - rate * update) > current:
            rate *= 0.5
        beta -= rate * update
    raise RuntimeError("ridge logistic regression failed to converge")


def seed_diagnostic(groups, seeds, threshold, penalty):
    features, targets = {}, {}
    for seed in seeds:
        pairs = [(arm, row) for arm in STRONG for row in groups.get((arm, seed), [])]
        features[seed] = np.array([
            [*(row[k] for k in DETECTORS[1:]), row["step"], arm == "muon", arm == "sandwich",
             row["write_q95"]]
            for arm, row in pairs
        ], dtype=np.float64).reshape(-1, len(DETECTORS) + 3)
        targets[seed] = np.array([row["delta_nll"] >= threshold for _, row in pairs], dtype=bool)
    folds = []
    for seed in seeds:
        others = [other for other in seeds if other != seed]
        training_y = np.concatenate([targets[other] for other in others]) if others else np.array([], dtype=bool)
        test_y = targets[seed]
        fold = {"held_out_seed": seed, "training_seeds": others, "train_measurements": len(training_y),
                "train_damage_events": int(training_y.sum()), "test_measurements": len(test_y),
                "test_damage_events": int(test_y.sum()), "auroc_controls": None,
                "auroc_controls_and_write": None, "increment": None}
        if len(np.unique(training_y)) < 2:
            fold["status"] = "insufficient_training_classes"
        elif len(np.unique(test_y)) < 2:
            fold["status"] = "insufficient_held_out_classes"
        else:
            train = np.concatenate([features[other] for other in others])
            test = features[seed]
            control = roc_scores(ridge_logistic(train[:, :-1], training_y, test[:, :-1], penalty), test_y)
            full = roc_scores(ridge_logistic(train, training_y, test, penalty), test_y)
            fold.update(status="scored", auroc_controls=control, auroc_controls_and_write=full,
                        increment=full - control)
        folds.append(fold)
    increments = np.array([fold["increment"] for fold in folds if fold["status"] == "scored"])
    interval = None
    if increments.size >= 2:
        rng = np.random.default_rng(124185)
        samples = rng.choice(increments, size=(5000, increments.size), replace=True).mean(axis=1)
        interval = np.quantile(samples, [0.025, 0.975]).tolist()
    return {"held_out_folds": folds, "scored_seeds": int(increments.size),
            "mean_held_out_increment": float(increments.mean()) if increments.size else None,
            "seed_bootstrap_95_exploratory": interval,
            "uncertainty_scope": "Descriptive resampling of scored seeds, not event-level or a significance test."}


def endpoint(run):
    last = run["evaluation"][-1]
    return last["test_nll"], last["train_seconds"]


def finite(value):
    return value is not None and math.isfinite(value)


def valid_terminal(run, branch_steps):
    if len(run["trace"]) != branch_steps or not run["evaluation"]:
        return False
    step, evaluation = run["trace"][-1], run["evaluation"][-1]
    return (step["step"] == evaluation["step"] == branch_steps
            and all(finite(value) for value in (step["train_nll"], step["grad_norm"], evaluation["test_nll"])))


def analyze(config, results):
    training, criteria = config["training"], config["criteria"]
    seeds = training["test_seeds"]
    expected = {(a["name"], s, schedule) for a in training["arms"]
                for s in seeds for schedule in ("nominal", "stress")}
    runs = {}
    failures = []
    for key in sorted(expected):
        name, seed, schedule = key
        path = results / f"{name}-{seed}-{schedule}.json"
        if path.exists():
            run = json.loads(path.read_text())
            if (run["arm"], run["seed"], run["schedule"]) != key:
                raise ValueError(f"run identity mismatch for {path}")
            runs[key] = run
        elif (results / f"{name}-{seed}-prefix-diverged.json").exists():
            failures.append(key)
    omitted = expected - set(runs) - set(failures)
    threshold = criteria["damage_nll"]

    arm_summary = {}
    for name in sorted({key[0] for key in expected}):
        arm_summary[name] = []
        for seed in seeds:
            stress = runs.get((name, seed, "stress"))
            nominal = runs.get((name, seed, "nominal"))
            if stress is None or nominal is None:
                continue
            assert nominal["evaluation"][0] == stress["evaluation"][0], (name, seed)
            nll, seconds = endpoint(nominal)
            stress_nll, stress_seconds = endpoint(stress)
            diverged = any(not finite(stress["trace"][-1][key]) for key in ("train_nll", "grad_norm"))
            diverged = diverged or not finite(stress_nll)
            valid_nominal = valid_terminal(nominal, training["branch_steps"])
            valid_stress = valid_terminal(stress, training["branch_steps"])
            dangers = sum(row.get("delta_nll") is not None and row["delta_nll"] >= threshold for row in stress["probes"])
            arm_summary[name].append({"seed": seed, "nominal_test_nll": nll if valid_nominal else None,
                                      "nominal_train_seconds": seconds, "stress_test_nll": stress_nll if valid_stress else None,
                                      "stress_train_seconds": stress_seconds, "stress_diverged": diverged,
                                      "stress_damage_events": dangers})

    strong = set(STRONG)
    strong_rows = {}
    for (arm, seed, schedule), run in runs.items():
        if schedule != "stress" or arm not in strong:
            continue
        strong_rows[(arm, seed)] = [row for row in run["probes"]
                                    if all(finite(row.get(k)) for k in (*DETECTORS, "step", "delta_nll"))]
    rows = [row for group in strong_rows.values() for row in group]
    positive = sum(row["delta_nll"] >= threshold for row in rows)
    by_group = []
    for (arm, seed), group in sorted(strong_rows.items()):
        by_group.append({"arm": arm, "seed": seed, "measurements": len(group),
                         "damage_events": sum(row["delta_nll"] >= threshold for row in group),
                         "auroc": {metric: roc(group, metric, threshold) for metric in DETECTORS}})
    defined = [group["auroc"] for group in by_group if group["auroc"]["write_q95"] is not None]
    auc = {metric: statistics.mean(g[metric] for g in defined) if defined else None for metric in DETECTORS}
    signal_summary = seed_diagnostic(strong_rows, seeds, threshold, criteria["signal_ridge"])
    margin = signal_summary["mean_held_out_increment"]
    enough_seeds = signal_summary["scored_seeds"] >= criteria["min_scored_signal_seeds"]
    interventions = [row for row in rows if all(row.get(k) is not None for k in
                                               ("nll_direction", "nll_uniform", "nll_control"))]
    damaged_interventions = [r for r in interventions if r["delta_nll"] >= threshold]

    gaps = {}
    best_strong = {}
    for seed in seeds:
        vals = {name: a["nominal_test_nll"] for name, values in arm_summary.items()
                for a in values if a["seed"] == seed and a["nominal_test_nll"] is not None}
        if "gate" in vals and strong.issubset(vals):
            best_strong[seed] = min(strong, key=lambda name: (vals[name], name))
            gaps[seed] = vals[best_strong[seed]] - vals["gate"]

    matched_times = {}
    for seed, best_name in best_strong.items():
        pair = [runs[(name, seed, "nominal")] for name in ("gate", best_name)]
        target = max(endpoint(run)[0] for run in pair)
        times = {}
        for run in pair:
            for row, next_row in zip(run["evaluation"], run["evaluation"][1:]):
                if (row["test_nll"] is not None and next_row["test_nll"] is not None
                        and max(row["test_nll"], next_row["test_nll"]) <= target):
                    times[run["arm"]] = row["train_seconds"]
                    break
        matched_times[seed] = {"target_nll": target, "train_seconds": times}

    missing = len(omitted) > 0
    phenomenon = positive >= criteria["min_damage_events"] and len({g["seed"] for g in by_group if g["damage_events"]}) >= 2
    signal = margin >= criteria["auroc_margin"] if enough_seeds else None
    room = len(gaps) == len(seeds) and bool(gaps) and min(gaps.values()) >= criteria["min_nominal_gap_nll"]
    return {
        "arms": arm_summary,
        "omitted_arms": [list(key) for key in sorted(omitted)],
        "prefix_failures": [list(key) for key in failures],
        "phenomenon": {"qualifies": None if missing else phenomenon, "damage_events": positive, "measurements": len(rows)},
        "signal": {"qualifies": signal if phenomenon and not missing else None, **signal_summary,
                   "marginal_macro_auroc_descriptive": auc, "marginal_groups_descriptive": by_group},
        "headroom": {"qualifies": room if len(gaps) == len(seeds) and not missing else None,
                     "nominal_nll_gate_advantage": gaps, "matched_target_seconds": matched_times},
        "first_version": {
            "matched_one_step_rows": len(interventions),
            "damaged_one_step_rows": len(damaged_interventions),
            "mean_direction_minus_uniform_nll_on_damaged": (
                statistics.mean(r["nll_direction"] - r["nll_uniform"] for r in damaged_interventions)
                if damaged_interventions else None),
            "mean_direction_minus_control_nll_on_damaged": (
                statistics.mean(r["nll_direction"] - r["nll_control"] for r in damaged_interventions)
                if damaged_interventions else None),
            "caveat": "one-step counterfactual only, excluded from advancement criteria",
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    metadata = json.loads((args.results / "run.json").read_text())
    report = analyze(metadata["config"], args.results)
    report["smoke"] = metadata["smoke"]
    for key in ("frozen_commit", "frozen_tree", "prereg_sha256", "config_sha256"):
        report[key] = metadata.get(key)
    write_json(args.results / "aggregate.json", report)
    print("[codex] mechanical gate-optimizer aggregate saved", flush=True)


if __name__ == "__main__":
    main()
