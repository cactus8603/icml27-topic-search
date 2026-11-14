import argparse
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aggregate import DETECTORS, STRONG, seed_diagnostic
from data import write_json


def finite(value):
    return value is not None and math.isfinite(value)


def status(branch, training, multiplier):
    terminal = training["branch_steps"]
    if not branch["trace"] or not branch["evaluations"]:
        return "missing_measurements"
    trace = branch["trace"]
    evaluations = branch["evaluations"]
    if (any(not finite(row[k]) for row in trace for k in ("train_nll", "grad_norm"))
            or any(not finite(row["dev_nll"]) for row in evaluations)):
        return "nonfinite"
    if [row["step"] for row in trace] != list(range(1, terminal + 1)):
        return "incomplete"
    scheduled_evals = sorted({0, terminal, *range(training["eval_every"], terminal, training["eval_every"])})
    if [row["step"] for row in evaluations] != scheduled_evals:
        return "missing_measurements"
    if multiplier != 1:
        scheduled_probes = list(range(training["probe_every"], terminal + 1, training["probe_every"]))
        if [row["step"] for row in branch["probes"]] != scheduled_probes:
            return "missing_measurements"
        if any(not finite(row.get(k)) for row in branch["probes"]
               for k in (*DETECTORS, "delta_nll", "nll_before", "nll_after")):
            return "nonfinite_probe"
    return "completed"


def collapse(branch, nominal, criteria):
    reference = {row["step"]: row["dev_nll"] for row in nominal["evaluations"]}
    differences = [row["dev_nll"] - reference[row["step"]] for row in branch["evaluations"][1:]]
    streak, longest = 0, 0
    limit = criteria["collapse_gap_nll"]
    for value in differences:
        streak = streak + 1 if value > limit else 0
        longest = max(longest, streak)
    return {
        "terminal_nominal_gap": differences[-1],
        "longest_bad_eval_streak": longest,
        "eligible": differences[-1] <= limit and longest < criteria["collapse_consecutive_evals"],
    }


def read_branch(results, arm, seed, multiplier):
    path = results / f"{arm}-{seed}-{multiplier}x.json"
    if not path.exists():
        return None
    branch = json.loads(path.read_text())
    if (branch["arm"], branch["seed"], branch["multiplier"]) != (arm, seed, multiplier):
        raise ValueError(f"branch identity does not match path: {path}")
    return branch


def rescue_summary(damaged):
    fields = ("nll_direction", "nll_uniform", "nll_control", "nll_wo_rollback",
              "uniform_delta_norm_relative_error", "control_change_norm_relative_error")
    paired = [
        row for row in damaged if row.get("rescue", {}).get("status") == "measured"
        and all(finite(row["rescue"].get(key)) for key in fields)
        and max(row["rescue"]["uniform_delta_norm_relative_error"],
                row["rescue"]["control_change_norm_relative_error"]) <= 1e-4
    ]
    caps = [row["delta_nll"] for row in damaged]
    results = {
        "damaged_steps": len(damaged), "valid_norm_matched_steps": len(paired),
        "whole_step_rollback_to_before_cap_mean_nats": statistics.mean(caps) if caps else None,
        "whole_step_rollback_to_before_cap_total_nats": sum(caps) if caps else None,
        "variants": {},
        "scope": "Selected damaged probe steps only. The cap is the observed damage recoverable by returning to the "
                 "pre-step NLL. It is not a bound on achievable improvements or deployment headroom.",
    }
    for variant in ("direction", "uniform", "control", "wo_rollback"):
        gains = [row["nll_after"] - row["rescue"][f"nll_{variant}"] for row in paired]
        cap = sum(row["delta_nll"] for row in paired)
        share = sum(min(max(0.0, gain), row["delta_nll"]) for gain, row in zip(gains, paired, strict=True))
        results["variants"][variant] = {
            "mean_signed_rescue_nats": statistics.mean(gains) if gains else None,
            "fraction_of_paired_rollback_cap_recovered": share / cap if cap > 0 else None,
        }
    return results


def signal_summary(groups, seeds, criteria, eligible, damage_count):
    threshold = criteria["damage_nll"]
    measurements = {seed: [row for arm in STRONG for row in groups.get((arm, seed), ())]
                    for seed in seeds}
    counts = {seed: (len(rows), sum(row["delta_nll"] >= threshold for row in rows))
              for seed, rows in measurements.items()}
    coverage = []
    for seed in seeds:
        others = [other for other in seeds if other != seed]
        n_train, n_train_positive = (sum(counts[other][i] for other in others) for i in (0, 1))
        n_test, n_test_positive = counts[seed]
        if not 0 < n_train_positive < n_train:
            fold_status = "insufficient_training_classes"
        elif not 0 < n_test_positive < n_test:
            fold_status = "insufficient_held_out_classes"
        else:
            fold_status = "both_classes"
        coverage.append({"held_out_seed": seed, "training_seeds": others, "status": fold_status,
                         "train_measurements": n_train, "train_damage_events": n_train_positive,
                         "test_measurements": n_test, "test_damage_events": n_test_positive})
    covering_seeds = sum(row["status"] == "both_classes" for row in coverage)
    report = {"status": "ineligible_pressure" if not eligible else "insufficient_coverage",
              "seeds_with_class_coverage": covering_seeds, "fold_coverage": coverage}
    if (eligible and damage_count >= criteria["signal_min_damage"]
            and covering_seeds >= criteria["signal_min_scored_seeds"]):
        report.update(status="descriptive_only",
                      **seed_diagnostic(groups, seeds, threshold, criteria["signal_ridge"]))
    return report


def judge(metadata, results):
    config, training = metadata["config"], metadata["training"]
    criteria = config["criteria"]
    seeds = config["calibration_seeds"]
    grid = config["stress_multipliers"]
    nominal = {(a, seed): read_branch(results, a, seed, 1) for a in STRONG for seed in seeds}
    nominal_status = {key: status(run, training, 1) if run else "missing" for key, run in nominal.items()}
    by_multiplier = {}
    fully_observed = True
    for multiplier in grid:
        rows, healthy_rows, cells = {}, {}, []
        all_eligible = True
        for arm in STRONG:
            for seed in seeds:
                key = (arm, seed)
                branch = read_branch(results, arm, seed, multiplier)
                prefix_failure = (results / f"{arm}-{seed}-prefix_failure.json").is_file()
                if branch is None and not prefix_failure:
                    fully_observed = False
                state = status(branch, training, multiplier) if branch else ("prefix_failed" if prefix_failure else "missing")
                ref_state = nominal_status[key]
                if ref_state == "missing" and not prefix_failure:
                    fully_observed = False
                if state in {"incomplete", "missing_measurements"} or ref_state in {"incomplete", "missing_measurements"}:
                    fully_observed = False
                if branch and nominal[key] and branch["evaluations"][0] != nominal[key]["evaluations"][0]:
                    raise ValueError(f"continuations have inconsistent prefix metrics: {key}, {multiplier}x")
                guard = collapse(branch, nominal[key], criteria) if state == ref_state == "completed" else None
                eligible = guard is not None and guard["eligible"]
                all_eligible = all_eligible and eligible
                rows[key] = [
                    row for row in (branch["probes"] if branch else [])
                    if row["step"] > training["branch_warmup"]
                    and all(finite(row.get(k)) for k in (*DETECTORS, "delta_nll", "nll_before", "nll_after"))
                ]
                if eligible:
                    healthy_rows[key] = rows[key]
                cells.append({"arm": arm, "seed": seed, "stress_status": state, "nominal_status": ref_state,
                              "collapse_guard": guard, "post_ramp_probes": len(rows[key]),
                              "post_ramp_damage": sum(r["delta_nll"] >= criteria["damage_nll"] for r in rows[key])})
        def damaged_rows(groups):
            return [(a, seed, row) for (a, seed), group in groups.items() for row in group
                    if row["delta_nll"] >= criteria["damage_nll"]]

        damaged = damaged_rows(healthy_rows)
        damage_seeds = len({seed for _, seed, _ in damaged})
        damage_arms = len({arm for arm, _, _ in damaged})
        enough = (len(damaged) >= criteria["min_damage_events"] and damage_seeds >= criteria["min_damage_seeds"]
                  and damage_arms >= criteria["min_damage_arms"])
        by_multiplier[str(multiplier)] = {
            "all_cells_eligible": all_eligible, "phenomenon_qualifies": all_eligible and enough,
            "damaged_steps": len(damaged), "damage_seeds": damage_seeds, "damage_arms": damage_arms,
            "cells": cells, "signal": signal_summary(healthy_rows, seeds, criteria, all_eligible, len(damaged)),
            "local_rescue": rescue_summary([row for _, _, row in damaged]),
        }

    selected = next((m for m in grid if by_multiplier[str(m)]["phenomenon_qualifies"]), None) if fully_observed else None
    return {
        "status": "unable_incomplete_grid" if not fully_observed else ("advance_to_review" if selected else "no_eligible_multiplier"),
        "selected_multiplier": selected, "multipliers": by_multiplier, "smoke": metadata["smoke"],
        "separate_validation_plan": config["separate_validation_plan"] if selected else None,
        "frozen_commit": metadata["frozen_commit"], "frozen_tree": metadata["frozen_tree"],
        "config_sha256": metadata["config_sha256"], "prereg_sha256": metadata["prereg_sha256"],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("results", type=Path)
    args = p.parse_args()
    metadata = json.loads((args.results / "run.json").read_text())
    write_json(args.results / "calibration.json", judge(metadata, args.results))
    print("[codex] mechanical calibration summary saved", flush=True)


if __name__ == "__main__":
    main()
