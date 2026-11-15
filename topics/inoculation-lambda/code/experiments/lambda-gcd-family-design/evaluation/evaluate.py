"""Compare frozen leakage predictions on each held-out GCD defense family."""

import argparse
from dataclasses import dataclass
from itertools import combinations
import json
import math
from pathlib import Path
from statistics import fmean


@dataclass(frozen=True)
class Arm:
    id: str
    family: str
    lower: float
    upper: float
    scores: dict[str, float]

    @property
    def midpoint(self) -> float:
        return (self.lower + self.upper) / 2


def identifier(value) -> bool:
    return isinstance(value, str) and bool(value) and value == value.strip()


def finite_number(value, name: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def read_panel(payload) -> tuple[dict[str, str], list[Arm]]:
    if (not isinstance(payload, dict) or type(payload.get("schema")) is not int
            or payload["schema"] != 1 or payload.get("task") != "gcd"):
        raise ValueError("Expected a schema 1 GCD panel")
    methods = payload.get("methods")
    if (not isinstance(methods, dict) or len(methods) < 2 or methods.get("lambda") != "full"
            or any(not identifier(name) or support not in ("full", "common_support")
                   for name, support in methods.items())):
        raise ValueError("Declare lambda and at least one baseline with full or common_support coverage")
    rows = payload.get("arms")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a nonempty list of GCD defense arms")

    arms = []
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or not identifier(row.get("id")) or not identifier(row.get("family")):
            raise ValueError("Each arm needs a nonempty id and family")
        arm_id, family = row["id"], row["family"]
        if arm_id in seen:
            raise ValueError(f"Duplicate arm id {arm_id}")
        seen.add(arm_id)
        leakage = row.get("leakage")
        if not isinstance(leakage, dict):
            raise ValueError(f"Missing leakage bounds for {arm_id}")
        lower = finite_number(leakage.get("lower"), f"{arm_id} leakage lower")
        upper = finite_number(leakage.get("upper"), f"{arm_id} leakage upper")
        if not 0 <= lower <= upper <= 1:
            raise ValueError(f"Invalid leakage interval for {arm_id}")
        raw_scores = row.get("scores")
        if not isinstance(raw_scores, dict) or raw_scores.keys() - methods.keys():
            raise ValueError(f"{arm_id} scores must use the declared method IDs")
        scores = {}
        for method, support in methods.items():
            value = raw_scores.get(method)
            if value is None and support == "common_support":
                continue
            scores[method] = finite_number(value, f"{arm_id} {method} score")
        arms.append(Arm(arm_id, family, lower, upper, scores))
    return methods, sorted(arms, key=lambda arm: (arm.family, arm.id))


def sign(left: float, right: float) -> int:
    return int(left > right) - int(left < right)


def possible_signs(left: Arm, right: Arm) -> list[int]:
    signs = []
    if left.lower < right.upper:
        signs.append(-1)
    if left.lower <= right.upper and right.lower <= left.upper:
        signs.append(0)
    if left.upper > right.lower:
        signs.append(1)
    return signs


def concordance(predicted: int, observed: int) -> float:
    if predicted == 0 or observed == 0:
        return 0.5
    return float(predicted == observed)


def interval(point: float, lower: float, upper: float) -> dict[str, float]:
    return {"point": point, "lower": lower, "upper": upper}


def average(intervals: list[dict[str, float]]) -> dict[str, float]:
    return {key: fmean(item[key] for item in intervals) for key in ("point", "lower", "upper")}


def regret(selected: Arm, common: list[Arm]) -> dict[str, float]:
    point = selected.midpoint - min(arm.midpoint for arm in common)
    other_arms = [arm for arm in common if arm.id != selected.id]
    lower = max(0.0, selected.lower - min(arm.upper for arm in other_arms))
    upper = max(0.0, selected.upper - min(arm.lower for arm in other_arms))
    return interval(point, lower, upper)


def compare_family(arms: list[Arm], baseline: str) -> dict:
    common = [arm for arm in arms if baseline in arm.scores]
    counts = {
        "family_arm_ids": [arm.id for arm in arms],
        "common_arm_ids": [arm.id for arm in common],
        "common_pairs": math.comb(len(common), 2),
    }
    if len(common) < 2:
        return counts | {"status": "insufficient_common_support"}
    methods = ("lambda", baseline)
    variable = {method: len({arm.scores[method] for arm in common}) > 1 for method in methods}

    measures = {"lambda_concordance": [], "baseline_concordance": [], "lambda_minus_baseline_concordance": []}
    comparisons_determined = 0
    for left, right in combinations(common, 2):
        midpoint_sign = sign(left.midpoint, right.midpoint)
        solutions = possible_signs(left, right)
        comparisons_determined += len(solutions) == 1
        predicted = sign(left.scores["lambda"], right.scores["lambda"])
        reference = sign(left.scores[baseline], right.scores[baseline])
        lambda_point, baseline_point = concordance(predicted, midpoint_sign), concordance(reference, midpoint_sign)
        lambda_values = [concordance(predicted, solution) for solution in solutions]
        baseline_values = [concordance(reference, solution) for solution in solutions]
        deltas = [first - second for first, second in zip(lambda_values, baseline_values, strict=True)]
        measures["lambda_concordance"].append(interval(lambda_point, min(lambda_values), max(lambda_values)))
        measures["baseline_concordance"].append(interval(baseline_point, min(baseline_values), max(baseline_values)))
        measures["lambda_minus_baseline_concordance"].append(
            interval(lambda_point - baseline_point, min(deltas), max(deltas))
        )

    preferred = {method: min(common, key=lambda arm: (arm.scores[method], arm.id)) for method in methods}
    first, second = preferred["lambda"], preferred[baseline]
    if first.id == second.id:
        regret_difference = interval(0.0, 0.0, 0.0)
    else:
        regret_difference = interval(first.midpoint - second.midpoint, first.lower - second.upper,
                                     first.upper - second.lower)
    return counts | {
        "status": "eligible" if all(variable.values()) else "constant_prediction",
        "method_scores_vary": variable,
        "outcome_direction_determined_pairs": comparisons_determined,
        **{name: average(rows) for name, rows in measures.items()},
        "selected_arms": {method: arm.id for method, arm in preferred.items()},
        "lambda_regret": regret(first, common),
        "baseline_regret": regret(second, common),
        "lambda_minus_baseline_regret": regret_difference,
    }


def evaluate(payload) -> dict:
    methods, arms = read_panel(payload)
    families = {}
    for arm in arms:
        families.setdefault(arm.family, []).append(arm)

    comparisons = {}
    for method, support in sorted(methods.items()):
        if method == "lambda":
            continue
        folds = {family: compare_family(rows, method) for family, rows in families.items()}
        eligible = [family for family, fold in folds.items() if fold["status"] == "eligible"]
        keys = ("lambda_concordance", "baseline_concordance", "lambda_minus_baseline_concordance",
                "lambda_regret", "baseline_regret", "lambda_minus_baseline_regret")
        macro = {key: average([folds[family][key] for family in eligible]) for key in keys} if eligible else None
        comparisons[method] = {
            "declared_support": support,
            "folds": folds,
            "macro_eligible_families": eligible,
            "macro_common_pairs": sum(folds[family]["common_pairs"] for family in eligible),
            "macro_is_across_multiple_families": len(eligible) >= 2,
            "macro_covers_all_families_and_arms": len(eligible) == len(families) and all(
                fold["family_arm_ids"] == fold["common_arm_ids"] for fold in folds.values()
            ),
            "macro": macro,
        }
    return {"schema": 1, "task": "gcd", "scope": "frozen_zero_shot_by_family", "point_outcome": "interval_midpoint",
            "family_ids": list(families), "comparisons": comparisons}


def unique_keys(pairs):
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("Duplicate key in the panel JSON")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, help="Write an output file that does not already exist")
    args = parser.parse_args()
    try:
        payload = json.loads(args.input.read_text(encoding="utf-8"), object_pairs_hook=unique_keys)
        result = json.dumps(evaluate(payload), indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        if args.output is None:
            print(result, end="")
        else:
            with args.output.open("x", encoding="utf-8") as handle:
                handle.write(result)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"GCD family evaluation: {exc}\n")


if __name__ == "__main__":
    main()
