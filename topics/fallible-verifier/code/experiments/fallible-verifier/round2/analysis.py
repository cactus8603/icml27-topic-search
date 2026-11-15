"""Equal-budget and full-training-support evaluation on a locked test split."""

import argparse
from collections import Counter, defaultdict
from itertools import product
import json
from pathlib import Path
import random

import numpy as np
from scipy.optimize import linprog

from protocol import CONFIG, ROOT, digest, require_freeze, write_json


REJECT, ACCEPT, GOLD = 0, 1, 2
DIRECT = ((REJECT,), (ACCEPT,), (GOLD,))
TREES = DIRECT + tuple(product(range(3), repeat=2))
CONTROL, CASCADE, EXTRA, FULL_CONTROL, FULL_CASCADE = (
    "equal_budget_gold_only", "equal_budget_cascade", "full_gold_same_probes",
    "full_support_gold_only", "full_support_cascade",
)


def warm_plan(n, config):
    c_s, c_g = config["strong_price_units"], config["gold_price_units"]
    # Gold-only uses an integer number of gold reviews. Rounding the probe
    # count to a multiple of c_g makes the strong spend exactly gold-divisible.
    base_gold = max(1, int(n * config["budget_fraction_at_gold_cost"]))
    probes = int(n * config["probe_fraction"]) // c_g * c_g
    if not 0 < probes <= n or probes * c_s >= base_gold * c_g:
        raise ValueError("warm-up sample is too small or budget cannot fund both observations")
    cascade_gold = base_gold - probes * c_s // c_g
    return probes, base_gold, cascade_gold


def cheap_state(row):
    return row["dataset"], row["solver_model"], row["weak"]


def available_states(rows):
    return sorted({cheap_state(row) for row in rows if row["weak"] is not None})


def sample_strata(strata, count, rng, uniform_share):
    if count < len(strata):
        return set(rng.sample([i for block in strata.values() for i in block], count))
    total = sum(map(len, strata.values()))
    targets = {key: count * (uniform_share * len(block) / total + (1 - uniform_share) / len(strata))
               for key, block in strata.items()}
    allocation = dict.fromkeys(strata, 1)
    for _ in range(count - len(strata)):
        key = max((key for key, block in strata.items() if allocation[key] < len(block)),
                  key=lambda key: targets[key] - allocation[key])
        allocation[key] += 1
    return {i for key, block in strata.items() for i in rng.sample(block, allocation[key])}


def warm_observations(train, config, seed):
    probes, base_gold, cascade_gold = warm_plan(len(train), config)
    rng = random.Random(seed)
    order = rng.sample(range(len(train)), len(train))
    probe_ids = set(order[:probes])
    probe_map = {i: train[i]["strong"] for i in probe_ids}
    cheap = defaultdict(list)
    augmented = defaultdict(list)
    for i, row in enumerate(train):
        x = cheap_state(row)
        cheap[x].append(i)
        augmented[x, probe_map.get(i, "missing")].append(i)
    base_audits = sample_strata(cheap, base_gold, random.Random(seed + 100_000), 0.5)
    cascade_audits = sample_strata(augmented, cascade_gold, random.Random(seed + 200_000), 0.5)
    return probe_map, base_audits, cascade_audits


def fit_distribution(rows, states, probes, audits, smoothing):
    """Fit the joint from observed probes and stratified audits only."""
    x_count = Counter(cheap_state(row) for row in rows)
    s_count = Counter((cheap_state(rows[i]), s) for i, s in probes.items())
    strata = Counter((cheap_state(row), probes.get(i, "missing")) for i, row in enumerate(rows))
    audited = Counter((cheap_state(rows[i]), probes.get(i, "missing")) for i in audits)
    n, pos, within_n, within_pos = (Counter() for _ in range(4))
    for i in audits:
        row = rows[i]
        y = row["gold"]
        if y is None:
            continue
        d, model, weak = cheap_state(row)
        probe = probes.get(i, "missing")
        stratum = (cheap_state(row), probe)
        # Prefer the estimated inclusion propensity to an unweighted pooled
        # prior, but retain effective counts for the individual S cells.
        weight = strata[stratum] / audited[stratum] * len(audits) / len(rows)
        keys = ((), (weak,), (weak, d), (weak, d, model))
        for key in keys:
            n[key] += weight
            pos[key] += weight * y
        within_n[weak, d, model, probe] += 1
        within_pos[weak, d, model, probe] += y

    def adjust(key, prior):
        return (pos[key] + smoothing * prior) / (n[key] + smoothing)

    distribution = np.zeros((len(states), 3, 2))
    for xi, x in enumerate(states):
        d, model, w = x
        rate = (pos[()] + 0.5) / (n[()] + 1)
        rate = adjust((w, d, model), adjust((w, d), adjust((w,), rate)))
        total_s = sum(s_count[(x, s)] for s in (0, 1, None))
        for si, s in enumerate((0, 1, None)):
            weight = (s_count[x, s] + 0.5) / (total_s + 1.5)
            child = (w, d, model, s)
            probability = (within_pos[child] + smoothing * rate) / (within_n[child] + smoothing)
            distribution[xi, si, :] = np.array([1 - probability, probability]) * weight * x_count[x]
    return distribution


def action_outcome(tree, s, costs):
    consult = len(tree) == 2
    decision = GOLD if consult and s is None else tree[s] if consult else tree[0]
    cost = costs[0] * consult + costs[1] * (decision == GOLD)
    return [cost, int(decision == ACCEPT), int(decision == REJECT), int(consult), int(decision == GOLD)]


def optimize(distribution, cascade, target, costs, states):
    options = TREES if cascade else DIRECT
    if not states:
        return {}
    objective = []
    coeffs = []
    for tensor in distribution:
        for tree in options:
            coefficients = np.zeros(3)
            for si, s in enumerate((0, 1, None)):
                cost, accept, reject, _, _ = action_outcome(tree, s, costs)
                coefficients += [tensor[si].sum() * cost, tensor[si, 0] * accept, tensor[si, 1] * reject]
            objective.append(coefficients[0])
            coeffs.append(coefficients[1:])
    solved = linprog(np.asarray(objective), A_ub=np.asarray(coeffs).T,
                     b_ub=target * distribution.sum(axis=(0, 1)),
                     A_eq=np.repeat(np.eye(len(states)), len(options), axis=1),
                     b_eq=np.ones(len(states)), bounds=(0, 1), method="highs")
    if not solved.success:
        raise ValueError(f"routing LP failed: {solved.message}")
    return dict(zip(states, zip([options] * len(states), solved.x.reshape(len(states), -1))))


def outcomes(rows, policy, costs):
    result = {}
    for row in rows:
        options, probabilities = policy.get(cheap_state(row), ((DIRECT[2],), (1,)))
        result[row["id"]] = sum((p * np.asarray(action_outcome(tree, row["strong"], costs))
                                 for tree, p in zip(options, probabilities, strict=True)), start=np.zeros(5))
    return result


def strong_signal(train, test, smoothing):
    if not train:
        return {}
    stats = Counter()
    positives = Counter()
    for row in train:
        if row["gold"] is None or row["weak"] is None:
            continue
        d, model, w = cheap_state(row)
        for key in ((w,), (w, d), (w, d, model), (w, d, model, row["strong"])):
            stats[key] += 1
            positives[key] += row["gold"]

    def adjust(key, prior):
        return (positives[key] + smoothing * prior) / (stats[key] + smoothing)

    results = {}
    for row in test:
        if row["gold"] is None or row["weak"] is None:
            continue
        d, model, w = cheap_state(row)
        x = (w, d, model)
        p_w = (positives[w,] + 0.5) / (stats[w,] + 1)
        p_parent = adjust((w, d), p_w)
        p = adjust(x, p_parent)
        s = row["strong"]
        n = stats[x]
        m = min(stats[(*x, 0)], stats[(*x, 1)])
        if s is None or m == 0:
            updated = p
        else:
            p_cell_raw = positives[(*x, s)] / stats[(*x, s)]
            p_cell = (n * p_cell_raw + smoothing * p_parent) / (n + smoothing)
            updated = p + (m / (m + smoothing)) * (p_cell - p)
        results[row["id"]] = [(row["gold"] - p) ** 2, (row["gold"] - updated) ** 2]
    return results


def quality_metrics(rows, score):
    table = defaultdict(lambda: np.zeros(10))
    for row in rows:
        gold, weak, strong = row["gold"], row["weak"], row["strong"]
        values = table[row["question_id"]]
        values[:8] += [gold == 0, gold == 0 and strong == ACCEPT,
                       gold == 0 and weak == strong == ACCEPT, gold is None,
                       gold is None and strong == ACCEPT, weak is None, strong is None, 1]
        values[8:] += score.get(row["id"], (0, 0))
    return table


def grouped(rows, first, second):
    """Sufficient statistics for cost, class risks and unknown-label worst-case bounds."""
    groups = defaultdict(lambda: np.zeros(22))
    for row in rows:
        old, new = first[row["id"]], second[row["id"]]
        values = groups[row["question_id"]]
        y = row["gold"]
        values[:8] += [1, old[0], new[0], old[3], new[3], old[4], new[4], y is None]
        for label, offset in ((0, 8), (1, 15)):
            a, b = old[1 + label], new[1 + label]
            values[offset:offset + 7] += [y == label, a * (y == label), b * (y == label),
                                          a * (y is None), b * (y is None),
                                          max(0.0, b - a) * (y is None), min(0.0, b - a) * (y is None)]
    return groups


def ratios(values, first_warm, second_warm):
    n, first, second, first_strong, second_strong, first_gold, second_gold, unknown = values[:8]
    old_cost = first / n + first_warm
    new_cost = second / n + second_warm
    output = dict(old_cost=old_cost, new_cost=new_cost,
                  relative_saving=np.nan if old_cost <= 0 else 1 - new_cost / old_cost,
                  old_deployment=first / n, new_deployment=second / n,
                  old_strong=first_strong / n, new_strong=second_strong / n,
                  old_gold=first_gold / n, new_gold=second_gold / n)
    for name, offset in (("fa", 8), ("fr", 15)):
        count, a, b, unknown_a, unknown_b, plus_delta, minus_delta = values[offset:offset + 7]
        if not count:
            output |= {key: np.nan for key in (f"old_{name}", f"new_{name}", f"{name}_increase", f"old_{name}_upper",
                                                f"new_{name}_upper", f"{name}_increase_upper")}
            continue
        hi = b - a + plus_delta
        output |= {f"old_{name}": a / count, f"new_{name}": b / count,
                   f"{name}_increase": (b - a) / count,
                   f"old_{name}_upper": min(1, (a + unknown_a) / (count + unknown_a)),
                   f"new_{name}_upper": min(1, (b + unknown_b) / (count + unknown_b)),
                   f"{name}_increase_upper": hi / (count if hi >= 0 else count + unknown)}
    return output


def bootstrap(rows, draws, seed):
    observed = {row["question_id"]: row["dataset"] for row in rows}
    groups = sorted(observed)
    by_dataset = defaultdict(list)
    for index, question in enumerate(groups):
        by_dataset[observed[question]].append(index)
    rng = np.random.default_rng(seed)
    weights = np.zeros((draws, len(groups)), dtype=int)
    for block in by_dataset.values():
        weights[:, block] = rng.multinomial(len(block), np.ones(len(block)) / len(block), size=draws)
    return groups, weights


def paired_intervals(point, samples, tail):
    report = {}
    for key, value in point.items():
        valid = np.asarray([sample[key] for sample in samples])
        valid = valid[np.isfinite(valid)]
        bounds = np.percentile(valid, [100 * tail, 2.5, 97.5, 100 * (1 - tail)]).tolist() if len(valid) else [None] * 4
        report[key] = {"point": float(value) if np.isfinite(value) else None, "ci95": bounds[1:3],
                       "adjusted_lower": bounds[0], "adjusted_upper": bounds[-1], "defined_draws": len(valid)}
    return report


def classify(result, limit, larger_is_better, ready, draws, allow_opposite=True):
    if not ready or result["defined_draws"] != draws:
        return "inconclusive"
    low, high = result["adjusted_lower"], result["adjusted_upper"]
    if larger_is_better:
        if low >= limit:
            return "supported"
        if allow_opposite and high < limit:
            return "below_floor"
    else:
        if high <= limit:
            return "supported"
        if allow_opposite and low > limit:
            return "above_limit"
    return "inconclusive"


def measure(train, test, config):
    unit = config["gold_price_units"]
    costs = config["strong_price_units"] / unit, 1.0
    sites = available_states(train)
    target = config["train_risk_target"]
    smooth = config["smoothing"]
    sums = {label: defaultdict(lambda: np.zeros(5))
            for label in (CONTROL, CASCADE, EXTRA, FULL_CONTROL, FULL_CASCADE)}
    probes, base_gold, cascade_gold = warm_plan(len(train), config)
    n_seeds = len(config["audit_seeds"])
    audit_missing = Counter()

    def accumulate(label, info_s, info_y, cascade, weight):
        fit = fit_distribution(train, sites, info_s, info_y, smooth)
        policy = optimize(fit, cascade, target, costs, sites)
        for key, values in outcomes(test, policy, costs).items():
            sums[label][key] += weight * values
        audit_missing[label] += weight * sum(train[i]["gold"] is None for i in info_y)

    for seed in config["audit_seeds"]:
        probe_map, base, reviewed = warm_observations(train, config, seed)
        if len(base) * unit != len(reviewed) * unit + len(probe_map) * config["strong_price_units"]:
            raise ValueError("unequal warm-up spend")
        accumulate(CONTROL, {}, base, False, 1 / n_seeds)
        accumulate(CASCADE, probe_map, reviewed, True, 1 / n_seeds)
        accumulate(EXTRA, probe_map, range(len(train)), True, 1 / n_seeds)
    all_strong = {i: row["strong"] for i, row in enumerate(train)}
    accumulate(FULL_CONTROL, {}, range(len(train)), False, 1)
    accumulate(FULL_CASCADE, all_strong, range(len(train)), True, 1)

    plan = {CONTROL: (0, base_gold), CASCADE: (probes, cascade_gold),
            EXTRA: (probes, len(train)), FULL_CONTROL: (0, len(train)),
            FULL_CASCADE: (len(train), len(train))}
    deployment = config["warm_amortization_candidates"]
    warm = {key: (strong * costs[0] + gold) / deployment for key, (strong, gold) in plan.items()}
    groups, boot = bootstrap(test, config["bootstrap_draws"], config["bootstrap_seed"])
    tail = config["bootstrap_joint_alpha"] / config["max_test_family"]
    comparisons = {}
    for first, second in ((CONTROL, CASCADE), (CASCADE, EXTRA), (CASCADE, FULL_CASCADE),
                          (FULL_CONTROL, FULL_CASCADE)):
        table = grouped(test, sums[first], sums[second])
        matrix = np.array([table[group] for group in groups])
        for view, amortization in (("amortized", warm), ("deployment_only", dict.fromkeys(warm, 0.0))):
            first_warm, second_warm = amortization[first], amortization[second]
            comparisons[f"{second}_vs_{first}_{view}"] = paired_intervals(
                ratios(matrix.sum(axis=0), first_warm, second_warm),
                [ratios(values, first_warm, second_warm) for values in boot @ matrix],
                tail,
            )

    scores = strong_signal(train, test, smooth)
    quality = quality_metrics(test, scores)
    matrix = np.array([quality[group] for group in groups])

    def quality_stats(values):
        negative, strong_fa, shared, unknown, unknown_accept, weak_na, strong_na, size, low, high = values
        return {"strong_fa": strong_fa / negative if negative else np.nan,
                "shared_fa": shared / strong_fa if strong_fa else np.nan,
                "strong_fa_unknown_lower": strong_fa / (negative + unknown) if negative else np.nan,
                "shared_fa_unknown_lower": shared / (strong_fa + unknown_accept) if strong_fa else np.nan,
                "weak_parse_failure": weak_na / size, "strong_parse_failure": strong_na / size,
                "gold_parse_coverage": 1 - unknown / size,
                "matched_brier_gain": 1 - high / low if low else np.nan}

    diag = paired_intervals(quality_stats(matrix.sum(axis=0)),
                            [quality_stats(values) for values in boot @ matrix], tail)
    counts = {str(label): sum(row["gold"] == label for row in test) for label in (0, 1, None)}
    train_known = sum(row["gold"] is not None for row in train)
    coverage = min(train_known / len(train), diag["gold_parse_coverage"]["point"])
    ready = coverage >= config["min_gold_parse_coverage"] and min(counts["0"], counts["1"]) >= config["min_test_each_truth"]
    primary = comparisons[f"{CASCADE}_vs_{CONTROL}_amortized"]
    headroom = comparisons[f"{FULL_CASCADE}_vs_{FULL_CONTROL}_deployment_only"]

    def classification(table, metric, limit, larger_is_better, partial_truth_bound=False):
        return classify(table[metric], limit, larger_is_better, ready, config["bootstrap_draws"],
                        allow_opposite=not partial_truth_bound or counts["None"] == 0)

    def risk_status(table):
        return {
            key: classification(table, key, limit, False, partial_truth_bound=True)
            for risk in ("fa", "fr")
            for key, limit in ((f"old_{risk}_upper", config["max_absolute_risk"]),
                               (f"new_{risk}_upper", config["max_absolute_risk"]),
                               (f"{risk}_increase_upper", config["max_paired_risk_increase"]))
        }

    def budget_status(table):
        return {"risk_status": risk_status(table),
                "cost_saving": classification(table, "relative_saving", config["min_cost_saving"], True)}

    return {
        "counts": {"train_rows": len(train), "train_gold_resolved": train_known, "test_rows": len(test),
                   "test_questions": len(groups), "test_gold": counts, "test_signal_rows": len(scores),
                   "unresolved_train_gold": dict(Counter(row.get("gold_unknown_reason") or "unspecified"
                                                         for row in train if row["gold"] is None)),
                   "unresolved_test_gold": dict(Counter(row.get("gold_unknown_reason") or "unspecified"
                                                        for row in test if row["gold"] is None))},
        "warm_plan": {key: {"strong": strong, "gold": gold, "parse_unknown_gold_mean": audit_missing[key],
                            "price_units": strong * config["strong_price_units"] + gold * unit,
                            "amortized_gold_units": warm[key]} for key, (strong, gold) in plan.items()},
        "diagnostics": diag, "comparisons": comparisons,
        "adjusted_one_sided_alpha": tail,
        "mechanical": {
            "coverage_sufficient": ready,
            "phenomenon": {
                "strong_false_accept": classification(diag, "strong_fa_unknown_lower", config["min_fa"], True,
                                                      partial_truth_bound=True),
                "shared_false_accept": classification(diag, "shared_fa_unknown_lower", config["min_shared_fa"], True,
                                                      partial_truth_bound=True),
            },
            "conditional_signal_on_parsed_gold": classification(diag, "matched_brier_gain", config["min_brier_gain"], True),
            "full_support_opportunity": budget_status(headroom),
            "equal_budget_cost_comparison": budget_status(primary),
        },
    }


def merge(config, root, frozen):
    files = {label: root / f"{label}.json" for label in ("gold", "weak", "strong", "prompts")}
    sources = {label: json.loads(path.read_text()) for label, path in files.items()}
    for label, value in sources.items():
        if value["freeze"] != frozen or value["config_sha256"] != digest(CONFIG):
            raise ValueError(f"stale {label} input")
    gold = sources["gold"]["rows"]
    prompts = sources["prompts"]["rows"]
    if [r["id"] for r in gold] != [r["id"] for r in prompts]:
        raise ValueError("gold and prompt manifest IDs diverge")
    by_role = {}
    runtime = {}
    for role in ("weak", "strong"):
        source = sources[role]
        specification = config["models"][role]
        if (source["prompts_sha256"] != digest(files["prompts"]) or source["model"] != specification["id"]
                or source["model_revision"] != specification["revision"]):
            raise ValueError(f"stale model or manifest for {role}")
        records = source["records"]
        if set(records) != {row["id"] for row in gold} or len(records) != len(gold):
            raise ValueError(f"missing, duplicate or extra {role} verdicts")
        by_role[role] = records
        runtime[role] = {"seconds": source["runtime_seconds"], "vllm": source["vllm"],
                         "input_truncations": sum(v["input_truncated"] for v in records.values()),
                         "output_truncations": sum(v["finish"] == "length" for v in records.values()),
                         "parse_failures": sum(v["verdict"] is None for v in records.values()),
                         "prompt_tokens": sum(v["prompt_tokens"] for v in records.values()),
                         "output_tokens": sum(v["output_tokens"] for v in records.values())}
    rows = [row | {role: by_role[role][row["id"]]["verdict"] for role in by_role} for row in gold]
    train = [row for row in rows if row["split"] == "train"]
    test = [row for row in rows if row["split"] == "test"]
    if (len(train) + len(test) != len(rows) or not train or not test
            or {row["question_id"] for row in train} & {row["question_id"] for row in test}):
        raise ValueError("invalid train or held-out test split")
    return measure(train, test, config) | {"model_runtime": runtime,
                                          "input_sha256": {name: digest(path) for name, path in files.items()},
                                          "scheduler": sources["prompts"]["scheduler"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    arguments = parser.parse_args()
    if not arguments.run_dir.resolve().is_relative_to(ROOT / "results"):
        parser.error("--run-dir must be under round2/results/")
    destination = arguments.run_dir / "summary.json"
    if destination.exists():
        raise FileExistsError(destination)
    frozen = require_freeze(arguments.freeze)
    config = json.loads(CONFIG.read_text())
    result = merge(config, arguments.run_dir, frozen)
    write_json(destination, {"freeze": frozen, "config_sha256": digest(CONFIG)} | result)
    print("[codex] Wrote held-out paired analysis after freeze.")


if __name__ == "__main__":
    main()
