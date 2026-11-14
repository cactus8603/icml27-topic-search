"""Sampling and optimal mixtures of small verification trees."""

from collections import Counter, defaultdict
from itertools import product

import numpy as np
from scipy.optimize import linprog


ACCEPT, REJECT, GOLD = range(3)


def actions(cascade):
    direct = ((ACCEPT,), (REJECT,), (GOLD,))
    return direct + tuple(product(range(3), repeat=2)) if cascade else direct


def outcome(action, strong, gold, strong_cost, gold_cost):
    consulted = len(action) == 2
    decision = action[strong] if consulted else action[0]
    reviewed = decision == GOLD
    return (
        strong_cost * consulted + gold_cost * reviewed,
        int(decision == ACCEPT and gold == 0),
        int(decision == REJECT and gold == 1),
        int(consulted),
        int(reviewed),
    )


def fit_policy(distribution, cascade, target, strong_cost, gold_cost):
    """Minimize cost under two class-conditional error constraints."""
    options = actions(cascade)
    objective = []
    error_coefficients = []
    for weak in range(2):
        for action in options:
            coefficients = sum(
                (
                    distribution[weak, strong, gold]
                    * np.asarray(outcome(action, strong, gold, strong_cost, gold_cost)[:3])
                    for strong, gold in product(range(2), repeat=2)
                ),
                start=np.zeros(3),
            )
            objective.append(coefficients[0])
            error_coefficients.append(coefficients[1:])

    equality = np.repeat(np.eye(2), len(options), axis=1)
    result = linprog(
        objective,
        A_ub=np.asarray(error_coefficients).T,
        b_ub=target * distribution.sum(axis=(0, 1)),
        A_eq=equality,
        b_eq=np.ones(2),
        bounds=(0, 1),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"infeasible verification policy: {result.message}")
    return options, result.x.reshape(2, -1)


def deploy(policy, rows, strong_cost, gold_cost):
    options, probabilities = policy
    stats = {}
    for row in rows:
        values = [0.0] * 7
        for action, probability in zip(options, probabilities[row.weak]):
            if probability > 0:
                for i, value in enumerate(outcome(action, row.strong, row.gold, strong_cost, gold_cost)):
                    values[i] += float(probability) * value
        values[5 + row.gold] = 1
        if row.group not in stats:
            stats[row.group] = [0.0] * 7
        for i, value in enumerate(values):
            stats[row.group][i] += value
    return stats


def full_distribution(rows):
    mass = np.zeros((2, 2, 2))
    for row in rows:
        mass[row.weak, row.strong, row.gold] += 1
    return mass / len(rows)


def split_by_question(rows, folds, rng):
    groups = defaultdict(set)
    for row in rows:
        groups[row.group].add(row.dataset)
    by_dataset = defaultdict(list)
    for group, datasets in sorted(groups.items()):
        by_dataset[min(datasets)].append(group)
    assignment = {}
    offset = 0
    for block in by_dataset.values():
        rng.shuffle(block)
        for index, group in enumerate(block):
            assignment[group] = (index + offset) % folds
        offset += len(block)
    if len(set(assignment.values())) != folds:
        raise ValueError("too few unique questions for grouped cross-validation")
    return assignment


def probe_indices(weak, fraction, rng):
    result = []
    for value in range(2):
        indices = [i for i, verdict in enumerate(weak) if verdict == value]
        count = min(len(indices), max(1, round(len(indices) * fraction)))
        result.extend(rng.sample(indices, count))
    return sorted(result)


def stratified_audits(weak, observed_strong, count, uniform_fraction, rng):
    strata = defaultdict(list)
    for index, value in enumerate(weak):
        strata[value, observed_strong.get(index, -1)].append(index)
    if count < len(strata):
        return sorted(rng.sample(range(len(weak)), count))
    targets = {
        key: count * (uniform_fraction * len(indices) / len(weak)
                      + (1 - uniform_fraction) / len(strata))
        for key, indices in strata.items()
    }
    allocation = {key: 1 for key in strata}
    for _ in range(count - len(strata)):
        key = max(
            (key for key, indices in strata.items() if allocation[key] < len(indices)),
            key=lambda key: (targets[key] - allocation[key], key),
        )
        allocation[key] += 1
    return sorted(
        index for key, indices in sorted(strata.items())
        for index in rng.sample(indices, allocation[key])
    )


def sparse_distribution(weak, observed_strong, observed_gold, smoothing):
    """Fit from the exposed masks. Post-stratify gold by observable probe strata."""
    global_rate = (sum(observed_gold.values()) + 0.5) / (len(observed_gold) + 1)
    count_weak = Counter(weak)
    stratum_sizes = Counter((w, observed_strong.get(i, -1)) for i, w in enumerate(weak))
    labeled_strata = defaultdict(list)
    for index, gold in observed_gold.items():
        labeled_strata[weak[index], observed_strong.get(index, -1)].append(gold)

    result = np.zeros((2, 2, 2))
    for w in range(2):
        if not count_weak[w]:
            continue
        labeled_weak = [y for (v, _), values in labeled_strata.items() if w == v for y in values]
        weak_rate = (sum(labeled_weak) + smoothing * global_rate) / (len(labeled_weak) + smoothing)
        rates = {}
        for s in (-1, 0, 1):
            labels = labeled_strata[w, s]
            rates[s] = (sum(labels) + smoothing * weak_rate) / (len(labels) + smoothing)
        probe_count = stratum_sizes[w, 0] + stratum_sizes[w, 1]
        if not probe_count:
            result[w, 0, :] = (1 - rates[-1], rates[-1])
        else:
            for s in range(2):
                share = stratum_sizes[w, s] / probe_count
                result[w, s, :] = share * np.asarray([1 - rates[s], rates[s]])
        result[w, :, :] *= count_weak[w] / len(weak)
    return result
