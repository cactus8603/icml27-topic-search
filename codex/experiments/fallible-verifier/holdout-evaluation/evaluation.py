"""CPU-only input validation and candidate scoring tools for an unselected policy."""

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np


TOPIC = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(TOPIC / "real_holdout"), str(TOPIC / "round2")]
import analysis as reference
from prepare import EXPECTED_OUTPUTS, VARIANTS
from protocol import hash_order


PANEL_COMMIT = "37a8a783bd7b1f7648dfbf0376cbbb99099cad58"
SPLIT_GROUPS = {"train": 250, "test": 747}
ROLES = ("weak", "strong")
MODEL_PINS = {
    "weak": ("Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca"),
    "strong": ("Qwen/Qwen3-32B-AWQ", "0499c3ac83fdef8810b907a23894ba91e95eddd8"),
}
DIGEST = re.compile(r"[0-9a-f]{64}")
GOLD_FIELDS = {"id", "question_id", "split", "variant", "gold", "gold_status", "publisher_is_correct"}
GOLD_STATES = {"corroborated", "empty_candidate", "publisher_reference_mismatch",
               "unparseable_final_answer", "publisher_label_conflict"}
MODEL_STATES = {"observed", "abstain", "missing"}
ACTION_INDEX = {reference.ACCEPT: 0, reference.REJECT: 1, reference.GOLD: 2}
ARMS = (reference.CONTROL, reference.CASCADE, reference.EXTRA, reference.FULL_CONTROL, reference.FULL_CASCADE)
PAIRS = ((reference.CONTROL, reference.CASCADE), (reference.CASCADE, reference.EXTRA),
         (reference.CASCADE, reference.FULL_CASCADE), (reference.FULL_CONTROL, reference.FULL_CASCADE))


def unique_fields(pairs):
    output = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("Duplicate JSON field")
        output[key] = value
    return output


def load_json(payload):
    return json.loads(payload, object_pairs_hook=unique_fields)


def checked_bytes(payload, expected):
    if not DIGEST.fullmatch(expected) or hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError("Input artifact bytes do not match their frozen receipt")


@dataclass(frozen=True)
class Binding:
    """Chosen later by the formal protocol, independently of test measurements."""

    freeze_commit: str
    measurement_sha256: str
    decision_rule_sha256: str

    def metadata(self):
        if (not re.fullmatch(r"[0-9a-f]{40}", self.freeze_commit)
                or any(not DIGEST.fullmatch(value) for value in (self.measurement_sha256, self.decision_rule_sha256))):
            raise ValueError("An explicit freeze, measurement protocol and decision rule are required")
        return {
            "freeze_commit": self.freeze_commit, "measurement_sha256": self.measurement_sha256,
            "decision_rule_sha256": self.decision_rule_sha256, "panel_commit": PANEL_COMMIT,
            "panel_prompts_sha256": EXPECTED_OUTPUTS["prompts.jsonl"]["sha256"],
            "panel_gold_sha256": EXPECTED_OUTPUTS["gold.jsonl"]["sha256"],
        }


@dataclass(frozen=True)
class Split:
    name: str
    binding: Binding
    question_ids: frozenset
    rows: tuple


def ingest_split(gold_bytes, role_bytes, hashes, binding, split, prompt_manifest_sha256):
    """Join ID-only grade rows and binary role receipts, retaining explicit missing statuses."""
    if split not in SPLIT_GROUPS or not DIGEST.fullmatch(prompt_manifest_sha256):
        raise ValueError("Unknown split or invalid input-manifest digest")
    if set(role_bytes) != set(ROLES) or set(hashes) != set(ROLES):
        raise ValueError("Both independently pinned model receipts and gold are required")
    expected_binding = binding.metadata()
    checked_bytes(gold_bytes, expected_binding["panel_gold_sha256"])
    split_field = (f',"split":"{split}","variant":').encode("ascii")
    gold = {}
    clusters = defaultdict(set)
    for line in gold_bytes.splitlines():
        # Match only the last two fields of the canonical JSONL row, before decoding.
        if not any(line.endswith(split_field + f'"{variant}"}}'.encode("ascii")) for variant in VARIANTS):
            continue
        row = load_json(line)
        if set(row) != GOLD_FIELDS:
            raise ValueError("Gold must contain only IDs and published/corroborated grade fields")
        key, qid, variant = row["id"], row["question_id"], row["variant"]
        y, status = row["gold"], row["gold_status"]
        if (row["split"] != split or not isinstance(qid, str) or not DIGEST.fullmatch(qid)
                or variant not in VARIANTS or key != hash_order(qid, variant) or key in gold
                or status not in GOLD_STATES or type(row["publisher_is_correct"]) is not bool
                or (y is not None and type(y) is not bool)
                or (status == "corroborated") != (y is not None)
                or (y is not None and y != row["publisher_is_correct"])):
            raise ValueError("Gold row has invalid identity, split or truth status")
        gold[key] = row
        clusters[qid].add(variant)
    if len(clusters) != SPLIT_GROUPS[split] or any(items != set(VARIANTS) for items in clusters.values()):
        raise ValueError("The full split needs all four variants in every registered question cluster")

    by_role = {}
    for role in ROLES:
        checked_bytes(role_bytes[role], hashes[role])
        receipt = load_json(role_bytes[role])
        model, revision = MODEL_PINS[role]
        pins = dict(schema=1, role=role, model=model, model_revision=revision, split=split,
                    binding=expected_binding, prompt_manifest_sha256=prompt_manifest_sha256)
        if set(receipt) != {"records", *pins} or any(receipt[k] != v for k, v in pins.items()):
            raise ValueError(f"Stale or misbound measurement receipt for {role}")
        records = {}
        for record in receipt["records"]:
            if set(record) != {"id", "verdict", "status"} or record["id"] in records:
                raise ValueError(f"Duplicate or invalid {role} measurement row")
            verdict, status = record["verdict"], record["status"]
            if (status not in MODEL_STATES or (status == "observed") != (verdict is not None)
                    or (verdict is not None and (type(verdict) is not int or verdict not in (0, 1)))):
                raise ValueError(f"Invalid {role} measurement status or verdict")
            records[record["id"]] = (verdict, status)
        if records.keys() != gold.keys():
            raise ValueError(f"Missing, extra or wrong-split {role} IDs")
        by_role[role] = records

    rows = []
    for key, row in sorted(gold.items(), key=lambda item: (item[1]["question_id"], VARIANTS.index(item[1]["variant"]))):
        rows.append({"id": key, "question_id": row["question_id"], "dataset": "gsm8k",
                     "solver_model": row["variant"], "gold": row["gold"], "gold_status": row["gold_status"],
                     **{role: by_role[role][key][0] for role in ROLES},
                     **{f"{role}_status": by_role[role][key][1] for role in ROLES}})
    return Split(split, binding, frozenset(clusters), tuple(rows))


@dataclass(frozen=True)
class BinaryFit:
    train: Split
    policies: dict
    plan: dict
    unknown_gold_by_arm: dict
    cost_units: tuple
    amortization_candidates: int


def fit_round2_binary(train, config):
    """Fit the Round 2 binary LP as a candidate policy on development data only."""
    if train.name != "train":
        raise ValueError("Policy fitting requires the development split")
    c_s, c_g, amortize = (config[key] for key in ("strong_price_units", "gold_price_units",
                                                 "warm_amortization_candidates"))
    seeds = config["audit_seeds"]
    target, smoothing = config["train_risk_target"], config["smoothing"]
    if (any(type(x) is not int or x <= 0 for x in (c_s, c_g, amortize)) or not seeds
            or len(set(seeds)) != len(seeds) or any(type(x) is not int for x in seeds)
            or not 0 <= target <= 1 or not math.isfinite(smoothing) or smoothing <= 0
            or any(not 0 < config[key] <= 1 for key in ("probe_fraction", "budget_fraction_at_gold_cost"))):
        raise ValueError("Supply explicit valid Round 2 policy inputs")
    costs = c_s / c_g, 1.0
    rows = train.rows
    states = reference.available_states(rows)
    probes, base_gold, cascade_gold = reference.warm_plan(len(rows), config)
    if probes * c_s + cascade_gold * c_g != base_gold * c_g:
        raise ValueError("Warm-up prices do not produce equal candidate-policy spending")
    policies = {label: [] for label in ARMS}
    unknown = Counter()

    def fit(arm, strong, gold, cascade):
        distribution = reference.fit_distribution(rows, states, strong, gold, smoothing)
        policies[arm].append(reference.optimize(distribution, cascade, target, costs, states))
        unknown[arm] += sum(rows[i]["gold"] is None for i in gold)

    all_indices = range(len(rows))
    for seed in seeds:
        seen, base, cascade = reference.warm_observations(rows, config, seed)
        fit(reference.CONTROL, {}, base, False)
        fit(reference.CASCADE, seen, cascade, True)
        fit(reference.EXTRA, seen, all_indices, True)
    fit(reference.FULL_CONTROL, {}, all_indices, False)
    fit(reference.FULL_CASCADE, {i: row["strong"] for i, row in enumerate(rows)}, all_indices, True)
    plan = {reference.CONTROL: (0, base_gold), reference.CASCADE: (probes, cascade_gold),
            reference.EXTRA: (probes, len(rows)), reference.FULL_CONTROL: (0, len(rows)),
            reference.FULL_CASCADE: (len(rows), len(rows))}
    return BinaryFit(train, policies, plan, {arm: unknown[arm] / len(policies[arm]) for arm in ARMS},
                     costs, amortize)


def coverage(split):
    return {
        "questions": len(split.question_ids), "candidates": len(split.rows),
        "gold": dict(Counter("unknown" if r["gold"] is None else ("correct" if r["gold"] else "wrong")
                             for r in split.rows)),
        "unknown_gold": dict(Counter(r["gold_status"] for r in split.rows if r["gold"] is None)),
        **{role: dict(Counter(r[f"{role}_status"] for r in split.rows)) for role in ROLES},
    }


def compare_actions(test, first, second, *, costs, first_warm, second_warm, bootstrap):
    """Compare already fixed expected action masses without selecting thresholds or models."""
    if test.name != "test":
        raise ValueError("Policy comparison requires the locked test split")
    c_s, c_g = costs
    warm = (first_warm, second_warm)
    if (any(not math.isfinite(c) or c <= 0 for c in costs)
            or any(not math.isfinite(v) or v < 0 for v in warm)):
        raise ValueError("Invalid declared deployment or warm-up price")
    draws, seed, tail = (bootstrap[key] for key in ("draws", "seed", "tail"))
    if type(draws) is not int or draws <= 0 or type(seed) is not int or seed < 0 or not 0 < tail < 0.5:
        raise ValueError("Supply an explicit paired resampling plan")
    ids = {row["id"] for row in test.rows}
    if first.keys() != ids or second.keys() != ids:
        raise ValueError("Both policies need expected actions for every locked candidate")

    def convert(outcomes):
        converted = {}
        for row in test.rows:
            key = row["id"]
            mass = np.asarray(outcomes[key], dtype=float)
            # Masses are direct accept/reject/gold, then strong-to-accept/reject/gold.
            if (mass.shape != (6,) or not np.isfinite(mass).all() or (mass < -1e-8).any()
                    or not np.isclose(mass.sum(), 1.0, rtol=0, atol=1e-8)):
                raise ValueError("An action distribution has invalid or incomplete probability mass")
            da, dr, dg, sa, sr, sg = mass.clip(0, 1)
            strong = sa + sr + sg
            if row["weak"] is None and not np.isclose(dg, 1, rtol=0, atol=1e-8):
                raise ValueError("A missing or abstaining weak measurement must fall back to direct gold")
            if row["strong"] is None and sa + sr > 1e-8:
                raise ValueError("A paid unavailable strong measurement must fall back to gold")
            gold = dg + sg
            converted[key] = np.asarray([c_s * strong + c_g * gold, da + sa, dr + sr, strong, gold])
        return converted

    groups, weights = reference.bootstrap(test.rows, draws, seed)
    table = reference.grouped(test.rows, convert(first), convert(second))
    matrix = np.asarray([table[group] for group in groups])
    ratios = lambda values: reference.ratios(values, first_warm, second_warm)
    return {
        "coverage": coverage(test),
        "bootstrap": {"draws": draws, "seed": seed, "tail": tail},
        "first_warm_cost_per_candidate": first_warm, "second_warm_cost_per_candidate": second_warm,
        "risk_and_cost": reference.paired_intervals(ratios(matrix.sum(axis=0)),
                                                   [ratios(values) for values in weights @ matrix], tail),
    }


def policy_actions(rows, policies):
    """Expand Round 2 policy mixtures into auditable direct/consulted action masses."""
    dist = {}
    fallback = ((reference.DIRECT[reference.GOLD],), (1,))
    for row in rows:
        totals = np.zeros(6)
        for policy in policies:
            options, weights = policy.get(reference.cheap_state(row), fallback)
            for tree, weight in zip(options, weights, strict=True):
                if len(tree) == 1:
                    index = ACTION_INDEX[tree[0]]
                else:
                    strong = row["strong"]
                    decision = reference.GOLD if strong is None else tree[strong]
                    index = 3 + ACTION_INDEX[decision]
                totals[index] += weight / len(policies)
        dist[row["id"]] = totals
    return dist


def compare_round2_binary(fit, test, bootstrap):
    """Evaluate the explicitly configured candidate policy suite with a sealed split."""
    if fit.train.binding != test.binding or fit.train.question_ids & test.question_ids:
        raise ValueError("Train and locked test provenance or question isolation differs")
    if test.name != "test":
        raise ValueError("Pass only the locked test split")
    costs = fit.cost_units
    actions = {arm: policy_actions(test.rows, fit.policies[arm]) for arm in ARMS}
    plans = {}
    for arm, (strong, gold) in fit.plan.items():
        plans[arm] = {"strong_queries": strong, "gold_reviews": gold,
                      "unresolved_gold_reviews": fit.unknown_gold_by_arm[arm],
                      "amortized": (strong * costs[0] + gold * costs[1]) / fit.amortization_candidates}
    comparisons = {}
    for old, new in PAIRS:
        for view in ("deployment_only", "amortized"):
            first, second = (0.0, 0.0) if view == "deployment_only" else (plans[old]["amortized"], plans[new]["amortized"])
            comparisons[f"{new}_vs_{old}_{view}"] = compare_actions(
                test, actions[old], actions[new], costs=costs, first_warm=first, second_warm=second, bootstrap=bootstrap,
            )
    return {"development_coverage": coverage(fit.train), "warm_up": plans, "comparisons": comparisons}


def assess_equal_risk(comparison, train_coverage, criteria):
    """Apply separately declared risk limits, without selecting or searching thresholds."""
    intervals = comparison["risk_and_cost"]
    draws = comparison["bootstrap"]["draws"]
    test_coverage = comparison["coverage"]
    observed = test_coverage["gold"]
    unknown = observed.get("unknown", 0)
    minimum_coverage = criteria["min_gold_parse_coverage"]
    ready = (
        min(1 - unknown / test_coverage["candidates"],
            1 - train_coverage["gold"].get("unknown", 0) / train_coverage["candidates"]) >= minimum_coverage
        and min(observed.get("correct", 0), observed.get("wrong", 0)) >= criteria["min_test_each_truth"]
    )
    absolute = criteria["max_absolute_risk"]
    increase = criteria["max_paired_risk_increase"]
    required = {key: limit for risk in ("fa", "fr")
                for key, limit in ((f"old_{risk}_upper", absolute), (f"new_{risk}_upper", absolute),
                                   (f"{risk}_increase_upper", increase))}
    risks = {name: reference.classify(intervals[name], target, False, ready, draws, allow_opposite=unknown == 0)
             for name, target in required.items()}
    saving = reference.classify(intervals["relative_saving"], criteria["min_cost_saving"], True, ready, draws)
    return {"coverage_sufficient": ready, "risk_status": risks, "cost_saving_status": saving,
            "equal_risk_supported": all(state == "supported" for state in risks.values()),
            "equal_risk_saving_supported": saving == "supported" and all(state == "supported" for state in risks.values())}
