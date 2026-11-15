"""[codex] CPU-only, group-held-out analyses of the exposed development cache.

All strategy searches are exploratory. The held-out estimate conditions on the
five fitted selectors; it is not an independent confirmation or a selection-
adjusted confidence interval. No sealed-panel path is ever needed here.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo

import numpy as np


HERE = Path(__file__).resolve().parent
LEGACY_PATH = HERE / "analyze.py"
if not LEGACY_PATH.exists():
    LEGACY_PATH = HERE.parents[1] / "topics/compositional-retrieval/code/dev-3090/analyze.py"
_spec = spec_from_file_location("retrieval_legacy_analysis", LEGACY_PATH)
A = module_from_spec(_spec)
_spec.loader.exec_module(A)

K = 50
FOLDS = 5
FOLD_SEED = 2026111663
BOOT_SEED = 2026111662
DRAWS = 10_000
RANDOM_SELECT_SEEDS = tuple(range(2026111670, 2026111680))
RANDOM_ALLOCATION_SEEDS = tuple(range(2026111700, 2026111764))
SHUFFLE_SEED = 2026111664
REFERENCE_BUDGETS = (25, 50, 75, 100, 150, 200)
FORBIDDEN = "retrieval-independent-panel"
METRIC_KEYS = ("ap_pp", "recall_at_50_pp", "admission_recall_pp")


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def safe_path(path):
    path = Path(path).expanduser().resolve()
    if any(FORBIDDEN in part for part in path.parts):
        raise ValueError("The sealed independent panel is forbidden in development analysis")
    return path


def fold_assignment(groups):
    return np.array([int(hashlib.sha256(canonical([FOLD_SEED, int(group)])).hexdigest(), 16) % FOLDS
                     for group in groups], dtype=int)


def connected_query_groups(queries):
    """Join sentinel groups and every query sharing a relevant candidate image."""
    original = [int(query["group"]) for query in queries]
    parent = {group: group for group in original}

    def find(group):
        while parent[group] != group:
            parent[group] = parent[parent[group]]
            group = parent[group]
        return group

    owner = {}
    for query, group in zip(queries, original):
        for image in query["relevant"]:
            if image in owner:
                left, right = find(group), find(owner[image])
                parent[max(left, right)] = min(left, right)
            else:
                owner[image] = group
    return np.asarray([find(group) for group in original], dtype=int)


def positive_fold_overlap(queries, folds):
    """Audit metadata only: no relevant image may cross an OOF train/test split."""
    image_folds = {}
    for query, fold in zip(queries, folds, strict=True):
        for image in query["relevant"]:
            image_folds.setdefault(image, set()).add(int(fold))
    shared = {image for image, assigned in image_folds.items() if len(assigned) > 1}
    return {"images_in_multiple_folds": len(shared),
            "affected_queries": sum(bool(set(query["relevant"]) & shared) for query in queries),
            "image_ids": sorted(shared)}


def shuffled_conditions(index, seed=SHUFFLE_SEED):
    """Whole-query derangement, rather than an ineffective swap of AND operands."""
    index = np.asarray(index)
    if len(index) < 2:
        raise ValueError("At least two queries are required for the shuffled-condition control")
    order = np.random.default_rng(seed).permutation(len(index))
    donor = np.empty(len(index), dtype=int)
    donor[order] = np.roll(order, 1)
    identical = np.all(np.sort(index[donor], axis=1) == np.sort(index, axis=1), axis=1)
    return index[donor], {"seed": seed, "donor_query_indices": donor.tolist(),
                          "self_donors": int((donor == np.arange(len(index))).sum()),
                          "same_condition_pair_queries": int(identical.sum())}


def corrected_policy_name(name):
    """Preserve frozen scores while correcting their probability terminology."""
    return name.replace("llr_", "logsigmoid_")


def uncalibrated_score_details(name):
    if "llr_" not in name:
        return {}
    return {"score_interpretation": "Sum of log-sigmoid-transformed pool z-scores (or rank fusion thereof); "
                                    "uncalibrated product heuristic, not a log-likelihood ratio"}


def measure(order, relevance, budgets):
    budgets = np.broadcast_to(np.asarray(budgets, dtype=int), (len(order),)).copy()
    if np.any(budgets <= 0) or np.any(budgets > order.shape[1]):
        raise ValueError("Reranking budgets must fit inside the candidate pool")
    labels = np.take_along_axis(relevance, order, axis=1)
    positives = relevance.sum(axis=1)
    admitted = np.array([labels[i, :budget].sum() for i, budget in enumerate(budgets)])
    hits50 = labels[:, :K].sum(axis=1)
    legacy = A.measure(order, relevance, budgets)
    return {"ap_pp": 100 * A.average_precision(labels),
            "recall_at_50_pp": 100 * hits50 / positives,
            "admission_recall_pp": 100 * admitted / positives,
            "admission_oracle_ap_pp": legacy["oracle"],
            "hits_at_50": hits50, "admitted_hits": admitted, "budget": budgets}


def summary(metrics):
    return {"map_pp": float(np.mean(metrics["ap_pp"])),
            "recall_at_50_pp": float(np.mean(metrics["recall_at_50_pp"])),
            "admission_recall_pp": float(np.mean(metrics["admission_recall_pp"])),
            "admission_oracle_map_pp": float(np.mean(metrics["admission_oracle_ap_pp"])),
            "queries_without_top50_hit": int(np.sum(metrics["hits_at_50"] == 0)),
            "queries_without_admitted_hit": int(np.sum(metrics["admitted_hits"] == 0)),
            "mean_pairs_per_query": float(np.mean(metrics["budget"])),
            "total_cross_pairs": int(np.sum(metrics["budget"])),
            "budget_counts": {str(b): int(np.sum(metrics["budget"] == b)) for b in np.unique(metrics["budget"])}}


def bootstrap_plan(groups, *, draws=DRAWS, seed=BOOT_SEED):
    _, codes = np.unique(np.asarray(groups), return_inverse=True)
    counts = np.bincount(codes)
    picks = np.random.default_rng(seed).integers(0, len(counts), (draws, len(counts)))
    return codes, picks, counts[picks].sum(axis=1)


def paired(base, alternate, plan):
    codes, picks, denominator = plan
    result = {}
    for key in METRIC_KEYS:
        delta = alternate[key] - base[key]
        totals = np.bincount(codes, weights=delta)
        samples = totals[picks].sum(axis=1) / denominator
        low, high = np.quantile(samples, [0.025, 0.975])
        se = float(np.std(samples, ddof=1))
        result[key] = {"point_pp": float(np.mean(delta)), "ci95_pp": [float(low), float(high)],
                       "bootstrap_se_pp": se, "ci_half_width_pp": float((high - low) / 2),
                       "normal_approx_mde80_pp": float((1.959963984540054 + 0.8416212335729143) * se)}
    return result


def validate_cache(cache, panel):
    ids, queries = panel["ids"], panel["queries"]
    core, cross = np.asarray(cache["core_order"]), np.asarray(cache["cross"])
    if len(ids) != len(set(ids)) or core.shape != (len(queries), len(ids)):
        raise ValueError("Candidate/query alignment is malformed")
    if not np.issubdtype(core.dtype, np.integer) or not np.all(np.sort(core, axis=1) == np.arange(len(ids))):
        raise ValueError("Each CORE order must be a complete candidate permutation")
    if cross.ndim != 2 or cross.shape[0] != len(queries) or not max(REFERENCE_BUDGETS) <= cross.shape[1] <= len(ids):
        raise ValueError("Analysis v2 requires cross scores through rank 200")
    if not np.isfinite(cross).all():
        raise ValueError("Cross scores must be finite")
    for name in ("core_embed", "cs_clip"):
        image, full, cond = (np.asarray(cache[f"{name}_{part}"]) for part in ("image", "full", "condition"))
        if image.ndim != 2 or full.shape != (len(queries), image.shape[1]) or len(image) != len(ids):
            raise ValueError("Image/full-query vectors are not aligned")
        if cond.ndim != 2 or cond.shape[1] != image.shape[1] or not all(np.isfinite(x).all() for x in (image, full, cond)):
            raise ValueError("Condition vectors must be aligned and finite")
    index = np.asarray(panel["condition_index"])
    if index.ndim != 2 or len(index) != len(queries) or index.shape[1] != 2 or np.any(index < 0):
        raise ValueError("This diagnostic panel requires two valid condition indices per query")
    if any(np.any(index >= len(cache[f"{name}_condition"])) for name in ("core_embed", "cs_clip")):
        raise ValueError("Condition index outside vector table")
    inverse = {key: col for col, key in enumerate(ids)}
    relevance = np.zeros(core.shape, dtype=bool)
    for row, query in enumerate(queries):
        if not query["relevant"] or not set(query["relevant"]) <= set(ids):
            raise ValueError("Every query must have valid in-pool positives")
        relevance[row, [inverse[key] for key in query["relevant"]]] = True
    return core.astype(np.int64), cross, relevance


def require_replay(base, meta, published=None):
    expected = A.PUBLISHED if published is None else published
    actual = {"map_pp": float(np.mean(base["ap_pp"])),
              "oracle_map_pp": float(np.mean(base["admission_oracle_ap_pp"])),
              "recall_pp": float(np.mean(base["recall_at_50_pp"])),
              "queries_without_hit": int(np.sum(base["hits_at_50"] == 0))}
    if meta.get("replay", {}).get("state") != "PASS":
        raise ValueError("Cache baseline replay did not PASS; stop before policy analysis")
    if any(abs(actual[key] - value) > (0 if key == "queries_without_hit" else A.TOLERANCE_PP)
           for key, value in expected.items()):
        raise ValueError("Recomputed baseline did not match the frozen replay; stop before policy analysis")
    return actual


def validate_logic_vectors(cache, panel, meta, calibration, logic_vectors, logic):
    """Check the four-template mean and bind its rows to the exact cache/source.

    These checks establish artifact consistency; the producing committed runner
    remains responsible for encoding the stated texts with the stated weights.
    """
    required = {"metadata_json", "condition_texts"}
    for name in ("core_embed", "cs_clip"):
        required.update({f"{name}_lcse_condition", f"{name}_template_vectors"})
    if not required <= set(logic_vectors):
        raise ValueError("Four-template LCSE vectors need complete provenance and per-template vectors")
    encoded = np.asarray(logic_vectors["metadata_json"])
    if encoded.shape != () or encoded.dtype.kind != "U":
        raise ValueError("LCSE metadata_json must be one Unicode JSON string")
    try:
        provenance = json.loads(encoded.item())
    except (TypeError, ValueError) as error:
        raise ValueError("Invalid LCSE metadata JSON") from error
    if (not isinstance(provenance, dict) or provenance.get("schema") != 1
            or provenance.get("scope") != "lcse-four-template-condition-vectors"):
        raise ValueError("Unsupported LCSE vector provenance schema/scope")
    texts = np.asarray(logic_vectors["condition_texts"])
    if texts.ndim != 1 or texts.dtype.kind != "U" or texts.tolist() != panel.get("condition_texts"):
        raise ValueError("LCSE condition text rows differ from the frozen condition table")
    if provenance.get("templates") != list(logic.CONCEPT_TEMPLATES):
        raise ValueError("LCSE template protocol differs from the author's four templates")
    source = calibration.get("source", {})
    matches = {
        "calibration_source_manifest_sha256": source.get("manifest_sha256"),
        "calibration_design_sha256": source.get("calibration_design_sha256"),
        "eval_manifest_sha256": meta.get("panel_sha256"),
    }
    for key, expected in matches.items():
        if (not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected)
                or provenance.get(key) != expected):
            raise ValueError(f"LCSE provenance mismatch: {key}")
    revisions = meta.get("revisions")
    if (not isinstance(revisions, dict) or not {"core_embed", "cs_clip", "core_reranker"} <= set(revisions)
            or any(not re.fullmatch(r"[0-9a-f]{40}", str(value)) for value in revisions.values())
            or provenance.get("model_revisions") != revisions):
        raise ValueError("LCSE model revisions differ from the cache")
    for key in ("source", "freeze"):
        expected = meta.get("bundle", {}).get(key)
        if (not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{40}", expected)
                or provenance.get(key) != expected):
            raise ValueError(f"LCSE frozen code provenance mismatch: {key}")
    # Supplement encoding may be a later commit than the main cache. Record its
    # own committed source without demanding the two different runners be equal.
    if not re.fullmatch(r"[0-9a-f]{40}", str(provenance.get("bundle_commit", ""))):
        raise ValueError("LCSE supplement must record its own committed source")
    for name in ("core_embed", "cs_clip"):
        mean = np.asarray(logic_vectors[f"{name}_lcse_condition"])
        templates = np.asarray(logic_vectors[f"{name}_template_vectors"])
        expected = cache[f"{name}_condition"].shape
        if mean.shape != expected or expected[0] != len(texts) or not np.isfinite(mean).all():
            raise ValueError("Four-template LCSE means are not aligned with the cache")
        if templates.shape == (expected[0] * 4, expected[1]):
            templates = templates.reshape(expected[0], 4, expected[1])
        if templates.shape != (expected[0], 4, expected[1]) or not np.isfinite(templates).all():
            raise ValueError("LCSE requires four finite template vectors per condition")
        if not np.allclose(np.linalg.norm(templates, axis=-1), 1.0, atol=.005, rtol=0):
            raise ValueError("Each LCSE template embedding must be a unit vector")
        if not np.allclose(mean, templates.mean(axis=1), atol=1e-7, rtol=1e-6):
            raise ValueError("LCSE condition vectors must equal the template mean without renormalization")
    return provenance


def features(cache, index):
    result = {}
    for name in ("core_embed", "cs_clip"):
        image = cache[f"{name}_image"]
        full = cache[f"{name}_full"] @ image.T
        conditions = cache[f"{name}_condition"] @ image.T
        sd_full = full.std(axis=1, keepdims=True)
        sd_cond = conditions.std(axis=1, keepdims=True)
        if np.any(sd_full <= 0) or np.any(sd_cond <= 0):
            raise ValueError("Zero-variance scores cannot be standardized")
        result[name] = {"full_raw": full, "condition_raw": conditions[index].transpose(0, 2, 1),
                        "full_z": (full - full.mean(axis=1, keepdims=True)) / sd_full,
                        "condition_z": ((conditions - conditions.mean(axis=1, keepdims=True)) / sd_cond)[index].transpose(0, 2, 1)}
    return result


def shortlist_features(full, core, depth, *, raw=False):
    top = core[:, :depth]
    suffix = "raw" if raw else "z"
    return {f"{name}_full": np.take_along_axis(value[f"full_{suffix}"], top, axis=1)
            for name, value in full.items()} | {
        f"{name}_cond": np.take_along_axis(value[f"condition_{suffix}"], top[:, :, None], axis=1)
        for name, value in full.items()}


def oof_selection(methods, eligible, groups, folds):
    """Fit one selector per training split; held-out labels never choose it."""
    if set(np.unique(folds)) != set(range(FOLDS)):
        raise ValueError("All five fixed group folds must be represented")
    names = sorted(eligible)
    if not names:
        raise ValueError("At least one eligible policy is required")
    first = methods[names[0]]
    result = {key: np.empty_like(value) for key, value in first.items()}
    records, selected = [], np.empty(len(groups), dtype=object)
    for fold in range(FOLDS):
        held = folds == fold
        train = ~held
        # Freeze before outcomes: train mean AP, then fixed Recall@50, then lexical id.
        winner = min(names, key=lambda name: (-float(methods[name]["ap_pp"][train].mean()),
                                              -float(methods[name]["recall_at_50_pp"][train].mean()), name))
        if set(np.asarray(groups)[held]) & set(np.asarray(groups)[train]):
            raise AssertionError("A query group crossed the held-out boundary")
        for key in result:
            result[key][held] = methods[winner][key][held]
        selected[held] = winner
        records.append({"fold": fold, "selected_policy": winner, "training_queries": int(train.sum()),
                        "heldout_queries": int(held.sum()),
                        "training_map_pp": float(methods[winner]["ap_pp"][train].mean()),
                        "heldout_map_pp": float(methods[winner]["ap_pp"][held].mean()),
                        "heldout_recall_at_50_pp": float(methods[winner]["recall_at_50_pp"][held].mean())})
    return result, records, selected.tolist()


def uncertainty_scores(scores, core):
    ranked = np.take_along_axis(scores, core[:, :200], axis=1)
    result = {"margin50_200": -(ranked[:, 49] - ranked[:, 199]),
              "margin1_50": -(ranked[:, 0] - ranked[:, 49])}
    for temperature in (1.0, 0.1, 0.01):
        logits = (ranked - ranked[:, :1]) / temperature
        prob = np.exp(logits)
        prob /= prob.sum(axis=1, keepdims=True)
        result[f"entropy_top200_t{temperature:g}"] = -(prob * np.log(np.maximum(prob, np.finfo(float).tiny))).sum(axis=1)
    return result


def exact_batch_budgets(uncertainty):
    if len(uncertainty) % 4:
        raise ValueError("Exact 75/50/50/25 allocation requires a query count divisible by four")
    return A.quartile_budgets(uncertainty)


def describe_strata(base, alternate, positives, groups, *, draws):
    strata = {"one_positive": positives == 1, "two_or_more": positives >= 2,
              "two_or_three": (positives >= 2) & (positives <= 3), "four_or_more": positives >= 4}
    output = {}
    for name, selected in strata.items():
        if not selected.any():
            output[name] = {"queries": 0}
            continue
        sub_base = {key: value[selected] for key, value in base.items()}
        sub_alt = {key: value[selected] for key, value in alternate.items()}
        output[name] = {"queries": int(selected.sum()), "groups": len(set(np.asarray(groups)[selected])),
                        "summary": summary(sub_alt),
                        "paired_against_core50": paired(sub_base, sub_alt, bootstrap_plan(np.asarray(groups)[selected], draws=draws))}
    return output


def analyse(cache, panel, meta, *, published=None, draws=DRAWS, logic=None, calibration=None, logic_vectors=None,
            evaluation_context=None):
    core, cross, relevance = validate_cache(cache, panel)
    groups = np.asarray([q["group"] for q in panel["queries"]])
    connected_groups = connected_query_groups(panel["queries"])
    original_folds, folds = fold_assignment(groups), fold_assignment(connected_groups)
    original_overlap = positive_fold_overlap(panel["queries"], original_folds)
    connected_overlap = positive_fold_overlap(panel["queries"], folds)
    if connected_overlap["images_in_multiple_folds"]:
        raise AssertionError("A relevant candidate image crossed connected OOF folds")
    positives = relevance.sum(axis=1)
    base_order = A.final_order(core, core[:, :K], cross[:, :K])
    base = measure(base_order, relevance, K)
    if evaluation_context is None:
        replay = require_replay(base, meta, published)
        evaluation = {"mode": "published-replay", "published_reproduction": "PASS", "formal_evidence": False}
    else:
        if published is not None:
            raise ValueError("A platform study cannot replace the published reference metrics")
        from platform_context import require_baseline
        actual = {"map_pp": float(np.mean(base["ap_pp"])),
                  "oracle_map_pp": float(np.mean(base["admission_oracle_ap_pp"])),
                  # Preserve the producer's aggregate arithmetic for exact receipt equality.
                  "recall_pp": float(100 * np.mean(base["hits_at_50"] / positives)),
                  "queries_without_hit": int(np.sum(base["hits_at_50"] == 0))}
        replay = require_baseline(actual, meta, evaluation_context)
        evaluation = dict(evaluation_context)
    logic_parameters = {}
    if logic is not None:
        if calibration is None or logic_vectors is None:
            raise ValueError("LCSE requires independent calibration and four-template concept vectors")
        logic_parameters = {name: logic.validate_calibration_artifact(calibration, name, excluded_image_ids=panel["ids"])
                            for name in ("core_embed", "cs_clip")}
        validate_logic_vectors(cache, panel, meta, calibration, logic_vectors, logic)
    plan = bootstrap_plan(groups, draws=draws)
    methods, descriptions = {"core50": base}, {"core50": {"family": "baseline", "budget": 50}}
    eligible = {"core50"}
    full = features(cache, np.asarray(panel["condition_index"]))
    shuffled_index, shuffle_info = shuffled_conditions(panel["condition_index"])
    shuffled = features(cache, shuffled_index)

    def evaluate(name, prefix, budgets, *, family, **details):
        top = core[:, :cross.shape[1]]
        score = A.cross_for(prefix, top, cross)
        order = A.final_order(core, prefix, score)
        methods[name] = measure(order, relevance, budgets)
        descriptions[name] = {"family": family, **details}

    for budget in REFERENCE_BUDGETS:
        evaluate(f"reference/rerank{budget}", core[:, :budget], budget,
                 family="fixed_budget_reference", budget=budget,
                 interpretation="Budget reference; reranking all candidates is not a mathematical ceiling")
    for depth in (100, 200):
        top = core[:, :depth]
        feats = shortlist_features(full, core, depth)
        original_policies = A.policies(feats)
        policies = {corrected_policy_name(name): value for name, value in original_policies.items()}
        policy_details = {corrected_policy_name(name): uncalibrated_score_details(name) for name in original_policies}
        shuffled_logic = {}
        if logic is not None:
            raw = shortlist_features(full, core, depth, raw=True)
            for name in ("core_embed", "cs_clip"):
                cosines = logic_vectors[f"{name}_lcse_condition"] @ cache[f"{name}_image"].T
                condition_cosines = cosines[np.asarray(panel["condition_index"])].transpose(0, 2, 1)
                condition_cosines = np.take_along_axis(condition_cosines, top[:, :, None], axis=1)
                shuffled_cosines = np.take_along_axis(cosines[shuffled_index].transpose(0, 2, 1), top[:, :, None], axis=1)
                for variant, options in (("paperparams_transfer", {"mu": .22, "beta": 30., "gamma": -1.}),
                                         ("train_presence_calibrated", logic_parameters[name])):
                    extra = logic.aggregate_scores(raw[f"{name}_full"], condition_cosines, **options)
                    if any(np.asarray(value).shape != top.shape or not np.isfinite(value).all() for value in extra.values()):
                        raise ValueError("LCSE/logic admission scores must be finite and aligned with candidates")
                    for key, value in extra.items():
                        policy_name = f"logic_{name}_{variant}_{key}"
                        policies[policy_name] = value
                        policy_details[policy_name] = {"concept_vectors": "four-template mean, no renormalization",
                            "logic_variant": variant, "logic_parameters": options,
                            "interpretation": "Fixed paper parameters transferred to this backbone" if variant == "paperparams_transfer"
                                              else "Independent train-presence NLL calibration adaptation, not original multi-operator calibration"}
                    shuffled_logic[f"logic_{name}_{variant}_lcse_and"] = logic.lcse_and(
                        raw[f"{name}_full"], shuffled_cosines, **options)
        for name, score in policies.items():
            key = f"m1/top{depth}/{name}"
            evaluate(key, A.admit(top, score), K, family="m1_policy", candidate_depth=depth, **policy_details.get(name, {}))
            eligible.add(key)
        for name, score in shuffled_logic.items():
            evaluate(f"control/top{depth}/shuffled_{name}", A.admit(top, score), K,
                     family="shuffled_condition_control", candidate_depth=depth, shuffle_seed=SHUFFLE_SEED)
        shuffled_policies = A.policies(shortlist_features(shuffled, core, depth))
        for name in ("min_cond_core", "min_cond_cs", "llr_core_t1", "llr_cs_t1"):
            evaluate(f"control/top{depth}/shuffled_{corrected_policy_name(name)}", A.admit(top, shuffled_policies[name]), K,
                     family="shuffled_condition_control", candidate_depth=depth, shuffle_seed=SHUFFLE_SEED,
                     **uncalibrated_score_details(name))
        for seed in RANDOM_SELECT_SEEDS:
            random_score = np.random.default_rng(seed).random(top.shape)
            evaluate(f"control/top{depth}/random50_seed{seed}", A.admit(top, random_score), K,
                     family="random_selection_control", candidate_depth=depth, seed=seed)

    for name, uncertainty in uncertainty_scores(full["core_embed"]["full_raw"], core).items():
        budgets = exact_batch_budgets(uncertainty)
        methods[f"m2/{name}"] = measure(A.budgeted(core, cross, budgets), relevance, budgets)
        descriptions[f"m2/{name}"] = {"family": "m2_batch_allocation", "uncertainty": name,
                                        "levels": [75, 50, 50, 25], "requires_unlabelled_query_batch": True}
    random_budgets = np.repeat([75, 50, 50, 25], len(groups) // 4)
    for seed in RANDOM_ALLOCATION_SEEDS:
        budgets = np.random.default_rng(seed).permutation(random_budgets)
        key = f"control/m2/random_allocation_seed{seed}"
        methods[key] = measure(A.budgeted(core, cross, budgets), relevance, budgets)
        descriptions[key] = {"family": "random_allocation_control", "seed": seed,
                             "requires_unlabelled_query_batch": True}

    oof, fold_records, chosen = oof_selection(methods, eligible, connected_groups, folds)
    methods["oof/m1_single_selector"] = oof
    descriptions["oof/m1_single_selector"] = {"family": "group_heldout_single_selector",
        "selection": "training mean AP descending; fixed Recall@50 descending; lexical policy id ascending",
        "eligible_policies": sorted(eligible), "fold_seed": FOLD_SEED,
        "grouping": "connected components of original sentinel groups and shared relevant image IDs",
        "ci_scope": "Conditional development interval, fixed fitted fold selectors; no selection refitting in bootstrap",
        "folds": fold_records, "selected_policy_by_query": chosen,
        "heldout_fold_map_sd_pp": float(np.std([r["heldout_map_pp"] for r in fold_records], ddof=1))}
    rows = {}
    for name, metrics in methods.items():
        rows[name] = {**descriptions[name], "summary": summary(metrics),
                      "paired_against_core50": paired(base, metrics, plan),
                      "positive_count_strata": describe_strata(base, metrics, positives, groups, draws=draws)}
    rows["oof/m1_single_selector"]["paired_against_core50_connected_groups"] = paired(
        base, oof, bootstrap_plan(connected_groups, draws=draws))
    rows["oof/m1_single_selector"]["connected_group_positive_count_strata"] = describe_strata(
        base, oof, positives, connected_groups, draws=draws)
    rows["oof/m1_single_selector"]["connected_interval_scope"] = (
        "Conditional development bootstrap over connected groups; original sentinel-group interval retained for comparison. "
        "Selectors are fixed, not refitted in bootstrap.")
    random_methods = [metrics for name, metrics in methods.items() if name.startswith("control/m2/")]
    random_mean = {key: np.mean([metrics[key] for metrics in random_methods], axis=0) for key in METRIC_KEYS}
    for name in methods:
        if name.startswith("m2/"):
            rows[name]["paired_against_random_allocation_expectation"] = paired(random_mean, methods[name], plan)
            rows[name]["random_allocation_comparison_scope"] = (
                f"Monte Carlo expectation over {len(random_methods)} fixed permutations; descriptive, not a randomization p-value")
    ranked = sorted((name for name in methods if name.startswith("m1/")),
                    key=lambda name: (-rows[name]["summary"]["map_pp"], name))
    random_controls = [rows[name]["paired_against_core50"]["ap_pp"]["point_pp"]
                       for name in methods if name.startswith("control/m2/")]
    kind = A.kinds(panel["image_categories"], panel["ids"], panel["queries"])
    report = {"schema": 2, "scope": ("local-relative platform exploration; published reproduction MISMATCH; no independent confirmation"
              if evaluation_context is not None else "exposed development panel; no independent confirmation"),
        "evaluation_context": evaluation, "formal_evidence": False,
        "baseline_replay": replay, "queries": len(groups), "groups": len(set(groups)), "pool": len(panel["ids"]),
        "fold_assignment": folds.tolist(), "fold_seed": FOLD_SEED,
        "original_sentinel_fold_assignment": original_folds.tolist(),
        "oof_grouping": {"original_sentinel_groups": len(set(groups)),
            "connected_groups": len(set(connected_groups)),
            "component_id": "minimum original sentinel group ID in each connected component",
            "original_positive_crossfold_overlap": original_overlap,
            "connected_positive_crossfold_overlap": connected_overlap,
            "fold_query_counts": {str(fold): int((folds == fold).sum()) for fold in range(FOLDS)},
            "scope": "OOF selection and extra OOF intervals use connected groups; legacy paired intervals keep original sentinel groups"},
        "methods": rows, "exploratory_policy_ranking": ranked, "shuffled_condition_control": shuffle_info,
        "d1_recall": A.d1_recall(core, relevance, kind, full["core_embed"]["condition_z"].transpose(0, 2, 1)),
        "d2_ranking": A.d2_ranking(base_order, relevance, kind), "d3_implemented": False,
        "random_allocation_map_gain_distribution_pp": {"seeds": list(RANDOM_ALLOCATION_SEEDS),
            "mean": float(np.mean(random_controls)), "quantiles_025_50_975": np.quantile(random_controls, [.025, .5, .975]).tolist()},
        "bootstrap": {"draws": draws, "seed": BOOT_SEED, "unit": "original sentinel query group",
            "scope": "Paired descriptive intervals; exploratory grid is not multiplicity-adjusted",
            "precision": "MDE80 is normal approximation (1.96+0.84)*observed paired bootstrap SE; planning estimate, not a power guarantee"},
        "cost_scope": "Cross-pair counts only. Same calls can have different token/pixel padding, batch packing and GPU time; total indexing, storage and latency need separate measurement.",
        "m2_scope": "Quartiles use the entire unlabelled query batch, not online independent queries. Online use requires train-frozen thresholds or a budget controller; no claim of identical runtime from identical calls.",
        "limits": ["All M1 policies target the exposed panel; OOF does not undo earlier human exposure.",
                   "Original sentinel-group intervals may retain shared-positive dependence; connected OOF groups remove that image overlap but not every category dependence.",
                   "M1 OOF selects only exact 50-call policies; M2 remains a separate exploratory batch allocation.",
                   "Fixed-budget reranking is a cost reference, never an oracle or ceiling."]}
    per_query = {"schema": 2, "evaluation_context": evaluation, "formal_evidence": False,
                 "queries": [{"index": i, "key": q["key"], "text": q["text"],
                  "group": int(q["group"]), "connected_group": int(connected_groups[i]),
                  "fold": int(folds[i]), "positives": int(positives[i])}
                  for i, q in enumerate(panel["queries"])],
                 "methods": {name: {key: value.tolist() for key, value in metrics.items()} for name, metrics in methods.items()}}
    return report, per_query


def markdown(report):
    evaluation = report.get("evaluation_context", {})
    replay_text = ("Published reproduction remains MISMATCH. Local-relative platform exploration only; not formal evidence."
                   if evaluation.get("mode") == "platform-relative-exploration" else "Baseline replay passed.")
    lines = ["# Retrieval development analysis v2", "", "[codex] Exposed panel; development estimates only.", "",
             replay_text + " Fixed Recall@50 and admission Recall@budget are reported separately.", "",
             "| method | mAP | ΔmAP [95%] | Recall@50 | admission recall | calls/query |",
             "|---|---:|---:|---:|---:|---:|"]
    names = ["core50", "oof/m1_single_selector"] + [f"reference/rerank{b}" for b in REFERENCE_BUDGETS]
    names += report["exploratory_policy_ranking"][:12]
    names += [name for name in report["methods"] if name.startswith("m2/")]
    for name in names:
        row = report["methods"][name]
        s, d = row["summary"], row["paired_against_core50"]["ap_pp"]
        lo, hi = d["ci95_pp"]
        lines.append(f"| {name} | {s['map_pp']:.2f} | {d['point_pp']:+.2f} [{lo:+.2f}, {hi:+.2f}] | "
                     f"{s['recall_at_50_pp']:.2f} | {s['admission_recall_pp']:.2f} | {s['mean_pairs_per_query']:.1f} |")
    connected = report["methods"]["oof/m1_single_selector"]["paired_against_core50_connected_groups"]["ap_pp"]
    lo, hi = connected["ci95_pp"]
    grouping = report["oof_grouping"]
    lines += ["", "OOF policies are chosen on four connected-group folds and evaluated on the fifth. Their intervals condition on these fitted selectors.",
              f"OOF ΔmAP with connected-group bootstrap: {connected['point_pp']:+.2f} [{lo:+.2f}, {hi:+.2f}]. "
              f"Groups: {grouping['original_sentinel_groups']} original, {grouping['connected_groups']} connected; "
              f"cross-fold shared positive images after joining: {grouping['connected_positive_crossfold_overlap']['images_in_multiple_folds']}. "
              "The table retains the original sentinel-group intervals for comparability.",
              "The complete exploratory ranking and all per-query metrics are in the JSON files.", "", report["m2_scope"], "", report["cost_scope"], "",
              "D3 is not implemented in this cache: condition-count experiments need a separately frozen panel."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lcse-calibration", type=Path)
    parser.add_argument("--lcse-vectors", type=Path)
    parser.add_argument("--platform-plan", type=Path)
    parser.add_argument("--platform-receipt", type=Path)
    args = parser.parse_args()
    if bool(args.lcse_calibration) != bool(args.lcse_vectors):
        parser.error("--lcse-calibration and --lcse-vectors must be supplied together")
    if bool(args.platform_plan) != bool(args.platform_receipt):
        parser.error("--platform-plan and --platform-receipt must be supplied together")
    cache_dir, out = safe_path(args.cache), safe_path(args.out)
    if out.exists():
        raise FileExistsError(out)
    cache, panel = A.load(cache_dir)
    meta = json.loads((cache_dir / "meta.json").read_text())
    evaluation_context = None
    if args.platform_plan:
        from platform_context import validate_context
        supplementary = (args.lcse_calibration, args.lcse_vectors) if args.lcse_calibration else ()
        evaluation_context = validate_context(cache_dir, args.platform_plan, args.platform_receipt,
                                              supplementary_paths=supplementary)
    calibration, logic, logic_vectors = None, None, None
    if args.lcse_calibration:
        path = safe_path(args.lcse_calibration)
        vectors_path = safe_path(args.lcse_vectors)
        calibration = json.loads(path.read_text())
        logic_vectors = dict(np.load(vectors_path, allow_pickle=False))
        import logic_baselines as logic
    report, per_query = analyse(cache, panel, meta, logic=logic, calibration=calibration, logic_vectors=logic_vectors,
                                evaluation_context=evaluation_context)
    if calibration is not None:
        report["lcse_calibration"] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "artifact": calibration}
        report["lcse_vectors"] = {"path": str(vectors_path), "sha256": hashlib.sha256(vectors_path.read_bytes()).hexdigest(),
                                   "aggregation": "mean of four unit template vectors; no renormalization",
                                   "provenance": json.loads(np.asarray(logic_vectors["metadata_json"]).item())}
    report["cache_meta"] = meta
    report["generated_at_taipei"] = datetime.now(ZoneInfo("Asia/Taipei")).isoformat()
    out.mkdir(parents=True, exist_ok=False)
    for name, value in (("report.json", report), ("per_query.json", per_query)):
        (out / name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (out / "report.md").write_text(markdown(report))
    print(out / "report.md")


if __name__ == "__main__":
    main()
