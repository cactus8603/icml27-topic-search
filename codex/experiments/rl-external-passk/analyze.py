"""Replay the published DeepScaleR 32k paired counts on CPU."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np


UPSTREAM = "ac7c3fc58d5e325447607e778bdcbc897182cf7e"
CRITERIA_SHA = "213b578bba416db880cdf17af18fbdddf5eac9f8f3fa4ce95116abc3cbf3a81e"
INPUT_SHA = {
    "base.jsonl": "184d3c7446d8f2ff389195909ca408a2b2bb0984ad726e4549c48adee8af1fc4",
    "rl.jsonl": "9da1ffef9e04f951427112a7ee88d3105956952ab30229886a901e2864f5bb45",
}
SOURCE_COUNTS = {"olympiadbench": 633, "gaokao2023en": 358, "amc23": 39, "aime24": 30}
PROMPTS = 1060
DRAWS = 128
BOOTSTRAP = 4000
ALPHA = 0.05
SEED = 0


def checked_bytes(path, expected_sha):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha:
        raise ValueError(f"Input SHA-256 differs: {path.name}")
    return data


def read_counts(data, draws):
    rows = {}
    for raw in data.splitlines():
        if not raw.strip():
            continue
        row = json.loads(raw)
        if not isinstance(row, dict) or set(row) != {"id", "source", "c", "n"}:
            raise ValueError("Expected exactly id, source, c and n")
        prompt, source, success, total = (row[key] for key in ("id", "source", "c", "n"))
        if (type(prompt) not in (int, str) or prompt == "" or prompt in rows
                or not isinstance(source, str) or not source
                or type(total) is not int or type(success) is not int
                or total != draws or not 0 <= success <= total):
            raise ValueError("Invalid or duplicate paired count row")
        rows[prompt] = (source, success)
    return rows


def paired_counts(base, rl, source_counts):
    if base.keys() != rl.keys() or len(base) != sum(source_counts.values()):
        raise ValueError("The arms have different prompt coverage")
    ids = sorted(base, key=lambda key: (isinstance(key, str), key))
    if any(base[key][0] != rl[key][0] for key in ids):
        raise ValueError("The arms disagree on a prompt source")
    if Counter(base[key][0] for key in ids) != source_counts:
        raise ValueError("Prompt source totals differ from the paper metadata")
    return tuple(np.array([arm[key][1] for key in ids], dtype=int) for arm in (base, rl))


def pass_lookup(draws, pass_at_k_vec):
    possible = np.arange(draws + 1)
    return np.column_stack([pass_at_k_vec(possible, draws, k) for k in range(1, draws + 1)])


def simultaneous_band(differences, *, bootstrap=BOOTSTRAP, seed=SEED):
    """The paper's Gaussian prompt-multiplier sup-t band over the full grid."""
    m = len(differences)
    means = differences.mean(axis=0)
    se = differences.std(axis=0, ddof=1) / np.sqrt(m)
    centered = differences - means
    scale = m * np.maximum(se, 1e-12)
    maxima = np.empty(bootstrap)
    rng = np.random.default_rng(seed)
    for start in range(0, bootstrap, 128):
        end = min(start + 128, bootstrap)
        noise = rng.standard_normal((end - start, m))
        maxima[start:end] = np.max(np.abs(noise @ centered) / scale, axis=1)

    critical = float(np.quantile(maxima, 1 - ALPHA))
    lower = np.where(se < 1e-12, -1.0, means - critical * se)
    upper = np.where(se < 1e-12, 1.0, means + critical * se)
    return lower, upper, critical


def certify(lower, upper):
    positive = (np.flatnonzero(lower > 0) + 1).tolist()
    negative = (np.flatnonzero(upper < 0) + 1).tolist()
    first_inconclusive = next((k for k, bound in enumerate(lower, 1) if bound <= 0), len(lower) + 1)
    return {
        "first_loss_interval": [first_inconclusive, negative[0] if negative else None],
        "certified_positive_budgets": positive,
        "certified_negative_budgets": negative,
    }


def verify_public_band(band):
    expected = {
        "first_loss_interval": [11, 61],
        "certified_positive_budgets": list(range(1, 11)),
        "certified_negative_budgets": list(range(61, 129)),
    }
    if band != expected:
        raise ValueError("Dense-grid reproduction differs from the author's published result")


def focal_rates(base, rl, lo, hi, k):
    base_rate, rl_rate = float(base[:, k - 1].mean()), float(rl[:, k - 1].mean())
    return {
        "base_pct": 100 * base_rate,
        "rl_pct": 100 * rl_rate,
        "rl_minus_base_pp": 100 * (rl_rate - base_rate),
        "paired_dense_grid_band_pp": [100 * float(lo[k - 1]), 100 * float(hi[k - 1])],
    }


def gate_reference(endpoints, criteria, k):
    rules = criteria["gate0"]
    low, high = rules["pass_window_pct"]
    required_loss = rules["pair_rl_pk_below_base_min_pp"]
    required_gain = rules["pair_rl_p1_gain_min_pp"]
    truncation_max = rules["trunc_rate_diff_max_pp"]
    truncation = {"base_pct": 7.8, "rl_pct": 0.7, "absolute_diff_pp": 7.1}
    focal = endpoints[str(k)]
    return {
        "scope": "reference_checks_on_external_32k_aggregate_only",
        "rl_pass_at_1_gain_min_pp": required_gain,
        "rl_pass_at_1_gain_meets_reference": endpoints["1"]["rl_minus_base_pp"] >= required_gain,
        "rl_pass_at_k_loss_min_pp": required_loss,
        "rl_pass_at_k_loss_meets_reference": focal["rl_minus_base_pp"] <= -required_loss,
        "pass_at_k_window_pct": [low, high],
        "base_pass_at_k_in_reference_window": low <= focal["base_pct"] <= high,
        "rl_pass_at_k_in_reference_window": low <= focal["rl_pct"] <= high,
        "published_truncation": truncation | {
            "source": "arXiv:2609.22547v3 Table 1, clean 32k DeepScaleR row",
            "exclusive_diff_max_pp": truncation_max,
            "meets_reference": truncation["absolute_diff_pp"] < truncation_max,
        },
        "official_gate0_evaluable": False,
    }


def response_mix(base, rl, lookup, draws, k, step, mix_curve):
    """Finite-sample R2, with a matched-pass@1 endpoint-randomization reference."""
    pis = np.linspace(0.0, 1.0, round(1 / step) + 1)
    p1, pk = mix_curve(rl, base, draws, k, pis)
    base_k, rl_k = lookup[base, k - 1].mean(), lookup[rl, k - 1].mean()
    linear = (1 - pis) * base_k + pis * rl_k
    points = [
        {"pi_rl": float(pi), "pass_at_1_pct": 100 * float(p), "pass_at_k_pct": 100 * float(q),
         "matched_per_prompt_randomized_endpoint_pct": 100 * float(b),
         "delta_vs_matched_endpoint_pp": 100 * float(q - b)}
        for pi, p, q, b in zip(pis, p1, pk, linear, strict=True)
    ]
    return {
        "k": k,
        "matched_pass_at_1_range_pct": [100 * float(p1.min()), 100 * float(p1.max())],
        "points": points,
        "max_pass_at_k_grid_point": max(points, key=lambda row: row["pass_at_k_pct"]),
        "wise_ft_and_sparse_replacement_observed": False,
    }


def hindsight_endpoints(base, rl, lookup, k):
    base_k, rl_k = lookup[base, k - 1], lookup[rl, k - 1]
    choose_rl = rl_k > base_k
    best = max(float(base_k.mean()), float(rl_k.mean()))
    oracle = float(np.maximum(base_k, rl_k).mean())
    return {
        "k": k,
        "best_single_endpoint_pct": 100 * best,
        "empirical_hindsight_endpoint_pass_at_k_pct": 100 * oracle,
        "hindsight_minus_best_single_endpoint_pp": 100 * (oracle - best),
        "hindsight_policy_empirical_pass_at_1_pct": 100 * float(np.where(choose_rl, lookup[rl, 0], lookup[base, 0]).mean()),
        "matched_pass_at_1": False,
        "scope": "same observed counts, per-prompt endpoint choice, ties to base, fixed k",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("check", "run"))
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--job")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.freeze):
        raise ValueError("Expected a full repository freeze commit")
    criteria = json.loads(checked_bytes(args.source / "configs/criteria.json", CRITERIA_SHA))
    blobs = {name: checked_bytes(args.assets / name, sha) for name, sha in INPUT_SHA.items()}
    sys.path.insert(0, str(args.source / "scripts"))
    from aggregate import mix_curve, pass_at_k_vec

    if args.phase == "check":
        print(json.dumps({"asset_bytes": {name: len(data) for name, data in blobs.items()},
                          "upstream_commit": UPSTREAM, "verified_sha256": INPUT_SHA}, sort_keys=True))
        return
    if not re.fullmatch(r"[1-9][0-9]*", args.job or "") or args.output is None:
        raise ValueError("Use the frozen Slurm launcher for replay")
    base, rl = paired_counts(*(read_counts(blobs[name], DRAWS) for name in INPUT_SHA), SOURCE_COUNTS)
    lookup = pass_lookup(DRAWS, pass_at_k_vec)
    lower, upper, critical = simultaneous_band(lookup[rl] - lookup[base])
    band = certify(lower, upper)
    verify_public_band(band)
    endpoints = {str(k): focal_rates(lookup[base], lookup[rl], lower, upper, k) for k in (1, 64, 128)}
    k = criteria["eval"]["k_main"]
    result = {
        "freeze": args.freeze,
        "slurm_job_id": int(args.job),
        "upstream_commit": UPSTREAM,
        "input_sha256": INPUT_SHA,
        "criteria_sha256": CRITERIA_SHA,
        "prompts": PROMPTS,
        "draws_per_arm_and_prompt": DRAWS,
        "source_counts": SOURCE_COUNTS,
        "numpy_version": np.__version__,
        "published_dense_grid_reproduction": band | {"bootstrap": BOOTSTRAP, "seed": SEED,
                                                       "alpha": ALPHA, "critical_value": critical},
        "paired_focal_metrics": endpoints,
        "original_gate_reference_checks": gate_reference(endpoints, criteria, k),
        "response_mix": response_mix(base, rl, lookup, DRAWS, k, criteria["gate1"]["mix_curve"]["pi_step"], mix_curve),
        "hindsight_endpoints": [hindsight_endpoints(base, rl, lookup, budget) for budget in (64, 128)],
    }
    encoded = json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n"
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(encoded)


if __name__ == "__main__":
    main()
