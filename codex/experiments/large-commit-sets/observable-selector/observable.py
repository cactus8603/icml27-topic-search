"""Replay a fixed LLaDA selector using only the logits of the natural decoding step."""

import argparse
import hashlib
import json
import math
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import median
from zoneinfo import ZoneInfo


SOURCE_FREEZE = "c979eba0260a2a6f5bd3a1de5cf6affb4ea7a5af"
VERDICT_SHA = "c7397ad04f021b7a5a107464b22342f2268e25b10a7b38e213c798fe9831814d"
JOBS = {"train": "26441", "test": "26442", "verdict": "26443"}
SOURCE_RECEIPTS = {
    "train": "f3ad5fba802c733287259ec2827aaf9a387e28cecff29293c1bde52c6162fd39",
    "test": "8144334c52bb955006e3b8fde7392ab277f7f2c8337ffd40f0d470cd083e7082",
}
SHARED = Path("/work/ChihChungHsu/cactus8603/icml")
DATA = SHARED / "data"
ROUND2 = Path(__file__).resolve().parent.parent / "round2"
STAGE = "large-commit-sets-observable-selector"
MINIMUM_TRAIN_STATE = 20


@dataclass(frozen=True)
class Observation:
    step: int
    first_swap_confidence_loss: float
    shortest_swap_span: int

    def state(self, confidence_cutoff, span_cutoff):
        return (
            self.step,
            self.first_swap_confidence_loss > confidence_cutoff,
            self.shortest_swap_span > span_cutoff,
        )


def observe(anchor, settings):
    if anchor is None:
        return None
    candidates = anchor["candidates"]
    confidence = {}
    for candidate in candidates:
        for index, value in zip(candidate["positions"], candidate["confidence"], strict=True):
            if not math.isfinite(value) or value > 0 or confidence.setdefault(index, value) != value:
                raise ValueError("Position has invalid or inconsistent natural-step confidence.")
    baseline = set(candidates[0]["positions"])
    swaps = []
    for candidate in candidates[1:]:
        alternative = set(candidate["positions"])
        (dropped,) = baseline - alternative
        (added,) = alternative - baseline
        loss, span = confidence[dropped] - confidence[added], abs(added - dropped)
        if not 0 <= loss <= settings["max_swap_logprob_gap"] or span > settings["max_swap_distance"]:
            raise ValueError("Candidate differs from the natural-step swap constraints.")
        swaps.append((loss, span, dropped, added))
    if swaps != sorted(swaps):
        raise ValueError("Candidate swaps differ from the frozen natural-step order.")
    return Observation(anchor["step"], swaps[0][0], min(swap[1] for swap in swaps))


def correct_choices(row):
    anchor = row["anchor"]
    return (
        tuple(int(candidate["correct"]) for candidate in anchor["candidates"])
        if anchor is not None
        else (int(row["baseline_correct"]),)
    )


def fit_states(observations, choices, minimum_size=MINIMUM_TRAIN_STATE):
    eligible = [row for row in observations if row is not None]
    if not eligible:
        raise ValueError("No eligible train rows for state cutoffs.")
    cutoffs = (
        median(row.first_swap_confidence_loss for row in eligible),
        median(row.shortest_swap_span for row in eligible),
    )
    members = defaultdict(list)
    for observation, outcomes in zip(observations, choices, strict=True):
        if observation is not None:
            members[observation.state(*cutoffs)].append(outcomes)
    table = {
        state: best_option(outcomes)
        for state, outcomes in members.items()
        if len(outcomes) >= minimum_size
    }
    return cutoffs, table, {state: len(rows) for state, rows in members.items()}


def best_option(outcomes):
    totals = tuple(map(sum, zip(*outcomes, strict=True)))
    return max(range(len(totals)), key=totals.__getitem__)


def score_states(observations, outcomes, cutoffs, table):
    states = [row.state(*cutoffs) if row is not None else None for row in observations]
    selected = [table.get(state, 0) for state in states]
    in_support = defaultdict(list)
    for state, choices in zip(states, outcomes, strict=True):
        if state in table:
            in_support[state].append(choices)
    hindsight = {state: best_option(choices) for state, choices in in_support.items()}
    diagnostic = {
        "confidence": [choices[0] for choices in outcomes],
        "train_fitted": [choices[index] for choices, index in zip(outcomes, selected, strict=True)],
        "state_oracle": [
            choices[hindsight.get(state, 0)]
            for state, choices in zip(states, outcomes, strict=True)
        ],
        "supported_pool_oracle": [
            max(choices) if state in table else choices[0]
            for state, choices in zip(states, outcomes, strict=True)
        ],
        "three_candidate_oracle": [max(choices) for choices in outcomes],
    }
    return (
        diagnostic,
        sum(len(rows) for rows in in_support.values()),
        sum(index != 0 for index in selected),
    )


def verify_published_phenomenon(rows, verdict, threshold):
    original = verdict["phenomenon"]
    eligible = [row["anchor"]["candidates"][0] for row in rows if row["anchor"] is not None]
    flips = sum(
        any(
            (a >= threshold) != (j >= threshold)
            for a, j in zip(candidate["additive_margins"], candidate["joint_margins"], strict=True)
        )
        for candidate in eligible
    )
    if (
        flips != original["events"]
        or len(eligible) != original["prompts"]
        or len(eligible) != verdict["eligible_prompts"]
    ):
        raise ValueError("Phenomenon counts disagree with the published Round 2 verdict.")
    return original


def round2_source():
    sys.path.insert(0, str(ROUND2))
    import score

    return score


def read_source(source, results):
    verdict_path = results / f"verdict-{JOBS['verdict']}.json"
    paths = {key: results / f"{key}-{JOBS[key]}.jsonl" for key in ("train", "test")}
    if any(not path.is_file() or path.is_symlink() for path in (*paths.values(), verdict_path)):
        raise ValueError("Require exact, regular Round 2 receipt files.")
    verdict_raw = verdict_path.read_bytes()
    if hashlib.sha256(verdict_raw).hexdigest() != VERDICT_SHA:
        raise ValueError("Published Round 2 verdict hash changed.")
    verdict = json.loads(verdict_raw)
    config, original, _, splits, provenance = source.load_design(
        DATA / "gsm8k-hf/main/train-00000-of-00001.parquet",
        DATA / "gsm8k/main/test-00000-of-00001.parquet",
        SHARED / "envs/biasgen/bin/python",
    )
    provenance["artifacts_sha256"] = source.sha(ROUND2 / "artifacts.json")
    expected = dict(provenance)
    for key in paths:
        expected[f"{key}_job_id"] = JOBS[key]
        expected[f"{key}_doc_ids"] = len(splits[key])
    if expected != verdict["metadata"]:
        raise ValueError("Source metadata does not match the published Round 2 verdict.")
    settings = original["decoder"]
    if settings["candidate_count"] != 3 or settings["generation_length"] // settings["quota"] != 32:
        raise ValueError("Expected the original fixed three candidates and 32-step budget.")
    rows = {}
    for key, path in paths.items():
        if source.sha(path) != SOURCE_RECEIPTS[key]:
            raise ValueError(f"Source receipt fingerprint differs for {key}.")
        metadata, rows[key] = source.load_artifact(path, key, provenance, splits[key], settings)
        if metadata["job_id"] != JOBS[key] or source.sha(path) != SOURCE_RECEIPTS[key]:
            raise ValueError(f"Source job ID or receipt bytes changed for {key}.")
    return config, settings, rows, verdict


def analyze(source, config, settings, rows, verdict):
    cutoffs, table, support_counts = fit_states(
        [observe(row["anchor"], settings) for row in rows["train"]],
        [correct_choices(row) for row in rows["train"]],
    )
    observations = [observe(row["anchor"], settings) for row in rows["test"]]
    evidence, supported, changed = score_states(
        observations, [correct_choices(row) for row in rows["test"]], cutoffs, table,
    )
    n = len(observations)
    counts = {key: sum(values) for key, values in evidence.items()}
    accuracy = {key: count / n for key, count in counts.items()}
    original_accuracy = verdict["descriptive"]["accuracy_by_arm"]
    if (
        n != verdict["test_prompts"]
        or accuracy["confidence"] != original_accuracy["confidence"]
        or accuracy["three_candidate_oracle"] != original_accuracy["pool_oracle"]
    ):
        raise ValueError("Test size, confidence or three-candidate oracle changed.")
    confidence, fitted, state, eligible_pool, full_pool = (counts[key] for key in evidence)
    if not max(confidence, fitted) <= state <= eligible_pool <= full_pool:
        raise ValueError("Expected nested finite-panel optima.")
    deltas = [
        method - ref
        for method, ref in zip(evidence["train_fitted"], evidence["confidence"], strict=True)
    ]
    threshold = config["descriptive_selector_accuracy_gain_target"]
    comparison = source.gate(
        source.paired_bootstrap(deltas, config), threshold,
        sum(row is not None for row in observations) >= config["minimum_eligible"],
    )
    return {
        "phenomenon_replayed": verify_published_phenomenon(rows["test"], verdict, config["margin_threshold"]),
        "test_prompts": n,
        "test_prompts_in_train_supported_states": supported,
        "test_prompts_with_changed_action": changed,
        "train_cutoffs": {"first_swap_confidence_loss": cutoffs[0], "shortest_swap_span": cutoffs[1]},
        "train_states": [
            {"step": step, "loss_above_cutoff": loss, "span_above_cutoff": span, "train_prompts": count,
             "eligible": (step, loss, span) in table, "action": table.get((step, loss, span), 0)}
            for (step, loss, span), count in sorted(support_counts.items())
        ],
        "accuracy": accuracy,
        "correct_counts": counts,
        "signal_train_fitted_minus_confidence": comparison | {
            "newly_correct": deltas.count(1), "newly_wrong": deltas.count(-1),
        },
        "headroom_pp": {
            "fitted_to_state_oracle": 100 * (state - fitted) / n,
            "confidence_to_state_oracle": 100 * (state - confidence) / n,
            "state_oracle_to_supported_pool_oracle": 100 * (eligible_pool - state) / n,
            "supported_pool_oracle_to_three_candidate_oracle": 100 * (full_pool - eligible_pool) / n,
            "confidence_to_three_candidate_oracle": 100 * (full_pool - confidence) / n,
            "fitted_to_three_candidate_oracle": 100 * (full_pool - fitted) / n,
        },
        "deployment_model_forwards_per_prompt": 32,
        "analysis_model_forwards": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--job")
    mode.add_argument("--local-run-id")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.freeze):
        parser.error("Use the full reviewed commit.")
    source = round2_source()
    identity = {
        "scope": "round2_observable_state_selector_historical_diagnostic",
        "freeze_commit": args.freeze,
        "source_freeze_commit": SOURCE_FREEZE,
        "source_jobs": JOBS,
        "source_verdict_sha256": VERDICT_SHA,
        "source_receipt_sha256": SOURCE_RECEIPTS,
        "minimum_train_state": MINIMUM_TRAIN_STATE,
    }
    if args.check_only:
        config = json.loads((ROUND2 / "config.json").read_text())
        if (config["train_sample_size"], config["test_sample_size"]) != (480, 1000):
            raise ValueError("Source design changed.")
        print(json.dumps(identity | {"stage": "static_protocol_checked"}, sort_keys=True))
        return
    if os.environ.get("PILOT_FREEZE_COMMIT") != SOURCE_FREEZE:
        parser.error("Run with the sealed source loader.")
    if args.job and (not re.fullmatch(r"[1-9][0-9]*", args.job) or os.getenv("SLURM_JOB_ID") != args.job):
        parser.error("Slurm identifier does not match the execution environment.")
    if args.local_run_id and not re.fullmatch(r"local-[a-zA-Z0-9]+", args.local_run_id):
        parser.error("Invalid local run identity.")
    run_id = args.job or args.local_run_id
    directory = SHARED / "runs" / STAGE / run_id
    if directory.exists():
        raise FileExistsError(directory)
    config, settings, rows, verdict = read_source(source, source.results_directory())
    result = analyze(source, config, settings, rows, verdict)
    receipt = identity | {
        "run_id": run_id,
        "execution_mode": "slurm_cpu" if args.job else "local_cpu",
        "read_at_taipei": datetime.now(ZoneInfo("Asia/Taipei")).isoformat(timespec="seconds"),
    }
    directory.mkdir(parents=True, exist_ok=False)
    payload = (json.dumps(result | receipt, sort_keys=True, allow_nan=False) + "\n").encode()
    path = directory / "summary.json"
    with path.open("xb") as handle:
        handle.write(payload)
    print(json.dumps(receipt | {"summary": str(path), "summary_sha256": hashlib.sha256(payload).hexdigest()},
                     sort_keys=True))


if __name__ == "__main__":
    main()
