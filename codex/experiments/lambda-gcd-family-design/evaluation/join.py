"""Verify the completed four-family GCD receipts and compare frozen predictors."""

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
from math import isclose
import os
from pathlib import Path
import re
from statistics import fmean
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import average, evaluate, finite_number, interval, unique_keys


ROOT = Path("/work/ChihChungHsu/cactus8603/icml")
DATA_JOB, PHM_JOB, READY_JOB = "26821", "26822", "26823"
DATA_SHA = "01adff19f4fdc7598bfca7fed06ae892345dd1075e8f773fa6f0160bc6335a71"
PREFIX = "lambda-gcd-family-"
DESIGN_STAGE = "gcd_family_holdout_cpu_design_v1"
STEP_SCOPE = "gcd_family_step0_predictions"
OUTCOME_SCOPE = "gcd_family_training_outcome_v1"
READY_SCOPE = "synthetic_gcd_training_readiness"
SCOPE = "gcd_family_frozen_zero_shot_comparison_v1"
COUNT = 51
SEEDS = (0, 1, 2)
SCORE_FIELDS = ("lambda", "gradient_cosine", "negative_target_suffix_nll", "train_suffix_nll_delta",
                "train_gradient_rms", "trainable_parameter_count")
PHM_SCOPES = {
    "none": "default_control",
    "ip_single": "original_formula_single_prefix",
    "ip_bank": "weighted_bank_adaptation",
    "pps": "prefix_only_diagnostic",
    "layer_restricted": "prefix_only_diagnostic",
}
PHM_METHODS = {
    "phm_original_formula_gcd": "original_formula_single_prefix",
    "phm_weighted_bank_gcd": "weighted_bank_adaptation",
    "phm_prefix_only_gcd": "prefix_only_diagnostic",
}
JOB_ID = re.compile(r"[1-9][0-9]*")
FREEZE = re.compile(r"[0-9a-f]{40}")
SHA = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class Jobs:
    step_array: str
    step_join: str
    outcome_array: str
    outcome_judge: str

    def names(self):
        return {DATA_JOB: PREFIX + "preflight", PHM_JOB: "lambda-gcd-phm", READY_JOB: PREFIX + "train-ready",
                self.step_array: PREFIX + "step0", self.step_join: PREFIX + "step0-join",
                self.outcome_array: PREFIX + "outcomes", self.outcome_judge: PREFIX + "outcome-judge"}


def successful_allocations(jobs, query=subprocess.check_output):
    names = jobs.names()
    if len(names) != 7 or any(not JOB_ID.fullmatch(job) for job in names):
        raise ValueError("The seven source IDs must be distinct Slurm jobs")
    raw = query(["sacct", "-X", "-n", "-P", "-j", ",".join(names),
                 "-o", "JobID,State,ExitCode,JobName%60"], text=True)
    expected = set()
    for job, name in names.items():
        ids = (f"{job}_{task}" for task in range(COUNT)) if job in (jobs.step_array, jobs.outcome_array) else (job,)
        expected.update(f"{number}|COMPLETED|0:0|{name}" for number in ids)
    actual = raw.splitlines()
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Expected only successful source jobs, including both complete 51-cell arrays")


def recorded_freezes(ledger, names):
    found = {}
    for line in ledger.read_text(encoding="utf-8").splitlines():
        words = line.split()
        if len(words) < 8 or words[0] != "[codex]" or words[3] != "submit" or words[4] not in names:
            continue
        job = words[4]
        if job in found or words[5] != "inoculation-lambda" or not FREEZE.fullmatch(words[-1]):
            raise ValueError(f"Ambiguous or incompatible recorded freeze for {job}")
        found[job] = words[-1]
    if set(found) != set(names):
        raise ValueError("One or more completed sources lack a recorded frozen submission")
    return found


def trusted_file(root, relative):
    path = root / relative
    if path.is_symlink() or not path.is_file() or path.resolve() != root / relative:
        raise ValueError(f"Expected an unaliased regular file: {relative}")
    return path


def binding(obj, expected, subject):
    if not isinstance(obj, dict) or any(obj.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Unexpected {subject} binding")


def source_json(root, path, log, expected, *, sha_key="sha256", path_key=None, pinned=None):
    lines = trusted_file(root, log).read_bytes().strip().splitlines()
    if not lines:
        raise ValueError(f"Missing successful receipt declaration in {log}")
    record = json.loads(lines[-1], object_pairs_hook=unique_keys)
    binding(record, expected, log)
    digest = record.get(sha_key)
    if not isinstance(digest, str) or not SHA.fullmatch(digest) or (pinned is not None and digest != pinned):
        raise ValueError(f"Unexpected declared SHA-256 in {log}")
    target = trusted_file(root, path)
    if path_key is not None:
        declared = Path(record[path_key])
        if not declared.is_absolute() or declared.is_symlink() or declared.resolve() != target:
            raise ValueError(f"Unexpected declared path in {log}")
    raw = target.read_bytes()
    if sha256(raw).hexdigest() != digest:
        raise ValueError(f"Receipt SHA-256 differs from {log}")
    return json.loads(raw, object_pairs_hook=unique_keys), digest


def checked_inputs(root, jobs, freezes):
    data_path = f"data/lambda-gcd/family-design-{DATA_JOB}/manifest.json"
    manifest, manifest_sha = source_json(root, data_path, f"tmp/{PREFIX}preflight-{DATA_JOB}.log", {},
                                         sha_key="manifest_sha256", pinned=DATA_SHA)
    binding(manifest, {"stage": DESIGN_STAGE, "train_seeds": list(SEEDS), "cpu_pass": True,
                       "training_outcomes_opened": False}, "CPU design")
    split_hashes = {name: item["sha256"] for name, item in manifest["splits"].items()}
    ready, ready_sha = source_json(root, f"runs/{PREFIX}train-ready/{READY_JOB}/summary.json",
                                  f"tmp/{PREFIX}train-ready-{READY_JOB}.log", {"scope": READY_SCOPE, "gpu_cells": 8},
                                  path_key="output")
    binding(ready, {"schema": 1, "scope": READY_SCOPE, "freeze": freezes[READY_JOB],
                    "family_design": {"manifest_sha256": manifest_sha, "split_sha256": split_hashes},
                    "validated_author_configs": COUNT, "assets": manifest["assets"]}, "GPU readiness")

    phm, phm_sha = source_json(root, f"runs/{PREFIX}phm/{PHM_JOB}/score.json", f"tmp/lambda-gcd-phm-{PHM_JOB}.log",
                               {"job": PHM_JOB, "mode": "score"}, sha_key="receipt_sha256")
    binding(phm, {"schema": 1, "job": PHM_JOB, "mode": "score", "freeze": freezes[PHM_JOB],
                   "design_sha256": manifest_sha, "calibration_sha256": split_hashes["phm-calibration"],
                   "model_snapshot": manifest["assets"]["model_snapshot"], "calibration_scope": "gcd_neutral_adaptation",
                   "calibration_rows": manifest["phm_interface"]["calibration_rows"]}, "PHM panel")

    step_path = f"runs/{PREFIX}step0/{jobs.step_array}/summary.json"
    step, step_sha = source_json(root, step_path, f"tmp/{PREFIX}step0-join-{jobs.step_join}.log", {"scope": STEP_SCOPE},
                                 path_key="path")
    out_path = f"runs/{PREFIX}outcomes/{jobs.outcome_array}/summary.json"
    outcome, out_sha = source_json(root, out_path, f"tmp/{PREFIX}outcome-judge-{jobs.outcome_judge}.log", {"scope": OUTCOME_SCOPE},
                                    path_key="output")
    common = {"schema": 1, "manifest_sha256": manifest_sha, "training_readiness_sha256": ready_sha}
    binding(step, common | {"scope": STEP_SCOPE, "array": int(jobs.step_array), "freeze": freezes[jobs.step_join],
                            "method_direction": "smaller_predicts_less_leakage"}, "step-0 summary")
    binding(outcome, common | {"scope": OUTCOME_SCOPE, "array": jobs.outcome_array,
                               "freeze": freezes[jobs.outcome_judge], "prediction_panel_sha256": step_sha}, "outcome summary")
    check_cells(root, manifest, jobs, freezes, common, step, step_sha, outcome)
    return manifest, step, outcome, phm, {"manifest": manifest_sha, "readiness": ready_sha, "phm": phm_sha,
                                         "step0_panel": step_sha, "outcome_panel": out_sha}


def check_cells(root, manifest, jobs, freezes, common, step, step_sha, outcome):
    roster = {f"{arm}:{seed}" for arm in manifest["arms"] for seed in SEEDS}
    rows = outcome["runs"]
    if (len(roster) != COUNT or set(step["cells"]) != roster or set(outcome["cell_receipt_sha256"]) != roster
            or len(rows) != COUNT or {f"{row['arm']}:{row['seed']}" for row in rows} != roster):
        raise ValueError("The frozen roster must match all 51 prediction and outcome cells")
    for task, run in enumerate(rows):
        arm, seed = run["arm"], run["seed"]
        key = f"{arm}:{seed}"
        prediction = step["cells"][key]
        step_cell, step_cell_sha = source_json(
            root, f"runs/{PREFIX}step0/{jobs.step_array}/cell-{task:02d}.json",
            f"tmp/{PREFIX}step0-{jobs.step_array}_{task}.log", {"scope": "gcd_family_step0_cell"},
            path_key="path", pinned=prediction["cell_receipt_sha256"],
        )
        result_cell, _ = source_json(
            root, f"runs/{PREFIX}outcomes/{jobs.outcome_array}/cell-{task}.json",
            f"tmp/{PREFIX}outcomes-{jobs.outcome_array}_{task}.log", {"scope": OUTCOME_SCOPE},
            path_key="output", pinned=outcome["cell_receipt_sha256"][key],
        )
        initial = prediction["initial_A_B_sha256"]
        if not SHA.fullmatch(initial) or set(prediction["scores"]) != set(SCORE_FIELDS):
            raise ValueError(f"Missing frozen prediction values or initialization SHA: {key}")
        recipe = manifest["arms"][arm]
        cell = common | {"arm": arm, "seed": seed, "recipe_sha256": recipe["recipe_sha256_by_seed"][str(seed)],
                         "training_sha256": recipe["training_sha256"], "initial_A_B_sha256": initial}
        binding(step_cell, cell | {"scope": "gcd_family_step0_cell", "task": task, "array": int(jobs.step_array),
                                   "freeze": freezes[jobs.step_array], "scores": prediction["scores"]}, f"prediction {key}")
        binding(result_cell, cell | {"scope": OUTCOME_SCOPE, "task": task, "array": jobs.outcome_array,
                                     "freeze": freezes[jobs.outcome_array], "prediction_panel_sha256": step_sha,
                                     "prediction_cell_sha256": step_cell_sha}, f"training outcome {key}")


def predictor_panel(manifest, step, outcome, phm):
    arms = manifest["arms"]
    by_scope = {arm: PHM_SCOPES[row["family"]] for arm, row in arms.items()}
    declared_scopes = {row["id"]: row["scope"] for row in manifest["phm_interface"]["arms"]}
    if phm["method_scope_by_arm"] != by_scope or declared_scopes != by_scope:
        raise ValueError("PHM scopes must match the frozen design and family-specific applications")
    phm_scores = phm["predicted_leakage_scores_by_arm"]
    observed = {row["id"]: row for row in outcome["arms"]}
    runs = {(row["arm"], row["seed"]): row for row in outcome["runs"]}
    if set(phm_scores) != set(arms) or set(observed) != set(arms) or len(outcome["arms"]) != len(arms):
        raise ValueError("The PHM and outcome arm lists must match the frozen roster")
    methods = dict.fromkeys(SCORE_FIELDS, "full") | {"gradient_cosine": "common_support"}
    methods.update(dict.fromkeys((*PHM_METHODS, "phm_gcd_adaptation_all"), "common_support"))
    result, matched = [], set()
    for arm in sorted(arms.keys() - {"none"}):
        row = observed[arm]
        if row["family"] != arms[arm]["family"] or type(row["id_matched_all_seeds"]) is not bool:
            raise ValueError(f"Invalid outcome family or ID status for {arm}")
        by_seed = [runs[arm, seed] for seed in SEEDS]
        if row["id_matched_all_seeds"] != all(item["id_matching_status"] == "matched" for item in by_seed):
            raise ValueError(f"Per-seed ID status disagrees with the arm summary: {arm}")
        for index, bound in enumerate(("lower", "upper")):
            values = [item["leakage_bounds"][index] for item in by_seed]
            if not isclose(fmean(values), row["leakage"][bound], rel_tol=0, abs_tol=1e-12):
                raise ValueError(f"Per-seed leakage disagrees with the arm summary: {arm}")
        scores = {}
        for method in SCORE_FIELDS:
            values = [step["cells"][f"{arm}:{seed}"]["scores"][method] for seed in SEEDS]
            if method == "gradient_cosine" and any(value is None for value in values):
                scores[method] = None
            else:
                scores[method] = fmean(finite_number(value, method) for value in values)
        scores.update({method: phm_scores[arm] if by_scope[arm] == scope else None for method, scope in PHM_METHODS.items()})
        scores["phm_gcd_adaptation_all"] = phm_scores[arm]
        if row["id_matched_all_seeds"]:
            matched.add(arm)
        result.append({"id": arm, "family": row["family"], "leakage": row["leakage"], "scores": scores})
    return {"schema": 1, "task": "gcd", "methods": methods, "arms": result}, matched


def report_panel(panel):
    if not panel["arms"]:
        return None
    report = evaluate(panel)
    leakage = {row["id"]: row["leakage"] for row in panel["arms"]}
    for comparison in report["comparisons"].values():
        folds = comparison["folds"]
        for fold in folds.values():
            if "selected_arms" not in fold:
                continue
            values = {}
            for method, arm in fold["selected_arms"].items():
                low, high = leakage[arm]["lower"], leakage[arm]["upper"]
                values[method] = interval((low + high) / 2, low, high)
            fold["selected_leakage"] = values
        eligible = comparison["macro_eligible_families"]
        if eligible:
            selected = [folds[family]["selected_leakage"] for family in eligible]
            comparison["macro"]["selected_leakage"] = {method: average([row[method] for row in selected])
                                                         for method in selected[0]}
    return report


def run(root, ledger, jobs, freeze, job, query=subprocess.check_output):
    successful_allocations(jobs, query)
    return make_report(root.resolve(), ledger, jobs, freeze, job)


def make_report(root, ledger, jobs, freeze, job):
    freezes = recorded_freezes(ledger, jobs.names())
    manifest, step, outcome, phm, digests = checked_inputs(root, jobs, freezes)
    panel, matched = predictor_panel(manifest, step, outcome, phm)
    id_panel = panel | {"arms": [row for row in panel["arms"] if row["id"] in matched]}
    return {"schema": 1, "scope": SCOPE, "freeze": freeze, "job": job,
            "source_jobs": jobs.names(), "source_freezes": freezes, "source_sha256": digests,
            "id_matched_arm_ids": sorted(matched), "defended_range_lower": outcome["defended_range_lower"],
            "leakage_reliability": outcome["leakage_reliability"],
            "phm_scope_by_arm": phm["method_scope_by_arm"],
            "all_defended": report_panel(panel), "id_matched_sensitivity": report_panel(id_panel)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--ledger", required=True, type=Path)
    for name in ("step0-array", "step0-join", "outcome-array", "outcome-judge"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    job = os.environ.get("SLURM_JOB_ID", "")
    if not JOB_ID.fullmatch(job) or not FREEZE.fullmatch(args.freeze):
        parser.error("Run from a non-array Slurm allocation with a full frozen commit")
    jobs = Jobs(args.step0_array, args.step0_join, args.outcome_array, args.outcome_judge)
    if job in jobs.names() or os.environ.get("SLURM_ARRAY_JOB_ID"):
        parser.error("Final comparison requires its own non-array Slurm allocation")
    report = run(ROOT, args.ledger, jobs, args.freeze, job)
    output = ROOT / f"runs/{PREFIX}comparison/{job}/summary.json"
    raw = (json.dumps(report, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=False)
    output.write_bytes(raw)
    print(json.dumps({"scope": SCOPE, "output": str(output), "sha256": sha256(raw).hexdigest(), "job": job}))


if __name__ == "__main__":
    main()
