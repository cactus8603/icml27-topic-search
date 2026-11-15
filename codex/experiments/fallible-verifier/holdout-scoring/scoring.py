"""Build role receipts for a frozen two-mapping verifier policy without gold labels."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import sys


TOPIC = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(TOPIC), str(TOPIC / "real_holdout"), str(TOPIC / "holdout-evaluation")]
import dev_likelihood as assay
from receipt_contract import Binding, MODEL_PINS, SPLIT_GROUPS, checked_bytes, load_json
from phase3_likelihood import label_ids, logprobs_params, run_gpu, tokenize, two_mapping, verify_model
from round2.protocol import CONFIG, digest, hash_order, question_key, write_json


PROFILES = ("raw", "neutral")
OUTPUT = assay.prepare.SHARED / "runs/fallible-verifier/gsm8k-policy-scoring"


def panel_rows(payload, split):
    if split not in SPLIT_GROUPS:
        raise ValueError("Unknown panel split")
    checked_bytes(payload, assay.prepare.EXPECTED_OUTPUTS["prompts.jsonl"]["sha256"])
    suffixes = tuple(f',"split":"{split}","variant":"{variant}"}}'.encode()
                     for variant in assay.prepare.VARIANTS)
    groups = defaultdict(dict)
    for line in payload.splitlines():
        if not line.endswith(suffixes):
            continue
        row = load_json(line)
        if set(row) != assay.PROMPT_KEYS or row["split"] != split:
            raise ValueError("Prompt row has the wrong shape or split")
        qid = sha256(question_key(row["question"]).encode()).hexdigest()
        variant = row["variant"]
        if (row["question_id"] != qid or row["id"] != hash_order(qid, variant)
                or variant in groups[qid] or not isinstance(row["solution"], str)):
            raise ValueError("Invalid or duplicate prompt identity")
        groups[qid][variant] = row
    variants = set(assay.prepare.VARIANTS)
    if len(groups) != SPLIT_GROUPS[split] or any(set(group) != variants for group in groups.values()):
        raise ValueError("Missing questions or candidate variants")
    if any(len({row["question"] for row in group.values()}) != 1 for group in groups.values()):
        raise ValueError("The four variants must share the exact question")
    return {qid: {key: groups[qid][key] for key in assay.prepare.VARIANTS} for qid in sorted(groups)}


def selected_requests(groups, tokenizer, maximum, profile):
    if profile not in PROFILES:
        raise ValueError("Unknown prompt profile")
    requests, aliases, _ = assay.requests_for(groups, tokenizer, sys.maxsize)
    expected = {(qid, (row["id"], profile), swap) for qid, group in groups.items()
                for row in group.values() for swap in (False, True)}
    selected = {key: aliases[key] if len(requests[aliases[key]]["prompt_token_ids"]) <= maximum else None
                for key in expected}
    needed = set(selected.values()) - {None}
    return {key: prompt for key, prompt in requests.items() if key in needed}, selected


def manifest_sha(groups, profile):
    """Hash the ordered role-independent inputs, independent of tokenizer aliases."""
    payload = {"profile": profile, "questions": [list(group.values()) for group in groups.values()]}
    return sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def threshold(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("Abstention width needs a finite nonnegative bound")
    return float(value)


def label(semantic, bound):
    if semantic is None:
        return None, "missing"
    if not math.isfinite(semantic):
        raise ValueError("Invalid native semantic log odds")
    if abs(semantic) <= bound:
        return None, "abstain"
    return int(semantic > 0), "observed"


def make_receipt(groups, aliases, scores, role, split, profile, bound, binding):
    model, revision = MODEL_PINS[role]
    if scores.keys() != set(aliases.values()) - {None}:
        raise ValueError("Native score coverage differs from the requested aliases")
    deltas = {key: scores[actual] if actual is not None else None for key, actual in aliases.items()}
    records = []
    for qid, group in groups.items():
        for row in group.values():
            score = two_mapping(deltas, qid, (row["id"], profile))
            verdict, state = label(score[0] if score is not None else None, bound)
            records.append({"id": row["id"], "verdict": verdict, "status": state})
    return {"schema": 1, "role": role, "model": model, "model_revision": revision,
            "split": split, "binding": binding.metadata(), "prompt_manifest_sha256": manifest_sha(groups, profile),
            "records": records}


def load_policy(measurement_path, decision_path, freeze):
    measurement, decision = measurement_path.read_bytes(), decision_path.read_bytes()
    spec, rule = load_json(measurement), load_json(decision)
    if (set(spec) != {"schema", "profile", "panel_commit", "native_likelihood_commit"}
            or spec["schema"] != 1 or spec["profile"] not in PROFILES
            or spec["panel_commit"] != assay.PANEL_COMMIT
            or spec["native_likelihood_commit"] != assay.NATIVE_COMMIT):
        raise ValueError("Unknown measurement protocol")
    if set(rule) != {"schema", "abstain_abs_log_odds"} or rule["schema"] != 1:
        raise ValueError("Unknown verdict rule")
    bounds = rule["abstain_abs_log_odds"]
    if set(bounds) != set(MODEL_PINS):
        raise ValueError("A frozen bound is required for both roles")
    binding = Binding(freeze, sha256(measurement).hexdigest(), sha256(decision).hexdigest())
    binding.metadata()
    return spec["profile"], {role: threshold(bound) for role, bound in bounds.items()}, binding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", required=True, choices=tuple(MODEL_PINS))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--split", choices=tuple(SPLIT_GROUPS))
    parser.add_argument("--freeze")
    args = parser.parse_args()
    if os.environ.get("FV_REPORT_STAGE_FD3") == "1":
        assay.REPORT_STAGES = True
        sys.excepthook = assay.report_failure
    assay.report_stage("provenance")
    if version("vllm") != "0.33.0" or digest(CONFIG) != assay.prepare.ROUND2_CONFIG_SHA256:
        raise ValueError("Frozen runtime or model config differs")
    config = load_json(CONFIG.read_bytes())
    model = config["models"][args.role]
    if (model["id"], model["revision"]) != MODEL_PINS[args.role]:
        raise ValueError("The scoring and evaluation model pins differ")
    from transformers import AutoTokenizer

    assay.report_stage("model_files")
    directory = assay.MODEL_ROOTS[args.role]
    verify_model(directory, model, args.check_only)
    tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True)
    label_token_ids = label_ids(tokenizer)
    params = logprobs_params(label_token_ids)
    maximum = config["max_input_tokens"]
    if not (assay.PANEL / "coverage.json").is_file():
        raise ValueError("No completion receipt for the frozen panel")
    if args.check_only:
        # The source receipt binds the entire file. Decode only development rows.
        groups = panel_rows((assay.PANEL / "prompts.jsonl").read_bytes(), "train")
        checks = {}
        for profile in PROFILES:
            requests, aliases = selected_requests(groups, tokenizer, maximum, profile)
            checks[profile] = {"logical_requests": len(aliases), "unique_requests": len(requests),
                               "overlength_logical_requests": sum(value is None for value in aliases.values()),
                               "input_manifest_sha256": manifest_sha(groups, profile)}
        print(json.dumps({"role": args.role, "train_questions": len(groups), "profiles": checks}, sort_keys=True))
        return

    here = Path(__file__).resolve().parent
    profile, bounds, binding = load_policy(here / "measurement.json", here / "decision.json", args.freeze or "")
    job = os.environ.get("SLURM_JOB_ID", "")
    if not job.isascii() or not job.isdecimal():
        raise ValueError("A recorded Slurm job is required")
    output = OUTPUT / job / f"{args.split}-{args.role}.json"
    if output.exists():
        raise FileExistsError(output)
    assay.report_stage("registered_prompts")
    groups = panel_rows((assay.PANEL / "prompts.jsonl").read_bytes(), args.split)
    requests, aliases = selected_requests(groups, tokenizer, maximum, profile)
    smoke = {choice: tokenize(tokenizer, f"The two allowed choices are A and B. Say {choice}.", maximum)
             for choice in ("A", "B")}
    assay.report_stage("native_gpu")
    scores = run_gpu(directory, args.role, config, requests, smoke, params, label_token_ids)
    assay.report_stage("receipt")
    receipt = make_receipt(groups, aliases, scores, args.role, args.split, profile, bounds[args.role], binding)
    write_json(output, receipt)
    print(json.dumps({"receipt_sha256": digest(output), "split": args.split, "role": args.role,
                      "status": Counter(row["status"] for row in receipt["records"])}, sort_keys=True))


if __name__ == "__main__":
    main()
