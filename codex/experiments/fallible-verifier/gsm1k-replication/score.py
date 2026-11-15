"""Score both GSM1k replication splits using the frozen GSM8K verdict policy."""

import argparse
from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import sys


TOPIC = Path(__file__).resolve().parents[1]
SCORING_SOURCE_COMMIT = "ac3950820be8c7f4f1528ecc553b9cf51aad894b"
sys.path.insert(0, str(TOPIC / "holdout-scoring"))
import scoring
from receipt_contract import DIGEST, checked_bytes


PROFILE = "neutral"
VARIANTS = ("base-0", "base-1", "oat-0", "oat-1")
SPLIT_GROUPS = {"train": 250, "test": 747}
BOUNDS = {"weak": 0.0, "strong": 0.25}
MAXIMUM = 2048
assay = scoring.assay


def load_coverage(path, expected_sha256):
    payload = path.read_bytes()
    checked_bytes(payload, expected_sha256)
    coverage = scoring.load_json(payload)
    expected_splits = {split: {"questions": count, "candidates": count * len(VARIANTS)}
                       for split, count in SPLIT_GROUPS.items()}
    if (set(coverage) != {"schema", "scope", "prepared_sha256", "solver_receipt_sha256",
                          "variants_in_output_order", "splits", "derived"}
            or coverage["schema"] != 1 or coverage["scope"] != "gsm1k_replication_assembled"
            or coverage["variants_in_output_order"] != list(VARIANTS)
            or coverage["splits"] != expected_splits):
        raise ValueError("Unexpected scoring coverage schema or roster")
    solver_receipts = coverage["solver_receipt_sha256"]
    if (set(solver_receipts) != {"base", "oat"}
            or not DIGEST.fullmatch(coverage["prepared_sha256"])
            or any(not DIGEST.fullmatch(value) for value in solver_receipts.values())):
        raise ValueError("Invalid prepared panel or solver receipt fingerprints")
    if set(coverage["derived"]) != {"prompts.jsonl", "gold.jsonl"}:
        raise ValueError("Invalid panel file roster")
    for file in coverage["derived"].values():
        if (set(file) != {"sha256", "bytes"} or not DIGEST.fullmatch(file["sha256"])
                or type(file["bytes"]) is not int or file["bytes"] <= 0):
            raise ValueError("Invalid panel file fingerprint")
    return coverage


def panel_groups(prompt_bytes, coverage):
    file = coverage["derived"]["prompts.jsonl"]
    checked_bytes(prompt_bytes, file["sha256"])
    if len(prompt_bytes) != file["bytes"]:
        raise ValueError("Unexpected prompt file length")

    splits = {split: {} for split in SPLIT_GROUPS}
    question_splits = {}
    for line in prompt_bytes.splitlines():
        row = scoring.load_json(line)
        if (set(row) != assay.PROMPT_KEYS or row["split"] not in splits or row["variant"] not in VARIANTS
                or not isinstance(row["question"], str) or not isinstance(row["solution"], str)):
            raise ValueError("Invalid prompt row shape")
        split, variant = row["split"], row["variant"]
        normalized = scoring.question_key(row["question"])
        qid = sha256(normalized.encode()).hexdigest()
        if (not normalized or row["question_id"] != qid or row["id"] != scoring.hash_order(qid, variant)
                or question_splits.setdefault(qid, split) != split):
            raise ValueError("Invalid prompt identity or overlapping splits")
        group = splits[split].setdefault(qid, {})
        if variant in group:
            raise ValueError("Duplicate candidate identity")
        group[variant] = row

    ordered = {}
    for split, groups in splits.items():
        if len(groups) != SPLIT_GROUPS[split]:
            raise ValueError("Unexpected split size")
        for group in groups.values():
            if set(group) != set(VARIANTS) or len({row["question"] for row in group.values()}) != 1:
                raise ValueError("An exact question and four candidates are required per group")
        ordered[split] = {qid: {variant: groups[qid][variant] for variant in VARIANTS} for qid in sorted(groups)}
    return ordered


@dataclass(frozen=True)
class PanelBinding:
    freeze_commit: str
    decision_rule_sha256: str
    panel_prompts_sha256: str
    panel_gold_sha256: str
    panel_coverage_sha256: str

    def metadata(self):
        return {
            "freeze_commit": self.freeze_commit,
            "profile": PROFILE,
            "scoring_source_commit": SCORING_SOURCE_COMMIT,
            "native_likelihood_commit": assay.NATIVE_COMMIT,
            "decision_rule_sha256": self.decision_rule_sha256,
            "panel_prompts_sha256": self.panel_prompts_sha256,
            "panel_gold_sha256": self.panel_gold_sha256,
            "panel_coverage_sha256": self.panel_coverage_sha256,
        }


def decision_rule_sha():
    payload = (TOPIC / "holdout-scoring" / "decision.json").read_bytes()
    rule = scoring.load_json(payload)
    if rule != {"schema": 1, "abstain_abs_log_odds": BOUNDS}:
        raise ValueError("The original verdict policy bounds are required")
    return sha256(payload).hexdigest()


def panel_binding(freeze, coverage, coverage_sha256):
    if not re.fullmatch(r"[0-9a-f]{40}", freeze or "") or not DIGEST.fullmatch(coverage_sha256):
        raise ValueError("A full protocol commit and coverage fingerprint are required")
    return PanelBinding(freeze, decision_rule_sha(), coverage["derived"]["prompts.jsonl"]["sha256"],
                        coverage["derived"]["gold.jsonl"]["sha256"], coverage_sha256)


def prepare_requests(splits, tokenizer):
    requests, split_aliases = {}, {}
    for split, groups in splits.items():
        selected, aliases = scoring.selected_requests(groups, tokenizer, MAXIMUM, PROFILE)
        if requests.keys() & selected.keys():
            raise ValueError("Native request keys overlap across splits")
        requests.update(selected)
        split_aliases[split] = aliases
    return requests, split_aliases


def make_split_receipts(splits, split_aliases, scores, role, binding):
    expected = {key for aliases in split_aliases.values() for key in aliases.values() if key is not None}
    if scores.keys() != expected:
        raise ValueError("Native score identities differ from both split manifests")
    receipts = {}
    for split, groups in splits.items():
        aliases = split_aliases[split]
        split_scores = {key: scores[key] for key in set(aliases.values()) - {None}}
        receipts[split] = scoring.make_receipt(groups, aliases, split_scores, role, split, PROFILE, BOUNDS[role], binding)
    return receipts


def runtime(role, check_only):
    if version("vllm") != "0.33.0" or scoring.digest(scoring.CONFIG) != assay.prepare.ROUND2_CONFIG_SHA256:
        raise ValueError("Frozen runtime or model config differs")
    config = scoring.load_json(scoring.CONFIG.read_bytes())
    if config["max_input_tokens"] != MAXIMUM:
        raise ValueError("Frozen input token bound differs")
    model = config["models"][role]
    if (model["id"], model["revision"]) != scoring.MODEL_PINS[role]:
        raise ValueError("The scoring and evaluation model pins differ")
    from transformers import AutoTokenizer

    assay.report_stage("model_files")
    directory = assay.MODEL_ROOTS[role]
    scoring.verify_model(directory, model, check_only)
    tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True)
    ids = scoring.label_ids(tokenizer)
    params = scoring.logprobs_params(ids)
    smoke = {choice: scoring.tokenize(tokenizer, f"The two allowed choices are A and B. Say {choice}.", MAXIMUM)
             for choice in ("A", "B")}
    return directory, config, tokenizer, ids, params, smoke


def check_requests(tokenizer):
    question = "A bag has two beads. How many beads are in the bag?"
    qid = sha256(scoring.question_key(question).encode()).hexdigest()
    solutions = ("A: 2", "Result: 2", "A: 3", "A:" + " bead" * (MAXIMUM + 1))
    group = {variant: {"id": scoring.hash_order(qid, variant), "question_id": qid, "split": "train",
                       "variant": variant, "question": question, "solution": solution}
             for variant, solution in zip(VARIANTS, solutions, strict=True)}
    splits = {"train": {qid: group}}
    requests, split_aliases = prepare_requests(splits, tokenizer)
    aliases = split_aliases["train"]
    if (len(aliases) != 8 or len(requests) != 4 or sum(key is None for key in aliases.values()) != 2):
        raise ValueError("Synthetic probes failed neutral deduplication or overlength handling")
    return {"logical_requests": len(aliases), "unique_requests": len(requests),
            "overlength_logical_requests": sum(key is None for key in aliases.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", required=True, choices=tuple(scoring.MODEL_PINS))
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--freeze")
    parser.add_argument("--panel-dir", type=Path)
    parser.add_argument("--coverage-sha256")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    if os.environ.get("FV_REPORT_STAGE_FD3") == "1":
        assay.REPORT_STAGES = True
        sys.excepthook = assay.report_failure
    assay.report_stage("provenance")
    options = (args.freeze, args.panel_dir, args.coverage_sha256, args.output_dir)
    if args.check_only:
        if any(value is not None for value in options):
            parser.error("Check-only uses synthetic probes without an official panel")
        decision_rule_sha()
        _, _, tokenizer, _, _, _ = runtime(args.role, check_only=True)
        print(json.dumps({"role": args.role, **check_requests(tokenizer)}, sort_keys=True))
        return

    if (any(value is None for value in options)
            or not re.fullmatch(r"[1-9][0-9]*", os.environ.get("SLURM_JOB_ID", ""))):
        parser.error("Scoring needs a recorded Slurm job, freeze, panel, coverage digest and output directory")
    outputs = {split: args.output_dir / f"{split}-{args.role}.json" for split in SPLIT_GROUPS}
    if any(path.exists() for path in outputs.values()):
        raise FileExistsError("Role receipts already exist")

    directory, config, tokenizer, ids, params, smoke = runtime(args.role, check_only=False)
    assay.report_stage("registered_prompts")
    coverage = load_coverage(args.panel_dir / "coverage.json", args.coverage_sha256)
    binding = panel_binding(args.freeze, coverage, args.coverage_sha256)
    splits = panel_groups((args.panel_dir / "prompts.jsonl").read_bytes(), coverage)
    requests, split_aliases = prepare_requests(splits, tokenizer)
    assay.report_stage("native_gpu")
    scores = scoring.run_gpu(directory, args.role, config, requests, smoke, params, ids)
    assay.report_stage("receipt")
    receipts = make_split_receipts(splits, split_aliases, scores, args.role, binding)
    written = []
    try:
        for split, path in outputs.items():
            scoring.write_json(path, receipts[split])
            written.append(path)
    except BaseException:
        for path in written:
            path.unlink()
        raise
    for split, path in outputs.items():
        print(json.dumps({"receipt_sha256": scoring.digest(path), "split": split, "role": args.role}, sort_keys=True))


if __name__ == "__main__":
    main()
