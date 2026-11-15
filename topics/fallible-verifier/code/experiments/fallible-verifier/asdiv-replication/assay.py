"""Assemble, score, and evaluate the sealed ASDiv-A replication panel."""

import argparse
from dataclasses import dataclass, replace
import importlib.util
import json
import os
from pathlib import Path
import re
import sys


HERE = Path(__file__).resolve().parent
TOPIC = HERE.parent
sys.path[:0] = [str(TOPIC / "gsm1k-replication"), str(TOPIC / "holdout-evaluation"),
                str(TOPIC / "instruct-candidate-coverage"), str(HERE)]
import score
from panel import candidate_tables, file_record, json_bytes, write_bundle
from receipt_contract import DIGEST, MODEL_PINS, checked_bytes, load_json


PREPARED_SHA256 = "110881d5e9a7ba12d2776a30dbebe8ef919f76a81ef39b2a829790ffd2b2b8d8"
DATASET = "asdiv-a"
COVERAGE_SCOPE = "asdiv_replication_assembled"
SUMMARY_SCOPE = "asdiv_external_replication_instruct_oat"
ROLES = ("base", "oat")
FREEZE = re.compile(r"[0-9a-f]{40}")
JOB = re.compile(r"[1-9][0-9]*")

# The frozen scoring code imports its own prepare.py under the bare module name.
prepare_spec = importlib.util.spec_from_file_location("asdiv_replication_prepare", HERE / "prepare.py")
prepare = importlib.util.module_from_spec(prepare_spec)
sys.modules[prepare_spec.name] = prepare
prepare_spec.loader.exec_module(prepare)


def assemble(prepared_dir, receipts, freeze, job, warm_gate_sha256):
    from generate_instruct import verify_proposals as verify_instruct
    from generate_oat import verify_proposals as verify_oat

    if set(receipts) != set(ROLES) or not DIGEST.fullmatch(warm_gate_sha256):
        raise ValueError("Both role receipts and the verified warm-up gate fingerprint are required")
    questions, references = prepare.verified_rows(prepared_dir, PREPARED_SHA256)
    identity = {
        "expected_freeze": freeze, "expected_job": job,
        "expected_warm_gate_sha256": warm_gate_sha256,
    }
    _, base = verify_instruct(*receipts["base"], "asdiv-full", questions, PREPARED_SHA256, **identity)
    _, oat = verify_oat(*receipts["oat"], questions, PREPARED_SHA256, **identity)
    if base.keys() & oat.keys():
        raise ValueError("ASDiv-A solver receipts overlap on candidate slots")
    payloads, stats = candidate_tables(questions, references, base | oat)
    coverage = {
        "schema": 1, "scope": COVERAGE_SCOPE, "dataset": DATASET, "freeze_commit": freeze, "slurm_job_id": job,
        "prepared_sha256": PREPARED_SHA256, "warm_gate_sha256": warm_gate_sha256,
        "solver_receipt_sha256": {role: receipts[role][1] for role in ROLES},
        "variants_in_output_order": list(score.VARIANTS), "splits": stats,
        "derived": {name: file_record(data) for name, data in payloads.items()},
    }
    return coverage, payloads | {"coverage.json": json_bytes(coverage)}


def load_coverage(path, expected_sha256, freeze, job):
    payload = path.read_bytes()
    checked_bytes(payload, expected_sha256)
    coverage = load_json(payload)
    expected = {
        "schema": 1, "scope": COVERAGE_SCOPE, "dataset": DATASET, "freeze_commit": freeze, "slurm_job_id": job,
        "prepared_sha256": PREPARED_SHA256, "variants_in_output_order": list(score.VARIANTS),
        "splits": {split: {"questions": count, "candidates": count * len(score.VARIANTS)}
                   for split, count in prepare.SPLIT_COUNTS.items()},
    }
    if (set(coverage) != {*expected, "warm_gate_sha256", "solver_receipt_sha256", "derived"}
            or any(coverage[name] != value for name, value in expected.items())):
        raise ValueError("Unexpected ASDiv-A coverage identity or roster")
    receipts = coverage["solver_receipt_sha256"]
    if (set(receipts) != set(ROLES) or any(not DIGEST.fullmatch(value) for value in receipts.values())
            or not DIGEST.fullmatch(coverage["warm_gate_sha256"])):
        raise ValueError("Invalid ASDiv-A solver or warm-up gate fingerprints")
    if set(coverage["derived"]) != {"prompts.jsonl", "gold.jsonl"}:
        raise ValueError("Invalid ASDiv-A panel file roster")
    for file in coverage["derived"].values():
        if (set(file) != {"sha256", "bytes"} or not DIGEST.fullmatch(file["sha256"])
                or type(file["bytes"]) is not int or file["bytes"] <= 0):
            raise ValueError("Invalid ASDiv-A panel file fingerprint")
    return coverage


@dataclass(frozen=True)
class PanelBinding:
    scoring: score.PanelBinding
    job: str
    warm_gate_sha256: str

    def metadata(self):
        return self.scoring.metadata() | {
            "dataset": DATASET, "slurm_job_id": self.job, "prepared_sha256": PREPARED_SHA256,
            "warm_gate_sha256": self.warm_gate_sha256,
        }


def load_panel(panel_dir, coverage_sha256, freeze, job):
    coverage = load_coverage(panel_dir / "coverage.json", coverage_sha256, freeze, job)
    binding = PanelBinding(score.panel_binding(freeze, coverage, coverage_sha256), job, coverage["warm_gate_sha256"])
    groups = score.panel_groups((panel_dir / "prompts.jsonl").read_bytes(), coverage)
    return binding, groups


def score_panel(panel_dir, coverage_sha256, freeze, job, role, output_dir):
    outputs = {split: output_dir / f"{split}-{role}.json" for split in prepare.SPLIT_COUNTS}
    if any(path.exists() or path.is_symlink() for path in outputs.values()):
        raise FileExistsError("ASDiv-A role receipts already exist")
    directory, config, tokenizer, ids, params, smoke = score.runtime(role, check_only=False)
    score.assay.report_stage("registered_prompts")
    binding, splits = load_panel(panel_dir, coverage_sha256, freeze, job)
    requests, split_aliases = score.prepare_requests(splits, tokenizer)
    score.assay.report_stage("native_gpu")
    scores = score.scoring.run_gpu(directory, role, config, requests, smoke, params, ids)
    score.assay.report_stage("receipt")
    receipts = score.make_split_receipts(splits, split_aliases, scores, role, binding)
    written = []
    try:
        for split, path in outputs.items():
            score.scoring.write_json(path, receipts[split])
            written.append(path)
    except BaseException:
        for path in written:
            path.unlink()
        raise
    return {split: score.scoring.digest(path) for split, path in outputs.items()}


def evaluate(panel_dir, coverage_sha256, freeze, job, score_dir):
    import evaluate as frozen_evaluation

    binding, groups = load_panel(panel_dir, coverage_sha256, freeze, job)
    gold = frozen_evaluation.index_gold((panel_dir / "gold.jsonl").read_bytes(), groups, binding)
    splits, hashes = {}, {}
    for split, group in groups.items():
        paired, hashes[split] = {}, {}
        for role in MODEL_PINS:
            paired[role], hashes[split][role] = frozen_evaluation.read_role(
                score_dir / f"{split}-{role}.json", role, split, group, binding,
            )
        aligned = frozen_evaluation.make_split(group, gold, paired, split, binding)
        splits[split] = replace(aligned, rows=tuple(row | {"dataset": DATASET} for row in aligned.rows))

    config = frozen_evaluation.CONFIG.read_bytes()
    config_sha = frozen_evaluation.assay.prepare.ROUND2_CONFIG_SHA256
    checked_bytes(config, config_sha)
    return {
        "scope": SUMMARY_SCOPE, "dataset": DATASET, "slurm_job_id": job,
        "criteria_sha256": config_sha, "binding": binding.metadata(), "scoring_receipt_sha256": hashes,
        **frozen_evaluation.readback(splits["train"], splits["test"], load_json(config)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("assemble")
    build.add_argument("--freeze", required=True)
    build.add_argument("--prepared-dir", type=Path, required=True)
    build.add_argument("--warm-gate-sha256", required=True)
    for role in ROLES:
        build.add_argument(f"--{role}-receipt", type=Path, required=True)
        build.add_argument(f"--{role}-receipt-sha256", required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    measure = commands.add_parser("score")
    measure.add_argument("--role", choices=tuple(MODEL_PINS), required=True)
    measure.add_argument("--check-only", action="store_true")
    measure.add_argument("--output-dir", type=Path)
    assess = commands.add_parser("evaluate")
    assess.add_argument("--score-dir", type=Path, required=True)
    assess.add_argument("--output", type=Path, required=True)
    for command in (measure, assess):
        required = command is assess
        command.add_argument("--freeze", required=required)
        command.add_argument("--panel-dir", type=Path, required=required)
        command.add_argument("--coverage-sha256", required=required)
    args = parser.parse_args()

    if args.command == "score":
        if os.environ.get("FV_REPORT_STAGE_FD3") == "1":
            score.assay.REPORT_STAGES = True
            sys.excepthook = score.assay.report_failure
        score.assay.report_stage("provenance")
        options = (args.freeze, args.panel_dir, args.coverage_sha256, args.output_dir)
        if args.check_only:
            if any(value is not None for value in options):
                parser.error("Check-only uses synthetic probes without an official panel")
            score.decision_rule_sha()
            _, _, tokenizer, _, _, _ = score.runtime(args.role, check_only=True)
            print(json.dumps({"role": args.role, **score.check_requests(tokenizer)}, sort_keys=True))
            return
        if any(value is None for value in options):
            parser.error("Supply the freeze, panel directory, coverage fingerprint and scoring output directory")

    job = os.environ.get("SLURM_JOB_ID", "")
    if not JOB.fullmatch(job) or not FREEZE.fullmatch(args.freeze or ""):
        parser.error("Supply the active Slurm job and full ASDiv-A protocol freeze")
    if args.command == "assemble":
        receipts = {role: (getattr(args, f"{role}_receipt"), getattr(args, f"{role}_receipt_sha256")) for role in ROLES}
        _, payloads = assemble(args.prepared_dir, receipts, args.freeze, job, args.warm_gate_sha256)
        write_bundle(args.output_dir, payloads)
        print(json.dumps({"coverage_sha256": file_record(payloads["coverage.json"])["sha256"]}, sort_keys=True))
    elif args.command == "score":
        hashes = score_panel(args.panel_dir, args.coverage_sha256, args.freeze, job, args.role, args.output_dir)
        for split, digest in hashes.items():
            print(json.dumps({"receipt_sha256": digest, "split": split, "role": args.role}, sort_keys=True))
    else:
        if args.output.exists() or args.output.is_symlink():
            raise FileExistsError(args.output)
        report = evaluate(args.panel_dir, args.coverage_sha256, args.freeze, job, args.score_dir)
        score.scoring.write_json(args.output, report)
        print(json.dumps({"slurm_job_id": job, "summary_sha256": score.scoring.digest(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
