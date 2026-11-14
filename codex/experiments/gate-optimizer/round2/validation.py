import argparse
import copy
import hashlib
import json
import os
import time
from pathlib import Path

import torch

from experiment import cell, configuration, prepare, sha256, synthetic, windows, write_json
from judge import judge


CALIBRATION_COMMIT = "9e4f3025621ad9c86ccd43cae25148603bb50954"
CALIBRATION_SHA256 = "28676645e5e7722c4f07ebb084bcb2cfab2492590310bde1b8b9c060f0d511df"
MULTIPLIER = 4


def verify_calibration(path, calibration, config_sha256):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != CALIBRATION_SHA256:
        raise ValueError("calibration differs from the independently reviewed aggregate")
    prior = json.loads(data)
    if (
        prior["smoke"] or prior["frozen_commit"] != CALIBRATION_COMMIT
        or prior["config_sha256"] != config_sha256 or prior["status"] != "advance_to_review"
        or prior["selected_multiplier"] != MULTIPLIER
        or prior["separate_validation_plan"] != calibration["separate_validation_plan"]
    ):
        raise ValueError("reviewed calibration does not authorize the frozen validation")


def validation_summary(metadata, output):
    plan = metadata["config"]["separate_validation_plan"]
    # Adapt only the population and coverage thresholds of the frozen round-two judge.
    input_record = copy.deepcopy(metadata)
    config = input_record["config"]
    config["calibration_seeds"] = plan["seeds"]
    config["stress_multipliers"] = [MULTIPLIER]
    config["criteria"]["signal_min_damage"] = plan["min_damage_events"]
    config["criteria"]["signal_min_scored_seeds"] = plan["min_scored_seeds"]
    mechanical = judge(input_record, output)
    pressure = mechanical["multipliers"][str(MULTIPLIER)]
    signal = pressure["signal"]
    increment = signal.get("mean_held_out_increment")
    if mechanical["status"] == "unable_incomplete_grid":
        decision = "unable_incomplete_grid"
    elif not pressure["all_cells_eligible"]:
        decision = "unable_quality_guard"
    elif (
        pressure["damaged_steps"] < plan["min_damage_events"]
        or signal.get("scored_seeds", 0) < plan["min_scored_seeds"]
    ):
        decision = "unable_insufficient_coverage"
    elif increment >= plan["min_average_increment_auc"]:
        decision = "signal_pass"
    else:
        decision = "signal_below_margin"
    return {
        "status": decision,
        "multiplier": MULTIPLIER,
        "validation_plan": plan,
        "pressure": pressure,
        "calibration_sha256": CALIBRATION_SHA256,
        **{key: metadata[key] for key in ("smoke", "slurm_id", "frozen_commit", "frozen_tree",
                                         "config_sha256", "prereg_sha256", "corpus_sha256", "device")},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--calibration-result", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--seconds", type=int, default=3500)
    parser.add_argument("--frozen-commit")
    parser.add_argument("--frozen-tree")
    parser.add_argument("--prereg-sha256")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    deadline = time.monotonic() + args.seconds
    calibration = json.loads(args.config.read_text())
    config_hash = sha256(args.config)
    verify_calibration(args.calibration_result, calibration, config_hash)
    if sha256(args.base_config) != calibration["base_config_sha256"]:
        raise ValueError("base config differs from the frozen experiment")
    base = configuration(json.loads(args.base_config.read_text()), calibration, args.smoke)
    if args.smoke:
        torch.set_num_threads(1)
        device = torch.device("cpu")
        synthetic(base, args.corpus)
    else:
        if not all((args.frozen_commit, args.frozen_tree, args.prereg_sha256)):
            parser.error("validation requires frozen commit, tree and PREREG hash")
        if sha256(Path(__file__).with_name("PREREG_validation.md")) != args.prereg_sha256:
            raise ValueError("validation PREREG hash mismatch")
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA with BF16 required")
        device = torch.device("cuda")
        prepare(base, args.corpus)
    args.output.mkdir(parents=True, exist_ok=False)
    t = base["training"]
    streams = windows(args.corpus, base["model"], t)
    metadata = {
        "config": calibration, "training": t, "model": base["model"], "smoke": args.smoke,
        "torch": torch.__version__, "frozen_commit": args.frozen_commit, "frozen_tree": args.frozen_tree,
        "config_sha256": config_hash, "prereg_sha256": args.prereg_sha256,
        "corpus_sha256": sha256(args.corpus / "corpus.json") if not args.smoke else None,
        "slurm_id": os.getenv("SLURM_JOB_ID"),
        "device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
    }
    write_json(args.output / "run.json", metadata)
    try:
        for seed in calibration["separate_validation_plan"]["seeds"]:
            for arm in t["arms"]:
                cell(base["model"], t, arm, calibration["base_rates"][arm["name"]], seed,
                     [MULTIPLIER], calibration["criteria"]["damage_nll"], streams, args.output,
                     device, deadline, args.smoke)
    finally:
        write_json(args.output / "validation.json", validation_summary(metadata, args.output))


if __name__ == "__main__":
    main()
