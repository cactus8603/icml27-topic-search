"""CPU-only paired transfer on the fixed 12-query seed 0/6 pilot."""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from zoneinfo import ZoneInfo

import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
GIT_REPO = Path(os.environ.get("MULTIFID_GIT_REPO", REPO))
TOPIC = REPO / "killtests/multifidelity-attribution"
PILOT = HERE.parent / "multifidelity-second-seed"
RUNS = Path("/work/ChihChungHsu/cactus8603/icml/runs/multifid")
OLD = RUNS / "exact_probe/target12-599469e"
LATER = RUNS / "second_seed/26535"
JOB = 26535
SEED6_FREEZE = "c6ff6aabd5229f3bd65966a741b5ad7789187c94"
OLD_RECEIPT = REPO / "codex/multifidelity-heldout/results.json"
OLD_RECEIPT_SHA = "211d732034aef0b4f4136df331492d96688100a8ca721337d5104073df537ad3"
OLD_MANIFEST_SHA = "b9be9b845a015a6fffdac9bda229e74d4e993526cfcf0e83c33208e8ab37e0f4"
OLD_COMPLETION_SHAS = (
    "f144d7f2b30fbdb66a223aad52f7785570b54bbee61518d2ab87a7b16d642c02",
    "32ebcbab094fa74f4888279ef0f6bea8aec6a8ff2598b1846f4575effb54fa19",
)
N = 50_000
BOOT_SEED = 2026111538
BOOT_DRAWS = 10_000
MIN_COVERAGE = 9
SEEDS = (0, 6)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pin_checkout(commit, paths):
    for path in paths:
        relative = path.relative_to(REPO).as_posix()
        body = subprocess.check_output(["git", "-C", str(GIT_REPO), "show", f"{commit}:{relative}"])
        require(hashlib.sha256(body).hexdigest() == sha(path), f"Frozen source changed: {relative}")


def check_inputs():
    pin_checkout(SEED6_FREEZE, [PILOT / name for name in ("pilot.py", "config.json", "cifar.sha256")])
    sys.path.insert(0, str(PILOT))
    from pilot import plan

    spec = read(PILOT / "config.json")
    cfg, queries, masks, identity = plan(TOPIC, spec)
    require(cfg["train_seed"] == 0 and identity["train_seed"] == 6 and cfg["crossseed_train_seed"] == 5,
            "Unexpected seed roles")
    require(identity["mask_ids"] == list(range(100, 200)) and len(queries) == 12, "Unexpected panel")
    require(sha(OLD / "manifest.json") == OLD_MANIFEST_SHA and sha(OLD_RECEIPT) == OLD_RECEIPT_SHA,
            "Changed historical receipt")
    metadata = read(OLD / "manifest.json")
    require(metadata["git_head"] == spec["source_commit"] and metadata["train_seed"] == 0
            and metadata["positions"] == identity["query_positions"] and metadata["cifar_indices"] == queries.tolist()
            and metadata["mask_test"] == [100, 199], "Seed-0 manifest does not match seed 6")
    upstream = read(OLD_RECEIPT)["inputs"]
    proofs = {}
    for i, (start, end) in enumerate(spec["exact_chunks"]):
        folder = OLD / f"shard-{i}"
        marker = folder / "complete.json"
        require(sha(marker) == OLD_COMPLETION_SHAS[i], "Changed seed-0 exact shard receipt")
        done = read(marker)
        artifact = folder / "exact" / f"Y_{start}_{end}.npy"
        require(done["start"] == start and done["end"] == end and done["frozen_source_copy"],
                "Unexpected seed-0 exact shard")
        proofs[artifact] = done["sha256"][str(artifact)]
    stage0 = RUNS / "stage0"
    for key in ["cheap/ekfac_1e-8.npy", *(f"retrain/loss_{j}.npy" for j in identity["mask_ids"])]:
        proofs[stage0 / key] = upstream["file_sha256"][key]
    for path, digest in proofs.items():
        require(sha(path) == digest, f"Changed or incomplete seed-0 input: {path.name}")
    return spec, masks, identity, proofs


def completed():
    command = ["sacct", "-j", str(JOB), "-X", "-n", "-P", "--format", "JobIDRaw,State,ExitCode"]
    lines = subprocess.check_output(command, text=True).splitlines()
    require([line.strip() for line in lines if line.strip()] == [f"{JOB}|COMPLETED|0:0"],
            "Wait for the frozen seed-6 job to complete successfully before opening its artifacts")


def load_seed0(spec, identity, proofs):
    positions = identity["query_positions"]
    query_count = len(positions)
    exact_files = [OLD / f"shard-{i}/exact/Y_{s}_{e}.npy" for i, (s, e) in enumerate(spec["exact_chunks"])]
    cheap_file = RUNS / "stage0/cheap/ekfac_1e-8.npy"
    loss_files = [RUNS / f"stage0/retrain/loss_{j}.npy" for j in identity["mask_ids"]]
    require(set(exact_files + [cheap_file] + loss_files) == set(proofs), "A seed-0 input lacks provenance")
    exact = np.column_stack([np.load(path, allow_pickle=False) for path in exact_files])
    cheap = np.load(cheap_file, mmap_mode="r", allow_pickle=False)
    losses = np.stack([np.load(path, allow_pickle=False) for path in loss_files])
    require(exact.shape == (N, query_count) and cheap.shape == (N, 400) and losses.shape == (100, 401),
            "Unexpected seed-0 score or target shape")
    return exact, cheap[:, positions], losses[:, positions]


def load_seed6(spec, identity):
    expected = identity | {"freeze_commit": SEED6_FREEZE, "slurm_job_id": JOB}
    summary = LATER / "summary.json"
    summary_sha = sha(summary)
    report = read(summary)
    require(all(report[key] == value for key, value in expected.items())
            and report["exploratory_only"] is True and report["kronfluence_commit"] == spec["kronfluence_commit"],
            "Unexpected seed-6 receipt identity")
    files = [LATER / name for name in ("trajectory.npz", "ekfac.npz")]
    require(set(report["artifact_sha256"]) == {path.name for path in files}, "Unexpected seed-6 artifact set")
    proofs = {path: report["artifact_sha256"][path.name] for path in files}
    require(all(sha(path) == digest for path, digest in proofs.items()), "Changed seed-6 artifacts")
    with np.load(files[0], allow_pickle=False) as a, np.load(files[1], allow_pickle=False) as b:
        require(all(json.loads(data["provenance"].item()) == expected
                    and np.array_equal(data["query_indices"], identity["cifar_queries"]) for data in (a, b)),
                "Different seed-6 producer metadata")
        require(np.array_equal(a["mask_ids"], identity["mask_ids"])
                and b["trajectory_sha256"].item() == proofs[files[0]]
                and b["damping"].item() == spec["ekfac_damping"]
                and 0 <= a["replay_normalized_rmse"].item() < 0.02
                and 0 <= b["max_abs_export_error"].item() < identity["export_fidelity_limit"],
                "Seed-6 masks, pairing, replay, or export validation failed")
        arrays = a["exact"], b["matrix"], a["retrain_loss"]
    proofs[summary] = summary_sha
    return arrays, proofs


def assess(seed_data, design, blocks, spearman):
    k = sum(blocks)
    require(set(seed_data) == set(SEEDS) and len(blocks) == 2, "Expected both seeds and two query blocks")
    for exact, cheap, target in seed_data.values():
        require(exact.shape == cheap.shape == (design.shape[1], k) and target.shape == (design.shape[0], k)
                and all(np.isfinite(a).all() for a in (exact, cheap, target)), "Non-finite values or wrong dimensions")
    pred = [(design @ exact, design @ cheap) for exact, cheap, _ in (seed_data[s] for s in SEEDS)]
    scores = {
        f"{source}_to_{target}": {name: spearman(matrix, seed_data[target][2])
                                  for name, matrix in zip(("exact", "ekfac_1e-8"), pred[i], strict=True)}
        for i, source in enumerate(SEEDS) for target in SEEDS
    }
    truth_agreement = spearman(seed_data[0][2], seed_data[6][2])
    okay = np.isfinite(truth_agreement)
    for metrics in scores.values():
        okay &= np.isfinite(metrics["exact"]) & np.isfinite(metrics["ekfac_1e-8"])
    coverage = {"strict_status": "complete" if okay.all() else "undefined_query",
                "common_scorable_queries": int(okay.sum()), "undefined_panel_offsets": np.flatnonzero(~okay).tolist()}
    if okay.sum() < MIN_COVERAGE:
        return coverage
    indices = np.flatnonzero(okay)
    rng = np.random.default_rng(BOOT_SEED)
    samples = rng.choice(indices, size=(BOOT_DRAWS, len(indices)), replace=True)

    def statistics(values):
        return {"mean": float(values[okay].mean()),
                "ci95": np.quantile(values[samples].mean(axis=1), [0.025, 0.975]).tolist(),
                "query_block_means": [float(values[group].mean()) if len(group) else None
                                      for group in (indices[indices < blocks[0]], indices[indices >= blocks[0]])]}

    gaps = {key: values["exact"] - values["ekfac_1e-8"] for key, values in scores.items()}
    return coverage | {
        "prediction_seed_to_target_seed": {key: {**{n: statistics(v) for n, v in values.items()},
                                                  "exact_minus_ekfac": statistics(gaps[key])}
                                           for key, values in scores.items()},
        "cross_minus_within_same_target_gap": {"target_0": statistics(gaps["6_to_0"] - gaps["0_to_0"]),
                                               "target_6": statistics(gaps["0_to_6"] - gaps["6_to_6"])},
        "cross_seed_retrain_loss_rank_agreement": statistics(truth_agreement),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "score"))
    parser.add_argument("--freeze")
    args = parser.parse_args()
    if args.mode == "score":
        require(re.fullmatch(r"[0-9a-f]{40}", args.freeze or ""), "Provide the reviewed full comparison commit")
        pin_checkout(args.freeze, [HERE / name for name in ("compare.py", "PROTOCOL.md", "test_compare.py")])
        completed()
    spec, masks, identity, old_proofs = check_inputs()
    context = {"query_positions": identity["query_positions"], "cifar_queries": identity["cifar_queries"],
               "mask_ids": identity["mask_ids"], "packed_mask_sha256": identity["packed_mask_sha256"],
               "seed_6_job": JOB, "seed_6_commit": SEED6_FREEZE, "verified_seed_0_files": len(old_proofs)}
    if args.mode == "score":
        from pilot import spearman

        timestamp = datetime.now(ZoneInfo("Asia/Taipei")).isoformat(timespec="seconds")
        a = load_seed0(spec, identity, old_proofs)
        b, new_proofs = load_seed6(spec, identity)
        context |= {"comparison_commit": args.freeze, "read_started_taipei": timestamp, "exploratory": True,
                    "comparison_slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                    "inputs_sha256": {str(p): d for p, d in (old_proofs | new_proofs).items()},
                    "metrics": assess({0: a, 6: b}, -masks.astype(np.float64), (6, 6), spearman)}
        require(all(sha(path) == digest for path, digest in (old_proofs | new_proofs).items()),
                "An input changed during the comparison")
    print(json.dumps(context, ensure_ascii=False, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    main()
