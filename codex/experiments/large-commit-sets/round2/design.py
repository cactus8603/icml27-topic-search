"""Frozen splits, probe coverage design and provenance shared by both modes."""

import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
sys.path.insert(0, str(PARENT))
import pilot

CONFIG_PATH = HERE / "config.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_order(indices, seed):
    return sorted(indices, key=lambda index: hashlib.sha256(f"{seed}:{index}".encode()).digest())


def normalize(text):
    return " ".join(text.casefold().split())


def read_rows(parquet, python, expected):
    if sha(parquet) != expected["sha256"]:
        raise ValueError(f"GSM8K file hash mismatch: {parquet}")
    output = subprocess.check_output([str(python), "-B", str(PARENT / "read_parquet.py"), str(parquet)])
    rows = json.loads(output)
    if len(rows) != expected["row_count"]:
        raise ValueError(f"GSM8K file row count mismatch: {parquet}")
    return rows


def split_ids(train, test, config, round1):
    seed = config["split_seed"]
    original_test = stable_order(range(len(test)), round1["main"]["sample_seed"])[
        : round1["main"]["sample_size"]
    ]
    test_keys = [normalize(row["question"]) for row in test]
    all_test_keys = set(test_keys)
    used = {test_keys[i] for i in original_test}
    smoke_blocklist = set(config["excluded_train_doc_ids"])
    train_keys = [normalize(row["question"]) for row in train]
    used_train = {train_keys[i] for i in smoke_blocklist}

    def deduplicated(order, keys, prohibited):
        accepted = []
        for index in order:
            key = keys[index]
            if key not in prohibited:
                prohibited.add(key)
                accepted.append(index)
        return accepted

    test_indices = deduplicated(stable_order(range(len(test)), seed + ":test"), test_keys, used)
    train_pool = deduplicated(
        stable_order(range(len(train)), seed + ":train"), train_keys, all_test_keys | used_train
    )
    smoke_count = config["smoke_sample_size"]
    splits = {
        "smoke": train_pool[:smoke_count],
        "train": train_pool[smoke_count : smoke_count + config["train_sample_size"]],
        "test": test_indices[: config["test_sample_size"]],
    }
    counts = {mode: len(ids) for mode, ids in splits.items()}
    expected = {mode: config[f"{mode}_sample_size"] for mode in splits}
    if counts != expected:
        raise RuntimeError(f"Insufficient disjoint questions: {counts!r}")
    return splits


def load_design(train_path, test_path, parquet_python):
    config = json.loads(CONFIG_PATH.read_text())
    round1 = json.loads(pilot.CONFIG_PATH.read_text())
    train = read_rows(train_path, parquet_python, round1["smoke"])
    test = read_rows(test_path, parquet_python, round1["main"])
    splits = split_ids(train, test, config, round1)
    provenance = {
        "config_sha256": sha(CONFIG_PATH),
        "parent_config_sha256": sha(pilot.CONFIG_PATH),
        "train_sha256": round1["smoke"]["sha256"],
        "test_sha256": round1["main"]["sha256"],
        "checkpoint_revision": round1["checkpoint"]["revision"],
        "freeze_commit": os.environ["PILOT_FREEZE_COMMIT"],
    }
    return config, round1, {"smoke": train, "train": train, "test": test}, splits, provenance


def joint_rows(pools):
    """Merge identical reveal inputs. Each output target remains masked."""
    rows = defaultdict(set)
    for members in pools:
        for reveal, target in pilot.joint_holdouts(members):
            rows[reveal].add(target)
    return {reveal: sorted(targets) for reveal, targets in sorted(rows.items())}


def doubleton_rows(pools, budget):
    """Spend each probe on unique reachable within-candidate interactions."""
    possible = defaultdict(set)
    for cid, members in enumerate(pools):
        for reveal in combinations(members, 2):
            possible[reveal].update((cid, target) for target in members if target not in reveal)
    counts = defaultdict(int)
    rows = {}
    for _ in range(budget):
        if not possible:
            raise ValueError("Not enough reachable doubletons for the frozen probe budget.")
        # Penalize previously covered candidate-target combinations, then
        # prefer coverage across more candidates and shorter reveal distance.
        reveal = min(
            possible,
            key=lambda pair: (
                -sum(1 / (1 + counts[target]) for target in sorted(possible[pair])),
                pair[1] - pair[0],
                pair,
            ),
        )
        coverage = possible.pop(reveal)
        rows[reveal] = sorted({target for _, target in coverage})
        for target in coverage:
            counts[target] += 1
    if any(not counts[cid, target] for cid, members in enumerate(pools) for target in members):
        raise RuntimeError("Doubleton budget left a candidate-target without pair coverage.")
    return rows


def counts_and_path(path, mode):
    result_path = results_directory()
    if path.resolve().parent != result_path or path.suffix != ".jsonl":
        raise ValueError(f"{mode} needs a fresh JSONL file directly under results/.")


def results_directory():
    return Path(os.environ.get("PILOT_RESULTS_PATH", HERE / "results")).resolve()


def verify_shards(model_path):
    """Check the staged index and exact shard bytes against public LFS OIDs."""
    smaller_files = json.loads((HERE / "artifacts.json").read_text())
    for filename, expected in smaller_files["support_sha256"].items():
        if sha(model_path / filename) != expected:
            raise ValueError(f"Snapshot file differs: {filename}")
    index = json.loads((model_path / "model.safetensors.index.json").read_text())
    if set(index["weight_map"].values()) != set(smaller_files["lfs_sha256"]):
        raise ValueError("The safetensors index differs from this frozen snapshot.")
    for filename, expected in smaller_files["lfs_sha256"].items():
        with (model_path / filename).open("rb") as shard:
            actual = hashlib.file_digest(shard, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"Checkpoint shard differs: {filename}")
