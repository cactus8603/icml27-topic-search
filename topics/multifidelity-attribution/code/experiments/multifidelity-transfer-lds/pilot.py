"""Compare frozen seed-0 attributions against seed-0 and seed-7 retraining."""

import argparse
from datetime import datetime
from hashlib import file_digest, sha256
from importlib.metadata import version
from importlib.util import module_from_spec, spec_from_file_location
import json
import os
from pathlib import Path
import re
import sys
from time import monotonic
from zoneinfo import ZoneInfo

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / "killtests/multifidelity-attribution"
SHARED = Path("/work/ChihChungHsu/cactus8603/icml")
RUNS = SHARED / "runs/multifid"
CIFAR = SHARED / "data/cifar10/cifar-10-batches-py"
HELPER = HERE.parent / "multifidelity-heldout-lds/pilot.py"
MANIFEST = HERE / "inputs.json"
INPUTS = json.loads(MANIFEST.read_bytes())

TARGET_SEEDS = [0, 7]
TRAIN_SIZE = 50_000
QUERIES = 50
FORWARD_QUERIES = 400
MASK_START = 392
MASK_IDS = list(range(MASK_START, 456))
TECHNICAL_MASKS = [100, 101]
REPLAY_LIMIT = 0.02
TRANSFER_REFERENCE = 0.05
BOOTSTRAP_SEED = 0
BOOTSTRAP_DRAWS = 10_000
EXACT_FILES = [f"stage0/exact/Y_{start}_{start + 10}.npy" for start in range(0, QUERIES, 10)]
EKFAC_FILE = "stage0/cheap/ekfac_1e-8.npy"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_helpers():
    with HELPER.open("rb") as stream:
        digest = file_digest(stream, "sha256").hexdigest()
    require(digest == INPUTS["source_sha256"][HELPER.relative_to(ROOT).as_posix()],
            "Frozen held-out LDS helper differs")
    spec = spec_from_file_location("frozen_heldout_lds", HELPER)
    helpers = module_from_spec(spec)
    spec.loader.exec_module(helpers)
    return helpers


frozen = load_helpers()


def mask_plan(cfg, removal_mask):
    masks = np.stack([removal_mask(cfg, j, TRAIN_SIZE) for j in range(MASK_IDS[-1] + 1)])
    packed = np.packbits(masks, axis=1)
    historical = {mask.tobytes() for mask in packed[:MASK_START]}
    current = {mask.tobytes() for mask in packed[MASK_START:]}
    require(len(historical) == MASK_START and len(current) == len(MASK_IDS)
            and current.isdisjoint(historical), "Mask content overlaps")
    digest = sha256(packed[MASK_START:].tobytes()).hexdigest()
    require(digest == INPUTS["heldout_mask_sha256"], "Mask fingerprint differs")
    return masks[MASK_START:], digest


def preflight():
    frozen.verify_files(ROOT, INPUTS["source_sha256"])
    frozen.verify_files(CIFAR, INPUTS["data_sha256"])
    frozen.verify_files(RUNS, INPUTS["run_sha256"])
    packages = {name: version(name) for name in INPUTS["packages"]}
    require(packages == INPUTS["packages"], "Frozen package versions differ")
    sys.path.insert(0, str(SOURCE / "scripts"))
    from common import query_indices, removal_mask_np

    cfg = frozen.read_json(SOURCE / "configs/stage0.json")
    require((cfg["train_seed"], cfg["ensemble_seeds"], cfg["crossseed_train_seed"],
             cfg["k_total"], cfg["k_stage0"], cfg["mask_seed"], cfg["p_drop"],
             cfg["n_train"], cfg["matmul_precision"])
            == (0, [1, 2, 3, 4], 5, FORWARD_QUERIES, QUERIES, 1234, 0.01, None, "default"),
            "Historical Stage 0 configuration differs")
    pool = query_indices(cfg)
    require(len(pool) == FORWARD_QUERIES and np.unique(pool).size == FORWARD_QUERIES,
            "Expected distinct source query IDs")
    query_bytes = pool[:QUERIES].astype("<i8").tobytes()
    require(sha256(query_bytes).hexdigest() == INPUTS["query_ids_i64_sha256"],
            "Source and target query IDs differ")
    masks, mask_hash = mask_plan(cfg, removal_mask_np)
    identity = {
        "predictor_seed": cfg["train_seed"],
        "target_seeds": TARGET_SEEDS,
        "query_positions": list(range(QUERIES)),
        "query_ids": pool[:QUERIES].tolist(),
        "forward_query_pool_sha256": sha256(pool.astype("<i8").tobytes()).hexdigest(),
        "mask_ids": MASK_IDS,
        "packed_mask_sha256": mask_hash,
        "inputs_manifest_sha256": frozen.fingerprint(MANIFEST),
        "packages": packages,
    }
    return cfg, pool, masks, identity


def taipei_now():
    return datetime.now(ZoneInfo("Asia/Taipei")).isoformat(timespec="seconds")


def publish_json(path, value):
    data = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    frozen.publish(path, lambda stream: stream.write(data))


def technical_gate(errors):
    require(set(errors) == {str(mask_id) for mask_id in TECHNICAL_MASKS}
            and all(0 <= error < REPLAY_LIMIT for error in errors.values()),
            "Historical technical replay receipt differs")


def retrain(run, cfg, pool, masks, identity):
    import jax
    import jax.numpy as jnp
    from mf_core import ResNet9, Trainer, load_cifar, removal_mask

    require(os.environ.get("CIFAR_DIR") == str(CIFAR), "CIFAR path differs from verified inputs")
    device, = jax.devices()
    require(device.platform == "gpu" and "H200" in device.device_kind, "Expected one H200")
    x_train, y_train, x_test, y_test = load_cifar(cfg["n_train"])
    require(len(x_train) == TRAIN_SIZE, "Training set size differs")
    trainer = Trainer(ResNet9(cfg["width"]), x_train, y_train, cfg["hp"])
    x_query, y_query = jnp.asarray(x_test[pool]), jnp.asarray(y_test[pool])

    def query_losses(removed, seed):
        weights = jnp.asarray(~removed, dtype=jnp.float32)
        state = trainer.train(weights, seed)
        values = np.asarray(trainer.query_losses(state, x_query, y_query), dtype=np.float32)
        return frozen.checked(values, (FORWARD_QUERIES,), "retrained query losses")

    errors = {}
    for mask_id in TECHNICAL_MASKS:
        started = monotonic()
        observed = query_losses(removal_mask(cfg, mask_id, TRAIN_SIZE), cfg["train_seed"])
        path = RUNS / f"stage0/retrain/loss_{mask_id}.npy"
        old = frozen.checked(np.load(path, allow_pickle=False), (FORWARD_QUERIES + 1,), path.name)
        error = frozen.replay_error(observed, old[:FORWARD_QUERIES])
        require(error < REPLAY_LIMIT, f"Historical replay {mask_id} exceeded normalized RMSE {REPLAY_LIMIT}")
        errors[str(mask_id)] = error
        print(f"[codex] Historical mask {mask_id}: {monotonic() - started:.1f}s, normalized RMSE {error:.9g}.",
              flush=True)
    technical_gate(errors)

    losses = np.empty((len(TARGET_SEEDS), len(MASK_IDS), QUERIES), dtype=np.float32)
    for seed_idx, seed in enumerate(TARGET_SEEDS):
        for mask_idx, (mask_id, removed) in enumerate(zip(MASK_IDS, masks, strict=True)):
            started = monotonic()
            losses[seed_idx, mask_idx] = query_losses(removed, seed)[:QUERIES]
            print(f"[codex] Seed {seed}, fresh mask {mask_id}: {monotonic() - started:.1f}s.", flush=True)

    artifact = run / "retrain.npz"
    frozen.publish(artifact, lambda stream: np.savez(
        stream, losses=losses, target_seeds=TARGET_SEEDS, mask_ids=MASK_IDS,
        query_ids=identity["query_ids"], identity=json.dumps(identity, sort_keys=True)))
    publish_json(run / "retrain-receipt.json", {
        "schema": 1,
        "identity": identity,
        "retrain_sha256": frozen.fingerprint(artifact),
        "technical_replay_normalized_rmse": errors,
        "device_kind": device.device_kind,
        "completed_at_taipei": taipei_now(),
    })


def checked_retrain(run, identity):
    receipt_path = run / "retrain-receipt.json"
    receipt_hash = frozen.fingerprint(receipt_path)
    receipt = frozen.read_json(receipt_path)
    require(receipt.get("schema") == 1 and receipt.get("identity") == identity
            and "H200" in receipt.get("device_kind", ""), "Retrain producer or device differs")
    technical_gate(receipt["technical_replay_normalized_rmse"])
    path = run / "retrain.npz"
    digest = frozen.fingerprint(path)
    require(receipt.get("retrain_sha256") == digest, "Retraining artifact fingerprint differs")
    with np.load(path, allow_pickle=False) as artifact:
        require(json.loads(artifact["identity"].item()) == identity, "Retraining artifact identity differs")
        for field in ("mask_ids", "query_ids", "target_seeds"):
            ids = artifact[field]
            require(ids.dtype.kind == "i" and ids.tolist() == identity[field], f"Retraining {field} differ")
        loss = artifact["losses"]
        require(loss.dtype == np.float32, "Expected float32 masked losses")
        frozen.checked(loss, (len(TARGET_SEEDS), len(MASK_IDS), QUERIES), "masked losses")
    require(frozen.fingerprint(path) == digest and frozen.fingerprint(receipt_path) == receipt_hash,
            "Retrain input changed while reading")
    return loss, receipt, receipt_hash


def predict(masks, exact, ekfac):
    design = -masks.astype(np.float64)
    return {"exact": design @ np.asarray(exact, dtype=np.float64),
            "ekfac": design @ np.asarray(ekfac, dtype=np.float64)}


def paired_summary(predictions, truth):
    scores = {
        f"{method}_0_to_{seed}": frozen.column_spearman(values, target)
        for seed, target in zip(TARGET_SEEDS, truth, strict=True)
        for method, values in predictions.items()
    }
    scores["target_truth_rank_agreement"] = frozen.column_spearman(truth[0], truth[1])
    valid = np.logical_and.reduce([np.isfinite(values) for values in scores.values()])
    summary = {
        "comparison": "exact_minus_ekfac_0_to_7",
        "transfer_reference": TRANSFER_REFERENCE,
        "scorable_queries": int(valid.sum()),
        "undefined_query_positions": np.flatnonzero(~valid).tolist(),
    }
    if not valid.all():
        return summary | {"diagnostic_status": "untestable_incomplete_fixed_panel"}

    scores["exact_minus_ekfac_0_to_0"] = scores["exact_0_to_0"] - scores["ekfac_0_to_0"]
    scores["exact_minus_ekfac_0_to_7"] = scores["exact_0_to_7"] - scores["ekfac_0_to_7"]
    scores["cross_minus_within_advantage"] = scores["exact_minus_ekfac_0_to_7"] - scores["exact_minus_ekfac_0_to_0"]
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(valid.size, size=(BOOTSTRAP_DRAWS, valid.size))
    means = {name: float(values.mean()) for name, values in scores.items()}
    intervals = {name: np.quantile(values[draws].mean(axis=1), [0.025, 0.975]).tolist()
                 for name, values in scores.items()}
    primary = summary["comparison"]
    lower, upper = intervals[primary]
    status = "inconclusive"
    if means[primary] >= TRANSFER_REFERENCE and lower > 0:
        status = "positive_transfer_at_reference"
    elif upper < TRANSFER_REFERENCE:
        status = "below_reference"
    return summary | {
        "diagnostic_status": status,
        "means": means,
        "descriptive_95pct_intervals": intervals,
    }


def score(run, masks, identity):
    truth, receipt, receipt_hash = checked_retrain(run, identity)
    exact = np.column_stack([
        frozen.checked(np.load(RUNS / name, allow_pickle=False), (TRAIN_SIZE, 10), name)
        for name in EXACT_FILES
    ])
    ekfac = np.load(RUNS / EKFAC_FILE, mmap_mode="r", allow_pickle=False)
    require(ekfac.shape == (TRAIN_SIZE, FORWARD_QUERIES), "Frozen EK-FAC shape differs")
    source_ekfac = frozen.checked(ekfac[:, :QUERIES], (TRAIN_SIZE, QUERIES), EKFAC_FILE)
    predictions = predict(masks, exact, source_ekfac)
    require(all(np.isfinite(values).all() for values in predictions.values()), "Invalid mask predictions")
    frozen.verify_files(RUNS, {name: INPUTS["run_sha256"][name] for name in [*EXACT_FILES, EKFAC_FILE]})
    publish_json(run / "summary.json", {
        "schema": 1,
        "identity": identity,
        "retrain_receipt_sha256": receipt_hash,
        "retrain_sha256": receipt["retrain_sha256"],
        "technical_replay_normalized_rmse": receipt["technical_replay_normalized_rmse"],
        "metrics": paired_summary(predictions, truth),
        "diagnostic_only": True,
        "completed_at_taipei": taipei_now(),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("preflight", "retrain", "score"))
    parser.add_argument("--freeze")
    parser.add_argument("--run", type=Path)
    args = parser.parse_args()
    cfg, pool, masks, identity = preflight()
    if args.phase == "preflight":
        print("[codex] " + json.dumps(identity, sort_keys=True))
        return
    job = os.environ.get("SLURM_JOB_ID", "")
    require(re.fullmatch(r"[1-9][0-9]*", job) and re.fullmatch(r"[a-f0-9]{40}", args.freeze or "")
            and args.run == RUNS / "transfer-lds" / job, "Use the frozen Slurm launcher")
    identity.update(freeze=args.freeze, slurm_job_id=int(job))
    if args.phase == "retrain":
        retrain(args.run, cfg, pool, masks, identity)
    else:
        score(args.run, masks, identity)


if __name__ == "__main__":
    main()
