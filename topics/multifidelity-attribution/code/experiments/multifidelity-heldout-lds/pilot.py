"""Evaluate fixed attribution scores against fresh, seed-0 masked retraining."""

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
from scipy.stats import rankdata


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / "killtests/multifidelity-attribution"
PROJECTION = HERE.parent / "multifidelity-complement/pilot.py"
SHARED = Path("/work/ChihChungHsu/cactus8603/icml")
RUNS = SHARED / "runs/multifid"
CIFAR = SHARED / "data/cifar10/cifar-10-batches-py"
PRODUCER = "complement/26643"
PRODUCER_FREEZE = "1432b5dffa9527677f3cce697f138bbe437c40cb"
EXACT_FREEZE = "6d8344ab0fd355dea6b80c3ccf14685c6359f1f0"
CALIBRATION_EXACT = [f"stage0/exact/Y_{start}_{start + 10}.npy" for start in range(0, 50, 10)]
CALIBRATION_CHEAP = "stage0/cheap/ekfac_1e-8.npy"
CALIBRATION_GRAM = "stage0/cheap/gram_query_grads.npy"
TRAIN_SIZE = 50_000
TARGET = list(range(436, 448))
TARGET_IDS = [8388, 3928, 7642, 9514, 5500, 2395, 2463, 705, 3077, 867, 6105, 5970]
MASK_START = 328
MASK_IDS = list(range(MASK_START, 392))
TECHNICAL_MASKS = (100, 101)
REPLAY_LIMIT = 0.02
PERMUTATION_SEED = 20261114
BOOTSTRAP_SEED = 0


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fingerprint(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_bytes())


def checked(array, shape, label):
    require(array.shape == shape and array.dtype.kind == "f" and np.isfinite(array).all(),
            f"Invalid array: {label}")
    return array


def verify_files(root, expected):
    for name, digest in expected.items():
        require(fingerprint(root / name) == digest, f"Input fingerprint differs: {name}")


def mask_plan(cfg, removal_mask):
    # Stage 0 uses [0,200); direction coverage uses [200,328).
    all_masks = np.stack([removal_mask(cfg, j, TRAIN_SIZE) for j in range(MASK_IDS[-1] + 1)])
    packed = np.packbits(all_masks, axis=1)
    prior = {row.tobytes() for row in packed[:MASK_START]}
    targets = {row.tobytes() for row in packed[MASK_START:]}
    require(len(targets) == len(MASK_IDS) and targets.isdisjoint(prior), "Mask content overlaps")
    return all_masks[MASK_START:], sha256(packed[MASK_START:].tobytes()).hexdigest()


def preflight():
    manifest_path = HERE / "inputs.json"
    manifest = read_json(manifest_path)
    verify_files(ROOT, manifest["source_sha256"])
    verify_files(CIFAR, manifest["data_sha256"])
    verify_files(RUNS, manifest["run_sha256"])
    packages = {name: version(name) for name in manifest["packages"]}
    require(packages == manifest["packages"], "Could not verify frozen JAX runtime")
    sys.path.insert(0, str(SOURCE / "scripts"))
    from common import query_indices, removal_mask_np

    cfg = read_json(SOURCE / "configs/stage0.json")
    require((cfg["train_seed"], cfg["M"], cfg["p_drop"], cfg["k_total"], cfg["k_stage0"],
             cfg["matmul_precision"]) == (0, 200, 0.01, 400, 50, "default"),
            "Historical Stage 0 recipe differs")
    pool = query_indices(dict(cfg, k_total=TARGET[-1] + 1))
    require(len(pool) == 448 and np.unique(pool).size == len(pool) and pool[TARGET].tolist() == TARGET_IDS,
            "Target query positions or IDs differ")
    masks, mask_hash = mask_plan(cfg, removal_mask_np)
    require(mask_hash == manifest["heldout_mask_sha256"], "Mask fingerprint differs")
    frozen = manifest["run_sha256"]
    receipt = read_json(RUNS / PRODUCER / "summary.json")
    upstream = receipt["identity"]
    require(upstream.get("freeze") == PRODUCER_FREEZE and upstream.get("slurm_job_id") == 26643
            and upstream.get("train_seed") == 0 and upstream.get("damping") == 1e-8
            and upstream.get("calibration_query_ids") == pool[:50].tolist()
            and upstream.get("target_query_positions") == TARGET and upstream.get("target_query_ids") == TARGET_IDS,
            "Upstream producer or panel identity differs")
    expected_exact = {"slurm_job_id": 26628, "freeze": EXACT_FREEZE, "sha256": frozen[f"{PRODUCER}/exact.npz"]}
    require(receipt["exact_source"] == expected_exact, "Upstream exact source differs")
    for name in ("exact.npz", "cheap.npz"):
        require(receipt["artifact_sha256"][name] == frozen[f"{PRODUCER}/{name}"],
                f"Upstream receipt does not corroborate {name}")
    for name in [*CALIBRATION_EXACT, CALIBRATION_CHEAP, CALIBRATION_GRAM]:
        require(upstream["input_sha256"][str(RUNS / name)] == frozen[name],
                f"Calibration receipt does not corroborate {name}")
    with np.load(RUNS / PRODUCER / "exact.npz", allow_pickle=False) as exact:
        identity = upstream | {"freeze": EXACT_FREEZE, "slurm_job_id": 26628}
        require(json.loads(exact["identity"].item()) == identity, "Exact provenance differs")
        checked(exact["exact"], (TRAIN_SIZE, len(TARGET)), "exact columns")
    with np.load(RUNS / PRODUCER / "cheap.npz", allow_pickle=False) as cheap:
        require(json.loads(cheap["identity"].item()) == upstream
                and cheap["exact_sha256"].item() == expected_exact["sha256"], "EK-FAC provenance differs")
        checked(cheap["cheap"], (TRAIN_SIZE, len(TARGET)), "EK-FAC columns")
        checked(cheap["cross_gram"], (50, len(TARGET)), "cross Gram")
    for mask_id in TECHNICAL_MASKS:
        old = np.load(RUNS / f"stage0/retrain/loss_{mask_id}.npy", allow_pickle=False)
        checked(old, (401,), "historical technical replay losses")
    identity = {
        "query_positions": TARGET,
        "query_ids": TARGET_IDS,
        "train_seed": cfg["train_seed"],
        "mask_ids": MASK_IDS,
        "packed_mask_sha256": mask_hash,
        "inputs_manifest_sha256": fingerprint(manifest_path),
        "producer_summary_sha256": frozen[f"{PRODUCER}/summary.json"],
        "jax_packages": packages,
    }
    return cfg, pool, masks, identity


def publish(path, writer):
    intermediate = path.with_suffix(".partial")
    with intermediate.open("xb") as stream:
        writer(stream)
    os.link(intermediate, path)
    intermediate.unlink()


def replay_error(observed, reference):
    observed, reference = observed.astype(np.float64), reference.astype(np.float64)
    return float(np.sqrt(np.square(observed - reference).mean()) / max(reference.std(), 0.1))


def retrain(run, cfg, pool, masks, identity):
    import jax
    import jax.numpy as jnp
    from mf_core import ResNet9, Trainer, load_cifar, removal_mask

    device, = jax.devices()
    require(device.platform == "gpu" and "H200" in device.device_kind, "Expected one H200")
    x_train, y_train, x_test, y_test = load_cifar(cfg["n_train"])
    trainer = Trainer(ResNet9(cfg["width"]), x_train, y_train, cfg["hp"])
    old_queries = pool[:cfg["k_total"]]
    old_x, old_y = jnp.asarray(x_test[old_queries]), jnp.asarray(y_test[old_queries])
    full_x, full_y = jnp.asarray(x_test[pool]), jnp.asarray(y_test[pool])
    errors = {}
    for mask_id in TECHNICAL_MASKS:
        started = monotonic()
        weights = jnp.asarray(~removal_mask(cfg, mask_id, TRAIN_SIZE), dtype=jnp.float32)
        state = trainer.train(weights, cfg["train_seed"])
        observed = checked(np.asarray(trainer.query_losses(state, old_x, old_y)), (400,), "old panel replay")
        del state
        reference = np.load(RUNS / f"stage0/retrain/loss_{mask_id}.npy", allow_pickle=False)[:400]
        error = replay_error(observed, reference)
        require(error < REPLAY_LIMIT, f"Historical replay {mask_id} exceeded normalized RMSE {REPLAY_LIMIT}")
        errors[str(mask_id)] = error
        print(f"[codex] Historical mask {mask_id}: {monotonic() - started:.1f}s, normalized RMSE {error:.9g}.",
              flush=True)

    losses = []
    for mask_id, removed in zip(MASK_IDS, masks, strict=True):
        started = monotonic()
        weights = jnp.asarray(~removed, dtype=jnp.float32)
        state = trainer.train(weights, cfg["train_seed"])
        values = np.asarray(trainer.query_losses(state, full_x, full_y)[TARGET[0]:], dtype=np.float32)
        del state
        losses.append(checked(values, (len(TARGET),), "fresh target loss"))
        print(f"[codex] Fresh mask {mask_id}: {monotonic() - started:.1f}s.", flush=True)

    publish(run / "retrain.npz", lambda stream: np.savez(
        stream, losses=np.stack(losses), mask_ids=MASK_IDS, query_ids=TARGET_IDS,
        technical_replay=json.dumps(errors, sort_keys=True), identity=json.dumps(identity, sort_keys=True)))


def attribution_arms(ycal, ccal, exact, cheap, u, d, permutation, fit_alpha):
    spell = (ycal @ u) @ d
    residual = cheap - (ccal @ u) @ d
    alpha = fit_alpha(ycal, ccal, u)
    alpha_permuted = fit_alpha(ycal, ccal[permutation], u)
    return {
        "spell": spell,
        "ekfac": cheap,
        "hybrid": spell + alpha * residual,
        "perm_ekfac": cheap[permutation],
        "perm_hybrid": spell + alpha_permuted * residual[permutation],
        "exact": exact,
    }, {"alpha": alpha, "permuted_alpha": alpha_permuted}


def column_spearman(predicted, truth):
    ranks = [rankdata(values, axis=0) for values in (predicted, truth)]
    left, right = (rank - rank.mean(axis=0) for rank in ranks)
    denominator = np.linalg.norm(left, axis=0) * np.linalg.norm(right, axis=0)
    return np.divide((left * right).sum(axis=0), denominator,
                     out=np.full_like(denominator, np.nan), where=denominator > 0)


def gains(means):
    return {
        "hybrid_minus_spell": means["hybrid"] - means["spell"],
        "hybrid_minus_ekfac": means["hybrid"] - means["ekfac"],
        "hybrid_minus_stronger": means["hybrid"] - np.maximum(means["spell"], means["ekfac"]),
        "permutation_gain": means["perm_hybrid"] - np.maximum(means["spell"], means["perm_ekfac"]),
        "exact_minus_stronger": means["exact"] - np.maximum(means["spell"], means["ekfac"]),
    }


def paired_summary(scores):
    valid = np.logical_and.reduce([np.isfinite(values) for values in scores.values()])
    summary = {
        "scorable_queries": int(valid.sum()),
        "undefined_query_positions": np.asarray(TARGET)[~valid].tolist(),
        "complete_fixed_panel": bool(valid.all()),
    }
    if not valid.all():
        return summary
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(valid.size, size=(10_000, valid.size))
    means = {name: float(values.mean()) for name, values in scores.items()}
    mean_gains = {key: float(value) for key, value in gains(means).items()}
    boot_gains = gains({name: values[draws].mean(axis=1) for name, values in scores.items()})
    return summary | {
        "lds": means,
        "gains": mean_gains,
        "descriptive_95pct_intervals": {name: np.quantile(values, [0.025, 0.975]).tolist()
                                        for name, values in boot_gains.items()},
        "hybrid_gain_at_least_0_04_reference": mean_gains["hybrid_minus_stronger"] >= 0.04,
        "permutation_gain_at_most_0_01_reference": mean_gains["permutation_gain"] <= 0.01,
        "six_query_block_lds": [{name: float(values[block].mean()) for name, values in scores.items()}
                                for block in (slice(0, 6), slice(6, 12))],
    }


def score(run, masks, identity):
    from hybrid import fit_alpha

    spec = spec_from_file_location("frozen_complement_projection", PROJECTION)
    projection = module_from_spec(spec)
    spec.loader.exec_module(projection)
    retrain_path = run / "retrain.npz"
    retrain_hash = fingerprint(retrain_path)
    with np.load(retrain_path, allow_pickle=False) as losses:
        require(json.loads(losses["identity"].item()) == identity and losses["mask_ids"].tolist() == MASK_IDS
                and losses["query_ids"].tolist() == TARGET_IDS, "Retrain panel or producer differs")
        truth = checked(losses["losses"], (len(MASK_IDS), len(TARGET)), "masked losses")
        errors = json.loads(losses["technical_replay"].item())
        require(set(errors) == {str(j) for j in TECHNICAL_MASKS}
                and all(0 <= error < REPLAY_LIMIT for error in errors.values()), "Technical replay receipt differs")

    def calibration(name, columns=None):
        array = np.load(RUNS / name, mmap_mode="r", allow_pickle=False)
        return np.asarray(array if columns is None else array[columns], dtype=np.float64)

    ycal = np.column_stack([calibration(name) for name in CALIBRATION_EXACT])
    ccal = calibration(CALIBRATION_CHEAP, np.s_[:, :50])
    gram = calibration(CALIBRATION_GRAM, np.s_[:50, :50])
    with np.load(RUNS / PRODUCER / "exact.npz", allow_pickle=False) as exact, \
         np.load(RUNS / PRODUCER / "cheap.npz", allow_pickle=False) as cheap:
        u, d = projection.coordinates(gram, cheap["cross_gram"], budget=10)
        order = np.random.default_rng(PERMUTATION_SEED).permutation(TRAIN_SIZE)
        arms, coefficients = attribution_arms(ycal, ccal, exact["exact"].astype(np.float64),
                                              cheap["cheap"].astype(np.float64), u, d, order, fit_alpha)
    design = -masks.astype(np.float64)
    correlations = {name: column_spearman(design @ matrix, truth) for name, matrix in arms.items()}
    require(fingerprint(retrain_path) == retrain_hash, "Retrain receipt was modified")
    result = {
        "identity": identity,
        "retrain_sha256": retrain_hash,
        "technical_replay_normalized_rmse": errors,
        "coefficients": coefficients,
        "metrics": paired_summary(correlations),
        "diagnostic_only": True,
        "completed_at_taipei": datetime.now(ZoneInfo("Asia/Taipei")).isoformat(timespec="seconds"),
    }
    data = json.dumps(result, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    publish(run / "summary.json", lambda stream: stream.write(data))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("preflight", "retrain", "score"))
    parser.add_argument("--freeze")
    parser.add_argument("--run", type=Path)
    args = parser.parse_args()
    cfg, pool, masks, identity = preflight()
    if args.phase == "preflight":
        print(json.dumps(identity, sort_keys=True))
        return
    job = os.environ.get("SLURM_JOB_ID", "")
    require(re.fullmatch(r"[1-9][0-9]*", job) and re.fullmatch(r"[a-f0-9]{40}", args.freeze or "")
            and args.run == RUNS / "heldout-lds" / job, "Use the frozen Slurm launcher")
    identity.update(freeze=args.freeze, slurm_job_id=int(job))
    if args.phase == "retrain":
        retrain(args.run, cfg, pool, masks, identity)
    else:
        score(args.run, masks, identity)


if __name__ == "__main__":
    main()
