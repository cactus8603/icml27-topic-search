"""Evaluate the frozen seed-0 predictor panel on independent removal masks."""

import argparse
import json
import os
from datetime import datetime
from hashlib import sha256
from importlib.metadata import version
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
SHARED = Path("/work/ChihChungHsu/cactus8603/icml")
RUNS = SHARED / "runs/multifid"
CIFAR = SHARED / "data/cifar10/cifar-10-batches-py"
MANIFEST = HERE / "inputs.json"

sys.path[:0] = [str(SOURCE / "scripts"), str(HERE.parent / "multifidelity-heldout-lds"),
               str(HERE.parent / "multifidelity-signal-confirm")]
from common import query_indices, removal_mask_np
from hybrid import hybrid, select_U, spell
from pilot import checked, fingerprint, publish, read_json, replay_error, require, verify_files
from score import HISTORY_SHA, verify_inputs as verify_prior_signal_inputs


PANEL = slice(50, 100)
QUERIES = PANEL.stop - PANEL.start
TRAIN_SIZE = 50_000
FORWARD_QUERIES = 400
MASK_START = 456
MASK_IDS = list(range(MASK_START, 520))
TECHNICAL_MASKS = (100, 101)
REPLAY_LIMIT = 0.02
PRODUCER = "signal-confirm/26734"
PRODUCER_FREEZE = "09661dc26c24ea06623bae30fe6fdc80692dd0aa"
ARMS = ("exact", "ekfac", "spell", "hybrid", "ekfac_permuted", "hybrid_permuted")
REFERENCE_GAIN = 0.04
CONTROL_MAX_GAIN = 0.01
BOOTSTRAP_SEED = 0
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_BATCH = 64


def packed_mask_plan(masks, start):
    packed = np.packbits(masks, axis=1)
    historical = {row.tobytes() for row in packed[:start]}
    current = {row.tobytes() for row in packed[start:]}
    require(len(historical) == start and len(current) == len(masks) - start
            and current.isdisjoint(historical), "Mask content overlaps")
    return packed


def preflight(phase):
    inputs = read_json(MANIFEST)
    verify_files(ROOT, inputs["source_sha256"])
    verify_files(CIFAR, inputs["data_sha256"])
    verify_files(RUNS, inputs["run_sha256"])
    packages = inputs["gpu_packages" if phase == "retrain" else "cpu_packages"]
    require(all(version(name) == expected for name, expected in packages.items()),
            "Frozen numerical runtime differs")
    cfg = read_json(SOURCE / "configs/stage0.json")
    require((cfg["train_seed"], cfg["ensemble_seeds"], cfg["crossseed_train_seed"],
             cfg["k_total"], cfg["k_stage0"], cfg["mask_seed"], cfg["p_drop"],
             cfg["n_train"], cfg["matmul_precision"])
            == (0, [1, 2, 3, 4], 5, FORWARD_QUERIES, QUERIES, 1234, 0.01, None, "default"),
            "Historical Stage 0 configuration differs")
    pool = query_indices(cfg)
    ids = pool[PANEL].tolist()
    query_sha = sha256(pool[PANEL].astype("<i8").tobytes()).hexdigest()
    require(len(pool) == len(set(pool)) == FORWARD_QUERIES and query_sha == inputs["query_ids_i64_sha256"],
            "Source query identities differ")
    prior = read_json(RUNS / PRODUCER / "summary.json")
    require(prior["freeze"] == PRODUCER_FREEZE and prior["query_positions"] == [PANEL.start, PANEL.stop]
            and prior["cifar_ids"] == ids and prior["source_sha256"] == HISTORY_SHA
            and prior["signal"]["status"] == "PASS", "26734 summary lineage differs")
    producer_cfg = read_json(RUNS / PRODUCER / "config.json")
    require(producer_cfg == (cfg | {"run_dir": str(RUNS / PRODUCER), "tag": "signal-confirm"}),
            "26734 configuration differs from the frozen source configuration")

    masks = np.stack([removal_mask_np(cfg, j, TRAIN_SIZE) for j in range(MASK_IDS[-1] + 1)])
    packed = packed_mask_plan(masks, MASK_START)
    for field, section in {
        "used_mask_sha256": packed[:MASK_START],
        "heldout_mask_sha256": packed[MASK_START:],
        "previous_transfer_mask_sha256": packed[392:456],
    }.items():
        require(sha256(section.tobytes()).hexdigest() == inputs[field], f"Mask fingerprint differs: {field}")
    identity = {
        "target_seed": cfg["train_seed"],
        "query_positions": [PANEL.start, PANEL.stop],
        "query_ids": ids,
        "forward_query_pool_sha256": sha256(pool.astype("<i8").tobytes()).hexdigest(),
        "mask_ids": MASK_IDS,
        "packed_mask_sha256": inputs["heldout_mask_sha256"],
        "inputs_manifest_sha256": fingerprint(MANIFEST),
        "predictor_job": 26734,
        "predictor_freeze": PRODUCER_FREEZE,
        "predictor_summary_sha256": inputs["run_sha256"][f"{PRODUCER}/summary.json"],
    }
    return cfg, pool, masks, identity, prior


def publish_json(path, value):
    data = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    publish(path, lambda stream: stream.write(data))


def taipei_now():
    return datetime.now(ZoneInfo("Asia/Taipei")).isoformat(timespec="seconds")


def technical_gate(errors):
    require(set(errors) == {str(mask_id) for mask_id in TECHNICAL_MASKS}
            and all(0 <= error < REPLAY_LIMIT for error in errors.values()),
            "Historical technical replay receipt differs")


def retrain(run, cfg, pool, masks, identity):
    # Preserve the 26703 training and full forward pool. Save only the 26734 query slice.
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

    def query_losses(removed):
        weights = jnp.asarray(~removed, dtype=jnp.float32)
        state = trainer.train(weights, cfg["train_seed"])
        values = np.asarray(trainer.query_losses(state, x_query, y_query), dtype=np.float32)
        return checked(values, (FORWARD_QUERIES,), "retrained query losses")

    errors = {}
    for mask_id in TECHNICAL_MASKS:
        started = monotonic()
        legacy_mask = removal_mask(cfg, mask_id, TRAIN_SIZE)
        require(np.array_equal(legacy_mask, removal_mask_np(cfg, mask_id, TRAIN_SIZE)),
                "GPU-side and archived mask generation differ")
        observed = query_losses(legacy_mask)
        old = load_array(RUNS / f"stage0/retrain/loss_{mask_id}.npy", (FORWARD_QUERIES + 1,))
        error = replay_error(observed, old[:FORWARD_QUERIES])
        require(0 <= error < REPLAY_LIMIT, f"Historical replay {mask_id} exceeded normalized RMSE {REPLAY_LIMIT}")
        errors[str(mask_id)] = error
        print(f"[codex] Historical mask {mask_id}: {monotonic() - started:.1f}s, normalized RMSE {error:.9g}.",
              flush=True)
    technical_gate(errors)
    losses = np.empty((len(MASK_IDS), QUERIES), dtype=np.float32)
    for index, (mask_id, removed) in enumerate(zip(MASK_IDS, masks, strict=True)):
        started = monotonic()
        losses[index] = query_losses(removed)[PANEL]
        print(f"[codex] Seed 0, fresh mask {mask_id}: {monotonic() - started:.1f}s.", flush=True)

    artifact = run / "retrain.npz"
    publish(artifact, lambda stream: np.savez(
        stream, losses=losses, mask_ids=MASK_IDS, query_ids=identity["query_ids"],
        identity=json.dumps(identity, sort_keys=True)))
    publish_json(run / "retrain-receipt.json", {
        "schema": 1, "identity": identity, "retrain_sha256": fingerprint(artifact),
        "technical_replay_normalized_rmse": errors, "device_kind": device.device_kind,
        "gpu_packages": read_json(MANIFEST)["gpu_packages"], "completed_at_taipei": taipei_now(),
    })


def checked_retrain(run, identity):
    receipt_path = run / "retrain-receipt.json"
    receipt_hash = fingerprint(receipt_path)
    receipt = read_json(receipt_path)
    require(receipt.get("schema") == 1 and receipt.get("identity") == identity
            and "H200" in receipt.get("device_kind", "")
            and receipt.get("gpu_packages") == read_json(MANIFEST)["gpu_packages"],
            "Retrain producer, device or runtime differs")
    technical_gate(receipt["technical_replay_normalized_rmse"])
    path = run / "retrain.npz"
    digest = fingerprint(path)
    require(receipt.get("retrain_sha256") == digest, "Retraining artifact fingerprint differs")
    with np.load(path, allow_pickle=False) as artifact:
        require(json.loads(artifact["identity"].item()) == identity, "Retraining artifact identity differs")
        for field in ("mask_ids", "query_ids"):
            ids = artifact[field]
            require(ids.dtype.kind == "i" and ids.tolist() == identity[field], f"Retraining {field} differ")
        loss = artifact["losses"]
        require(loss.dtype == np.float32, "Expected float32 masked losses")
        checked(loss, (len(MASK_IDS), QUERIES), "masked losses")
    require(fingerprint(path) == digest and fingerprint(receipt_path) == receipt_hash,
            "Retrain input changed while reading")
    return loss, receipt, receipt_hash


def load_array(path, shape):
    return checked(np.load(path, allow_pickle=False, mmap_mode="r"), shape, path.name)


def fixed_arms():
    # Keep the float32 products, projection and alpha fit used in the 26734 scorer.
    exact = np.column_stack([
        load_array(RUNS / PRODUCER / f"exact/Y_{start}_{start + 10}.npy", (TRAIN_SIZE, 10))
        for start in range(PANEL.start, PANEL.stop, 10)
    ])
    cheap = load_array(RUNS / "stage0/cheap/ekfac_1e-8.npy", (TRAIN_SIZE, FORWARD_QUERIES))[:, PANEL]
    gram = load_array(RUNS / "stage0/cheap/gram_query_grads.npy", (FORWARD_QUERIES, FORWARD_QUERIES))[PANEL, PANEL]
    basis = select_U(gram, 10)
    criteria = read_json(SOURCE / "configs/criteria.json")
    seed = criteria["stage0"]["negative_control"]["seed"]
    cheap_perm = cheap[np.random.default_rng(seed).permutation(TRAIN_SIZE)]
    mixed, _ = hybrid(exact, cheap, basis)
    mixed_perm, _ = hybrid(exact, cheap_perm, basis)
    return dict(zip(ARMS, (exact, cheap, spell(exact, basis), mixed, cheap_perm, mixed_perm), strict=True))


def predictions(masks, arms):
    design = -masks.astype(np.float32)
    return np.stack([design @ arms[name] for name in ARMS])


def centered_ranks(values, axis):
    ranks = rankdata(values, axis=axis)
    return ranks - ranks.mean(axis=axis, keepdims=True)


def lds_from_ranks(predicted, truth, axis):
    denom = np.linalg.norm(predicted, axis=axis) * np.linalg.norm(truth, axis=axis)
    return np.divide((predicted * truth).sum(axis=axis), denom,
                     out=np.full(denom.shape, np.nan), where=denom > 0)


def point_scores(predicted, truth):
    require(np.isfinite(predicted).all() and np.isfinite(truth).all(), "Non-finite prediction or loss")
    return lds_from_ranks(centered_ranks(predicted, 1), centered_ranks(truth, 0)[None, ...], 1)


def replay_26734(cfg, arms, masks, prior):
    verify_prior_signal_inputs(cfg)
    losses = np.stack([
        load_array(RUNS / f"stage0/retrain/loss_{j}.npy", (FORWARD_QUERIES + 1,))[PANEL]
        for j in range(cfg["M"])
    ])
    scores = point_scores(predictions(masks[:cfg["M"]], arms), losses)
    require(np.isfinite(scores).all(), "Historical 26734 point scores are incomplete")
    reproduced = scores.mean(axis=1)
    published = [prior["lds"][name] for name in ARMS]
    error = float(np.max(np.abs(reproduced - published)))
    require(error <= 1e-8, "Historical 26734 LDS replay differs")
    return {"prior_job": 26734, "max_absolute_lds_error": error, "n_historical_masks": cfg["M"]}


def contrasts(means):
    exact, ekfac, spell_lds, hybrid_lds, perm_ekfac, perm_hybrid = means.T
    baseline = np.maximum(ekfac, spell_lds)
    return {
        "hybrid_minus_ekfac": hybrid_lds - ekfac,
        "hybrid_minus_spell": hybrid_lds - spell_lds,
        "hybrid_minus_best_baseline": hybrid_lds - baseline,
        "exact_minus_hybrid": exact - hybrid_lds,
        "exact_minus_best_observed": exact - np.maximum(baseline, hybrid_lds),
        "permuted_hybrid_minus_best_control": perm_hybrid - np.maximum(spell_lds, perm_ekfac),
    }


def joint_means(predicted, truth, mask_indices, query_indices_draws):
    n_draws, queries = query_indices_draws.shape
    means = np.empty((n_draws, len(ARMS)), dtype=np.float64)
    for start in range(0, n_draws, BOOTSTRAP_BATCH):
        end = min(start + BOOTSTRAP_BATCH, n_draws)
        rows = mask_indices[start:end, :, None]
        cols = query_indices_draws[start:end, None, :]
        truth_ranks = centered_ranks(truth[rows, cols], 1)
        for arm_index, values in enumerate(predicted):
            lds = lds_from_ranks(centered_ranks(values[rows, cols], 1), truth_ranks, 1)
            require(lds.shape == (end - start, queries), "Bootstrap score dimensions differ")
            means[start:end, arm_index] = lds.mean(axis=1)
    return means


def paired_summary(predicted, truth, draws=BOOTSTRAP_DRAWS):
    scores = point_scores(predicted, truth)
    valid = np.isfinite(scores).all(axis=0)
    coverage = {
        "scorable_queries": int(valid.sum()),
        "undefined_query_positions": (np.flatnonzero(~valid) + PANEL.start).tolist(),
    }
    if not valid.all():
        return coverage | {"diagnostic_status": "untestable_incomplete_fixed_panel"}

    n_masks, n_queries = truth.shape
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    queries = rng.integers(n_queries, size=(draws, n_queries))
    masks = rng.integers(n_masks, size=(draws, n_masks))
    means = scores.mean(axis=1)
    delta = {key: float(value) for key, value in contrasts(means).items()}
    query_means = scores[:, queries].mean(axis=2).T
    joint = joint_means(predicted, truth, masks, queries)
    retained = np.isfinite(joint).all(axis=1)

    def intervals(distribution):
        endpoints = {}
        for key, value in contrasts(distribution).items():
            # An undefined rank at a sampled mask set leaves a difference between -2 and +2.
            finite = np.isfinite(value)
            lower = np.quantile(np.where(finite, value, -2.0), 0.025)
            upper = np.quantile(np.where(finite, value, 2.0), 0.975)
            endpoints[key] = [float(lower), float(upper)]
        return endpoints

    joint_intervals = intervals(joint)
    if delta["permuted_hybrid_minus_best_control"] > CONTROL_MAX_GAIN:
        status = "invalid_permutation_control"
    else:
        lower, upper = joint_intervals["hybrid_minus_best_baseline"]
        status = "inconclusive"
        if delta["hybrid_minus_best_baseline"] >= REFERENCE_GAIN and lower > 0:
            status = "positive_at_practical_reference"
        elif upper < REFERENCE_GAIN:
            status = "below_practical_reference"
    return coverage | {
        "diagnostic_status": status, "reference_gain": REFERENCE_GAIN,
        "control_max_gain": CONTROL_MAX_GAIN,
        "mean_lds": dict(zip(ARMS, means.tolist(), strict=True)),
        "contrasts": delta,
        "paired_query_only_95pct_intervals": intervals(query_means),
        "paired_mask_and_query_95pct_intervals": joint_intervals,
        "paired_mask_and_query_draws_retained": int(retained.sum()),
        "bootstrap_draws": draws, "bootstrap_seed": BOOTSTRAP_SEED,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("preflight", "retrain", "score"))
    parser.add_argument("--freeze")
    parser.add_argument("--run", type=Path)
    args = parser.parse_args()
    cfg, pool, masks, identity, prior = preflight(args.phase)
    if args.phase == "preflight":
        witness = replay_26734(cfg, fixed_arms(), masks, prior)
        print("[codex] " + json.dumps({"identity": identity, "historical_replay": witness}, sort_keys=True))
        return
    job = os.environ.get("SLURM_JOB_ID", "")
    require(re.fullmatch(r"[1-9][0-9]*", job) and re.fullmatch(r"[a-f0-9]{40}", args.freeze or "")
            and args.run == RUNS / "new-mask-confirm" / job, "Use the frozen Slurm launcher")
    identity.update(freeze=args.freeze, slurm_job_id=int(job))
    test_masks = masks[MASK_START:]
    if args.phase == "retrain":
        retrain(args.run, cfg, pool, test_masks, identity)
    else:
        truth, receipt, receipt_hash = checked_retrain(args.run, identity)
        arms = fixed_arms()
        witness = replay_26734(cfg, arms, masks, prior)
        metrics = paired_summary(predictions(test_masks, arms), truth)
        verify_files(RUNS, read_json(MANIFEST)["run_sha256"])
        publish_json(args.run / "summary.json", {
            "schema": 1, "identity": identity, "retrain_receipt_sha256": receipt_hash,
            "retrain_sha256": receipt["retrain_sha256"], "historical_replay": witness,
            "technical_replay_normalized_rmse": receipt["technical_replay_normalized_rmse"],
            "metrics": metrics, "diagnostic_only": True, "completed_at_taipei": taipei_now(),
        })


if __name__ == "__main__":
    main()
