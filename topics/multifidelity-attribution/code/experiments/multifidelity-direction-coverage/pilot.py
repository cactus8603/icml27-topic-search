"""Fresh coordinate selection and held-out exact mask-span diagnostic."""

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np


FORMAL_COMMIT = "599469e3faf5dec7aae854cc929c6d265bdd706f"
CANDIDATES = np.arange(400, 424)
TARGETS = np.arange(424, 436)
FIT_MASKS = range(200, 264)
TEST_MASKS = range(264, 328)
N_DIRECTIONS = 10
UNIFORM_SEED = 7611501
BOOTSTRAP_SEED = 7611502
RANK_RTOL = 1e-10
PRACTICAL_DELTA = 0.10


def uniform_columns():
    rng = np.random.Generator(np.random.PCG64(UNIFORM_SEED))
    return np.sort(rng.choice(CANDIDATES.size, N_DIRECTIONS, replace=False))


def geometry_columns(fit_response):
    if fit_response.shape != (len(FIT_MASKS), CANDIDATES.size) or not np.isfinite(fit_response).all():
        raise ValueError("invalid cheap fit responses")
    centered = fit_response - fit_response.mean(axis=0)
    lengths = np.linalg.norm(centered, axis=0)
    scaled = np.divide(centered, lengths, out=np.zeros_like(centered), where=lengths > 0)
    residual = scaled.copy()
    remaining = np.ones(CANDIDATES.size, dtype=bool)
    chosen = []
    for _ in range(N_DIRECTIONS):
        norms = np.linalg.norm(residual, axis=0)
        largest = norms[remaining].max()
        if largest <= RANK_RTOL:
            raise ValueError("cheap fit geometry has rank below ten")
        # Break near-ties by the earliest frozen candidate position.
        column = np.flatnonzero(remaining & (norms >= largest - RANK_RTOL))[0]
        basis = residual[:, column] / norms[column]
        residual -= np.outer(basis, basis @ residual)
        chosen.append(int(column))
        remaining[column] = False
    return np.sort(chosen)


def query_plan(query_indices, cfg):
    if cfg["k_total"] != CANDIDATES[0] or cfg["M"] != FIT_MASKS.start:
        raise ValueError("fresh positions overlap the previous budget or leave an unexpected gap")
    ids = query_indices(dict(cfg, k_total=int(TARGETS[-1]) + 1))
    if ids.size != TARGETS[-1] + 1 or np.unique(ids).size != ids.size:
        raise ValueError("query permutation is not a unique fixed prefix")
    return ids


def span_score(fit, heldout, efit, eheldout):
    # The intercept is fitted on the fit masks. Centering the held-out prediction
    # separately makes the reported R² measure only variation across masks.
    centered = fit - fit.mean(axis=0)
    norms = np.linalg.norm(centered, axis=0)
    if not np.isfinite(norms).all() or np.any(norms == 0):
        raise ValueError("constant or non-finite fitted exact directions")
    basis, _, rank, _ = np.linalg.lstsq(centered / norms, efit - efit.mean(axis=0), rcond=RANK_RTOL)
    if rank != N_DIRECTIONS or not np.isfinite(basis).all():
        raise ValueError("fit-mask exact span has rank below ten or non-finite coefficients")
    prediction = ((heldout - fit.mean(axis=0)) / norms) @ basis + efit.mean(axis=0)
    pcenter = prediction - prediction.mean(axis=0)
    ecenter = eheldout - eheldout.mean(axis=0)
    total = np.sum(ecenter**2, axis=0)
    if np.any(total == 0) or np.any(np.sum(pcenter**2, axis=0) == 0):
        raise ValueError("constant held-out exact target or span prediction")
    return 1.0 - np.sum((ecenter - pcenter) ** 2, axis=0) / total


def score(fit, heldout, efit, eheldout, geometry, uniform):
    if not all(np.isfinite(x).all() for x in (fit, heldout, efit, eheldout)):
        raise ValueError("non-finite exact mask responses")
    if any(len(arm) != N_DIRECTIONS or len(set(arm)) != N_DIRECTIONS for arm in (geometry, uniform)):
        raise ValueError("per-arm exact budget is not ten distinct directions")
    g = span_score(fit[:, geometry], heldout[:, geometry], efit, eheldout)
    u = span_score(fit[:, uniform], heldout[:, uniform], efit, eheldout)
    delta = g - u
    if not np.isfinite(delta).all():
        raise ValueError("non-finite held-out R²")
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    samples = rng.integers(len(delta), size=(10000, len(delta)))
    lower, upper = np.quantile(np.median(delta[samples], axis=1), [0.025, 0.975])
    med = float(np.median(delta))
    return {
        "status": "COVERAGE_IMPROVEMENT" if med >= PRACTICAL_DELTA and lower > 0 else "INCONCLUSIVE",
        "geometry_r2_median": float(np.median(g)),
        "uniform_r2_median": float(np.median(u)),
        "paired_delta_r2_median": med,
        "paired_query_bootstrap_95": [float(lower), float(upper)],
        "geometry_queries_ahead": int(np.count_nonzero(delta > 0)),
        "evaluation_queries": len(delta),
    }


def head_gradient_fn(jax, jnp, trainer):
    def loss(head, params, buffers, image, label):
        return trainer._qloss_one((dict(params, fc=head), buffers), image, label)

    @jax.jit
    def gradient(params, buffers, images, labels):
        head = jax.vmap(jax.grad(loss), in_axes=(None, None, None, 0, 0))(
            params["fc"], params, buffers, images, labels
        )
        return jnp.concatenate((head["w"].reshape((images.shape[0], -1)), head["b"]), axis=1)

    return gradient


def cheap_fit_responses(jax, jnp, trainer, final, images, labels, fit_masks):
    gradient = head_gradient_fn(jax, jnp, trainer)
    params, _, buffers = final
    candidate = np.asarray(gradient(params, buffers, images, labels), dtype=np.float64)
    output = np.zeros((len(fit_masks), len(images)))
    for start in range(0, trainer.n, 200):
        stop = min(start + 200, trainer.n)
        train = np.asarray(gradient(params, buffers, trainer.X[start:stop], trainer.Y[start:stop]), dtype=np.float64)
        output -= fit_masks[:, start:stop] @ (train @ candidate.T)
    return output


def write_json(path, payload):
    data = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    temporary = path.with_suffix(".tmp")
    with temporary.open("xb") as stream:
        stream.write(data)
    temporary.rename(path)
    return hashlib.sha256(data).hexdigest()


def run(args, cfg, ids, source):
    import jax
    import jax.numpy as jnp
    from common import load_cifar, removal_mask_np
    from mf_core import ResNet9, Trainer

    device, = jax.devices()
    if device.platform != "gpu" or "H200" not in device.device_kind:
        raise ValueError(f"requires one NVIDIA H200, got {device}")
    xtr, ytr, xte, yte = load_cifar(cfg["n_train"])
    if len(ytr) != 50000 or len(yte) != 10000:
        raise ValueError("unexpected CIFAR-10 dimensions")
    trainer = Trainer(ResNet9(cfg["width"]), xtr, ytr, cfg["hp"])
    del xtr, ytr
    w = jnp.ones(trainer.n, dtype=jnp.float32)
    seed, keep = cfg["train_seed"], cfg["ckpt_every"]
    mask_design = np.stack([removal_mask_np(cfg, i, trainer.n) for i in (*FIT_MASKS, *TEST_MASKS)])
    if not mask_design.any(axis=1).all() or np.unique(np.packbits(mask_design, axis=1), axis=0).shape[0] != len(mask_design):
        raise ValueError("new mask design is empty or duplicate")
    fit_masks = -mask_design[:len(FIT_MASKS)].astype(np.float64)

    args.output.mkdir(parents=True, exist_ok=False)
    begin = time.monotonic()
    final, checkpoints = trainer.train(w, seed, keep_every=keep)
    jax.block_until_ready(final)
    print(f"source training replay elapsed {time.monotonic() - begin:.0f}s", flush=True)
    cheap = cheap_fit_responses(jax, jnp, trainer, final, jnp.asarray(xte[ids[CANDIDATES]]),
                               jnp.asarray(yte[ids[CANDIDATES]]), fit_masks)
    try:
        geometry = geometry_columns(cheap)
    except ValueError as error:
        write_json(args.output / "results.json", {"status": "UNTESTABLE", "reason": str(error),
                   "frozen_git_commit": args.commit, "completed_at": datetime.now().astimezone().isoformat()})
        print("UNTESTABLE", flush=True)
        return
    uniform = uniform_columns()
    union = np.union1d(geometry, uniform)
    positions = np.concatenate((CANDIDATES[union], TARGETS))
    receipt = {
        "frozen_git_commit": args.commit,
        "formal_source_git_commit": FORMAL_COMMIT,
        "stage_config_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "data_manifest_sha256": hashlib.sha256(Path(__file__).with_name("cifar.sha256").read_bytes()).hexdigest(),
        "mask_design_sha256": hashlib.sha256(np.packbits(mask_design, axis=1).tobytes()).hexdigest(),
        "jax_version": jax.__version__,
        "numpy_version": np.__version__,
        "device": device.device_kind,
        "candidate_positions": CANDIDATES.tolist(),
        "geometry_positions": CANDIDATES[geometry].tolist(),
        "uniform_positions": CANDIDATES[uniform].tolist(),
        "evaluation_positions": TARGETS.tolist(),
        "fit_mask_range": [FIT_MASKS.start, FIT_MASKS.stop],
        "heldout_mask_range": [TEST_MASKS.start, TEST_MASKS.stop],
        "exact_query_count": int(len(positions)),
        "selection_frozen_at": datetime.now().astimezone().isoformat(),
    }
    selection_sha = write_json(args.output / "selection.json", receipt)
    print(f"cheap-only selection frozen; exact queries {len(positions)}", flush=True)

    test_masks = -mask_design[len(FIT_MASKS):].astype(np.float64)
    fit, heldout = [], []
    for start in range(0, len(positions), cfg["qchunk"]):
        block = positions[start:start + cfg["qchunk"]]
        begin = time.monotonic()
        values, _ = trainer.metagrad(w, seed, jnp.asarray(xte[ids[block]]), jnp.asarray(yte[ids[block]]),
                                     keep, (final, checkpoints))
        fit.append(fit_masks @ values.T)
        heldout.append(test_masks @ values.T)
        del values
        print(f"exact block {start}:{start + len(block)} elapsed {time.monotonic() - begin:.0f}s", flush=True)
    fit, heldout = np.concatenate(fit, axis=1), np.concatenate(heldout, axis=1)
    n = len(union)
    try:
        result = score(fit[:, :n], heldout[:, :n], fit[:, n:], heldout[:, n:],
                       np.searchsorted(union, geometry), np.searchsorted(union, uniform))
    except (ValueError, np.linalg.LinAlgError) as error:
        result = {"status": "UNTESTABLE", "reason": str(error)}
    write_json(args.output / "results.json", result | {
        "selection_sha256": selection_sha,
        "completed_at": datetime.now().astimezone().isoformat(),
    })
    print(result["status"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source / "scripts"))
    from common import query_indices

    config = args.source / "configs/stage0.json"
    cfg = json.loads(config.read_text())
    if (cfg["train_seed"], cfg["n_train"], cfg["p_drop"], cfg["matmul_precision"]) != (0, None, 0.01, "default"):
        raise ValueError("unexpected source training/mask configuration")
    ids = query_plan(query_indices, cfg)
    if args.output is None:
        print(f"preflight OK: {len(CANDIDATES)} fresh candidates, {len(TARGETS)} fresh evaluation queries, "
              f"{len(FIT_MASKS)}+{len(TEST_MASKS)} fresh masks; uniform positions {CANDIDATES[uniform_columns()].tolist()}")
        return
    run(args, cfg, ids, config)


if __name__ == "__main__":
    main()
