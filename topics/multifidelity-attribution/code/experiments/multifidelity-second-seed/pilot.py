"""Seed-6 exact and fixed EK-FAC LDS on the existing matched 12-query panel."""

import argparse
from datetime import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "config.json"
N_TRAIN = 50000
RUNS = Path("/work/ChihChungHsu/cactus8603/icml/runs/multifid/second_seed")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def check_versions(spec, *packages):
    for name in packages:
        require(importlib.metadata.version(name) == spec["versions"][name], f"Unexpected {name} version")


def plan(source, spec):
    checked = {}
    for relative, expected in spec["source_sha256"].items():
        checked[relative] = digest(source / relative)
        require(checked[relative] == expected, f"Frozen Stage 0 source changed: {relative}")
    sys.path.insert(0, str(source / "scripts"))
    from common import CIFAR_DIR, query_indices, removal_mask_np

    data_hashes = {}
    for line in (HERE / "cifar.sha256").read_text().splitlines():
        expected, filename = line.split()
        data_hashes[filename] = digest(Path(CIFAR_DIR) / filename)
        require(data_hashes[filename] == expected, f"CIFAR-10 checksum differs: {filename}")
    cfg = json.loads((source / "configs/stage0.json").read_text())
    criteria = json.loads((source / "configs/criteria.json").read_text())
    occupied = {cfg["train_seed"], cfg["crossseed_train_seed"], *cfg["ensemble_seeds"]}
    require(spec["train_seed"] not in occupied, "The exploratory seed overlaps Stage 0/1")
    positions = np.concatenate([np.arange(start, end) for start, end in spec["exact_chunks"]])
    queries = query_indices(cfg)[positions]
    require(queries.tolist() == spec["cifar_queries"], "The previous 12-query panel changed")
    mask_ids = list(range(spec["mask_start"], spec["mask_end"]))
    masks = np.stack([removal_mask_np(cfg, i, N_TRAIN) for i in mask_ids])
    mask_digest = hashlib.sha256(np.packbits(masks, axis=1).tobytes()).hexdigest()
    require(mask_digest == spec["mask_sha256"], "The previous evaluation masks changed")
    identity = {
        "config_sha256": digest(CONFIG), "stage0_source_commit": spec["source_commit"],
        "stage0_source_sha256": checked, "cifar_sha256": data_hashes, "train_seed": spec["train_seed"],
        "query_positions": positions.tolist(), "cifar_queries": queries.tolist(), "mask_ids": mask_ids,
        "packed_mask_sha256": mask_digest, "removed_count_range": [int(masks.sum(1).min()), int(masks.sum(1).max())],
        "export_fidelity_limit": criteria["gates"]["export_fidelity_max_abs_logit_diff"],
    }
    return cfg, queries, masks, identity


def checked(value, shape, label):
    require(value.shape == shape and np.isfinite(value).all(), f"Bad shape or nonfinite values: {label}")
    return value


def produce_trajectory(run, spec, cfg, queries, masks, identity):
    check_versions(spec, "jax")
    import jax
    import jax.numpy as jnp
    from mf_core import ResNet9, Trainer, load_cifar

    checkpoint_file = run / "trajectory.npz"
    require(not checkpoint_file.exists(), "Trajectory already exists for this job")
    require(jax.devices()[0].platform == "gpu", "This phase requires CUDA JAX")
    require(cfg["matmul_precision"] == "default", "Unexpected training precision")
    xtr, ytr, xte, yte = load_cifar()
    require(len(xtr) == N_TRAIN and len(xte) == 10000, "Unexpected CIFAR-10 shape")
    xq, yq = jnp.asarray(xte[queries]), jnp.asarray(yte[queries])
    trainer = Trainer(ResNet9(cfg["width"]), xtr, ytr, cfg["hp"])
    w = jnp.ones(N_TRAIN, jnp.float32)
    seed = spec["train_seed"]
    final, checkpoints = trainer.train(w, seed, keep_every=cfg["ckpt_every"])
    reference = np.asarray(trainer.query_losses(final, xq, yq))
    blocks, losses = [], []
    offset = 0
    for start, end in spec["exact_chunks"]:
        size = end - start
        selection = slice(offset, offset + size)
        matrix, block_losses = trainer.metagrad(w, seed, xq[selection], yq[selection],
                                                cfg["ckpt_every"], (final, checkpoints))
        blocks.append(checked(matrix.T.astype(np.float32), (N_TRAIN, size), "exact query block"))
        losses.append(checked(block_losses, (size,), "exact reference query losses"))
        offset += size
        print(f"[codex] Finished seed {seed} exact queries {start}:{end}.", flush=True)
    del checkpoints
    replay = np.concatenate(losses)
    normalized_rmse = float(np.sqrt(np.mean((replay - reference) ** 2)) / max(reference.std(), 0.1))
    require(normalized_rmse < 0.02, "Exact replay and final-model query losses disagree")

    params, _, buffers = final
    saved = np.asarray(trainer.m.apply(params, buffers, xq, False)[0])
    with jax.default_matmul_precision("highest"):
        highest = np.asarray(trainer.m.apply(params, buffers, xq, False)[0])
    arrays = {f"{kind}.{name}.{key}": np.asarray(value)
              for kind, groups in (("p", params), ("b", buffers))
              for name, leaves in groups.items() for key, value in leaves.items()}
    targets = []
    for j, removed in zip(identity["mask_ids"], masks, strict=True):
        masked = trainer.train(jnp.asarray(~removed, dtype=jnp.float32), seed)
        targets.append(checked(np.asarray(trainer.query_losses(masked, xq, yq)), (len(queries),), "retrain losses"))
        if (j + 1) % 10 == 0:
            print(f"[codex] Finished seed {seed} masks through {j}.", flush=True)
    noise = [np.asarray(trainer.query_losses(trainer.train(w, seed), xq, yq))
             for _ in range(spec["no_removal_repeats"])]
    np.savez(checkpoint_file, **arrays, query_indices=queries, mask_ids=identity["mask_ids"],
             ref_logits=checked(saved, (len(queries), 10), "original JAX logits"),
             highest_logits=checked(highest, (len(queries), 10), "highest JAX logits"),
             exact=np.column_stack(blocks), baseline=reference, replay_loss=replay,
             replay_normalized_rmse=normalized_rmse, retrain_loss=np.stack(targets),
             noise_loss=np.stack(noise), provenance=json.dumps(identity, sort_keys=True))


def produce_ekfac(run, spec, cfg, queries, identity):
    check_versions(spec, "torch")
    import torch
    from torch.nn import functional as F
    import kronfluence
    from kronfluence.analyzer import Analyzer, prepare_model
    from kronfluence.arguments import FactorArguments, ScoreArguments
    from kronfluence.task import Task
    from cheap_torch import FID_MAX, load_model, to_t
    from common import load_cifar

    require(torch.cuda.is_available(), "This phase requires CUDA PyTorch")
    output = run / "ekfac.npz"
    require(not output.exists() and not (run / "kron").exists(), "EK-FAC artifacts already exist for this job")
    archive = Path(os.environ["MULTIFID_KRON_ARCHIVE"]).resolve()
    require(Path(kronfluence.__file__).resolve().is_relative_to(archive),
            "Kronfluence must import from the pinned archived checkout")

    class CrossEntropyTask(Task):
        def compute_train_loss(self, batch, model, sample=False):
            x, labels = batch
            logits = model(x)
            if sample:
                with torch.no_grad():
                    labels = torch.multinomial(F.softmax(logits.detach(), -1), 1).flatten()
            return F.cross_entropy(logits, labels, reduction="sum")

        def compute_measurement(self, batch, model):
            x, labels = batch
            return F.cross_entropy(model(x), labels, reduction="sum")

    checkpoint = run / "trajectory.npz"
    model, saved = load_model(cfg["width"], checkpoint)
    with saved:
        require(json.loads(saved["provenance"].item()) == identity, "JAX receipt identities differ")
        require(np.array_equal(saved["query_indices"], queries), "JAX query order differs")
        highest = checked(saved["highest_logits"], (len(queries), 10), "JAX export reference")
        ordinary = checked(saved["ref_logits"], (len(queries), 10), "ordinary JAX export")
    xtr, ytr, xte, yte = load_cifar()
    inputs = to_t(xte[queries])
    with torch.no_grad():
        torch_logits = model(inputs.cuda()).cpu().numpy()
    error = float(np.max(np.abs(torch_logits - highest)))
    original_error = float(np.max(np.abs(torch_logits - ordinary)))
    require(error < FID_MAX, "Original JAX-to-PyTorch export fidelity check failed")

    task = CrossEntropyTask()
    analyzer = Analyzer(analysis_name="ekfac", model=prepare_model(model, task), task=task,
                        output_dir=str(run / "kron"))
    train = torch.utils.data.TensorDataset(to_t(xtr), torch.tensor(ytr, dtype=torch.long))
    query = torch.utils.data.TensorDataset(inputs, torch.tensor(yte[queries], dtype=torch.long))
    torch.manual_seed(spec["factor_seed"])
    analyzer.fit_all_factors(factors_name="ekfac", dataset=train, per_device_batch_size=256,
                             factor_args=FactorArguments(strategy="ekfac"), overwrite_output_dir=False)
    analyzer.compute_pairwise_scores(
        scores_name="ekfac_1e-8", factors_name="ekfac", train_dataset=train, query_dataset=query,
        per_device_query_batch_size=len(queries), per_device_train_batch_size=256,
        score_args=ScoreArguments(damping_factor=spec["ekfac_damping"]), overwrite_output_dir=False,
    )
    result = -analyzer.load_pairwise_scores("ekfac_1e-8")["all_modules"].float().cpu().numpy().T
    np.savez(output, matrix=checked(result, (N_TRAIN, len(queries)), "EK-FAC scores"),
             trajectory_sha256=digest(checkpoint), query_indices=queries,
             max_abs_export_error=error, ordinary_export_error=original_error,
             damping=spec["ekfac_damping"], provenance=json.dumps(identity, sort_keys=True))


def spearman(prediction, target):
    from scipy.stats import rankdata

    rank_pred = rankdata(prediction, axis=0).astype(np.float64)
    rank_true = rankdata(target, axis=0).astype(np.float64)
    rank_pred -= rank_pred.mean(axis=0)
    rank_true -= rank_true.mean(axis=0)
    denom = np.linalg.norm(rank_pred, axis=0) * np.linalg.norm(rank_true, axis=0)
    return np.divide(np.sum(rank_pred * rank_true, axis=0), denom,
                     out=np.full_like(denom, np.nan), where=denom > 0)


def lds_table(exact, cheap, losses, noise, removed, spec):
    design = -removed.astype(np.float64)
    exact_lds = spearman(design @ exact.astype(np.float64), losses)
    cheap_lds = spearman(design @ cheap.astype(np.float64), losses)
    valid = np.isfinite(exact_lds) & np.isfinite(cheap_lds)
    risk_std = losses.std(0, ddof=1)
    ratio = np.divide(noise.std(0, ddof=1), risk_std,
                      out=np.full_like(risk_std, np.nan), where=risk_std > 0)
    summary = {
        "strict_status": "complete" if valid.all() else "undefined_query",
        "paired_scorable_queries": int(valid.sum()),
        "undefined_positions_in_panel": np.flatnonzero(~valid).tolist(),
        "noise_over_signal_mean": float(ratio.mean()) if np.isfinite(ratio).all() else None,
    }
    if valid.sum() < 9:
        return summary
    paired = exact_lds[valid] - cheap_lds[valid]
    rng = np.random.default_rng(spec["bootstrap_seed"])
    samples = rng.choice(paired, (spec["bootstrap_resamples"], len(paired)), replace=True).mean(1)
    return summary | {
        "exact_lds": float(exact_lds[valid].mean()), "ekfac_1e-8_lds": float(cheap_lds[valid].mean()),
        "paired_lds_gap": float(paired.mean()),
        "descriptive_query_bootstrap_ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
    }


def score(run, spec, queries, masks, identity):
    traj, ekfac = run / "trajectory.npz", run / "ekfac.npz"
    hashes = {path.name: digest(path) for path in (traj, ekfac)}
    k = len(queries)
    with np.load(traj, allow_pickle=False) as a, np.load(ekfac, allow_pickle=False) as b:
        require(all(json.loads(manifest["provenance"].item()) == identity for manifest in (a, b)),
                "Stage receipts disagree on experiment identity")
        require(all(np.array_equal(manifest["query_indices"], queries) for manifest in (a, b)),
                "Different query order in producer receipts")
        require(np.array_equal(a["mask_ids"], identity["mask_ids"]), "Reordered or missing masks")
        require(b["trajectory_sha256"].item() == hashes[traj.name] and b["damping"].item() == spec["ekfac_damping"],
                "EK-FAC prediction is not bound to the exact trajectory or damping")
        require(0 <= a["replay_normalized_rmse"].item() < 0.02, "Replay check missing or invalid")
        require(0 <= b["max_abs_export_error"].item() < identity["export_fidelity_limit"],
                "Export-fidelity receipt failed the bound")
        metrics = lds_table(
            checked(a["exact"], (N_TRAIN, k), "exact"), checked(b["matrix"], (N_TRAIN, k), "EK-FAC"),
            checked(a["retrain_loss"], (len(masks), k), "retrain"),
            checked(a["noise_loss"], (spec["no_removal_repeats"], k), "no removal"), masks, spec,
        )
        metrics["replay_normalized_rmse"] = a["replay_normalized_rmse"].item()
        metrics["max_abs_export_error"] = b["max_abs_export_error"].item()
        metrics["ordinary_export_error"] = b["ordinary_export_error"].item()
    require(hashes == {path.name: digest(path) for path in (traj, ekfac)}, "Input files changed during scoring")
    receipt = identity | {"artifact_sha256": hashes, "kronfluence_commit": spec["kronfluence_commit"],
                          "completed_at_taipei": datetime.now(ZoneInfo("Asia/Taipei")).isoformat(timespec="seconds"),
                          "exploratory_only": True, "metrics": metrics}
    with tempfile.NamedTemporaryFile(mode="w", dir=run, prefix=".summary.", delete=False) as file:
        pending = Path(file.name)
        try:
            json.dump(receipt, file, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)
            file.write("\n")
            file.close()
            os.link(pending, run / "summary.json")
        finally:
            pending.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("check", "jax", "ekfac", "score"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--freeze")
    parser.add_argument("--run", type=Path)
    args = parser.parse_args()
    spec = json.loads(CONFIG.read_text())
    cfg, queries, masks, identity = plan(args.source.resolve(), spec)
    if args.stage == "check":
        print(json.dumps(identity, sort_keys=True))
        return
    job = os.getenv("SLURM_JOB_ID", "")
    require(re.fullmatch(r"[1-9][0-9]*", job) and re.fullmatch(r"[0-9a-f]{40}", args.freeze or "")
            and args.run is not None and args.run.resolve() == RUNS.resolve() / job,
            "Use the reviewed Slurm runner and its isolated job directory")
    identity |= {"freeze_commit": args.freeze, "slurm_job_id": int(job)}
    if args.stage == "jax":
        produce_trajectory(args.run, spec, cfg, queries, masks, identity)
    elif args.stage == "ekfac":
        produce_ekfac(args.run, spec, cfg, queries, identity)
    else:
        score(args.run, spec, queries, masks, identity)


if __name__ == "__main__":
    main()
