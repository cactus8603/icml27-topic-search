"""Frozen-calibration complement alignment on 12 independently reserved queries."""

import argparse
from datetime import datetime
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import shutil
import sys
from zoneinfo import ZoneInfo

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / "killtests/multifidelity-attribution"
SPEC_DIR = HERE.parent / "multifidelity-second-seed"
OLD_RECEIPT = ROOT / "codex/multifidelity-heldout/results.json"
OLD_RECEIPT_SHA = "211d732034aef0b4f4136df331492d96688100a8ca721337d5104073df537ad3"
RUNS = Path("/work/ChihChungHsu/cactus8603/icml/runs/multifid")
BASE = RUNS / "stage0"
CHECKPOINT = BASE / "models/model_s0.npz"
GRAD_GRAM = BASE / "cheap/gram_query_grads.npy"
CHEAP = BASE / "cheap/ekfac_1e-8.npy"
FACTORS = BASE / "cheap/kron/ekfac/factors_ekfac"
Y_PATHS = tuple(BASE / f"exact/Y_{i}_{i+10}.npy" for i in range(0, 50, 10))
EXACT_SOURCE_JOB = 26628
EXACT_SOURCE_FREEZE = "6d8344ab0fd355dea6b80c3ccf14685c6359f1f0"
EXACT_SOURCE_SHA256 = "9e5f1a090cb8d3150fe6bf5021eb2c697f117454a01fda71c869507631311fc7"
EXACT_SOURCE = RUNS / "complement" / str(EXACT_SOURCE_JOB) / "exact.npz"
N = 50_000
HISTORICAL_QUERY_COUNT = 400
CAL = 50
B = 10
TARGET = np.arange(436, 448)
TARGET_IDS = (8388, 3928, 7642, 9514, 5500, 2395, 2463, 705, 3077, 867, 6105, 5970)
REPLAY = 4
DAMPING = 1e-8
REPLAY_RMSE_MAX = 0.02
TECHNICAL_REPLAY_TOL = 2e-4
BOOT_SEED = 2026111546
PERM_SEED = 20261114


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def pins(path, parent):
    result = {}
    for line in path.read_text().splitlines():
        digest, name = line.split()
        require(re.fullmatch(r"[0-9a-f]{64}", digest) and Path(name).name == name, "Invalid input digest manifest")
        result[parent / name] = digest
    return result


def tensor_count(path):
    with path.open("rb") as stream:
        length = int.from_bytes(stream.read(8), byteorder="little")
        require(0 < length < 2**20, f"Invalid factor header size: {path.name}")
        header = json.loads(stream.read(length))
    names = [key for key in header if key != "__metadata__"]
    require(bool(names) and all(key in header and "shape" in header[key] and "data_offsets" in header[key]
                                for key in names), f"Invalid factor tensor inventory: {path.name}")
    return len(names)


def check_inputs():
    spec = read(SPEC_DIR / "config.json")
    require(spec["train_seed"] == 6 and spec["ekfac_damping"] == DAMPING, "Unexpected historical baseline lock")
    tools = {relative: sha(SOURCE / relative) for relative in spec["source_sha256"]}
    require(tools == spec["source_sha256"], "Stage 0 files no longer match their historical inputs")
    cfg = read(SOURCE / "configs/stage0.json")
    criteria = read(SOURCE / "configs/criteria.json")
    require((cfg["train_seed"], cfg["crossseed_train_seed"], cfg["query_seed"], cfg["k_total"], cfg["k_stage0"],
             cfg["matmul_precision"]) == (0, 5, 20261114, 400, CAL, "default"), "Unexpected source query or seed")
    sys.path.insert(0, str(SOURCE / "scripts"))
    from common import CIFAR_DIR, query_indices

    pool = query_indices(dict(cfg, k_total=int(TARGET[-1] + 1)))
    require(len(pool) == 448 and np.unique(pool).size == len(pool) and TARGET[0] == 436 and CAL < TARGET[0]
            and tuple(pool[TARGET]) == TARGET_IDS,
            "Fresh queries overlap the historical 436-position prefix")
    require(sha(OLD_RECEIPT) == OLD_RECEIPT_SHA, "Historical input receipt changed")
    old_inputs = read(OLD_RECEIPT)["inputs"]
    old_files = {*Y_PATHS, CHEAP, GRAD_GRAM}
    factor_files = pins(HERE / "factors.sha256", FACTORS)
    require(set(FACTORS.iterdir()) == set(factor_files) and len(factor_files) == 13,
            "Incomplete or changed historical EK-FAC factors")
    require(read(FACTORS / "factor_arguments.json")["strategy"] == "ekfac", "Factors are not EK-FAC")
    for name in ("covariance_dataset_metadata.json", "lambda_dataset_metadata.json"):
        item = read(FACTORS / name)
        require(item["dataset_size"] == N and item["indices"] is None, "Factor data are not the full train inventory")

    cifar_files = pins(SPEC_DIR / "cifar.sha256", Path(CIFAR_DIR))
    checks = factor_files | cifar_files | {CHECKPOINT: old_inputs["checkpoint_sha256"]}
    checks.update({p: old_inputs["file_sha256"][p.relative_to(BASE).as_posix()] for p in old_files})
    for path, expected in checks.items():
        require(sha(path) == expected, f"Input SHA changed: {path.name}")

    # Asset preflight reads the NumPy headers and hashes only.
    for path, shape in [(CHEAP, (N, 400)), (GRAD_GRAM, (400, 400)), *[(p, (N, 10)) for p in Y_PATHS]]:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        require(array.shape == shape and array.dtype.kind == "f", f"Unexpected header: {path.name}")
    identity = {
        "train_seed": 0, "reserved_formal_seed": 5, "source_commit": spec["source_commit"],
        "kronfluence_commit": spec["kronfluence_commit"], "source_sha256": tools,
        "input_sha256": {str(path): expected for path, expected in checks.items()},
        "calibration_query_ids": pool[:CAL].tolist(),
        "target_query_positions": TARGET.tolist(), "target_query_ids": pool[TARGET].tolist(),
        "historical_source_receipt_sha256": OLD_RECEIPT_SHA,
        "factor_manifest_sha256": sha(HERE / "factors.sha256"),
        "factor_tensor_counts": {p.name: tensor_count(p) for p in factor_files if p.suffix == ".safetensors"},
        "damping": DAMPING,
        "export_fidelity_limit": criteria["gates"]["export_fidelity_max_abs_logit_diff"],
    }
    return cfg, spec["versions"], identity


def finite(array, shape, label):
    require(array.shape == shape and np.isfinite(array).all(), f"Bad shape or non-finite values: {label}")
    return array


def relative_l2(actual, reference, axis=None):
    denominator = np.linalg.norm(reference, axis=axis)
    require(np.all(np.isfinite(denominator)) and np.all(denominator > 0), "Undefined technical reference norm")
    return np.linalg.norm(actual - reference, axis=axis) / denominator


def metadata(archive, expected):
    require(json.loads(archive["identity"].item()) == expected, "Producer identity differs from the frozen panel")


def exact_identity(ids):
    return {**ids, "freeze": EXACT_SOURCE_FREEZE, "slurm_job_id": EXACT_SOURCE_JOB}


def check_exact_artifact(path, ids):
    require(path.is_file() and not path.is_symlink() and sha(path) == EXACT_SOURCE_SHA256,
            "Exact producer artifact does not match the frozen SHA")
    with np.load(path, allow_pickle=False) as archive:
        metadata(archive, exact_identity(ids))
        finite(archive["exact"], (N, len(TARGET)), "existing exact target columns")
        finite(archive["frozen_highest_logits"], (CAL + len(TARGET), 10), "existing frozen logits")
        require(0 <= archive["old_calibration_error"].item() < REPLAY_RMSE_MAX
                and 0 <= archive["replay_error"].item() < REPLAY_RMSE_MAX
                and 0 <= archive["replayed_state_relative_l2"].item() <= TECHNICAL_REPLAY_TOL,
                "Existing exact producer fails its original technical bounds")


def reuse_exact(run, ids):
    check_exact_artifact(EXACT_SOURCE, ids)
    destination = run / "exact.npz"
    with EXACT_SOURCE.open("rb") as source, destination.open("xb") as output:
        shutil.copyfileobj(source, output)
    check_exact_artifact(destination, ids)
    print(f"Reused exact artifact from job {EXACT_SOURCE_JOB}: {EXACT_SOURCE_SHA256}", flush=True)


def torch_query_batch(historical_ids, ids):
    start = HISTORICAL_QUERY_COUNT - len(TARGET)
    require(len(historical_ids) == HISTORICAL_QUERY_COUNT
            and list(historical_ids[:CAL]) == ids["calibration_query_ids"],
            "Historical query batch differs from the frozen calibration")
    query_ids = [*historical_ids[:start], *ids["target_query_ids"]]
    require(len(query_ids) == HISTORICAL_QUERY_COUNT and len(set(query_ids)) == len(query_ids),
            "Padded query batch repeats or omits a row")
    return query_ids, slice(start, HISTORICAL_QUERY_COUNT)


def produce_cheap(run, cfg, ids):
    import torch
    from torch.nn import functional as F
    import kronfluence
    from kronfluence.analyzer import Analyzer, prepare_model
    from kronfluence.arguments import ScoreArguments
    from kronfluence.task import Task
    from cheap_torch import Grads, load_model, to_t
    from common import load_cifar, query_indices

    require(torch.cuda.is_available(), "EK-FAC and full gradients require CUDA")
    archive = Path(os.environ["MULTIFID_KRON_ARCHIVE"]).resolve()
    require(Path(kronfluence.__file__).resolve().is_relative_to(archive), "Kronfluence source is not pinned")

    class CrossEntropy(Task):
        def compute_train_loss(self, batch, model, sample=False):
            if sample:
                raise ValueError("Historical factors are read-only. Do not refit them.")
            x, labels = batch
            return F.cross_entropy(model(x), labels, reduction="sum")

        def compute_measurement(self, batch, model):
            x, labels = batch
            return F.cross_entropy(model(x), labels, reduction="sum")

    model, stored = load_model(cfg["width"], CHECKPOINT)
    stored.close()
    xtr, labels, xte, yte = load_cifar(cfg["n_train"])
    query_ids, target_rows = torch_query_batch(query_indices(cfg), ids)
    query_x = to_t(xte[query_ids])
    query_y = torch.as_tensor(yte[query_ids], dtype=torch.long)
    check_exact_artifact(run / "exact.npz", ids)
    with np.load(run / "exact.npz", allow_pickle=False) as exact:
        high_precision = exact["frozen_highest_logits"]
    fidelity_rows = [*range(CAL), *range(target_rows.start, target_rows.stop)]
    with torch.no_grad():
        estimate = model(query_x[fidelity_rows].cuda()).cpu().numpy()
    export_error = float(np.max(np.abs(estimate - high_precision)))
    require(0 <= export_error < ids["export_fidelity_limit"], "Frozen JAX/PyTorch export differs")
    gradient = Grads(model)
    gradient_rows = gradient.flat("ce", query_x.cuda(), query_y.cuda())
    gram = (gradient_rows @ gradient_rows.T).cpu().numpy()
    new_gram = gram[:CAL, :CAL].astype(np.float64)
    old_gram = np.load(GRAD_GRAM, allow_pickle=False, mmap_mode="r")[:CAL, :CAL].astype(np.float64)
    gram_error = float(relative_l2(new_gram, old_gram))
    print(f"Calibration Gram replay relative L2: {gram_error:.9g}", flush=True)
    require(gram_error <= TECHNICAL_REPLAY_TOL, "Calibration Gram replay differs")
    cross_gram = gram[:CAL, target_rows].astype(np.float64)
    del gradient_rows, gradient
    torch.cuda.empty_cache()

    task = CrossEntropy()
    output = Path(os.environ["MULTIFID_KRON_OUTPUT"])
    scorer = Analyzer(analysis_name="ekfac", model=prepare_model(model, task), task=task, output_dir=str(output))
    factor_link = scorer.factors_output_dir(factors_name="ekfac")
    factor_link.parent.mkdir(parents=True, exist_ok=True)
    factor_link.symlink_to(FACTORS, target_is_directory=True)
    train = torch.utils.data.TensorDataset(to_t(xtr), torch.tensor(labels, dtype=torch.long))
    query = torch.utils.data.TensorDataset(query_x, query_y)
    scorer.compute_pairwise_scores(
        scores_name="ekfac_1e-8", factors_name="ekfac", train_dataset=train, query_dataset=query,
        per_device_query_batch_size=HISTORICAL_QUERY_COUNT, per_device_train_batch_size=256,
        score_args=ScoreArguments(damping_factor=DAMPING), overwrite_output_dir=False,
    )
    cheap = -scorer.load_pairwise_scores("ekfac_1e-8")["all_modules"].float().cpu().numpy().T
    finite(cheap, (N, HISTORICAL_QUERY_COUNT), "EK-FAC columns")
    old = np.load(CHEAP, allow_pickle=False, mmap_mode="r")
    err = relative_l2(cheap[:, :REPLAY].astype(np.float64), old[:, :REPLAY].astype(np.float64), axis=0)
    ekfac_error = float(np.max(err))
    print(f"EK-FAC calibration replay max relative L2: {ekfac_error:.9g}", flush=True)
    require(ekfac_error <= TECHNICAL_REPLAY_TOL, "Historical EK-FAC factors did not reproduce the frozen baseline")
    for name in FACTORS.iterdir():
        require(sha(name) == ids["input_sha256"][str(name)], "Source factors changed during scoring")
    np.savez(run / "cheap.npz", cheap=cheap[:, target_rows], cross_gram=cross_gram, gram_relative_l2=gram_error,
             ekfac_replay_relative_l2_max=ekfac_error, export_max_abs=export_error, exact_sha256=EXACT_SOURCE_SHA256,
             identity=json.dumps(ids, sort_keys=True))


def coordinates(gram, cross, budget):
    scale = float(np.max(np.abs(gram)))
    require(scale > 0 and np.max(np.abs(gram - gram.T)) / scale < 1e-6, "Degenerate or asymmetric Gram")
    symmetric = (gram + gram.T) * 0.5
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    vals, u = eigenvalues[-budget:][::-1], eigenvectors[:, -budget:][:, ::-1]
    require(vals[-1] > 0 and vals[-1] - eigenvalues[-budget - 1] > 1e-10 * vals[0],
            "Zero eigenvalue or ambiguous fixed basis cutoff")
    cal_projection = (u.T @ symmetric) / vals[:, None]
    require(np.max(np.abs(cal_projection - u.T)) <= 1e-10
            and np.max(np.abs(u.T @ u - np.eye(budget))) <= 1e-10,
            "Calibration coordinates do not reproduce the original projector")
    return u, (u.T @ cross) / vals[:, None]


def cosine(a, b, required=True):
    norm = np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0)
    valid = np.isfinite(norm) & (norm > 0)
    if required:
        require(valid.all(), "Undefined complement cosine")
    result = (a[:, valid] * b[:, valid]).sum(axis=0) / norm[valid]
    require(np.isfinite(result).all(), "Nonfinite cosine")
    return np.clip(result, -1, 1)


def summarize(ycal, ccal, y, c, g, cross, budget=B):
    u, d = coordinates(g, cross, budget)
    yp, cp = (ycal @ u) @ d, (ccal @ u) @ d
    residual_y, residual_c = y - yp, c - cp
    perm = np.random.default_rng(PERM_SEED).permutation(len(c))
    rho = cosine(residual_y, residual_c)
    permuted = cosine(residual_y, residual_c[perm])
    samples = np.random.default_rng(BOOT_SEED).integers(len(rho), size=(10_000, len(rho)))

    def mean_or_null(values):
        return float(values.mean()) if len(values) else None

    def residual_norm_ratio(new, old):
        den = np.linalg.norm(old, axis=0)
        nonzero = den > 0
        return mean_or_null(np.linalg.norm(new[:, nonzero], axis=0) / den[nonzero])

    raw = cosine(y, c, required=False)
    spell = cosine(y, yp, required=False)
    return {"mean_heldout_complement_cosine": float(rho.mean()),
            "descriptive_query_bootstrap_ci95": np.quantile(rho[samples].mean(axis=1), [0.025, 0.975]).tolist(),
            "heldout_block_means": [float(x.mean()) for x in np.array_split(rho, 2)],
            "row_permutation_mean_cosine": float(permuted.mean()),
            "row_permutation_within_reference_abs_0_05": bool(abs(permuted.mean()) <= .05),
            "raw_ekfac_alignment_mean": mean_or_null(raw), "raw_ekfac_alignment_queries": len(raw),
            "spell_term_alignment_mean": mean_or_null(spell), "spell_term_alignment_queries": len(spell),
            "remaining_mean_norm_ratio": {"exact": residual_norm_ratio(residual_y, y),
                                          "ekfac": residual_norm_ratio(residual_c, c)},
            "covered_queries": len(rho)}


def score(run, ids):
    ycal = np.column_stack([np.load(p, allow_pickle=False) for p in Y_PATHS]).astype(np.float64)
    ccal = np.load(CHEAP, allow_pickle=False, mmap_mode="r")[:, :CAL].astype(np.float64)
    g = np.load(GRAD_GRAM, allow_pickle=False, mmap_mode="r")[:CAL, :CAL].astype(np.float64)
    check_exact_artifact(run / "exact.npz", ids)
    artifacts = {name: sha(run / name) for name in ("exact.npz", "cheap.npz")}
    with np.load(run / "exact.npz", allow_pickle=False) as a, np.load(run / "cheap.npz", allow_pickle=False) as c:
        metadata(c, ids)
        require(c["exact_sha256"].item() == artifacts["exact.npz"], "EK-FAC receipt points to a different replay")
        technical = {key: float(z[key].item()) for z, fields in (
            (a, ("old_calibration_error", "replay_error", "replayed_state_relative_l2")),
            (c, ("gram_relative_l2", "ekfac_replay_relative_l2_max", "export_max_abs")),
        ) for key in fields}
        require(all(0 <= v < (REPLAY_RMSE_MAX if key in ("old_calibration_error", "replay_error")
                               else ids["export_fidelity_limit"] if key == "export_max_abs"
                               else TECHNICAL_REPLAY_TOL * (1 + 1e-12)) for key, v in technical.items()),
                "Producer technical receipts exceed bounds")
        data = (finite(ycal, (N, CAL), "cal Y"), finite(ccal, (N, CAL), "cal C"),
                finite(a["exact"], (N, len(TARGET)), "target Y").astype(np.float64),
                finite(c["cheap"], (N, len(TARGET)), "target C").astype(np.float64),
                finite(g, (CAL, CAL), "cal Gram"), finite(c["cross_gram"], (CAL, len(TARGET)), "cross Gram"))
        stats = summarize(*data)
    require(all(sha(run / name) == digest for name, digest in artifacts.items()), "Producer artifacts changed")
    receipt = {"identity": ids, "artifact_sha256": artifacts, "technical": technical, "metrics": stats,
               "exact_source": {"slurm_job_id": EXACT_SOURCE_JOB, "freeze": EXACT_SOURCE_FREEZE,
                                "sha256": EXACT_SOURCE_SHA256},
               "exploratory_only": True,
               "completed_at_taipei": datetime.now(ZoneInfo("Asia/Taipei")).isoformat(timespec="seconds")}
    pending = run / "summary.partial"
    with pending.open("x") as stream:
        json.dump(receipt, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
    os.link(pending, run / "summary.json")
    pending.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("preflight", "reuse-exact", "cheap", "score"))
    parser.add_argument("--freeze")
    parser.add_argument("--run", type=Path)
    args = parser.parse_args()
    cfg, versions, ids = check_inputs()
    if args.phase == "preflight":
        check_exact_artifact(EXACT_SOURCE, ids)
        print(json.dumps(ids, sort_keys=True))
        return
    job = os.environ.get("SLURM_JOB_ID", "")
    expected = RUNS / "complement" / job
    require(re.fullmatch(r"[1-9][0-9]*", job) and re.fullmatch(r"[a-f0-9]{40}", args.freeze or "")
            and args.run == expected, "Use the frozen Slurm launcher for every producer")
    ids |= {"freeze": args.freeze, "slurm_job_id": int(job)}
    if args.phase == "reuse-exact":
        reuse_exact(args.run, ids)
    elif args.phase == "cheap":
        require(version("torch") == versions["torch"], "Unexpected torch version")
        produce_cheap(args.run, cfg, ids)
    else:
        score(args.run, ids)


if __name__ == "__main__":
    main()
