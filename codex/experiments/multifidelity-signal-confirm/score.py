"""Apply the existing Stage 0 signal rule to a new fixed query panel."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

SOURCE = Path(__file__).resolve().parents[3] / "killtests/multifidelity-attribution"
sys.path.insert(0, str(SOURCE / "scripts"))
import common
import hybrid
from stage0_criteria import column_cosine, grade_interval, lds_q, stable_mean

START, END, CHUNK, BUDGET = 50, 100, 10, 10
HISTORY = Path("/work/ChihChungHsu/cactus8603/icml/runs/multifid/stage0")
HISTORY_SHA = "07383dfad5e16bd3ffb3d2b7c9ec367d9e4d02232848157bde88ffb914cda1c2"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_inputs(cfg):
    if (cfg["train_seed"], cfg["k_total"], cfg["M"], cfg["qchunk"], cfg["p_drop"]) != (0, 400, 200, CHUNK, 0.01):
        raise ValueError("Frozen training configuration differs")
    paths = ["cheap/ekfac_1e-8.npy", "cheap/gram_query_grads.npy", "exact/ql_0_10.npy", "exact/Y_0_10.npy"]
    paths.extend(f"retrain/loss_{index}.npy" for index in range(cfg["M"]))
    manifest = "".join(f"{digest(HISTORY / name)}  {name}\n" for name in paths)
    if hashlib.sha256(manifest.encode()).hexdigest() != HISTORY_SHA:
        raise ValueError("Frozen source manifest differs")


def array(path, shape):
    data = np.load(path, allow_pickle=False, mmap_mode="r")
    if data.shape != shape or not np.isfinite(data).all():
        raise ValueError(f"Bad shape or non-finite values: {path}")
    return data


def replay(run):
    reference = array(HISTORY / "exact/ql_0_10.npy", (CHUNK,))
    current = array(run / "exact/ql_0_10.npy", (CHUNK,))
    loss_error = float(np.sqrt(np.mean((current - reference) ** 2)) / max(float(np.std(reference)), 0.1))
    old_y = array(HISTORY / "exact/Y_0_10.npy", (50_000, CHUNK)).astype(np.float64)
    new_y = array(run / "exact/Y_0_10.npy", old_y.shape)
    norm_y = np.linalg.norm(old_y)
    if not norm_y > 0:
        raise ValueError("Zero norm for the historical metagradient replay")
    metagrad_error = float(np.linalg.norm(new_y - old_y) / norm_y)
    if loss_error >= 0.02 or metagrad_error >= 0.02:
        raise ValueError(f"Historical replay exceeded its allowance: {loss_error=}, {metagrad_error=}")
    return {"loss_normalized_rmse": loss_error, "metagradient_relative_l2": metagrad_error}


def judge(rho, rho_perm, gain_perm, criteria):
    bootstrap = criteria["bootstrap"]
    draws = np.random.RandomState(bootstrap["seed"]).randint(0, len(rho), size=(bootstrap["n"], len(rho)))
    tail = (1 - bootstrap["ci"]) * 0.5
    ci = np.quantile(rho[draws].mean(axis=1), [tail, 1 - tail]).tolist()
    rho_mean, rho_perm_mean = stable_mean(rho), stable_mean(rho_perm)
    stage0 = criteria["stage0"]
    limit = stage0["negative_control"]
    passed = abs(rho_perm_mean) <= limit["rho_abs_max"] and gain_perm <= limit["gain_max"]
    status = grade_interval(rho_mean, ci, stage0["signal"]) if passed else "INVALID"
    return {"status": status, "rho_perp": rho_mean, "ci": ci, "control": {
        "pass": passed, "rho_perp_permuted": rho_perm_mean, "hybrid_gain_permuted": gain_perm,
    }}


def summarize(cfg, criteria, run):
    panel = slice(START, END)
    losses = np.stack([array(HISTORY / "retrain" / f"loss_{j}.npy", (401,))[panel] for j in range(cfg["M"])])
    n = 50_000
    masks = np.stack([common.removal_mask_np(cfg, j, n) for j in range(cfg["M"])])
    exact = np.concatenate([array(run / "exact" / f"Y_{i}_{i + CHUNK}.npy", (n, CHUNK))
                            for i in range(START, END, CHUNK)], axis=1)
    cheap = array(HISTORY / "cheap/ekfac_1e-8.npy", (n, 400))[:, panel]
    gram = array(HISTORY / "cheap/gram_query_grads.npy", (400, 400))[panel, panel]
    basis = hybrid.select_U(gram, BUDGET)
    stage0 = criteria["stage0"]
    perm = np.random.default_rng(stage0["negative_control"]["seed"]).permutation(n)
    cheap_perm = cheap[perm]
    spell = hybrid.spell(exact, basis)
    mixed, _ = hybrid.hybrid(exact, cheap, basis)
    mixed_perm, _ = hybrid.hybrid(exact, cheap_perm, basis)
    methods = {"exact": exact, "spell": spell, "ekfac": cheap, "hybrid": mixed,
               "ekfac_permuted": cheap_perm, "hybrid_permuted": mixed_perm}
    scores = {name: lds_q(values, masks, losses) for name, values in methods.items()}
    exact_residual = exact - spell
    rho = column_cosine(exact_residual, cheap - hybrid.spell(cheap, basis))
    rho_perm = column_cosine(exact_residual, cheap_perm - hybrid.spell(cheap_perm, basis))
    required = {name: scores[name] for name in ("spell", "ekfac_permuted", "hybrid_permuted")}
    required.update(rho=rho, rho_permuted=rho_perm)
    invalid = {name: (np.flatnonzero(~np.isfinite(values)) + START).tolist()
               for name, values in required.items() if not np.isfinite(values).all()}
    if invalid:
        return {"signal": {"status": "UNTESTABLE", "undefined_positions": invalid}}
    lds = {name: stable_mean(values) if np.isfinite(values).all() else None for name, values in scores.items()}
    perm_gain = lds["hybrid_permuted"] - max(lds["spell"], lds["ekfac_permuted"])
    gain = lds["hybrid"] - lds["ekfac"] if lds["hybrid"] is not None and lds["ekfac"] is not None else None
    return {"signal": judge(rho, rho_perm, perm_gain, criteria), "lds": lds, "hybrid_minus_ekfac": gain}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "configure", "replay", "score"))
    parser.add_argument("--run", type=Path)
    parser.add_argument("--freeze")
    args = parser.parse_args()
    config = json.loads((SOURCE / "configs/stage0.json").read_text())
    if args.mode == "preflight":
        verify_inputs(config)
        print("Frozen attribution inputs verified")
    elif args.mode == "configure":
        config.update(run_dir=str(args.run), tag="signal-confirm")
        (args.run / "config.json").write_text(json.dumps(config))
    elif args.mode == "replay":
        print(json.dumps(replay(args.run)))
    else:
        verify_inputs(config)
        criteria = json.loads((SOURCE / "configs/criteria.json").read_text())
        result = summarize(config, criteria, args.run)
        result.update(freeze=args.freeze, query_positions=[START, END],
                      cifar_ids=common.query_indices(config)[START:END].tolist(), source_sha256=HISTORY_SHA,
                      replay=replay(args.run))
        temporary = args.run / "summary.pending"
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.run / "summary.json")


if __name__ == "__main__":
    main()
