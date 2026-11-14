import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
ROOT = EXPERIMENT.parents[1]
SCRIPTS = EXPERIMENT / "scripts"
STAGE_CONFIG = EXPERIMENT / "configs/stage0.json"
STAGE_CRITERIA = EXPERIMENT / "configs/criteria.json"
PRIOR = ROOT / "codex/multifidelity-heldout"
BASE = Path("/work/ChihChungHsu/cactus8603/icml")
RUNS = BASE / "runs/multifid/exact_probe"
CIFAR = BASE / "data/cifar10/cifar-10-batches-py"
CHUNKS = ((118, 124), (268, 274))
REL_SVD_TOL = 1e-10


def load_prior():
    spec = importlib.util.spec_from_file_location("mf_heldout", PRIOR / "evaluate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


heldout = load_prior()
require = heldout.require
sha256 = heldout.sha256
now = heldout.now
read_array = heldout.read_array


def json_read(path):
    return json.loads(path.read_text())


def json_create(path, value):
    encoded = json.dumps(value, indent=2, allow_nan=False, ensure_ascii=False) + "\n"
    with path.open("x") as stream:
        stream.write(encoded)


def run_path(run_id):
    require(re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", run_id) is not None, "invalid run ID")
    return RUNS / run_id


def probe_positions():
    rng = np.random.default_rng(7611402)
    starts = [int(rng.integers(lo, hi - 6 + 1)) for lo, hi in ((50, 225), (225, 400))]
    chunks = tuple((start, start + 6) for start in starts)
    require(chunks == CHUNKS, "preselected chunks differ from frozen protocol")
    return np.concatenate([np.arange(i, j) for i, j in chunks])


def stage_config():
    cfg = json_read(STAGE_CONFIG)
    require((cfg["train_seed"], cfg["k_total"], cfg["k_stage0"], cfg["qchunk"], cfg["query_seed"]) ==
            (0, 400, 50, 10, 20261114), "stage config is not the original trajectory")
    return cfg


def source_paths():
    return [HERE / file for file in ("PROTOCOL.md", "probe.py", "exact.sbatch", "score.sbatch")] + [
        SCRIPTS / file for file in ("exact_y.py", "common.py", "jax_common.py", "mf_core.py")
    ] + [STAGE_CONFIG, STAGE_CRITERIA, PRIOR / "evaluate.py", PRIOR / "results.json"]


def data_paths():
    return [CIFAR / f"data_batch_{i}" for i in range(1, 6)] + [CIFAR / "test_batch"]


def identity_files(paths):
    return {str(path): sha256(path) for path in paths}


def check_hashes(expected):
    for file, digest in expected.items():
        require(sha256(Path(file)) == digest, f"input hash changed: {file}")


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def freeze_sources(commit):
    require(re.fullmatch(r"[a-f0-9]{40}", commit) is not None, "expected full frozen Git commit")
    require(git("rev-parse", f"{commit}^{{commit}}") == commit, "invalid frozen Git commit")
    frozen = {}
    for path in source_paths():
        relative = path.relative_to(ROOT).as_posix()
        data = subprocess.check_output(["git", "-C", str(ROOT), "show", f"{commit}:{relative}"])
        require(path.read_bytes() == data, f"live source differs from {commit}: {relative}")
        frozen[relative] = hashlib.sha256(data).hexdigest()
    return frozen


def plan(run, cfg):
    pos = probe_positions()
    order = np.random.RandomState(cfg["query_seed"]).permutation(10000)[:cfg["k_total"]]
    return {"run_dir": str(run), "query_chunks_end_exclusive": CHUNKS, "positions": pos.tolist(),
            "cifar_indices": order[pos].tolist(), "train_seed": cfg["train_seed"],
            "producer": (SCRIPTS / "exact_y.py").relative_to(ROOT).as_posix(), "mask_fit": [0, 99], "mask_test": [100, 199]}


def prepare(run, commit):
    frozen = freeze_sources(commit)
    manifest = plan(run, stage_config()) | {
        "frozen_at": now(), "git_head": commit,
        "source_sha256": frozen, "cifar_sha256": identity_files(data_paths()),
    }
    (RUNS / "logs").mkdir(parents=True, exist_ok=True)
    run.mkdir(exist_ok=False)
    json_create(run / "manifest.json", manifest)


def verify_plan(run, commit):
    manifest = json_read(run / "manifest.json")
    require(manifest["git_head"] == commit, "submitted frozen commit differs from run manifest")
    require(manifest["source_sha256"] == freeze_sources(commit), "frozen sources differ from run manifest")
    require(set(manifest["cifar_sha256"]) == {str(path) for path in data_paths()}, "CIFAR manifest is incomplete")
    check_hashes(manifest["cifar_sha256"])
    current = json.loads(json.dumps(plan(run, stage_config())))
    require(all(manifest[key] == value for key, value in current.items()), "selection differs from run manifest")
    return manifest


def shard_config(cfg, shard_dir):
    return cfg | {"run_dir": str(shard_dir), "tag": str(shard_dir / "report")}


def producer_command(executable, producer, cfg_path, start, end):
    return [executable, str(producer), "--cfg", str(cfg_path), "--qstart", str(start), "--qend", str(end)]


def launch_shard(run, shard, cfg, producer, environment, frozen_sources=None):
    require(0 <= shard < len(CHUNKS), "invalid array index")
    start, end = CHUNKS[shard]
    work = run / f"shard-{shard}"
    work.mkdir(exist_ok=False)
    config = work / "config.json"
    json_create(config, shard_config(cfg, work))
    if frozen_sources is not None:
        # exact_y imports these adjacent modules, so run verified copies from the isolated shard.
        checkout = work / "producer"
        checkout.mkdir()
        for name in ("exact_y.py", "common.py", "jax_common.py", "mf_core.py"):
            source = SCRIPTS / name
            body = source.read_bytes()
            require(hashlib.sha256(body).hexdigest() == frozen_sources[source.relative_to(ROOT).as_posix()],
                    f"source changed while copying: {source}")
            (checkout / name).write_bytes(body)
        producer = checkout / "exact_y.py"
    command = producer_command(sys.executable, producer, config, start, end)
    subprocess.run(command, cwd=work, check=True, env=environment)
    array_path, loss_path = chunk_paths(work, start, end)
    read_array(array_path, (cfg["n_train"] or 50000, end - start))
    read_array(loss_path, (end - start,))
    json_create(work / "complete.json", {"start": start, "end": end, "finished_at": now(),
                                         "command": command, "frozen_source_copy": frozen_sources is not None,
                                         "slurm_job_id": environment.get("SLURM_JOB_ID"),
                                         "sha256": identity_files([config, array_path, loss_path])})


def chunk_paths(shard_dir, start, end):
    exact = shard_dir / "exact"
    return exact / f"Y_{start}_{end}.npy", exact / f"ql_{start}_{end}.npy"


def get_chunks(run, n, cfg, frozen_sources=None):
    arrays, baselines, completion_hashes = [], [], {}
    for i, (start, end) in enumerate(CHUNKS):
        shard = run / f"shard-{i}"
        marker = shard / "complete.json"
        done = json_read(marker)
        require((done["start"], done["end"]) == (start, end), "invalid shard completion metadata")
        array_path, loss_path = chunk_paths(shard, start, end)
        require(set(done["sha256"]) == {str(path) for path in (shard / "config.json", array_path, loss_path)},
                "unexpected shard artifacts")
        check_hashes(done["sha256"])
        require(json_read(shard / "config.json") == shard_config(cfg, shard), "shard config changed trajectory")
        if frozen_sources is not None:
            require(done["frozen_source_copy"], "shard ran without a frozen source copy")
            check_hashes({str(shard / "producer" / path.name): frozen_sources[path.relative_to(ROOT).as_posix()]
                          for path in source_paths() if path.parent == SCRIPTS})
        arrays.append(read_array(array_path, (n, end - start)))
        baselines.append(read_array(loss_path, (end - start,)))
        completion_hashes[str(marker)] = sha256(marker)
    return np.column_stack(arrays).astype(np.float64), np.concatenate(baselines), completion_hashes


def centered(x):
    return x - x.mean(axis=0)


def r2(y, pred, allow_constant=False):
    base = np.square(centered(y)).sum(axis=0)
    if not allow_constant:
        require(np.all(base > 0), "constant exact mask projection")
    error = np.square(centered(y) - centered(pred)).sum(axis=0)
    return 1 - np.divide(error, base, out=np.full_like(base, np.nan), where=base > 0)


def ranks(x):
    return centered(rankdata(x, axis=0).astype(np.float64))


def spearman(target, pred):
    a, b = ranks(target), ranks(pred)
    denom = np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0)
    result = np.full(len(denom), np.nan)
    return np.divide((a * b).sum(axis=0), denom, out=result, where=denom > 0)


def basis(a):
    q, singular, _ = np.linalg.svd(a, full_matrices=False)
    require(len(singular) >= 1 and singular[-1] > REL_SVD_TOL * singular[0], "exact direction matrix rank below budget")
    return q, singular


def geometric_diagnostics(a, target, removed, d, residual):
    q, singular = basis(a)
    measure = np.square(target).sum(axis=0)
    require(np.all(measure > 0), "zero target exact gradient")
    projected = q.T @ target
    geometrical = np.square(projected).sum(axis=0) / measure
    f = removed @ q
    exact = removed @ target
    weight_projection = f @ projected
    spell = (removed @ a) @ d
    design = np.column_stack([np.ones(100), f[:100]])
    fit, _, rank, singular_mask = np.linalg.lstsq(design, exact[:100], rcond=REL_SVD_TOL)
    require(rank == design.shape[1], "mask oracle design rank below budget")
    span_mask_fit = fit[0] + f @ fit[1:]
    s0, p0 = centered(spell[:100]), centered(weight_projection[:100])
    den = np.square(s0).sum(axis=0)
    require(np.all(den > 0), "constant calibration Spell prediction")
    beta = np.maximum(0, (s0 * p0).sum(axis=0) / den)
    prediction = {"exact": exact, "spell": spell, "residual_only": residual, "hybrid": spell + residual,
                  "weight_projection": weight_projection, "weight_projection_hybrid": weight_projection + residual,
                  "mask_fit_projection": span_mask_fit, "mask_fit_hybrid": span_mask_fit + residual,
                  "scaled_spell_hybrid": spell * beta + residual}
    on_test = {key: value[100:] for key, value in prediction.items()}
    geo = {"weight_projection_energy_fraction": geometrical.tolist(), "direction_singular_values": singular.tolist(),
           "mask_design_min_singular_value": float(singular_mask[-1]), "beta_nonnegative_scale": beta.tolist(),
           "test_r2_vs_exact": {key: r2(on_test["exact"], on_test[key]).tolist()
                                for key in ("spell", "weight_projection", "mask_fit_projection")},
           "test_r2_spell_vs_weight_projection": nullable_vector(r2(on_test["weight_projection"], on_test["spell"], allow_constant=True)),
           "test_spearman_spell_vs_exact": nullable_vector(spearman(on_test["exact"], on_test["spell"])),
           "test_spearman_spell_vs_weight_projection": nullable_vector(spearman(on_test["weight_projection"], on_test["spell"])),
           "test_mask_sd": {key: value.std(axis=0, ddof=1).tolist() for key, value in on_test.items()}}
    return prediction, geo


def nullable_vector(values):
    return [float(value) if np.isfinite(value) else None for value in values]


def diagnostics_tag(lds, mask_fit_r2):
    fit = float(np.median(mask_fit_r2))
    if lds["exact"] < .88:
        return ["UNTESTABLE_EXACT_LDS_BELOW_0_88"]
    comps = ("spell", *heldout.CHEAP_NAMES)
    baseline = max(lds[key] for key in comps)
    tags = []
    if fit < .25 and max(lds["weight_projection_hybrid"], lds["mask_fit_hybrid"]) < baseline + .04:
        tags.append("LOW_DIRECTION_SPAN_COVERAGE")
    reference = max(baseline, lds["hybrid"]) + .04
    if lds["weight_projection_hybrid"] >= reference:
        tags.append("GRAM_TRANSPORT_HEADROOM")
    elif lds["mask_fit_hybrid"] >= reference:
        tags.append("MASK_METRIC_FIT_HEADROOM")
    if lds["scaled_spell_hybrid"] >= reference:
        tags.append("SCALE_HEADROOM")
    return tags or ["MECHANISM_UNRESOLVED"]


def score_predictions(pred, target, geo, positions):
    scores = {key: spearman(target, value[100:]) for key, value in pred.items()}
    valid = np.ptp(target, axis=0) > 0
    required = [key for key in scores if key != "residual_only"]
    bad_predictions = {key: positions[valid & ~np.isfinite(scores[key])].tolist() for key in required
                       if np.any(valid & ~np.isfinite(scores[key]))}
    strict = "UNTESTABLE_CONSTANT_TARGET" if not valid.all() else "SCORED"
    if bad_predictions:
        strict = "UNTESTABLE_CONSTANT_PREDICTION"
    out = {"strict_status": strict, "query_count": int(valid.sum()), "constant_query_positions": positions[~valid].tolist(),
           "constant_predictions_on_scorable": bad_predictions, "per_query_lds": {key: nullable_vector(vals) for key, vals in scores.items()}}
    if bad_predictions or valid.sum() < 9:
        return out | {"mechanism_tags": ["UNTESTABLE_DIAGNOSTIC"]}
    means = {key: float(values[valid].mean()) if np.isfinite(values[valid]).all() else None for key, values in scores.items()}
    comparators = ("spell", *heldout.CHEAP_NAMES)
    winner = max(comparators, key=means.__getitem__)
    rng = np.random.default_rng(7611403)
    idx = rng.integers(0, valid.sum(), (10000, valid.sum()))
    bootstrap = {key: values[valid][idx].mean(axis=1) for key, values in scores.items() if means[key] is not None}
    boot_baseline = np.maximum.reduce([bootstrap[key] for key in comparators])
    effects = {}
    for key in ("hybrid", "weight_projection_hybrid", "mask_fit_hybrid", "scaled_spell_hybrid"):
        effects[key] = {
            "delta_vs_best": means[key] - means[winner],
            "delta_vs_best_ci95": np.percentile(bootstrap[key] - boot_baseline, (2.5, 97.5)).tolist(),
            "delta_vs_frozen_hybrid": means[key] - means["hybrid"],
            "delta_vs_frozen_hybrid_ci95": np.percentile(bootstrap[key] - bootstrap["hybrid"], (2.5, 97.5)).tolist(),
        }
    blocks = {}
    for start, end in CHUNKS:
        keep = valid & (positions >= start) & (positions < end)
        if np.any(keep):
            blocks[f"{start}:{end}"] = {key: float(values[keep].mean()) if np.isfinite(values[keep]).all() else None
                                       for key, values in scores.items()}
    mask_r2 = np.asarray(geo["test_r2_vs_exact"]["mask_fit_projection"])[valid]
    return out | {"mean_lds": means, "best_baseline": winner, "effects": effects, "end_exclusive_blocks_lds": blocks,
                  "mask_fit_heldout_r2_median": float(np.median(mask_r2)),
                  "mechanism_tags": diagnostics_tag(means, mask_r2)}


def summarize(run, manifest, begin):
    cfg = stage_config()
    criteria = json_read(STAGE_CRITERIA)
    source = Path(cfg["run_dir"])
    prior = json_read(PRIOR / "results.json")
    # Bind exact calibration, cheap controls, mask targets and checkpoint to the earlier logged inputs.
    require(sha256(STAGE_CONFIG) == prior["method_and_timing"]["config_sha256"], "stage config differs from heldout run")
    require(sha256(STAGE_CRITERIA) == prior["method_and_timing"]["original_criteria_sha256"], "criteria differ from heldout run")
    require(sha256(PRIOR / "evaluate.py") == prior["method_and_timing"]["script_sha256"], "heldout evaluator differs from recorded run")
    check_hashes({str(source / file): digest for file, digest in prior["inputs"]["file_sha256"].items()})
    check_hashes(prior["inputs"]["additional_file_sha256"])
    n, k, total, loss, removed, gram, audit, _ = heldout.load(cfg, criteria, source)
    require(audit["checkpoint_sha256"] == prior["inputs"]["checkpoint_sha256"], "seed0 checkpoint differs from heldout run")
    require(audit["sha256_packed_removal_masks"] == prior["inputs"]["sha256_packed_removal_masks"], "masks differ from heldout run")
    pos = probe_positions()
    require(np.array_equal(np.array(audit["query_indices"])[pos], manifest["cifar_indices"]), "probe query indices mismatched")
    target, probe_losses, marker_hashes = get_chunks(run, n, cfg, manifest["source_sha256"])
    with np.load(source / "models/model_s0.npz") as model:
        baseline = model["query_loss"][pos].astype(np.float64)
    error = probe_losses.astype(np.float64) - baseline
    normalized_rmse = float(np.sqrt(np.square(error).mean()) / max(float(baseline.std()), 0.1))
    provenance = {"frozen_plan_sha256": sha256(run / "manifest.json"), "shard_marker_hashes": marker_hashes,
                  "source_run_dir": str(source), "normalized_probe_query_loss_rmse": normalized_rmse,
                  "probe_query_loss_max_abs_difference": float(np.max(np.abs(error))),
                  "probe_query_loss_block_abs_errors": np.max(np.abs(error.reshape(2, 6)), axis=1).tolist(),
                  "checkpoint_sha256": audit["checkpoint_sha256"], "mask_sha256": audit["sha256_packed_removal_masks"],
                  "probe_query_losses": probe_losses.tolist(), "checkpoint_query_losses": baseline.tolist()}
    out = plan(run, cfg) | {"read_started_at": begin, "frozen_git_commit": manifest["git_head"], "provenance": provenance,
                            "status": "exploratory_exact_probe_same_trajectory"}
    if normalized_rmse >= .02:
        return out | {"assessment": {"strict_status": "UNTESTABLE_TRAJECTORY_CHECK", "mechanism_tags": []}}
    u, _, coords, _ = heldout.ten_directions(gram, k, heldout.BUDGET)
    a = np.zeros((n, heldout.BUDGET), dtype=np.float64)
    for start in range(0, k, cfg["qchunk"]):
        end = start + cfg["qchunk"]
        a += read_array(source / "exact" / f"Y_{start}_{end}.npy", (n, end - start)) @ u[start:end]
    selection = {}
    cheats = {}
    for name in heldout.CHEAP_NAMES:
        c = read_array(source / "cheap" / f"{name}.npy", (n, total), mmap=True)
        mask_predictions = removed @ c
        selection[name] = float(heldout.lds(mask_predictions[:100, :k], loss[:100, :k]).mean())
        cheats[name] = mask_predictions[:, pos].astype(np.float64)
    selected = max(heldout.CHEAP_NAMES, key=selection.__getitem__)
    require(selected == prior["fit"]["selected_cheap"], "cheap method does not match heldout selection")
    require(all(abs(selection[key] - prior["fit"]["cheap_calibration_lds"][key]) < 1e-5 for key in selection), "selection LDS mismatch")
    c = np.load(source / "cheap" / f"{selected}.npy", mmap_mode="r")
    c_measured = c[:, :k] @ u
    alpha, _ = heldout.fit_alpha(a, c_measured)
    require(np.isclose(alpha, prior["fit"]["alpha"], atol=0, rtol=1e-9), "fit alpha differs from heldout run")
    d = coords[:, pos]
    residual = alpha * (cheats[selected] - (removed @ c_measured) @ d)
    try:
        pred, geom = geometric_diagnostics(a, target, removed, d, residual)
        assessment = score_predictions(pred | cheats, loss[100:, pos], geom, pos)
    except ValueError as exc:
        return out | {"assessment": {"strict_status": "UNTESTABLE_PROBE", "reason": str(exc), "mechanism_tags": []}}
    assessment["constant_query_cifar_indices"] = [audit["query_indices"][p] for p in assessment["constant_query_positions"]]
    return out | {"calibration": {"selected": selected, "selected_alpha": alpha, "selection_lds": selection,
                                  "exact_budget": heldout.BUDGET, "new_diagnostic_exact_columns": len(pos)},
                  "geometry": geom, "assessment": assessment}


def self_test(test_dir):
    rng = np.random.default_rng(50)
    a = rng.normal(size=(50, 10))
    d = rng.normal(size=(10, 12))
    target = a @ d
    m = rng.normal(size=(200, 50))
    residual = rng.normal(size=(200, 12)) * .02
    pred, geo = geometric_diagnostics(a, target, m, d, residual)
    np.testing.assert_allclose(geo["weight_projection_energy_fraction"], 1, atol=1e-12)
    np.testing.assert_allclose(geo["test_r2_vs_exact"]["mask_fit_projection"], 1, atol=1e-12)
    np.testing.assert_allclose(pred["scaled_spell_hybrid"], pred["hybrid"], atol=1e-11)
    require(np.isnan(r2(np.ones((4, 1)), np.arange(4)[:, None], allow_constant=True)[0]),
            "undefined optional R² test failed")
    delta = pred["exact"][100:, :].copy()
    delta[:, 2] = 1
    alternatives = {name: pred["exact"] + residual for name in heldout.CHEAP_NAMES}
    score = score_predictions(pred | alternatives, delta, geo, probe_positions())
    require(score["strict_status"] == "UNTESTABLE_CONSTANT_TARGET" and score["query_count"] == 11, "constant target test failed")
    # Preserve coefficient fitting isolation when the held-out values of the target change.
    m_changed = m.copy()
    m_changed[100:] = -m[100:]
    _, alt_geo = geometric_diagnostics(a, target, m_changed, d, residual)
    np.testing.assert_array_equal(geo["beta_nonnegative_scale"], alt_geo["beta_nonnegative_scale"])
    # Fake exact producer receives the real CLI flags and cfg_paths-compatible run root.
    test_dir = test_dir.resolve()
    test_dir.mkdir(exist_ok=False)
    fake = test_dir / "fake_exact.py"
    fake.write_text('''import argparse, json, pathlib, numpy as np
p=argparse.ArgumentParser()
p.add_argument("--cfg"); p.add_argument("--qstart", type=int); p.add_argument("--qend", type=int)
a=p.parse_args()
c=json.loads(pathlib.Path(a.cfg).read_text())
r=pathlib.Path(c["run_dir"])
assert pathlib.Path(c["tag"]).parent == r
(r/"exact").mkdir()
s,e=a.qstart,a.qend
np.save(r/"exact" / f"Y_{s}_{e}.npy", np.full((c["n_train"],e-s),s))
np.save(r/"exact" / f"ql_{s}_{e}.npy", np.arange(s,e))
''')
    cfg = stage_config() | {"n_train": 7}
    for i in range(len(CHUNKS)):
        launch_shard(test_dir, i, cfg, fake, os.environ.copy())
    y, baseline, markers = get_chunks(test_dir, 7, cfg)
    np.testing.assert_array_equal(baseline, probe_positions())
    require(y.shape == (7, 12) and len(markers) == 2, "fake exact chain failed")
    corrupt, _ = chunk_paths(test_dir / "shard-0", *CHUNKS[0])
    with corrupt.open("ab") as stream:
        stream.write(b"tampered")
    try:
        get_chunks(test_dir, 7, cfg)
    except ValueError as exc:
        require("input hash changed" in str(exc), "unexpected tamper test failure")
    else:
        raise AssertionError("altered shard did not fail closed")
    print("Synthetic span/transport/scale, constant target, shard CLI and tamper checks passed")


def main():
    ap = argparse.ArgumentParser(description="Fixed 12-query exact diagnostic")
    sub = ap.add_subparsers(dest="action", required=True)
    for name in ("plan", "prepare", "produce", "score"):
        parser = sub.add_parser(name)
        parser.add_argument("run_id")
        if name != "plan":
            parser.add_argument("commit")
        if name == "produce":
            parser.add_argument("shard", type=int)
    sub.add_parser("self-test").add_argument("test_dir", type=Path)
    args = ap.parse_args()
    if args.action == "self-test":
        self_test(args.test_dir)
        return
    run = run_path(args.run_id)
    if args.action == "plan":
        print(json.dumps(plan(run, stage_config()), indent=2))
    elif args.action == "prepare":
        prepare(run, args.commit)
    elif args.action == "produce":
        manifest = verify_plan(run, args.commit)
        env = os.environ.copy() | {"REQUIRE_GPU": "1", "CIFAR_DIR": str(CIFAR), "JAX_PLATFORMS": "cuda",
                                   "PYTHONDONTWRITEBYTECODE": "1"}
        env.pop("JAX_DEFAULT_MATMUL_PRECISION", None)
        launch_shard(run, args.shard, stage_config(), SCRIPTS / "exact_y.py", env, manifest["source_sha256"])
    else:
        begin = now()
        manifest = verify_plan(run, args.commit)
        output = run / "results.json"
        require(not output.exists(), "result already exists")
        try:
            result = summarize(run, manifest, begin)
        except (ValueError, OSError, KeyError) as exc:
            result = {"read_started_at": begin, "frozen_git_commit": args.commit,
                      "assessment": {"strict_status": "UNTESTABLE_INPUT_PROVENANCE", "reason": str(exc), "mechanism_tags": []}}
        json_create(output, result | {"completed_at": now()})
        print(result["assessment"]["strict_status"], result["assessment"]["mechanism_tags"])


if __name__ == "__main__":
    main()
