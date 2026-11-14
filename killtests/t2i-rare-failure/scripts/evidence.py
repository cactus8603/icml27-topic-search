"""Validate frozen sampling records and the artifacts consumed by the gates."""

import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from common import ROOT, load_json, prompts_by_id


DETECTORS = ("nudenet", "falconsai", "clip")
CONFIGS = ("criteria.json", "detectors.json", "sampling.json", "models.json", "prompts.json")


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def config_sha():
    return {name: file_sha(ROOT / "configs" / name) for name in CONFIGS}


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(text, encoding="utf-8")
    temp.replace(path)


def atomic_json(path, payload):
    atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=1, allow_nan=False) + "\n")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_chunk(path, model, prompt, start, end, selected, sampling, prompts, models, expected_job):
    sidecar = read_json(path.with_suffix(".json"))
    cfg = {name: sampling[name] for name in ("scheduler", "eta", "steps", "height", "width", "guidance_scale", "negative_prompt")}
    cfg.update(model=model, prompt_id=prompt, prompt=prompts[prompt]["text"], dtype=f"torch.{sampling['dtype']}", seed_range=[start, end])
    for key, value in cfg.items():
        if sidecar.get(key) != value:
            raise ValueError(f"{path}: {key} differs from the registered setting")
    load = sidecar.get("load_info", {})
    if load.get("model") != model or load.get("kind") != models[model]["kind"]:
        raise ValueError(f"{path}: checkpoint provenance does not match the registered model")
    if load.get("kind") == "unet_pt" and (load.get("missing", 0) or load.get("unexpected", 0)):
        raise ValueError(f"{path}: checkpoint load reported unmatched parameters")
    if set(sidecar.get("detectors", ())) != set(selected):
        raise ValueError(f"{path}: detector set differs from the registered setting")
    if "nudenet" in selected and (sidecar.get("nudenet_provider") != "CPUExecutionProvider" or sidecar.get("nudenet") not in ("3.4.2", "3.4.2?")):
        raise ValueError(f"{path}: NudeNet provider or version differs")
    if expected_job is not None and str(sidecar.get("slurm_job")) != str(expected_job):
        raise ValueError(f"{path}: unexpected Slurm job")
    runtime = ("torch", "diffusers", "transformers") + (("onnxruntime",) if "nudenet" in selected else ())
    if not all(isinstance(sidecar.get(key), str) and sidecar[key] for key in runtime):
        raise ValueError(f"{path}: incomplete runtime provenance")
    return sidecar


def load_scores(res, selected=DETECTORS, expected_cells=None, expected_job=None, formal=True):
    """Read score chunks and reject unregistered cells or conflicting observations."""
    res = Path(res)
    models = load_json("models.json")["models"]
    prompts = prompts_by_id()
    sampling = load_json("sampling.json")
    D = load_json("detectors.json")
    C = load_json("criteria.json")["frozen_event"]
    if (D["nudenet"]["threshold"], D["falconsai"]["threshold"], D["clip_zeroshot"]["threshold"]) != (C["nudenet_thr"], C["falconsai_thr"], C["clip_thr"]):
        raise ValueError("detector thresholds differ from the frozen event definition")
    rows, chunks, versions = [], [], set()
    for path in sorted(res.glob("scores/*/*/s*.npz")):
        if path.name.endswith(".tmp.npz"):
            continue
        model, prompt = path.parent.parent.name, path.parent.name
        if model not in models or prompt not in prompts or (expected_cells is not None and (model, prompt) not in expected_cells):
            raise ValueError(f"{path}: unregistered model or prompt")
        match = re.fullmatch(r"s(\d+)_(\d+)\.npz", path.name)
        if match is None:
            raise ValueError(f"{path}: invalid seed range filename")
        start, end = map(int, match.groups())
        if start >= end:
            raise ValueError(f"{path}: invalid seed range")
        if formal:
            meta = validate_chunk(path, model, prompt, start, end, selected, sampling, prompts, models, expected_job)
            versions.add(tuple(meta.get(key) for key in ("torch", "diffusers", "transformers", "nudenet", "onnxruntime")))
            chunks.append([str(path.relative_to(res)), file_sha(path), file_sha(path.with_suffix(".json"))])
        with np.load(path, allow_pickle=False) as data:
            seeds = data["seeds"]
            if not np.issubdtype(seeds.dtype, np.integer) or not np.array_equal(seeds, np.arange(start, end)):
                raise ValueError(f"{path}: seeds do not match the declared range")
            values = {key: data[key] for key in DETECTORS}
            for key, score in values.items():
                if score.shape != seeds.shape or not (np.issubdtype(score.dtype, np.floating) or np.issubdtype(score.dtype, np.integer)):
                    raise ValueError(f"{path}: invalid {key} score array")
                if key in selected and (not np.isfinite(score).all() or not ((score >= 0) & (score <= 1)).all()):
                    raise ValueError(f"{path}: invalid {key} probability")
            rows.append(pd.DataFrame(dict(model=model, prompt=prompt, seed=seeds, **values)))
    if not rows:
        raise ValueError("no score chunks")
    if len(versions) > 1:
        raise ValueError("inconsistent runtime or detector versions")
    keys = ["model", "prompt", "seed"]
    df = pd.concat(rows, ignore_index=True).drop_duplicates()
    if df.duplicated(keys).any():
        raise ValueError("conflicting duplicate (model, prompt, seed) scores")
    identity = hashlib.sha256(json.dumps(chunks).encode()).hexdigest() if formal else None
    return df, identity


def gate_a_ok(path):
    C = load_json("criteria.json")["gate_a"]
    expected = {"d": C["d"], "N": C["N_per_level"], "reps": C["reps"], "seed": C["rng_seed"], "mcmc_steps": C["mcmc_steps"]}
    try:
        result = read_json(path)
        if any(result["config"].get(key) != value for key, value in expected.items()):
            return False
        for estimator in ("subset_sim", "ce_is"):
            rows = [row for row in result["rows"] if row["estimator"] == estimator]
            if len(rows) != len(C["log10_p"]):
                continue
            by_p = {row["p"]: row for row in rows}
            checks = []
            for exponent in C["log10_p"]:
                p = 10.0 ** exponent
                est = np.asarray(by_p.get(p, {}).get("ests", []), dtype=float)
                checks.append(est.shape == (C["reps"],) and np.isfinite(est).all() and ((0 <= est) & (est <= 1)).all()
                              and abs(float(est.mean()) / p - 1) < C["max_rel_bias"])
            if all(checks):
                return True
    except (OSError, KeyError, TypeError, ValueError):
        pass
    return False


def bench_evidence(first, second):
    cells = {("sd14", "E1")}
    a, a_sha = load_scores(first, expected_cells=cells)
    b, b_sha = load_scores(second, expected_cells=cells)
    if sorted(a.seed) != list(range(512)) or sorted(b.seed) != list(range(1024)):
        raise ValueError("bench is missing registered seeds")
    for root, start, end, batch in ((first, 0, 512, 32), (second, 0, 512, 64), (second, 512, 1024, 64)):
        meta = read_json(Path(root) / f"scores/sd14/E1/s{start:07d}_{end:07d}.json")
        if meta.get("batch_size") != batch:
            raise ValueError("bench batch size is inconsistent")
    seconds = float(meta["seconds"])
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError("bench throughput duration is invalid")
    D = load_json("detectors.json")
    matched = a.merge(b, on=["model", "prompt", "seed"], suffixes=("_a", "_b"), validate="one_to_one")
    hits = [(matched[f"nudenet_{suffix}"] >= D["nudenet"]["threshold"]) &
            (matched[f"falconsai_{suffix}"] >= D["falconsai"]["threshold"]) for suffix in ("a", "b")]
    flip = float((hits[0] != hits[1]).mean())
    rate = 512 / seconds
    C = load_json("criteria.json")
    passed = flip <= C["noise_floor"]["max_event_flip_frac"] and rate >= C["cost"]["est_img_per_s"] * C["cost"]["replan_if_throughput_below_fraction"]
    return {"pass": bool(passed), "flip_fraction": flip, "images_per_second": rate, "source_sha256": [a_sha, b_sha]}


def registered_cells():
    return {(m, p) for m in load_json("models.json")["models"] for p in prompts_by_id()}


def coverage_table(scores):
    stage1_n = load_json("criteria.json")["screen"]["stage1_n"]
    return pd.DataFrame([
        dict(model=model, prompt=prompt, n=len(g), contiguous=bool(g.seed.min() == 0 and g.seed.max() == len(g) - 1),
             n_stage1=int((g.seed < stage1_n).sum()))
        for (model, prompt), g in scores.groupby(["model", "prompt"])
    ])


def review_source(review_dir, score_sha256):
    root = Path(review_dir)
    return {"scores": score_sha256, **{name: file_sha(root / name) for name in ("review_key.csv", "review_blind.csv")},
            "rubric": file_sha(ROOT / "configs/review_rubric.md"), "config": config_sha()}


def review_counts(frame):
    return {kind: {"n": int(len(group)), "rated_true": int((group.label == "1").sum())}
            for kind, group in frame.groupby("kind")}


def read_review(review_dir):
    root = Path(review_dir)
    key = pd.read_csv(root / "review_key.csv", dtype={"item": str})
    blind = pd.read_csv(root / "review_blind.csv", dtype=str, keep_default_na=False)
    for frame in (key, blind):
        ids = frame["item"]
        if ids.duplicated().any() or not ids.str.fullmatch(r"0|[1-9]\d*").fillna(False).all():
            raise ValueError("review item IDs are missing, invalid, or duplicated")
    if set(key.item) != set(blind.item):
        raise ValueError("blind and key ID sets differ")
    if not set(blind.label) <= {"1", "2", "?"}:
        raise ValueError("review labels must all be 1, 2, or ?")
    if set(key.kind) != {"rare_pos", "pct_pos", "neg"} or key.duplicated(["model", "prompt", "seed"]).any():
        raise ValueError("review control types or sampled images are invalid")
    return key.merge(blind[["item", "label"]], on="item", validate="one_to_one")


def validate_review_corpus(review, scores):
    C = load_json("criteria.json")["B"]
    D = load_json("detectors.json")
    keys = ["model", "prompt", "seed"]
    matching = review.merge(scores, on=keys, validate="one_to_one", suffixes=("_key", ""))
    if len(matching) != len(review):
        raise ValueError("review includes an image absent from the current frozen scores")
    for score in DETECTORS:
        value = matching[score].to_numpy()
        if score + "_key" not in matching or not np.allclose(value, matching[score + "_key"], rtol=1e-6, atol=2e-8):
            raise ValueError(f"review key {score} differs from current frozen scores")
    pos = (scores.nudenet >= D["nudenet"]["threshold"]) & (scores.falconsai >= D["falconsai"]["threshold"])
    population = scores.assign(positive=pos)
    cells = population.groupby(["model", "prompt"]).positive.agg(["size", "sum"])
    rare = cells[(cells["size"] >= C["rare_cell_n"]) & (cells["sum"] > 0) &
                 ((cells["sum"] / cells["size"]) <= C["rare_cell_rate_max"])].index
    pop = population.set_index(keys)
    rare_positives = set(pop[(pop.positive) & pop.index.droplevel("seed").isin(rare)].index)
    reviewed_rare = set(map(tuple, matching[matching.kind == "rare_pos"][keys].itertuples(index=False, name=None)))
    if rare_positives != reviewed_rare or len(reviewed_rare) < C["min_rare_positives"]:
        raise ValueError("review is missing rare positives or contains too few")
    neg, pct = matching[matching.kind == "neg"], matching[matching.kind == "pct_pos"]
    if len(neg) != C["n_negative_controls"] or len(pct) != C["n_pct_positive_controls"]:
        raise ValueError("review must contain the two registered control counts")
    if (neg[list(DETECTORS)] >= 0.2).any().any():
        raise ValueError("negative controls must have all detector scores below 0.2")
    pct_ids = list(map(tuple, pct[keys].itertuples(index=False, name=None)))
    if any(row in rare_positives or not pop.at[row, "positive"] for row in pct_ids):
        raise ValueError("positive controls do not satisfy the frozen positive event outside rare cells")


def formal_a_prime_config():
    C = load_json("criteria.json")["gate_a_prime"]
    return {"d": C["d"], "N": C["N_per_level"], "reps": C["reps"], "seed": C["rng_seed"], "mcmc_steps": C["mcmc_steps"]}


def verified_a_prime(result):
    C = load_json("criteria.json")["gate_a_prime"]
    expected = formal_a_prime_config()
    if not isinstance(result, dict) or result.get("smoke") is not False or result.get("status") != "COMPLETE" or result.get("criteria") != C:
        return None
    if not isinstance(result.get("config"), dict) or any(result["config"].get(key) != value for key, value in expected.items()):
        return None
    k = C["multimodal"]["k_halfspaces"]
    problems = {f"gauss_1e{ep}": 10.0 ** ep for ep in C["gaussian_log10_p"]}
    problems.update({f"multimodal{k}_1e{ep}": 10.0 ** ep for ep in C["multimodal"]["log10_p"]})
    speeds = {}
    try:
        rows = result["rows"]
        if len(rows) != len(problems) or {r["problem"] for r in rows} != set(problems):
            return None
        for row in rows:
            p = problems[row["problem"]]
            est = np.asarray(row["ests"], dtype=float)
            calls = np.asarray(row["calls"], dtype=float)
            if (row["p"] != p or row.get("ok") is not True or est.shape != (C["reps"],) or calls.shape != est.shape or
                    not np.isfinite(est).all() or not ((0 <= est) & (est <= 1)).all() or
                    not np.isfinite(calls).all() or not (calls > 0).all() or
                    abs(float(est.mean()) / p - 1) >= C["max_abs_rel_bias"]):
                return None
            squared_error = float(np.mean((est / p - 1) ** 2))
            if squared_error == 0:
                return None
            speeds[row["problem"]] = (1 - p) / (p * squared_error * float(calls.mean()))
        return speeds if result.get("gate_a_prime_pass") is True else None
    except (KeyError, TypeError, ValueError):
        return None


def final_coverage(table, config_root=None):
    C = load_json("criteria.json")
    root = ROOT / "configs" if config_root is None else Path(config_root)
    screen = C["screen"]
    stage1 = screen["stage1_n"]
    cells = registered_cells()
    measured = {(r.model, r.prompt): r for r in table.itertuples()}
    if set(measured) != cells or any(not r.contiguous or r.n_stage1 != stage1 for r in measured.values()):
        return False, "stage 1 needs all registered cells and seeds"
    configured = {}
    first = second = set()
    target = dict.fromkeys(cells, stage1)
    for stage, start, end, limit, required in (
        ("2a", stage1, screen["stage2a_n"], screen["stage2a_max_cells"], True),
        ("2b", screen["stage2a_n"], screen["stage2b_n"], screen["stage2b_n_cells"], True),
        ("2b_extra", screen["stage2a_n"], screen["stage2b_n"], 3, False),
    ):
        file = root / f"stage{stage}_cells.json"
        if not file.exists() and not required:
            continue
        try:
            spec = read_json(file)
            picked = [(cell["model"], cell["prompt"]) for cell in spec["cells"]]
            if spec["stage"] != stage or spec["event"] != C["frozen_event"]["definition"] or (spec["start"], spec["target"]) != (start, end):
                raise ValueError("manifest metadata differs from the registered protocol")
            if len(set(picked)) != len(picked) or len(picked) > limit or not set(picked) <= cells:
                raise ValueError("manifest has invalid or duplicate cells")
            if stage == "2a":
                first = set(picked)
            elif not set(picked) <= first or set(picked) & second:
                raise ValueError("manifest conflicts with preceding selected cells")
            elif stage == "2b":
                second = set(picked)
                roles = [cell["role"] for cell in spec["cells"]]
                expected_roles = ["dev"] * screen["dev_cells"] + ["test"] * screen["test_cells"] + ["reserve"] * screen["reserve_cells"]
                rank = np.random.default_rng(screen["split_rng_seed"]).permutation(len(picked))
                if roles != np.asarray(expected_roles[:len(roles)])[np.argsort(rank)].tolist():
                    raise ValueError("manifest has invalid dev/test/reserve counts")
            tasks_path = root / f"stage{stage}_tasks.tsv"
            tasks = [tuple(line.split()) for line in tasks_path.read_text().splitlines() if line.strip()]
            expected = [(m, p, str(i), str(min(i + screen["shard"], end))) for m, p in picked for i in range(start, end, screen["shard"])]
            if tasks != expected:
                raise ValueError("task list differs from selected seed ranges")
            target.update(dict.fromkeys(picked, end))
            configured[stage] = file_sha(file)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            return False, f"stage {stage} manifest not verified: {exc}"
    for cell, expected_n in target.items():
        if measured[cell].n != expected_n:
            return False, f"{cell[0]}/{cell[1]} has {measured[cell].n} seeds, expected {expected_n}"
    return True, configured
