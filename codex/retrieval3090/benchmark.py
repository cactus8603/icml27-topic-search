"""[codex] Component cost benchmark; never evaluates retrieval effectiveness.

Fresh query encoding is timed separately from frozen cached-vector admission.
The latter recomputes dot products and whole-gallery score normalization. Cross
scoring uses the frozen candidate pool and discards model scores after checking
their shape/finite values. Batched throughput is not single-query latency.
"""
from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import re
import subprocess
from time import perf_counter

import numpy as np

SEED = 2026111665
REPETITIONS = 2
QUERY_COUNT = 8
BUDGETS = (25, 50, 75, 100, 150, 200)
MODELS = ("core_embed", "cs_clip")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def fingerprint(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def safe(path):
    path = Path(path).expanduser().resolve()
    if any("retrieval-independent-panel" in part for part in path.parts):
        raise ValueError("The sealed panel is forbidden in a development cost benchmark")
    return path


def atomic_json(path, value):
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("w") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    partial.replace(path)


class Store:
    """One atomically committed record per completed task; interrupted tasks retry."""
    def __init__(self, out, identity):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.path = self.out / "checkpoint.json"
        self.identity = identity
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {
            "schema": 1, "identity": identity, "records": {}, "loads": []}
        if self.data.get("identity") != identity or self.data.get("schema") != 1:
            raise ValueError("Benchmark checkpoint source/environment/input provenance changed")
        atomic_json(self.path, self.data)

    def completed(self, key):
        return key in self.data["records"]

    def put(self, key, value):
        if self.completed(key):
            raise ValueError("A completed cost measurement cannot be silently replaced")
        self.data["records"][key] = value
        atomic_json(self.path, self.data)
        atomic_json(self.out / "progress.json", {"scope": "cost only", "last_completed": key,
                                                 "completed_tasks": len(self.data["records"])})

    def load_event(self, value):
        self.data["loads"].append(value)
        atomic_json(self.path, self.data)


def timed(call, *, synchronize=lambda: None, clock=perf_counter):
    synchronize()
    start = clock()
    value = call()
    synchronize()
    seconds = float(clock() - start)
    if not np.isfinite(seconds) or seconds < 0:
        raise ValueError("Invalid clock measurement")
    return value, seconds


def selected_queries(queries, count=QUERY_COUNT):
    if len(queries) < count:
        raise ValueError("Not enough queries for the fixed cost sample")
    return sorted(range(len(queries)), key=lambda index: (
        fingerprint([SEED, queries[index]["text"], queries[index]["key"]]), index))[:count]


def task_order(tasks, label):
    return sorted(tasks, key=lambda task: fingerprint([SEED, label, task]))


def zscore(value):
    std = value.std(axis=0, keepdims=True)
    if np.any(std <= 0):
        raise ValueError("Constant gallery similarities cannot be normalized")
    return (value - value.mean(axis=0, keepdims=True)) / std


def validate_vectors(cache, panel):
    """Technical label-free alignment checks, without opening cross score files."""
    count, queries = len(panel["ids"]), len(panel["queries"])
    index = np.asarray(panel["condition_index"])
    if len(set(panel["ids"])) != count or index.shape != (queries, 2) or index.dtype.kind not in "iu":
        raise ValueError("Malformed candidate IDs or two-condition index")
    for model in MODELS:
        image, full, cond = (np.asarray(cache[f"{model}_{part}"]) for part in ("image", "full", "condition"))
        if (image.ndim != 2 or image.shape[0] != count or full.shape != (queries, image.shape[1])
                or cond.shape != (len(panel["condition_texts"]), image.shape[1])
                or np.any(index < 0) or np.any(index >= len(cond))):
            raise ValueError("Benchmark vector rows are not aligned with the frozen panel")
        if not all(np.isfinite(value).all() and np.allclose(np.linalg.norm(value, axis=-1), 1, atol=.005)
                   for value in (image, full, cond)):
            raise ValueError("Benchmark requires finite unit-normalized index and query vectors")
    return index


def cpu_features(cache, condition_index, query, logic_vectors=None, *, clock=perf_counter):
    """Actual label-free gallery dot products, whole-pool z-scores and search."""
    output, components = {}, {}
    for model in MODELS:
        image = cache[f"{model}_image"]
        def full_search():
            full = (cache[f"{model}_full"][query:query + 1] @ image.T)[0]
            return full, zscore(full)
        (full, full_z), components[f"{model}/full_dot_and_zscore"] = timed(full_search, clock=clock)
        def condition_search():
            conditions = (cache[f"{model}_condition"][condition_index[query]] @ image.T).T
            return zscore(conditions)
        condition_z, components[f"{model}/two_condition_dot_and_zscore"] = timed(condition_search, clock=clock)
        output[model] = {"full_raw": full, "full_z": full_z, "condition_z": condition_z}
        if logic_vectors is not None:
            output[model]["logic_condition_raw"], components[f"{model}/four_template_mean_dot"] = timed(
                lambda: (logic_vectors[f"{model}_lcse_condition"][condition_index[query]] @ image.T).T, clock=clock)
    # Real stable search is included, even though admission below retains the
    # original all-query GEMM order to avoid changing frozen candidate membership.
    output["computed_core_order"], components["core_embed/stable_gallery_search"] = timed(
        lambda: np.argsort(-output["core_embed"]["full_raw"], kind="stable"), clock=clock)
    output["component_seconds"] = components
    return output


def percentile(value):
    order = np.argsort(value, kind="stable")
    sorted_value = value[order]
    edges = np.r_[0, 1 + np.flatnonzero(sorted_value[:-1] != sorted_value[1:]), len(value)]
    means = (edges[:-1] + edges[1:] + 1) / (2 * len(value))
    result = np.empty(len(value), dtype=float)
    result[order] = np.repeat(means, np.diff(edges))
    return result


def policy_names(with_logic=False):
    base = ["core", "cs_full"]
    for model in ("core", "cs"):
        base += [f"min_cond_{model}"] + [f"logsigmoid_{model}_t{tau:g}" for tau in (.5, 1., 2.)]
    for name in ("min_cond_core", "min_cond_cs", "logsigmoid_core_t1", "logsigmoid_cs_t1", "cs_full"):
        base += [f"fuse_{name}_w{weight:g}" for weight in (.25, .5, .75)]
    if with_logic:
        for model in MODELS:
            for variant in ("paperparams_transfer", "train_presence_calibrated"):
                base += [f"logic_{model}_{variant}_{kind}" for kind in
                         ("holistic", "mean_cond", "min_cond", "product_cond", "logsigmoid_cond", "lcse_and")]
    return [f"m1/top{depth}/{name}" for depth in (100, 200) for name in base]


def policy_scores(name, features, top, parameters=None):
    if name.startswith("fuse_"):
        match = re.fullmatch(r"fuse_(.+)_w(0\.25|0\.5|0\.75)", name)
        if not match:
            raise ValueError("Unsupported rank-fusion policy")
        weight = float(match[2])
        score = policy_scores(match[1], features, top, parameters)
        return (1 - weight) * percentile(-np.arange(len(top), dtype=float)) + weight * percentile(score)
    if name == "core":
        return -np.arange(len(top), dtype=float)
    if name == "cs_full":
        return features["cs_clip"]["full_z"][top]
    match = re.fullmatch(r"min_cond_(core|cs)|logsigmoid_(core|cs)_t(0\.5|1|2)", name)
    if match:
        short = match[1] or match[2]
        cond = features["core_embed" if short == "core" else "cs_clip"]["condition_z"][top]
        return cond.min(axis=-1) if match[1] else -np.logaddexp(0, -cond / float(match[3])).sum(axis=-1)
    match = re.fullmatch(r"logic_(core_embed|cs_clip)_(paperparams_transfer|train_presence_calibrated)_(.+)", name)
    if not match:
        raise ValueError("Unsupported M1 policy; use the corrected logsigmoid policy IDs")
    model, variant, kind = match.groups()
    options = {"mu": .22, "beta": 30., "gamma": -1.} if variant == "paperparams_transfer" else (parameters or {}).get(model)
    if options is None or "logic_condition_raw" not in features[model]:
        raise ValueError("Logic policy requires independently calibrated four-template vectors")
    full = features[model]["full_raw"][top]
    cond = features[model]["logic_condition_raw"][top]
    # Use the analysis helper itself: its float64 conversion is material for
    # product rankings when sigmoid values underflow in float32. Computing the
    # shared helper's six controls is included in this conservative CPU timing.
    from logic_baselines import aggregate_scores
    controls = aggregate_scores(full, cond, **options)
    if kind not in controls:
        raise ValueError("Unsupported logic aggregation")
    return controls[kind]


def admit(policy, features, frozen_order, parameters=None):
    match = re.fullmatch(r"m1/top(100|200)/(.+)", policy)
    if not match or len(frozen_order) < int(match[1]):
        raise ValueError("Policy needs a top100/top200 frozen candidate pool")
    top = frozen_order[:int(match[1])]
    scores = np.asarray(policy_scores(match[2], features, top, parameters))
    if scores.shape != top.shape or not np.isfinite(scores).all():
        raise ValueError("Invalid policy scores")
    selected = np.lexsort((np.arange(len(top)), -scores))[:50]
    return top[np.sort(selected)]


def encode_tasks(model_name, queries):
    return task_order([[model_name, query, repetition, kind] for repetition in range(REPETITIONS)
                       for query in queries for kind in ("full", "two_conditions", "four_templates")], "encoding")


def run_cross_tasks(store, scorer, tasks, sampled, panel, orders, original, batch_size,
                    *, selected_policy=None, selected_prefixes=None, synchronize=lambda: None,
                    clock=perf_counter):
    """One global batch-packed pass per budget/repetition, RGB decode included."""
    for arm, repetition in tasks:
        key = f"cross/{arm}/rep{repetition}"
        if store.completed(key):
            continue
        query_order = task_order(list(sampled), f"cross/{arm}/rep{repetition}/queries")
        budget = int(arm[1:]) if arm.startswith("b") else 50
        prefixes = [orders[q, :budget] if arm != "policy50" else selected_prefixes[q] for q in query_order]
        captions = [panel["queries"][query]["text"] for query in query_order for _ in range(budget)]
        ids = [panel["ids"][int(index)] for prefix in prefixes for index in prefix]
        images = [lambda image_id=image_id: original.rgb(original.image_path(image_id)) for image_id in ids]
        scores, seconds = timed(lambda: original.cross_scores(scorer, captions, images, batch_size),
                                synchronize=synchronize, clock=clock)
        scores = np.asarray(scores)
        if scores.shape != (len(sampled) * budget,) or not np.isfinite(scores).all():
            raise ValueError("Reranker produced invalid technical output")
        store.put(key, {"component": "warm_batch_cross_rerank_with_rgb_decode", "arm": arm,
            "repetition": repetition, "budget": budget, "query_indices": query_order,
            "seconds": seconds, "pairs": len(ids), "mean_seconds_per_query": seconds / len(sampled),
            "candidate_ids_sha256": fingerprint(ids), "policy": selected_policy if arm == "policy50" else None,
            "batch_size": batch_size, "packing": "global sampled-query order, batch <= 4 across query boundaries",
            "scope": "Batched component throughput, not single-query end-to-end latency"})
        print(json.dumps({"cost_stage": key, "seconds": seconds}), flush=True)


def package_version(name):
    try:
        return version(name)
    except Exception:
        return None


def driver_version():
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                capture_output=True, text=True, check=True, timeout=5)
        return sorted(set(result.stdout.strip().splitlines()))
    except (OSError, subprocess.SubprocessError):
        return None


def main():
    import torch
    import cache_scores as legacy
    import analysis_v2 as analysis

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--lcse-vectors", type=Path)
    parser.add_argument("--lcse-calibration", type=Path)
    parser.add_argument("--policy")
    args = parser.parse_args()
    if bool(args.lcse_vectors) != bool(args.lcse_calibration):
        parser.error("The LCSE vectors and calibration artifact must be provided together")
    cache_dir, out = safe(args.cache), safe(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "writer.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        began = perf_counter()
        frozen_panel, config, snapshots, context = legacy.score.preflight()
        panel = json.loads((cache_dir / "panel.json").read_text())
        meta = json.loads((cache_dir / "meta.json").read_text())
        if meta.get("replay", {}).get("state") != "PASS":
            raise ValueError("Cache baseline must pass replay before cost benchmarking")
        if (panel["ids"] != frozen_panel["ids"]
                or [(q["text"], q["key"]) for q in panel["queries"]] != [(q["text"], q["key"]) for q in frozen_panel["queries"]]
                or meta.get("revisions") != context["revisions"]
                or meta.get("model_config_sha256") != context["model_config_sha256"]):
            raise ValueError("Benchmark cache does not match the frozen exposed panel/model sources")
        cache = dict(np.load(cache_dir / "vectors.npz", allow_pickle=False))
        index = validate_vectors(cache, panel)
        expected_texts, expected_index = legacy.condition_texts(frozen_panel)
        if panel["condition_texts"] != expected_texts or panel["condition_index"] != expected_index:
            raise ValueError("Benchmark condition text/order differs from the original cache protocol")
        sampled = selected_queries(panel["queries"])
        # Preparation only, not an online timing: reproduce the original all-query
        # GEMM/ranking to keep every cost arm on the same frozen candidate pool.
        orders = legacy.original.ranks(cache["core_embed_full"] @ cache["core_embed_image"].T)
        parameters, logic_vectors = None, None
        inputs = {name: file_sha(cache_dir / name) for name in ("vectors.npz", "panel.json", "meta.json")}
        if args.lcse_vectors:
            import logic_baselines as logic
            vectors_path, calibration_path = safe(args.lcse_vectors), safe(args.lcse_calibration)
            logic_vectors = dict(np.load(vectors_path, allow_pickle=False))
            calibration = json.loads(calibration_path.read_text())
            parameters = {model: logic.validate_calibration_artifact(calibration, model, excluded_image_ids=panel["ids"])
                          for model in MODELS}
            analysis.validate_logic_vectors(cache, panel, meta, calibration, logic_vectors, logic)
            inputs.update(lcse_vectors=file_sha(vectors_path), lcse_calibration=file_sha(calibration_path))
        policies = policy_names(logic_vectors is not None)
        if args.policy is not None and args.policy not in policies:
            raise ValueError("Unsupported policy or missing verified LCSE artifacts")
        if config["batch_size"]["cross_pair"] != 4:
            raise ValueError("This cost protocol requires the frozen cross batch size 4")
        environment = {"python": platform.python_version(), "platform": platform.platform(),
            "packages": {name: package_version(name) for name in ("numpy", "torch", "transformers", "open_clip_torch")},
            "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "device": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0)),
            "device_memory": torch.cuda.get_device_properties(0).total_memory,
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "driver_versions": driver_version(),
            "cpu_threads": {"torch": torch.get_num_threads(), "torch_interop": torch.get_num_interop_threads(),
                **{key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}}}
        identity = {"seed": SEED, "repetitions": REPETITIONS, "sampled_queries": sampled,
            "inputs": inputs, "environment": environment, "config": config,
            "source": os.environ["DEV_SOURCE"], "freeze": os.environ["DEV_FREEZE"],
            "commit": os.environ["DEV_COMMIT"], "dirty": os.environ["DEV_DIRTY"], "policy": args.policy}
        if identity["dirty"] != "0":
            raise ValueError("Cost measurements require committed source")
        store = Store(out, identity)
        sync = torch.cuda.synchronize
        store.load_event({"component": "invocation_preflight_and_input_loading", "seconds": perf_counter() - began})

        # Online CPU components really recompute from cached query/index vectors;
        # their boundary with separately timed fresh-query encoding is explicit.
        selected_prefixes = {}
        for query, repetition in task_order([[q, r] for q in sampled for r in range(REPETITIONS)], "cpu"):
            task = f"cpu/search/query{query}/rep{repetition}"
            features, seconds = timed(lambda q=query: cpu_features(cache, index, q, logic_vectors))
            if not store.completed(task):
                store.put(task, {"component": "CPU dot products, whole-gallery zscore, stable CORE search",
                    "seconds": seconds, "query_index": query, "repetition": repetition,
                    "gallery_images": len(panel["ids"]), "query_vector_source": "frozen cached vectors",
                    "per_component_seconds": features["component_seconds"],
                    "scope": "Joint features for every policy; add only the model/components needed by a selected policy"})
            for policy in task_order(policies, f"cpu/admission/{query}/{repetition}"):
                task = f"cpu/admission/{policy}/query{query}/rep{repetition}"
                if store.completed(task):
                    continue
                prefix, seconds = timed(lambda p=policy: admit(p, features, orders[query], parameters))
                store.put(task, {"component": "CPU policy scoring helper plus admission selection", "seconds": seconds,
                    "query_index": query, "repetition": repetition, "policy": policy,
                    "logic_helper_scope": "Shared float64 six-control helper overhead included" if "/logic_" in policy else None,
                    "selected_candidate_indices_sha256": fingerprint(prefix.tolist())})
            if args.policy:
                selected_prefixes[query] = admit(args.policy, features, orders[query], parameters)

        for model_name, model_class in (("core_embed", legacy.original.Encoder), ("cs_clip", legacy.original.ClipEncoder)):
            tasks = encode_tasks(model_name, sampled)
            if all(store.completed(f"encode/{name}/query{query}/rep{rep}/{kind}") for name, query, rep, kind in tasks):
                continue
            torch.cuda.reset_peak_memory_stats()
            model, load_seconds = timed(lambda: model_class(model_name, snapshots[model_name], config, "cuda"), synchronize=sync)
            _, warmup_seconds = timed(lambda: model.encode([panel["queries"][sampled[0]]["text"]]), synchronize=sync)
            store.load_event({"component": "new_model_instance_load", "model": model_name, "seconds": load_seconds,
                "warmup_seconds": warmup_seconds, "filesystem_cache_state": "uncontrolled; not guaranteed disk-cold"})
            for name, query, repetition, kind in tasks:
                task = f"encode/{name}/query{query}/rep{repetition}/{kind}"
                if store.completed(task):
                    continue
                if kind == "full":
                    texts = [panel["queries"][query]["text"]]
                elif kind == "two_conditions":
                    texts = [panel["condition_texts"][i] for i in index[query]]
                else:
                    from logic_baselines import CONCEPT_TEMPLATES
                    names = {int(cat): value for cat, value in frozen_panel["names"].items()}
                    texts = [template.format(names[cat]) for cat in panel["queries"][query]["key"] for template in CONCEPT_TEMPLATES]
                vectors, seconds = timed(lambda: legacy.original.unit_scores(model, texts, config["batch_size"]["embed_text"]), synchronize=sync)
                if np.asarray(vectors).shape[0] != len(texts) or not np.isfinite(vectors).all():
                    raise ValueError("Invalid technical text encoder output")
                store.put(task, {"component": "warm fresh-query text encoding", "model": name, "kind": kind,
                    "seconds": seconds, "query_index": query, "repetition": repetition, "texts": len(texts),
                    "text_sha256": fingerprint(texts), "batch_size": config["batch_size"]["embed_text"],
                    "scope": "Encoding component only; resulting vectors do not replace frozen admission inputs"})
            del model
            gc.collect()
            torch.cuda.empty_cache()

        arms = [f"b{budget}" for budget in BUDGETS] + (["policy50"] if args.policy else [])
        tasks = task_order([[arm, rep] for arm in arms for rep in range(REPETITIONS)], "cross")
        if any(not store.completed(f"cross/{arm}/rep{rep}") for arm, rep in tasks):
            torch.cuda.reset_peak_memory_stats()
            scorer, seconds = timed(lambda: legacy.original.CrossScorer("core_reranker", snapshots["core_reranker"], config, "cuda"), synchronize=sync)
            first_query = sampled[0]
            warm_ids = [panel["ids"][int(i)] for i in orders[first_query, :4]]
            _, warmup = timed(lambda: legacy.original.cross_scores(scorer,
                [panel["queries"][first_query]["text"]] * 4,
                [lambda key=key: legacy.original.rgb(legacy.original.image_path(key)) for key in warm_ids], 4), synchronize=sync)
            store.load_event({"component": "new_model_instance_load", "model": "core_reranker", "seconds": seconds,
                              "warmup_seconds": warmup, "filesystem_cache_state": "uncontrolled"})
            run_cross_tasks(store, scorer, tasks, sampled, panel, orders, legacy.original, 4,
                selected_policy=args.policy, selected_prefixes=selected_prefixes, synchronize=sync)
            del scorer
            gc.collect()
            torch.cuda.empty_cache()
        report = {"schema": 1, "scope": "component cost measurements only; no retrieval effectiveness evaluated",
            "identity": identity, "model_loads_and_startup": store.data["loads"], "measurements": store.data["records"],
            "indexed_storage_bytes": {name: int(value.nbytes) for name, value in cache.items()},
            "boundary": "Fresh encoding is measured separately; CPU search/admission and reranking retain frozen cached-vector candidates. "
                        "Independent component sums are not an observed end-to-end pass or single-query latency.",
            "indexing": "Prebuilt image embeddings are assumed; image encoding and index construction are excluded from online components.",
            "reuse": "Cached condition embeddings remove two-condition/four-template query encoding; cached full query embeddings remove full-query encoding. "
                     "Fresh-query use pays the corresponding measured encoding components. Model switching pays separately recorded model loads.",
            "packing": "Reranker uses frozen batch4 over eight sampled queries. CPU admission is per query. GPU sync brackets every timed call.",
            "resume": "Completed tasks are reused only with equal source, input SHA, policy and environment. Loads from each invocation remain separate."}
        atomic_json(out / "report.json", report)
        atomic_json(out / "progress.json", {"stage": "complete", "scope": "cost only", "tasks": len(store.data["records"])})
        print(out / "report.json")


if __name__ == "__main__":
    main()
