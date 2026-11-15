"""[codex] Exact baseline-cross reuse around the unchanged frozen MCMR runner.

First produce a direct mcmr_pilot.py m1/top200/core cache, sharing --index.
After COCO selection is locked, use this entry point for the selected policy:
  --stage prepare|cache|analyze --baseline BASE --index INDEX
  --selection-lock SELECTED_POLICY_JSON --data DATA --images IMAGES --out RUN
  --policy m1/topD/NAME

Only cross_union's batch-1 compute is intercepted. Eight common pairs are always
freshly evaluated and must match exactly before any lookup. All other inference,
admission, pair order, checkpoint writing, and analysis use the frozen runner.
Use this entry point for analysis too: native meta explicitly requires its reuse
receipt. This is a local external development pilot, not H200 reproduction.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import fcntl
import hashlib
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace

import numpy as np

import mcmr_pilot as M

SCHEMA = "mcmr-baseline-cross-reuse-v1"
PILOT_COUNTS = (128, 8320)
REPEAT_COUNT = 8
HEX64 = re.compile(r"[0-9a-f]{64}\Z")


def wrapper_source():
    """Do not run a producer or trust a receipt from uncommitted wrapper bytes."""
    path = Path(__file__).resolve()
    relative = path.relative_to(M.ROOT).as_posix()
    body = path.read_bytes()
    if M.git_bytes("HEAD", relative) != body:
        raise ValueError("Reuse wrapper must be committed before cache or analyze")
    return {"path": relative, "sha256": hashlib.sha256(body).hexdigest(),
            "commit": subprocess.check_output(["git", "-C", str(M.ROOT), "rev-parse", "HEAD"], text=True).strip()}


def validate_wrapper_commit(commit, source):
    if (not re.fullmatch(r"[0-9a-f]{40}", str(commit))
            or source.get("path") != "codex/retrieval3090/mcmr_reuse.py"
            or hashlib.sha256(M.git_bytes(commit, source["path"])).hexdigest() != source.get("sha256")):
        raise ValueError("Receipt wrapper execution commit does not contain the bound source bytes")


def array_hash(array):
    array = np.ascontiguousarray(array)
    return M.digest({"shape": list(array.shape), "dtype": array.dtype.str,
                     "bytes_sha256": hashlib.sha256(array.tobytes()).hexdigest()})


@contextmanager
def shared(directory):
    """Keep baseline/index immutable for the whole wrapper operation."""
    directory = M.safe(directory)
    with (directory / "writer.lock").open("r") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        yield directory


def selection_binding(path, policy, policy_file):
    path = M.safe(path)
    value = M.read_json(path)
    if (value.get("schema") != 1 or value.get("state") != "LOCKED"
            or value.get("policy_id") != policy["policy_id"]
            or value.get("policy_sha256") != M.sha(policy_file)
            or value.get("selected_before_external_effects") is not True
            or value.get("formal_evidence") is not False
            or not value.get("selection_rule")
            or not all(isinstance(value.get(k), str) and HEX64.fullmatch(value[k])
                       for k in ("analysis_sha256", "platform_plan_sha256"))):
        raise ValueError("A matching, pre-external-effect LOCKED COCO selection file is required")
    return {"path": str(path), "sha256": M.sha(path), "policy_id": policy["policy_id"],
            "policy_sha256": M.sha(policy_file), "selection": value}


def validate_native(directory, policy, *, allow_reuse=False):
    """Integrity check only; deliberately computes no retrieval metrics."""
    directory = M.safe(directory)
    panel = M.read_json(directory / "panel.json")
    if M.read_json(directory / "policy.json") != policy:
        raise ValueError("Prepared policy differs")
    meta = M.read_json(directory / "meta.json")
    identity = meta.get("identity", {})
    models = M.frozen_config()["models"]
    required = {"core_embed", "core_reranker"} | ({"cs_clip"} if policy["backbone"] == "cs_clip" else set())
    accepted_states = {"REUSE_PENDING", "REUSE_COMPLETE"} if allow_reuse else {"COMPLETE"}
    marker = meta.get("cross_execution", {})
    if (meta.get("state") not in accepted_states
            or (allow_reuse and (marker.get("mode") != SCHEMA
                                or marker.get("state") != meta["state"].removeprefix("REUSE_")))
            or meta.get("cache_sha256") != M.sha(directory / "cache.npz")
            or identity.get("panel_sha256") != M.sha(directory / "panel.json")
            or identity.get("policy_sha256") != M.sha(directory / "policy.json")
            or identity.get("sources") != M.script_sources()
            or identity.get("source") != M.SOURCE or identity.get("official_CORE_code") != M.CORE_CODE
            or identity.get("model_config_sha256") != M.CONFIG_SHA
            or identity.get("model_revisions") != {name: models[name]["revision"] for name in required}
            or meta.get("cross_admission_dependency") is not False
            or not re.fullmatch(r"[0-9a-f]{40}", str(identity.get("commit", "")))):
        raise ValueError("Native cache COMPLETE/source/model/panel/cache SHA integrity differs")
    if (len(panel["queries"]), len(panel["candidates"])) != PILOT_COUNTS:
        raise ValueError("Only the fixed 128-query/8320-candidate pilot is allowed")
    with np.load(directory / "cache.npz", allow_pickle=False) as arrays:
        vectors = {name: np.array(arrays[name], copy=True) for name in arrays.files}
    return panel, meta, vectors


def score_pairs(vectors):
    pairs, cross = np.asarray(vectors["pairs"]), np.asarray(vectors["cross"])
    if (pairs.dtype != np.dtype("int64") or pairs.ndim != 2 or pairs.shape[1] != 2
            or cross.dtype != np.dtype("float32") or cross.shape != (len(pairs),)
            or not np.isfinite(cross).all() or len(set(map(tuple, pairs.tolist()))) != len(pairs)):
        raise ValueError("Cross pair order/shape/dtype/uniqueness is malformed")
    return pairs.tolist(), cross


def baseline_bundle(directory, index_directory):
    """Read only direct baseline scores/provenance, never method metrics."""
    directory, index_directory = M.safe(directory), M.safe(index_directory)
    policy = M.parse_policy("m1/top200/core")
    panel, meta, vectors = validate_native(directory, policy)
    if (meta.get("cross_execution") is not None or (directory / "reuse-intent.json").exists()
            or (directory / "reuse-receipt.json").exists()):
        raise ValueError("Baseline must be a direct original-runner cache, not a reuse chain")
    if set(vectors) != {"core_embed_image", "core_embed_full", "pairs", "cross"}:
        raise ValueError("Baseline may only contain method-independent CORE vectors and prefix50 scores")
    identity = meta["identity"]
    adapter = identity.get("adapter", {})
    if adapter.get("batch_sizes", {}).get("cross") != 1 or adapter.get("CS_candidate_input") is not None:
        raise ValueError("Direct baseline must use cross batch1 and native CORE joint evidence")
    index_meta = M.read_json(index_directory / "index-meta.json")
    expected = M.index_identity(SimpleNamespace(joint_batch=adapter["batch_sizes"]["joint_image"]), panel,
                               M.frozen_config(), Path(identity["model_revisions"]["core_embed"]),
                               identity["environment"], index_meta["identity"]["processor"], M.script_sources())
    index_vectors, index_receipt = M.read_shared_index(index_directory, expected)
    if identity.get("shared_index") != index_receipt:
        raise ValueError("Baseline was not made from this exact shared index")
    assert_index_vectors(vectors, index_vectors)
    pairs, cross = score_pairs(vectors)
    core, selected = M.admission(panel, vectors, policy)
    if pairs != M.cross_union(core, selected) or len(pairs) != M.BUDGET * len(panel["queries"]):
        raise ValueError("Baseline pairs are not the actual CORE prefix50 in CORE order")
    return SimpleNamespace(path=directory, panel=panel, meta=meta, vectors=vectors, core=core,
                           pairs=pairs, cross=cross, lookup={tuple(p): i for i, p in enumerate(pairs)},
                           index_vectors=index_vectors, index=index_receipt,
                           binding={"path": str(directory), "cache_sha256": M.sha(directory / "cache.npz"),
                                    "meta_sha256": M.sha(directory / "meta.json"),
                                    "panel_sha256": M.sha(directory / "panel.json"),
                                    "policy_sha256": M.sha(directory / "policy.json"),
                                    "pairs_sha256": M.digest(pairs), "cross_array_sha256": array_hash(cross)})


def assert_index_vectors(vectors, index_vectors):
    for key in ("core_embed_image", "core_embed_full"):
        a, b = np.asarray(vectors[key]), np.asarray(index_vectors[key])
        if a.dtype != np.float32 or b.dtype != np.float32 or a.shape != b.shape or not np.array_equal(a, b):
            raise ValueError("Final/baseline candidate and full-query vectors must exactly equal the shared index")


def validate_target_identity(identity, baseline):
    reference = baseline.meta["identity"]
    for key in ("schema", "scope", "source", "official_CORE_code", "sources", "panel_sha256",
                "model_config_sha256", "environment", "shared_index"):
        if identity.get(key) != reference.get(key):
            raise ValueError(f"Reuse target identity differs from baseline: {key}")
    for key in ("core_embed", "core_reranker"):
        if identity.get("model_revisions", {}).get(key) != reference["model_revisions"][key]:
            raise ValueError("Reuse target CORE/reranker model revision differs")
    a, b = deepcopy(identity.get("adapter", {})), deepcopy(reference["adapter"])
    a.pop("CS_candidate_input", None)
    b.pop("CS_candidate_input", None)
    if a != b or a.get("batch_sizes", {}).get("cross") != 1:
        raise ValueError("Reuse target processor/joint/cross batch or prompt adapter differs")


def pair_order_guard(items, baseline):
    if len(items) != len(set(map(tuple, items))) or not set(baseline.lookup) <= set(map(tuple, items)):
        raise ValueError("Target union must contain each baseline pair exactly once")
    if any(not isinstance(qi, int) or not isinstance(ci, int) or not 0 <= qi < len(baseline.core)
           or not 0 <= ci < baseline.core.shape[1] for qi, ci in items):
        raise ValueError("Target pair indices are invalid")
    expected = []
    for qi, row in enumerate(baseline.core):
        chosen = {ci for q, ci in items if q == qi}
        if not M.BUDGET <= len(chosen) <= 2*M.BUDGET or not chosen <= set(row[:200]):
            raise ValueError("Target union must be at most 100 pairs/query inside the original top200")
        expected.extend([[qi, int(ci)] for ci in row if ci in chosen])
    if items != expected:
        raise ValueError("Target union pair order differs from CORE order")


class ReuseCompute:
    """Shape/dtype-preserving lookup after a mandatory fresh, exact batch1 probe."""
    def __init__(self, baseline, items, compute):
        pair_order_guard(items, baseline)
        self.baseline, self.compute = baseline, compute
        self.returned_reused = self.returned_fresh = 0
        common = [pair for pair in items if tuple(pair) in baseline.lookup][:REPEAT_COUNT]
        if len(common) != REPEAT_COUNT:
            raise ValueError("Eight common pairs are required for fresh exact-repeat verification")
        expected = np.asarray([baseline.cross[baseline.lookup[tuple(p)]] for p in common], dtype=np.float32)
        fresh = []
        for pair in common:
            value = np.asarray(compute([pair]))
            if value.shape != (1,) or value.dtype != expected.dtype or not np.isfinite(value).all():
                raise ValueError("Fresh repeat changed batch1 shape/dtype or contains nonfinite scores")
            fresh.append(value[0])
        fresh = np.asarray(fresh, dtype=np.float32)
        if not np.array_equal(fresh, expected) or array_hash(fresh) != array_hash(expected):
            raise ValueError("Fresh eight-pair batch1 repeat differs: no baseline scores may be reused")
        self.repeat = {"pairs": common, "count": REPEAT_COUNT, "batch": 1, "exact_equal": True,
                       "expected_array_sha256": array_hash(expected), "fresh_array_sha256": array_hash(fresh),
                       "not_a_full_rerun": True}

    def __call__(self, rows):
        if len(rows) != 1:
            raise ValueError("Cross reuse is restricted to unchanged batch1 calls")
        key = tuple(rows[0])
        if key in self.baseline.lookup:
            self.returned_reused += 1
            return self.baseline.cross[[self.baseline.lookup[key]]].copy()
        value = np.asarray(self.compute(rows))
        if value.shape != (1,) or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError("Fresh novel-pair score changed shape/dtype or is nonfinite")
        self.returned_fresh += 1
        return value


@contextmanager
def intercept(args, baseline, binding):
    original_helpers, original_json = M.checkpoint_helpers, M.atomic_json
    state = {}
    target = M.safe(args.out)
    def helpers():
        helper = original_helpers()
        def batches(path, items, batch_size, compute, identity, **kwargs):
            if identity.get("stage") != "cross_union":
                return helper.batches(path, items, batch_size, compute, identity, **kwargs)
            if M.safe(path) != target / "checkpoints/cross_union.npy" or batch_size != 1 or state:
                raise ValueError("Unexpected cross stage location, batch, or repeated dispatch")
            validate_target_identity(identity, baseline)
            reused = ReuseCompute(baseline, items, compute)
            state.update(compute=reused, items=deepcopy(items))
            # Isolate resumed cross checkpoints from direct/other-baseline execution.
            return helper.batches(path, items, batch_size, reused, identity | {"cross_reuse": binding}, **kwargs)
        return SimpleNamespace(**(vars(helper) | {"batches": batches}))
    def publish(path, value):
        if M.safe(path) == target / "meta.json" and value.get("state") == "COMPLETE":
            # The unmodified native analyzer must reject this artifact, including
            # after completion; it does not know how to verify wrapper receipts.
            value = dict(value, state="REUSE_PENDING", cross_execution={"mode": SCHEMA, "state": "PENDING",
                                                                       "required_receipt": "reuse-receipt.json"})
        return original_json(path, value)
    M.checkpoint_helpers, M.atomic_json = helpers, publish
    try:
        yield state
    finally:
        M.checkpoint_helpers, M.atomic_json = original_helpers, original_json


def checkpoint_bindings(directory):
    return {name: M.sha(directory / "checkpoints" / name) for name in (
        "provenance.json", "cross-pairs.json", "cross_union.npy", "cross_union.state.json")}


def validate_live_bindings(binding):
    baseline, index, selection = (binding[k] for k in ("baseline", "index", "selection_lock"))
    base_path, index_path = M.safe(baseline["path"]), M.safe(index["path"])
    for field, filename in (("cache_sha256", "cache.npz"), ("meta_sha256", "meta.json"),
                            ("panel_sha256", "panel.json"), ("policy_sha256", "policy.json")):
        if M.sha(base_path / filename) != baseline[field]:
            raise ValueError("Baseline source files changed during reuse")
    if (M.sha(index_path / "index-meta.json") != index["meta_sha256"]
            or M.sha(index_path / "index.npz") != index["index_sha256"]
            or M.sha(selection["path"]) != selection["sha256"]):
        raise ValueError("Index or locked COCO selection changed during reuse")


def audit_final(directory, policy, baseline):
    panel, meta, vectors = validate_native(directory, policy, allow_reuse=True)
    validate_target_identity(meta["identity"], baseline)
    assert_index_vectors(vectors, baseline.index_vectors)
    if panel != baseline.panel:
        raise ValueError("Target panel differs from baseline")
    pairs, cross = score_pairs(vectors)
    core, selected = M.admission(panel, vectors, policy)
    if pairs != M.cross_union(core, selected):
        raise ValueError("Final union differs from the frozen policy's cheap-vector admission")
    pair_order_guard(pairs, baseline)
    reused = [(i, baseline.lookup[tuple(pair)]) for i, pair in enumerate(pairs) if tuple(pair) in baseline.lookup]
    if not np.array_equal(cross[[i for i, _ in reused]], baseline.cross[[j for _, j in reused]]):
        raise ValueError("Final reused scores differ from baseline")
    checkpoint = np.load(directory / "checkpoints/cross_union.npy", allow_pickle=False)
    if checkpoint.dtype != cross.dtype or checkpoint.shape != cross.shape or not np.array_equal(checkpoint, cross):
        raise ValueError("Final cache scores differ from their cross checkpoint")
    if M.read_json(directory / "checkpoints/provenance.json") != meta["identity"]:
        raise ValueError("Final checkpoint provenance differs")
    if M.read_json(directory / "checkpoints/cross-pairs.json") != pairs:
        raise ValueError("Final checkpoint pair order differs")
    return panel, meta, vectors, pairs, len(reused)


def finalize(directory, policy, baseline, binding, state):
    if not state or "compute" not in state:
        raise ValueError("No intercepted cross stage; cannot publish reuse receipt")
    validate_live_bindings(binding)
    panel, meta, vectors, pairs, reused = audit_final(directory, policy, baseline)
    compute = state["compute"]
    native = dict(meta)
    native.pop("cross_execution", None)
    native["state"] = "COMPLETE"
    validate_wrapper_commit(native["identity"]["commit"], binding["wrapper"])
    receipt = {"schema": SCHEMA, "state": "COMPLETE", "binding": binding,
               "wrapper_execution_commit": native["identity"]["commit"],
               "final_cache_sha256": M.sha(directory / "cache.npz"), "native_meta_sha256": M.digest(native),
               "checkpoint_sha256": checkpoint_bindings(directory), "pairs_sha256": M.digest(pairs),
               "counts": {"final_pairs": len(pairs), "reused_final_pairs": reused,
                          "fresh_final_pairs": len(pairs)-reused, "extra_fresh_verification_pairs": REPEAT_COUNT,
                          "current_invocation_returned_reused": compute.returned_reused,
                          "current_invocation_computed_fresh_final": compute.returned_fresh,
                          "resume_note": "Final counts describe the completed array; invocation counters exclude earlier attempts. Each invocation repeats eight fresh pairs."},
               "fresh_repeat": compute.repeat, "scope": M.SCOPE, "formal_evidence": False,
               "cross_execution": "Native runner plus verified baseline lookup; not every final score was freshly computed"}
    M.atomic_json(directory / "reuse-receipt.json", receipt)
    meta["state"] = "REUSE_COMPLETE"
    meta["cross_execution"] = {"mode": SCHEMA, "state": "COMPLETE", "required_receipt": "reuse-receipt.json",
                               "receipt_sha256": M.sha(directory / "reuse-receipt.json")}
    M.atomic_json(directory / "meta.json", meta)
    return receipt


def validate_receipt(directory, policy, baseline, binding):
    validate_live_bindings(binding)
    panel, meta, vectors, pairs, reused = audit_final(directory, policy, baseline)
    marker = meta.get("cross_execution", {})
    receipt = M.read_json(directory / "reuse-receipt.json")
    native = dict(meta)
    native.pop("cross_execution", None)
    native["state"] = "COMPLETE"
    if (marker != {"mode": SCHEMA, "state": "COMPLETE", "required_receipt": "reuse-receipt.json",
                   "receipt_sha256": M.sha(directory / "reuse-receipt.json")}
            or receipt.get("schema") != SCHEMA or receipt.get("state") != "COMPLETE"
            or receipt.get("binding") != binding or receipt.get("formal_evidence") is not False
            or receipt.get("final_cache_sha256") != M.sha(directory / "cache.npz")
            or receipt.get("native_meta_sha256") != M.digest(native)
            or receipt.get("checkpoint_sha256") != checkpoint_bindings(directory)
            or receipt.get("pairs_sha256") != M.digest(pairs)):
        raise ValueError("Wrapper receipt/cache/baseline/index/source/selection integrity differs")
    if receipt.get("wrapper_execution_commit") != meta["identity"]["commit"]:
        raise ValueError("Wrapper execution commit differs from the final cache provenance")
    validate_wrapper_commit(receipt["wrapper_execution_commit"], binding["wrapper"])
    counts = receipt.get("counts", {})
    if any(counts.get(key) != value for key, value in {
        "final_pairs": len(pairs), "reused_final_pairs": reused, "fresh_final_pairs": len(pairs)-reused,
        "extra_fresh_verification_pairs": REPEAT_COUNT}.items()):
        raise ValueError("Reuse receipt pair counts differ")
    common = [p for p in pairs if tuple(p) in baseline.lookup][:REPEAT_COUNT]
    expected = np.asarray([baseline.cross[baseline.lookup[tuple(p)]] for p in common], dtype=np.float32)
    repeat = receipt.get("fresh_repeat", {})
    if (repeat.get("pairs") != common or repeat.get("count") != REPEAT_COUNT or repeat.get("batch") != 1
            or repeat.get("exact_equal") is not True or repeat.get("expected_array_sha256") != array_hash(expected)
            or repeat.get("fresh_array_sha256") != array_hash(expected)):
        raise ValueError("Receipt does not attest the fixed eight-pair exact fresh repeat")
    return panel, meta, vectors, receipt


def binding_for(baseline, selection, source):
    # Commit is recorded but additional unrelated commits do not invalidate bytes.
    return {"schema": SCHEMA, "wrapper": {k: source[k] for k in ("path", "sha256")},
            "baseline": baseline.binding, "index": baseline.index, "selection_lock": selection}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "images", "out", "baseline", "index", "selection-lock"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--depth", type=int, choices=(100, 200))
    parser.add_argument("--lcse-calibration", type=Path)
    parser.add_argument("--joint-batch", type=int, default=1)
    parser.add_argument("--cross-batch", type=int, default=1, choices=(1,))
    parser.add_argument("--stage", required=True, choices=("prepare", "cache", "analyze"))
    args = parser.parse_args()
    for name in ("data", "images", "out", "baseline", "index", "selection_lock"):
        setattr(args, name, M.safe(getattr(args, name)))
    if not 1 <= args.joint_batch <= 8:
        parser.error("joint-batch must remain within the frozen range 1..8")
    paths = [args.out, args.baseline, args.index, args.data, args.images]
    if any(args.out == p or args.out in p.parents or p in args.out.parents for p in paths[1:]):
        parser.error("Target run must be separate from baseline/index/source directories")
    calibration = M.read_json(args.lcse_calibration) if args.lcse_calibration else None
    policy = M.parse_policy(args.policy, args.depth, calibration)
    if args.lcse_calibration:
        policy["calibration_sha256"] = M.sha(args.lcse_calibration)
    with M.writer(args.out):
        if args.stage == "prepare":
            M.prepare(args, policy)
            selection_binding(args.selection_lock, policy, args.out / "policy.json")
            return
        source = wrapper_source()
        with shared(args.baseline), shared(args.index):
            baseline = baseline_bundle(args.baseline, args.index)
            selection = selection_binding(args.selection_lock, policy, args.out / "policy.json")
            binding = binding_for(baseline, selection, source)
            if args.stage == "cache":
                panel = M.prepare(args, policy)
                if M.sha(args.out / "panel.json") != baseline.binding["panel_sha256"]:
                    raise ValueError("Prepared target panel must match the baseline bytes exactly")
                if M.environment() != baseline.meta["identity"]["environment"]:
                    raise ValueError("Actual process environment differs from baseline")
                M.fixed_json(args.out / "reuse-intent.json", binding)
                with intercept(args, baseline, binding) as state:
                    M.run_gpu(args, panel, policy)
                receipt = finalize(args.out, policy, baseline, binding, state)
                validate_receipt(args.out, policy, baseline, binding)
                print(M.canonical({"stage": "cache_complete", "research_metrics": False,
                                   "counts": receipt["counts"], "wrapper_commit": source["commit"]}).decode(), flush=True)
                return
            if M.read_json(args.out / "reuse-intent.json") != binding:
                raise ValueError("Reuse intent differs from analysis context")
            panel, meta, vectors, receipt = validate_receipt(args.out, policy, baseline, binding)
            report = M.analyze_arrays(panel, vectors, policy, vectors["pairs"].tolist(), vectors["cross"])
            report.update(provenance=meta, cross_reuse=receipt, formal_evidence=False)
            M.atomic_json(args.out / "analysis.json", report)
            print(M.canonical({"stage": "analyzed", "report": str(args.out / "analysis.json"),
                               "scope": M.SCOPE, "formal_evidence": False}).decode(), flush=True)


if __name__ == "__main__":
    main()
