"""[codex] Explicit, hash-bound local platform studies; never a replay waiver.

Only a committed study plan and a complete producer consistency receipt may
authorize platform-relative exploration. Historical replay remains MISMATCH.
This module does not inspect sealed data or choose policies or thresholds.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import subprocess


MODE = "platform-relative-exploration"
METRICS = ("map_pp", "oracle_map_pp", "recall_pp", "queries_without_hit")
ARTIFACTS = ("meta.json", "panel.json", "vectors.npz", "cross.npz")
CHECKS = ("complete_depth200", "vector_shapes_finite_unit", "core_order_recomputed",
          "top50_source_identity", "baseline_recomputed", "fixed_sample_repeat")
SOURCE_FILES = tuple(f"{model}_{part}.npy" for model in ("core_embed", "cs_clip")
                     for part in ("image", "full", "condition")) + ("cross_top50.npy",)
OPERATIONS = ("D1", "D2", "M1", "M2", "D3", "blind-export", "cost-benchmark", "MCMR-development-pilot")
EXPECTED_ENVIRONMENT = {
    "torch": "2.14.0+cu126", "torchvision": "0.29.0+cu126", "transformers": "5.20.2",
    "numpy": "2.3.5", "Pillow": "11.3.0", "open_clip_torch": "2.32.0", "cuda": "12.6",
    "cudnn": 91002, "device": "NVIDIA GeForce RTX 3090", "capability": [8, 6],
    "cpu_threads": {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}}
EXPECTED = {
    "schema": "retrieval-platform-plan-v1", "scope": "exposed-development-only", "mode": MODE,
    "formal_evidence": False,
    "source": "2daedf522c6d64b860277657d3067b61da3a0f2b",
    "freeze": "569b6c053f0f06006b7eab01ab64b815cd136702",
    "algorithm_commit": "4c5c20188e3db6f9aeef5e1f9ef77c4dd7f9eb52",
    "source_cache_commit": "4c5c20188e3db6f9aeef5e1f9ef77c4dd7f9eb52",
    "panel_sha256": "8f43eedbcef31fe5057c26705e97bf9f213f162612d9a15b53da11bb9bacf78d",
    "counts": {"images": 8191, "queries": 136, "original_groups": 118}, "depth": 200, "top_k": 50,
    "batch_size": {"embed_image": 8, "embed_text": 32, "cross_pair": 4},
    "model_config_sha256": "862591c60c5609f16727d902993df0cba9cab96ded72611d69d263a02fe4ad1d",
    "model_revisions": {
        "core_embed": "cc9870ea92d64ebd5f8f9fc368d1120453564158",
        "core_reranker": "64e1a36cd6ecace459a4c7cd8f7d0ef5d6fee30b",
        "cs_clip": "afd6e90f39fe49137fe15d6a462d994eae4811f5"},
    "environment": EXPECTED_ENVIRONMENT,
    "published_acceptance": {"map_pp": 54.53, "oracle_map_pp": 74.02, "recall_pp": 73.77,
                             "queries_without_hit": 31, "tolerance_pp": .10}}
_TOKEN = object()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def safe(path):
    path = Path(path).expanduser()
    if path.is_symlink():
        raise ValueError("Platform study inputs must be ordinary files/directories")
    path = path.resolve()
    if any("retrieval-independent-panel" in part for part in path.parts):
        raise ValueError("Sealed-panel paths are forbidden in platform studies")
    return path


def _json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON keys in platform study provenance")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError(f"Nonfinite JSON value in platform study provenance: {value}")
    value = json.loads(safe(path).read_text(), object_pairs_hook=pairs, parse_constant=invalid)
    if not isinstance(value, dict):
        raise ValueError("Platform study provenance must be a JSON object")
    return value


def _hex(value, width):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{" + str(width) + r"}", value) is not None


def _committed_plan(plan_path, commit):
    if not _hex(commit, 40):
        raise ValueError("Platform receipt requires a full producer commit")
    try:
        root = Path(subprocess.run(["git", "-C", str(plan_path.parent), "rev-parse", "--show-toplevel"],
            check=True, capture_output=True, text=True).stdout.strip()).resolve()
        relative = plan_path.relative_to(root).as_posix()
        stored = subprocess.run(["git", "-C", str(root), "show", f"{commit}:{relative}"],
            check=True, capture_output=True).stdout
    except (subprocess.CalledProcessError, ValueError) as error:
        raise ValueError("Platform plan is not present in the producer commit") from error
    if stored != plan_path.read_bytes():
        raise ValueError("Platform plan differs from its producer commit")
    return {"repository_path": relative, "producer_commit": commit}


class _Context(dict):
    """JSON-serializable context issued only after artifact validation."""
    def __init__(self, value, *, token, meta):
        if token is not _TOKEN:
            raise ValueError("Platform contexts must be issued by validate_context")
        super().__init__(value)
        self._binding = hashlib.sha256(canonical(value)).hexdigest()
        self._meta_binding = hashlib.sha256(canonical(meta)).hexdigest()


def _issued(context, meta):
    if (not isinstance(context, _Context)
            or context._binding != hashlib.sha256(canonical(context)).hexdigest()
            or context._meta_binding != hashlib.sha256(canonical(meta)).hexdigest()):
        raise ValueError("Platform context is unvalidated, modified, or bound to a different cache meta")


def _baseline(value):
    if not isinstance(value, dict) or set(value) != set(METRICS):
        raise ValueError("Platform baseline must contain the four exact replay metrics")
    if (any(isinstance(value[key], bool) or not isinstance(value[key], (float, int))
            or not math.isfinite(value[key]) for key in METRICS)
            or not isinstance(value["queries_without_hit"], int)
            or value["queries_without_hit"] < 0):
        raise ValueError("Platform baseline metrics must be finite and no-hit must be an integer")
    return dict(value)


def require_baseline(actual, meta, context):
    """Bind analysis' freshly recomputed baseline to the immutable receipt."""
    _issued(context, meta)
    actual = _baseline(actual)
    if actual != context["baseline"]:
        raise ValueError("Recomputed local baseline differs from platform consistency receipt")
    return actual


def runtime_environment():
    """GPU producers/cost runners call this; CPU consumers never initialize CUDA."""
    from importlib.metadata import version
    import os
    import torch
    environment = {name: version(name) for name in
                   ("torch", "torchvision", "transformers", "numpy", "Pillow", "open_clip_torch")}
    environment.update(cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                       device=torch.cuda.get_device_name(0), capability=list(torch.cuda.get_device_capability(0)),
                       cpu_threads={name: os.environ.get(name) for name in
                                    ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})
    return environment


def _validate_plan(plan):
    for key, value in EXPECTED.items():
        # Canonical JSON also distinguishes false from 0 and integer from float.
        if key not in plan or canonical(plan[key]) != canonical(value):
            raise ValueError(f"Unsupported fixed platform study plan: {key}")
    operations = plan.get("allowed_operations")
    if (not isinstance(operations, list) or len(operations) != len(OPERATIONS)
            or set(operations) != set(OPERATIONS)):
        raise ValueError("Platform plan has unsupported allowed operations")
    if plan.get("owner") != "[codex]" or plan.get("historical_baseline_was_read") is not True:
        raise ValueError("Platform plan must disclose its owner and prior baseline exposure")


def _supplements(paths, meta, environment):
    artifacts, parents = [], {}
    required = {"lcse_vectors.npz", "d3_embeddings.npz", "lcse_calibration.json"}
    for raw in paths:
        path = safe(raw)
        if path.name not in required or not path.is_file():
            raise ValueError("Unsupported supplementary platform artifact")
        parent = path.parent
        if parent not in parents:
            supplement_meta, provenance = _json(parent / "meta.json"), _json(parent / "provenance.json")
            outputs = supplement_meta.get("outputs_sha256")
            if (supplement_meta.get("status") != "complete" or not isinstance(outputs, dict)
                    or not required <= outputs.keys()):
                raise ValueError("Supplement needs a complete parent metadata/output SHA receipt")
            for name, digest in outputs.items():
                if (not isinstance(name, str) or Path(name).name != name or not _hex(digest, 64)
                        or file_sha(safe(parent / name)) != digest):
                    raise ValueError("Supplement parent output SHA differs")
            if any(provenance.get(key) != environment[key] for key in ("torch", "cuda", "device")):
                raise ValueError("Supplement runtime differs from the fixed platform environment")
            for key in ("source", "freeze"):
                if provenance.get(key) != meta["bundle"][key]:
                    raise ValueError("Supplement frozen source differs from platform cache")
            if not _hex(provenance.get("commit"), 40) or provenance.get("config") != meta.get("config"):
                raise ValueError("Supplement requires committed, same-configuration provenance")
            if (not _hex(provenance.get("calibration_design_sha256"), 64)
                    or not _hex(provenance.get("d3_texts_sha256"), 64)
                    or any(supplement_meta.get(key) != value for key, value in provenance.items())):
                raise ValueError("Supplement parent metadata and provenance differ")
            parents[parent] = {"parent_meta_sha256": file_sha(parent / "meta.json"),
                               "parent_provenance_sha256": file_sha(parent / "provenance.json"),
                               "producer_commit": provenance["commit"],
                               "runtime": {key: provenance[key] for key in ("torch", "cuda", "device")}}
        artifacts.append({"name": path.name, "sha256": file_sha(path), **parents[parent]})
    return artifacts


def validate_context(cache_dir, plan_path, receipt_path, *, supplementary_paths=()):
    """Validate the one predeclared platform study without changing replay state.

    The committed producer owns numerical/shape/order checks. Consumers verify
    their complete evidence receipt, all final byte hashes, the original failure
    and platform identities. Analysis additionally calls require_baseline.
    """
    cache, plan_path, receipt_path = safe(cache_dir), safe(plan_path), safe(receipt_path)
    plan, receipt = _json(plan_path), _json(receipt_path)
    _validate_plan(plan)
    if (receipt.get("schema") != "retrieval-platform-receipt-v1"
            or receipt.get("plan_sha256") != file_sha(plan_path)
            or receipt.get("formal_evidence") is not False
            or receipt.get("platform_consistency") != "VALID_LOCAL"
            or receipt.get("environment") != plan["environment"]):
        raise ValueError("Invalid platform consistency receipt schema/plan/scope/environment")
    committed = _committed_plan(plan_path, receipt.get("producer_commit"))
    if committed["repository_path"] != "codex/retrieval3090/platform_plan.json":
        raise ValueError("Platform plan must be the fixed repository study plan")
    checks = receipt.get("checks")
    if not isinstance(checks, dict) or set(checks) != set(CHECKS) or any(checks[key] is not True for key in CHECKS):
        raise ValueError("Platform receipt lacks complete producer invariant/repeat checks")
    outputs = receipt.get("outputs_sha256")
    if not isinstance(outputs, dict) or set(outputs) != set(ARTIFACTS):
        raise ValueError("Platform receipt must bind exactly the four complete cache artifacts")
    for name in ARTIFACTS:
        if not _hex(outputs[name], 64) or file_sha(safe(cache / name)) != outputs[name]:
            raise ValueError(f"Platform cache artifact SHA differs: {name}")
    meta, panel = _json(cache / "meta.json"), _json(cache / "panel.json")
    published = receipt.get("published_reproduction")
    if (not isinstance(published, dict) or set(published) != {"state", "metrics"}
            or published["state"] != "MISMATCH" or meta.get("replay", {}).get("state") != "MISMATCH"):
        raise ValueError("Platform studies require the preserved published MISMATCH; PASS is not a platform receipt")
    baseline = _baseline(published["metrics"])
    replay_core = meta["replay"].get("core", {})
    if any(replay_core.get(key) != baseline[key] for key in METRICS):
        raise ValueError("Platform receipt baseline differs from unchanged cache replay")
    published_expected = plan["published_acceptance"]
    if all(abs(baseline[key] - published_expected[key]) <= (0 if key == "queries_without_hit" else
            published_expected["tolerance_pp"]) for key in METRICS):
        raise ValueError("Platform MISMATCH cannot disguise an actually passing historical replay")
    bundle = meta.get("bundle", {})
    if (any(bundle.get(key) != plan[key] for key in ("source", "freeze"))
            or bundle.get("commit") != receipt["producer_commit"] or bundle.get("dirty") != "0"
            or meta.get("panel_sha256") != plan["panel_sha256"]
            or meta.get("revisions") != plan["model_revisions"]
            or meta.get("model_config_sha256") != plan["model_config_sha256"]
            or meta.get("depth") != 200 or meta.get("top_k") != 50
            or meta.get("config", {}).get("batch_size") != plan["batch_size"]
            or meta.get("execution_environment") != plan["environment"]):
        raise ValueError("Platform cache source/model/config/environment identity differs")
    if (len(panel.get("ids", [])) != plan["counts"]["images"]
            or len(set(panel["ids"])) != len(panel["ids"])
            or len(panel.get("queries", [])) != plan["counts"]["queries"]
            or len({query.get("group") for query in panel["queries"]}) != plan["counts"]["original_groups"]):
        raise ValueError("Platform cache exposed panel counts differ")
    source_digest = receipt.get("source_checkpoint_provenance_sha256")
    sources = receipt.get("source_checkpoint_files")
    if (not _hex(source_digest, 64) or meta.get("source_checkpoint_provenance_sha256") != source_digest
            or not isinstance(sources, dict) or set(sources) != set(SOURCE_FILES)
            or any(not isinstance(row, dict) or set(row) != {"sha256", "state_sha256"}
                   or any(not _hex(row[key], 64) for key in row) for row in sources.values())):
        raise ValueError("Platform receipt lacks complete source checkpoint identities")
    supplementary = _supplements(supplementary_paths, meta, plan["environment"])
    context = {"mode": MODE, "formal_evidence": False, "published_reproduction": published,
               "platform_consistency": "VALID_LOCAL", "baseline": baseline,
               "plan_sha256": receipt["plan_sha256"], "receipt_sha256": file_sha(receipt_path),
               "environment": plan["environment"], "cache_artifacts_sha256": outputs,
               "producer_commit": receipt["producer_commit"], "source_checkpoint_provenance_sha256": source_digest,
               "supplementary_artifacts": supplementary}
    return _Context(context, token=_TOKEN, meta=meta)
