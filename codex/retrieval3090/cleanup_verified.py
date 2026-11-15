"""[codex] Remove only verified duplicate platform/MCMR checkpoint arrays.

Run this file directly in the NumPy environment, after a completed round.
The default is a dry run; --apply is required for the three fixed allowlisted
paths. Original failed COCO caches, models, inputs and state files are retained.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from time import time_ns
from zoneinfo import ZoneInfo

import numpy as np

from platform_context import validate_context


REPO = Path(__file__).resolve().parents[2]
RUNS_ROOT = REPO / "dl/retrieval3090/runs"
BLOCK_BYTES = 16 * 1024 * 1024


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def fingerprint(value):
    # Exact cache_v2 serialization, deliberately different from canonical().
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _stamp(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns,
            info.st_uid, info.st_nlink, info.st_mode)


def owned(path, *, directory=False):
    """Check every component below the fixed runs root without resolving links."""
    path, root = Path(path).expanduser().absolute(), RUNS_ROOT.absolute()
    if (".." in path.parts or not path.is_relative_to(root)
            or any("retrieval-independent-panel" in part for part in path.parts)):
        raise ValueError("Cleanup is limited to the fixed local runs root")
    if root.resolve() != root:
        raise ValueError("The runs root must not traverse symlinks")
    current = root
    for position, part in enumerate((None, *path.relative_to(root).parts)):
        if part is not None:
            current /= part
        info = current.lstat()
        is_last = current == path
        if stat.S_ISLNK(info.st_mode) or info.st_uid != os.getuid():
            raise ValueError("Cleanup inputs must be owned and must not traverse symlinks")
        if not is_last or directory:
            if not stat.S_ISDIR(info.st_mode):
                raise ValueError("Cleanup ancestor is not a directory")
        elif not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError("Cleanup inputs must be ordinary single-link files")
    return path, info


def _record(path):
    path, info = owned(path)
    return {"path": str(path), "sha256": sha(path), "bytes": info.st_size, "_stamp": _stamp(info)}


def _read_json(path):
    record = _record(path)
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate keys in cleanup provenance")
            result[key] = value
        return result
    def nonfinite(value):
        raise ValueError(f"Nonfinite JSON in cleanup provenance: {value}")
    value = json.loads(Path(record["path"]).read_text(), object_pairs_hook=pairs, parse_constant=nonfinite)
    if not isinstance(value, dict):
        raise ValueError("Cleanup provenance must be a JSON object")
    return value, record


@contextmanager
def _writer_lock(directory):
    directory, _ = owned(directory, directory=True)
    if directory == RUNS_ROOT.absolute():
        raise ValueError("Cleanup requires an individual run directory")
    lock_path, before = owned(directory / "writer.lock")
    descriptor = os.open(lock_path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        if _stamp(os.fstat(stream.fileno())) != _stamp(before):
            raise ValueError("Writer lock changed while opening it")
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield directory


def _checkpoint(path, expected, expected_fingerprint):
    row = _record(path)
    state_path = Path(path).with_suffix(".state.json")
    state, state_record = _read_json(state_path)
    value = np.load(path, mmap_mode="r", allow_pickle=False)
    if (value.shape != expected.shape or value.dtype != expected.dtype or value.dtype.kind != "f"
            or state.get("completed") != value.shape[0]
            or state.get("array_shape") != list(value.shape) or state.get("dtype") != value.dtype.str
            or state.get("sha256") != row["sha256"] or state.get("fingerprint") != expected_fingerprint):
        raise ValueError("Checkpoint shape/dtype/completion/SHA/input identity differs")
    rows_per_block = max(1, BLOCK_BYTES // max(1, value.dtype.itemsize * int(np.prod(value.shape[1:]))))
    for start in range(0, len(value), rows_per_block):
        actual, final = value[start:start + rows_per_block], expected[start:start + rows_per_block]
        if not np.isfinite(actual).all() or not np.isfinite(final).all() or not np.array_equal(actual, final):
            raise ValueError("Checkpoint is not an exact finite duplicate of its final array")
    row.update(shape=list(value.shape), dtype=value.dtype.str, state_sha256=state_record["sha256"],
               input_fingerprint=expected_fingerprint, removed=False)
    return row, state_record


def _recheck(records):
    for row in records:
        path, info = owned(row["path"])
        if _stamp(info) != row["_stamp"] or sha(path) != row["sha256"]:
            raise ValueError("A verified cleanup input changed; nothing further may be deleted")


def _public(row):
    return {key: value for key, value in row.items() if not key.startswith("_")}


def _finish(directory, mode, rows, retained, *, apply, evidence):
    # Validate EVERY planned file and all retained evidence before the first unlink.
    _recheck([*retained, *rows])
    receipt = {"schema": "verified-duplicate-cleanup-v1", "actor": "[codex]", "mode": mode,
               "created_at_taipei": datetime.now(ZoneInfo("Asia/Taipei")).isoformat(),
               "status": "validated" if apply else "dry-run", "apply": bool(apply),
               "run": str(directory), "evidence": evidence,
               "verified_duplicate_bytes": sum(row["bytes"] for row in rows), "removed_bytes": 0,
               "files": [_public(row) for row in rows], "retained": [_public(row) for row in retained]}
    receipt_path = directory / f"cleanup-verified-{time_ns()}-{'apply' if apply else 'dry-run'}.json"
    descriptor = os.open(receipt_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        def save():
            stream.seek(0)
            stream.write(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")
            stream.truncate()
            stream.flush()
            os.fsync(stream.fileno())
        save()
        if apply:
            try:
                _recheck([*retained, *rows])
                for index, row in enumerate(rows):
                    _recheck([row])
                    Path(row["path"]).unlink()
                    receipt["files"][index]["removed"] = True
                    receipt["removed_bytes"] += row["bytes"]
                    save()
                receipt["status"] = "applied"
            except BaseException:
                receipt["status"] = "interrupted"
                save()
                raise
            save()
    return receipt_path, receipt


def cleanup_platform(cache, plan_path, receipt_path, *, apply=False):
    """Only the continuation's cross_rank51to200.npy is eligible."""
    with _writer_lock(cache) as cache:
        context = validate_context(cache, plan_path, receipt_path)
        if (context.get("mode") != "platform-relative-exploration" or context.get("formal_evidence") is not False
                or context.get("published_reproduction", {}).get("state") != "MISMATCH"
                or context.get("platform_consistency") != "VALID_LOCAL"):
            raise ValueError("Cleanup requires a complete validated local platform cache")
        retained = [_record(cache / name) for name in ("meta.json", "panel.json", "vectors.npz", "cross.npz")]
        if {Path(row["path"]).name: row["sha256"] for row in retained} != context["cache_artifacts_sha256"]:
            raise ValueError("Platform artifacts changed after context validation")
        # The explicit input receipt must live in this run, never the failed source cache.
        supplied, _ = owned(receipt_path)
        if supplied.parent != cache or supplied.name != "platform-receipt.json":
            raise ValueError("Use this platform run's own consistency receipt")
        receipt, receipt_record = _read_json(supplied)
        provenance, provenance_record = _read_json(cache / "platform-provenance.json")
        panel, _ = _read_json(cache / "panel.json")
        if (receipt_record["sha256"] != context["receipt_sha256"]
                or provenance.get("schema") != "retrieval-platform-continuation-v1"
                or provenance.get("commit") != context["producer_commit"]
                or provenance.get("plan_sha256") != context["plan_sha256"]
                or provenance.get("environment") != context["environment"]
                or provenance.get("source_checkpoint_provenance_sha256") != context["source_checkpoint_provenance_sha256"]
                or provenance.get("source_checkpoint_files") != receipt.get("source_checkpoint_files")):
            raise ValueError("Platform continuation provenance differs from its consistency receipt")
        path = cache / "checkpoints/cross_rank51to200.npy"
        with np.load(cache / "cross.npz", allow_pickle=False) as final:
            core, cross = final["core_order"], final["cross"]
            query_count, pool_count = len(panel["queries"]), len(panel["ids"])
            if (set(final.files) != {"core_order", "cross"} or cross.shape != (query_count, 200)
                    or core.shape != (query_count, pool_count) or not np.issubdtype(core.dtype, np.integer)
                    or not np.all(np.sort(core, axis=1) == np.arange(pool_count))):
                raise ValueError("Final platform cross/index shapes or candidate order are incomplete")
            keys = [panel["ids"][int(column)] for column in core[:, 50:200].flat]
            queries = [query["text"] for query in panel["queries"] for _ in range(150)]
            expected = fingerprint({"identity": provenance | {"stage": "rank51to200", "queries": queries, "images": keys},
                                    "items": query_count * 150, "batch_size": 4})
            row, state = _checkpoint(path, cross[:, 50:200].reshape(-1), expected)
        row["final_array"] = "cross.npz:cross[:,50:200].reshape(-1)"
        return _finish(cache, "platform-cache", [row], [*retained, receipt_record, provenance_record, state],
                       apply=apply, evidence={"plan_sha256": context["plan_sha256"],
                                              "platform_receipt_sha256": context["receipt_sha256"],
                                              "source_cache": "retained without inspection or deletion"})


def cleanup_index(directory, *, apply=False):
    """Only the two policy-independent CORE shared-index checkpoints qualify."""
    with _writer_lock(directory) as directory:
        meta, meta_record = _read_json(directory / "index-meta.json")
        panel, panel_record = _read_json(directory / "panel.json")
        provenance, provenance_record = _read_json(directory / "index-checkpoints/provenance.json")
        final_record = _record(directory / "index.npz")
        identity = meta.get("identity", {})
        if ((directory / "policy.json").exists() or meta.get("state") != "COMPLETE"
                or identity.get("schema") != "mcmr-core-shared-index-v1" or identity != provenance
                or meta.get("index_sha256") != final_record["sha256"]
                or not re.fullmatch(r"[0-9a-f]{40}", str(meta.get("commit", "")))
                or identity.get("panel_sha256") != digest(panel)
                or identity.get("candidate_inputs_sha256") != digest(panel["candidates"])
                or identity.get("query_inputs_sha256") != digest([query["text"] for query in panel["queries"]])):
            raise ValueError("Shared index is incomplete, policy-dependent or has changed provenance/SHA")
        counts = {"core_embed_image": len(panel["candidates"]), "core_embed_full": len(panel["queries"])}
        if (not all(counts.values()) or identity.get("candidates") != counts["core_embed_image"]
                or identity.get("queries") != counts["core_embed_full"] or identity.get("embedding_dim") != 2048):
            raise ValueError("Shared index candidate/query/dimension identity differs")
        retained, rows = [meta_record, panel_record, provenance_record, final_record], []
        with np.load(directory / "index.npz", allow_pickle=False) as final:
            if set(final.files) != set(counts):
                raise ValueError("Shared index may contain only the two independent CORE vector arrays")
            for stage, key, items, batch in (
                    ("core_joint", "core_embed_image", list(range(counts["core_embed_image"])), identity["batch_sizes"]["joint_image"]),
                    ("core_full", "core_embed_full", [query["text"] for query in panel["queries"]], identity["batch_sizes"]["text"])):
                value = final[key]
                if (value.shape != (counts[key], 2048) or value.dtype.kind != "f" or not np.isfinite(value).all()
                        or not np.allclose(np.linalg.norm(value, axis=1), 1, atol=2e-4)):
                    raise ValueError("Final shared-index embeddings are not finite unit vectors with the expected shape")
                expected = fingerprint({"identity": identity | {"stage": stage, "items_sha256": digest(items)},
                                        "items": len(items), "batch_size": batch})
                row, state = _checkpoint(directory / "index-checkpoints" / f"{stage}.npy", value, expected)
                row["final_array"] = f"index.npz:{key}"
                rows.append(row)
                retained.append(state)
        return _finish(directory, "mcmr-shared-index", rows, retained, apply=apply,
                       evidence={"index_sha256": final_record["sha256"], "index_meta_sha256": meta_record["sha256"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    platform = sub.add_parser("platform-cache")
    platform.add_argument("--cache", type=Path, required=True)
    platform.add_argument("--platform-plan", type=Path, required=True)
    platform.add_argument("--platform-receipt", type=Path, required=True)
    platform.add_argument("--apply", action="store_true")
    index = sub.add_parser("mcmr-index")
    index.add_argument("--index", type=Path, required=True)
    index.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.command == "platform-cache":
        result, _ = cleanup_platform(args.cache, args.platform_plan, args.platform_receipt, apply=args.apply)
    else:
        result, _ = cleanup_index(args.index, apply=args.apply)
    print(result)


if __name__ == "__main__":
    main()
