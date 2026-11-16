"""[codex] Read-only identity checks for a public development benchmark registry.

Only explicitly listed artifacts are opened. A PASS verifies the registry's
identities and comparison references, never a model, score, or research claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath


ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN = ("retrieval-independent-panel", "private", "sealed")
HEX_SHA = re.compile(r"[0-9a-fA-F]{64}\Z")
LIST_INDEX = re.compile(r"(?:0|[1-9][0-9]*)\Z")


class VerificationError(ValueError):
    def __init__(self, message, *, suite_sha256=None, artifact_count=None, cell_count=None):
        super().__init__(message)
        self.suite_sha256 = suite_sha256
        self.artifact_count = artifact_count
        self.cell_count = cell_count


def _nonempty(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _forbidden(parts):
    return any(word in part.casefold() for part in parts for word in FORBIDDEN)


def allowed_path(root, value, *, relative=True):
    """Resolve one allowlisted path; reject escapes before opening any file."""
    value = _nonempty(value, "path")
    posix, windows = PurePosixPath(value), PureWindowsPath(value)
    if "\\" in value or windows.drive:
        raise ValueError("Paths must use repository-relative POSIX components")
    if ".." in posix.parts or _forbidden(posix.parts):
        raise ValueError("Path has a forbidden or parent component")
    if relative and posix.is_absolute():
        raise ValueError("Artifact paths must be repository-relative")
    candidate = Path(value) if posix.is_absolute() else root / value
    resolved = candidate.resolve(strict=True)
    try:
        inside = resolved.relative_to(root)
    except ValueError as error:
        raise ValueError("Path or symlink escapes the repository root") from error
    if _forbidden(inside.parts):
        raise ValueError("Resolved path has a forbidden component")
    if not stat.S_ISREG(resolved.stat().st_mode):
        raise ValueError("Allowlisted paths must identify regular files")
    return resolved


def _read_and_hash(path, *, keep_bytes=False):
    digest, chunks = hashlib.sha256(), []
    with path.open("rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("Allowlisted paths must identify regular files")
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            if keep_bytes:
                chunks.append(block)
    return digest.hexdigest(), b"".join(chunks) if keep_bytes else None


def _json(payload):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError(f"Non-JSON numeric constant: {value}")

    return json.loads(payload.decode("utf-8"), object_pairs_hook=unique,
                      parse_constant=invalid_constant)


def json_pointer(document, pointer):
    """RFC 6901 pointer, with strict escapes and array-index grammar."""
    if not isinstance(pointer, str):
        raise ValueError("JSON pointer must be a string")
    if pointer == "":
        return document
    if not pointer.startswith("/"):
        raise ValueError("JSON pointer must be empty or start with /")
    value = document
    for encoded in pointer[1:].split("/"):
        if re.search(r"~(?![01])", encoded):
            raise ValueError(f"Invalid JSON pointer escape: {pointer}")
        token = encoded.replace("~1", "/").replace("~0", "~")
        if isinstance(value, dict):
            if token not in value:
                raise ValueError(f"Missing JSON pointer member: {pointer}")
            value = value[token]
        elif isinstance(value, list):
            if not LIST_INDEX.fullmatch(token):
                raise ValueError(f"Invalid JSON pointer list index: {pointer}")
            index = int(token)
            if index >= len(value):
                raise ValueError(f"JSON pointer list index out of range: {pointer}")
            value = value[index]
        else:
            raise ValueError(f"JSON pointer traverses a scalar: {pointer}")
    return value


def typed_equal(actual, expected):
    """Recursive JSON equality; e.g. true, 1, and 1.0 remain distinct."""
    if type(actual) is not type(expected):
        return False
    if isinstance(actual, dict):
        return actual.keys() == expected.keys() and all(
            typed_equal(actual[key], expected[key]) for key in actual)
    if isinstance(actual, list):
        return len(actual) == len(expected) and all(
            typed_equal(a, b) for a, b in zip(actual, expected))
    return actual == expected


def _validate_registry(suite):
    if not isinstance(suite, dict) or type(suite.get("schema")) is not int or suite["schema"] != 1:
        raise ValueError("Registry schema must be integer 1")
    _nonempty(suite.get("suite_id"), "suite_id")
    if suite.get("formal_evidence") is not False:
        raise ValueError("formal_evidence must be false")
    artifacts = suite.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("artifacts must be a nonempty object")
    for key, spec in artifacts.items():
        _nonempty(key, "artifact key")
        if not isinstance(spec, dict):
            raise ValueError(f"Artifact {key} must be an object")
        _nonempty(spec.get("path"), f"Artifact {key} path")
        if not isinstance(spec.get("sha256"), str) or not HEX_SHA.fullmatch(spec["sha256"]):
            raise ValueError(f"Artifact {key} needs a 64-hex SHA256")
        if spec.get("kind") not in ("json", "file"):
            raise ValueError(f"Artifact {key} kind must be json or file")
        if "expect" in spec and (spec["kind"] != "json" or not isinstance(spec["expect"], dict)):
            raise ValueError(f"Artifact {key} expect requires kind=json and an object")
    for name in ("datasets", "methods"):
        if not isinstance(suite.get(name), dict):
            raise ValueError(f"{name} must be an object")
        for key in suite[name]:
            _nonempty(key, f"{name} key")
    cells = suite.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError("cells must be a nonempty list")
    seen = set()
    for cell in cells:
        if not isinstance(cell, dict):
            raise ValueError("Every cell must be an object")
        for key in ("id", "dataset", "method", "role", "cost_class"):
            _nonempty(cell.get(key), f"Cell {key}")
        if cell["id"] in seen:
            raise ValueError(f"Duplicate cell id: {cell['id']}")
        seen.add(cell["id"])
        if cell["dataset"] not in suite["datasets"] or cell["method"] not in suite["methods"]:
            raise ValueError(f"Unknown dataset or method for cell {cell['id']}")
        refs = cell.get("artifact_refs")
        if not isinstance(refs, list) or any(not isinstance(ref, str) or ref not in artifacts for ref in refs):
            raise ValueError(f"Unknown or malformed artifact_refs for cell {cell['id']}")
        if "baseline_for_comparison" in cell:
            baseline = _nonempty(cell["baseline_for_comparison"], "baseline_for_comparison")
            if baseline not in suite["methods"]:
                raise ValueError(f"Unknown baseline method: {baseline}")
    for cell in cells:
        if "baseline_for_comparison" in cell and not any(
                other["dataset"] == cell["dataset"]
                and other["cost_class"] == cell["cost_class"]
                and other["method"] == cell["baseline_for_comparison"] for other in cells):
            raise ValueError(f"Cell {cell['id']} lacks a baseline on the same dataset and cost_class")


def verify_suite(suite_path, root=ROOT):
    """Return identity-only PASS, or raise VerificationError without writing."""
    suite_sha = artifact_count = cell_count = None
    try:
        root = Path(root).resolve(strict=True)
        if not root.is_dir():
            raise ValueError("root must be a directory")
        path = allowed_path(root, str(suite_path), relative=False)
        suite_sha, payload = _read_and_hash(path, keep_bytes=True)
        suite = _json(payload)
        if isinstance(suite, dict):
            if isinstance(suite.get("artifacts"), dict):
                artifact_count = len(suite["artifacts"])
            if isinstance(suite.get("cells"), list):
                cell_count = len(suite["cells"])
        _validate_registry(suite)
        # Resolve all declared paths before reading artifact payloads. Nothing in
        # datasets, methods, or artifact JSON is followed as another file path.
        paths = {key: allowed_path(root, spec["path"]) for key, spec in suite["artifacts"].items()}
        for key, spec in suite["artifacts"].items():
            actual_sha, body = _read_and_hash(paths[key], keep_bytes=spec["kind"] == "json")
            if actual_sha != spec["sha256"].lower():
                raise ValueError(f"Artifact {key} SHA256 mismatch")
            if spec["kind"] == "json":
                document = _json(body)
                for pointer, expected in spec.get("expect", {}).items():
                    if not typed_equal(json_pointer(document, pointer), expected):
                        raise ValueError(f"Artifact {key} expectation mismatch at {pointer!r}")
        return {"status": "PASS", "scope": "registry-identity-only", "formal_evidence": False,
                "suite_id": suite["suite_id"], "suite_sha256": suite_sha,
                "artifact_count": len(suite["artifacts"]), "cell_count": len(suite["cells"])}
    except (OSError, ValueError, TypeError, OverflowError, RecursionError) as error:
        raise VerificationError(str(error), suite_sha256=suite_sha,
                                artifact_count=artifact_count, cell_count=cell_count) from error


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    verify = sub.add_parser("verify", help="Verify only the registry and allowlisted artifact identities")
    verify.add_argument("--suite", required=True, type=Path)
    verify.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    try:
        result = verify_suite(args.suite, args.root)
    except VerificationError as error:
        result = {"status": "FAIL", "scope": "registry-identity-only", "formal_evidence": False,
                  "suite_sha256": error.suite_sha256, "artifact_count": error.artifact_count,
                  "cell_count": error.cell_count, "error": str(error)}
        print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
        return 1
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
