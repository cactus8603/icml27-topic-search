"""[codex] Strict, portable evidence helpers for native benchmark qualification."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read(path):
    def unique(pairs):
        output = {}
        for key, value in pairs:
            if key in output:
                raise ValueError("Duplicate evidence key: " + key)
            output[key] = value
        return output
    def invalid(value):
        raise ValueError("Nonfinite evidence: " + value)
    return json.loads(Path(path).read_text(), object_pairs_hook=unique, parse_constant=invalid)


def entry(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path)}


def checked(item):
    path = Path(item["path"]).resolve()
    if sha(path) != item["sha256"]:
        raise ValueError("Evidence payload changed: " + str(path))
    return path


def write(path, value):
    """Never replace an existing result or silently update a frozen receipt."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if read(path) != value:
            raise ValueError("Existing evidence differs: " + str(path))
        return
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
        indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
