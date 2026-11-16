"""[codex] Verify isolated dependency bytes before importing official code."""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.plan.read_text())["dependencies"]
    for relative, expected in spec["files"].items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT / "dl/retrieval-matrix/deps"):
            raise ValueError("Dependency path escapes its isolated target")
        with path.open("rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() != expected:
                raise ValueError("Frozen official dependency payload differs")
    for name, expected in spec["packages"].items():
        if version(name) != expected:
            raise ValueError(f"Frozen package version differs: {name}")
    import torch
    import transformers
    if torch.__version__ != spec["packages"]["torch"] or transformers.__version__ != spec["packages"]["transformers"]:
        raise ValueError("Actual imported module version differs from distribution metadata")
    print(json.dumps({"owner": "[codex]", "dependencies": "PASS", "files": len(spec["files"]),
                      "torch": torch.__version__, "transformers": transformers.__version__}))


if __name__ == "__main__":
    main()
