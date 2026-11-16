"""[codex] CPU-only complete source/data preflight; does not approve a plan."""
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path
import argparse
import json
import os
import runner as R


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path(__file__).parent / "plan.core-mcq.v1.json")
    parser.add_argument("--out", type=Path, default=R.ROOT / "dl/retrieval-matrix-rerank/core-mcq-preflight.json")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU preflight requires CUDA_VISIBLE_DEVICES='' before process start")
    plan = json.loads(args.plan.read_text())
    # Structural validation is independent of whether the reviewer has approved.
    structural = deepcopy(plan)
    for spec in structural["models"].values():
        spec["verified_protocol"] = True
    R.validate_plan(structural)
    for relative, digest in plan["source_files"].items():
        if R.sha(R.ROOT / relative) != digest:
            raise ValueError("Draft source bytes changed since plan construction")
    for entry in plan["external_sources"] + [plan["preparation"]]:
        R.artifact(entry)
    rows = R.load_examples(plan, verify_images=True)
    for name, expected in plan["environment"]["packages"].items():
        if version(name) != expected:
            raise ValueError("Isolated runtime metadata differs from frozen environment")
    value = {"owner": "[codex]", "state": "PASS", "technical_only": True,
             "gpu_inference": False, "weights_loaded": False, "independent_approval": False,
             "plan_sha256": R.sha(args.plan), "examples": len(rows),
             "pairs": sum(len(row["pairs"]) for row in rows), "unique_images": 9425,
             "counts": plan["counts"], "image_bytes_verified": True,
             "source_files": plan["source_files"], "packages": plan["environment"]["packages"]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: item for key, item in value.items() if key not in {"source_files", "packages"}}))


if __name__ == "__main__":
    main()
