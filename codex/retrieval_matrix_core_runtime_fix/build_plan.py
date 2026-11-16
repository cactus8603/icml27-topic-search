"""[codex] Preserve failed-smoke protocols in separately frozen repair plans."""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from codex.retrieval_matrix import runner as R
from codex.retrieval_matrix_core_runtime_fix import compat
from codex.retrieval_matrix_core_runtime_fix.runner import OWN, validate_repair


def build(path):
    original = R.read(path)
    if original.get("phase") not in {"P0", "MCMR-native-MMEB-v2"} or original.get("state") != "FROZEN":
        raise ValueError("Only the original frozen CORE/Qwen3 embedding protocols may be repaired")
    result = copy.deepcopy(original)
    for spec in result["models"].values():
        if spec["official"]["wrapper"] != "models/qwen3_vl/qwen3_vl_wrapper.py":
            raise ValueError("This repair does not cover Qwen2 or other model families")
        spec["runtime_compatibility"] = copy.deepcopy(compat.IDENTITY)
    result["runtime_repair"] = {"identity": copy.deepcopy(compat.IDENTITY),
        "parent_plan": {"path": str(path.resolve()), "sha256": R.sha(path)},
        "reason": "Original synthetic smoke failed before any benchmark inference: released bare decorator is incompatible with pinned Transformers4.57.3 factory API",
        "outputs": "new isolated directory; preserve every previous failure; no automatic replay"}
    result["prereg"] = "codex/retrieval_matrix_core_runtime_fix/PREREG.md"
    for name in OWN:
        relative = "codex/retrieval_matrix_core_runtime_fix/"+name
        result["source_files"][relative] = R.sha(ROOT/relative)
    validate_repair(result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-plan", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Never overwrite an existing repair plan")
    args.out.write_text(json.dumps(build(args.parent_plan), ensure_ascii=False, sort_keys=True, indent=2)+"\n")
    print(json.dumps({"path": str(args.out), "sha256": R.sha(args.out), "requires_commit": True}))
