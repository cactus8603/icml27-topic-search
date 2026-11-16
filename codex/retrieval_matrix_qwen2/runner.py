"""[codex] Independent frozen entry for GME / VLM2Vec-V2 P0 reproduction.

Reuses the committed matrix producer/checkpoints/CPU scorer without editing its
source or dependency environment. The sole model extension applies V2 DoRA.
"""
from pathlib import Path
from importlib.metadata import version
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from codex.retrieval_matrix import runner as R
from codex.retrieval_matrix_qwen2.adapters import Qwen2OfficialModel

original_gate = R.source_gate
original_environment = R.runtime_environment
R.models.OfficialModel = Qwen2OfficialModel
R.SOURCE_FILES += ("codex/retrieval_matrix_qwen2/runner.py", "codex/retrieval_matrix_qwen2/adapters.py")


def source_gate(plan_path, plan, freeze_commit):
    output = original_gate(plan_path, plan, freeze_commit)
    status = subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain", "--", str(HERE.relative_to(ROOT))], text=True)
    if status.strip():
        raise ValueError("Qwen2 production requires its own clean committed topic directory")
    return output


R.source_gate = source_gate


def runtime_environment(spec):
    # The shared producer predates PEFT. Extend its actual-environment receipt
    # here, without weakening any existing package or module checks.
    import peft
    import flash_attn
    reduced = {**spec, "environment": {key: value for key, value in spec.get("environment", {}).items() if key != "peft"}}
    actual = original_environment(reduced)
    actual["peft"] = version("peft")
    actual["modules"]["peft"] = {"version": peft.__version__, "path": str(Path(peft.__file__).resolve())}
    actual["modules"]["flash_attn"] = {"version": flash_attn.__version__, "path": str(Path(flash_attn.__file__).resolve())}
    for key, expected in spec.get("environment", {}).items():
        if actual.get(key) != expected:
            raise ValueError(f"Frozen Qwen2 runtime differs: {key}")
    return actual


R.runtime_environment = runtime_environment

if __name__ == "__main__":
    R.main()
