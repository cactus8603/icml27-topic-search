"""[codex] A native-qualified CLIP retains its complete P0 model identity."""
from pathlib import Path

from codex.retrieval_matrix_extra import adapters
from codex.retrieval_matrix_native_p0 import qualification as Q

PHASE = "P2-qualified-fixed-clip-baselines"
SUPPORTED = {"siglip2"}
ROOT = Path(__file__).resolve().parents[2]


def validate_model(spec):
    adapters.validate_spec(spec)
    if spec.get("family") != "CLIP" or spec.get("extra_family") not in SUPPORTED:
        raise ValueError("This CLIP panel currently supports the reviewed SigLIP2 native P0 path only")


def admit_models(plan):
    if plan.get("phase") != PHASE or plan.get("retriever_family") != "CLIP":
        raise ValueError("A separate qualified CLIP panel phase is required")
    if not plan.get("models") or set(plan["models"]) != set(plan.get("qualification", {})):
        raise ValueError("Every CLIP model requires exactly its own native qualification evidence")
    result = {}
    for key, spec in plan["models"].items():
        validate_model(spec)
        receipt = Q.admit(plan["qualification"][key], spec)
        if receipt.get("model") != key or receipt.get("family") != "CLIP":
            raise ValueError("Native qualification belongs to a different model or family")
        if plan.get("parent_p0_plan") != receipt["p0_plan"]:
            raise ValueError("CLIP panel parent plan differs from the qualified native P0")
        parent = Q.read(Q.checked(receipt["p0_plan"]))
        dependencies = Q.entry(ROOT/"codex/retrieval_matrix_extra/dependencies.json")
        if parent.get("dependencies") != dependencies or plan.get("dependencies") != dependencies:
            raise ValueError("CLIP panel dependencies must equal its qualified native P0 dependency entry")
        settings = parent["settings"]
        if (plan.get("parent_native_settings") != settings
                or settings.get("seed") != 2026111665 or settings.get("allow_tf32") is not False
                or plan.get("settings", {}).get("seed") != settings["seed"]
                or plan["settings"].get("allow_tf32") is not False
                or plan["settings"].get("batch_invariance_cosine_min") != settings["smoke_cosine_min"]
                or plan["settings"].get("batch_invariance_max_abs") != settings["smoke_max_abs"]):
            raise ValueError("CLIP panel changes the qualified forward or technical smoke settings")
        result[key] = receipt
    return result


def check_declared_sources(plan, root):
    """The additional dependencies cannot disappear from the source SHA table."""
    for folder in ("retrieval_matrix_clip_panel", "retrieval_matrix_extra", "retrieval_matrix_native_p0"):
        for path in sorted((Path(root)/"codex"/folder).iterdir()):
            if path.suffix in {".py", ".md", ".sh", ".json"}:
                relative = str(path.relative_to(root))
                if plan.get("source_files", {}).get(relative) != Q.sha(path):
                    raise ValueError("CLIP P2 source or dependency is not frozen: "+relative)
