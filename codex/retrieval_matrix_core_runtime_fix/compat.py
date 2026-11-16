"""[codex] Apply the pinned Transformers decorator's required factory call.

The released wrapper uses bare @check_model_inputs. Transformers 4.57.3
requires @check_model_inputs(). Recover the untouched original forward from
that known factory closure, then apply the real library decorator correctly.
No source payload, model math, attention backend or processor is replaced.
"""
from __future__ import annotations

import inspect
from pathlib import Path

from codex.retrieval_matrix import models as M

BaseOfficialModel = M.OfficialModel
IDENTITY = {
    "id": "core-qwen3-transformers457-check-model-inputs-factory-v1",
    "transformers": "4.57.3",
    "change": "apply check_model_inputs() to the unchanged original forward recovered from its bare-decorator closure",
    "model_math_changed": False,
}


def repair_class(cls):
    import transformers
    from transformers.utils.generic import check_model_inputs
    if transformers.__version__ != IDENTITY["transformers"]:
        raise ValueError("This explicit API compatibility repair only supports pinned Transformers4.57.3")
    marker = cls.__dict__.get("_codex_decorator_factory_repair")
    if marker:
        if marker["identity"] != IDENTITY or cls.forward is not marker["wrapped"]:
            raise ValueError("The repaired forward was subsequently modified")
        return marker["original"]
    broken = cls.__dict__.get("forward")
    if (not inspect.isfunction(broken) or broken.__module__ != "transformers.utils.generic"
            or broken.__name__ != "wrapped_fn"):
        raise ValueError("Refuse to repair a different or already valid model forward")
    closed = inspect.getclosurevars(broken).nonlocals
    original = closed.get("tie_last_hidden_states")
    if (set(closed) != {"tie_last_hidden_states"} or not inspect.isfunction(original)
            or original.__module__ != cls.__module__ or original.__qualname__ != cls.__name__+".forward"
            or "input_ids" not in inspect.signature(original).parameters):
        raise ValueError("The original released forward is not the expected bare-decorator closure")
    wrapped = check_model_inputs()(original)
    cls.forward = wrapped
    cls._codex_decorator_factory_repair = {"identity": dict(IDENTITY), "original": original, "wrapped": wrapped}
    return original


def official_class(spec):
    official = spec["official"]
    if official["wrapper"] != "models/qwen3_vl/qwen3_vl_wrapper.py" or official["model_class"] != "Qwen3VLForEmbedding":
        raise ValueError("The repair must not be applied to another model architecture")
    M.verified_files(official["root"], official["files"])
    module = M.official_module(official["root"], official["wrapper"], official["files"])
    return getattr(module, official["model_class"])


class CompatibleOfficialModel(BaseOfficialModel):
    def __init__(self, spec, *, device="cuda"):
        if spec.get("runtime_compatibility") != IDENTITY:
            raise ValueError("An independently frozen explicit compatibility model specification is required")
        repair_class(official_class(spec))
        super().__init__(spec, device=device)

    def signature(self):
        return {**super().signature(), "runtime_compatibility": dict(IDENTITY)}
