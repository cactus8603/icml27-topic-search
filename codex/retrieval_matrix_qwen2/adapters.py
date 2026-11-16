"""[codex] CORE's released Qwen2 evaluation, with explicit V2 DoRA loading.

GME invokes the unmodified released Qwen2 wrapper. VLM2Vec-V2 adds the adapter
loading step missing from CORE's generic loader: PEFT attaches the pinned DoRA
to the complete Qwen2 model and merges it, as upstream VLM2Vec's inference
loader does. CORE's processor, i2t_emb, pooling and normalization remain intact.
"""
from __future__ import annotations

import gc
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from codex.retrieval_matrix import models as M

BaseOfficialModel = M.OfficialModel


def core_wrapper(spec):
    official = spec["official"]
    M.verified_files(official["root"], official["files"])
    return M.official_module(official["root"], official["wrapper"], official["files"])


def coverage(expected, actual):
    missing, unexpected = sorted(expected.keys()-actual.keys()), sorted(actual.keys()-expected.keys())
    shapes = sorted(k for k in expected.keys() & actual.keys() if tuple(expected[k].shape) != tuple(actual[k].shape))
    return {"expected": len(expected), "actual": len(actual), "missing": missing, "unexpected": unexpected,
            "shape_mismatch": shapes, "dora_magnitude_vectors": sum("lora_magnitude_vector" in k for k in actual),
            "all_keys_and_shapes_match": not (missing or unexpected or shapes)}


def validate_v2_config(folder):
    config = json.loads((Path(folder)/"adapter_config.json").read_text())
    if (config.get("peft_type") != "LORA" or config.get("use_dora") is not True
            or config.get("r") != 16 or config.get("lora_alpha") != 64
            or config.get("base_model_name_or_path") != "Qwen/Qwen2-VL-2B-Instruct"):
        raise ValueError("VLM2Vec-V2 must use the pinned official rank16 DoRA configuration and Qwen2-VL-2B base")
    return config


def adapter_state(folder):
    import torch
    return torch.load(Path(folder)/"adapter_model.bin", map_location="cpu", weights_only=True, mmap=True)


def meta_coverage(spec, *, official_vlm2vec=False):
    """Check every saved adapter key/shape without allocating base weights."""
    import torch
    from transformers import Qwen2VLConfig
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
    validate_v2_config(spec["snapshot"])
    if official_vlm2vec:
        source = spec["vlm2vec_source"]
        M.verified_files(source["root"], source["files"])
        sys.path.insert(0, source["root"])
        from src.model.vlm_backbone.qwen2_vl import Qwen2VLForConditionalGeneration, Qwen2VLConfig
        cls = Qwen2VLForConditionalGeneration
    else:
        cls = getattr(core_wrapper(spec), spec["official"]["model_class"])
    config = Qwen2VLConfig.from_pretrained(spec["base"]["snapshot"], local_files_only=True)
    config._name_or_path = spec["base"]["snapshot"]
    with torch.device("meta"):
        base = cls(config)
    lora = LoraConfig.from_pretrained(spec["snapshot"], local_files_only=True)
    base = get_peft_model(base, lora, low_cpu_mem_usage=True)
    result = coverage(get_peft_model_state_dict(base), adapter_state(spec["snapshot"]))
    result.update(model_class=cls.__module__+"."+cls.__name__, device="meta", weights_allocated=False,
                  base_parameters=sum(parameter.numel() for parameter in base.parameters()))
    del base
    gc.collect()
    if not result["all_keys_and_shapes_match"] or result["dora_magnitude_vectors"] != 140:
        raise ValueError(f"VLM2Vec-V2 adapter coverage failed: {result}")
    return result


class Qwen2OfficialModel(BaseOfficialModel):
    def __init__(self, spec, *, device="cuda"):
        if spec.get("variant") != "vlm2vec-v2-dora":
            if spec.get("variant") != "gme-qwen2":
                raise ValueError("The Qwen2 runner only supports pinned GME or VLM2Vec-V2")
            super().__init__(spec, device=device)
            return
        import torch
        from peft import PeftModel, get_peft_model_state_dict
        self.spec, self.device = M.validate_spec(spec), device
        self.dtype = torch.bfloat16
        if spec["loader"] != {"torch_dtype": "bfloat16", "attn_implementation": "flash_attention_2"}:
            raise ValueError("V2 uses the frozen released full BF16 / FlashAttention2 inference settings")
        M.verified_files(spec["snapshot"], spec["files"], require_weights=True)
        M.verified_files(spec["base"]["snapshot"], spec["base"]["files"], require_weights=True)
        validate_v2_config(spec["snapshot"])
        expected_coverage = meta_coverage(spec)
        wrapper = core_wrapper(spec)
        cls = getattr(wrapper, spec["official"]["model_class"])
        # Upstream's inference loader attaches to the full model, then merges.
        # Materializing on CPU here is construction, never inference offload.
        base = cls.from_pretrained(spec["base"]["snapshot"], local_files_only=True,
                                   torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
        attached = PeftModel.from_pretrained(base, spec["snapshot"], local_files_only=True, is_trainable=False)
        loaded, saved = get_peft_model_state_dict(attached), adapter_state(spec["snapshot"])
        actual_coverage = coverage(loaded, saved)
        if not actual_coverage["all_keys_and_shapes_match"]:
            raise ValueError("Materialized VLM2Vec DoRA key/shape coverage differs")
        unequal = [key for key in saved if not torch.equal(loaded[key].detach().cpu(), saved[key].to(loaded[key].dtype))]
        if unequal:
            raise ValueError(f"PEFT did not load every original adapter tensor exactly: {unequal[:8]}")
        del loaded, saved
        self.model = attached.merge_and_unload(safe_merge=True).to(device=device, dtype=torch.bfloat16).eval()
        if any("lora_" in name for name, _ in self.model.named_parameters()):
            raise ValueError("Adapter merge left unexpected live LoRA parameters")
        for parameter in self.model.parameters():
            if parameter.device.type != device.split(":")[0] or parameter.dtype != torch.bfloat16:
                raise ValueError("The fully merged V2 model must reside entirely on the selected GPU in BF16")
        proc_cls = getattr(wrapper, spec["official"]["processor_class"])
        self.processor = proc_cls.from_pretrained(spec["snapshot"], local_files_only=True, **spec["processor"])
        self.embedding = M.official_module(spec["official"]["root"], spec["official"]["embedding"], spec["official"]["files"])
        self.adapter_receipt = {"meta_coverage": expected_coverage, "loaded_coverage": actual_coverage,
            "every_saved_adapter_tensor_loaded_exactly": True, "merged": True,
            "loading_source": spec["vlm2vec_source"],
            "disclosed_change": "CORE's released loader omitted VLM2Vec-V2 DoRA. Attach the official adapter to its complete base before merging; retain CORE's released Qwen2 evaluation wrapper and tokenizer/processor/i2t/pooling."}

    def signature(self):
        output = super().signature()
        if hasattr(self, "adapter_receipt"):
            output["adapter_receipt"] = self.adapter_receipt
        return output
