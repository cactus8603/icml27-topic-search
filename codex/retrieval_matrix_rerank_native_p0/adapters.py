"""[codex] Invoke pinned frozen backends; declared validation wrappers only."""
from codex.retrieval_matrix_rerank import backends as B


def create(variant):
    backend=B.BACKENDS[variant["model"]["backend"]](variant["model"])
    kind=variant["model"]["backend"]
    if kind=="swift_released":return backend
    import torch
    model=backend.engine.model if kind in {"qwen_native","mcmr_pointwise"} else backend.engine
    if getattr(model,"is_quantized",False):raise ValueError("Native P0 cannot substitute quantization")
    if any(p.device.type!="cuda" or (p.is_floating_point() and p.dtype!=torch.bfloat16) for p in model.parameters()):
        raise ValueError("Native P0 requires all original BF16 weights on assigned GPU; no CPU offload")
    if kind=="jina_native":
        import inspect
        if B.sha(inspect.getfile(type(backend.engine)))!=variant["model"]["files"]["modeling.py"]:
            raise ValueError("Actual imported Jina remote code differs from original pinned source")
    if kind=="qwen_native":
        # Original helper can silently replace failed vision preprocessing by NULL.
        # Validate its unchanged return before compute_scores, never save a fallback.
        tokenize=backend.engine.tokenize
        def checked_tokenize(messages,**kwargs):
            inputs=tokenize(messages,**kwargs)
            if ("pixel_values" not in inputs or "image_grid_thw" not in inputs
                    or len(inputs["image_grid_thw"])!=len(messages)):
                raise ValueError("Official Qwen vision helper silently dropped an image; fail before model forward")
            return inputs
        backend.engine.tokenize=checked_tokenize
    return backend
