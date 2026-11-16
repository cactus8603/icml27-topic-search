"""[codex] Thin invocation layer for hash-pinned *official* model code.

No research prompts, custom pooling, model class substitution, quantization,
CPU offload, attention fallback, or automatic batch reduction are provided.
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import re
import sys

import numpy as np


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verified_files(root, files, *, require_weights=False):
    root = Path(root).resolve()
    if not isinstance(files, dict) or not files:
        raise ValueError("A nonempty source/model SHA table is required")
    for name, expected in files.items():
        path = root / name
        if (Path(name).is_absolute() or ".." in Path(name).parts
                or not re.fullmatch(r"[0-9a-f]{64}", expected)
                or not path.resolve().is_relative_to(root) or file_sha(path) != expected):
            raise ValueError(f"Pinned official artifact differs or escapes root: {name}")
    if require_weights:
        import json
        if not any(name.endswith((".safetensors", ".bin")) for name in files):
            raise ValueError("The pinned model table must include actual weights")
        for name in files:
            if name.endswith(".index.json"):
                index = json.loads((root/name).read_text())
                if not set(index.get("weight_map", {}).values()) <= files.keys():
                    raise ValueError("Model weight index names an undeclared shard")


def official_module(root, relative, files):
    if relative not in files:
        raise ValueError("Official imported module is absent from the pinned SHA table")
    path = Path(root)/relative
    # A private name avoids importing an unrelated installed `models` package.
    name = "retrieval_matrix_official_"+file_sha(path)
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            del sys.modules[name]
            raise
    return sys.modules[name]


def validate_spec(spec):
    if spec.get("kind") not in {"official_embedding", "official_reranker"}:
        raise ValueError("Only a verified official embedding/reranker implementation is supported")
    if not re.fullmatch(r"[0-9a-f]{40}", spec.get("revision", "")) or not spec.get("repo"):
        raise ValueError("Model repository and immutable full revision are required")
    if Path(spec["snapshot"]).name != spec["revision"]:
        raise ValueError("The snapshot basename must identify the pinned model revision")
    official = spec["official"]
    if official.get("verified") is not True:
        raise ValueError("Official protocol has not passed the independent verification gate")
    for key in ("wrapper", "model_class"):
        if not isinstance(official.get(key), str) or not official[key]:
            raise ValueError(f"Missing official import field: {key}")
    if spec["kind"] == "official_embedding":
        for key in ("processor_class", "embedding", "model_name"):
            if not isinstance(official.get(key), str) or not official[key]:
                raise ValueError(f"Missing official embedding field: {key}")
    for key in ("image", "text", "cross"):
        if type(spec.get("batches", {}).get(key)) is not int or spec["batches"][key] <= 0:
            raise ValueError("Image/text/cross batches must each be fixed positive integers")
    if spec.get("loader", {}).get("torch_dtype", "bfloat16") not in {"bfloat16", "float16", "float32"}:
        raise ValueError("Unsupported explicit full-model dtype")
    forbidden = {"load_in_8bit", "load_in_4bit", "quantization_config", "device_map", "max_memory", "offload_folder"}
    if forbidden & spec.get("loader", {}).keys():
        raise ValueError("Model substitution, quantization, and CPU offload are forbidden")
    return spec


class OfficialModel:
    def __init__(self, spec, *, device="cuda"):
        import torch
        self.spec = validate_spec(spec)
        self.device = device
        official = spec["official"]
        verified_files(spec["snapshot"], spec["files"], require_weights=True)
        verified_files(official["root"], official["files"])
        kwargs = dict(spec.get("loader", {}))
        kwargs["torch_dtype"] = getattr(torch, kwargs.get("torch_dtype", "bfloat16"))
        self.dtype = kwargs["torch_dtype"]
        if spec["kind"] == "official_embedding":
            wrapper = official_module(official["root"], official["wrapper"], official["files"])
            cls = getattr(wrapper, official["model_class"])
            self.model = cls.from_pretrained(spec["snapshot"], device_map=device, local_files_only=True, **kwargs).eval()
            processor_cls = getattr(wrapper, official["processor_class"])
            self.processor = processor_cls.from_pretrained(spec["snapshot"], local_files_only=True, **spec.get("processor", {}))
            self.embedding = official_module(official["root"], official["embedding"], official["files"])
        else:
            if official.get("backend") != "ms-swift-generative-reranker":
                raise ValueError("Only the independently verified released ms-swift reranker backend may run")
            from swift.infer_engine import InferRequest, RequestConfig, TransformersEngine
            self.InferRequest, self.RequestConfig = InferRequest, RequestConfig
            self.scorer = TransformersEngine(spec["snapshot"], task_type="generative_reranker", device_map=device,
                                             **spec.get("processor", {}), **kwargs)
            self.model = self.scorer.model
        if getattr(self.model, "is_quantized", False):
            raise ValueError("Quantized inference cannot stand in for an official full model")
        for parameter in self.model.parameters():
            if parameter.device.type != device.split(":")[0] or (parameter.is_floating_point() and parameter.dtype != self.dtype):
                raise ValueError("Official model parameters differ from the requested full-model placement/dtype")

    def encode(self, items, *, image=False):
        import torch
        if self.spec["kind"] != "official_embedding":
            raise ValueError("This official model is a reranker, not an encoder")
        with torch.inference_mode():
            value = self.embedding.i2t_emb(self.spec["official"]["model_name"],
                                          [self.model, self.processor], list(items), is_query=not image)
            # This is the normalization used by official tasks/composite_dataset.
            value = torch.nn.functional.normalize(value, dim=-1)
        return value.float().cpu().numpy()

    def score(self, pairs):
        if self.spec["kind"] != "official_reranker":
            raise ValueError("This official model is an encoder, not a reranker")
        instruction = self.spec["official"]["instruction"]
        requests = [self.InferRequest(messages=[{"role": "system", "content": instruction},
                    {"role": "user", "content": text}, {"role": "assistant", "content": "<image>"}],
                    images=[str(path)]) for text, path in pairs]
        responses = self.scorer.infer(requests, self.RequestConfig(temperature=0, max_tokens=1500))
        # The release silently writes 0 on parse errors. A scientific run must
        # expose such failures instead of turning them into plausible scores.
        values = [float(response.choices[0].message.content[0]) for response in responses]
        if len(values) != len(pairs):
            raise ValueError("Official reranker did not return exactly one score per original pair")
        return np.asarray(values, dtype=np.float32)

    def signature(self):
        signature = {"implementation": self.spec["official"], "loader": self.spec.get("loader", {}),
                     "processor_configuration": self.spec.get("processor", {}), "batches": self.spec["batches"]}
        if hasattr(self, "processor"):
            tokenizer = self.processor.tokenizer
            signature["actual_processor"] = {"class": type(self.processor).__name__,
                "padding_side": tokenizer.padding_side, "eos_token": tokenizer.eos_token,
                "eod_id": getattr(self.processor, "eod_id", None),
                "min_pixels": getattr(self.processor, "min_pixels", None),
                "max_pixels": getattr(self.processor, "max_pixels", None),
                "instruction_standalone": getattr(self.processor, "instruction_standalone", None)}
        return signature
