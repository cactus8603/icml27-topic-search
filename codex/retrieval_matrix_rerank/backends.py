"""[codex] Thin, explicit calls to pinned released reranker implementations.

No model is instantiated at import. GPU production belongs to a separately
frozen runner. Input rows contain id/query/image and optional candidate_text.
Every result preserves the official raw score in addition to its scalar score.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(spec):
    revision = spec["revision"]
    root = Path(spec["snapshot"]).resolve()
    if root.name != revision or len(revision) != 40:
        raise ValueError("A full immutable model snapshot revision is required")
    files = spec["files"]
    if not files or not any(n.endswith((".bin", ".safetensors")) for n in files):
        raise ValueError("All actual model payload checksums must be frozen")
    for name, expected in files.items():
        path = root / name
        if Path(name).is_absolute() or ".." in Path(name).parts or not path.resolve().is_relative_to(root) or sha(path) != expected:
            raise ValueError("Pinned checkpoint bytes differ")
        if name.endswith(".index.json"):
            shards = set(json.loads(path.read_text()).get("weight_map", {}).values())
            if not shards or not shards <= files.keys():
                raise ValueError("Pinned model index has missing weight shards")
    if spec.get("verified_protocol") is not True:
        raise ValueError("Protocol requires an independent review before production")
    return root


def module(spec):
    path = Path(spec["path"]).resolve()
    if sha(path) != spec["sha256"]:
        raise ValueError("Official source bytes differ")
    name = "official_rerank_" + spec["sha256"]
    if name not in sys.modules:
        source = importlib.util.spec_from_file_location(name, path)
        value = importlib.util.module_from_spec(source)
        sys.modules[name] = value
        source.loader.exec_module(value)
    return sys.modules[name]


def checked_scores(rows, values, raw=None):
    values = [float(value) for value in values]
    if len(values) != len(rows) or not all(math.isfinite(value) for value in values):
        raise ValueError("Official backend failed a complete fixed pair list")
    if raw is None:
        raw = values
    return [{"id": row["id"], "score": value, "official_raw": original}
            for row, value, original in zip(rows, values, raw, strict=True)]


class SwiftReleased:
    """Exact CORE released engine, prompt and float32 activated score path."""
    def __init__(self, spec):
        import torch
        from swift.infer_engine import TransformersEngine, InferRequest, RequestConfig
        snapshot = verify(spec)
        # The official CORE main.py sets this value before constructing engine.
        if os.environ.get("IMAGE_MAX_TOKEN_NUM") != "1200":
            raise ValueError("Set frozen IMAGE_MAX_TOKEN_NUM=1200 before Swift imports")
        if (os.environ.get("GENERATIVE_RERANKER_POSITIVE_TOKEN", "yes") != "yes"
                or os.environ.get("GENERATIVE_RERANKER_NEGATIVE_TOKEN", "no") != "no"):
            raise ValueError("Published Swift score labels must remain yes/no")
        for name, expected in spec["swift_sources"].items():
            if sha(name) != expected:
                raise ValueError("Pinned ms-swift source differs")
        self.InferRequest, self.RequestConfig = InferRequest, RequestConfig
        self.engine = TransformersEngine(str(snapshot), task_type="generative_reranker",
            torch_dtype=torch.float16, attn_impl="flash_attention_2", device_map="cuda:0",
            max_batch_size=spec.get("max_batch_size", 32), max_length=1500,
            use_hf=True, download_model=False)
        if (getattr(self.engine.model, "is_quantized", False)
                or any(parameter.device.type != "cuda" or
                       (parameter.is_floating_point() and parameter.dtype != torch.float16)
                       for parameter in self.engine.model.parameters())):
            raise ValueError("Released CORE must remain full float16 on GPU without quantization or CPU offload")
        self.signature = {"backend": "CORE-released-ms-swift", "model_type": self.engine.model.model_info.model_type,
                          "template": self.engine.template.template_meta.template_type,
                          "task_type": self.engine.template.task_type, "image_max_token_num": 1200,
                          "max_length": 1500, "dtype": "float16", "raw_score": "content[0]; sigmoid(float32(logit_yes-logit_no))"}
        if self.signature["template"] != "qwen3_vl_reranker":
            raise ValueError("Official checkpoint args must select qwen3_vl_reranker")

    def score_pairs(self, rows, *, instruction):
        requests = []
        for row in rows:
            if row.get("candidate_text"):
                raise ValueError("CORE MCQ release receives image-only documents; fused adaptation must be separately frozen")
            requests.append(self.InferRequest(messages=[{"role": "system", "content": instruction},
                {"role": "user", "content": row["query"]}, {"role": "assistant", "content": "<image>"}],
                images=[str(row["image"])]))
        response = self.engine.infer(requests, self.RequestConfig(temperature=0, max_tokens=1500))
        raw = [value.choices[0].message.content for value in response]
        # Fail loudly rather than copying the upstream silent parse-error->zero.
        return checked_scores(rows, [float(value[0]) for value in raw], raw)


class QwenNative:
    """Official Qwen API for P3; a distinct protocol from CORE's Swift path."""
    def __init__(self, spec):
        import torch
        snapshot = verify(spec)
        official = module(spec["source"])
        self.engine = official.Qwen3VLReranker(str(snapshot), torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2", **spec.get("processor", {}))
        self.signature = {"backend": "Qwen-official-native", "dtype": "bfloat16",
                          "max_length": self.engine.max_length, "min_pixels": self.engine.min_pixels,
                          "max_pixels": self.engine.max_pixels, "raw_score": "official process scalar; native-dtype sigmoid"}

    def score_pairs(self, rows, *, instruction):
        values = []
        for row in rows:
            document = {"image": str(row["image"])}
            if row.get("candidate_text") is not None:
                document["text"] = row["candidate_text"]
            score = self.engine.process({"instruction": instruction, "query": {"text": row["query"]},
                                         "documents": [document]})
            if len(score) != 1:
                raise ValueError("Native Qwen did not return one score for original pair")
            values.extend(score)
        return checked_scores(rows, values)


class JinaNative:
    def __init__(self, spec):
        from transformers import AutoModel
        snapshot = verify(spec)
        self.engine = AutoModel.from_pretrained(str(snapshot), local_files_only=True,
            torch_dtype="auto", trust_remote_code=True, attn_implementation="flash_attention_2").to("cuda:0").eval()
        self.signature = {"backend": "Jina-official-remote-code", "max_length": 2048,
                          "doc_type": "image", "raw_score": "official compute_score scalar"}

    def score_pairs(self, rows, *, instruction=None):
        if any(row.get("candidate_text") for row in rows):
            raise ValueError("Jina released image-document protocol does not consume extra metadata")
        values = self.engine.compute_score([[row["query"], str(row["image"])] for row in rows],
                                           max_length=2048, doc_type="image")
        if not isinstance(values, (list, tuple)):
            values = [values]
        return checked_scores(rows, values)


class MCMRPointwise:
    """Calls the author's unchanged LabelScorer; retains both branch logprobs."""
    def __init__(self, spec):
        snapshot = verify(spec)
        self.official = module(spec["source"])
        tokenizer, processor, model = self.official.load_qwen(str(snapshot))
        self.engine = self.official.LabelScorer(tokenizer, processor, model)
        self.signature = {"backend": "MCMR-official-Qwen3VL8-pointwise", "dtype": "bfloat16",
                          "labels": ["True", "False"], "true_ids": self.engine.true_ids,
                          "false_ids": self.engine.false_ids, "two_forwards_per_pair": True,
                          "candidate_renderer": "official render_candidate_text, max2048 characters",
                          "raw_score": "p_true, true_logprob, false_logprob"}

    def score_pairs(self, rows, *, instruction=None):
        from PIL import Image
        values, raw = [], []
        for row in rows:
            text = (self.official.render_candidate_text(row["candidate_raw"])
                    if "candidate_raw" in row else row.get("candidate_text", ""))
            with Image.open(row["image"]) as image:
                score, true_lp, false_lp = self.engine.score_pair(row["query"], text, image.convert("RGB"))
            values.append(score)
            raw.append({"p_true": score, "true_logprob": true_lp, "false_logprob": false_lp})
        return checked_scores(rows, values, raw)


BACKENDS = {"swift_released": SwiftReleased, "qwen_native": QwenNative,
            "jina_native": JinaNative, "mcmr_pointwise": MCMRPointwise}
