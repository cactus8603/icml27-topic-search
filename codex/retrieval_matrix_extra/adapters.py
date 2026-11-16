"""[codex] Pinned author inference paths for additional official baselines.

All models expose encode(items, image=False) and signature(), matching the frozen
matrix checkpoint producer. Dataset labels never enter this interface.
"""
from __future__ import annotations

import copy
import importlib
from functools import lru_cache
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from codex.retrieval_matrix import models as M

CORE_MODEL = M.OfficialModel
CORE_VALIDATE = M.validate_spec
FAMILIES = {"siglip2", "clic", "clic-clips", "author-openclip", "triplet-separate", "unime-v2-native", "gme-core"}


def validate_spec(spec):
    if spec.get("extra_family") not in FAMILIES or spec.get("kind") != "official_embedding":
        raise ValueError("An explicit supported extra baseline family is required")
    if spec.get("qualification") != "P0_REQUIRED":
        raise ValueError("An acquired checkpoint cannot silently become a qualified P2 baseline")
    for field in ("repo", "revision", "variant", "snapshot", "files", "sources", "native_protocol"):
        if not spec.get(field):
            raise ValueError(f"Missing fixed baseline identity: {field}")
    if not re.fullmatch(r"[0-9a-f]{40}", spec["revision"]):
        raise ValueError("Author repository/model revision must be a full immutable commit")
    expected_loader = ({"torch_dtype":"bfloat16","attn_implementation":"flash_attention_2"}
                       if spec["extra_family"] in {"gme-core","unime-v2-native"} else {"torch_dtype":"float32"})
    if spec["loader"] != expected_loader:
        raise ValueError("Loader must exactly match the full native dtype/attention protocol; no quantization, offload or ignored overrides")
    for key in ("image", "text", "cross"):
        if type(spec["batches"][key]) is not int or spec["batches"][key] < 1:
            raise ValueError("Batch sizes must be explicit positive integers")
    if len(spec.get("published_targets", [])) < 2:
        raise ValueError("P0 preparation must name at least two benchmark targets")
    return spec


def verify(spec):
    validate_spec(spec)
    M.verified_files(spec["snapshot"], spec["files"], require_weights=True)
    for source in spec["sources"].values():
        M.verified_files(source["root"], source["files"])
    for support in spec.get("support", {}).values():
        M.verified_files(support["snapshot"], support["files"])


def _source(spec, key):
    source = spec["sources"][key]
    M.verified_files(source["root"], source["files"])
    return Path(source["root"])


@lru_cache(maxsize=4)
def original_function(source, name):
    """Import one unchanged author function without unrelated training imports."""
    # The AST-extracted function is executed from the pinned original source;
    # no copied regular expression or alternative text normalization is used.
    import ast
    source = Path(source)
    tree = ast.parse(source.read_text())
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = {"re": re}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[name]


def clic_pre_caption(text):
    source = ROOT/"dl/retrieval-matrix-extra/sources/clic/eval/retrieval_datasets.py"
    return original_function(str(source), "pre_caption")(text, max_words=30)


def openclip_parts(spec, *, weights=True, device="cpu"):
    """Use the author's exact factory/config while routing downloads to pinned local bytes."""
    import torch
    if spec["extra_family"] == "clic-clips":
        sys.path.insert(0, str(_source(spec, "clic")))
        library = importlib.import_module("clips")
    else:
        library = importlib.import_module("open_clip")
        if library.__version__ != "2.31.0":
            raise ValueError("CLIC's author-specified open_clip_torch==2.31.0 is required")
    factory = importlib.import_module(library.__name__+".factory")
    if "config_file" in spec:
        import json
        config = json.loads((Path(spec["snapshot"])/spec["config_file"]).read_text())
        model_cfg = copy.deepcopy(config["model_cfg"])
        # Pin the otherwise mutable bare bert-base-uncased tokenizer reference.
        if spec["extra_family"] == "clic-clips":
            model_cfg["text_cfg"]["hf_tokenizer_name"] = spec["support"]["bert"]["snapshot"]
        alias = "extra_"+spec["variant"].replace("-", "_")
        factory._MODEL_CONFIGS[alias] = model_cfg
        pp = config["preprocess_cfg"]
        name = alias
        kwargs = {"image_mean": pp["mean"], "image_std": pp["std"],
                  "image_interpolation": pp["interpolation"], "image_resize_mode": pp["resize_mode"]}
    else:
        # The original NegCLIP release explicitly calls this standard factory
        # name with a local checkpoint. Do not silently force QuickGELU.
        name, kwargs = "ViT-B-32", {}
    if weights:
        model, _, processor = library.create_model_and_transforms(name,
            pretrained=str(Path(spec["snapshot"])/spec["weight_file"]), precision="fp32",
            pretrained_hf=False, device=device, **kwargs)
    else:
        # A meta model checks all checkpoint keys/shapes without materializing weights.
        with torch.device("meta"):
            model = library.create_model(name, pretrained=None, precision="fp32", pretrained_hf=False, device="meta")
        from dataclasses import asdict
        from open_clip.transform import PreprocessCfg, image_transform_v2
        pp = asdict(PreprocessCfg()) | {k: v for k, v in kwargs.items() if not k.startswith("image_")}
        if "config_file" in spec:
            pp.update(config["preprocess_cfg"])
            pp["size"] = model.visual.image_size
        processor = image_transform_v2(PreprocessCfg(**pp), is_train=False)
    tokenizer = library.get_tokenizer(name)
    return model.eval(), processor, tokenizer


def triplet_parts(spec, *, weights=True, device="cpu"):
    import torch
    from transformers import CLIPTokenizerFast, CLIPImageProcessor
    source = spec["sources"]["checkpoint_code"]
    module = M.official_module(source["root"], "utils.py", source["files"])
    folder = Path(spec["snapshot"])
    output = []
    for kind in ("Vision", "Text"):
        cls = getattr(module, f"CLIP{kind}EncoderOnly")
        config_cls = getattr(module, f"CLIP{kind}EncoderOnlyConfig")
        subfolder = folder/("vision-encoder" if kind == "Vision" else "text-encoder")
        config = config_cls.from_pretrained(subfolder, local_files_only=True)
        if config.model_name != "openai/clip-vit-base-patch32" or config.pretrained or config.lora:
            raise ValueError("Expected the author's scratch-trained full ViT-B/32 checkpoint")
        config.model_name = spec["support"]["clip"]["snapshot"]
        if weights:
            model, info = cls.from_pretrained(subfolder, config=config, local_files_only=True, output_loading_info=True)
            if any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
                raise ValueError(f"Separate encoder checkpoint did not load strictly: {info}")
            model = model.to(device=device, dtype=torch.float32)
        else:
            with torch.device("meta"):
                model = cls(config)
        output.append(model.eval())
    base = spec["support"]["clip"]["snapshot"]
    output.extend([CLIPTokenizerFast.from_pretrained(base, local_files_only=True),
                   CLIPImageProcessor.from_pretrained(base, local_files_only=True)])
    return output


def unime_parts(spec, *, weights=True):
    # Keep the author's model, pooling, processor, batching and resize routines.
    # Import under its intended src package, in a separate baseline process.
    sys.path.insert(0, str(_source(spec, "unime_v2")/"Embedding"))
    from src.arguments import ModelArguments, DataArguments
    from src.model import MMEBModel
    from src.model_utils import load_processor, Qwen2_VL_process_fn
    process_image = original_function(str(_source(spec, "unime_v2")/"Embedding/src/dataset.py"), "process_image")
    args = ModelArguments(model_name=spec["snapshot"], processor_name=spec["snapshot"],
                          pooling="eos", normalize=True)
    args.model_backbone = "qwen2_vl"
    data_args = DataArguments(max_len=4096, image_resolution="high")
    processor = load_processor(args)
    model = MMEBModel.load(args) if weights else None
    return model, processor, Qwen2_VL_process_fn, process_image, data_args


class ExtraOfficialModel:
    def __init__(self, spec, *, device="cuda"):
        import torch
        verify(spec)
        self.spec, self.device = spec, device
        self.family = spec["extra_family"]
        self.dtype = getattr(torch, spec["loader"]["torch_dtype"])
        if self.family == "gme-core":
            core_spec = {**spec, "official": spec["core_official"]}
            self.delegate = CORE_MODEL(core_spec, device=device)
        elif self.family in {"clic", "clic-clips", "author-openclip"}:
            self.model, self.processor, self.tokenizer = openclip_parts(spec, device=device)
        elif self.family == "siglip2":
            from transformers import SiglipModel, AutoProcessor
            self.model = SiglipModel.from_pretrained(spec["snapshot"], local_files_only=True,
                         torch_dtype=torch.float32).to(device).eval()
            self.processor = AutoProcessor.from_pretrained(spec["snapshot"], local_files_only=True)
        elif self.family == "triplet-separate":
            self.parts = triplet_parts(spec, device=device)
        elif self.family == "unime-v2-native":
            self.model, self.processor, self.unime_process, self.unime_resize, self.data_args = unime_parts(spec)
            self.model = self.model.to(device=device, dtype=torch.bfloat16).eval()

    def encode(self, items, *, image=False):
        import torch
        from PIL import Image
        items = list(items)
        if hasattr(self, "delegate"):
            return self.delegate.encode(items, image=image)
        with torch.inference_mode():
            if self.family in {"clic", "clic-clips", "author-openclip"}:
                if image:
                    batch = torch.stack([self.processor(Image.open(item).convert("RGB")) for item in items]).to(self.device)
                    value = self.model.encode_image(batch)
                else:
                    if self.spec["native_protocol"].get("text_cleaner") == "clic-pre-caption-30":
                        items = [clic_pre_caption(item) for item in items]
                    value = self.model.encode_text(self.tokenizer(items).to(self.device))
            elif self.family == "siglip2":
                if image:
                    batch = self.processor(images=[Image.open(item).convert("RGB") for item in items], return_tensors="pt").to(self.device)
                    value = self.model.get_image_features(**batch)
                else:
                    batch = self.processor(text=[item.lower() for item in items], padding="max_length", max_length=64,
                                           truncation=True, return_tensors="pt").to(self.device)
                    value = self.model.get_text_features(**batch)
            elif self.family == "triplet-separate":
                source = self.spec["sources"]["core"]
                embedding = M.official_module(source["root"], "embedding.py", source["files"])
                value = embedding.i2t_emb("tripletclip", self.parts, items, is_query=not image)
            elif self.family == "unime-v2-native":
                images = [self.unime_resize(Image.open(item).convert("RGB"), "high") for item in items] if image else [None]*len(items)
                texts = ["<|image_pad|> Find an image caption describing the given image."]*len(items) if image else items
                batch = self.unime_process({"text": texts, "image": images}, processor=self.processor, max_length=4096)
                batch = {key: value.to(self.device) if torch.is_tensor(value) else value for key, value in batch.items()}
                value = self.model(qry=batch)["qry_reps"]
            else:
                raise ValueError("Unsupported frozen baseline family")
            value = torch.nn.functional.normalize(value, dim=-1)
        return value.float().cpu().numpy()

    def signature(self):
        return {"variant": self.spec["variant"], "extra_family": self.family,
                "native_protocol": self.spec["native_protocol"], "sources": self.spec["sources"],
                "loader": self.spec["loader"], "batches": self.spec["batches"],
                "qualification": "P0_REQUIRED", "local_path_routing_only": True,
                "actual_classes": [type(value).__module__+"."+type(value).__name__ for value in
                                   (getattr(self, "model", None), getattr(self, "processor", None)) if value is not None]}
