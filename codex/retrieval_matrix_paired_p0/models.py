"""[codex] CLIC author FP32 batch1 helpers and exact paired tokenizer policy."""
from contextlib import contextmanager
import copy
import importlib
import os
from pathlib import Path
import sys

from codex.retrieval_matrix_extra import adapters as A
from codex.retrieval_matrix_native_p0.evidence import ROOT, digest, read


@contextmanager
def source_directory(path):
    previous = Path.cwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(previous)


def verify(spec):
    if (spec.get("family") != "CLIP" or spec["variant"] not in {"clic_b32", "clic_l14", "clic_clips_l14"}
            or spec["native_protocol"].get("text_cleaner") != "raw-caption-to-author-tokenizer"
            or spec["batches"] != {"image": 1, "text": 1, "cross": 1}):
        raise ValueError("Paired CLIC needs its explicit raw-caption FP32 author batch1 profile")
    baseline = read(ROOT/"codex/retrieval_matrix_extra/registry.json")["models"][spec["variant"]]
    if spec["baseline_model_spec_sha256"] != digest(baseline):
        raise ValueError("Paired identity does not bind the unchanged acquired official checkpoint")
    for key in ("repo", "revision", "snapshot", "files", "sources", "loader", "support", "config_file", "weight_file"):
        if spec.get(key) != baseline.get(key):
            raise ValueError("Paired native task cannot substitute checkpoint/source bytes: " + key)
    A.verify(spec)


def parts(spec, *, weights=True, device="cpu"):
    """Author modelcards use OpenCLIP models, including CLIC-CLIPS weights.

    Route only the factory configuration and pretrained path to fixed local
    bytes. CLIPS gets the author's CustomTokenizer branch, never the lowercase
    alias/HFTokenizer branch from the earlier generic acquisition wrapper.
    """
    import torch
    import open_clip
    from open_clip import factory
    if open_clip.__version__ != "2.31.0":
        raise ValueError("The CLIC-pinned OpenCLIP version is required")
    config = read(Path(spec["snapshot"])/spec["config_file"])
    model_cfg = copy.deepcopy(config["model_cfg"])
    alias = "PAIRED_"+spec["variant"].upper()
    if spec["extra_family"] == "clic-clips":
        model_cfg["text_cfg"]["hf_tokenizer_name"] = spec["support"]["bert"]["snapshot"]
    factory._MODEL_CONFIGS[alias] = model_cfg
    pp = config["preprocess_cfg"]
    if weights:
        model, _, transform = open_clip.create_model_and_transforms(alias,
            pretrained=str(Path(spec["snapshot"])/spec["weight_file"]), precision="fp32",
            pretrained_hf=False, device=device, image_mean=pp["mean"], image_std=pp["std"],
            image_interpolation=pp["interpolation"], image_resize_mode=pp["resize_mode"])
    else:
        with torch.device("meta"):
            model = open_clip.create_model(alias, pretrained=None, precision="fp32", pretrained_hf=False, device="meta")
        from open_clip.transform import PreprocessCfg, image_transform_v2
        transform = image_transform_v2(PreprocessCfg(size=model.visual.image_size, **pp), is_train=False)
    if spec["extra_family"] == "clic-clips":
        source = Path(spec["sources"]["clic"]["root"])
        sys.path.insert(0, str(source))
        clips = importlib.import_module("clips")
        clips_factory = importlib.import_module("clips.factory")
        clips_factory._MODEL_CONFIGS[alias] = model_cfg
        # The unchanged author constructor resolves ./clips/vocab.txt; preserve
        # its intended directory solely while constructing the tokenizer.
        with source_directory(source):
            tokenizer = clips.get_tokenizer(alias, context_length=model.context_length)
        if type(tokenizer).__name__ != "CLIPS_Tokenizer":
            raise ValueError("CLIPS must use the author's custom BOS/EOS/trailing-class tokenizer")
    else:
        tokenizer = open_clip.get_tokenizer(alias)
    return model.eval(), transform, tokenizer


class PairedModel:
    def __init__(self, spec, device="cuda"):
        verify(spec)
        self.spec, self.device = spec, device
        self.model, self.transform, self.tokenizer = parts(spec, device=device)
        source = Path(spec["sources"]["clic"]["root"])/"eval/eval_utils.py"
        self.load_text = A.original_function(str(source), "load_encode_text")
        self.load_image = A.original_function(str(source), "load_encode_img")
        # original_function's default namespace is designed for text cleaners.
        # Inject the exact imports used by these unchanged official helpers.
        import torch
        from PIL import Image
        for function in (self.load_text, self.load_image):
            function.__globals__.update(F=torch.nn.functional, Image=Image)

    def encode(self, items, *, image=False):
        import torch
        if len(items) != 1 or not isinstance(items[0], str):
            raise ValueError("Author paired inference is exactly one raw path or caption per call")
        with torch.inference_mode():
            if image:
                value = self.load_image(self.device, items[0], self.model, self.transform)
            else:
                value = self.load_text(self.device, self.model, items[0], self.tokenizer)
        return value.float().cpu().numpy()

    def signature(self):
        return {"model_class": type(self.model).__module__+"."+type(self.model).__name__,
            "tokenizer_class": type(self.tokenizer).__module__+"."+type(self.tokenizer).__name__,
            "transform": repr(self.transform), "dtype": "float32", "paired_batch": 1,
            "text_policy": "raw-caption-to-author-tokenizer", "normalization": "unchanged author load_encode helpers FP32 F.normalize",
            "baseline_spec_sha256": self.spec["baseline_model_spec_sha256"]}


# Private reuse of the frozen native producer does not alter that module or
# the SigLIP2 model. The local facade supplies this generic constructor only.
Siglip2 = PairedModel
