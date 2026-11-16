"""[codex] Official Google HF SigLIP2 port, explicit native FP32 protocol."""
from pathlib import Path

from .evidence import sha


def verify(spec):
    if (spec.get("family") != "CLIP" or spec.get("extra_family") != "siglip2"
            or spec.get("repo") != "google/siglip2-so400m-patch14-384"
            or spec.get("revision") != "e8e487298228002f3d8a82e0cd5c8ea9c567f57f"
            or spec.get("loader") != {"torch_dtype": "float32"}):
        raise ValueError("This native encoder only implements the pinned Google fixed-384 SigLIP2 port")
    for name, expected in spec["files"].items():
        if sha(Path(spec["snapshot"])/name) != expected:
            raise ValueError("Pinned Google checkpoint payload differs: " + name)


class Siglip2:
    def __init__(self, spec, device="cuda"):
        import torch
        from transformers import AutoProcessor, SiglipModel
        verify(spec)
        self.spec, self.device = spec, device
        self.model, info = SiglipModel.from_pretrained(spec["snapshot"], local_files_only=True,
            torch_dtype=torch.float32, output_loading_info=True)
        if any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise ValueError("Official Google HF checkpoint must load every parameter strictly")
        self.model = self.model.to(device).eval()
        self.processor = AutoProcessor.from_pretrained(spec["snapshot"], local_files_only=True)

    def encode(self, items, *, image=False):
        import torch
        from PIL import Image
        with torch.inference_mode():
            if image:
                pictures = []
                for path in items:
                    with Image.open(path) as picture:
                        pictures.append(picture.convert("RGB"))
                batch = self.processor(images=pictures, return_tensors="pt").to(self.device)
                value = self.model.get_image_features(**batch)
            else:
                if any(not isinstance(item, str) for item in items):
                    raise ValueError("Encoder only accepts caption strings, without labels/metadata")
                batch = self.processor(text=[item.lower() for item in items], padding="max_length",
                    max_length=64, truncation=True, return_tensors="pt").to(self.device)
                value = self.model.get_text_features(**batch)
            return torch.nn.functional.normalize(value.float(), dim=-1).cpu().numpy()

    def signature(self):
        return {"model_class": type(self.model).__module__ + "." + type(self.model).__name__,
            "processor_class": type(self.processor).__module__ + "." + type(self.processor).__name__,
            "torch_dtype": "float32", "text_lowercase": True, "max_length": 64,
            "padding": "max_length", "truncation": True, "normalization_dtype": "float32",
            "image_preprocessing": self.processor.image_processor.to_dict(),
            "disclosed_adaptation": self.spec["native_protocol"]["disclosed_adaptation"]}
