"""獨立影像／文字向量，以及逐對分數。模型與輸入格式由補量設定固定。"""
import json

import numpy as np


def qwen_inputs(processor, messages, config):
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True,
        return_tensors="pt",
        processor_kwargs={
            "text_kwargs": {"padding": True},
            "images_kwargs": {"size": {"shortest_edge": config["image_min_pixels"],
                                      "longest_edge": config["image_max_pixels"]}},
        },
    )
    if inputs["input_ids"].shape[1] > config["max_tokens"]:
        raise ValueError("補量輸入超過固定 token 上限")
    return inputs


def conversation(text, image=None, system=None):
    content = [] if image is None else [{"type": "image", "image": image}]
    if text is not None:
        content.append({"type": "text", "text": text})
    messages = [{"role": "system", "content": [{"type": "text", "text": system}]}] if system else []
    return messages + [{"role": "user", "content": content}]


def reranker_conversation(text, image, config):
    return [
        {"role": "system", "content": [{"type": "text", "text": config["system"]}]},
        {"role": "user", "content": [
            {"type": "text", "text": "<Instruct>: " + config["instruct"]},
            {"type": "text", "text": "<Query>:"},
            {"type": "text", "text": text},
            {"type": "text", "text": "\n<Document>:"},
            {"type": "image", "image": image},
        ]},
    ]


class Encoder:
    def __init__(self, name, snapshot, config, device):
        import torch
        from transformers import AutoProcessor, Qwen3VLModel, SiglipModel

        self.name = name
        self.device = device
        self.config = config.get(name)
        self.processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
        if name == "core_embed":
            self.processor.tokenizer.padding_side = "left"
        cls = Qwen3VLModel if name == "core_embed" else SiglipModel
        dtype = torch.float32 if device == "cpu" else torch.bfloat16
        self.model, loading = cls.from_pretrained(snapshot, local_files_only=True, dtype=dtype,
                                                  attn_implementation="sdpa", output_loading_info=True)
        if loading["missing_keys"] or loading["mismatched_keys"]:
            raise ValueError(f"{name} 權重不完整：{loading}")
        self.model.to(device).eval()

    def encode(self, items, *, images=False):
        import torch
        from torch.nn.functional import normalize

        with torch.inference_mode():
            if self.name == "core_embed":
                messages = ([conversation(None, image, self.config["image_system"]) for image in items] if images else
                            [conversation(caption, system=self.config["text_system"]) for caption in items])
                inputs = qwen_inputs(self.processor, messages, self.config).to(self.device)
                output = self.model(**inputs, use_cache=False).last_hidden_state[:, -1]
            elif images:
                inputs = self.processor(images=items, return_tensors="pt").to(self.device)
                output = self.model.get_image_features(**inputs).pooler_output
            else:
                inputs = self.processor(text=[caption.lower() for caption in items], padding="max_length",
                                        max_length=64, truncation=True, return_tensors="pt").to(self.device)
                output = self.model.get_text_features(**inputs).pooler_output
            return normalize(output.float(), dim=-1).cpu().numpy()


class ClipEncoder:
    def __init__(self, name, snapshot, config, device):
        import open_clip
        import torch
        from open_clip.transform import PreprocessCfg, image_transform_v2

        self.device = device
        if name == "cs_clip":
            from common import load_model
            # The stage-1 E2 encoder is used only for the optional descriptive comparison.
            self.model, self.preprocess, self.tokenizer = load_model(name, device, cs_clip_path=snapshot / "open_clip_pytorch_model.bin")
        else:
            options = json.loads((snapshot / "open_clip_config.json").read_text())
            architecture = options["model_cfg"]
            weights = snapshot / "open_clip_model.safetensors"
            self.model = open_clip.CLIP(**architecture).to(device).eval()
            open_clip.load_checkpoint(self.model, str(weights), weights_only=True)
            preprocess_config = PreprocessCfg(**options["preprocess_cfg"])
            self.preprocess = image_transform_v2(preprocess_config, is_train=False)
            self.tokenizer = open_clip.get_tokenizer("ViT-L-14")
        self.model = self.model.to(dtype=torch.float32)

    def encode(self, items, *, images=False):
        import torch
        from torch.nn.functional import normalize

        with torch.inference_mode():
            if images:
                vectors = self.model.encode_image(torch.stack([self.preprocess(image) for image in items]).to(self.device))
            else:
                vectors = self.model.encode_text(self.tokenizer(items).to(self.device))
        return normalize(vectors.float(), dim=-1).cpu().numpy()


class CrossScorer:
    def __init__(self, name, snapshot, config, device):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.name = name
        self.config = config[name]
        self.device = device
        self.processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
        self.processor.tokenizer.padding_side = "left"
        dtype = torch.float32 if device == "cpu" else torch.bfloat16
        self.model, loading = Qwen3VLForConditionalGeneration.from_pretrained(
            snapshot, dtype=dtype, local_files_only=True, attn_implementation="sdpa", output_loading_info=True,
        )
        if loading["missing_keys"] or loading["mismatched_keys"]:
            raise ValueError(f"{name} 權重不完整：{loading}")
        self.model.to(device).eval()
        self.yes = self.token_id(self.config.get("yes", "yes"))
        if name == "core_reranker":
            self.no = self.token_id("no")

    def token_id(self, text):
        tokenizer = self.processor.tokenizer
        ids = tokenizer.encode(text, add_special_tokens=False)
        if ids != [tokenizer.get_vocab()[text]]:
            raise ValueError(f"{text!r} 不是當前 tokenizer 的單 token")
        return ids[0]

    def score(self, pairs):
        import torch

        if self.name == "core_reranker":
            messages = [reranker_conversation(caption, image, self.config) for caption, image in pairs]
        else:
            prompt = self.config["prompt"]
            messages = [conversation(prompt.format(caption=caption), image) for caption, image in pairs]
        inputs = qwen_inputs(self.processor, messages, self.config).to(self.device)
        with torch.inference_mode():
            logits = self.model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1, :].float()
            if self.name == "core_reranker":
                # 官方的二元 sigmoid 與此 log-odds 單調等價，log-odds 避免飽和平手。
                scores = logits[:, self.yes] - logits[:, self.no]
            else:
                scores = logits.log_softmax(dim=-1)[:, self.yes]
        return scores.cpu().numpy()


def unit_scores(encoder, items, batch_size, *, images=False):
    """傳入影像 opener 時逐批解碼，避免全庫的原圖同時留在 RAM。"""
    vectors = []
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        vectors.append(encoder.encode([opener() for opener in batch] if images else batch, images=images))
    return np.concatenate(vectors)
