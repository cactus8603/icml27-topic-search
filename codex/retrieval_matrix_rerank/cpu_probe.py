"""[codex] Probe official APIs/templates without loading any model weights."""
from pathlib import Path
from importlib.metadata import version
import argparse
import json
import os

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "dl/retrieval-matrix-rerank/cpu-probe.json")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU probe requires CUDA_VISIBLE_DEVICES='' before Python starts")
    os.environ["IMAGE_MAX_TOKEN_NUM"] = "1200"
    import torch
    from PIL import Image
    from swift.infer_engine import TransformersEngine, InferRequest, RequestConfig
    from swift.model import get_processor
    from swift.template import get_template
    from swift.utils import get_generative_reranker_logits
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from transformers import AutoModelForVision2Seq
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
    snapshot = ROOT / "hf_home/hub/models--Alibaba-NLP--core-reranker-2b/snapshots/64e1a36cd6ecace459a4c7cd8f7d0ef5d6fee30b"
    processor = get_processor(str(snapshot), use_hf=True, download_model=False, task_type="generative_reranker")
    template = get_template(processor, max_length=1500)
    template.set_mode("transformers")
    synthetic = ROOT / "dl/retrieval-matrix-rerank/sources/synthetic.png"
    Image.new("RGB", (64, 64), (128, 64, 32)).save(synthetic)
    request = InferRequest(messages=[{"role": "system", "content": "Determine if the following image is relevant to the query. "},
        {"role": "user", "content": "A synthetic color image."}, {"role": "assistant", "content": "<image>"}], images=[str(synthetic)])
    encoded = template.encode(request)
    large = ROOT / "dl/retrieval-matrix-rerank/sources/synthetic-max-image.png"
    Image.new("RGB", (1280, 960), (128, 64, 32)).save(large)
    full_request = InferRequest(messages=request.messages, images=[str(large)])
    maximum = template.encode(full_request)
    assert int(maximum["image_grid_thw"].prod()) // 4 == 1200
    assert template.template_meta.template_type == "qwen3_vl_reranker"
    generator = torch.Generator().manual_seed(23)
    weights = torch.randn(len(processor.tokenizer), 4, generator=generator)
    hidden = torch.randn(2, 3, 4, generator=generator)
    native = get_generative_reranker_logits(weights, processor.tokenizer, hidden)
    yes, no = [processor.tokenizer.convert_tokens_to_ids(word) for word in ("yes", "no")]
    expected = torch.nn.functional.linear(hidden, weights[[yes, no]])
    assert torch.equal(native, expected[..., 0:1] - expected[..., 1:2])
    jina = ROOT / "hf_home/hub/models--jinaai--jina-reranker-m0/snapshots/94bfe0aeb2d4dd7978362699cddd5893d4e0adc8"
    jina_cls = get_class_from_dynamic_module("modeling.JinaVLForRanking", str(jina), local_files_only=True)
    result = {"owner": "[codex]", "state": "PASS", "technical_only": True, "weights_loaded": False,
              "gpu_inference": False, "packages": {name: version(name) for name in ["torch", "transformers", "ms-swift", "qwen-vl-utils", "flash-attn", "peft"]},
              "core_auto_model_type": processor.model_info.model_type,
              "core_auto_template": template.template_meta.template_type,
              "core_args_required": True, "core_decoded_synthetic": processor.tokenizer.decode(encoded["input_ids"]),
              "core_input_tokens": len(encoded["input_ids"]), "core_image_grid": encoded["image_grid_thw"].tolist(),
              "full_size_synthetic_input_tokens": len(maximum["input_ids"]),
              "full_size_synthetic_image_grid": maximum["image_grid_thw"].tolist(),
              "swift_binary_projection_check": "PASS", "jina_remote_class": jina_cls.__name__,
              "mcmr_auto_model_class": AutoModelForVision2Seq._model_mapping[Qwen3VLConfig].__name__}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
