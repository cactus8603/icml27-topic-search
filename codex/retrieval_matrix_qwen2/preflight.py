"""[codex] CPU-only official processor and V2 DoRA key-coverage evidence."""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import sys
from time import monotonic

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from codex.retrieval_matrix import runner as R
from codex.retrieval_matrix_qwen2 import adapters as A


def run(plan, output):
    began = monotonic()
    import torch
    import flash_attn
    import transformers
    import peft
    from PIL import Image
    data = R.read(plan)
    report = {"owner": "[codex]", "technical_only": True, "no_gpu_calls": True, "no_base_weights_loaded": True,
              "plan_sha256": R.sha(plan), "models": {},
              "environment": {"torch": torch.__version__, "transformers": transformers.__version__,
                              "peft": peft.__version__, "flash_attn": flash_attn.__version__,
                              "qwen-vl-utils": version("qwen-vl-utils"), "cuda_runtime": torch.version.cuda,
                              "python": sys.version, "python_executable": sys.executable}}
    for key, spec in data["models"].items():
        wrapper = A.core_wrapper(spec)
        proc = getattr(wrapper, spec["official"]["processor_class"]).from_pretrained(
            spec["snapshot"], local_files_only=True, **spec["processor"])
        text = proc(texts=["", "A synthetic color image."], images=[None, None], videos=[None, None],
                    instruction=["Find me an everyday image that matches the given caption: "]*2,
                    is_query=False, max_length=1024, truncation=False)
        image = proc(texts=[None], images=[Image.new("RGB", (224, 224), (20, 128, 64))], videos=[None],
                     instruction=["Represent the given image."], is_query=False, max_length=1024, truncation=False)
        row = {"processor": type(proc).__name__, "text_input_ids": text["input_ids"].tolist(),
               "text_attention_mask": text["attention_mask"].tolist(), "decoded": proc.tokenizer.batch_decode(text["input_ids"]),
               "image_shapes": {name: list(value.shape) for name, value in image.items()},
               "padding_side": proc.tokenizer.padding_side, "eos_id": proc.eos_id}
        if spec["variant"] == "vlm2vec-v2-dora":
            row["core_evaluation_adapter_coverage"] = A.meta_coverage(spec)
            row["upstream_native_adapter_coverage"] = A.meta_coverage(spec, official_vlm2vec=True)
        report["models"][key] = row
    report.update(state="PASS", seconds=monotonic()-began)
    R.atomic_json(output, report)
    print(json.dumps({"owner": "[codex]", "state": "PASS", "out": str(output), "models": list(report["models"])}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.plan, args.out)
