"""[codex] CPU-only artifact, native processor, and exact model-key qualification prep.

Uses only synthetic colored images and fixed synthetic text. No benchmark split,
labels, research embeddings, GPU device or performance result is accessed.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import struct
import sys
from time import monotonic

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from codex.retrieval_matrix_extra import adapters as A


def header(path):
    if str(path).endswith(".safetensors"):
        with Path(path).open("rb") as stream:
            length = struct.unpack("<Q",stream.read(8))[0]
            if length > 64*1024*1024:
                raise ValueError("Implausible safetensors header")
            value=json.loads(stream.read(length))
        return {key:tuple(row["shape"]) for key,row in value.items() if key != "__metadata__"}
    import torch
    value=torch.load(path,map_location="cpu",weights_only=True,mmap=True)
    value=value.get("state_dict",value)
    return {key:tuple(row.shape) for key,row in value.items()}


def key_coverage(model, actual):
    expected={key:tuple(value.shape) for key,value in model.state_dict().items()}
    # HF safe serialization omits the duplicate lm_head when tied to embeddings.
    tied = {"lm_head.weight"} if getattr(getattr(model,"config",None),"tie_word_embeddings",False) else set()
    absent=sorted(expected.keys()-actual.keys())
    missing=sorted(set(absent)-tied)
    unexpected=sorted(actual.keys()-expected.keys())
    mismatched=sorted(key for key in expected.keys()&actual.keys() if expected[key] != actual[key])
    result={"expected_keys":len(expected),"checkpoint_keys":len(actual),"missing":missing,
        "omitted_tied_aliases":sorted(set(absent)&tied),"unexpected":unexpected,"shape_mismatch":mismatched,
        "parameters":sum(value.numel() for value in model.parameters()),"device":"meta","base_weights_allocated":False,
        "class":type(model).__module__+"."+type(model).__name__,"all_keys_and_shapes_match":not(missing or unexpected or mismatched)}
    if not result["all_keys_and_shapes_match"]:
        raise ValueError(f"Official model/checkpoint key coverage mismatch: {result}")
    return result


def check_one(spec):
    import torch
    from PIL import Image
    from transformers import AutoProcessor, SiglipConfig, SiglipModel, Qwen2VLConfig
    A.verify(spec)
    family=spec["extra_family"]
    image=Image.new("RGB",(96,64),(64,128,192))
    texts=["A synthetic color image.",""]
    models=[]
    processor_receipt={}
    actual={}
    for name in spec["files"]:
        if name.endswith((".safetensors",".bin")):
            part=header(Path(spec["snapshot"])/name)
            if family == "triplet-separate":
                prefix="vision." if name.startswith("vision-") else "text."
                part={prefix+key:value for key,value in part.items()}
            if actual.keys()&part.keys():
                raise ValueError("Duplicate stored tensor names across declared shards")
            actual.update(part)
    if family in {"clic","clic-clips","author-openclip"}:
        model,processor,tokenizer=A.openclip_parts(spec,weights=False)
        models=[("",model)]
        if spec["native_protocol"].get("text_cleaner")=="clic-pre-caption-30":
            texts=[A.clic_pre_caption(text) for text in texts]
        tokens=tokenizer(texts)
        processor_receipt={"image_tensor_shape":list(processor(image).shape),"token_shape":list(tokens.shape),
                           "empty_text_retained":True,"token_ids":tokens.tolist()}
    elif family == "siglip2":
        processor=AutoProcessor.from_pretrained(spec["snapshot"],local_files_only=True)
        batch=processor(text=[value.lower() for value in texts],padding="max_length",max_length=64,truncation=True,return_tensors="pt")
        pixels=processor(images=[image],return_tensors="pt")
        processor_receipt={"token_shape":list(batch["input_ids"].shape),"token_ids":batch["input_ids"].tolist(),
            "image_tensor_shape":list(pixels["pixel_values"].shape),"lowercase_explicit":True,"padding":"max_length"}
        with torch.device("meta"):
            model=SiglipModel(SiglipConfig.from_pretrained(spec["snapshot"],local_files_only=True))
        models=[("",model)]
    elif family == "triplet-separate":
        parts=A.triplet_parts(spec,weights=False)
        models=[("vision.",parts[0]),("text.",parts[1])]
        batch=parts[2](texts,padding=True,max_length=77,truncation=True,return_tensors="pt")
        processor_receipt={"token_shape":list(batch["input_ids"].shape),"token_ids":batch["input_ids"].tolist(),
            "image_tensor_shape":list(parts[3](images=[image],return_tensors="pt")["pixel_values"].shape)}
    elif family == "unime-v2-native":
        _,processor,process,resize,_=A.unime_parts(spec,weights=False)
        # The author's raw empty text produces zero tokens; it is not an official
        # caption in the native P0 datasets. Record rather than invent a template.
        batch=process({"text":texts[:1],"image":[None]},processor=processor,max_length=4096)
        batch_image=process({"text":["<|image_pad|> Find an image caption describing the given image."],
                             "image":[resize(image,"high")]},processor=processor,max_length=4096)
        processor_receipt={"token_shape":list(batch["input_ids"].shape),"token_ids":batch["input_ids"].tolist(),
            "image_token_shape":list(batch_image["input_ids"].shape),"image_tensor_shape":list(batch_image["pixel_values"].shape),
            "max_length":4096,"image_resize":"official process_image high 1344x1344"}
        from src.vlm_backbone.qwen2_vl import Qwen2VLForConditionalGeneration
        config=Qwen2VLConfig.from_pretrained(spec["snapshot"],local_files_only=True)
        config._attn_implementation="eager"
        with torch.device("meta"):
            model=Qwen2VLForConditionalGeneration(config)
        models=[("",model)]
    elif family == "gme-core":
        official=spec["core_official"]
        wrapper=A.M.official_module(official["root"],official["wrapper"],official["files"])
        processor=getattr(wrapper,official["processor_class"]).from_pretrained(spec["snapshot"],local_files_only=True,**spec["processor"])
        batch=processor(texts=texts,images=[None]*2,videos=[None]*2,instruction=["ignored"]*2,is_query=False,max_length=1024,truncation=False)
        batch_image=processor(texts=[None],images=[image],videos=[None],instruction=["ignored"],is_query=False,max_length=1024,truncation=False)
        processor_receipt={"token_shape":list(batch["input_ids"].shape),"token_ids":batch["input_ids"].tolist(),
            "image_token_shape":list(batch_image["input_ids"].shape),"empty_text_retained":True}
        config=Qwen2VLConfig.from_pretrained(spec["snapshot"],local_files_only=True)
        config._attn_implementation="eager"
        with torch.device("meta"):
            model=getattr(wrapper,official["model_class"])(config)
        models=[("",model)]
    coverage=[]
    for prefix,model in models:
        tensors={key.removeprefix(prefix):shape for key,shape in actual.items() if key.startswith(prefix)}
        coverage.append(key_coverage(model,tensors))
    del models
    gc.collect()
    return {"state":"PASS","processor":processor_receipt,"coverage":coverage,"identity":spec,
            "technical_only":True,"research_metrics":False,"native_gpu_attention_unchanged":True,
            "meta_attention":"eager for shape construction only; no weights allocated and no inference"}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry",type=Path,default=ROOT/"codex/retrieval_matrix_extra/registry.json")
    parser.add_argument("--models",nargs="+")
    parser.add_argument("--out",type=Path,default=ROOT/"dl/retrieval-matrix-extra/cpu-preflight.json")
    args=parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU preflight requires CUDA_VISIBLE_DEVICES= before starting Python")
    registry=json.loads(args.registry.read_text())
    result={"schema":1,"state":"RUNNING","technical_only":True,"research_metrics":False,"gpu_used":False,
        "registry_sha256":A.M.file_sha(args.registry),"models":{},"environment":{}}
    for name in ("torch","transformers","tokenizers","numpy","open_clip","peft"):
        module=__import__(name)
        result["environment"][name]={"version":getattr(module,"__version__",None),"path":str(Path(module.__file__).resolve())}
    result["environment"]["open-clip-torch-distribution"] = version("open-clip-torch")
    started=monotonic()
    try:
        for key in args.models or list(registry["models"]):
            result["models"][key]=check_one(registry["models"][key])
            args.out.write_text(json.dumps(result,indent=2)+"\n")
            print(key,"CPU PASS",flush=True)
        result["state"]="PASS"
    except BaseException as exc:
        result["state"]="FAILED"
        result["error"]={"type":type(exc).__name__,"message":str(exc)}
        raise
    finally:
        result["elapsed_seconds"]=monotonic()-started
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(result,indent=2)+"\n")


if __name__=="__main__":
    main()
