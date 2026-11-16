"""[codex] Real repaired imports and random CPU multimodal/DoRA forward only."""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from time import monotonic
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from codex.retrieval_matrix_qwen2_runtime_fix import contracts as C

def tiny_config():
    from transformers import Qwen2VLConfig
    config=Qwen2VLConfig(vocab_size=128,hidden_size=32,intermediate_size=64,num_hidden_layers=1,
        num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=128,
        rope_scaling={"type":"mrope","mrope_section":[1,1,2]},
        vision_config={"depth":1,"embed_dim":32,"hidden_size":32,"num_heads":4,
            "mlp_ratio":2,"patch_size":2,"temporal_patch_size":1,"spatial_merge_size":2},
        image_token_id=120,video_token_id=121,vision_start_token_id=122,vision_end_token_id=123,
        bos_token_id=1,eos_token_id=2,pad_token_id=0,use_cache=False)
    config._attn_implementation="eager"
    return config

def tiny_forward(spec):
    import torch
    from peft import LoraConfig,PeftModel,get_peft_model,get_peft_model_state_dict
    from codex.retrieval_matrix_qwen2 import adapters as A
    wrapper=A.core_wrapper(spec);cls=getattr(wrapper,spec["official"]["model_class"])
    torch.manual_seed(2026111665)
    model=cls(tiny_config()).to(dtype=torch.bfloat16).eval()
    text={"input_ids":torch.tensor([[0,1,5,2],[1,6,7,2]]),
        "attention_mask":torch.tensor([[0,1,1,1],[1,1,1,1]])}
    image={"input_ids":torch.tensor([[1,120,5,2]]),"attention_mask":torch.ones((1,4),dtype=torch.long),
        "pixel_values":torch.randn(4,12,dtype=torch.bfloat16),"image_grid_thw":torch.tensor([[1,2,2]])}
    with torch.inference_mode():
        text_value=model(**text);image_value=model(**image)
    if (text_value.shape!=(2,32) or image_value.shape!=(1,32)
            or not torch.isfinite(text_value).all() or not torch.isfinite(image_value).all()
            or text_value.dtype!=torch.bfloat16 or image_value.dtype!=torch.bfloat16):
        raise ValueError("Random original Qwen2 wrapper CPU forward is invalid")
    # FP32 isolates attach/load/merge correctness from BF16 rounding in this technical probe.
    base=cls(tiny_config()).float().eval();original=deepcopy(base)
    attached=get_peft_model(base,LoraConfig(r=2,lora_alpha=4,use_dora=True,target_modules=["q_proj","v_proj"]))
    with torch.no_grad():
        for name,value in attached.named_parameters():
            if "lora_A" in name:value.fill_(.02)
            if "lora_B" in name:value.fill_(.03)
    attached.eval()
    with torch.inference_mode():expected=attached(**text)
    with tempfile.TemporaryDirectory(prefix="qwen2-fix-dora-",dir=ROOT/"hf_home/tmp") as folder:
        attached.save_pretrained(folder)
        restored=PeftModel.from_pretrained(original,folder,is_trainable=False).eval()
        left,right=get_peft_model_state_dict(attached),get_peft_model_state_dict(restored)
        if not A.coverage(left,right)["all_keys_and_shapes_match"] or any(not torch.equal(left[k],right[k]) for k in left):
            raise ValueError("Random DoRA roundtrip changed an adapter tensor")
        with torch.inference_mode():restored_value=restored(**text)
        torch.testing.assert_close(restored_value,expected,atol=1e-6,rtol=1e-5)
        merged=restored.merge_and_unload(safe_merge=True).eval()
        with torch.inference_mode():merged_value=merged(**text)
        torch.testing.assert_close(merged_value,expected,atol=1e-6,rtol=1e-5)
        if any("lora_" in name for name,_ in merged.named_parameters()):raise ValueError("Random merge left live adapters")
    return {"class":cls.__module__+"."+cls.__name__,"random_model_parameters":sum(p.numel() for p in model.parameters()),
        "bf16_text_shape":list(text_value.shape),"bf16_image_shape":list(image_value.shape),
        "technical_attention":"eager on CPU only; production remains original FlashAttention2",
        "full_model_dora_roundtrip_and_merge":"PASS","dora_merge_max_abs":float((merged_value-expected).abs().max()),
        "real_checkpoint_weights_loaded":False}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--out",type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise ValueError("Never overwrite an independent CPU preflight receipt")
    began=monotonic();base=C.parent();manifest=C.read(C.HERE/"overlay.json");C.verify_overlay(manifest)
    modules=C.actual_modules()
    if modules!=manifest["modules"]:raise ValueError("Actual repaired runtime differs")
    import torch
    from transformers import BloomPreTrainedModel,Qwen2VLForConditionalGeneration
    from flash_attn.bert_padding import pad_input,unpad_input
    if torch.cuda.is_initialized():raise ValueError("CPU preflight cannot initialize CUDA")
    value=torch.randn(2,4,8);mask=torch.tensor([[1,1,0,0],[1,1,1,0]])
    unpadded,indices,*_=unpad_input(value,mask)
    padded=pad_input(unpadded,indices,2,4)
    torch.testing.assert_close(padded[mask.bool()],value[mask.bool()])
    technical=tiny_forward(base["models"]["gme2"])
    from codex.retrieval_matrix_qwen2 import preflight as P
    details=a.out.with_name(a.out.stem+"-original-processor-meta.json")
    P.run(C.ROOT/C.BASE_REL,details)
    if torch.cuda.is_initialized():raise ValueError("CPU preflight initialized CUDA")
    result={"owner":"[codex]","state":"CPU_TECHNICAL_PASS","research_metrics":False,
        "real_model_effects_read":False,"gpu_calls":False,"cuda_initialized":False,
        "original_embedding_plan":C.entry(C.ROOT/C.BASE_REL),"overlay":C.entry(C.HERE/"overlay.json"),
        "actual_modules":modules,"bloom_and_qwen2_actualimports":"PASS","flash_padding_einops_operation":"PASS",
        "random_forward":technical,"original_processors_and_full_v2_meta_coverage":C.entry(details),
        "sources":{str((C.HERE/name).relative_to(C.ROOT)):C.sha(C.HERE/name) for name in C.OWN},
        "seconds":monotonic()-began,"not_covered":"Real weights/GPU FlashAttention2 execution; no automatic retry authorized"}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True)+"\n")
    print({"owner":"[codex]","state":result["state"],"out":str(a.out),"sha256":C.sha(a.out),"cuda_initialized":False})

if __name__=="__main__":main()
