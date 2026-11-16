"""[codex] Strict payload/meta/import and tiny random-weight CPU forwards only."""
import argparse
import importlib
from importlib.metadata import version
from pathlib import Path
import os
import sys
import tempfile
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from codex.retrieval_matrix_triplet_native_p0 import contracts as C


def runtime_audit(*,hash_files=True):
    import torch
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or torch.cuda.is_initialized():
        raise ValueError('CPU-only preflight requires CUDA hidden before process start')
    dependency=C.read(C.checked(C.dependency_reference()));actual={}
    for name,wanted in dependency['modules'].items():
        module=importlib.import_module(name)
        distribution={'open_clip':'open-clip-torch','qwen_vl_utils':'qwen-vl-utils'}.get(name,name)
        actual[name]={'path':str(Path(module.__file__).resolve()),'actual_version':getattr(module,'__version__',None),
            'distribution_version':version(distribution)}
        if actual[name]!=wanted:raise ValueError('Actual prepared module differs: '+name)
    # Force the lazy CUDA-availability-dependent Transformers dependency path.
    # Import this module directly without initializing CUDA or allocating a GPU.
    padding=importlib.import_module('flash_attn.bert_padding')
    from codex.retrieval_matrix_qwen2_runtime_fix import contracts as overlay
    manifest=C.read(C.frozen('codex/retrieval_matrix_qwen2_runtime_fix/overlay.json',C.OVERLAY_REF))
    overlay.verify_overlay(manifest)
    if hash_files:
        for relative,wanted in dependency['files'].items():
            if C.sha(ROOT/relative)!=wanted:raise ValueError('Frozen dependency payload differs: '+relative)
    if torch.cuda.is_initialized():raise ValueError('CPU imports unexpectedly initialized CUDA')
    return {'modules':actual,'dependency_files':len(dependency['files']),'all_files_hashed':bool(hash_files),
        'direct_flash_attn_bert_padding':str(Path(padding.__file__).resolve()),'cuda_initialized':False}


def tiny_forwards(spec):
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from codex.retrieval_matrix_extra import adapters as A
    with tempfile.TemporaryDirectory(dir=ROOT/'hf_home/tmp',prefix='triplet-tiny-cpu-') as temporary:
        if spec['extra_family']=='triplet-separate':
            from transformers import CLIPConfig,CLIPTextConfig,CLIPVisionConfig,CLIPTokenizerFast,CLIPImageProcessor
            source=spec['sources']['checkpoint_code'];module=A.M.official_module(source['root'],'utils.py',source['files'])
            text=CLIPTextConfig(vocab_size=49408,hidden_size=32,intermediate_size=64,num_hidden_layers=1,
                num_attention_heads=4,max_position_embeddings=77,projection_dim=512)
            vision=CLIPVisionConfig(hidden_size=32,intermediate_size=64,num_hidden_layers=1,num_attention_heads=4,
                image_size=224,patch_size=32,projection_dim=512)
            CLIPConfig.from_text_vision_configs(text,vision,projection_dim=512).save_pretrained(temporary)
            model_text=module.CLIPTextEncoderOnly(module.CLIPTextEncoderOnlyConfig(model_name=temporary,pretrained=False)).eval()
            model_image=module.CLIPVisionEncoderOnly(module.CLIPVisionEncoderOnlyConfig(model_name=temporary,pretrained=False)).eval()
            support=spec['support']['clip']['snapshot']
            tokenizer=CLIPTokenizerFast.from_pretrained(support,local_files_only=True)
            processor=CLIPImageProcessor.from_pretrained(support,local_files_only=True)
            tokens=tokenizer(['A synthetic color image.',''],padding=True,truncation=True,max_length=77,return_tensors='pt')
            pixels=processor(images=[Image.new('RGB',(96,64),(64,128,192))],return_tensors='pt')
            with torch.inference_mode():
                image=F.normalize(model_image(pixels),dim=-1);text=F.normalize(model_text(**tokens).text_embeds,dim=-1)
        else:
            import open_clip
            model=open_clip.model.CLIP(embed_dim=512,vision_cfg={'image_size':224,'layers':1,'width':32,'patch_size':32,'head_width':8},
                text_cfg={'context_length':77,'vocab_size':49408,'width':32,'heads':4,'layers':1}).eval()
            tokens=open_clip.get_tokenizer('ViT-B-32')(['A synthetic color image.',''])
            with torch.inference_mode():
                image=F.normalize(model.encode_image(torch.zeros(1,3,224,224)),dim=-1)
                text=F.normalize(model.encode_text(tokens),dim=-1)
        if (image.shape!=(1,512) or text.shape!=(2,512) or image.dtype!=torch.float32 or text.dtype!=torch.float32
                or not torch.isfinite(image).all() or not torch.isfinite(text).all() or torch.cuda.is_initialized()):
            raise ValueError('Tiny source-class CPU synthetic forward failed')
        return {'image_shape':list(image.shape),'text_shape':list(text.shape),'dtype':str(image.dtype),
            'random_tiny_weights_only':True,'actual_pretrained_forward':False,'cuda_initialized':False}


def audit():
    from codex.retrieval_matrix_extra import preflight as extra
    result={'schema':1,'state':'CPU_TECHNICAL_PASS','native_qualification':False,'technical_only':True,
        'research_metrics':False,'runtime':runtime_audit(),'models':{},
        'source_files':{name:C.sha(C.HERE/name) for name in C.OWN},'registry':C.entry(ROOT/'codex/retrieval_matrix_extra/registry.json')}
    for key in C.VARIANTS:
        spec=C.model(key)
        result['models'][key]={'meta_import_payload_processor_audit':extra.check_one(spec),
            'tiny_random_synthetic_forward':tiny_forwards(spec),'source_gaps':C.gaps(key)}
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();C.write(args.out,audit());print('CPU technical audit complete; native protocol remains blocked')


if __name__=='__main__':main()
