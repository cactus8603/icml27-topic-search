"""[codex] Actual repaired imports and random original-class CPU forward only."""
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic
import argparse
import json
import os
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from codex.retrieval_matrix_native_runtime_fix import contracts as C


def tiny_forward(spec):
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoProcessor,SiglipConfig,SiglipModel
    from codex.retrieval_matrix_native_p0 import models
    processor=AutoProcessor.from_pretrained(spec['snapshot'],local_files_only=True)
    config=SiglipConfig(text_config={'vocab_size':len(processor.tokenizer),'hidden_size':16,'intermediate_size':32,
        'num_hidden_layers':1,'num_attention_heads':2,'max_position_embeddings':64,'projection_size':16},
        vision_config={'hidden_size':16,'intermediate_size':32,'num_hidden_layers':1,'num_attention_heads':2,
            'image_size':384,'patch_size':14,'num_channels':3},projection_dim=16)
    config._attn_implementation='eager'
    torch.manual_seed(spec.get('seed',2026111665))
    model=SiglipModel(config).float().eval()
    # No from_pretrained: exercise the exact original wrapper methods on random weights.
    wrapper=models.Siglip2.__new__(models.Siglip2)
    wrapper.spec,wrapper.device,wrapper.model,wrapper.processor=spec,'cpu',model,processor
    with TemporaryDirectory(prefix='native-fix-tiny-',dir=ROOT/'hf_home/tmp') as folder:
        image=Path(folder)/'synthetic.png';Image.new('RGB',(96,64),(64,128,192)).save(image)
        text=wrapper.encode(['A synthetic color image.',''])
        vision=wrapper.encode([str(image)],image=True)
        encoded=processor(text=['a synthetic color image.'],images=[Image.open(image).convert('RGB')],
            padding='max_length',max_length=64,truncation=True,return_tensors='pt')
        with torch.inference_mode():joint=model(**encoded)
    if (text.shape!=(2,16) or vision.shape!=(1,16) or text.dtype!=np.float32 or vision.dtype!=np.float32
            or not np.isfinite(text).all() or not np.isfinite(vision).all()):
        raise ValueError('Random original native text/image CPU forward failed')
    np.testing.assert_allclose(np.linalg.norm(text,axis=1),1,atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(vision,axis=1),1,atol=1e-6)
    torch.testing.assert_close(torch.from_numpy(text[:1]),joint.text_embeds,atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(torch.from_numpy(vision),joint.image_embeds,atol=1e-6,rtol=1e-5)
    if any(p.dtype!=torch.float32 or p.device.type!='cpu' for p in model.parameters()):
        raise ValueError('Tiny native probe must remain FP32 CPU')
    return {'wrapper_class':type(wrapper).__module__+'.'+type(wrapper).__name__,
        'model_class':type(model).__module__+'.'+type(model).__name__,
        'processor_class':type(processor).__module__+'.'+type(processor).__name__,
        'parameters':sum(p.numel() for p in model.parameters()),'text_shape':list(text.shape),'image_shape':list(vision.shape),
        'pixel_shape':list(encoded['pixel_values'].shape),'token_shape':list(encoded['input_ids'].shape),
        'dtype':'float32','exact_original_encode_and_joint_forward_agree':'PASS',
        'attention':'CPU eager technical probe only; production original configuration unchanged',
        'real_checkpoint_weights_loaded':False}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise ValueError('Never overwrite a CPU technical receipt')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU probe requires CUDA hidden')
    began=monotonic();plan=C.read(a.plan);dependencies=C.validate(plan,imports=True)
    from codex.retrieval_matrix_native_runtime_fix import runner
    from transformers import BloomPreTrainedModel,SiglipModel
    from flash_attn.bert_padding import unpad_input,pad_input
    import torch
    if torch.cuda.is_initialized():raise ValueError('CPU preflight cannot initialize CUDA')
    value=torch.randn(2,4,8);mask=torch.tensor([[1,1,0,0],[1,1,1,0]])
    unpadded,indices,*_=unpad_input(value,mask);padded=pad_input(unpadded,indices,2,4)
    torch.testing.assert_close(padded[mask.bool()],value[mask.bool()])
    actual=runner.runtime(plan,gpu=False)
    technical=tiny_forward(plan['model_spec'])
    if torch.cuda.is_initialized():raise ValueError('CPU probe initialized CUDA')
    receipt={'owner':'[codex]','state':'CPU_TECHNICAL_PASS','plan':C.entry(a.plan),
        'source_files':plan['source_files'],'dependencies':plan['dependencies'],'runtime':actual,
        'dependency_files_checked':len(dependencies['files']),'original_dependency_files':2766,
        'added_overlay_files':32,'actual_modules':C.actual_modules(dependencies),
        'actual_Bloom_Siglip_imports':'PASS','flash_padding_einops_operation':'PASS','random_forward':technical,
        'real_model_weights_loaded':False,'real_model_effects_read':False,'gpu_calls':False,'cuda_initialized':False,
        'seconds':monotonic()-began,'limits':'No real-weight/GPU forward or benchmark qualification; no retry authorized.'}
    C.write(a.out,receipt);print(json.dumps({'state':receipt['state'],'out':str(a.out),'sha256':C.sha(a.out),'cuda_initialized':False}))


if __name__=='__main__':main()
