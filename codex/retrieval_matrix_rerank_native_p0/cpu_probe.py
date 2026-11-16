"""[codex] Official processors/classes only; no real weights/GPU/model effects."""
from pathlib import Path
import argparse,json,os,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
os.environ['HF_HOME']=str(ROOT/'hf_home');os.environ['HF_MODULES_CACHE']=str(ROOT/'hf_home/modules')
from codex.retrieval_matrix_rerank import backends as B
from codex.retrieval_matrix_rerank_native_p0 import contracts as C


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError("CPU-only probe requires hidden CUDA")
    os.environ['IMAGE_MAX_TOKEN_NUM']='1200'
    import torch
    from PIL import Image
    from transformers import AutoProcessor
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from swift.model import get_processor
    from swift.template import get_template
    from swift.infer_engine import InferRequest
    source=ROOT/'codex/retrieval_matrix_rerank_native_p0/backend_specs.draft.json'
    specs=C.R.read(source)['models'];out=a.out.parent;out.mkdir(parents=True,exist_ok=True)
    image=out/'synthetic-1280x960.png';Image.new('RGB',(1280,960),(64,128,32)).save(image)
    results={}
    for key in ('qwen2','qwen8'):
        proc=get_processor(specs[key]['snapshot'],use_hf=True,download_model=False,task_type='generative_reranker')
        template=get_template(proc,max_length=1500);template.set_mode('transformers')
        request=InferRequest(messages=[{'role':'system','content':'Determine if the following image is relevant to the query. '},
            {'role':'user','content':'A synthetic color image.'},{'role':'assistant','content':'<image>'}],images=[str(image)])
        encoded=template.encode(request)
        assert template.template_meta.template_type=='qwen3_vl_reranker'
        assert int(encoded['image_grid_thw'].prod())//4==1200
        results[key]={'state':'PASS','auto_model_type':proc.model_info.model_type,'template':template.template_meta.template_type,
            'input_tokens':len(encoded['input_ids']),'image_grid':encoded['image_grid_thw'].tolist(),
            'decoded_template':proc.tokenizer.decode(encoded['input_ids'])}
    for key in ('qwen_native2','qwen_native8'):
        module=B.module(specs[key]['source']);engine=module.Qwen3VLReranker.__new__(module.Qwen3VLReranker)
        engine.processor=AutoProcessor.from_pretrained(specs[key]['snapshot'],local_files_only=True,trust_remote_code=True,padding_side='left')
        for name,value in {'max_length':10240,'min_pixels':4096,'max_pixels':1843200,'total_pixels':module.MAX_TOTAL_PIXELS,
            'fps':1,'max_frames':64,'default_instruction':'Given a search query, retrieve relevant candidates that answer the query.'}.items():setattr(engine,name,value)
        messages=engine.format_mm_instruction(query_text='A synthetic color image.',doc_image=str(image),
            instruction='Determine if the following image is relevant to the query. ')
        encoded=engine.tokenize([messages]);assert len(encoded['image_grid_thw'])==1
        results[key]={'state':'PASS','class':module.Qwen3VLReranker.__name__,'messages':messages,
            'input_tokens':int(encoded['input_ids'].shape[1]),'image_grid':encoded['image_grid_thw'].tolist(),
            'decoded_template':engine.processor.tokenizer.decode(encoded['input_ids'][0])}
    jina=specs['jina_m0'];cls=get_class_from_dynamic_module('modeling.JinaVLForRanking',jina['snapshot'],local_files_only=True)
    module=sys.modules[cls.__module__]
    prompt=module.formatting_prompts_func('A synthetic color image.',str(image),doc_type='image')
    assert '**Document**:' in prompt and '**Query**:' in prompt
    jina_processor=AutoProcessor.from_pretrained(jina['snapshot'],max_pixels=602112,min_pixels=3136,trust_remote_code=True,local_files_only=True)
    jina_input=jina_processor(text=[prompt],images=[Image.open(image).convert('RGB')],return_tensors='pt',padding=True,truncation=True,max_length=2047)
    assert len(jina_input['image_grid_thw'])==1
    score_input=torch.cat([jina_input['input_ids'],torch.full((1,1),100,dtype=jina_input['input_ids'].dtype)],dim=1)
    results['jina_m0']={'state':'PASS','class':cls.__name__,'prompt':prompt,'logit_bias':module.LOGIT_BIAS,
        'score_token_id':100,'max_length':2048,'image_max_pixels':602112,'image_min_pixels':3136,
        'processor_class':jina_processor.__class__.__name__,'image_processor_class':jina_processor.image_processor.__class__.__name__,
        'image_processor_config':jina_processor.image_processor.to_dict(),
        'input_tokens_including_reward':int(score_input.shape[1]),'reward_last_token':int(score_input[0,-1]),
        'image_grid':jina_input['image_grid_thw'].tolist()}
    module=B.module(specs['qwen_instruct8']['source']);proc=AutoProcessor.from_pretrained(specs['qwen_instruct8']['snapshot'],local_files_only=True,use_fast=False)
    true=proc.tokenizer('True',add_special_tokens=False).input_ids;false=proc.tokenizer('False',add_special_tokens=False).input_ids
    assert len(true)==len(false)==1
    messages=module.build_messages('A synthetic color image.','',Image.open(image).convert('RGB'))
    text=proc.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    encoded=proc(text=[text],images=[Image.open(image).convert('RGB')],return_tensors='pt')
    results['qwen_instruct8']={'state':'PASS','true_ids':true,'false_ids':false,'system_prompt':module.CONFIG['SYSTEM_PROMPT'],
        'decoded_template':text,'input_tokens':int(encoded['input_ids'].shape[1]),'image_grid':encoded['image_grid_thw'].tolist(),
        'candidate_text':'','adaptation':'image-only MCQ with no native product metadata; no published matching MCQ targets'}
    assert not torch.cuda.is_initialized()
    import importlib
    modules={}
    for name in ('torch','torchvision','transformers','numpy','tokenizers','qwen_vl_utils','swift','flash_attn','peft'):
        imported=importlib.import_module(name);path=Path(imported.__file__).resolve()
        modules[name]={'path':str(path),'actual_version':getattr(imported,'__version__',None),'sha256':C.R.sha(path)}
    C.R.atomic_json(a.out,{'owner':'[codex]','state':'PASS','technical_only':True,'cuda_initialized':False,
        'weights_loaded':False,'model_effects_produced':False,'modules':modules,'backend_specs':C.entry(source),'variants':results})
    print(json.dumps({'state':'PASS','variants':list(results),'receipt':str(a.out),'sha256':C.R.sha(a.out),'cuda_initialized':False}))


if __name__=='__main__':main()
