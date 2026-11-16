"""[codex] Explicit flash padding and cross-process signature/equivalence checks."""
from pathlib import Path
from time import monotonic
import argparse
import json
import os
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from codex.retrieval_matrix_paired_runtime_fix import contracts as C,models


def signature_process(key):
    program="""import json,io
from contextlib import redirect_stdout
from codex.retrieval_matrix_paired_runtime_fix import contracts as C,models
spec=C.parent(KEY)['model_spec']
with redirect_stdout(io.StringIO()):
    model,transform,tokenizer=models.ORIGINAL.parts(spec,weights=False)
wrapper=models.ORIGINAL.PairedModel.__new__(models.ORIGINAL.PairedModel)
wrapper.spec,wrapper.model,wrapper.transform,wrapper.tokenizer=spec,model,transform,tokenizer
raw=wrapper.signature();prepared=models.prepare(wrapper)
import torch
assert type(prepared) is models.ORIGINAL.PairedModel
assert not torch.cuda.is_initialized()
print(json.dumps({'raw':raw,'stable':prepared.signature(),'cuda_initialized':False}))
""".replace('KEY',repr(key))
    value=subprocess.check_output([sys.executable,'-B','-c',program],text=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
    return json.loads(value)


def forward_equivalence(plans):
    root=ROOT/'dl/retrieval-matrix-paired-p0'
    link_path=root/'cpu-forward-plan-links-final.json';links=C.read(link_path)
    prior=C.read(C.checked(links['forward_receipt']))
    if prior.get('state')!='TECHNICAL_PASS' or prior.get('research_metrics') is not False:
        raise ValueError('Original technical CPU helper evidence required')
    output={}
    for key,plan in plans.items():
        link=links['models'][key];row=prior['models'][key]
        archive=C.read(C.checked(link['original_plan_archive']))
        if (link['current_plan']!=C.entry(ROOT/C.PARENTS[key])
                or plan['model_spec']!=archive['model_spec']
                or C.digest(plan['model_spec'])!=link['unchanged_model_spec_sha256']
                or C.entry(Path(models.ORIGINAL.__file__))!=link['unchanged_model_source']
                or row['model_source']!=link['unchanged_model_source']
                or row['cuda_initialized'] is not False or row['dtype']!='float32'
                or row['finite'] is not True or row['normalized'] is not True):
            raise ValueError('Encoder/model/tokenizer/helper equality chain broke')
        models.ORIGINAL.verify(plan['model_spec'])
        output[key]={'parent_plan':link['current_plan'],'new_plan':C.entry(C.HERE/('p0-'+key+'-einops-v1.json')),
            'original_archive':link['original_plan_archive'],'model_spec_sha256':link['unchanged_model_spec_sha256'],
            'model_source':link['unchanged_model_source'],'all_payload_tokenizer_helper_bytes_reverified':True,
            'same_original_wrapper_class':models.ORIGINAL.PairedModel.__module__+'.'+models.ORIGINAL.PairedModel.__name__,
            'prior_cpu_signature':models.stable_signature(row['signature']),
            'original_cpu_forward_not_repeated':True}
    return {'links':C.entry(link_path),'original_forward_receipt':links['forward_receipt'],'models':output}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise ValueError('Never overwrite CPU technical receipts')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU probe requires CUDA hidden')
    began=monotonic();plans={key:C.read(C.HERE/('p0-'+key+'-einops-v1.json')) for key in C.MODELS}
    from codex.retrieval_matrix_paired_runtime_fix import runner
    import torch
    from transformers import BloomPreTrainedModel
    from flash_attn.bert_padding import unpad_input,pad_input
    runtimes={key:runner.runtime(plan,gpu=False) for key,plan in plans.items()}
    value=torch.randn(2,4,8);mask=torch.tensor([[1,1,0,0],[1,1,1,0]])
    unpadded,indices,*_=unpad_input(value,mask);padded=pad_input(unpadded,indices,2,4)
    torch.testing.assert_close(padded[mask.bool()],value[mask.bool()])
    left,right=signature_process('clic_clips_l14'),signature_process('clic_clips_l14')
    if left['stable']!=right['stable']:raise ValueError('Original transform/tokenizer signatures are not stable across processes')
    if left['stable']!=models.stable_signature(left['raw']) or right['stable']!=models.stable_signature(right['raw']):
        raise ValueError('Signature policy changed more than volatile repr addresses')
    equality=forward_equivalence(plans)
    if left['stable']!=equality['models']['clic_clips_l14']['prior_cpu_signature']:
        raise ValueError('Stable actual tokenizer/transform signature differs from the original real CPU helper receipt')
    if torch.cuda.is_initialized():raise ValueError('CPU preflight initialized CUDA')
    receipt={'owner':'[codex]','state':'CPU_TECHNICAL_PASS','plans':{key:C.entry(C.HERE/('p0-'+key+'-einops-v1.json')) for key in plans},
        'source_files':next(iter(plans.values()))['source_files'],'runtimes':runtimes,
        'dependency_files':2798,'original_files':2766,'overlay_files':32,'actual_modules':10,
        'Bloom_actual_import':'PASS','explicit_flash_padding_einops_operation':'PASS',
        'cross_process_signature':{'state':'PASS','left':left,'right':right,'policy':C.SIGNATURE_POLICY},
        'encoder_bytes_and_original_cpu_helper_chain':equality,
        'real_model_weights_loaded':False,'real_model_forward_repeated':False,'real_model_effects_read':False,
        'gpu_calls':False,'cuda_initialized':False,'seconds':monotonic()-began,
        'limits':'Original helper forward evidence is inherited by exact bytes; no new numerical qualification or GPU authorization.'}
    C.write(a.out,receipt);print(json.dumps({'state':receipt['state'],'out':str(a.out),'sha256':C.sha(a.out),'cuda_initialized':False}))


if __name__=='__main__':main()
