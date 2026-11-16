"""[codex] Private shared producer; no import-time mutation of frozen modules."""
import importlib.util
import inspect
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import uuid
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from codex.retrieval_matrix_panel_runtime_next import contracts as C,build_plan as B
from codex.retrieval_matrix_panel import data,data_audit


def private_module(label,path):
    name='.'.join(path.parent.relative_to(ROOT).parts)+'._panel_next_'+label+'_'+uuid.uuid4().hex
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def original_class_guard():
    from codex.retrieval_matrix import models
    cls=models.OfficialModel
    if (not inspect.isclass(cls) or cls.__module__!='codex.retrieval_matrix.models'
            or cls.__qualname__!='OfficialModel'):
        raise ValueError('Start the isolated new launcher in a fresh process; shared model dispatch was already patched')
    return cls


def create_runtime(plan):
    # Never import either legacy runner: both patch canonical R at import time.
    original_class_guard()
    engine=private_module('matrix',ROOT/'codex/retrieval_matrix/runner.py')
    original_load,original_gate=engine.load_plan,engine.source_gate
    base_runtime,base_rank=engine.runtime_environment,engine.rank_blocks
    base_data,base_labels,base_analyze=engine.load_data,engine.load_labels,engine.analyze
    family=plan['family_adapter'];key=next(iter(plan['models']));expected=plan['models'][key]
    if family=='core-qwen3-9943159':
        from codex.retrieval_matrix_core_runtime_fix import compat
        encoder=compat.CompatibleOfficialModel
    elif family=='qwen2-einops-9943159':
        from codex.retrieval_matrix_qwen2.adapters import Qwen2OfficialModel
        encoder=Qwen2OfficialModel
    else:raise ValueError('Unknown repaired MLLM family')

    def validate_spec(spec):
        if spec!=expected:raise ValueError('The complete P2 model spec differs from its genuine repaired P0')

    def construct(spec,**kwargs):
        original_class_guard();validate_spec(spec)
        model=encoder(spec,**kwargs)
        if model.signature()!=plan['parent_model_signature']:
            raise ValueError('Actual P2 repaired class/processor/adapter signature differs from qualified P0')
        return model

    def load_plan(path):
        current=original_load(path)
        if current!=plan or current.get('reviewed') is not True:raise ValueError('A new independent P2 review/freeze is required')
        base,_=C.admit(current);C.check_sources(current,base)
        return current

    def source_gate(path,current,commit):
        value=original_gate(path,current,commit)
        base,_=C.admit(current);C.check_sources(current,base)
        C.dependencies(base,family,imports=False)
        folders={str(Path(relative).parent) for relative in current['source_files'] if relative.startswith('codex/')}
        for folder in sorted(folders):
            if subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain','--',folder],text=True).strip():
                raise ValueError('Every executing repaired P2 source folder must remain clean: '+folder)
        paths=[engine.checked_file(current[name]) for name in ('development_audit','development_manifest',
            'development_sources','development_images_summary','development_images_receipt')]
        data_audit.verify(*paths);data_audit.bind_datasets(current,C.Q.read(paths[1]));B.verify_datasets(current)
        return value

    def runtime_environment(spec):
        if (os.environ.get('RETRIEVAL_CENTRAL_OWNER')!='[codex]' or not os.environ.get('RETRIEVAL_CENTRAL_TASK')
                or not os.environ.get('CUDA_VISIBLE_DEVICES')):
            raise ValueError('Repaired MLLM P2 requires the root central dispatcher')
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise ValueError('Require exactly one assigned CUDA device')
        if Path(sys.executable).resolve()!=(ROOT/C.FAMILIES[family]['python']).resolve():
            raise ValueError('Use the exact original repaired parent Python entry point')
        validate_spec(spec);base=C.parent(family);C.dependencies(base,family,imports=True)
        # Preserve the original 994 runtime semantics, including the fact that
        # it does not apply/record the declarative seed or TF32/matmul defaults.
        if family=='core-qwen3-9943159':
            actual=base_runtime(spec);actual['runtime_compatibility']=dict(spec['runtime_compatibility'])
        else:
            import peft,flash_attn
            from importlib.metadata import version
            from codex.retrieval_matrix_qwen2_runtime_fix import contracts as repair
            reduced={**spec,'environment':{k:v for k,v in spec.get('environment',{}).items() if k!='peft'}}
            actual=base_runtime(reduced);actual['peft']=version('peft')
            for name,module in (('peft',peft),('flash_attn',flash_attn)):
                actual['modules'][name]={'version':module.__version__,'path':str(Path(module.__file__).resolve())}
            for name,wanted in spec.get('environment',{}).items():
                if actual.get(name)!=wanted:raise ValueError('Qualified Qwen2 environment differs: '+name)
            actual['modules']['einops']=repair.actual_modules()['einops']
            actual['runtime_fix']={'overlay_manifest':base['runtime_fix']['overlay'],
                'parent_freeze_commit':base['runtime_fix']['parent_freeze_commit']}
        C.runtime_matches(actual,plan['parent_runtime'])
        return actual

    def load_data(spec,**kwargs):
        return data.load_data(spec,engine,**kwargs) if spec['format']=='cola_single' else base_data(spec,**kwargs)

    def load_labels(spec,bundle):
        return data.load_labels(spec,bundle,engine) if spec['format']=='cola_single' else base_labels(spec,bundle)

    def analyze(out,spec,bundle,settings,*,index_out=None):
        if spec['format']!='cola_single':return base_analyze(out,spec,bundle,settings,index_out=index_out)
        if index_out is not None:raise ValueError('P3 is outside this fixed P2 baseline adapter')
        identity=engine.read(out/'provenance.json')
        if identity['dataset_fingerprint']!=bundle['fingerprint']:raise ValueError('COLA index inputs differ')
        _,order,meta=engine.validate_index(out,identity,bundle,settings)
        report,detail=engine.metrics.evaluate(order,[row['id'] for row in bundle['gallery']],load_labels(spec,bundle))
        report.update(schema=1,model=identity['model'],dataset=identity['dataset'],gallery=len(bundle['gallery']),
            raw_queries=bundle['raw_queries'],excluded=[],retained_empty=[],
            protocol='full gallery admission; remove non-hard images only during scoring without refill; threshold-group AP authoritative',
            provenance={'index_meta_sha256':engine.sha(out/'index-meta.json'),'index':meta},
            official=data.official_metrics(out,spec,bundle,engine))
        engine.atomic_json(out/'results/report.json',report);engine.atomic_json(out/'results/per-query.json',detail)

    def rank_blocks(image,text,*,block_queries=64):
        import numpy as np
        if np.asarray(image).dtype!=np.float32 or np.asarray(text).dtype!=np.float32:
            raise ValueError('P2 ranking requires normalized FP32 arrays')
        return base_rank(image,text,block_queries=block_queries)

    engine.models=SimpleNamespace(OfficialModel=construct,validate_spec=validate_spec)
    engine.SOURCE_FILES+=tuple('codex/retrieval_matrix_panel_runtime_next/'+name for name in C.OWN)
    engine.load_plan,engine.source_gate,engine.runtime_environment=load_plan,source_gate,runtime_environment
    engine.load_data,engine.load_labels,engine.analyze=load_data,load_labels,analyze
    engine.json_rows=lambda path:data.lf_rows(path) if path.suffix=='.jsonl' else engine.read(path)
    engine.rank_blocks=rank_blocks
    return engine


def main():
    import argparse
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--stage',choices=('smoke','index','analyze'),required=True)
    args,remaining=parser.parse_known_args();plan=C.Q.read(args.plan)
    C.admit(plan)
    create_runtime(plan).main(['--plan',str(args.plan),'--out',str(C.output_path(args.out)),'--stage',args.stage,*remaining])


if __name__=='__main__':main()
