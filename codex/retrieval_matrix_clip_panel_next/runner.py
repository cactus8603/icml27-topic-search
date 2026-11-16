"""[codex] Private frozen panel modules; exact qualified family-specific forward."""
import importlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import uuid
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from codex.retrieval_matrix_clip_panel_next import contracts as C
from codex.retrieval_matrix_clip_panel_next import build_plan as B
from codex.retrieval_matrix_panel import data,data_audit


def private_module(label,path):
    package='.'.join(Path(path).parent.relative_to(ROOT).parts)
    spec=importlib.util.spec_from_file_location(package+'._clip_next_'+label+'_'+uuid.uuid4().hex,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def create_runtime(plan):
    engine=private_module('matrix',ROOT/'codex/retrieval_matrix/runner.py')
    legacy=private_module('clip',ROOT/'codex/retrieval_matrix_clip_panel/runner.py')
    # Rebind only this private module's globals. Shared frozen modules remain intact.
    legacy.R=engine
    legacy.base_load,legacy.base_labels=engine.load_data,engine.load_labels
    legacy.base_analyze,legacy.base_json_rows=engine.analyze,engine.json_rows
    original_gate,original_load=engine.source_gate,engine.load_plan
    original_rank=engine.rank_blocks
    key=next(iter(plan['models']));expected_spec=plan['models'][key]

    def validate_spec(spec):
        if spec!=expected_spec:raise ValueError('Panel encoder spec differs from its qualified parent')

    def construct(spec):
        validate_spec(spec)
        model=C.model_class(plan['qualification_adapter'])(spec)
        if model.signature()!=plan['parent_model_signature']:
            raise ValueError('Actual P2 model class/processor/raw tokenizer differs from its qualified P0')
        import torch
        if any(p.device.type!='cuda' or (p.is_floating_point() and p.dtype!=torch.float32) for p in model.model.parameters()):
            raise ValueError('Keep the qualified full FP32 model on the assigned GPU')
        return model

    def load_plan(path):
        current=original_load(path)
        if current.get('reviewed') is not True:raise ValueError('P2 needs its own independent review after genuine parent admission')
        if current!=plan:raise ValueError('Runtime plan differs from its admitted fixed plan')
        C.admit_models(current);C.check_declared_sources(current)
        return current

    def source_gate(path,current,commit):
        value=original_gate(path,current,commit)
        C.admit_models(current);C.check_declared_sources(current)
        folders={str(Path(relative).parent) for relative in current['source_files'] if relative.startswith('codex/')}
        for folder in sorted(folders):
            if subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain','--',folder],text=True).strip():
                raise ValueError('Every executing CLIP source directory must remain clean: '+folder)
        dependencies=C.read(C.checked(current['dependencies']))
        for relative,expected in dependencies['files'].items():
            path=(ROOT/relative).resolve()
            if not path.is_relative_to(ROOT/'dl') or C.sha(path)!=expected:raise ValueError('Qualified dependency bytes changed')
        paths=[engine.checked_file(current[name]) for name in ('development_audit','development_manifest',
            'development_sources','development_images_summary','development_images_receipt')]
        data_audit.verify(*paths);data_audit.bind_datasets(current,C.read(paths[1]))
        B.verify_datasets(current)
        for name in ('cola-gqa','cola-paco'):
            if current['datasets'][name].get('use')!='fixed baseline test only; no fitting or method selection':
                raise ValueError('COLA may only be a fixed baseline test')
        return value

    def runtime_environment(spec):
        if (os.environ.get('RETRIEVAL_CENTRAL_OWNER')!='[codex]' or not os.environ.get('RETRIEVAL_CENTRAL_TASK')
                or not os.environ.get('CUDA_VISIBLE_DEVICES')):
            raise ValueError('CLIP next GPU production requires the root central dispatcher')
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise ValueError('Require one centrally assigned GPU')
        validate_spec(spec)
        parent=C.read(C.checked(plan['parent_p0_plan']))
        runtime=C.registered_module(plan['qualification_adapter'],'runtime_module').runtime(parent,gpu=True)
        return {'native_qualified_runtime':runtime,'qualification_adapter':plan['qualification_adapter'],
            'parent_p0_plan':plan['parent_p0_plan'],'ranking_dtype':'float32','tie_rule':'stable-gallery-order'}

    def rank_blocks(images,texts,*,block_queries=64):
        import numpy as np
        images,texts=np.asarray(images),np.asarray(texts)
        if images.dtype!=np.float32 or texts.dtype!=np.float32:raise ValueError('Panel ranking requires FP32 normalized vectors')
        return original_rank(images,texts,block_queries=block_queries)

    engine.models=SimpleNamespace(OfficialModel=construct,validate_spec=validate_spec)
    engine.SOURCE_FILES+=tuple(relative for relative in [*C.COMMON,*C.LEGACY,
        *('codex/retrieval_matrix_clip_panel_next/'+name for name in C.OWN)] if relative not in engine.SOURCE_FILES)
    engine.load_plan,engine.source_gate,engine.runtime_environment=load_plan,source_gate,runtime_environment
    engine.json_rows=lambda path:data.lf_rows(path) if path.suffix=='.jsonl' else legacy.base_json_rows(path)
    engine.load_data,engine.load_labels,engine.analyze=legacy.load_data,legacy.load_labels,legacy.analyze
    engine.rank_blocks=rank_blocks
    return engine


def main():
    import argparse
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--plan',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--stage',required=True,choices=('smoke','index','analyze'));a,remaining=p.parse_known_args()
    plan=C.read(a.plan);C.admit_models(plan)
    out=C.output_path(a.out)
    create_runtime(plan).main(['--plan',str(a.plan),'--out',str(out),'--stage',a.stage,*remaining])


if __name__=='__main__':main()
