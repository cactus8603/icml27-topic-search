"""[codex] Real admission first; no P2 plan without actual qualified evidence."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import argparse
import json
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from codex.retrieval_matrix_clip_panel_next import contracts as C
from codex.retrieval_matrix_panel import data_audit


def legacy_builder():
    path=ROOT/'codex/retrieval_matrix_clip_panel/build_plan.py'
    if subprocess.check_output(['git','-C',str(ROOT),'show',C.LEGACY_COMMIT+':codex/retrieval_matrix_clip_panel/build_plan.py'])!=path.read_bytes():
        raise ValueError('Original fixed panel dataset adapter differs from its frozen source')
    spec=importlib.util.spec_from_file_location('clip_next_private_builder',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def build(base_path,receipt_path,adapter_name,audit_path):
    base=C.read(base_path);key=base['model_key'];spec=base['model_spec']
    reference=C.adapter_reference(adapter_name);facade=C.adapter(reference)
    if key not in C.ADAPTERS[adapter_name]['models'] or spec.get('family')!='CLIP':
        raise ValueError('Parent model does not belong to the selected qualification facade')
    # No development manifest, test data, data audit, or model init precedes this call.
    qualified=facade.admit(C.entry(receipt_path),spec)
    candidate={'schema':1,'state':'REVIEW_REQUIRED','reviewed':False,'phase':C.PHASE,'purpose':C.PURPOSE,'retriever_family':'CLIP',
        'models':{key:deepcopy(spec)},'qualification':{key:C.entry(receipt_path)},
        'qualification_adapter':reference,'parent_p0_plan':C.entry(base_path),
        'parent_native_settings':deepcopy(base['settings']),'parent_model_signature':C.model_signature(qualified),
        'dependencies':deepcopy(base['dependencies']),'settings':C.panel_settings(base),'fixed_baseline_test':True}
    # The completed admission above is reused only within this builder call.
    C.binding(candidate,qualified)
    development=ROOT/'dl/retrieval-matrix/data/coco-development-v2'
    panel=ROOT/'codex/retrieval_matrix_panel'
    audit_args=(audit_path,development/'manifest.json',panel/'development-sources.json',
        development/'images-summary.json',development/'dev/images-receipt.jsonl')
    data_audit.verify(*audit_args)
    result={**candidate,'owner':'[codex]','scope':'Qualified fixed CLIP baselines, COCO devv2 and COLA fixed test',
        'datasets':legacy_builder().datasets(development),
        'prereg':'codex/retrieval_matrix_clip_panel_next/PREREG.md',
        'development_audit':C.entry(audit_path),'development_manifest':C.entry(development/'manifest.json'),
        'development_sources':C.entry(panel/'development-sources.json'),
        'development_images_summary':C.entry(development/'images-summary.json'),
        'development_images_receipt':C.entry(development/'dev/images-receipt.jsonl'),
        'not_covered':['P3 rerank','new method selection on COLA','native paper ranking reproduction'],
        'disclosure':'Exact qualified encoder forward; panel ranking uses FP32 CPU dot and stable gallery-order ties. These rankings are not native Google or paired-table reproduction. COLA is fixed baseline test only.'}
    result['source_files']=C.parent_sources(result)
    data_audit.bind_datasets(result,C.read(development/'manifest.json'))
    return result


def verify_datasets(plan):
    development=ROOT/'dl/retrieval-matrix/data/coco-development-v2'
    if plan.get('datasets')!=legacy_builder().datasets(development):
        raise ValueError('Panel/COLA data entries differ from the original fixed v2 dataset adapter')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('base-plan','qualification','development-audit','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--adapter',required=True,choices=tuple(C.ADAPTERS));a=p.parse_args()
    if a.out.exists():raise ValueError('Never overwrite any P2 plan')
    value=build(a.base_plan,a.qualification,a.adapter,a.development_audit)
    from codex.retrieval_matrix_native_p0.evidence import write
    write(a.out,value);print(json.dumps({'state':'PREPARED_REQUIRES_COMMIT','path':str(a.out),'sha256':C.sha(a.out)}))


if __name__=='__main__':main()
