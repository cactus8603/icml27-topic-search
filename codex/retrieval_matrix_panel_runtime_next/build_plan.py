"""[codex] Genuine repaired P0 first, then audited fixed dev-v2/COLA metadata."""
from copy import deepcopy
from pathlib import Path
import argparse
import json
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from codex.retrieval_matrix_panel_runtime_next import contracts as C
from codex.retrieval_matrix_panel import data_audit
Q=C.Q


def datasets(development):
    manifest=Q.read(development/'manifest.json')
    if manifest['conjunction']['queries']<1000:raise ValueError('Formal conjunction dev requires >=1000 queries')
    result={}
    for name,folder,panel in (('conjunction-dev','dev','conjunction'),('negation-dev','negation','negation')):
        path=development/folder;meta=manifest[panel]
        for file,expected in meta['outputs'].items():
            if Q.sha(path/file)!=expected:raise ValueError('Development metadata changed')
        result[name]={'format':'conjunction','split':'dev','gallery':Q.entry(path/'gallery.json'),
            'queries':Q.entry(path/'queries.jsonl'),'images_receipt':Q.entry(development/'dev/images-receipt.jsonl'),
            'images_dir':str(development/'dev/images'),'expected_gallery':20000,
            'expected_queries':meta['queries'],'expected_eligible':meta['queries'],
            'scope':'formal conjunction development' if folder=='dev' else 'negation diagnostic only; insufficient 1000-query power'}
    cola=ROOT/'dl/retrieval-matrix/data/cola'
    for subset,gallery,queries in (('gqa',2589,320),('paco',7921,400)):
        result['cola-'+subset]={'format':'cola_single','split':'public','subset':subset,
            'parquet':Q.entry(cola/'data'/('singleobjects_'+subset+'.parquet')),
            'vocabulary':Q.entry(cola/'data'/('singleobjects_'+subset+'_labels.json')),
            'images_receipt':Q.entry(cola/'file_receipt.jsonl'),'image_root':str(cola),
            'allowed_shared_image_roots':[str(ROOT/'dl/retrieval3090/data/coco2017/val2017'),
                str(ROOT/'dl/claude-panel/conj-panel-v1/dev/images')],
            'expected_gallery':gallery,'expected_queries':queries,'expected_eligible':queries,
            'use':'fixed baseline test only; no fitting or method selection'}
    return result


def build(family,key,receipt_path,audit_path):
    base,qualified,signature,environment=C.admit_parent(family,key,Q.entry(receipt_path))
    development=ROOT/'dl/retrieval-matrix/data/coco-development-v2';panel=ROOT/'codex/retrieval_matrix_panel'
    data_audit.verify(audit_path,development/'manifest.json',panel/'development-sources.json',
        development/'images-summary.json',development/'dev/images-receipt.jsonl')
    result={'schema':1,'owner':'[codex]','state':'REVIEW_REQUIRED','reviewed':False,'phase':C.PHASE,
        'retriever_family':'MLLM','family_adapter':family,'models':{key:deepcopy(base['models'][key])},
        'qualification':{key:Q.entry(receipt_path)},'parent_p0_plan':qualified['p0_plan'],
        'parent_native_settings':deepcopy(base['settings']),'settings':deepcopy(base['settings']),
        'parent_model_signature':signature,'parent_runtime':environment,'dependencies':deepcopy(base['dependencies']),
        'fixed_baseline_test':True,'datasets':datasets(development),
        'ranking':{'dtype':'float32','backend':'CPU dot','ties':'stable-gallery-order','main_k':100,'top_ks':[50,100,200]},
        'development_audit':Q.entry(audit_path),'development_manifest':Q.entry(development/'manifest.json'),
        'development_sources':Q.entry(panel/'development-sources.json'),
        'development_images_summary':Q.entry(development/'images-summary.json'),
        'development_images_receipt':Q.entry(development/'dev/images-receipt.jsonl'),
        'prereg':'codex/retrieval_matrix_panel_runtime_next/PREREG.md','source_files':C.source_files(base),
        'not_covered':['P3','new methods','COLA method selection','native paper ranking reproduction'],
        'disclosure':'Same qualified repaired MLLM encoder, batches and recorded runtime. Original P0 does not set/record actual seed or TF32/matmul profile; no retrospective claim is made. Panel uses FP32 CPU dot/stable gallery ties; COLA fixed caption adaptation only.'}
    data_audit.bind_datasets(result,Q.read(development/'manifest.json'))
    return result


def verify_datasets(plan):
    if plan.get('datasets')!=datasets(ROOT/'dl/retrieval-matrix/data/coco-development-v2'):
        raise ValueError('P2 datasets differ from the exact complete fixed dev-v2/COLA adapter')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family',choices=tuple(C.FAMILIES),required=True);parser.add_argument('--model',required=True)
    for name in ('qualification','development-audit','out'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():raise ValueError('Never overwrite a P2 plan')
    result=build(args.family,args.model,args.qualification,args.development_audit)
    args.out.write_text(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'state':'REVIEW_REQUIRED','sha256':Q.sha(args.out)}))


if __name__=='__main__':main()
