"""[codex] Source-review drafts, not executable model plans or qualifications."""
from copy import deepcopy
from pathlib import Path
import argparse
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from codex.retrieval_matrix_triplet_native_p0 import contracts as C


def build(key):
    spec=C.model(key);targets={}
    for row in spec['published_targets']:
        benchmark='COCO' if row['benchmark']=='MSCOCO' else row['benchmark']
        targets[benchmark]={'as_printed_registry_table':row['table'],'as_printed_registry_metrics':row['metrics'],
            'paper':{'path':row['source_file'],'sha256':row['source_sha256']},
            'checkpoint_identity':{'variant':key,'repo':spec['repo'],'revision':spec['revision']},
            'primary_metrics':None if key in C.TABLE5 else deepcopy(row['metrics'])}
        if key in C.TABLE5:
            targets[benchmark]['table5_as_printed_R5']=deepcopy(C.TABLE5[key][benchmark])
            targets[benchmark]['direction_resolution']='UNRESOLVED_TABLE5_VS_TABLE13; never choose by model results'
    sources=C.source_manifest()
    return {'schema':1,'owner':'[codex]','phase':'P0_TRIPLET_SOURCE_REVIEW','state':'SOURCE_REVIEW_REQUIRED',
        'reviewed':False,'executable':False,'model_key':key,'model_spec':spec,'model_spec_sha256':C.digest(spec),
        'parent_registry':C.entry(C.ROOT/'codex/retrieval_matrix_extra/registry.json'),'parent_ref':C.EXTRA_REF,
        'targets_by_canonical_benchmark':targets,'source_gaps':C.gaps(key),
        'author_explicit':{'paper_tables':['5','13'] if key in C.TABLE5 else ['16'],
            'author_repo_commit':spec['sources']['tripletclip']['revision'],
            'paper_names_datasets':['MSCOCO','Flickr30k'],'paper_metric':'R@5; Table13 additionallyR@1/10',
            'paper_evaluation_reference':'https://github.com/LAION-AI/CLIP_benchmark',
            'author_retrieval_command_released':False},
        'reference_not_author_exact':{'laion_commit':C.BENCHMARK_REF,'selection':'last upstream commit before2024-11-01',
            'annotation_source':'mehdidc/retrieval_annotations release1.0.0',
            'reference_score':'CPU texts_emb @ images_emb.T; GPU torch.topk in original image-batch chunks',
            'reference_normalization':'F.normalize inside original autocast context',
            'reference_amp_default':True,'reference_image_batch_default':64,
            'reference_text_order':'all captions from each image, flatten image-batch groups, no custom text cleaner'},
        'preparation_only_settings':{'precision':'float32','image_batch':spec['batches']['image'],
            'text_batch':spec['batches']['text'],'not_author_qualified':True},
        'data_readiness':{'annotations':'PENDING_ROOT_ACQUIRE_AUDIT','coco_images_cached':5000,
            'flickr_images_cached':1000,'caption_count':'determined from exact release annotation; never assume25000/25010'},
        'dependencies':C.dependency_reference(),'external_sources':sources,
        'qualification_contract':C.qualification_contract(),
        'source_files':{'codex/retrieval_matrix_triplet_native_p0/'+name:C.sha(C.HERE/name) for name in C.OWN}}


def validate(plan):
    if plan!=build(plan.get('model_key')):raise ValueError('Source-review draft changed its exact variant, targets, gaps, source or execution prohibition')
    return plan


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out-dir',type=Path,default=C.HERE)
    args=parser.parse_args()
    for key in C.VARIANTS:
        path=args.out_dir/('draft-'+key+'.json');C.write(path,build(key));print(path)


if __name__=='__main__':main()
