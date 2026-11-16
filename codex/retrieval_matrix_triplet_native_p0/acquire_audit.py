"""[codex] Root-invoked exact LAION annotation acquisition and offline audit.

No images/models are downloaded. This tool never creates qualifications or
claims the chosen LAION revision was the exact original author invocation.
"""
from collections import defaultdict
from pathlib import Path
import argparse
import hashlib
import json
import sys
import urllib.request
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from codex.retrieval_matrix_triplet_native_p0 import contracts as C

ASSETS={
    'coco':{'name':'coco_test_karpathy.json','id':81280122,'size':4116303,'updated_at':'2022-10-17T10:12:13Z'},
    'flickr':{'name':'flickr30k_test_karpathy.txt','id':81280309,'size':398044,'updated_at':'2022-10-17T10:13:09Z'},
}
BASE='https://github.com/mehdidc/retrieval_annotations/releases/download/1.0.0/'


def assets():
    metadata=C.read(C.SOURCES/'annotation-release-1.0.0.json')
    if metadata['tag_name']!='1.0.0':raise ValueError('The original benchmark annotation release changed')
    output={}
    for key,expected in ASSETS.items():
        rows=[x for x in metadata['assets'] if x['name']==expected['name']]
        if len(rows)!=1 or any(rows[0].get(name)!=value for name,value in expected.items()):
            raise ValueError('Fixed annotation asset ID/bytes/date differs')
        if rows[0]['browser_download_url']!=BASE+expected['name']:raise ValueError('Unexpected annotation URL')
        output[key]=rows[0]
    return output


def acquire(destination,*,fetch=None):
    fetch=fetch or (lambda url:urllib.request.urlopen(url,timeout=90).read())
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    receipt={'schema':1,'state':'COMPLETE','scope':'public-annotation-acquisition-only',
        'source_metadata':C.entry(C.SOURCES/'annotation-release-1.0.0.json'),'files':{}}
    for key,item in assets().items():
        path=destination/item['name']
        if not path.exists():
            payload=fetch(item['browser_download_url'])
            if len(payload)!=item['size']:raise ValueError('Incomplete annotation download')
            temporary=path.with_suffix(path.suffix+'.partial')
            if temporary.exists():raise ValueError('Preserve any partial evidence; do not silently overwrite')
            temporary.write_bytes(payload);temporary.replace(path)
        if path.stat().st_size!=item['size']:raise ValueError('Existing annotation byte count differs')
        receipt['files'][key]={'entry':C.entry(path),'asset_id':item['id'],'bytes':item['size'],
            'url':item['browser_download_url'],'release_updated_at':item['updated_at'],
            'upstream_digest':item.get('digest'),'note':'Release lacks upstream SHA; exact acquired SHA requires independent audit/freeze.'}
    C.write(destination/'acquisition.json',receipt)
    return receipt


def annotation_rows(key,path):
    """Match referenced torchvision COCO and LAION Flickr ordering/text policy."""
    if key=='coco':
        value=C.read(path);images={};annotations=defaultdict(list);caption_ids=set()
        for row in value['images']:
            image_id=row['id']
            if type(image_id) is not int or image_id in images:raise ValueError('Duplicate/noninteger COCO image ID')
            images[image_id]=row
        for row in value['annotations']:
            if row['image_id'] not in images or row['id'] in caption_ids or not isinstance(row['caption'],str):
                raise ValueError('Invalid or duplicate original COCO annotation identity')
            caption_ids.add(row['id']);annotations[row['image_id']].append(row)
        return [{'id':str(key),'filename':Path(images[key]['file_name']).name,
            'captions':[row['caption'] for row in annotations[key]],'caption_ids':[str(row['id']) for row in annotations[key]]}
            for key in sorted(images)]
    if key!='flickr':raise ValueError('Only the two named native retrieval datasets are allowed')
    grouped={}
    with Path(path).open() as stream:
        next(stream)  # The exact LAION source unconditionally discards header.
        for line in stream:
            line=line.strip()
            if line:
                filename,caption=line.split('.jpg,')
                filename+='.jpg';grouped.setdefault(filename,[]).append(caption)
    return [{'id':Path(name).stem,'filename':name,'captions':captions,
        'caption_ids':[name+'#'+str(i) for i in range(len(captions))]} for name,captions in grouped.items()]


def bind_images(rows,cache,*,expected_images,verify_images):
    """Only exact cached filenames are reused; no best-effort substitute."""
    if len(rows)!=expected_images or set(row['filename'] for row in rows)!=set(cache):
        raise ValueError('Official annotation image set differs from complete shared cache; acquire exact missing images first')
    gallery=[];queries=[];labels=[];seen=set()
    for index,row in enumerate(rows):
        stored=cache[row['filename']];path=Path(stored['path'])
        if not row['captions'] or len(row['captions'])!=len(row['caption_ids']):raise ValueError('An original image lost its captions')
        if (not path.is_file() or path.stat().st_size!=stored['bytes']
                or (verify_images and C.sha(path)!=stored['sha256'])):raise ValueError('Shared cached image payload differs')
        if verify_images:
            from PIL import Image
            with Image.open(path) as image:image.verify()
        gallery.append({'id':row['id'],'path':str(path.resolve()),'sha256':stored['sha256'],'bytes':stored['bytes'],
            'source_metadata':row})
        for text,caption_id in zip(row['captions'],row['caption_ids']):
            if caption_id in seen:raise ValueError('Original caption ID is not unique')
            seen.add(caption_id);queries.append({'id':caption_id,'text':text})
            labels.append({'query_id':caption_id,'positive_image_index':index})
    return {'gallery':gallery,'queries':queries,'labels':labels,'counts':{'images':len(gallery),'texts':len(queries),
        'excluded_images':0,'excluded_texts':0},'ordering':'source-defined annotation ordering; no caption cleaning or first-five truncation'}


def inference_items(bundle):
    return [row['path'] for row in bundle['gallery']],[row['text'] for row in bundle['queries']]


def shared_cache(name,*,verify_images):
    from codex.retrieval_matrix_native_p0 import data
    plan=C.read(C.frozen('codex/retrieval_matrix_native_p0/p0-siglip2.json','387d4fa5293967705b93aeba235cb9ffe643cc42'))
    bundle=data.load(plan['benchmarks'][name]['data'],verify_images=verify_images)
    return {Path(row['path']).name:{'path':row['path'],'sha256':row['image_sha256'],'bytes':row['image_bytes']}
        for row in bundle['gallery']}


def audit(destination,*,verify_images):
    destination=Path(destination);acquisition=C.read(destination/'acquisition.json');result={
        'schema':1,'scope':'exact-reference-annotation-and-cache-audit','state':'COMPLETE','benchmarks':{},
        'acquisition':C.entry(destination/'acquisition.json'),'images_hashed_decoded':bool(verify_images),
        'does_not_resolve_author_direction_amp_batch':True,'qualification':False}
    for name in ASSETS:
        path=C.checked(acquisition['files'][name]['entry']);rows=annotation_rows(name,path)
        bundle=bind_images(rows,shared_cache(name,verify_images=verify_images),expected_images=5000 if name=='coco' else 1000,
            verify_images=verify_images)
        out=destination/(name+'-reference-bundle.json');C.write(out,bundle)
        result['benchmarks'][name]={'bundle':C.entry(out),'annotation':C.entry(path),'counts':bundle['counts'],
            'gallery_order_sha256':C.digest([x['id']for x in bundle['gallery']]),
            'caption_order_text_sha256':C.digest(bundle['queries']),'mapping_sha256':C.digest(bundle['labels'])}
    C.write(destination/'offline-audit.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('list','acquire','audit'))
    parser.add_argument('--destination',type=Path,default=C.DATA);parser.add_argument('--execute-root-acquisition',action='store_true')
    parser.add_argument('--verify-images',action='store_true');args=parser.parse_args()
    if args.action=='list':print(json.dumps(assets(),sort_keys=True));return
    if args.action=='acquire':
        if not args.execute_root_acquisition:parser.error('Root must explicitly authorize this exact public annotation acquisition')
        value=acquire(args.destination)
    else:value=audit(args.destination,verify_images=args.verify_images)
    print(json.dumps({'scope':value['scope'],'state':value['state']}))


if __name__=='__main__':main()
