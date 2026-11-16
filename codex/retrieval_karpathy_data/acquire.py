"""[codex] Acquire only original Karpathy COCO test images, with official ZIP CRC evidence."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import struct
import time
import urllib.request
import zipfile
import zlib
from zoneinfo import ZoneInfo

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'dl/retrieval-matrix/data/karpathy-original'
SOURCE_SHA = '2fd999220673258012acfb411a4e7e66af7d488050b2519b0badcc49b7600b8d'
ANNOTATIONS_SHA = '113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268'
CENTRAL_SHA = 'cd282f9bd992a327931de58ff2c8ed38f36ae341e701e12e4ca8ba12ce659b2c'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def jsonl(path, values):
    with Path(path).open('w') as stream:
        for row in values:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')


def parse_central(data):
    """Read standard central records; individual COCO JPEG sizes fit uint32."""
    result, offset = {}, 0
    while offset < len(data):
        if len(data) - offset < 46:
            raise ValueError('Truncated central directory header')
        fields = struct.unpack_from('<4s6H3I5H2I', data, offset)
        if fields[0] != b'PK\x01\x02':
            raise ValueError('Unexpected central directory signature')
        name_size, extra_size, comment_size = fields[10:13]
        end = offset + 46 + name_size + extra_size + comment_size
        if end > len(data):
            raise ValueError('Truncated central directory variable fields')
        name = data[offset + 46:offset + 46 + name_size].decode('utf-8' if fields[3] & 0x800 else 'cp437')
        if name in result or not name or fields[9] == 0xffffffff:
            raise ValueError('Duplicate/empty name or unsupported ZIP64 image size')
        result[name] = {'crc32': f'{fields[7]:08x}', 'bytes': fields[9]}
        offset = end
    return result


def source_rows(source=BASE / 'dataset_coco.json', *, enforce=True):
    if enforce and sha(source) != SOURCE_SHA:
        raise ValueError('Original Karpathy caption JSON SHA differs')
    rows = [row for row in json.loads(Path(source).read_text())['images'] if row['split'] == 'test']
    if enforce and (len(rows) != 5000 or sum(len(row['sentences']) for row in rows) != 25010):
        raise ValueError('Original image/caption denominator differs')
    if len({row['cocoid'] for row in rows}) != len(rows):
        raise ValueError('Repeated original COCO image ID')
    if any(row['filepath'] != 'val2014' or row['filename'] != f'COCO_val2014_{row["cocoid"]:012d}.jpg' for row in rows):
        raise ValueError('Original COCO2014 filepath/name changed')
    return rows


def inspect_bytes(data, expected, dimensions):
    if len(data) != expected['bytes'] or f'{zlib.crc32(data):08x}' != expected['crc32']:
        raise ValueError('Image bytes/CRC differ from official val2014 ZIP central directory')
    with Image.open(io.BytesIO(data)) as image:
        if image.format != 'JPEG' or list(image.size) != list(dimensions):
            raise ValueError('Image format/dimensions differ from official COCO metadata')
        image.verify()
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(), 'crc32': expected['crc32'],
            'width': dimensions[0], 'height': dimensions[1], 'format': 'JPEG'}


def prepare_dimensions(rows, output):
    source = ROOT / 'dl/retrieval3090/data/coco2017/annotations_trainval2017.zip'
    if sha(source) != ANNOTATIONS_SHA:
        raise ValueError('Official COCO image metadata archive SHA differs')
    wanted = {row['cocoid'] for row in rows}
    dimensions, flickr_ids = {}, {}
    with zipfile.ZipFile(source) as archive:
        for name in ('annotations/instances_train2017.json', 'annotations/instances_val2017.json'):
            with archive.open(name) as stream:
                metadata = json.load(stream)
            # Compare URL namespace only; no semantic labels or result files.
            for image in metadata['images']:
                image_id = image['id']
                if image_id in wanted:
                    dimensions[str(image_id)] = [image['width'], image['height']]
                url = image.get('flickr_url', '')
                if url:
                    photo_id = url.rsplit('/', 1)[-1].split('_', 1)[0]
                    flickr_ids.setdefault(photo_id, []).append(image_id)
            del metadata
    if set(dimensions) != set(map(str, wanted)):
        raise ValueError('Official image metadata does not cover every original Karpathy test image')
    write_json(output / 'dimensions.json', dimensions)
    flickr_source = ROOT / 'dl/retrieval-matrix/data/flickr30k_test/karpathy_test_manifest.jsonl'
    flickr_rows = [json.loads(line) for line in flickr_source.open() if line.strip()]
    overlaps = []
    for row in flickr_rows:
        filename = row.get('filename', Path(row.get('image', row.get('path', ''))).name)
        if not filename:
            raise ValueError('Flickr source manifest lacks an original image filename')
        photo_id = filename.split('_', 1)[0].split('.', 1)[0]
        if photo_id in flickr_ids:
            overlaps.append({'flickr_filename': filename, 'flickr_photo_id': photo_id,
                             'coco_ids': flickr_ids[photo_id]})
    gallery_paths = {
        'original_dev_v1': ROOT / 'dl/claude-panel/conj-panel-v1/dev/gallery.json',
        'original_test_v1': ROOT / 'dl/claude-panel/conj-panel-v1/test/gallery.json',
        'development_v2': ROOT / 'dl/retrieval-matrix/data/coco-development-v2/dev/gallery.json'}
    gallery_sets = {key: {row['id'] for row in json.loads(path.read_text())}
                    for key, path in gallery_paths.items()}
    for row in overlaps:
        row['gallery_intersections'] = {key: sorted(ids.intersection(row['coco_ids']))
                                        for key, ids in gallery_sets.items()}
    write_json(output / 'flickr-coco-source-id-audit.json', {'owner': '[codex]',
        'scope': 'Exact original Flickr photo ID against official COCO2017 flickr_url; no visual-near-duplicate claim',
        'coco_annotations_sha256': ANNOTATIONS_SHA, 'flickr_manifest_sha256': sha(flickr_source),
        'gallery_sha256': {key: sha(path) for key, path in gallery_paths.items()},
        'flickr_images': len(flickr_rows), 'overlapping_flickr_images': len(overlaps), 'overlaps': overlaps,
        'cola_gqa_to_coco': 'unknown; no official VisualGenome image_data mapping found in existing caches'})
    return dimensions


def manifest_rows(rows, image_root):
    return [{'id': str(row['cocoid']), 'cocoid': row['cocoid'], 'imgid': row['imgid'],
             'filename': row['filename'], 'filepath': row['filepath'], 'split': 'test',
             'path': str((image_root / row['filename']).absolute()),
             'captions': [sentence['raw'] for sentence in row['sentences']],
             'sentids': [sentence['sentid'] for sentence in row['sentences']]}
            for row in rows]


def acquire(workers=12):
    rows = source_rows()
    output = BASE / 'coco-test'
    images = output / 'images'
    images.mkdir(parents=True, exist_ok=True)
    central_file = BASE / 'val2014-central-directory.bin'
    if sha(central_file) != CENTRAL_SHA:
        raise ValueError('Official val2014 central directory evidence SHA differs')
    central = parse_central(central_file.read_bytes())
    if len(central) != 40505:
        raise ValueError('Official val2014 archive member denominator changed')
    inventory_path = BASE / 'coco-cache-inventory.json'
    inventory = json.loads(inventory_path.read_text())
    if inventory['source_metadata_sha256'] != SOURCE_SHA:
        raise ValueError('Cache inventory does not correspond to the fixed original test split')
    dimensions = prepare_dimensions(rows, output)
    began = time.monotonic()

    def one(row):
        image_id, name = row['cocoid'], row['filename']
        dest = images / name
        expected = central['val2014/' + name]
        size = dimensions[str(image_id)]
        url = 'http://images.cocodataset.org/val2014/' + name
        rejected = []
        if dest.exists():
            detail = inspect_bytes(dest.read_bytes(), expected, size)
            return {'id': str(image_id), 'path': str(dest.absolute()), 'filename': name, **detail,
                    'source_url': url, 'acquisition': 'reused_cache' if dest.is_symlink() else 'downloaded',
                    'cache_path': str(dest.resolve()) if dest.is_symlink() else None}
        for raw in inventory['candidates'].get(str(image_id), []):
            cached = Path(raw).resolve()
            if not cached.is_relative_to(ROOT) or 'retrieval-independent-panel' in str(cached):
                raise ValueError('Cache source escapes fixed public workspace scope')
            try:
                detail = inspect_bytes(cached.read_bytes(), expected, size)
            except (ValueError, OSError) as error:
                rejected.append({'path': str(cached), 'reason': str(error)})
                continue
            dest.symlink_to(cached)
            return {'id': str(image_id), 'path': str(dest.absolute()), 'filename': name, **detail,
                    'source_url': url, 'acquisition': 'reused_cache', 'cache_path': str(cached),
                    'rejected_cache_candidates': rejected}
        for attempt in range(4):
            try:
                request = urllib.request.Request(url, headers={'User-Agent': 'codex-public-data-audit/1.0'})
                with urllib.request.urlopen(request, timeout=60) as response:
                    data = response.read(10_000_001)
                detail = inspect_bytes(data, expected, size)
                temp = dest.with_name(dest.name + '.partial')
                temp.write_bytes(data)
                temp.replace(dest)
                return {'id': str(image_id), 'path': str(dest.absolute()), 'filename': name, **detail,
                        'source_url': url, 'acquisition': 'downloaded', 'cache_path': None,
                        'rejected_cache_candidates': rejected}
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(1 + attempt)

    receipts, failures = {}, []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_id = {executor.submit(one, row): str(row['cocoid']) for row in rows}
        for future in as_completed(future_to_id):
            key = future_to_id[future]
            try:
                receipts[key] = future.result()
            except Exception as error:
                failures.append({'id': key, 'error': type(error).__name__ + ': ' + str(error)})
            completed = len(receipts) + len(failures)
            if completed % 100 == 0 or completed == len(rows):
                progress = {'owner': '[codex]', 'expected': len(rows), 'ok': len(receipts),
                    'downloaded': sum(r['acquisition'] == 'downloaded' for r in receipts.values()),
                    'reused': sum(r['acquisition'] == 'reused_cache' for r in receipts.values()),
                    'failed': len(failures), 'wall_seconds': time.monotonic() - began, 'research_metrics': False}
                write_json(output / 'progress.json', progress)
                print(json.dumps(progress), flush=True)
    if failures:
        write_json(output / 'failures.json', failures)
        raise ValueError('Some original images failed; preserve partial download for exact resume')
    receipt_rows = [receipts[str(row['cocoid'])] for row in rows]
    jsonl(output / 'image-receipt.jsonl', receipt_rows)
    jsonl(output / 'manifest.jsonl', manifest_rows(rows, images))
    result = {'owner': '[codex]', 'status': 'PASS', 'time': datetime.now(ZoneInfo('Asia/Taipei')).strftime('%Y-%m-%d %H:%M'),
        'scope': 'Fixed official baseline test acquisition only; no model execution or development selection',
        'source_json_sha256': SOURCE_SHA, 'central_directory_sha256': CENTRAL_SHA,
        'official_crc32_and_bytes_checked': True, 'coco_annotations_sha256': ANNOTATIONS_SHA,
        'cache_inventory_sha256': sha(inventory_path), 'images': len(rows),
        'raw_captions': sum(len(row['sentences']) for row in rows), 'caption_rule': 'All original raw strings including whitespace; no first5 truncation',
        'downloaded_images': sum(r['acquisition'] == 'downloaded' for r in receipt_rows),
        'reused_images': sum(r['acquisition'] == 'reused_cache' for r in receipt_rows),
        'image_bytes': sum(r['bytes'] for r in receipt_rows), 'wall_seconds': time.monotonic() - began,
        'outputs': {name: sha(output / name) for name in ('manifest.jsonl', 'image-receipt.jsonl', 'dimensions.json', 'flickr-coco-source-id-audit.json')}}
    write_json(output / 'acquisition.json', result)
    print(json.dumps(result), flush=True)


def audit():
    output = BASE / 'coco-test'
    acquisition = json.loads((output / 'acquisition.json').read_text())
    for name, expected in acquisition['outputs'].items():
        if sha(output / name) != expected:
            raise ValueError('Acquisition output differs: ' + name)
    rows = source_rows()
    manifest = [json.loads(line) for line in (output / 'manifest.jsonl').open()]
    if manifest != manifest_rows(rows, output / 'images'):
        raise ValueError('Original raw captions/IDs/order/whitespace differ')
    central = parse_central((BASE / 'val2014-central-directory.bin').read_bytes())
    if sha(BASE / 'val2014-central-directory.bin') != CENTRAL_SHA:
        raise ValueError('Source central directory differs')
    receipt = [json.loads(line) for line in (output / 'image-receipt.jsonl').open()]
    if [row['id'] for row in receipt] != [row['id'] for row in manifest]:
        raise ValueError('Receipt does not preserve complete source image order')
    for row in receipt:
        values = inspect_bytes(Path(row['path']).read_bytes(), central['val2014/' + row['filename']], [row['width'], row['height']])
        if any(row[key] != value for key, value in values.items()):
            raise ValueError('Image changed after download: ' + row['filename'])
    result = {'owner': '[codex]', 'status': 'PASS', 'scope': 'Offline complete original raw-caption/image/SHA/ZIP-CRC verification',
              'images': len(receipt), 'captions': sum(len(row['captions']) for row in manifest),
              'acquisition_sha256': sha(output / 'acquisition.json')}
    write_json(output / 'offline-audit.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--audit', action='store_true')
    args = parser.parse_args()
    audit() if args.audit else acquire(args.workers)
