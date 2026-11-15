"""[codex] Independent train calibration and additional holistic D3 encodings."""
import argparse
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
from time import monotonic

import numpy as np
import cache_scores as legacy
from cache_v2 import atomic_json, batches, fingerprint
from logic_baselines import CONCEPT_TEMPLATES, fit_presence_calibration


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def fixed_json(path, value):
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError(f'Refusing to replace a different design: {path}')
    if not path.exists():
        atomic_json(path, value)


def prepare(out):
    runtime = Path(os.environ['RETRIEVAL_ROOT'])
    data = runtime / 'data'
    anchor_path = data / 'coco2017-anchor-rerank/manifest.json'
    previous_path = data / 'coco2017-rerank/manifest.json'
    conjunction_path = data / 'coco2017-rank-and/manifest.json'
    anchor, previous, conjunction = [json.loads(p.read_text()) for p in
                                     (anchor_path, previous_path, conjunction_path)]
    # Only the historically exposed DEV partition is eligible for calibration.
    source = conjunction['parts']['dev']
    excluded = sorted(set(anchor['ids']) | set(previous['ids']))
    excluded_sha = set(anchor['jpeg_sha256']) | set(previous['jpeg_sha256'])
    candidates = [(int(key), digest) for key, digest in zip(source['ids'], source['jpg_sha256'], strict=True)
                  if key not in excluded and digest not in excluded_sha]
    candidates.sort(key=lambda pair: hashlib.sha256(f'2026111664:{pair[0]}'.encode()).hexdigest())
    selected, seen_sha = [], set()
    for key, digest in candidates:
        if digest in seen_sha:
            continue
        path = data / 'coco2017-rank-and/images' / f'{key:012d}.jpg'
        if not path.exists() or sha(path) != digest:
            raise ValueError(f'Historical calibration JPEG failed verification: {path}')
        selected.append({'image_id': key, 'sha256': digest, 'path': str(path)})
        seen_sha.add(digest)
        if len(selected) == 512:
            break
    if len(selected) != 512:
        raise ValueError('Insufficient disjoint exposed train calibration images')
    conditions, index = legacy.condition_texts(anchor)
    names = {int(key): name for key, name in anchor['names'].items()}
    text_to_cat = {legacy.joint_text([name]): cat for cat, name in names.items()}
    categories = [text_to_cat[text] for text in conditions]
    ids = [row['image_id'] for row in selected]
    presence = legacy.image_categories(ids)
    # Confirm that all calibration images are actually train2017, without consulting
    # the sealed panel. Both manifests were already excluded when that pool was built.
    import ijson
    import zipfile
    with zipfile.ZipFile(data / 'coco2017/annotations_trainval2017.zip') as archive:
        with archive.open('annotations/instances_train2017.json') as stream:
            train_ids = {int(row['id']) for row in ijson.items(stream, 'images.item')}
    if not set(ids) <= train_ids:
        raise ValueError('Calibration contains an image outside train2017')
    out.mkdir(parents=True, exist_ok=True)
    design = {'schema': 1, 'seed': 2026111664, 'split': 'train2017',
              'scope': 'historically-exposed-conjunction-DEV-only',
              'source_manifest_sha256': sha(conjunction_path),
              'evaluation_manifest_sha256': sha(anchor_path),
              'previous_manifest_sha256': sha(previous_path),
              'selected': selected, 'excluded_image_ids': excluded,
              'category_ids': categories, 'category_names': [names[cat] for cat in categories],
              'condition_texts': conditions, 'condition_index': index,
              'labels': [[int(cat in presence[key]) for cat in categories] for key in ids],
              'templates': list(CONCEPT_TEMPLATES), 'images_disjoint_by_id_and_jpeg_sha': True}
    fixed_json(out / 'calibration_design.json', design)
    print(json.dumps({'stage': 'calibration_design_ready', 'images': len(selected),
                      'concepts': len(categories), 'design_sha256': sha(out / 'calibration_design.json')}), flush=True)


def _encode(args):
    import torch
    if os.environ.get('DEV_DIRTY') != '0':
        raise ValueError('Research encodings require a committed code reference')
    design_path = args.design / 'calibration_design.json'
    design = json.loads(design_path.read_text())
    d3_path = args.d3 / 'texts.jsonl'
    d3_design = json.loads((args.d3 / 'design.json').read_text())
    from diagnostics import validate_design
    validate_design(d3_design)
    d3 = [json.loads(line) for line in d3_path.read_text().splitlines() if line.strip()]
    if d3 != d3_design['texts']:
        raise ValueError('D3 text rows differ from the frozen score-blind design')
    texts = [row['text'] for row in d3]
    if len(texts) != len(set(texts)):
        raise ValueError('D3 text table must be unique')
    config, snapshots = legacy.original.models()
    data = Path(os.environ['RETRIEVAL_ROOT']) / 'data'
    source_manifests = {'source_manifest_sha256': data / 'coco2017-rank-and/manifest.json',
                        'evaluation_manifest_sha256': data / 'coco2017-anchor-rerank/manifest.json',
                        'previous_manifest_sha256': data / 'coco2017-rerank/manifest.json'}
    if any(sha(path) != design[field] for field, path in source_manifests.items()):
        raise ValueError('Calibration source manifest changed')
    if any(sha(Path(row['path'])) != row['sha256'] for row in design['selected']):
        raise ValueError('Calibration JPEG changed after design preparation')
    args.out.mkdir(parents=True, exist_ok=True)
    cp = args.out / 'checkpoints'
    cp.mkdir(exist_ok=True)
    identity = {'commit': os.environ['DEV_COMMIT'], 'source': os.environ['DEV_SOURCE'],
                'freeze': os.environ['DEV_FREEZE'], 'calibration_design_sha256': sha(design_path),
                'd3_texts_sha256': sha(d3_path), 'config': config,
                'torch': torch.__version__, 'cuda': torch.version.cuda,
                'device': torch.cuda.get_device_name(0)}
    fixed_json(args.out / 'provenance.json', identity)
    lcse = {'condition_texts': np.asarray(design['condition_texts'])}
    d3_vectors, calibration, timings = {'texts': np.asarray(texts)}, {}, {}
    templates = [template.format(name) for name in design['category_names'] for template in CONCEPT_TEMPLATES]
    ids = [row['image_id'] for row in design['selected']]
    labels = np.asarray(design['labels'], dtype=np.uint8)
    for name, cls in (('core_embed', legacy.original.Encoder), ('cs_clip', legacy.original.ClipEncoder)):
        print(f'[codex] Additional encodings: {name}', flush=True)
        began = monotonic()
        torch.cuda.reset_peak_memory_stats()
        model = cls(name, snapshots[name], config, 'cuda')
        def encode_text(rows):
            return model.encode(rows, images=False)
        def encode_image(rows):
            return model.encode([legacy.original.rgb(Path(row['path'])) for row in rows], images=True)
        def stage(part, items, batch, fn):
            return batches(cp / f'{name}_{part}.npy', items, batch, fn,
                           identity | {'model': name, 'stage': part, 'items_fingerprint': fingerprint(items)},
                           status=args.out / 'progress.json')
        template_vectors = stage('templates', templates, config['batch_size']['embed_text'], encode_text)
        means = template_vectors.reshape(len(design['category_names']), len(CONCEPT_TEMPLATES), -1).mean(axis=1)
        # Mean of normalized template vectors implements mean cosine. Do NOT renormalize.
        lcse[f'{name}_lcse_condition'] = means
        lcse[f'{name}_template_vectors'] = template_vectors.reshape(len(design['category_names']), len(CONCEPT_TEMPLATES), -1)
        image_vectors = stage('calibration_image', design['selected'], config['batch_size']['embed_image'], encode_image)
        cosines = image_vectors @ means.T
        artifact = fit_presence_calibration(cosines, labels, image_ids=ids,
                                           excluded_image_ids=design['excluded_image_ids'], backbone=name,
                                           source_manifest_sha256=design['source_manifest_sha256'])
        artifact['source']['calibration_design_sha256'] = sha(design_path)
        if calibration:
            if calibration['source'] != artifact['source']:
                raise ValueError('Calibration sources differ across backbones')
            calibration['models'].update(artifact['models'])
        else:
            calibration = artifact
        d3_vectors[f'{name}_text'] = stage('d3_text', texts, config['batch_size']['embed_text'], encode_text)
        timings[name] = legacy.score.timeline(torch, began, len(texts) + len(templates) + len(ids), 'supplement_inputs')
        del model
        gc.collect()
        torch.cuda.empty_cache()
    lcse_metadata = {'schema': 1, 'scope': 'lcse-four-template-condition-vectors',
                     'templates': list(CONCEPT_TEMPLATES),
                     'calibration_design_sha256': sha(design_path),
                     'calibration_source_manifest_sha256': design['source_manifest_sha256'],
                     'eval_manifest_sha256': design['evaluation_manifest_sha256'],
                     'model_revisions': {name: path.name for name, path in snapshots.items()},
                     'source': os.environ['DEV_SOURCE'], 'freeze': os.environ['DEV_FREEZE'],
                     'bundle_commit': os.environ['DEV_COMMIT']}
    lcse['metadata_json'] = np.asarray(json.dumps(lcse_metadata, sort_keys=True))
    np.savez(args.out / 'lcse_vectors.npz', **lcse)
    d3_vectors['metadata_json'] = np.asarray(json.dumps({
        'schema': 'd3-holistic-encodings-v1', 'design_sha256': sha(args.d3 / 'design.json'),
        'texts_sha256': d3_design['texts_sha256'],
        'gallery_ids_sha256': d3_design['gallery_ids_sha256'],
        'model_revisions': {name: path.name for name, path in snapshots.items()
                            if name in ('core_embed', 'cs_clip')},
        'model_config_sha256': legacy.original.MODEL_CONFIG_SHA, 'normalization': 'unit_l2',
        'source': os.environ['DEV_SOURCE'], 'freeze': os.environ['DEV_FREEZE'],
        'commit': os.environ['DEV_COMMIT']}, sort_keys=True))
    np.savez(args.out / 'd3_embeddings.npz', **d3_vectors)
    atomic_json(args.out / 'lcse_calibration.json', calibration)
    atomic_json(args.out / 'meta.json', identity | {'timings': timings, 'status': 'complete',
                'outputs_sha256': {filename: sha(args.out / filename) for filename in
                                   ('lcse_vectors.npz', 'd3_embeddings.npz', 'lcse_calibration.json')}})
    atomic_json(args.out / 'progress.json', {'stage': 'complete'})
    print('[codex] Supplementary encodings complete', flush=True)


def encode(args):
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / 'writer.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _encode(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--out', type=Path, required=True)
    inference = sub.add_parser('encode')
    inference.add_argument('--design', type=Path, required=True)
    inference.add_argument('--d3', type=Path, required=True)
    inference.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.out)
    else:
        encode(args)


if __name__ == '__main__':
    main()
