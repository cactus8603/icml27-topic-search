"""[codex] Freeze local assets and existing code identities for CORE scaling."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
LEGACY_REF = '2daedf522c6d64b860277657d3067b61da3a0f2b'
MODEL_PATH = 'killtests/compositional-retrieval/scripts/headroom_models.py'
CONFIG_PATH = 'killtests/compositional-retrieval/configs/params_headroom.json'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    baseline = ROOT / 'dl/retrieval3090/runs/platform-torch214-20261116'
    images = ROOT / 'dl/retrieval3090/data/coco2017-anchor-rerank/images'
    manifest = images.parent / 'manifest.json'
    config_bytes = subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{LEGACY_REF}:{CONFIG_PATH}'])
    model_bytes = subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{LEGACY_REF}:{MODEL_PATH}'])
    config = json.loads(config_bytes)
    prior = json.loads((ROOT / 'dl/retrieval3090/public-download-receipt.json').read_text())['files']
    acquired = json.loads((ROOT / 'dl/retrieval3090/scaling-assets/asset-manifest.json').read_text())
    models = {}
    for key, name in [('embed2', 'core_embed'), ('cross2', 'core_reranker')]:
        spec = config['models'][name]
        folder = ROOT / 'hf_home/hub' / ('models--' + spec['repo'].replace('/', '--')) / 'snapshots' / spec['revision']
        files = {}
        for filename in spec['files']:
            path = folder / filename
            digest = sha(path)
            if prior.get(str(path), {}).get('sha256') != digest:
                raise ValueError(f'Old model lacks a valid download receipt: {path}')
            files[filename] = digest
        models[key] = {'repo': spec['repo'], 'revision': spec['revision'], 'snapshot': str(folder),
                       'files': files, 'embedding_dim': 2048 if key == 'embed2' else None,
                       'license': 'cc-by-4.0'}
    for key, spec in acquired['models'].items():
        files = {}
        for row in spec['files']:
            path = Path(row['path'])
            if row['checksum_kind'] == 'sha256':
                digest = row['expected_checksum']
            else:
                content = path.read_bytes()
                expected = hashlib.sha1(f'blob {len(content)}\0'.encode() + content).hexdigest()
                if len(content) != row['expected_size'] or expected != row['expected_checksum']:
                    raise ValueError(f'Public small-file checksum failed: {path}')
                digest = hashlib.sha256(content).hexdigest()
            files[row['file']] = digest
        models[key] = {k: spec[k] for k in ['repo', 'revision', 'snapshot', 'license']}
        models[key].update(files=files, embedding_dim=4096 if key == 'embed8' else None)
    source_names = ['runner.py', 'analyze.py', 'acquire.py', 'build_plan.py', 'run.sh',
                    'test_runner.py', 'test_analyze.py']
    plan = {
        'schema': 1, 'state': 'FROZEN', 'owner': '[codex]',
        'scope': 'exposed-development-only; CORE scaling; not independent confirmation',
        'formal_evidence': False,
        'source': {'frozen_commit': LEGACY_REF, 'model_path': MODEL_PATH, 'config_path': CONFIG_PATH,
                   'config_sha256': hashlib.sha256(config_bytes).hexdigest(),
                   'model_sha256': hashlib.sha256(model_bytes).hexdigest()},
        'source_files': {str((HERE / name).relative_to(ROOT)): sha(HERE / name) for name in source_names},
        'data': {'baseline_dir': str(baseline), 'panel_sha256': sha(baseline / 'panel.json'),
                 'vectors_sha256': sha(baseline / 'vectors.npz'), 'cross_sha256': sha(baseline / 'cross.npz'),
                 'images_dir': str(images), 'image_filename_template': '{id:012d}.jpg',
                 'images_manifest_path': str(manifest), 'images_manifest_sha256': sha(manifest),
                 'images': 8191, 'queries': 136, 'connected_groups': 110,
                 'baseline_meta_sha256': sha(baseline / 'meta.json'),
                 'baseline_receipt_sha256': sha(baseline / 'platform-receipt.json')},
        'models': models,
        'settings': {'top_k': 50, 'embed_image_batch': 8, 'embed_text_batch': 32,
                     'cross_batch': 4, 'seed': 2026111668, 'dtype': 'bfloat16', 'precision': 'bfloat16',
                     'attention': 'sdpa', 'quantization': False, 'cpu_offload': False},
        'reuse': {'embed2': True, 'cross22': True},
        'environment': {'torch': '2.14.0+cu126', 'transformers': '5.20.2', 'numpy': '2.3.5',
                        'Pillow': '11.3.0', 'torchvision': '0.29.0+cu126', 'cuda': '12.6'},
        'analysis': {'seed': 2026111668, 'bootstrap_draws': 10000, 'epsilon_pp': 2.0,
                     'grouping': 'same original group or shared positive; connected components',
                     'oracle': 'perfectly sort fixed top50; retain retriever full tail',
                     'primary': 'remaining ranking gap under retrieve8/rerank8',
                     'secondary': '100 minus retrieve8 oracle; same-pool gap recovery; factorial effects'},
        'gpu_queue': {'min_free_mib': 20480, 'max_utilization': 20, 'two_checks': True,
                      'reserve_margin_mib': 1024},
    }
    target = HERE / 'plan.json'
    target.write_text(json.dumps(plan, indent=2, sort_keys=True) + '\n')
    print(f'[codex] Wrote {target}; SHA256 {sha(target)}')


if __name__ == '__main__':
    main()
