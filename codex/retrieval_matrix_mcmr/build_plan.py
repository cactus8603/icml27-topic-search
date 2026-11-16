"""[codex] Derive an independently frozen native MCMR batch from a verified P0 plan."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from codex.retrieval_matrix import build_plan as B
from codex.retrieval_matrix_mcmr import runner as N


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def build(reference_path, output, keys=None):
    reference_path = Path(reference_path).resolve()
    reference = json.loads(reference_path.read_text())
    if reference.get('state') != 'FROZEN' or reference.get('schema') != 1:
        raise ValueError('Native model identities must derive from an already frozen verified P0 plan')
    keys = keys or list(reference['models'])
    models = {key: deepcopy(reference['models'][key]) for key in keys}
    families = {'qwen2' if model.get('variant') in {'gme-qwen2', 'vlm2vec-v2-dora'} else 'qwen3'
                for model in models.values()}
    if len(families) != 1:
        raise ValueError('Separate plans are required for the isolated Qwen2 and Qwen3 environments')
    for model in models.values():
        model['batches']['image'], model['batches']['text'] = 1, 32
        model['batch_deviation'] = 'Native fused product batch1 / query32 fixed before model scores; no product or metadata deduplication'
    data = ROOT / 'dl/retrieval-matrix/data/mcmr-mmeb-v3-native-v1'
    native_manifest = json.loads((data / 'manifest.json').read_text())
    for name, expected in native_manifest['outputs'].items():
        if sha(data / name) != expected:
            raise ValueError('Prepared native conversion artifact SHA differs: ' + name)
    for name, expected in native_manifest['raw_input_sha256'].items():
        if sha(ROOT / 'dl/retrieval3090/data/mcmr' / name) != expected:
            raise ValueError('Original raw native source differs: ' + name)
    audit_path = data / 'tokenlength-audit.json'
    audit = json.loads(audit_path.read_text())
    if (not audit['all_profiles_fit_conservative_context'] or audit['native_manifest_sha256'] != sha(data / 'manifest.json')
            or audit['script_sha256'] != sha(HERE / 'token_audit.py')):
        raise ValueError('Complete native input must pass the frozen CPU context audit first')
    by_model = {(row['repo'], row['revision']): row for row in audit['profiles']}
    for model in models.values():
        profile = by_model[(model['repo'], model['revision'])]
        if (profile['model_config_sha256'] != model['files']['config.json']
                or profile['tokenizer_sha256'] != model['files']['tokenizer.json']
                or profile['tokenizer_config_sha256'] != model['files']['tokenizer_config.json']
                or not profile['conservative_context_fits_all']):
            raise ValueError('Context audit model identity differs')
    byte_identity_path = ROOT / 'dl/retrieval-matrix/data/mmeb-mcmr/full-archive-identity.json'
    byte_identity = json.loads(byte_identity_path.read_text())
    if not byte_identity['archive_source_lfs_digest_verified'] or any(
            not row['exact_byte_match_with_original'] or row['sha256'] != native_manifest['raw_input_sha256'][name]
            for name, row in byte_identity['metadata'].items()):
        raise ValueError('Complete MMEB/original metadata byte identity must be verified before freezing')
    full = ROOT / 'dl/retrieval-matrix/data/mcmr-full'
    full_audit = json.loads((full / 'independent-audit.json').read_text())
    if full_audit['status'] != 'PASS' or full_audit['missing_candidate_images'] != 0:
        raise ValueError('Complete native image extraction must pass independent verification')
    dataset = {'format': 'mcmr-mmeb-native', 'split': 'public', 'expected_gallery': 104981,
        'expected_queries': 3997, 'expected_unique_images': 103176,
        'native_manifest': B.entry(data / 'manifest.json'),
        'corpus': B.entry(data / 'corpus.jsonl'), 'queries': B.entry(data / 'queries.jsonl'),
        'instances': B.entry(data / 'instances.jsonl'), 'token_audit': B.entry(audit_path),
        'raw_byte_identity': B.entry(byte_identity_path),
        'images_receipt': B.entry(full / 'images-receipt.jsonl'), 'images_dir': str(full / 'images'),
        'full_images_independent_audit': B.entry(full / 'independent-audit.json'),
        'similarity_batch': 1000, 'top_k': 100, 'checkpoint_product_chunk': 64,
        'use': 'fixed full native MCMR baseline only; MMEB official renderer; CORE exact conversion unverified',
        'strict_core_p0_match_eligible': False}
    source_files = deepcopy(reference['source_files'])
    source_files[str(reference_path.relative_to(ROOT))] = sha(reference_path)
    for relative in N.OWN_SOURCES:
        source_files[relative] = sha(ROOT / relative)
    for relative, expected in source_files.items():
        if sha(ROOT / relative) != expected:
            raise ValueError('Executing source differs from its frozen model-reference plan: ' + relative)
    plan = {'schema': 1, 'state': 'FROZEN', 'owner': '[codex]', 'phase': 'MCMR-native-MMEB-v2',
            'runtime_family': next(iter(families)), 'prereg': 'codex/retrieval_matrix_mcmr/PREREG.md',
            'reference_plan': B.entry(reference_path), 'models': models, 'datasets': {'mcmr': dataset},
            'source_files': source_files, 'dependencies': deepcopy(reference['dependencies']),
            'settings': deepcopy(reference['settings']), 'fixed_baseline_test': False,
            'gpu_queue': deepcopy(reference['gpu_queue']),
            'matrix_rerank_settings_unused': True,
            'known_differences': ['MMEB-V3 official native renderer; CORE original converter missing, so not strict CORE P0',
                'Released VLM2Vec fused dispatch omits vlm2vec name; route its fully loaded DoRA through identical generic gme branch',
                'Native fused batch1 / query32 fixed before metrics',
                'Storage checkpoint chunk64 amortizes receipt IO; each product still encoded separately with model batch1',
                'GPU topk called on each one-dimensional score row, matching released CORE tie behavior',
                'Only native top100 indices/score values persisted; full BF16 score blocks computed over complete corpus',
                'Image file may repeat across products; each product keeps its own metadata and checkpoint identity']}
    if output.exists():
        raise ValueError('Never overwrite an existing native frozen plan')
    output.write_text(json.dumps(plan, sort_keys=True, indent=2) + '\n')
    print(json.dumps({'owner': '[codex]', 'plan': str(output), 'sha256': sha(output), 'models': keys,
                      'strict_core_p0_match_eligible': False}))
    return plan


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-plan', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--models', nargs='+')
    args = parser.parse_args()
    build(args.reference_plan, args.out, args.models)
