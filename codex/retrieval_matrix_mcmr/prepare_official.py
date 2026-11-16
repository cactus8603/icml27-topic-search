"""[codex] Fixed official MMEB-V3 native MCMR renderer, CPU preparation only."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[2]
OFFICIAL = ROOT / 'dl/retrieval-matrix/data/mmeb-mcmr'
RAW = ROOT / 'dl/retrieval3090/data/mcmr'
IMAGES = ROOT / 'dl/retrieval-matrix/data/mcmr-full/images'
PINS = {
    'candidates.jsonl': '05242fb414333a99c562a986edb8b2d1997db1726f731b6131ab97bf557fcbc0',
    'query.jsonl': '28925c2d015a75fda763d7bd3092bfd01bf96c8d3e8cd2397702e1fda049e68c',
}
SOURCE_PINS = {
    'vlm2vec-source/src/data/dataset/mcmr_dataset.py': '356e9cf511b173fe9f21fe2561042e190c18a49c4a9441d78b4a1c4c288f77da',
    'vlm2vec-source/src/data/eval_dataset/mcmr_dataset.py': '820b19135ab36dbc6e99aeb392cf84002277d00c415b84b762722d945bfeca38',
    'vlm2vec-source/experiments/public/eval/image.yaml': '989d04a13f369a08e60741bfe726cbe1428bef5e2ead26bccf5a04e6b841d45f',
    'gme.py': 'e02a4adb9d4baa053db7d4c0160efef2f88e35040e05153e942fd900a865e550',
    'qwen3_pointwise.py': 'dce636de090a7e022f9227fc3aafd639b4259c07c74c61b8676161edf0cb3ce7',
}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def official_module():
    for filename, expected in SOURCE_PINS.items():
        if sha(OFFICIAL / filename) != expected:
            raise ValueError('Official source SHA mismatch: ' + filename)
    path = OFFICIAL / 'vlm2vec-source/src/data/dataset/mcmr_dataset.py'
    spec = importlib.util.spec_from_file_location('pinned_mmeb_mcmr_native', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.CANDIDATE_TEXT_FIELDS != ('title', 'description', 'features', 'price', 'Date First Available'):
        raise ValueError('Official renderer field order changed')
    return module


def comparison_function(filename, name):
    """Import only the fixed pure renderer AST, excluding Torch/model loading."""
    path = OFFICIAL / filename
    if sha(path) != SOURCE_PINS[filename]:
        raise ValueError('Comparison source SHA mismatch')
    tree = ast.parse(path.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(functions) != 1:
        raise ValueError('Expected exactly one renderer function')
    namespace = {'Dict': Dict, 'Any': Any}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


def write_row(stream, row):
    stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')


def prepare(out: Path):
    out = out.absolute()
    if out.is_symlink() or not out.resolve().is_relative_to(ROOT / 'dl/retrieval-matrix/data'):
        raise ValueError('Only an isolated ignored matrix data directory is permitted')
    if out.exists() and any(out.iterdir()):
        raise ValueError('Never overwrite a completed or partial native adapter version')
    for filename, expected in PINS.items():
        if sha(RAW / filename) != expected:
            raise ValueError('Original data SHA mismatch: ' + filename)
    module = official_module()
    gme = comparison_function('gme.py', 'render_text')
    qwen = comparison_function('qwen3_pointwise.py', 'render_candidate_text')
    query_rows = module.load_mcmr_query_rows(str(RAW / 'query.jsonl'))
    if len(query_rows) != 3997 or len({row['qry_id'] for row in query_rows}) != 3997:
        raise ValueError('Expected all 3997 original unique queries')
    image_receipt = IMAGES.parent / 'images-receipt.jsonl'
    if sha(image_receipt) != '26ca885a59d6c9a47d1883018d83cc7afa3ccfb7d6acdc67640906096d967376':
        raise ValueError('Original full image receipt SHA mismatch')
    image_hashes = {row['file']: row['sha256'] for row in
                    map(json.loads, image_receipt.read_text().splitlines())}
    out.mkdir(parents=True, exist_ok=True)
    candidate_ids, image_names = set(), set()
    comparison = Counter()
    samples = []
    char_lengths = []
    with (out / 'corpus.jsonl').open('w') as corpus:
        for line in (RAW / 'candidates.jsonl').open():
            raw = json.loads(line)
            cid = str(raw.get('candidate_id', '')).strip()
            if not cid or cid in candidate_ids:
                raise ValueError('Missing or duplicated candidate ID')
            # Corpus admission has no access to query labels or qrel.
            image = module.resolve_candidate_image_path(raw, str(IMAGES))
            if image is None:
                raise ValueError('Candidate image is missing: ' + cid)
            path = Path(image)
            if path.name not in image_hashes:
                raise ValueError('Candidate image absent from verified archive receipt')
            text = module.build_mcmr_candidate_text(raw)
            if not text:
                text = f'candidate_id: {cid}'  # exact official parser fallback
            candidate_ids.add(cid)
            image_names.add(path.name)
            write_row(corpus, {'id': cid, 'text': text, 'image': str(path), 'modality': 'image-text'})
            alternative_gme, alternative_qwen = gme(raw, 2048), qwen(raw)
            comparison['candidates'] += 1
            comparison['mmeb_equals_original_gme'] += text == alternative_gme
            comparison['mmeb_equals_original_qwen_pointwise'] += text == alternative_qwen
            comparison['original_gme_equals_original_qwen_pointwise'] += alternative_gme == alternative_qwen
            comparison['mmeb_chars_gt_2048'] += len(text) > 2048
            char_lengths.append(len(text))
            if len(samples) < 3:
                samples.append({'id': cid, 'mmeb_official': text,
                                'original_gme_2048': alternative_gme,
                                'original_qwen_pointwise_2048': alternative_qwen})
    if len(candidate_ids) != 104981 or len(image_names) != 103176:
        raise ValueError('Full official candidate/image counts differ')
    with (out / 'queries.jsonl').open('w') as queries, (out / 'instances.jsonl').open('w') as qrel:
        for row in query_rows:
            if not set(row['label_names']) <= candidate_ids:
                raise ValueError('A query positive is missing from complete native corpus')
            write_row(queries, {'id': row['qry_id'], 'text': row['qry_text']})
            write_row(qrel, {'qid': row['qry_id'], 'pos': row['label_names']})
    (out / 'renderer_comparison.json').write_text(json.dumps({
        'scope': 'string provenance only; no model scoring or renderer selection by effects',
        'fixed_renderer': 'MMEB-V3 official build_mcmr_candidate_text',
        'counts': dict(comparison), 'candidate_character_range': [min(char_lengths), max(char_lengths)],
        'samples_first_three_source_rows': samples}, ensure_ascii=False, indent=2) + '\n')
    manifest = {
        'owner': '[codex]', 'adapter_version': 'mcmr-mmeb-v3-native-v1',
        'renderer_repo': 'TIGER-AI-Lab/VLM2Vec',
        'renderer_revision': 'f658ebeb49704784330ffe9e68a10796698d75e3',
        'renderer_function': 'src/data/dataset/mcmr_dataset.py:build_mcmr_candidate_text',
        'schema': 'CORE task-compatible queries/corpus/instances JSONL',
        'core_exact_conversion': 'unverified; original CORE conversion scripts not released',
        'raw_dataset': 'Lux1997/MCMR', 'raw_revision': '12d53f4063804ee6a976024c7375ab2fbe4368c8',
        'raw_input_sha256': PINS, 'official_source_sha256': SOURCE_PINS,
        'candidate_ids': len(candidate_ids), 'unique_candidate_images': len(image_names),
        'queries': len(query_rows), 'query_instruction': module.DEFAULT_QUERY_INSTRUCTION,
        'candidate_fields_order': list(module.CANDIDATE_TEXT_FIELDS),
        'candidate_separator': '\n', 'list_item_separator': ' ', 'renderer_character_truncation': None,
        'corpus_admission_uses_qrel_or_labels': False, 'image_and_metadata_both_required': True,
        'full_image_receipt_sha256': sha(image_receipt), 'producer_sha256': sha(__file__),
        'spec_sha256': sha(Path(__file__).with_name('SPEC.md')),
        'outputs': {name: sha(out / name) for name in
                    ('corpus.jsonl', 'queries.jsonl', 'instances.jsonl', 'renderer_comparison.json')},
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(json.dumps({k: manifest[k] for k in ('adapter_version', 'queries', 'candidate_ids',
                                             'unique_candidate_images', 'core_exact_conversion')}, indent=2))
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path,
                        default=ROOT / 'dl/retrieval-matrix/data/mcmr-mmeb-v3-native-v1')
    prepare(parser.parse_args().out)
