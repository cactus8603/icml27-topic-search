"""[codex] CPU full-corpus token counts; no truncation, model weights, or GPU."""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import logging
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Union
import unicodedata
from zoneinfo import ZoneInfo

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
HUB = ROOT / 'hf_home/hub'
CORE = ROOT / 'dl/retrieval-matrix/sources/core'
MODELS = [
    ('Alibaba-NLP/core-emb-2b', 'cc9870ea92d64ebd5f8f9fc368d1120453564158', 'qwen3'),
    ('Alibaba-NLP/core-emb-8b', '123aef9e5e9e76fb686420f077fce3ef3b580c2b', 'qwen3'),
    ('Qwen/Qwen3-VL-Embedding-2B', '9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda', 'qwen3'),
    ('Qwen/Qwen3-VL-Embedding-8B', '2c4565515e0f265c6511776e7193b22c0968ddc7', 'qwen3'),
    ('Alibaba-NLP/gme-Qwen2-VL-2B-Instruct', '9cfa6413f704a7c1cf5064d240748e10c876b286', 'qwen2'),
    ('VLM2Vec/VLM2Vec-V2.0', 'e39ff079b8275ef876d3656da8c0bddbff3c4dde', 'qwen2'),
]
REFERENCE_PLANS = (
    ('codex/retrieval_matrix/p0-qwen3-batch1.json', 'f4d5c65bdb95c5d858b2b0cbc1af8a78a297709e648a753c67deeb6759fcfc23'),
    ('codex/retrieval_matrix_qwen2/p0-qwen2-batch1.json', 'bc4f0d14d365b5250f1147a5920b81dbab6afe20a82df5495ca6346da1752c37'),
)
TOKENIZATION_FIELDS = (
    'repo', 'revision', 'tokenizer_sha256', 'tokenizer_config_sha256',
    'model_config_sha256', 'preprocessor_config_sha256', 'formatter_source_sha256',
    'formatter_source', 'corpus_bare_text_tokens',
    'corpus_official_chat_with_one_image_placeholder_plus_eod', 'query_text_only_tokens',
    'conversation_prefix', 'conversation_suffix', 'corpus_arg_max_tokens_id',
    'tokenizer_truncation', 'model_inference',
)


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def runtime_specs():
    result = {}
    for relative, expected in REFERENCE_PLANS:
        path = ROOT / relative
        if sha(path) != expected:
            raise ValueError('Frozen runtime processor reference plan differs: ' + relative)
        plan = json.loads(path.read_text())
        if plan.get('state') != 'FROZEN':
            raise ValueError('Runtime processor bounds require an already frozen model plan')
        for key, spec in plan['models'].items():
            identity = (spec['repo'], spec['revision'])
            if identity in result:
                raise ValueError('Duplicated runtime model identity')
            result[identity] = {'model_key': key, 'spec': spec, 'reference_plan': relative,
                                'reference_plan_sha256': expected}
    return result


def image_bounds(processor, runtime, chat_max, context):
    """Bound both payload and actual runtime override in units of merged image tokens."""
    patch_size, merge_size = int(processor['patch_size']), int(processor['merge_size'])
    factor = patch_size * merge_size
    payload_upper = math.ceil(processor['max_pixels'] / factor ** 2)
    overrides = runtime['spec']['processor']
    # The frozen loader passes these overrides to from_pretrained. In
    # particular 1,310,720 pixels / (14*2)^2 requires ceil(...)=1,672 tokens;
    # a hard-coded 1,280-token Qwen2 fallback was an invalid upper bound.
    runtime_max_pixels = int(overrides['max_pixels'])
    if runtime_max_pixels <= 0:
        raise ValueError('Runtime processor max_pixels must be positive')
    runtime_upper = math.ceil(runtime_max_pixels / factor ** 2)
    image_upper = max(payload_upper, runtime_upper)
    full_upper = chat_max + image_upper - 1
    return {'payload_image_token_upper': payload_upper,
            'runtime_image_token_upper': runtime_upper,
            'runtime_max_fused_tokens': chat_max + runtime_upper - 1,
            'conservative_image_token_upper': image_upper,
            'conservative_max_fused_tokens': full_upper,
            'conservative_context_fits_all': full_upper <= context,
            'image_bound_provenance': {
                'rule': 'max(ceil(payload_max_pixels/factor^2), ceil(runtime_max_pixels/factor^2)); replace one image placeholder',
                'patch_size': patch_size, 'merge_size': merge_size, 'merged_image_factor': factor,
                'payload_max_pixels': int(processor['max_pixels']), 'runtime_max_pixels': runtime_max_pixels,
                'runtime_processor_overrides': deepcopy(overrides),
                'runtime_processor_overrides_sha256': digest(overrides),
                'reference_plan': runtime['reference_plan'],
                'reference_plan_sha256': runtime['reference_plan_sha256'],
                'model_key': runtime['model_key']}}


def formatter(kind):
    if kind == 'qwen3':
        path = CORE / 'models/qwen3_vl/qwen3_vl_wrapper.py'
        classname = 'Qwen3VLForEmbeddingProcessor'
    else:
        path = CORE / 'models/qwen2_vl/qwen2_vl_wrapper.py'
        classname = 'Qwen2VLProcessorForEmbedding'
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == classname)
    function = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'format_model_input')
    namespace = {'Image': Image, 'List': List, 'Union': Union, 'Dict': Dict, 'Any': Any,
                 'unicodedata': unicodedata, 'logger': logging.getLogger(__name__)}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['format_model_input'], path


def summarize(values):
    ordered = sorted(values)
    return {'items': len(values), 'min': ordered[0], 'max': ordered[-1],
            'p50': ordered[math.ceil(len(ordered) * .5) - 1],
            'p95': ordered[math.ceil(len(ordered) * .95) - 1],
            'p99': ordered[math.ceil(len(ordered) * .99) - 1],
            'gt4096': sum(value > 4096 for value in values)}


def run(data, output):
    from transformers import AutoTokenizer
    manifest = json.loads((data / 'manifest.json').read_text())
    for name in ['corpus.jsonl', 'queries.jsonl', 'instances.jsonl']:
        if sha(data / name) != manifest['outputs'][name]:
            raise ValueError('Native prepared data checksum mismatch')
    # Valid product strings contain Unicode line-separator characters. JSONL
    # records are delimited by ASCII LF, not Python str.splitlines().
    with (data / 'corpus.jsonl').open() as stream:
        corpus = [json.loads(line) for line in stream]
    with (data / 'queries.jsonl').open() as stream:
        queries = [json.loads(line) for line in stream]
    cache, results, runtimes = {}, [], runtime_specs()
    for repo, revision, kind in MODELS:
        snapshot = HUB / ('models--' + repo.replace('/', '--')) / 'snapshots' / revision
        config = json.loads((snapshot / 'config.json').read_text())
        processor = json.loads((snapshot / 'preprocessor_config.json').read_text())
        context = config.get('text_config', {}).get('max_position_embeddings') or config.get('max_position_embeddings')
        token_file = snapshot / 'tokenizer.json'
        token_sha = sha(token_file)
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        tokenizer.backend_tokenizer.no_truncation()
        tokenizer.backend_tokenizer.no_padding()
        method, source = formatter(kind)
        # Exact official format_model_input with a placeholder local image;
        # no image data is opened by this pure conversation-formatting method.
        receiver = SimpleNamespace(instruction_standalone=True, min_pixels=4096, max_pixels=1310720)
        marker = 'MCMR_AUDIT_TEXT_SENTINEL_9018'
        messages = method(receiver, marker, 'dummy.jpg', None, False,
                          'Represent the given image and text.')
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if formatted.count(marker) != 1:
            raise ValueError('Cannot isolate exact official conversation text')
        prefix, suffix = formatted.split(marker)
        key = (token_sha, prefix, suffix)
        if key not in cache:
            bare, chat, query_counts = [], [], []
            backend = tokenizer.backend_tokenizer
            for offset in range(0, len(corpus), 256):
                texts = [row['text'] for row in corpus[offset:offset + 256]]
                bare.extend(len(x.ids) for x in backend.encode_batch(texts, add_special_tokens=False))
                # Both released CORE wrappers append one end-of-document token.
                chat.extend(len(x.ids) + 1 for x in backend.encode_batch(
                    [prefix + text + suffix for text in texts], add_special_tokens=True))
            for offset in range(0, len(queries), 256):
                query_counts.extend(len(x.ids) for x in backend.encode_batch(
                    [row['text'] for row in queries[offset:offset + 256]], add_special_tokens=False))
            cache[key] = (bare, chat, query_counts)
        bare, chat, query_counts = cache[key]
        bounds = image_bounds(processor, runtimes[(repo, revision)], max(chat), context)
        result = {'repo': repo, 'revision': revision, 'tokenizer_sha256': token_sha,
                  'tokenizer_config_sha256': sha(snapshot / 'tokenizer_config.json'),
                  'model_config_sha256': sha(snapshot / 'config.json'),
                  'preprocessor_config_sha256': sha(snapshot / 'preprocessor_config.json'),
                  'formatter_source_sha256': sha(source), 'formatter_source': str(source.relative_to(ROOT)),
                  'corpus_bare_text_tokens': summarize(bare),
                  'corpus_official_chat_with_one_image_placeholder_plus_eod': summarize(chat),
                  'query_text_only_tokens': summarize(query_counts),
                  **bounds, 'model_context': context,
                  'conversation_prefix': prefix, 'conversation_suffix': suffix,
                  'corpus_arg_max_tokens_id': corpus[max(range(len(chat)), key=chat.__getitem__)]['id'],
                  'tokenizer_truncation': False, 'model_inference': False}
        results.append(result)
        print(repo, 'bare', max(bare), 'chat', max(chat), 'image upper', bounds['conservative_image_token_upper'],
              'full upper', bounds['conservative_max_fused_tokens'], 'context', context, flush=True)
    report = {'owner': '[codex]', 'scope': 'CPU native corpus token length; no model weights or GPU',
              'image_bound_revision': 2,
              'native_manifest_sha256': sha(data / 'manifest.json'),
              'script_sha256': sha(Path(__file__)), 'profiles': results,
              'all_profiles_fit_conservative_context': all(x['conservative_context_fits_all'] for x in results),
              'no_candidates_or_queries_filtered': True,
              'interpretation': 'max_length=4096 with truncation=False is not a hard 4096 cap; image expansion is bounded separately; formal batch settings still require a runtime preflight'}
    output.write_text(json.dumps(report, indent=2) + '\n')
    return report


def refresh_image_bounds(data, original, output):
    """Reuse the original complete tokenizer statistics, recording the correction explicitly."""
    previous_bytes = original.read_bytes()
    previous = json.loads(previous_bytes)
    if previous['script_sha256'] != 'ed2b8a810794486ddd7db1340c6e162a154585286ae5b85180fbd328991b396c':
        raise ValueError('This correction must start from the preserved original full-tokenization audit')
    if previous['native_manifest_sha256'] != sha(data / 'manifest.json'):
        raise ValueError('Original tokenizer statistics do not match prepared data')
    manifest = json.loads((data / 'manifest.json').read_text())
    for name in ('corpus.jsonl', 'queries.jsonl', 'instances.jsonl'):
        if sha(data / name) != manifest['outputs'][name]:
            raise ValueError('Original complete data changed while reusing tokenizer statistics')
    if {(row['repo'], row['revision']) for row in previous['profiles']} != {(repo, revision) for repo, revision, _ in MODELS}:
        raise ValueError('Reused token statistics must retain all six original model identities')
    saved = data / 'tokenlength-audit.original-v1.json'
    if saved.exists() and saved.read_bytes() != previous_bytes:
        raise ValueError('Never overwrite a differing original audit')
    saved.write_bytes(previous_bytes)
    report, runtimes = deepcopy(previous), runtime_specs()
    unchanged = [{key: row[key] for key in TOKENIZATION_FIELDS} for row in previous['profiles']]
    changes = []
    for row in report['profiles']:
        snapshot = HUB / ('models--' + row['repo'].replace('/', '--')) / 'snapshots' / row['revision']
        for name, key in [('config.json', 'model_config_sha256'), ('tokenizer.json', 'tokenizer_sha256'),
                          ('tokenizer_config.json', 'tokenizer_config_sha256'),
                          ('preprocessor_config.json', 'preprocessor_config_sha256')]:
            if sha(snapshot / name) != row[key]:
                raise ValueError('Original tokenization model identity changed: ' + name)
        if sha(ROOT / row['formatter_source']) != row['formatter_source_sha256']:
            raise ValueError('Original official chat formatter changed')
        old_bounds = {key: row[key] for key in ('payload_image_token_upper', 'conservative_image_token_upper',
                                               'conservative_max_fused_tokens', 'conservative_context_fits_all')}
        processor = json.loads((snapshot / 'preprocessor_config.json').read_text())
        row.update(image_bounds(processor, runtimes[(row['repo'], row['revision'])],
            row['corpus_official_chat_with_one_image_placeholder_plus_eod']['max'], row['model_context']))
        changes.append({'repo': row['repo'], 'revision': row['revision'], 'previous': old_bounds,
                        'corrected_runtime_image_token_upper': row['runtime_image_token_upper'],
                        'corrected_runtime_max_fused_tokens': row['runtime_max_fused_tokens'],
                        'corrected_conservative_image_token_upper': row['conservative_image_token_upper'],
                        'corrected_conservative_max_fused_tokens': row['conservative_max_fused_tokens']})
    current = [{key: row[key] for key in TOKENIZATION_FIELDS} for row in report['profiles']]
    if current != unchanged:
        raise ValueError('Image-bound correction must not alter any tokenizer statistics')
    probes = []
    for filename, expected in (
        ('independent-processor-qwen3.json', '67cf923fdac469f6eb9e660bf05cfdd8012c1192e71d39242f3b9cb07ba4c62f'),
        ('independent-processor-qwen2.json', 'cbd8df4047f1cdd555b5b04531dd15ee07c7e33a4b91cb5b138f6e62523b9a69')):
        path = data / filename
        if sha(path) != expected:
            raise ValueError('Independent processor correction evidence differs')
        probe = json.loads(path.read_text())
        if probe['state'] != 'PASS' or probe['gpu_inference'] or probe['weights_loaded']:
            raise ValueError('Correction evidence must be a synthetic CPU processor-only probe')
        probes.append({'path': str(path.relative_to(ROOT)), 'sha256': expected,
                       'models': deepcopy(probe['models'])})
    report.update({'script_sha256': sha(Path(__file__)), 'image_bound_revision': 2,
        'all_profiles_fit_conservative_context': all(row['conservative_context_fits_all'] for row in report['profiles']),
        'image_bound_correction': {'owner': '[codex]',
            'time': datetime.now(ZoneInfo('Asia/Taipei')).strftime('%Y-%m-%d %H:%M'),
            'reason': 'Independent actual-processor CPU probes found Qwen2 runtime image expansion exceeds the previous hard-coded 1280 bound; use frozen max_pixels1310720 and factor28 => ceil(1310720/784)=1672',
            'previous_audit_path': str(saved.relative_to(ROOT)), 'previous_audit_sha256': sha(saved),
            'previous_script_sha256': previous['script_sha256'],
            'corpus_or_queries_retokenized': False, 'all_tokenization_fields_unchanged': True,
            'unchanged_tokenization_fields_sha256': digest(unchanged),
            'independent_processor_evidence': probes, 'profile_changes': changes}})
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'owner': '[codex]', 'status': 'PASS', 'corpus_or_queries_retokenized': False,
                      'output': str(output), 'sha256': sha(output), 'changes': changes}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'dl/retrieval-matrix/data/mcmr-mmeb-v3-native-v1')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--refresh-image-bounds-from', type=Path,
                        help='Reuse preserved full tokenizer statistics; change only image bounds with explicit provenance')
    args = parser.parse_args()
    output = args.output or args.data / 'tokenlength-audit.json'
    if args.refresh_image_bounds_from:
        refresh_image_bounds(args.data, args.refresh_image_bounds_from, output)
    else:
        run(args.data, output)
