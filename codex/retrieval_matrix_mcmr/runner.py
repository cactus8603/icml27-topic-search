"""[codex] Independent native MCMR entry; complete product identities, no label admission."""
from __future__ import annotations

import argparse
import gc
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys
from time import monotonic

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from codex.retrieval_matrix import runner as R
from codex.retrieval_matrix_mcmr.adapters import NativeOfficialModel

OWN_SOURCES = tuple('codex/retrieval_matrix_mcmr/' + name for name in
                    ('runner.py', 'adapters.py', 'build_plan.py', 'run.sh', 'PREREG.md',
                     'SPEC.md', 'sources.json', 'prepare_official.py', 'token_audit.py', 'test_native.py'))


def rows(path):
    # Product metadata contains Unicode line separators inside valid strings.
    # ASCII LF, not str.splitlines(), is the JSONL record delimiter.
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def source_gate(plan_path, plan, freeze_commit):
    if not set(OWN_SOURCES) <= plan['source_files'].keys():
        raise ValueError('Native plan must bind all independently executing source files')
    result = R.source_gate(plan_path, plan, freeze_commit)
    scope = [str(HERE.relative_to(ROOT))]
    if any(spec.get('variant') in {'gme-qwen2', 'vlm2vec-v2-dora'} for spec in plan['models'].values()):
        scope.append('codex/retrieval_matrix_qwen2')
    status = subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain', '--', *scope], text=True)
    if status.strip():
        raise ValueError('Native MCMR/Qwen2 source directories must be clean before producing model outputs')
    return result


def load_plan(path):
    plan = R.load_plan(path)
    if plan.get('phase') != 'MCMR-native-MMEB-v2' or set(plan['datasets']) != {'mcmr'}:
        raise ValueError('This entry requires its separately frozen MCMR native plan')
    spec = plan['datasets']['mcmr']
    if spec.get('format') != 'mcmr-mmeb-native' or spec.get('expected_gallery') != 104981 or spec.get('expected_queries') != 3997:
        raise ValueError('Native MCMR must retain all original candidates and queries')
    if not {'raw_byte_identity', 'token_audit'} <= spec.keys():
        raise ValueError('Native production requires complete source-byte identity and CPU context evidence')
    if spec.get('checkpoint_product_chunk') != 64:
        raise ValueError('Native MCMR v2 fixes disk checkpoint chunks at 64 products; model batch remains 1')
    for model in plan['models'].values():
        if model['batches']['image'] != 1 or model['batches']['text'] != 32 or model['loader']['torch_dtype'] != 'bfloat16':
            raise ValueError('Native MCMR fixes fused batch1, query32, native BF16')
    return plan


def load_data(spec, *, verify_images=False):
    manifest_path = R.checked_file(spec['native_manifest'])
    manifest = R.read(manifest_path)
    if manifest['adapter_version'] != 'mcmr-mmeb-v3-native-v1' or manifest['corpus_admission_uses_qrel_or_labels'] is not False:
        raise ValueError('Only the fixed official MMEB native renderer is admitted')
    if 'raw_byte_identity' in spec:
        byte_identity = R.read(R.checked_file(spec['raw_byte_identity']))
        if byte_identity.get('archive_source_lfs_digest_verified') is not True:
            raise ValueError('MMEB source archive lacks full published SHA verification')
        for name in ('query.jsonl', 'candidates.jsonl'):
            row = byte_identity['metadata'][name]
            if row['exact_byte_match_with_original'] is not True or row['sha256'] != manifest['raw_input_sha256'][name]:
                raise ValueError('MMEB and original raw metadata byte identity is unproven')
    if 'token_audit' in spec:
        token_audit = R.read(R.checked_file(spec['token_audit']))
        if not token_audit.get('all_profiles_fit_conservative_context') or token_audit['native_manifest_sha256'] != R.sha(manifest_path):
            raise ValueError('Complete prepared native text lacks matching CPU context evidence')
    paths = {}
    for name in ('corpus', 'queries'):
        paths[name] = R.checked_file(spec[name])
        if R.sha(paths[name]) != manifest['outputs'][name + '.jsonl']:
            raise ValueError('Native data artifact differs from original conversion receipt')
    image_receipt_path = R.checked_file(spec['images_receipt'])
    if R.sha(image_receipt_path) != manifest['full_image_receipt_sha256']:
        raise ValueError('Original full image receipt differs from prepared native corpus')
    receipt_rows = rows(image_receipt_path)
    image_receipts = {row['file']: row for row in receipt_rows}
    if len(image_receipts) != len(receipt_rows):
        raise ValueError('Repeated source image receipt filename')
    image_root = R.safe(spec['images_dir'])
    gallery, seen_ids, checked_paths = [], set(), set()
    for row in rows(paths['corpus']):
        cid, text = str(row['id']), row['text']
        if not cid or cid in seen_ids or not isinstance(text, str) or not text.strip():
            raise ValueError('Empty or duplicated candidate identity/metadata')
        path = R.safe(row['image'])
        if not path.is_relative_to(image_root) or not path.is_file():
            raise ValueError('Missing or escaping candidate image path')
        receipt = image_receipts[path.name]
        if verify_images and path not in checked_paths:
            if R.sha(path) != receipt['sha256'] or path.stat().st_size != receipt['bytes']:
                raise ValueError('Candidate image differs from original archive receipt')
            checked_paths.add(path)
        # Verification may reuse a file hash, but product entries and embeddings
        # are never deduplicated by image path. Text stays on its own ID row.
        gallery.append({'id': cid, 'path': str(path), 'text': text,
                        'sha256': receipt['sha256'], 'bytes': receipt['bytes']})
        seen_ids.add(cid)
    queries, seen_queries = [], set()
    for row in rows(paths['queries']):
        qid, text = str(row['id']), row['text']
        if not qid or qid in seen_queries or not isinstance(text, str) or not text.strip():
            raise ValueError('Empty or duplicated native query')
        queries.append({'id': qid, 'text': text})
        seen_queries.add(qid)
    if len(gallery) != spec['expected_gallery'] or len(queries) != spec['expected_queries']:
        raise ValueError('Native MCMR original denominator changed')
    # Deliberately never open instances/qrel in the inference data loader.
    return {'gallery': gallery, 'queries': queries,
            'fingerprint': R.digest({'gallery': gallery, 'queries': queries, 'dataset': spec})}


def runtime_environment(spec):
    if spec.get('variant') not in {'gme-qwen2', 'vlm2vec-v2-dora'}:
        return R.runtime_environment(spec)
    import peft
    import flash_attn
    reduced = {**spec, 'environment': {key: val for key, val in spec.get('environment', {}).items() if key != 'peft'}}
    result = R.runtime_environment(reduced)
    result['peft'] = version('peft')
    result['modules']['peft'] = {'version': peft.__version__, 'path': str(Path(peft.__file__).resolve())}
    result['modules']['flash_attn'] = {'version': flash_attn.__version__, 'path': str(Path(flash_attn.__file__).resolve())}
    for key, expected in spec.get('environment', {}).items():
        if result.get(key) != expected:
            raise ValueError('Frozen native Qwen2 runtime differs: ' + key)
    return result


def smoke(out, spec, identity, settings):
    from PIL import Image
    import torch
    if (out / 'smoke.json').exists():
        R.require_smoke(out, identity)
        return
    model = NativeOfficialModel(spec)
    path = out / 'synthetic-product.png'
    Image.new('RGB', (224, 224), (80, 130, 170)).save(path)
    products = [{'id': 'synthetic-a', 'path': str(path), 'text': 'title: A blue synthetic object.'},
                {'id': 'synthetic-b', 'path': str(path), 'text': 'title: A second synthetic object.\nfeatures: Different longer synthetic metadata.'}]
    synthetic_queries = [{'id': str(i), 'text': 'A synthetic object.' if i == 0 else
                           'Find a synthetic object that is described by synthetic metadata.'}
                          for i in range(spec['batches']['text'])]
    fused = model.encode_candidates(products)
    text = model.encode_queries(synthetic_queries)
    if fused.ndim != 2 or text.ndim != 2 or fused.shape[1] != text.shape[1] or not np.isfinite(fused).all() or not np.isfinite(text).all():
        raise ValueError('Synthetic native fused/query encoding has invalid shape or values')
    single_fused = model.encode_candidates(products[:1])[0]
    single_text = model.encode_queries(synthetic_queries[:1])[0]
    invariance = {}
    for name, one, batch in [('image', single_fused, fused[0]), ('text', single_text, text[0])]:
        invariance[name] = {'max_abs_difference': float(np.max(np.abs(one - batch))),
                            'cosine': float(one @ batch / (np.linalg.norm(one) * np.linalg.norm(batch)))}
    # A separate batch1 synthetic input exercises the production image cap and
    # proves truncation=False does not silently enforce max_length=4096. The
    # real complete-corpus CPU bound is lower; no real candidate is inspected.
    large_path = out / 'synthetic-context-product.png'
    Image.new('RGB', (1024, 1280), (80, 130, 170)).save(large_path)
    long_text = 'title: ' + 'synthetic object ' * 1400
    with Image.open(large_path) as opened:
        probe = model.backend.processor(texts=[long_text], images=[opened], videos=[None],
            instruction=['Represent the given image and text.'], is_query=False,
            max_length=4096, truncation=False)
    untruncated_length = int(probe['attention_mask'][0].sum())
    del probe
    if untruncated_length <= 4096:
        raise ValueError('Synthetic full-context probe did not preserve tokens beyond the nominal max_length')
    context_value = model.encode_candidates([{'id': 'synthetic-context', 'path': str(large_path), 'text': long_text}])
    if context_value.shape != (1, fused.shape[1]) or not np.isfinite(context_value).all():
        raise ValueError('Synthetic batch1 full-context fused inference failed')
    passed = R.invariance_passed(invariance, settings)
    R.fixed_json(out / 'smoke.json', {'state': 'PASS' if passed else 'FAILED', 'technical_only': True,
        'research_metrics': False, 'identity': identity, 'signature': model.signature(),
        'batch1_vs_fixed_batch': invariance, 'image_semantics': 'fused product image+own metadata',
        'synthetic_fused_batch': 2, 'production_fused_batch': 1, 'text_batch': len(synthetic_queries),
        'synthetic_untruncated_fused_tokens': untruncated_length,
        'synthetic_context_probe_uses_real_product_data': False,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'peak_reserved_bytes': torch.cuda.max_memory_reserved()})
    if not passed:
        raise ValueError('Synthetic native batch invariance failed; no automatic fallback')


def top100_encode(query_vectors, corpus_vectors, batch, dtype):
    import torch
    if query_vectors.dtype != dtype or corpus_vectors.dtype != dtype:
        raise ValueError('Native similarity input dtype differs from frozen model dtype')
    scores = query_vectors[batch] @ corpus_vectors.T
    # Preserve the release's one-dimensional topk call shape. CUDA may select
    # different equal-score boundary members for a batched two-dimensional call.
    selected = [torch.topk(row, k=min(100, len(corpus_vectors))) for row in scores]
    indices = np.stack([value.indices.int().cpu().numpy() for value in selected])
    values = np.stack([value.values.float().cpu().numpy() for value in selected])
    return np.concatenate([indices.astype(np.float32), values], axis=1)


def encode_product_chunk(model, products):
    # Storage chunks amortize receipt IO; model inference stays exactly batch1.
    return np.concatenate([model.encode_candidates([product]) for product in products], axis=0)


def validate_index(out, identity, bundle):
    meta = R.read(out / 'index-meta.json')
    if meta.get('state') != 'COMPLETE' or meta['identity'] != identity:
        raise ValueError('Incomplete or different native index identity')
    for name in ('corpus', 'queries', 'native-top100'):
        if R.array_receipt(out / (name + '.npy')) != meta['arrays'][name]:
            raise ValueError('Native index checkpoint receipt differs: ' + name)
    if R.sha(out / 'processor.json') != meta['processor_sha256']:
        raise ValueError('Native processor signature changed')
    corpus = np.load(out / 'corpus.npy', mmap_mode='r', allow_pickle=False)
    queries = np.load(out / 'queries.npy', mmap_mode='r', allow_pickle=False)
    native = np.load(out / 'native-top100.npy', mmap_mode='r', allow_pickle=False)
    if (len(corpus), len(queries), native.shape) != (len(bundle['gallery']), len(bundle['queries']), (len(bundle['queries']), 200)):
        raise ValueError('Native complete-gallery/query/top100 array shapes differ')
    for values in (corpus, queries):
        if not np.allclose(np.linalg.norm(values, axis=1), 1, atol=1e-2):
            raise ValueError('Native embeddings are not normalized')
    indices = native[:, :100]
    if not np.array_equal(indices, indices.astype(np.int32)) or np.any(indices < 0) or np.any(indices >= len(corpus)):
        raise ValueError('Native top100 indices are invalid')
    if any(len(set(row.astype(np.int32))) != 100 for row in indices) or np.any(np.diff(native[:, 100:], axis=1) > 0):
        raise ValueError('Native top100 contains duplicates or unsorted scores')
    return native, meta


def index(out, spec, identity, bundle, settings, dataset_spec):
    if (out / 'index-meta.json').exists():
        validate_index(out, identity, bundle)
        return
    import torch
    began = monotonic()
    model = NativeOfficialModel(spec)
    load_seconds = monotonic() - began
    R.fixed_json(out / 'processor.json', model.signature())
    processor_sha = R.sha(out / 'processor.json')
    # Each checkpoint item includes the complete distinct product identity.
    R.checkpoint(out / 'corpus.npy', bundle['gallery'], dataset_spec['checkpoint_product_chunk'],
                 lambda products: encode_product_chunk(model, products),
                 {**identity, 'stage': 'fused-product', 'processor_sha256': processor_sha}, progress=out / 'progress.json')
    R.checkpoint(out / 'queries.npy', bundle['queries'], 32, model.encode_queries,
                 {**identity, 'stage': 'native-query', 'processor_sha256': processor_sha}, progress=out / 'progress.json')
    del model
    gc.collect()
    torch.cuda.empty_cache()
    dtype = getattr(torch, spec['loader']['torch_dtype'])
    corpus = torch.from_numpy(np.load(out / 'corpus.npy')).to('cuda', dtype=dtype)
    queries = torch.from_numpy(np.load(out / 'queries.npy')).to('cuda', dtype=dtype)
    positions = {row['id']: i for i, row in enumerate(bundle['queries'])}
    R.checkpoint(out / 'native-top100.npy', bundle['queries'], dataset_spec['similarity_batch'],
                 lambda batch: top100_encode(queries, corpus, [positions[row['id']] for row in batch], dtype),
                 {**identity, 'stage': 'native-bf16-top100', 'top_k': 100, 'torch_topk': True,
                  'similarity_batch': dataset_spec['similarity_batch']}, progress=out / 'progress.json')
    arrays = {name: R.array_receipt(out / (name + '.npy')) for name in ('corpus', 'queries', 'native-top100')}
    R.atomic_json(out / 'index-meta.json', {'state': 'COMPLETE', 'identity': identity,
        'arrays': arrays, 'processor_sha256': processor_sha, 'model_load_seconds_this_invocation': load_seconds,
        'index_wall_seconds_this_invocation': monotonic() - began,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
        'normalization_dtype': 'bfloat16', 'similarity_dtype': 'bfloat16',
        'ranking': 'native GPU torch.topk100 on full original product-order corpus; distinct metadata per product',
        'checkpoint_top100_layout': 'first100=float32-exact integer product indices; last100=native BF16 score values saved as float32'})


def ordinal_metrics(native, gallery_ids, labels):
    details = []
    for row, label in zip(native, labels, strict=True):
        relevant = set(label['positives'])
        ranked = [gallery_ids[int(i)] for i in row[:100]]
        rank = next((i + 1 for i, key in enumerate(ranked[:10]) if key in relevant), None)
        details.append({'id': label['id'], 'Hit@1': float(ranked[0] in relevant),
                        'Recall@1': float(ranked[0] in relevant) / len(relevant),
                        'MRR@10': 0. if rank is None else 1. / rank})
    return {name: float(np.mean([row[name] for row in details])) for name in ('Hit@1', 'Recall@1', 'MRR@10')}, details


def analyze(out, dataset_spec, bundle, identity):
    native, meta = validate_index(out, identity, bundle)
    labels_raw = rows(R.checked_file(dataset_spec['instances']))
    lookup = {str(row['qid']): list(dict.fromkeys(map(str, row['pos']))) for row in labels_raw}
    if len(lookup) != len(labels_raw) or set(lookup) != {row['id'] for row in bundle['queries']}:
        raise ValueError('Labels do not preserve every original query exactly')
    ids = [row['id'] for row in bundle['gallery']]
    id_set = set(ids)
    if any(not positives or not set(positives) <= id_set for positives in lookup.values()):
        raise ValueError('Empty or missing original positive identities')
    labels = [{'id': row['id'], 'positives': lookup[row['id']]} for row in bundle['queries']]
    ordinal, detail = ordinal_metrics(native, ids, labels)
    import pytrec_eval
    qrels = {row['id']: {key: 1 for key in row['positives']} for row in labels}
    predictions = {label['id']: {ids[int(i)]: float(score) for i, score in zip(row[:100], row[100:], strict=True)}
                   for label, row in zip(labels, native, strict=True)}
    predictions10 = {label['id']: {ids[int(i)]: float(score) for i, score in zip(row[:10], row[100:110], strict=True)}
                     for label, row in zip(labels, native, strict=True)}
    requested = {'recall_1', 'recall_5', 'recall_10', 'recall_50', 'recall_100', 'ndcg_cut_5', 'ndcg_cut_10', 'map_cut_10'}
    evaluated = pytrec_eval.RelevanceEvaluator(qrels, requested).evaluate(predictions)
    mrr10 = pytrec_eval.RelevanceEvaluator(qrels, {'recip_rank'}).evaluate(predictions10)
    if set(evaluated) != set(qrels) or set(mrr10) != set(qrels):
        raise ValueError('Evaluator changed the original 3997 query denominator')
    fields = sorted(next(iter(evaluated.values())))
    aggregate = {field: float(np.mean([row[field] for row in evaluated.values()])) for field in fields}
    aggregate['MRR@10'] = float(np.mean([row['recip_rank'] for row in mrr10.values()]))
    report = {'owner': '[codex]', 'model': identity['model'], 'queries': len(labels), 'gallery_products': len(ids),
        'primary_pytrec': aggregate, 'native_torch_ordinal_diagnostic': ordinal,
        'recall_semantics': 'relevant fraction, not Hit@1; pytrec may reorder native top100 score ties by document ID',
        'mrr10_semantics': 'pytrec reciprocal rank within native top10 admission; all original queries retained',
        'published_targets_for_difference_only': identity['model_spec'].get('published_targets'),
        'strict_CORE_P0_match_claim': False, 'reason': 'MMEB official native text; CORE original conversion unverified',
        'provenance': meta, 'cost_ledger': R.read(out / 'invocation-costs.json') if (out / 'invocation-costs.json').exists() else []}
    R.atomic_json(out / 'results/report.json', report)
    R.atomic_json(out / 'results/per-query.json', detail)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=('smoke', 'index', 'analyze'))
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--smoke-out', type=Path)
    parser.add_argument('--freeze-commit', required=True)
    args = parser.parse_args(argv)
    began = monotonic()
    plan, out = load_plan(args.plan), R.safe(args.out)
    spec, dataset_spec = plan['models'][args.model], plan['datasets']['mcmr']
    with R.writer(out):
        succeeded = False
        try:
            sources = source_gate(args.plan, plan, args.freeze_commit)
            identity = {'schema': 1, 'owner': '[codex]', 'model': args.model, 'dataset': 'mcmr',
                        'plan_sha256': R.sha(args.plan), 'source_files': sources, 'freeze_commit': args.freeze_commit,
                        'model_spec': spec, 'settings': plan['settings']}
            if args.stage == 'smoke':
                identity['environment'] = runtime_environment(spec)
                R.fixed_json(out / 'mode.json', {'technical_only': True})
                smoke(out, spec, identity, plan['settings'])
            else:
                bundle = load_data(dataset_spec, verify_images=args.stage == 'index')
                identity['dataset_fingerprint'] = bundle['fingerprint']
                if args.stage == 'index':
                    identity['environment'] = runtime_environment(spec)
                    R.fixed_json(out / 'mode.json', {'technical_only': False})
                    R.fixed_json(out / 'provenance.json', identity)
                    R.require_smoke(args.smoke_out, identity)
                    index(out, spec, identity, bundle, plan['settings'], dataset_spec)
                else:
                    previous = R.read(out / 'provenance.json')
                    for key in ('model', 'plan_sha256', 'source_files', 'freeze_commit', 'dataset_fingerprint', 'settings'):
                        if previous.get(key) != identity.get(key):
                            raise ValueError('Native analysis provenance differs: ' + key)
                    analyze(out, dataset_spec, bundle, previous)
            succeeded = True
            print(json.dumps({'owner': '[codex]', 'stage': args.stage, 'state': 'COMPLETE', 'time': R.now(), 'out': str(out)}), flush=True)
        except BaseException as error:
            R.atomic_json(out / ('failure-' + args.stage + '.json'), {'owner': '[codex]', 'time': R.now(),
                'state': 'FAILED', 'error_type': type(error).__name__, 'error': str(error), 'automatic_fallback': False})
            raise
        finally:
            ledger_path = out / 'invocation-costs.json'
            ledger = R.read(ledger_path) if ledger_path.exists() else []
            ledger.append({'stage': args.stage, 'time': R.now(), 'completed': succeeded,
                           'wall_seconds': monotonic() - began, 'includes_failed_and_resume_invocations': True,
                           'scope': 'entire invocation after argument parsing, including source/image verification, loading, compute and IO'})
            R.atomic_json(ledger_path, ledger)


if __name__ == '__main__':
    main()
