"""[codex] Separate, preregistered local continuation after published replay fails.

The historical replay is never changed to PASS. Cached embeddings and original
top50 pairs are reused only after input/runtime/digest checks; only ranks51-200
require fresh GPU inference. This producer does not evaluate any policy.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
from time import monotonic

import numpy as np

import cache_scores as legacy
from cache_v2 import atomic_json, batches, file_sha, fingerprint
from parity_probe import checked_array
from platform_context import runtime_environment, validate_context


def safe(path):
    path = Path(path).expanduser().resolve()
    if any('retrieval-independent-panel' in part for part in path.parts):
        raise ValueError('Sealed data are forbidden in a platform development run')
    return path


def committed_plan(path, commit):
    payload = subprocess.run(['git', 'show', f'{commit}:codex/retrieval3090/platform_plan.json'],
                             check=True, capture_output=True).stdout
    if payload != path.read_bytes():
        raise ValueError('Platform plan differs from the producing committed source')
    plan = json.loads(payload)
    if (plan.get('schema') != 'retrieval-platform-plan-v1'
            or plan.get('scope') != 'exposed-development-only'
            or plan.get('formal_evidence') is not False):
        raise ValueError('Unsupported platform plan')
    return plan


def verify_repeat(source, probe, arrays, plan):
    record = json.loads((probe / 'probe.json').read_text())
    if (record.get('status') != 'technical_probe_complete'
            or record.get('commit') != plan['source_cache_commit']
            or record.get('source') != plan['source'] or record.get('freeze') != plan['freeze']
            or record.get('probe_vectors_sha256') != file_sha(probe / 'probe_vectors.npz')
            or record['environment']['packages']['torch'] != plan['environment']['torch']):
        raise ValueError('The same-runtime fixed repeat probe differs from its frozen context')
    with np.load(probe / 'probe_vectors.npz', allow_pickle=False) as vectors:
        for name, count in (('core_embed_image', 16), ('core_embed_full', 32)):
            if not np.array_equal(vectors[name], arrays[name][:count]):
                raise ValueError(f'The fixed-batch repeat is not exact: {name}')
    return {'probe_json_sha256': file_sha(probe / 'probe.json'),
            'probe_vectors_sha256': file_sha(probe / 'probe_vectors.npz')}


def verify_reranker_repeat(scorer, queries, images, expected):
    fresh = np.concatenate([np.asarray(scorer.score([
        (queries[i], images[i]()) for i in range(start, start + 4)]), dtype=float)
        for start in (0, 4)])
    if fresh.shape != (8,) or not np.isfinite(fresh).all() or not np.array_equal(fresh, expected[:8]):
        raise ValueError('Source reranker prefix did not repeat exactly in the continuation runtime')
    return fresh


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-cache', type=Path, required=True)
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    source, probe, plan_path, out = map(safe, (args.source_cache, args.probe, args.plan, args.out))
    if os.environ.get('DEV_DIRTY') != '0' or source == out:
        raise ValueError('A separate committed platform run is required')
    commit = os.environ['DEV_COMMIT']
    plan = committed_plan(plan_path, commit)
    environment = runtime_environment()
    if environment != plan['environment']:
        raise ValueError(f'Current environment differs from the fixed platform: {environment}')
    with ExitStack() as stack:
        source_lock = stack.enter_context((source / 'writer.lock').open('r'))
        fcntl.flock(source_lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        out.mkdir(parents=True, exist_ok=True)
        writer = stack.enter_context((out / 'writer.lock').open('a+'))
        fcntl.flock(writer, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prior = json.loads((source / 'checkpoint-provenance.json').read_text())
        replay = json.loads((source / 'baseline-top50-replay.json').read_text())
        if replay.get('state') != 'MISMATCH':
            raise ValueError('This study activates only after the original torch214 replay fails')
        try:
            legacy.score.check_replay(replay['metrics'])
        except ValueError:
            pass
        else:
            raise ValueError('A passing historical replay cannot activate the separate study')
        panel, config, snapshots, context = legacy.score.preflight()
        if (prior['commit'] != plan['source_cache_commit'] or prior['dirty'] != '0'
                or prior['panel_sha256'] != plan['panel_sha256'] or prior['config'] != config
                or prior['model_config_sha256'] != plan['model_config_sha256']
                or prior['revisions'] != plan['model_revisions'] or prior['depth'] != plan['depth']
                or config['batch_size'] != plan['batch_size']
                or any(prior[key] != context[key] for key in
                       ('panel_sha256', 'source_commit', 'model_config_sha256', 'revisions'))):
            raise ValueError('Source checkpoint does not match the declared frozen platform')
        old_environment = dict(prior['execution_environment'])
        old_environment['open_clip_torch'] = old_environment.pop('open_clip')
        if any(environment.get(key) != value for key, value in old_environment.items()):
            raise ValueError('Source checkpoint runtime differs from this continuation')
        texts = [query['text'] for query in panel['queries']]
        conditions, condition_index = legacy.condition_texts(panel)
        arrays, source_files = {}, {}
        for name in ('core_embed', 'cs_clip'):
            for part, items, batch in (('image', panel['ids'], 8), ('full', texts, 32),
                                       ('condition', conditions, 32)):
                key = f'{name}_{part}'
                arrays[key], details = checked_array(source, key, prior, items=items, batch_size=batch,
                    stage_identity={'stage': key, 'order': items})
                if (arrays[key].ndim != 2 or arrays[key].shape[0] != len(items)
                        or not np.allclose(np.linalg.norm(arrays[key], axis=1), 1, atol=1e-5, rtol=0)):
                    raise ValueError('Source embeddings are not normalized')
                source_files[f'{key}.npy'] = {'sha256': details['sha256'], 'state_sha256': details['state_sha256']}
            dimensions = {arrays[f'{name}_{part}'].shape[1] for part in ('image', 'full', 'condition')}
            if len(dimensions) != 1 or next(iter(dimensions)) != {'core_embed': 2048, 'cs_clip': 512}[name]:
                raise ValueError('Source model embedding dimensions differ')
        repeat = verify_repeat(source, probe, arrays, plan)
        core = legacy.original.ranks(arrays['core_embed_full'] @ arrays['core_embed_image'].T)
        first_keys = [panel['ids'][int(col)] for col in core[:, :50].flat]
        first_queries = [text for text in texts for _ in range(50)]
        first, details = checked_array(source, 'cross_top50', prior, items=first_keys, batch_size=4,
            stage_identity={'stage': 'top50', 'queries': first_queries, 'images': first_keys})
        source_files['cross_top50.npy'] = {'sha256': details['sha256'], 'state_sha256': details['state_sha256']}
        identity = {'schema': 'retrieval-platform-continuation-v1', 'commit': commit,
                    'source_checkpoint_provenance_sha256': file_sha(source / 'checkpoint-provenance.json'),
                    'source_checkpoint_files': source_files,
                    'source_replay_sha256': file_sha(source / 'baseline-top50-replay.json'),
                    'plan_sha256': file_sha(plan_path), 'environment': environment,
                    'config': config, 'repeat': repeat}
        provenance = out / 'platform-provenance.json'
        if provenance.exists() and json.loads(provenance.read_text()) != identity:
            raise ValueError('Continuation identity changed; use its original committed source')
        atomic_json(provenance, identity)
        cp = out / 'checkpoints'
        cp.mkdir(exist_ok=True)

        def encode(name, cls, snapshot, settings, images, groups):
            if settings != config or list(groups[0]) != texts or list(groups[1]) != conditions:
                raise ValueError('Legacy encoding order changed')
            return arrays[f'{name}_image'], [arrays[f'{name}_full'], arrays[f'{name}_condition']], {
                'scope': 'verified same-runtime source cache; not fresh online timing',
                'source': str(source), 'items': len(images)}

        cross_call = 0
        reranker_repeat = {}
        def cross_scores(scorer, queries, images, batch_size):
            nonlocal cross_call
            keys = [int(opener.__defaults__[0]) for opener in images]
            if cross_call == 0:
                cross_call += 1
                if keys != first_keys or queries != first_queries or batch_size != 4:
                    raise ValueError('Original top50 cross-pair order changed')
                # The earlier probe scored the older environment's shortlist.
                # Repeat THIS source's actual first 8 pairs in original batches.
                fresh = verify_reranker_repeat(scorer, queries, images, first)
                reranker_repeat.update(pairs=8, batches=[4, 4], input_images=keys[:8],
                    input_queries=queries[:8], fresh_scores=fresh.tolist(), max_abs_difference=0.0)
                atomic_json(out / 'reranker-repeat.json', reranker_repeat)
                return np.asarray(first)
            if cross_call != 1 or batch_size != 4:
                raise ValueError('Unexpected continuation cross call')
            cross_call += 1
            expected = [panel['ids'][int(col)] for col in core[:, 50:200].flat]
            if keys != expected or queries != [text for text in texts for _ in range(150)]:
                raise ValueError('Rank51-200 cross-pair order changed')
            rows = list(zip(queries, images, strict=True))
            def compute(batch):
                return np.asarray(scorer.score([(text, opener()) for text, opener in batch]), dtype=float)
            return batches(cp / 'cross_rank51to200.npy', rows, batch_size, compute,
                           identity | {'stage': 'rank51to200', 'queries': queries, 'images': keys},
                           status=out / 'progress.json')

        legacy.encode, legacy.original.cross_scores = encode, cross_scores
        try:
            legacy.run(out, plan['depth'])
        except SystemExit as error:
            if not str(error).startswith('Baseline did not replay the published 26806 CORE numbers'):
                raise
        meta_path = out / 'meta.json'
        meta = json.loads(meta_path.read_text())
        if (meta['replay']['state'] != 'MISMATCH' or meta['replay']['core'] != replay['metrics']
                or cross_call != 2 or reranker_repeat.get('pairs') != 8):
            raise ValueError('Published failure/baseline identity changed during continuation')
        with np.load(out / 'vectors.npz', allow_pickle=False) as vectors:
            if set(vectors.files) != set(arrays) or any(not np.array_equal(vectors[key], value)
                                                      for key, value in arrays.items()):
                raise ValueError('Final vectors differ from the verified same-runtime source arrays')
        with np.load(out / 'cross.npz', allow_pickle=False) as scores:
            if (not np.array_equal(scores['core_order'], core)
                    or scores['cross'].shape != (len(texts), 200)
                    or not np.isfinite(scores['cross']).all()
                    or not np.array_equal(scores['cross'][:, :50].reshape(-1), first)):
                raise ValueError('Final continuation cross/cache alignment failed')
        meta.update(config=config, execution_environment=environment,
                    source_checkpoint_provenance_sha256=identity['source_checkpoint_provenance_sha256'],
                    platform_scope='platform-relative-exploration', formal_evidence=False)
        atomic_json(meta_path, meta)
        receipt = {'schema': 'retrieval-platform-receipt-v1', 'plan_sha256': identity['plan_sha256'],
                   'producer_commit': commit, 'published_reproduction': meta['replay'],
                   'platform_consistency': 'VALID_LOCAL', 'formal_evidence': False,
                   'environment': environment, 'source_checkpoint_files': source_files,
                   'reranker_repeat_sha256': file_sha(out / 'reranker-repeat.json'),
                   'source_checkpoint_provenance_sha256': identity['source_checkpoint_provenance_sha256'],
                   'checks': {key: True for key in ('complete_depth200', 'vector_shapes_finite_unit',
                       'core_order_recomputed', 'top50_source_identity', 'baseline_recomputed', 'fixed_sample_repeat')},
                   'outputs_sha256': {name: file_sha(out / name) for name in
                                      ('meta.json', 'panel.json', 'vectors.npz', 'cross.npz')}}
        receipt['published_reproduction'] = {'state': 'MISMATCH', 'metrics': meta['replay']['core']}
        receipt_path = out / 'platform-receipt.json'
        atomic_json(receipt_path, receipt)
        validate_context(out, plan_path, receipt_path)
        atomic_json(out / 'progress.json', {'stage': 'complete', 'scope': 'local-relative-only',
                    'published_reproduction': 'MISMATCH', 'formal_evidence': False})
        print('[codex] Local cache complete; published replay remains MISMATCH; no policy was evaluated', flush=True)


if __name__ == '__main__':
    main()
