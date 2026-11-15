"""[codex] Resumable, batch-identical wrapper for the frozen development cache.

The scoring implementation, order, and batch composition are unchanged. A checkpoint
is reusable only with the same committed source, panel, models, depth and inputs.
"""
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


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + '\n')
    temporary.replace(path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def batches(path, items, batch_size, compute, identity, *, status=None):
    """Commit completed batches, never a partially written batch; retry it after failure."""
    state_path = path.with_suffix('.state.json')
    expected = fingerprint({'identity': identity, 'items': len(items), 'batch_size': batch_size})
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        'fingerprint': expected, 'completed': 0, 'active_seconds': 0.0, 'batches': []}
    if state['fingerprint'] != expected:
        raise ValueError(f'Checkpoint provenance differs: {path}')
    done = state['completed']
    result = np.load(path, mmap_mode='r+') if path.exists() else None
    if done and (result is None or result.shape[0] != len(items)):
        raise ValueError('Incomplete checkpoint metadata')
    if done == len(items):
        print(f'[codex] Reusing completed {path.name}', flush=True)
        return np.asarray(result)
    for start in range(done, len(items), batch_size):
        end = min(start + batch_size, len(items))
        began = monotonic()
        value = np.asarray(compute(items[start:end]))
        if value.shape[0] != end - start or not np.isfinite(value).all():
            raise ValueError(f'Invalid model output at {path.name}:{start}')
        if result is None:
            result = np.lib.format.open_memmap(path, mode='w+', dtype=value.dtype,
                                              shape=(len(items), *value.shape[1:]))
        if result.shape[1:] != value.shape[1:]:
            raise ValueError('Model output shape changed')
        result[start:end] = value
        result.flush()
        elapsed = monotonic() - began
        state['completed'] = end
        state['active_seconds'] += elapsed
        state['batches'].append({'start': start, 'end': end, 'seconds': elapsed})
        atomic_json(state_path, state)
        progress = {'stage': path.stem, 'completed': end, 'total': len(items),
                    'active_seconds': round(state['active_seconds'], 3)}
        if status:
            atomic_json(status, progress)
        if start == done or end == len(items) or end // batch_size % 25 == 0:
            print(json.dumps(progress), flush=True)
    return np.asarray(result)


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--depth', type=int, default=200)
    args = parser.parse_args()
    if args.depth != 200:
        parser.error('This frozen development protocol requires depth 200')
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / 'writer.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        panel, config, snapshots, context = legacy.score.preflight()
        identity = context | {'depth': args.depth, 'commit': os.environ['DEV_COMMIT'],
                              'dirty': os.environ['DEV_DIRTY'], 'config': config}
        manifest = args.out / 'checkpoint-provenance.json'
        if manifest.exists() and json.loads(manifest.read_text()) != json.loads(json.dumps(identity, default=str)):
            raise ValueError('Refusing to mix runs from different sources')
        atomic_json(manifest, identity)
        checkpoint = args.out / 'checkpoints'
        checkpoint.mkdir(exist_ok=True)
        status = args.out / 'progress.json'

        def encode(name, model_class, snapshot, settings, images, groups):
            paths = [checkpoint / f'{name}_{part}.npy' for part in ('image', 'full', 'condition')]
            torch.cuda.reset_peak_memory_stats()
            began = monotonic()
            # Loading even when cached preserves the simple legacy timing structure;
            # per-stage active compute is separately recorded in the checkpoint states.
            model = model_class(name, snapshot, settings, 'cuda')
            result = []
            for path, items, is_image in zip(paths, (images, *groups), (True, False, False), strict=True):
                batch = settings['batch_size']['embed_image' if is_image else 'embed_text']
                def compute(rows, flag=is_image):
                    return model.encode([opener() for opener in rows] if flag else rows, images=flag)
                item_identity = identity | {'stage': path.stem,
                                           'order': panel['ids'] if is_image else items}
                result.append(batches(path, items, batch, compute, item_identity, status=status))
            timing = legacy.score.timeline(torch, began, len(images), 'indexed_images')
            del model
            gc.collect()
            torch.cuda.empty_cache()
            return result[0], result[1:], timing

        cross_call = 0
        def cross_scores(scorer, queries, images, batch_size):
            nonlocal cross_call
            part = ('top50', 'rank51to200')[cross_call]
            cross_call += 1
            keys = [int(opener.__defaults__[0]) for opener in images]
            rows = list(zip(queries, images, strict=True))
            def compute(batch):
                # Legacy cross_scores converts the scalar array to float64.
                return np.asarray(scorer.score([(text, opener()) for text, opener in batch]), dtype=float)
            return batches(checkpoint / f'cross_{part}.npy', rows, batch_size, compute,
                           identity | {'stage': part, 'queries': queries, 'images': keys}, status=status)

        legacy.encode = encode
        legacy.original.cross_scores = cross_scores
        legacy.run(args.out, args.depth)
        atomic_json(status, {'stage': 'complete', 'completed': len(panel['queries']) * args.depth,
                             'total': len(panel['queries']) * args.depth})


if __name__ == '__main__':
    main()
