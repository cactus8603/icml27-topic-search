"""[codex] Technical two-image model check; no research metrics or sealed data."""
import argparse
import gc
from importlib.metadata import version
import json
import os
from pathlib import Path
from time import monotonic

import numpy as np
from PIL import Image
import torch
import cache_scores as cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit('A CUDA GPU with BF16 is required')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise FileExistsError(args.out)
    config, snapshots = cache.original.models()
    paths = sorted((Path(os.environ['RETRIEVAL_ROOT']) / 'data/coco2017-anchor-rerank/images').glob('*.jpg'))[:2]
    if len(paths) == 2:
        images = [cache.original.rgb(path) for path in paths]
        image_source = [str(path) for path in paths]
    else:
        images = [Image.new('RGB', (224, 224), color) for color in ('red', 'blue')]
        image_source = 'synthetic solid colors, technical execution only'
    captions = ['An image containing a red object.', 'An image containing a blue object.']
    phases = {}
    for name, model_cls in [('core_embed', cache.original.Encoder), ('cs_clip', cache.original.ClipEncoder),
                            ('core_reranker', cache.original.CrossScorer)]:
        print(f'[codex] Technical smoke: loading {name}', flush=True)
        torch.cuda.reset_peak_memory_stats()
        started = monotonic()
        model = model_cls(name, snapshots[name], config, 'cuda')
        if name == 'core_reranker':
            result = model.score(list(zip(captions, images, strict=True)))
            if result.shape != (2,) or not np.isfinite(result).all():
                raise ValueError('Invalid cross-encoder output')
            shapes = {'scores': list(result.shape)}
        else:
            vi = model.encode(images, images=True)
            vt = model.encode(captions)
            if vi.shape != vt.shape or len(vi) != 2 or not np.isfinite(vi).all() or not np.isfinite(vt).all():
                raise ValueError('Invalid encoder output')
            np.testing.assert_allclose(np.linalg.norm(vi, axis=1), 1, atol=1e-5)
            np.testing.assert_allclose(np.linalg.norm(vt, axis=1), 1, atol=1e-5)
            shapes = {'image_vectors': list(vi.shape), 'text_vectors': list(vt.shape)}
        phases[name] = cache.score.timeline(torch, started, 2, 'technical_inputs') | shapes
        del model
        gc.collect()
        torch.cuda.empty_cache()
        print(f'[codex] Technical smoke: {name} passed', flush=True)
    report = {
        'status': 'PASS', 'scope': 'technical environment validation only, not baseline replay or research evidence',
        'device': torch.cuda.get_device_name(0), 'cuda': torch.version.cuda, 'bf16': True,
        'versions': {name: version(name) for name in ['torch', 'torchvision', 'transformers', 'open_clip_torch', 'numpy', 'Pillow']},
        'source': os.environ['DEV_SOURCE'], 'freeze': os.environ['DEV_FREEZE'],
        'commit': os.environ['DEV_COMMIT'], 'dirty': os.environ['DEV_DIRTY'],
        'revisions': {name: path.name for name, path in snapshots.items()},
        'image_source': image_source, 'phases': phases,
    }
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
