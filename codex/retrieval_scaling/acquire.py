"""[codex] Acquire checksum-verified public CORE 8B checkpoints only."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'codex/retrieval3090'))
import download_assets as D

PINS = {
    'embed8': ('Alibaba-NLP/core-emb-8b', '123aef9e5e9e76fb686420f077fce3ef3b580c2b'),
    'cross8': ('Alibaba-NLP/core-reranker-8b', 'd95d36a34300e34b87cc1793827ac798642c4488'),
}
SMALL = ('config.json', 'chat_template.jinja', 'tokenizer.json', 'tokenizer_config.json',
         'preprocessor_config.json', 'video_preprocessor_config.json', 'special_tokens_map.json',
         'added_tokens.json', 'merges.txt', 'vocab.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, default=ROOT / 'dl/retrieval3090/scaling-assets')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--metadata-only', action='store_true')
    args = parser.parse_args()
    args.runtime.mkdir(parents=True, exist_ok=True)
    receipt = args.runtime / 'public-download-receipt.json'
    items, models = [], {}
    for key, (repo, revision) in PINS.items():
        weight_names = ([f'pytorch_model-{i:05d}-of-00004.bin' for i in range(1, 5)]
                        + ['pytorch_model.bin.index.json'] if key == 'embed8' else
                        [f'model-{i:05d}-of-00004.safetensors' for i in range(1, 5)]
                        + ['model.safetensors.index.json', 'generation_config.json'])
        model_items = list(D.hf_items('models', repo, revision, [*SMALL, *weight_names], D.HF_HOME))
        items.extend(model_items)
        models[key] = {'repo': repo, 'revision': revision, 'license': 'cc-by-4.0',
                       'snapshot': str(Path(model_items[0]['path']).parent),
                       'files': model_items, 'bytes': sum(row['expected_size'] for row in model_items)}
    metadata = {'owner': '[codex]', 'scope': 'public official CORE8B models; no sealed data',
                'models': models, 'bytes': sum(row['expected_size'] for row in items)}
    (args.runtime / 'asset-manifest.json').write_text(json.dumps(metadata, indent=2, sort_keys=True) + '\n')
    D.log(f"Pinned {len(items)} files, {metadata['bytes'] / 2**30:.3f} GiB")
    if args.metadata_only:
        return
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(D.download, item, receipt) for item in items]
        for future in futures:
            future.result()
    D.log('All CORE8B files passed size and checksum validation')


if __name__ == '__main__':
    main()
