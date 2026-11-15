"""Check the exact offline snapshot resolved by Hugging Face's default revision."""
import json
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


def cached_weights(model):
    root = Path(snapshot_download(model, local_files_only=True))
    if not (root / "config.json").is_file() or not (root / "tokenizer_config.json").is_file():
        return False
    index = root / "model.safetensors.index.json"
    if not index.is_file():
        return (root / "model.safetensors").is_file()
    shards = set(json.loads(index.read_text())["weight_map"].values())
    return bool(shards) and all((root / shard).is_file() for shard in shards)


if __name__ == "__main__":
    try:
        sys.exit(0 if cached_weights(sys.argv[1]) else 1)
    except (OSError, ValueError, KeyError):
        sys.exit(1)
