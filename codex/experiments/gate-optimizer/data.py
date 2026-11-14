import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, obj):
    path = Path(path)
    staging = path.with_suffix(path.suffix + ".partial")
    staging.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    staging.replace(path)


def prepare(config, output):
    import pyarrow.parquet as parquet
    from tokenizers import Tokenizer

    source = config["source"]
    vocab = config["model"]["vocab"]
    for key in ("parquet", "tokenizer"):
        if sha256(source[key]) != source[f"{key}_sha256"]:
            raise ValueError(f"unexpected input content: {key}")
    output.mkdir(parents=True, exist_ok=True)
    limits = source["token_limits"]
    manifest = output / "corpus.json"
    if manifest.exists():
        manifest_data = json.loads(manifest.read_text())
        if (
            manifest_data["source"] != source
            or manifest_data["vocab"] != vocab
            or any(sha256(output / f"{k}.bin") != v for k, v in manifest_data["sha256"].items())
        ):
            raise ValueError("existing corpus has different inputs or file hashes")
        return

    tokenizer = Tokenizer.from_file(source["tokenizer"])
    eos = tokenizer.token_to_id("<|endoftext|>")
    if tokenizer.get_vocab_size() != vocab or eos is None or not 0 < vocab < 65536:
        raise ValueError("tokenizer/vocabulary does not match config")
    groups = ("train", "dev", "probe", "test")
    counts = dict.fromkeys(groups, 0)
    docs = dict.fromkeys(groups, 0)
    seen = set()
    tagged = source["split_salt"].encode()
    handles = {k: (output / f"{k}.partial").open("wb") for k in groups}
    try:
        for batch in parquet.ParquetFile(source["parquet"]).iter_batches(batch_size=128, columns=["text"]):
            selected = []
            for document in batch.column(0).to_pylist():
                if not document or not document.strip():
                    continue
                document = document.strip()
                digest = hashlib.blake2b(tagged + document.encode(), digest_size=16).digest()
                if digest in seen:
                    continue
                seen.add(digest)
                index = int.from_bytes(digest[:4], "little") % 100
                group = "train" if index < 91 else groups[1 + (index - 91) // 3]
                if counts[group] < limits[group]:
                    selected.append((group, document))
            encoded = tokenizer.encode_batch([doc for _, doc in selected], add_special_tokens=False)
            for (group, _), tokens in zip(selected, encoded, strict=True):
                available = limits[group] - counts[group]
                if available <= 0:
                    continue
                ids = (tokens.ids + [eos])[:available]
                handles[group].write(np.asarray(ids, dtype="<u2").tobytes())
                counts[group] += len(ids)
                docs[group] += 1
            if counts == limits:
                break
    finally:
        for handle in handles.values():
            handle.close()
    if counts != limits:
        raise ValueError(f"insufficient local corpus tokens: {counts}")
    for group in groups:
        (output / f"{group}.partial").replace(output / f"{group}.bin")
    write_json(manifest, {
        "source": source, "vocab": vocab, "tokens": counts, "docs": docs,
        "sha256": {group: sha256(output / f"{group}.bin") for group in groups},
    })


class Windows:
    def __init__(self, path, sequence, width):
        self.tokens = np.memmap(path, dtype="<u2", mode="r")
        self.sequence = sequence
        self.windows_per_batch = width
        self.n = (len(self.tokens) - 1) // sequence
        if self.n < width:
            raise ValueError(f"insufficient tokens for one batch in {path}")

    def order(self, steps, seed):
        rng = np.random.default_rng(seed)
        count = steps * self.windows_per_batch
        order = np.concatenate([rng.permutation(self.n) for _ in range((count + self.n - 1) // self.n)])
        return order[:count].reshape(steps, self.windows_per_batch)

    def batch(self, rows, device):
        positions = rows[:, None] * self.sequence + np.arange(self.sequence + 1)
        tokens = torch.from_numpy(np.asarray(self.tokens[positions], dtype=np.int64)).to(device)
        return tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()


def synthetic(config, output):
    output.mkdir(parents=True, exist_ok=True)
    vocab = config["model"]["vocab"]
    for i, name in enumerate(("train", "dev", "probe", "test")):
        rng = np.random.default_rng(100 + i)
        rng.integers(0, vocab, size=6000, dtype=np.uint16).astype("<u2").tofile(output / f"{name}.bin")
