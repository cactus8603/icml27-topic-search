"""Pinned public parquet reader and synthetic fixtures."""

from dataclasses import dataclass
import hashlib
from pathlib import Path
import random
import urllib.request


REPO = "https://huggingface.co/datasets/Jacklu0831/llm-verification-raw"


@dataclass(frozen=True)
class Candidate:
    dataset: str
    solver_model: str
    group: str
    weak: int
    strong: int
    gold: int


def digest(path):
    with path.open("rb") as data:
        return hashlib.file_digest(data, "sha256").hexdigest()


def fetch(config, directory):
    """Write only under the explicit shared data directory chosen by the caller."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, sha in config["dataset_shards"].items():
        destination = directory / name
        if destination.is_file() and digest(destination) == sha:
            continue
        pending = directory / f".{name}.incoming"
        url = f"{REPO}/resolve/{config['dataset_revision']}/data/verifier_samples/{name}?download=true"
        request = urllib.request.Request(url, headers={"Accept-Encoding": "identity"})
        with pending.open("xb") as target:
            try:
                with urllib.request.urlopen(request, timeout=60) as source:
                    magic = source.read(4)
                    if source.status != 200 or magic != b"PAR1":
                        raise ValueError(f"expected a complete Parquet response for {name}")
                    target.write(magic)
                    while chunk := source.read(2**20):
                        target.write(chunk)
                if digest(pending) != sha:
                    raise ValueError(f"checksum mismatch for {name}")
            except BaseException:
                pending.unlink(missing_ok=True)
                raise
        pending.replace(destination)


def boolean(value, name):
    if not isinstance(value, bool):
        raise ValueError(f"unexpected {name} type: {type(value).__name__}")
    return int(value)


def load_pairs(config, directory):
    import pyarrow.parquet as pq

    keys = ("dataset", "question", "solver_model", "solver_output_sha256")
    fields = [*keys, "solver_correct", "verifier_model", "verifier_verdict"]
    models = {config["weak_model"], *config["strong_models"]}
    joined = {}
    for name, sha in config["dataset_shards"].items():
        path = directory / name
        if digest(path) != sha:
            raise ValueError(f"checksum mismatch for {name}")
        for batch in pq.ParquetFile(path).iter_batches(columns=fields, batch_size=4096):
            for row in batch.to_pylist():
                model = row["verifier_model"]
                if model not in models:
                    continue
                key = tuple(row[field] for field in keys)
                if any(not isinstance(value, str) for value in key):
                    raise ValueError("incomplete candidate key")
                gold = boolean(row["solver_correct"], "solver_correct")
                previous_gold, verdicts = joined.setdefault(key, (gold, {}))
                if previous_gold != gold or model in verdicts:
                    raise ValueError("candidate labels disagree or a verifier row is duplicated")
                value = row["verifier_verdict"]
                verdicts[model] = None if value is None else boolean(value, "verifier_verdict")

    result = {}
    for model in config["strong_models"]:
        pairs = []
        missing = 0
        for key, (gold, verdicts) in sorted(joined.items()):
            dataset, question, solver_model, _ = key
            weak = verdicts.get(config["weak_model"])
            strong = verdicts.get(model)
            if weak is None or strong is None:
                missing += 1
                continue
            group = hashlib.sha256(question.encode()).hexdigest()
            pairs.append(Candidate(dataset, solver_model, group, weak, strong, gold))
        if not pairs:
            raise ValueError(f"no complete pairs for {model}")
        result[model] = (pairs, missing)
    return result


def synthetic(config):
    rng = random.Random(20261114)
    result = {}
    for model_id, model in enumerate(config["strong_models"]):
        pairs = []
        for dataset in ("gsm", "aime", "dummy"):
            for question in range(15):
                for attempt in range(8):
                    gold = rng.randrange(2)
                    both_wrong = rng.random() < 0.10
                    weak = gold ^ int(both_wrong or rng.random() < 0.24)
                    strong = gold ^ int(both_wrong or rng.random() < (0.04 + 0.08 * model_id))
                    solver = f"toy-{attempt % 2}"
                    pairs.append(Candidate(dataset, solver, f"{dataset}-{question}", weak, strong, gold))
        result[model] = (pairs, 0)
    return result
