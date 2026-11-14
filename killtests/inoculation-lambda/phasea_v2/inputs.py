"""Pin the local model, tokenizer and PPS inputs shared by all Phase A stages."""
import argparse
import json
import re
import shutil
from pathlib import Path

import yaml

from .estimator import file_hash


REQUIRED_MODEL = "Qwen/Qwen2.5-7B-Instruct"
OPTIONAL_MODELS = {"Qwen/Qwen2.5-14B-Instruct", "allenai/Olmo-3-7B-Instruct"}


def snapshot_evidence(root):
    """Identify HF weight blobs and hash the metadata/tokenizer files actually loaded."""
    if root.parent.name != "snapshots" or not re.fullmatch(r"[0-9a-f]{40}", root.name):
        raise ValueError(f"not an exact Hugging Face snapshot: {root}")
    index = root / "model.safetensors.index.json"
    if (root / "model.safetensors").is_file():
        weights = {"model.safetensors"}
    elif index.is_file():
        weights = set(json.loads(index.read_text())["weight_map"].values())
    else:
        raise ValueError(f"no safetensors model weights: {root}")
    if not weights or not all(Path(name).name == name and name.endswith(".safetensors") for name in weights):
        raise ValueError(f"invalid safetensors index: {index}")
    required = {"config.json", "tokenizer_config.json", "tokenizer.json"}
    if index.is_file():
        required.add(index.name)
    # Local fast tokenizers and model configuration may read any of these files.
    metadata = required | {path.name for path in root.iterdir() if path.suffix in {".json", ".txt", ".jinja", ".model"}}
    blobs = {}
    for name in sorted(weights):
        path = root / name
        target = path.resolve(strict=True)
        if not path.is_symlink() or target.parent != root.parent.parent / "blobs" or not re.fullmatch(r"[0-9a-f]{64}", target.name):
            raise ValueError(f"weight is not a content-addressed Hugging Face blob: {path}")
        size = target.stat().st_size
        if not target.is_file() or size == 0:
            raise ValueError(f"missing or empty weight blob: {path}")
        blobs[name] = {"blob_sha256": target.name, "size": size}
    return dict(snapshot=str(root), revision=root.name, weights=blobs,
                metadata_sha256={name: file_hash(root / name) for name in sorted(metadata)})


def resolve_snapshot(model):
    from huggingface_hub import snapshot_download

    root = Path(snapshot_download(model, local_files_only=True)).resolve()
    expected_cache = "models--" + model.replace("/", "--")
    if root.parent.parent.name != expected_cache:
        raise ValueError(f"snapshot does not belong to {model}: {root}")
    return snapshot_evidence(root)


def prepare(config_dir, manifest_path):
    """Create one manifest without replacement, copying each unique PPS vector once."""
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    configs, models, vector_sources = {}, {}, {}
    for path in sorted(config_dir.glob("*.yaml")):
        if not path.name.startswith(("grant_", "pg_")):
            continue
        cfg = yaml.safe_load(path.read_text())
        vectors = {d["name"]: Path(cfg["data_root"]) / d["vector"] for d in cfg["defenses"] if d["kind"] == "pps"}
        configs[path.name] = {"config_sha256": file_hash(path), "model": cfg["model"], "vectors": {}}
        vector_sources[path.name] = vectors
    if not configs or {c["model"] for c in configs.values()} != {REQUIRED_MODEL, *OPTIONAL_MODELS}:
        raise ValueError("unexpected model set in frozen Phase A configurations")
    for model in sorted({REQUIRED_MODEL, *OPTIONAL_MODELS}):
        try:
            models[model] = resolve_snapshot(model)
        except (OSError, ValueError, KeyError) as exc:
            if model == REQUIRED_MODEL:
                raise
            models[model] = {"unavailable": str(exc)}
    vector_dir = manifest_path.parent / "pps_v2"
    vector_dir.mkdir()
    copied = {}
    for config, vectors in vector_sources.items():
        for defense, source in vectors.items():
            if source not in copied:
                digest = file_hash(source)
                target = vector_dir / f"{digest}.pt"
                if not target.exists():
                    with source.open("rb") as original, target.open("xb") as stream:
                        shutil.copyfileobj(original, stream)
                    target.chmod(0o444)
                if file_hash(target) != digest:
                    raise ValueError(f"PPS vector changed during input capture: {source}")
                copied[source] = {"path": str(target.relative_to(manifest_path.parent)), "sha256": digest, "source": str(source)}
            configs[config]["vectors"][defense] = copied[source]
    vector_dir.chmod(0o555)
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump({"version": 1, "models": models, "configs": configs}, stream, indent=2, sort_keys=True)
    manifest_path.chmod(0o444)


def read_manifest(path, *, content=None):
    document = json.loads(path.read_bytes() if content is None else content)
    if document["version"] != 1:
        raise ValueError("unsupported Phase A input manifest version")
    return document


def inputs_for_config(document, manifest_path, config_path, model, *, verify=False):
    cfg = document["configs"][config_path.name]
    if cfg["config_sha256"] != file_hash(config_path) or cfg["model"] != model:
        raise ValueError(f"configuration differs from shared run input manifest: {config_path}")
    evidence = document["models"][model]
    if "unavailable" in evidence:
        raise ValueError(f"model unavailable in shared run input manifest: {model}")
    snapshot = Path(evidence["snapshot"])
    vectors = {name: manifest_path.parent / row["path"] for name, row in cfg["vectors"].items()}
    hashes = {name: row["sha256"] for name, row in cfg["vectors"].items()}
    if verify:
        if snapshot_evidence(snapshot) != evidence:
            raise ValueError(f"model or tokenizer snapshot differs from shared run input manifest: {model}")
        for name, path in vectors.items():
            if file_hash(path) != hashes[name]:
                raise ValueError(f"PPS vector differs from shared run input manifest: {name}")
    return snapshot, vectors, {"model_snapshot": str(snapshot), "tokenizer_snapshot": str(snapshot),
                               "revision": evidence["revision"], "pps_sha256": hashes}


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("prepare")
    create.add_argument("config_dir", type=Path)
    create.add_argument("manifest", type=Path)
    check = commands.add_parser("available")
    check.add_argument("manifest", type=Path)
    check.add_argument("model")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.config_dir, args.manifest)
    else:
        model = read_manifest(args.manifest)["models"][args.model]
        if "unavailable" in model:
            parser.exit(1, f"SKIP {args.model}: {model['unavailable']}\n")


if __name__ == "__main__":
    main()
