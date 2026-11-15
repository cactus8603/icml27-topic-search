"""Freeze and evaluate author-labelled opposite relations in What'sUp Controlled Images B."""
import argparse
from dataclasses import dataclass
import hashlib
from importlib.metadata import version
import io
from itertools import combinations
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tarfile

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import COCO_DIR, EXP, ICML, jdump, paired_bootstrap_diff
from headroom import CONFIG as MODEL_CONFIG, data_path, encode_items, model_dir, rgb
from headroom_models import ClipEncoder, Encoder
from prospective_manifest import attach_shuffles, digest, file_sha
from prospective_rank import increment


RELEASE = ICML / "data/whatsup-controlled-b"
MANIFEST = EXP / "configs/controlled_spatial_manifest.json"
INPUTS = {
    "controlled_clevr_dataset.json": (86532, "2c0df1ab59d13d71442752b4b3c7353114d712a785b0850ec22e11c2b0c474c9"),
    "controlled_clevr.tar.gz": (65817200, "df8b23af7dc73c4bf98937777b7a5d85aab42598eace82fa7adad5802df2c493"),
}
CAPTION = re.compile(r"^A ([a-z]+) (to the left of|to the right of|in front of|behind) a ([a-z]+)$")
OPPOSITE = {"to the left of": "to the right of", "to the right of": "to the left of",
            "in front of": "behind", "behind": "in front of"}
RELATIONS = tuple(OPPOSITE)
WEIGHT = 0.125
BOOTSTRAP_SEED = 2026111512
SHUFFLE_SEED = 2026111511
BOOTSTRAP_DRAWS = 10000
NEAR_DISTANCE = 6
NUM_RELEASE_GROUPS = 102
NUM_COCO = 5000
RELEASE_BYTES_MAX = 96 * 1024**2
IMAGE_BYTES_MAX = 2 * 1024**2
MODEL_NAMES = ("cs_clip", "core_embed")
MODEL_CONFIG_SHA = "862591c60c5609f16727d902993df0cba9cab96ded72611d69d263a02fe4ad1d"
ANGLES = np.pi / 32 * np.outer(np.arange(8), np.arange(32) + 0.5)
DCT = np.sqrt(2 / 32) * np.cos(ANGLES)
DCT[0] /= np.sqrt(2)


@dataclass(frozen=True)
class Label:
    filename: str
    caption: str


def read_groups(rows, expected_groups):
    groups = {}
    filenames = set()
    object_pairs = set()
    for row in rows:
        captions = row["caption_options"]
        parsed = [CAPTION.fullmatch(caption) for caption in captions]
        if len(parsed) != len(RELATIONS) or not all(parsed):
            raise ValueError("Expected four labelled relation captions")
        subject, relation, obj = parsed[0].groups()
        if {match.groups() for match in parsed} != {(subject, r, obj) for r in RELATIONS}:
            raise ValueError("Caption options must describe the same two objects in all four relations")
        relation_token = relation.removeprefix("to the ").replace("in front", "in-front").replace(" ", "_")
        filename = f"{subject}_{relation_token}_{obj}.jpeg"
        if row["image_path"] != f"data/controlled_clevr/{filename}" or filename in filenames:
            raise ValueError(f"Invalid or repeated release annotation: {row['image_path']}")
        objects = (subject, obj)
        if objects not in groups:
            if subject == obj or frozenset(objects) in object_pairs:
                raise ValueError(f"Duplicate or invalid object pair: {objects}")
            object_pairs.add(frozenset(objects))
            groups[objects] = {}
        groups[objects][relation] = Label(filename, captions[0])
        filenames.add(filename)
    if len(groups) != expected_groups or any(group.keys() != OPPOSITE.keys() for group in groups.values()):
        raise ValueError("Release has a missing or incomplete relation quartet")
    return dict(sorted(groups.items()))


def read_release():
    for name, (size, sha) in INPUTS.items():
        path = RELEASE / name
        if path.stat().st_size != size or file_sha(path) != sha:
            raise ValueError(f"Unexpected release bytes: {path}")
    rows = json.loads((RELEASE / "controlled_clevr_dataset.json").read_text())
    groups = read_groups(rows, NUM_RELEASE_GROUPS)
    expected = {label.filename for group in groups.values() for label in group.values()}
    images = {}
    total_bytes = 0
    with tarfile.open(RELEASE / "controlled_clevr.tar.gz", "r|gz") as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            if member.isdir() and path.as_posix() == "controlled_clevr":
                continue
            if (not member.isfile() or path.parts != ("controlled_clevr", path.name)
                    or path.name not in expected or path.name in images):
                raise ValueError(f"Unexpected or repeated release archive member: {path}")
            total_bytes += member.size
            if not 0 < member.size <= IMAGE_BYTES_MAX or total_bytes > RELEASE_BYTES_MAX:
                raise ValueError("Uncompressed image size exceeds the release budget")
            with archive.extractfile(member) as image:
                images[path.name] = image.read()
    if images.keys() != expected:
        raise ValueError("Release images and annotations differ")
    return groups, images


@dataclass(frozen=True)
class Fingerprint:
    sha256: str
    perceptual: int
    difference: int


def fingerprint(data):
    with Image.open(io.BytesIO(data)) as image:
        gray = image.convert("L")
        small = np.asarray(gray.resize((32, 32), Image.Resampling.LANCZOS), dtype=float)
        frequencies = (DCT @ small @ DCT.T).ravel()[1:]  # leave the DC term out
        adjacent = np.asarray(gray.resize((9, 8), Image.Resampling.LANCZOS))
    phash = np.packbits(frequencies > np.median(frequencies))
    dhash = np.packbits(adjacent[:, 1:] > adjacent[:, :-1])
    return Fingerprint(hashlib.sha256(data).hexdigest(), int.from_bytes(phash), int.from_bytes(dhash))


def near_duplicate(left, right):
    return ((left.perceptual ^ right.perceptual).bit_count() <= NEAR_DISTANCE
            and (left.difference ^ right.difference).bit_count() <= NEAR_DISTANCE)


def prior_fingerprints(data_config):
    import pyarrow.parquet as pq

    coco = {}
    locations = sorted((COCO_DIR / "val2017").glob("*.jpg"))
    if len(locations) != NUM_COCO or not all(re.fullmatch(r"\d{12}\.jpg", path.name) for path in locations):
        raise ValueError("Incomplete COCO val2017 overlap screen")
    for path in locations:
        coco[path.name] = fingerprint(path.read_bytes())
    ht = {}
    manifest = {"COCO_val2017": {"count": len(coco),
                "content_digest": digest({name: item.sha256 for name, item in coco.items()})},
                "Half-Truths": {}}
    for subset, count in data_config["subset_sizes"].items():
        path = data_path(data_config, subset)
        parquet = pq.ParquetFile(path)
        if parquet.metadata.num_rows != count:
            raise ValueError(f"Unexpected Half-Truths row count in {subset}")
        i = 0
        for batch in parquet.iter_batches(columns=["image", "image_sha256"], batch_size=16):
            for row in batch.to_pylist():
                item = fingerprint(row["image"]["bytes"])
                if item.sha256 != row["image_sha256"]:
                    raise ValueError(f"Encoded image and declared hash differ: {subset} row {i}")
                ht[f"{subset}:{i}"] = item
                i += 1
        if i != count:
            raise ValueError(f"Incomplete Half-Truths overlap screen for {subset}")
        manifest["Half-Truths"][subset] = {"count": i, "file_sha256": file_sha(path)}
    return {"COCO_val2017": coco, "Half-Truths": ht}, manifest


def screen_groups(groups, release, prior):
    """Exclude exact matches and stop for adjudication on uncertain external overlap."""
    reasons = {}
    for objects, group in groups.items():
        for label in group.values():
            item = release[label.filename]
            for dataset, candidates in prior.items():
                for key, candidate in candidates.items():
                    if item.sha256 == candidate.sha256:
                        reasons.setdefault(objects, []).append(
                            [label.filename, dataset, key, "exact"])
                    elif near_duplicate(item, candidate):
                        raise ValueError(f"External near-duplicate candidate: {label.filename}, {dataset} {key}")
    files = [(objects, label.filename) for objects, group in groups.items() for label in group.values()]
    for (a, fa), (b, fb) in combinations(files, 2):
        if release[fa].sha256 == release[fb].sha256:
            reasons.setdefault(a, []).append([fa, "Controlled Images B", fb, "exact"])
            if a != b:
                reasons.setdefault(b, []).append([fb, "Controlled Images B", fa, "exact"])
    return reasons


def anchor_assignment(group_keys):
    queries = [{"k": 2, "key": objects, "anchor_key": objects,
                "anchor": f"a {objects[0]} and a {objects[1]}"} for objects in group_keys]
    assigned = attach_shuffles(queries, SHUFFLE_SEED)
    donors = [item["shuffled_donor_key"] for item in assigned]
    if any(donor is None for donor in donors) or len(set(donors)) != len(queries):
        raise ValueError("A unique, object-disjoint shuffled anchor is required for each quartet")
    return {item["key"]: (item["anchor"], item["shuffled_anchor"], item["shuffled_donor_key"])
            for item in assigned}


def make_manifest(groups, images, models):
    hashes = {name: fingerprint(data) for name, data in images.items()}
    prior, prior_manifest = prior_fingerprints(models["data"])
    reasons = screen_groups(groups, hashes, prior)
    included = [key for key in groups if key not in reasons]
    if len(included) < 2:
        raise ValueError("Overlap screening left too few complete quartets")
    donors = anchor_assignment(included)
    return {
        "source": "What’s Up With Vision-Language Models? Controlled Images B official release",
        "release": {name: {"bytes": size, "sha256": sha} for name, (size, sha) in INPUTS.items()},
        "model_config_sha256": MODEL_CONFIG_SHA,
        "overlap_screen": {"prior_inputs": prior_manifest,
                           "external_hashes": "SHA-256, 63-bit DCT pHash without DC, 64-bit horizontal dHash",
                           "external_max_hamming_in_each_near_hash": NEAR_DISTANCE,
                           "internal_hash": "SHA-256",
                           "excluded_groups": [{"objects": list(key), "matches": sorted(reasons[key])}
                                               for key in sorted(reasons)]},
        "weight": WEIGHT,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "shuffle_seed": SHUFFLE_SEED,
        "groups": [{"objects": list(key), "shuffled_donor": list(donors[key][2]),
                    "image_hashes": [hashes[groups[key][r].filename].sha256 for r in RELATIONS]}
                   for key in included],
        "relation_order": list(RELATIONS),
    }


def margins(encoder, groups, images, objects, donors, config, *, with_anchors):
    labels = [groups[key][r] for key in objects for r in RELATIONS]
    inputs = [lambda name=label.filename: rgb(io.BytesIO(images[name])) for label in labels]
    texts = [label.caption for label in labels]
    if with_anchors:
        texts.extend(donors[key][0] for key in objects)
        texts.extend(donors[key][1] for key in objects)
    iv, tv = encode_items(encoder, inputs, texts, config)
    n = len(RELATIONS)
    image_vectors = iv.reshape(len(objects), n, -1)
    delta = image_vectors - image_vectors[:, [RELATIONS.index(OPPOSITE[r]) for r in RELATIONS]]
    raw = np.einsum("grd,grd->gr", tv[:len(labels)].reshape(delta.shape), delta)
    result = {"plain": raw}
    if with_anchors:
        true, shuffled = np.split(tv[len(labels):], 2)
        result["true_anchor"] = increment(raw, np.einsum("grd,gd->gr", delta, true), WEIGHT)
        result["shuffled_anchor"] = increment(raw, np.einsum("grd,gd->gr", delta, shuffled), WEIGHT)
    return result


def summarize_scores(matrix):
    expected_shape = (len(matrix["plain"]), len(RELATIONS))
    if not all(values.shape == expected_shape and values.size and np.isfinite(values).all()
               for values in matrix.values()):
        raise ValueError("All arms require four finite directional margins per quartet")
    wins = {name: (values > 0) + 0.5 * (values == 0) for name, values in matrix.items()}
    group_scores = {name: values.mean(axis=1) for name, values in wins.items()}
    primary = paired_bootstrap_diff(group_scores["true_anchor"], group_scores["shuffled_anchor"],
                                    BOOTSTRAP_DRAWS, BOOTSTRAP_SEED)
    descriptive = paired_bootstrap_diff(group_scores["true_anchor"], group_scores["plain"],
                                        BOOTSTRAP_DRAWS, BOOTSTRAP_SEED)
    return {
        "primary_true_minus_shuffled_win_rate_pp": primary,
        "descriptive_true_minus_plain_win_rate_pp": descriptive,
        "groups": len(group_scores["plain"]),
        "directional_trials": int(wins["plain"].size),
        "arms": {name: {"win_rate_pp": float(values.mean() * 100),
                       "ties": int((values == 0.5).sum()),
                       "both_directions_strictly_correct_pp": float((values.reshape(-1, 2) == 1).all(axis=1).mean() * 100),
                       "by_relation_win_rate_pp": {r: float(values[:, i].mean() * 100)
                                                    for i, r in enumerate(RELATIONS)}}
                 for name, values in wins.items()},
    }


def run(out_dir, groups, images, manifest, config):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Formal scoring requires the Slurm GPU")
    commit = (out_dir / "frozen_commit.txt").read_text().strip()
    job_id = os.environ["SLURM_JOB_ID"]
    if not re.fullmatch(r"[0-9a-f]{40}", commit) or not job_id.isdecimal():
        raise ValueError("Invalid Slurm provenance")
    objects = [tuple(row["objects"]) for row in manifest["groups"]]
    donors = anchor_assignment(objects)
    cs = ClipEncoder("cs_clip", model_dir(config["models"]["cs_clip"]), config, "cuda")
    matrix = margins(cs, groups, images, objects, donors, config, with_anchors=True)
    del cs
    torch.cuda.empty_cache()
    core = Encoder("core_embed", model_dir(config["models"]["core_embed"]), config, "cuda")
    matrix["core_embed"] = margins(core, groups, images, objects, donors, config, with_anchors=False)["plain"]
    jdump({
        **summarize_scores(matrix),
        "scope": "opposite-relation pairwise ordering on What'sUp Controlled Images B",
        "manifest_sha256": file_sha(MANIFEST),
        "model_config_sha256": MODEL_CONFIG_SHA,
        "frozen_commit": commit,
        "slurm_job_id": job_id,
        "weight": WEIGHT,
        "package_versions": {name: version(name) for name in
                             ("torch", "torchvision", "transformers", "numpy", "open-clip-torch", "Pillow")},
    }, out_dir / "summary_controlled_spatial.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--make-manifest", action="store_true")
    actions.add_argument("--preflight", action="store_true")
    actions.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    if file_sha(MODEL_CONFIG) != MODEL_CONFIG_SHA:
        raise ValueError("Frozen model config digest changed")
    models = json.loads(MODEL_CONFIG.read_text())
    groups, images = read_release()
    computed = make_manifest(groups, images, models)
    if args.make_manifest:
        print(json.dumps(computed, ensure_ascii=False, indent=2))
        return
    if json.loads(MANIFEST.read_text()) != computed:
        raise ValueError("Frozen manifest differs from the re-derived input screen")
    for name in MODEL_NAMES:
        model_dir(models["models"][name])
    print(f"Validated {len(computed['groups'])} quartets, two offline models, and frozen manifest", flush=True)
    if args.out_dir is not None:
        if {p.name for p in args.out_dir.iterdir()} != {"frozen_commit.txt"}:
            raise ValueError("Formal scoring requires a fresh Slurm output directory")
        run(args.out_dir, groups, images, computed, models)


if __name__ == "__main__":
    main()
