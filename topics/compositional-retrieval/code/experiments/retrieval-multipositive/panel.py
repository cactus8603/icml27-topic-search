"""Build a score-blind, balanced COCO left/right panel from official metadata."""

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
from itertools import combinations
import json
from pathlib import Path
import re
import sys
from zipfile import ZipFile

import ijson


TOPIC = Path(__file__).resolve().parents[3] / "killtests" / "compositional-retrieval"
sys.path.insert(0, str(TOPIC / "scripts"))
from prospective_manifest import attach_shuffles, canonical_bytes, digest, file_sha, hash_select, heldout_assets

from common import COCO_DIR


COCO_SHA256 = "113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268"
CONFIG_SHA256 = {
    "assets_lock.json": "d8645acf04d055aca2e3382c1de52b9de113de6133719cc2b21ff48663c22d6e",
    "prospective_rank_manifest.json": "ade6a92afd554fe7ee9412b72d9c125b67024b95ae2124e58497360141092959",
    "controlled_spatial_manifest.json": "962c18756740a86fde7cfad0ecce3362a7d36d59226e16b85e63d47d3d79ebd8",
}
TRAIN_SHA256 = {
    "coco2017-rank-and": "9bd8a7e4bb167a84ae3e6b6836a248d9e8a92a28c51a3160ab3e599ed7e2d6cd",
    "coco2017-rerank": "df51d103caa1ec7968d46254b5757789e7f834e1f3a11dae1105b2f8c63ccd3f",
    "coco2017-anchor-rerank": "8f43eedbcef31fe5057c26705e97bf9f213f162612d9a15b53da11bb9bacf78d",
}
COCO_ORIGIN = "http://images.cocodataset.org/"
FLICKR_PHOTO = re.compile(r"/(\d+)_[0-9a-f]+")
CONTROLLED_ALIASES = {"mug": "cup", "plate": "bowl", "can": "bottle", "phone": "cell phone"}
GROUP_SEED, IMAGE_SEED, DONOR_SEED = 2026111571, 2026111572, 2026111573
MIN_GROUPS = 120
PER_DIRECTION = 4
AREA_FRACTION = 0.02
HORIZONTAL_GAP = 0.02
VERTICAL_OVERLAP = 0.25
DIRECTIONS = ("left", "right")


@dataclass(frozen=True)
class Prior:
    coco_ids: set
    known_image_shas: set
    controlled_pairs: set
    sources: dict


def verified_json(path, expected):
    body = path.read_bytes()
    if hashlib.sha256(body).hexdigest() != expected:
        raise ValueError(f"Input changed: {path}")
    return json.loads(body)


def prior_inventory():
    configs = {name: verified_json(TOPIC / "configs" / name, sha)
               for name, sha in CONFIG_SHA256.items()}
    train = {name: verified_json(COCO_DIR.parent / name / "manifest.json", sha)
             for name, sha in TRAIN_SHA256.items()}
    controlled = configs["controlled_spatial_manifest.json"]
    prior_val = configs["prospective_rank_manifest.json"]
    ht = configs["assets_lock.json"]["kbora/Half-Truths"]
    ids, shas, ht_digests = heldout_assets(ht["snapshot"])
    expected_ht_digests = {name: row["file_sha256"] for name, row in
                           controlled["overlap_screen"]["prior_inputs"]["Half-Truths"].items()}
    if ht_digests != expected_ht_digests:
        raise ValueError("Half-Truths parquet sources changed")
    for part in prior_val["splits"].values():
        ids.update(part["candidate_ids"])
        shas.update(part["candidate_image_sha256"])
    for part in train["coco2017-rank-and"]["parts"].values():
        ids.update(part["ids"])
        shas.update(part["jpg_sha256"])
    for dataset, removed_key in (("coco2017-rerank", "source_mismatches"),
                                 ("coco2017-anchor-rerank", "rejected")):
        panel = train[dataset]
        ids.update(panel["ids"])
        shas.update(panel["jpeg_sha256"])
        for row in panel[removed_key]:
            ids.add(row["id"])
            shas.add(row["sha256"])
    for group in controlled["groups"]:
        shas.update(group["image_hashes"])
    pairs = {
        tuple(sorted(CONTROLLED_ALIASES.get(name, name) for name in group["objects"]))
        for group in controlled["groups"]
    }
    sources = {
        "coco_annotations_sha256": COCO_SHA256,
        "source_config_sha256": CONFIG_SHA256,
        "train_panel_sha256": TRAIN_SHA256,
        "half_truths_revision": ht["sha"],
        "half_truths_parquet_sha256": ht_digests,
    }
    return Prior(ids, shas, pairs, sources)


def coco_rows(archive, split, field):
    with archive.open(f"annotations/instances_{split}2017.json") as stream:
        yield from ijson.items(stream, f"{field}.item", use_float=True)


def flickr_id(row):
    match = FLICKR_PHOTO.search(row["flickr_url"])
    if match is None:
        raise ValueError(f"Image {row['id']} lacks a Flickr source identity")
    return match[1]


def image_metadata(row):
    image_id = row["id"]
    filename = f"{image_id:012d}.jpg"
    url = f"{COCO_ORIGIN}train2017/{filename}"
    if row["file_name"] != filename or row["coco_url"] != url:
        raise ValueError(f"Inconsistent official image URL or filename for {image_id}")
    return {"file_name": filename, "url": url, "width": row["width"], "height": row["height"],
            "flickr_id": flickr_id(row)}


def relation(first, second, width):
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    if min(ay + ah, by + bh) - max(ay, by) < min(ah, bh) * VERTICAL_OVERLAP:
        return None
    if bx - (ax + aw) >= width * HORIZONTAL_GAP:
        return 0
    if ax - (bx + bw) >= width * HORIZONTAL_GAP:
        return 1
    return None


def relation_index(annotations, images, names, excluded_pairs):
    by_image = defaultdict(lambda: defaultdict(list))
    for row in annotations:
        if row["image_id"] in images:
            by_image[row["image_id"]][row["category_id"]].append((row["iscrowd"], row["area"], row["bbox"]))
    by_pair = defaultdict(lambda: ([], []))
    for image_id, annotated in by_image.items():
        image = images[image_id]
        width, height = image["width"], image["height"]
        boxes = [(cat, rows[0][2]) for cat, rows in sorted(annotated.items())
                 if len(rows) == 1 and not rows[0][0] and rows[0][1] >= AREA_FRACTION * width * height]
        for (a, box_a), (b, box_b) in combinations(boxes, 2):
            direction = relation(box_a, box_b, width)
            if direction is not None and tuple(sorted((names[a], names[b]))) not in excluded_pairs:
                by_pair[a, b][direction].append(image_id)
    return by_pair


def select_groups(pairs, images):
    used = set()
    groups = {}
    ordered = sorted(pairs, key=lambda key: (min(map(len, pairs[key])), sum(map(len, pairs[key])),
                                            hash_select(GROUP_SEED, key), key))
    for key in ordered:
        if min(map(len, pairs[key])) < PER_DIRECTION:
            continue
        sides = []
        reserved = set()
        for candidates in pairs[key]:
            chosen = []
            for image_id in sorted(candidates, key=lambda i: (hash_select(IMAGE_SEED, key, i), i)):
                source = images[image_id]["flickr_id"]
                if source in used or source in reserved:
                    continue
                reserved.add(source)
                chosen.append(image_id)
                if len(chosen) == PER_DIRECTION:
                    break
            sides.append(sorted(chosen))
        if all(len(chosen) == PER_DIRECTION for chosen in sides):
            used.update(reserved)
            groups[key] = sides
    return groups


def describe_groups(chosen, names):
    donor_rows = []
    for key in sorted(chosen):
        a, b = (names[k] for k in key)
        donor_rows.append({"k": 2, "key": key, "anchor_key": key,
                           "anchor": f"A photo with the {a} and the {b}."})
    attach_shuffles(donor_rows, DONOR_SEED)
    if (any(row["shuffled_donor_key"] is None for row in donor_rows)
            or {row["shuffled_donor_key"] for row in donor_rows} != set(chosen)):
        raise ValueError("Every group needs a unique, noun-disjoint shuffled donor")
    groups = []
    for row in donor_rows:
        key = row["key"]
        a, b = (names[k] for k in key)
        sides = chosen[key]
        groups.append({
            "key": key, "objects": [a, b], "anchor": row["anchor"],
            "shuffled_anchor": row["shuffled_anchor"], "shuffled_donor_key": row["shuffled_donor_key"],
            "images": sorted(sides[0] + sides[1]),
            "queries": [{"direction": direction, "text": f"A photo with the {a} to the {direction} of the {b}.",
                         "relevant": relevant} for direction, relevant in zip(DIRECTIONS, sides, strict=True)],
        })
    return groups


def build():
    prior = prior_inventory()
    annotation_file = COCO_DIR / "annotations_trainval2017.zip"
    if file_sha(annotation_file) != COCO_SHA256:
        raise ValueError("COCO train/val annotation ZIP changed")
    with ZipFile(annotation_file) as archive:
        train = list(coco_rows(archive, "train", "images"))
        val = list(coco_rows(archive, "val", "images"))
        ids = {row["id"] for row in train + val}
        if len(train) != 118287 or len(val) != 5000 or len(ids) != len(train) + len(val):
            raise ValueError("Incomplete or repeated official COCO roster")
        prior_ids = prior.coco_ids | {row["id"] for row in val}
        if not prior_ids <= ids:
            raise ValueError("A prior COCO ID is missing from the official train/val roster")
        prior_flickr = {flickr_id(row) for row in train + val if row["id"] in prior_ids}
        population = {row["id"]: image_metadata(row) for row in train
                      if row["id"] not in prior_ids and flickr_id(row) not in prior_flickr}
        names = {row["id"]: row["name"] for row in coco_rows(archive, "train", "categories")}
        pairs = relation_index(coco_rows(archive, "train", "annotations"), population, names, prior.controlled_pairs)
    chosen = select_groups(pairs, population)
    if len(chosen) < MIN_GROUPS:
        raise ValueError(f"Panel has only {len(chosen)} complete groups, expected at least {MIN_GROUPS}")
    groups = describe_groups(chosen, names)
    selected_ids = [i for q in groups for i in q["images"]]
    images = {i: population[i] for i in sorted(selected_ids)}
    selected_flickr = {image["flickr_id"] for image in images.values()}
    if (len(images) != len(selected_ids) or len(selected_flickr) != len(images)
            or prior_ids.intersection(images) or prior_flickr & selected_flickr):
        raise ValueError("Panel has COCO ID or Flickr source overlap")
    return {
        "schema": 1, "status": "source_ids_verified_only",
        "sources": prior.sources,
        "exclusions": {
            "coco_ids_sha256": digest(sorted(prior_ids)),
            "flickr_ids_sha256": digest(sorted(prior_flickr)),
            "known_prior_image_shas_sha256": digest(sorted(prior.known_image_shas)),
            "controlled_pairs_sha256": digest(sorted(prior.controlled_pairs)),
        },
        "coverage": {
            "train_images": len(train), "source_eligible_train_images": len(population),
            "eligible_categories": len(names), "groups": len(groups), "images": len(images),
            "queries": len(groups) * len(DIRECTIONS), "positives_per_query": PER_DIRECTION,
            "query_image_pairs": len(images) * len(DIRECTIONS),
        },
        "groups": groups, "images": {str(i): image for i, image in images.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    commands.add_parser("inventory")
    plan = commands.add_parser("plan")
    plan.add_argument("--out", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    panel = build()
    body = canonical_bytes(panel) + b"\n"
    if args.mode == "plan":
        with args.out.open("xb") as stream:
            stream.write(body)
    if args.mode == "verify" and args.manifest.read_bytes() != body:
        raise ValueError("Metadata manifest differs from the deterministic source reconstruction")
    print(json.dumps({"plan_sha256": hashlib.sha256(body).hexdigest(), "status": panel["status"],
                      "coverage": panel["coverage"]}, sort_keys=True))


if __name__ == "__main__":
    main()
