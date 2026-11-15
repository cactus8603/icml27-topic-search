"""Build a score-blind COCO train2017 panel disjoint from prior ranking runs."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from http.client import HTTPException
from io import BytesIO
from itertools import combinations
import json
from pathlib import Path
import sys
from threading import Lock
import time
from urllib.request import urlopen
from zipfile import ZipFile

import ijson
from PIL import Image


HERE = Path(__file__).resolve().parent
TOPIC = HERE.parents[2] / "killtests" / "compositional-retrieval"
sys.path.insert(0, str(TOPIC / "scripts"))
from common import COCO_DIR, ICML
from prospective_manifest import canonical_bytes, digest, file_sha, hash_select, heldout_assets, old_keys


DATA = ICML / "data" / "coco2017-rank-and"
ANNOTATIONS = COCO_DIR / "annotations_trainval2017.zip"
IMAGE_ORIGIN = "http://images.cocodataset.org/train2017/"
IMAGE_SPLIT_SEED = 2026111529
QUERY_SPLIT_SEED = 2026111530
PARTS = {"dev": 4096, "test": 8192}
QUERY_MINIMUM = {"dev": (50, 35), "test": (140, 85)}
AREA = 0.02
MAX_IMAGE_BYTES = 2 * 1024**2
MAX_PANEL_BYTES = 3 * 1024**3
SOURCES = {
    ANNOTATIONS: "113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268",
    TOPIC / "results/headroom/26393/e2_queries_main.json": "2aa2783101d65313f1193eaddef62af8dc523c0c5cfddfba213e5c08716dae94",
    TOPIC / "configs/prospective_rank_manifest.json": "ade6a92afd554fe7ee9412b72d9c125b67024b95ae2124e58497360141092959",
    TOPIC / "configs/assets_lock.json": "d8645acf04d055aca2e3382c1de52b9de113de6133719cc2b21ff48663c22d6e",
}


def coco_rows(field):
    with ZipFile(ANNOTATIONS) as archive, archive.open("annotations/instances_train2017.json") as stream:
        yield from ijson.items(stream, f"{field}.item", use_float=True)


def exclusions():
    for path, expected in SOURCES.items():
        if file_sha(path) != expected:
            raise ValueError(f"Changed source: {path.name}")
    snapshots = json.loads((TOPIC / "configs/assets_lock.json").read_text())
    ht = snapshots["kbora/Half-Truths"]
    ht_ids, ht_shas, ht_files = heldout_assets(ht["snapshot"])
    val_images = sorted((COCO_DIR / "val2017").glob("*.jpg"))
    if len(val_images) != 5000:
        raise ValueError("The original COCO validation pool is incomplete")
    val_ids = {int(path.stem) for path in val_images}
    val_shas = {file_sha(path) for path in val_images}
    old = old_keys(TOPIC / "results/headroom/26393/e2_queries_main.json")
    prior = json.loads((TOPIC / "configs/prospective_rank_manifest.json").read_text())
    for split in prior["splits"].values():
        old.update(tuple(q["key"]) for q in split["queries"])
        if set(split["candidate_ids"]) - val_ids or set(split["candidate_image_sha256"]) - val_shas:
            raise ValueError("The original ranking pool does not match the validation image inventory")
    return ht_ids | val_ids, ht_shas | val_shas, old, ht_files


def roster(excluded):
    population = [row for row in coco_rows("images") if row["id"] not in excluded]
    population.sort(key=lambda row: (hash_select(IMAGE_SPLIT_SEED, row["id"]), row["id"]))
    picked = {}
    start = 0
    for part, count in PARTS.items():
        batch = population[start:start + count]
        picked[part] = {row["id"]: [row["width"], row["height"]] for row in batch}
        if len(picked[part]) != count or any(row["file_name"] != f"{row['id']:012d}.jpg" for row in batch):
            raise ValueError("Unexpected image inventory or filename")
        start += count
    return picked


def image_path(data, image_id):
    return data / "images" / f"{image_id:012d}.jpg"


def validate_image(data, shape):
    with Image.open(BytesIO(data)) as image:
        if image.format != "JPEG" or image.size != tuple(shape):
            raise ValueError(f"Expected JPEG with annotated shape {shape}")
        image.load()


def get_image(root, record, publish):
    image_id, shape = record
    path = image_path(root, image_id)
    if path.is_file():
        validate_image(path.read_bytes(), shape)
        return
    failure = None
    for attempt in range(3):
        try:
            with urlopen(IMAGE_ORIGIN + path.name, timeout=35) as response:
                if response.geturl() != IMAGE_ORIGIN + path.name or response.status != 200:
                    raise ValueError("Unexpected response to public COCO image request")
                data = response.read(MAX_IMAGE_BYTES + 1)
            if len(data) > MAX_IMAGE_BYTES:
                raise ValueError("Image above the download limit")
            validate_image(data, shape)
            publish(path, data)
            return
        except (OSError, ValueError, HTTPException) as error:
            failure = error
            time.sleep(attempt + 1)
    raise OSError(f"Could not verify {path.name}: {failure}")


def categories(chosen):
    shapes = {key: shape for rows in chosen.values() for key, shape in rows.items()}
    image_categories = {key: set() for key in shapes}
    names = {row["id"]: row["name"] for row in coco_rows("categories")}
    for annotation in coco_rows("annotations"):
        image_id = annotation["image_id"]
        if image_id in shapes and not annotation["iscrowd"]:
            width, height = shapes[image_id]
            if annotation["area"] >= AREA * width * height:
                image_categories[image_id].add(annotation["category_id"])
    return names, image_categories


def form_queries(part, image_ids, category_sets, old):
    present = {cat: set() for key in image_ids for cat in category_sets[key]}
    for key in image_ids:
        for cat in category_sets[key]:
            present[cat].add(key)
    keys = {pair for key in image_ids for pair in combinations(sorted(category_sets[key]), 2)}
    queries = []
    for pair in sorted(keys - old):
        split = "dev" if hash_select(QUERY_SPLIT_SEED, pair) % 3 == 0 else "test"
        if split != part:
            continue
        left, right = (present[cat] for cat in pair)
        if not (left - right and right - left):
            continue
        relevant = sorted(left & right)
        sentinel = min(relevant, key=lambda key: (hash_select(QUERY_SPLIT_SEED, pair, key), key))
        queries.append({"key": list(pair), "relevant": relevant, "group": sentinel})
    return queries


def build(chosen, old, root=None, prior_shas=()):
    names, category_sets = categories(chosen)
    claims = set(prior_shas)
    parts = {}
    for part, rows in chosen.items():
        ids, shas = [], []
        for image_id in sorted(rows):
            if root is not None:
                sha = file_sha(image_path(root, image_id))
                if sha in claims:
                    continue
                claims.add(sha)
                shas.append(sha)
            ids.append(image_id)
        queries = form_queries(part, ids, category_sets, old)
        n_groups = len({q["group"] for q in queries})
        n_min, g_min = QUERY_MINIMUM[part]
        if len(ids) < PARTS[part] * .95 or len(queries) < n_min or n_groups < g_min:
            raise ValueError(f"Insufficient {part}: {len(ids)} images, {len(queries)} queries, {n_groups} groups")
        parts[part] = {"ids": ids, "jpg_sha256": shas, "queries": queries}
    return {"schema": 1, "names": names, "parts": parts}


def stats(manifest):
    return {part: {"candidates": len(split["ids"]), "queries": len(split["queries"]),
                   "single_positive_queries": sum(len(q["relevant"]) == 1 for q in split["queries"]),
                   "positive_image_groups": len({q["group"] for q in split["queries"]})}
            for part, split in manifest["parts"].items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "fetch"))
    args = parser.parse_args()
    prior_ids, prior_shas, prior_keys, ht_files = exclusions()
    chosen = roster(prior_ids)
    if args.mode == "plan":
        print(json.dumps({"roster_sha256": digest(chosen), "coverage_before_jpeg_dedup": stats(build(chosen, prior_keys))}))
        return
    if (DATA / "manifest.json").exists():
        raise FileExistsError("The prepared panel is already fixed")
    (DATA / "images").mkdir(parents=True, exist_ok=True)
    usage = sum(path.stat().st_size for path in (DATA / "images").glob("*.jpg"))
    limit = Lock()

    def publish(path, data):
        nonlocal usage
        with limit:
            if usage + len(data) > MAX_PANEL_BYTES:
                raise ValueError("Panel reached its 3 GiB disk limit")
            usage += len(data)
        temporary = path.with_suffix(".partial")
        with temporary.open("wb") as output:
            output.write(data)
        temporary.replace(path)

    with ThreadPoolExecutor(max_workers=16) as pool:
        rows = [(key, shape) for split in chosen.values() for key, shape in split.items()]
        print(json.dumps({"download_plan": stats(build(chosen, prior_keys)), "roster_sha256": digest(chosen)}), flush=True)
        for i, _ in enumerate(pool.map(lambda row: get_image(DATA, row, publish), rows), 1):
            if i % 256 == 0:
                print(f"{i}/{len(rows)} selected JPEGs checked", flush=True)
    manifest = build(chosen, prior_keys, DATA, prior_shas) | {
        "roster_sha256": digest(chosen), "prior_id_sha256": digest(sorted(prior_ids)),
        "prior_image_sha256": digest(sorted(prior_shas)), "prior_queries_sha256": digest(sorted(prior_keys)),
        "source_zip_sha256": SOURCES[ANNOTATIONS], "half_truths_file_sha256": ht_files,
    }
    with (DATA / "manifest.json").open("xb") as output:
        output.write(canonical_bytes(manifest) + b"\n")
    print(json.dumps({"manifest_sha256": file_sha(DATA / "manifest.json"), "coverage": stats(manifest)}))


if __name__ == "__main__":
    main()
