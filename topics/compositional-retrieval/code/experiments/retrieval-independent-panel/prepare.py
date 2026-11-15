"""Reconstruct the independent COCO retrieval plan from verified local metadata."""

import argparse
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import sys
from zipfile import ZipFile


HERE = Path(__file__).resolve().parent
POOL_SIZE = 8192
IMAGE_SEED = 2026111541
MIN_QUERIES = 100
MIN_GROUPS = 85
SPATIAL_SHA = "33363955507ec1c7cad716c3294f41c7892b12d944760631ce42fb85894295bb"
PLAN_SHA256 = "d7981026bfb1d56eeb3d8b4ed357e177677e6297cc8f778fcd163bac690823d6"


def load_prior(name, relative_path):
    spec = spec_from_file_location(name, HERE.parent / relative_path)
    module = module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


spatial = load_prior("retrieval_spatial_metadata", "retrieval-multipositive/panel.py")
rerank = load_prior("retrieval_rerank_metadata", "retrieval-rerank/prepare.py")
DATA = spatial.COCO_DIR.parent
DEFAULT_IMAGES = DATA / "coco2017-retrieval-independent" / "images"
LEGACY_QUERY_PATH = rerank.prior.TOPIC / "results/headroom/26393/e2_queries_main.json"
LEGACY_QUERY_SHA256 = rerank.prior.SOURCES[LEGACY_QUERY_PATH]


def select_images(train, validation, exposed_ids, count=POOL_SIZE):
    blocked_ids = exposed_ids | {image["id"] for image in validation}
    official_ids = {image["id"] for image in train + validation}
    if len(official_ids) != len(train) + len(validation) or not blocked_ids <= official_ids:
        raise ValueError("Official COCO metadata has repeated IDs or omits known images")
    blocked_sources = {spatial.flickr_id(image) for image in train + validation if image["id"] in blocked_ids}
    eligible = [(image, spatial.flickr_id(image)) for image in train if image["id"] not in blocked_ids]
    eligible = [(image, source) for image, source in eligible if source not in blocked_sources]
    eligible.sort(key=lambda item: (spatial.hash_select(IMAGE_SEED, item[0]["id"]), item[0]["id"]))
    selected, seen_sources = {}, set()
    for image, source in eligible:
        if source in seen_sources:
            continue
        seen_sources.add(source)
        if image["file_name"] != f"{image['id']:012d}.jpg":
            raise ValueError("Candidate filename differs from the official COCO ID")
        selected[image["id"]] = [image["width"], image["height"]]
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError("Insufficient unexposed candidates")
    return dict(sorted(selected.items())), blocked_ids, blocked_sources, len(eligible)


def pair_keys(queries, fields):
    return {tuple(sorted(query[field])) for query in queries for field in fields
            if query.get(field) is not None and len(query[field]) == 2}


def exposed_pairs(prior, spatial_plan, names):
    verified_legacy = spatial.verified_json(LEGACY_QUERY_PATH, LEGACY_QUERY_SHA256)
    keys = pair_keys((query for rows in verified_legacy["queries"].values() for query in rows), ("cats",))
    rank_name = "prospective_rank_manifest.json"
    rank = spatial.verified_json(spatial.TOPIC / "configs" / rank_name, spatial.CONFIG_SHA256[rank_name])
    for part in rank["splits"].values():
        keys.update(pair_keys(part["queries"], ("key", "anchor_key", "foil_key")))
    for dataset, sha in spatial.TRAIN_SHA256.items():
        panel = spatial.verified_json(DATA / dataset / "manifest.json", sha)
        for part in panel.get("parts", {"both": panel}).values():
            keys.update(pair_keys(part["queries"], ("key",)))
    keys.update(pair_keys(spatial_plan["groups"], ("key",)))
    category_ids = {name: category for category, name in names.items()}
    keys.update(tuple(sorted(category_ids[name] for name in pair)) for pair in prior.controlled_pairs
                if all(name in category_ids for name in pair))
    return keys


def plan():
    prior = spatial.prior_inventory()
    spatial_plan = spatial.verified_json(DATA / "coco2017-retrieval-multipositive/panel.json", SPATIAL_SHA)
    exposed_ids = prior.coco_ids | {int(image_id) for image_id in spatial_plan["images"]}
    if spatial.file_sha(rerank.prior.ANNOTATIONS) != spatial.COCO_SHA256:
        raise ValueError("COCO train/val annotation ZIP changed")
    with ZipFile(rerank.prior.ANNOTATIONS) as archive:
        train = list(spatial.coco_rows(archive, "train", "images"))
        validation = list(spatial.coco_rows(archive, "val", "images"))
    if (len(train), len(validation)) != (118287, 5000):
        raise ValueError("Official COCO roster is incomplete")
    roster, blocked_ids, blocked_sources, eligible = select_images(train, validation, exposed_ids)
    names, categories = rerank.prior.categories({"test": roster})
    blocked_keys = exposed_pairs(prior, spatial_plan, names)
    queries = rerank.annotations(roster, categories, blocked_keys)
    from e2_build_queries import joint_text

    for query in queries:
        query["text"] = joint_text([names[category] for category in query["key"]])
    coverage = rerank.coverage(roster, queries)
    return {
        "schema": 1,
        "seed": IMAGE_SEED,
        "sources": prior.sources | {
            "legacy_e2_queries_sha256": LEGACY_QUERY_SHA256,
            "latest_spatial_plan_sha256": SPATIAL_SHA,
        },
        "exclusions": {
            "coco_ids_sha256": spatial.digest(sorted(blocked_ids)),
            "flickr_ids_sha256": spatial.digest(sorted(blocked_sources)),
            "prior_pairs_sha256": spatial.digest(sorted(blocked_keys)),
            "prior_pairs": len(blocked_keys),
            "remaining_train_ids_before_source_deduplication": eligible,
        },
        "roster": roster,
        "queries": queries,
        "coverage": coverage,
    }


def summary(panel, images):
    paths = [images / f"{candidate:012d}.jpg" for candidate in panel["roster"]]
    supplied = sum(path.is_file() and not path.is_symlink() for path in paths)
    coverage = panel["coverage"]
    return {
        "status": "image_validation_required",
        "gpu_ready": False,
        "metadata_meets_coverage_floor": (coverage["queries"] >= MIN_QUERIES and coverage["groups"] >= MIN_GROUPS),
        "coverage": coverage,
        "cache_dir": str(images),
        "candidate_files_present": supplied,
        "candidate_files_missing": len(paths) - supplied,
        "plan_sha256": spatial.digest(panel),
        "roster_sha256": spatial.digest(panel["roster"]),
        "queries_sha256": spatial.digest(panel["queries"]),
        "sources": panel["sources"],
        "exclusions": panel["exclusions"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES, help="Directory to check for candidate files")
    args = parser.parse_args()
    receipt = summary(plan(), args.images)
    if receipt["plan_sha256"] != PLAN_SHA256:
        raise ValueError("Reconstructed metadata differs from the reviewed plan digest")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
