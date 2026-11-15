"""Prepare a single score-blind COCO pool, or verify its frozen JPEG inventory."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
from importlib.util import module_from_spec, spec_from_file_location
import hashlib
from io import BytesIO
from itertools import combinations
import json
from pathlib import Path
from tarfile import open as open_tar
from tempfile import NamedTemporaryFile
from threading import Lock
from urllib.request import urlopen

from PIL import Image, ImageSequence


HERE = Path(__file__).resolve().parent
PRIOR_STAGE = HERE.parent / "retrieval-conjunction"
spec = spec_from_file_location("conjunction_prepare", PRIOR_STAGE / "prepare.py")
prior = module_from_spec(spec)
spec.loader.exec_module(prior)
from prospective_manifest import canonical_bytes, digest, file_sha, hash_select


DATA = prior.ICML / "data" / "coco2017-rerank"
CONTROLLED = prior.ICML / "data" / "whatsup-controlled-b" / "controlled_clevr.tar.gz"
CONTROLLED_SHA = "df8b23af7dc73c4bf98937777b7a5d85aab42598eace82fa7adad5802df2c493"
PREVIOUS_SHA = "9bd8a7e4bb167a84ae3e6b6836a248d9e8a92a28c51a3160ab3e599ed7e2d6cd"
IMAGE_SEED = 2026111534
GROUP_SEED = 2026111535
POOL_SIZE = 4096
MIN_IMAGES = 4056
MIN_QUERIES = 60
MIN_GROUPS = 45
CAP = 5 * 1024**3 // 2
BATCH_SIZE = 64
FIRST_BATCH_RESERVE = BATCH_SIZE * prior.MAX_IMAGE_BYTES
IMAGE_FORMATS = ("JPEG", "PNG", "GIF", "WEBP", "TIFF", "BMP")


def previous_panel():
    path = prior.DATA / "manifest.json"
    if (file_sha(path) != PREVIOUS_SHA
            or json.loads((PRIOR_STAGE / "panel.json").read_text())["manifest_sha256"] != PREVIOUS_SHA):
        raise ValueError("26534 pool manifest changed")
    return json.loads(path.read_bytes())["parts"]


def exclusions():
    images, shas, queries, ht_files = prior.exclusions()
    rank_manifest = json.loads((prior.TOPIC / "configs/prospective_rank_manifest.json").read_bytes())
    for split in rank_manifest["splits"].values():
        for q in split["queries"]:
            for key in (q["foil_key"], q["anchor_key"]):
                if key is not None and len(key) == 2:
                    queries.add(tuple(sorted(key)))
    for split in previous_panel().values():
        images.update(split["ids"])
        shas.update(split["jpg_sha256"])
        queries.update(tuple(q["key"]) for q in split["queries"])
    if file_sha(CONTROLLED) != CONTROLLED_SHA:
        raise ValueError("Controlled Images exact-byte exclusion archive changed")
    with open_tar(CONTROLLED, "r|gz") as archive:
        for member in archive:
            if member.isfile() and member.name.lower().endswith((".jpg", ".jpeg", ".png")):
                with archive.extractfile(member) as stream:
                    shas.add(hashlib.sha256(stream.read()).hexdigest())
    return images, shas, queries, ht_files


def select_roster(excluded):
    population = [row for row in prior.coco_rows("images") if row["id"] not in excluded]
    population.sort(key=lambda row: (hash_select(IMAGE_SEED, row["id"]), row["id"]))
    roster = {row["id"]: [row["width"], row["height"]] for row in population[:POOL_SIZE]}
    if (len(roster) != POOL_SIZE
            or any(row["file_name"] != f"{row['id']:012d}.jpg" for row in population[:POOL_SIZE])):
        raise ValueError("Incomplete COCO image roster")
    return roster


def annotations(ids, categories, excluded):
    present = {}
    keys = set()
    for image_id in ids:
        cats = categories[image_id]
        for cat in cats:
            present.setdefault(cat, set()).add(image_id)
        keys.update(combinations(sorted(cats), 2))
    queries = []
    for pair in sorted(keys - excluded):
        left, right = (present[cat] for cat in pair)
        if left - right and right - left:
            relevant = sorted(left & right)
            group = min(relevant, key=lambda key: (hash_select(GROUP_SEED, pair, key), key))
            queries.append({"key": list(pair), "relevant": relevant, "group": group})
    return queries


def coverage(ids, queries):
    return {"candidates": len(ids), "queries": len(queries),
            "groups": len({q["group"] for q in queries}),
            "single_positive": sum(len(q["relevant"]) == 1 for q in queries)}


def check_coverage(summary):
    if (not MIN_IMAGES <= summary["candidates"] <= POOL_SIZE
            or summary["queries"] < MIN_QUERIES or summary["groups"] < MIN_GROUPS):
        raise ValueError(f"Fixed coverage floor unmet: {summary}")


def inspect_image(data, shape):
    with Image.open(BytesIO(data), formats=IMAGE_FORMATS) as image:
        format_name = image.format
        dimensions = list(image.size)
        for frame in ImageSequence.Iterator(image):
            frame.load()
    issue = None
    if format_name != "JPEG" or dimensions != shape:
        issue = {"actual": dimensions, "expected": shape, "format": format_name}
    return hashlib.sha256(data).hexdigest(), issue


def acquire(roster):
    directory = DATA / "images"
    directory.mkdir(parents=True, exist_ok=True)
    with (DATA / "download-bytes.log").open("a+", buffering=1) as ledger:
        fcntl.flock(ledger, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger.seek(0)
        charges = [int(line) for line in ledger]
        if not charges:
            charges = [sum(path.stat().st_size for path in directory.iterdir() if path.is_file())]
            ledger.write(f"{charges[0]}\n")
        # Include the full upper limit of the failed batch from before the persistent ledger.
        return acquire_images(roster, directory, ledger, sum(charges) + FIRST_BATCH_RESERVE)


def acquire_images(roster, directory, ledger, spend):
    if spend > CAP:
        raise ValueError("Existing data exceeds the byte cap")
    locked = Lock()

    def fetch(asset):
        nonlocal spend
        image_id, shape = asset
        path = prior.image_path(DATA, image_id)
        if path.is_file():
            return inspect_image(path.read_bytes(), shape)
        url = prior.IMAGE_ORIGIN + path.name
        with urlopen(url, timeout=35) as response:
            size = int(response.headers.get("Content-Length", 0))
            if response.geturl() != url or response.status != 200 or not 0 < size <= prior.MAX_IMAGE_BYTES:
                raise ValueError(f"Unexpected public asset response for {path.name}")
            with locked:
                if spend + size > CAP:
                    raise ValueError("Reached 2.5 GiB including existing and attempted assets")
                ledger.write(f"{size}\n")
                spend += size
            data = response.read(size)
        if len(data) != size:
            raise ValueError(f"Incomplete public asset {path.name}")
        try:
            result = inspect_image(data, shape)
        except OSError as error:
            raise ValueError(f"Could not decode the official source for image {image_id}") from error
        with NamedTemporaryFile(dir=directory, suffix=".partial", delete=False) as output:
            temp = Path(output.name)
            try:
                output.write(data)
                output.flush()
                temp.replace(path)
            finally:
                temp.unlink(missing_ok=True)
        return result

    rows = list(roster.items())
    received = {}
    with ThreadPoolExecutor(max_workers=16) as pool:
        for start in range(0, len(rows), BATCH_SIZE):
            batch = rows[start:start + BATCH_SIZE]
            for (image_id, _), result in zip(batch, pool.map(fetch, batch), strict=True):
                received[image_id] = result
            if (start + len(batch)) % 512 == 0:
                print(json.dumps({"images_verified": start + len(batch), "charged_bytes_upper_bound": spend}),
                      flush=True)
    return received


def prepare(mode):
    output = DATA / "manifest.json"
    if mode == "fetch" and output.exists():
        raise FileExistsError("Frozen asset manifest already exists")
    excluded_ids, excluded_shas, excluded_keys, half_truths = exclusions()
    roster = select_roster(excluded_ids)
    names, category_sets = prior.categories({"test": roster})
    all_ids = sorted(roster)
    unverified = annotations(all_ids, category_sets, excluded_keys)
    summary = coverage(all_ids, unverified)
    print(json.dumps({"annotation_plan": summary, "roster_sha256": digest(roster)}), flush=True)
    check_coverage(summary)
    if mode == "plan":
        return
    received = (acquire(roster) if mode == "fetch" else
                {image_id: inspect_image(prior.image_path(DATA, image_id).read_bytes(), shape)
                 for image_id, shape in roster.items()})
    ids, hashes = [], []
    seen = set(excluded_shas)
    source_mismatches = []
    for image_id in all_ids:
        sha, issue = received[image_id]
        if issue is not None:
            source_mismatches.append({"id": image_id, "sha256": sha, **issue})
        elif sha not in seen:
            seen.add(sha)
            ids.append(image_id)
            hashes.append(sha)
    queries = annotations(ids, category_sets, excluded_keys)
    summary = coverage(ids, queries)
    check_coverage(summary)
    manifest = {"schema": 1, "ids": ids, "jpeg_sha256": hashes, "queries": queries, "names": names,
                "source_mismatches": source_mismatches,
                "roster_sha256": digest(roster), "excluded_id_sha256": digest(sorted(excluded_ids)),
                "excluded_jpeg_sha256": digest(sorted(excluded_shas)),
                "excluded_keys_sha256": digest(sorted(excluded_keys)),
                "half_truths_sha256": half_truths, "controlled_archive_sha256": CONTROLLED_SHA,
                "conjunction_manifest_sha256": PREVIOUS_SHA,
                "annotations_sha256": prior.SOURCES[prior.ANNOTATIONS]}
    expected = canonical_bytes(manifest) + b"\n"
    if mode == "verify":
        if output.read_bytes() != expected:
            raise ValueError("Frozen manifest differs from the annotation-only reconstruction")
    else:
        with output.open("xb") as stream:
            stream.write(expected)
    print(json.dumps({"manifest_sha256": file_sha(output), "coverage": summary,
                      "source_mismatches": source_mismatches,
                      "jpeg_bytes": sum(path.stat().st_size for path in (DATA / "images").glob("*.jpg"))}), flush=True)


def preflight():
    path = DATA / "manifest.json"
    expected = json.loads((HERE / "panel.json").read_text())["manifest_sha256"]
    if file_sha(path) != expected:
        raise ValueError("Frozen new asset manifest changed")
    panel = json.loads(path.read_bytes())
    old = previous_panel()
    ids, hashes, queries = panel["ids"], panel["jpeg_sha256"], panel["queries"]
    id_set, sha_set = set(ids), set(hashes)
    keys = {tuple(q["key"]) for q in queries}
    if (panel["schema"] != 1 or ids != sorted(id_set) or len(ids) != len(hashes)
            or len(ids) != len(sha_set) or len(keys) != len(queries)):
        raise ValueError("Malformed candidate or query inventory")
    for rejected in panel["source_mismatches"]:
        sha, issue = inspect_image(prior.image_path(DATA, rejected["id"]).read_bytes(), rejected["expected"])
        if (rejected["id"] in id_set or issue is None
                or rejected != {"id": rejected["id"], "sha256": sha, **issue}):
            raise ValueError("Malformed source-image exclusion")
    for split in old.values():
        if (id_set.intersection(split["ids"]) or sha_set.intersection(split["jpg_sha256"])
                or keys.intersection(tuple(q["key"]) for q in split["queries"])):
            raise ValueError("New panel overlaps 26534")
    for q in queries:
        if (len(q["key"]) != 2 or sorted(set(q["key"])) != q["key"]
                or not all(str(cat) in panel["names"] for cat in q["key"])
                or q["relevant"] != sorted(set(q["relevant"])) or not id_set.issuperset(q["relevant"])
                or q["group"] not in q["relevant"]):
            raise ValueError("Malformed query or fixed relevance set")
    check_coverage(coverage(ids, queries))
    for image_id, sha in zip(ids, hashes, strict=True):
        if file_sha(prior.image_path(DATA, image_id)) != sha:
            raise ValueError(f"JPEG identity mismatch for {image_id}")
    return panel, expected


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "fetch", "verify"))
    prepare(parser.parse_args().mode)
