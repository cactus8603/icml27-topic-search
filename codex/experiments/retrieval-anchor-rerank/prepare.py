"""Build and verify the fresh COCO annotation-only anchor reranking panel."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
from threading import Lock
from urllib.request import urlopen


HERE = Path(__file__).resolve().parent
stage_path = HERE.parent / "retrieval-rerank"
spec = spec_from_file_location("last_rerank_prepare", stage_path / "prepare.py")
prior = module_from_spec(spec)
spec.loader.exec_module(prior)
from e2_build_queries import joint_text
from prospective_manifest import attach_shuffles, canonical_bytes, digest, file_sha, hash_select


DATA = prior.DATA.parent / "coco2017-anchor-rerank"
PREVIOUS_SHA = "df51d103caa1ec7968d46254b5757789e7f834e1f3a11dae1105b2f8c63ccd3f"
IMAGE_SEED = 2026111541
ROSTER_SHA = "eeba31b700ca84c69928df3a1f1267233da7b9b315808a5c510da9b62ae0121f"
DROP_SEED = 2026111542
SHUFFLE_SEED = 2026111543
POOL_SIZE = 8192
MIN_IMAGES = 8110
MIN_QUERIES = 100
MIN_GROUPS = 85
CAP = 5 * 1024**3 // 2


def previous_panel():
    path = prior.DATA / "manifest.json"
    if (file_sha(path) != PREVIOUS_SHA
            or json.loads((stage_path / "panel.json").read_bytes())["manifest_sha256"] != PREVIOUS_SHA):
        raise ValueError("Changed previous reranking panel")
    return json.loads(path.read_bytes())


def exclusions():
    ids, shas, keys, half_truths = prior.exclusions()
    panel = previous_panel()
    ids.update(panel["ids"])
    shas.update(panel["jpeg_sha256"])
    keys.update(tuple(q["key"]) for q in panel["queries"])
    for asset in panel["source_mismatches"]:
        ids.add(asset["id"])
        shas.add(asset["sha256"])
    return ids, shas, keys, half_truths


def roster(excluded):
    population = [row for row in prior.prior.coco_rows("images") if row["id"] not in excluded]
    population.sort(key=lambda row: (hash_select(IMAGE_SEED, row["id"]), row["id"]))
    selected = population[:POOL_SIZE]
    if len(selected) != POOL_SIZE or any(row["file_name"] != f"{row['id']:012d}.jpg" for row in selected):
        raise ValueError("Unexpected candidate count or file names")
    items = {row["id"]: [row["width"], row["height"]] for row in selected}
    if digest(items) != ROSTER_SHA:
        raise ValueError("Fresh annotation-only roster changed")
    return items


def with_anchors(queries, names):
    queries = [q.copy() for q in queries]
    for q in queries:
        key = q["key"]
        omitted = hash_select(DROP_SEED, "omit", key) % 2
        q["k"] = 2
        q["text"] = joint_text([names[cat] for cat in key])
        q["anchor_key"] = [key[1 - omitted]]
        q["anchor"] = joint_text([names[q["anchor_key"][0]]])
    attach_shuffles(queries, SHUFFLE_SEED)
    verify_anchors(queries, names)
    return queries


def verify_anchors(queries, names):
    by_key = {tuple(q["key"]): q for q in queries}
    donor_keys = [tuple(q["shuffled_donor_key"] or ()) for q in queries]
    if len(by_key) != len(queries) or sorted(donor_keys) != sorted(by_key):
        raise ValueError("Every new query needs one unique disjoint anchor donor")
    for q, d in zip(queries, donor_keys, strict=True):
        key = q["key"]
        expected_anchor = [key[1 - hash_select(DROP_SEED, "omit", key) % 2]]
        donor = by_key[d]
        if (len(key) != 2 or sorted(set(key)) != key or q["anchor_key"] != expected_anchor
                or q["text"] != joint_text([names[cat] for cat in key])
                or q["anchor"] != joint_text([names[cat] for cat in expected_anchor])
                or set(key) & set(donor["anchor_key"]) or q["shuffled_anchor"] != donor["anchor"]):
            raise ValueError("Invalid full, real-anchor, or shuffled-anchor construction")


def check_coverage(ids, queries):
    summary = prior.coverage(ids, queries)
    if (not MIN_IMAGES <= len(ids) <= POOL_SIZE
            or summary["queries"] < MIN_QUERIES or summary["groups"] < MIN_GROUPS):
        raise ValueError(f"Fixed annotation-only coverage floor not met: {summary}")
    return summary


def image_path(image_id):
    return prior.prior.image_path(DATA, image_id)


def acquire(selected):
    images = DATA / "images"
    images.mkdir(parents=True, exist_ok=True)
    with (DATA / "download-bytes.log").open("a+", buffering=1) as ledger:
        fcntl.flock(ledger, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger.seek(0)
        charges = [int(line) for line in ledger]
        if not charges:
            charges = [sum(path.stat().st_size for path in images.iterdir() if path.is_file())]
            ledger.write(f"{charges[0]}\n")
        return fetch_images(selected, images, ledger, sum(charges))


def fetch_images(selected, images, ledger, charged):
    if charged > CAP:
        raise ValueError("Existing acquisition exceeds the 2.5 GiB limit")
    lock = Lock()

    def fetch(item):
        nonlocal charged
        image_id, shape = item
        path = image_path(image_id)
        if path.is_file():
            return image_id, prior.inspect_image(path.read_bytes(), shape)
        url = prior.prior.IMAGE_ORIGIN + path.name
        with urlopen(url, timeout=35) as response:
            size = int(response.headers.get("Content-Length", 0))
            if response.geturl() != url or response.status != 200 or not 0 < size <= prior.prior.MAX_IMAGE_BYTES:
                raise ValueError(f"Unexpected public image response: {image_id}")
            with lock:
                if charged + size > CAP:
                    raise ValueError("Acquisition reached the 2.5 GiB cumulative cap")
                ledger.write(f"{size}\n")
                charged += size
            data = response.read(size)
        if len(data) != size:
            raise ValueError(f"Incomplete image: {image_id}")
        try:
            fingerprint = prior.inspect_image(data, shape)
        except OSError as error:
            raise ValueError(f"Could not fully decode a public image: {image_id}") from error
        with NamedTemporaryFile(dir=images, suffix=".partial", delete=False) as stream:
            pending = Path(stream.name)
            try:
                stream.write(data)
                stream.flush()
                pending.replace(path)
            finally:
                pending.unlink(missing_ok=True)
        return image_id, fingerprint

    items = list(selected.items())
    received = {}
    with ThreadPoolExecutor(max_workers=16) as workers:
        for start in range(0, len(items), 64):
            received.update(workers.map(fetch, items[start:start + 64]))
            if len(received) % 512 == 0:
                print(json.dumps({"verified_images": len(received), "charged_bytes": charged}), flush=True)
    return received


def assemble(selected, seen, received):
    ids, shas, rejected = [], [], []
    known = set(seen)
    for image_id in sorted(selected):
        sha, issue = received[image_id]
        if issue is not None:
            rejected.append({"id": image_id, "sha256": sha, "reason": "source_format_or_dimensions", **issue})
        elif sha in known:
            rejected.append({"id": image_id, "sha256": sha, "reason": "exact_duplicate"})
        else:
            known.add(sha)
            ids.append(image_id)
            shas.append(sha)
    return ids, shas, rejected


def prepare(mode):
    path = DATA / "manifest.json"
    if mode == "fetch" and path.exists():
        raise FileExistsError("Panel already prepared. Use verify.")
    blocked_ids, blocked_shas, blocked_keys, half_truths = exclusions()
    selected = roster(blocked_ids)
    names, categories = prior.prior.categories({"test": selected})
    draft = with_anchors(prior.annotations(sorted(selected), categories, blocked_keys), names)
    print(json.dumps({"annotation_plan": check_coverage(selected, draft), "roster_sha256": digest(selected)}),
          flush=True)
    if mode == "plan":
        return
    received = (acquire(selected) if mode == "fetch" else
                {image_id: prior.inspect_image(image_path(image_id).read_bytes(), shape)
                 for image_id, shape in selected.items()})
    ids, shas, rejected = assemble(selected, blocked_shas, received)
    queries = with_anchors(prior.annotations(ids, categories, blocked_keys), names)
    summary = check_coverage(ids, queries)
    panel = {"schema": 1, "ids": ids, "jpeg_sha256": shas, "queries": queries, "names": names,
             "rejected": rejected, "roster_sha256": digest(selected),
             "previous_rerank_sha256": PREVIOUS_SHA,
             "excluded_ids_sha256": digest(sorted(blocked_ids)),
             "excluded_jpegs_sha256": digest(sorted(blocked_shas)),
             "excluded_queries_sha256": digest(sorted(blocked_keys)),
             "coco_annotations_sha256": prior.prior.SOURCES[prior.prior.ANNOTATIONS],
             "half_truths_sha256": half_truths, "near_duplicate_screened": False}
    body = canonical_bytes(panel) + b"\n"
    if mode == "verify":
        if path.read_bytes() != body:
            raise ValueError("Local reconstruction does not match the frozen panel")
    else:
        with path.open("xb") as stream:
            stream.write(body)
    print(json.dumps({"manifest_sha256": file_sha(path), "coverage": summary, "rejections": len(rejected),
                      "near_duplicate_screened": False}), flush=True)


def preflight():
    expected = json.loads((HERE / "panel.json").read_bytes())["manifest_sha256"]
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("Bind the independently verified new manifest in panel.json first")
    path = DATA / "manifest.json"
    if file_sha(path) != expected:
        raise ValueError("Changed new manifest")
    panel = json.loads(path.read_bytes())
    previous = previous_panel()
    ids, shas = panel["ids"], panel["jpeg_sha256"]
    if (panel["schema"] != 1 or panel["near_duplicate_screened"] is not False or panel["roster_sha256"] != ROSTER_SHA
            or panel["previous_rerank_sha256"] != PREVIOUS_SHA
            or ids != sorted(set(ids)) or len(ids) != len(shas) or len(shas) != len(set(shas))
            or set(ids).intersection(previous["ids"]) or set(shas).intersection(previous["jpeg_sha256"])
            or {tuple(q["key"]) for q in panel["queries"]}.intersection(tuple(q["key"]) for q in previous["queries"])):
        raise ValueError("Malformed or overlapping candidate panel")
    for image_id, sha in zip(ids, shas, strict=True):
        if file_sha(image_path(image_id)) != sha:
            raise ValueError(f"Changed verified JPEG: {image_id}")
    for entry in panel["rejected"]:
        if entry["id"] in ids or file_sha(image_path(entry["id"])) != entry["sha256"]:
            raise ValueError("Changed excluded image receipt")
    candidates = set(ids)
    for q in panel["queries"]:
        if (q["relevant"] != sorted(set(q["relevant"])) or not candidates.issuperset(q["relevant"])
                or q["group"] not in q["relevant"]):
            raise ValueError("Malformed fixed relevance or cluster")
    verify_anchors(panel["queries"], {int(cat): name for cat, name in panel["names"].items()})
    return panel, expected, check_coverage(ids, panel["queries"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "fetch", "verify"))
    prepare(parser.parse_args().mode)
