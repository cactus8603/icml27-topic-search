#!/usr/bin/env python3
"""Rebuild only the public, exposed retrieval development panels, without models.

The original scripts/configuration are read from the fixed Git commit below.
Every generated panel must match its original SHA-256 before it is published.
This tool never reads or creates the independent validation panel or job 26601's
empirical receipt. Run --help for the public inputs and local directory layout.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from importlib.util import module_from_spec, spec_from_file_location
from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys
from tempfile import NamedTemporaryFile
from threading import Lock, local
import time
from urllib.parse import urlparse
from zipfile import ZipFile


SOURCE = "2daedf522c6d64b860277657d3067b61da3a0f2b"
TOPIC = "killtests/compositional-retrieval"
STAGES = ("retrieval-conjunction", "retrieval-rerank", "retrieval-anchor-rerank")
EXPECTED = {
    "conjunction": "9bd8a7e4bb167a84ae3e6b6836a248d9e8a92a28c51a3160ab3e599ed7e2d6cd",
    "rerank": "df51d103caa1ec7968d46254b5757789e7f834e1f3a11dae1105b2f8c63ccd3f",
    "anchor": "8f43eedbcef31fe5057c26705e97bf9f213f162612d9a15b53da11bb9bacf78d",
}
ALLOWLIST = tuple(
    f"{TOPIC}/{part}"
    for part in (
        "scripts/common.py", "scripts/e2_build_queries.py", "scripts/prospective_manifest.py",
        "configs/params.json", "configs/assets_lock.json",
        "configs/prospective_rank_manifest.json", "configs/controlled_spatial_manifest.json",
    )
) + tuple(f"codex/experiments/{stage}/{name}" for stage in STAGES for name in ("prepare.py", "panel.json"))
FORBIDDEN = "retrieval-independent-panel"


def allowed_path(value):
    path = Path(value).expanduser().resolve()
    if any(FORBIDDEN in part for part in path.parts):
        raise ValueError("Independent validation paths are forbidden")
    return path


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path):
    with allowed_path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checked_body(value, expected):
    body = canonical(value) + b"\n"
    actual = hashlib.sha256(body).hexdigest()
    if actual != expected:
        raise ValueError(f"Reconstruction differs: expected {expected}, got {actual}; no original manifest published")
    return body


def publish_exact(path, value, expected):
    path = allowed_path(path)
    body = checked_body(value, expected)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != body:
            raise ValueError(f"Refusing to replace a different manifest: {path}")
    else:
        with path.open("xb") as stream:
            stream.write(body)
    return path


def load_module(name, path):
    spec = spec_from_file_location(name, allowed_path(path))
    module = module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class Progress:
    def __init__(self, root):
        self.path = root / "rebuild-panel-status.json"
        self.started = time.monotonic()

    def __call__(self, stage, **details):
        record = {"source_commit": SOURCE, "stage": stage,
                  "elapsed_seconds": round(time.monotonic() - self.started, 1), **details}
        print(json.dumps(record, ensure_ascii=False), flush=True)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".partial")
        temporary.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
        temporary.replace(self.path)


def stage_sources(repo, root):
    revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "--verify", SOURCE], text=True).strip()
    if revision != SOURCE:
        raise ValueError("The complete fixed source commit is unavailable")
    archive = root / ".rebuild-public-source" / SOURCE
    for rel in ALLOWLIST:
        # The immutable allowlist deliberately excludes every validation-panel path.
        if FORBIDDEN in rel:
            raise AssertionError("Invalid source allowlist")
        body = subprocess.check_output(["git", "-C", str(repo), "show", f"{SOURCE}:{rel}"])
        target = archive / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != body:
            raise ValueError(f"Changed local frozen source: {target}")
        if not target.exists():
            target.write_bytes(body)
    return archive


class Images:
    """Acquire official COCO bytes only; later panel digests certify exact identity."""

    def __init__(self, root, fetch, workers, cap, progress, inspect):
        self.fetch, self.workers, self.cap = fetch, workers, cap
        self.progress, self.inspect = progress, inspect
        self.lock, self.thread = Lock(), local()
        self.charged = sum(p.stat().st_size for name in (
            "coco2017/val2017", "coco2017-rank-and/images", "coco2017-rerank/images",
            "coco2017-anchor-rerank/images") for p in (root / "data" / name).glob("*.jpg"))

    def one(self, split, directory, item, strict):
        from PIL import Image
        import requests

        image_id, shape = item
        name = f"{image_id:012d}.jpg"
        path = allowed_path(directory / name)
        if path.is_file():
            result = self.inspect(path.read_bytes(), shape)
            if strict and result[1] is not None:
                raise ValueError(f"Unexpected original COCO image shape/format: {path}")
            return image_id, result
        if not self.fetch:
            raise FileNotFoundError(f"Missing official image {split}/{name}; use --fetch-images to acquire it")
        if not hasattr(self.thread, "session"):
            self.thread.session = requests.Session()
        failure = None
        for attempt in range(4):
            # Both names address the official COCO bucket. The S3 endpoint avoids
            # certificate/proxy failures on the historical custom hostname.
            origins = ("https://s3.amazonaws.com/images.cocodataset.org",
                       "https://images.cocodataset.org", "http://images.cocodataset.org",
                       "https://s3.amazonaws.com/images.cocodataset.org")
            url = f"{origins[attempt]}/{split}/{name}"
            try:
                with self.thread.session.get(url, stream=True, timeout=(15, 45)) as response:
                    response.raise_for_status()
                    final = urlparse(response.url)
                    official = ((final.hostname == "images.cocodataset.org" and final.path == f"/{split}/{name}")
                                or (final.hostname == "s3.amazonaws.com"
                                    and final.path == f"/images.cocodataset.org/{split}/{name}"))
                    if not official:
                        raise ValueError("Unexpected image redirect")
                    chunks, size = [], 0
                    for chunk in response.iter_content(65536):
                        size += len(chunk)
                        if size > 2 * 1024**2:
                            raise ValueError("COCO image exceeds the frozen 2 MiB limit")
                        chunks.append(chunk)
                    data = b"".join(chunks)
                result = self.inspect(data, shape)
                if strict and result[1] is not None:
                    raise ValueError(f"Unexpected original COCO image shape/format: {name}")
                # Full decoding also rejects truncated content before publication.
                with Image.open(BytesIO(data)) as image:
                    image.load()
                with self.lock:
                    if self.charged + len(data) > self.cap:
                        raise ValueError("Local COCO acquisition reached --max-image-bytes")
                    self.charged += len(data)
                directory.mkdir(parents=True, exist_ok=True)
                with NamedTemporaryFile(dir=directory, suffix=".partial", delete=False) as stream:
                    temporary = Path(stream.name)
                    try:
                        stream.write(data)
                        stream.flush()
                        temporary.replace(path)
                    finally:
                        temporary.unlink(missing_ok=True)
                return image_id, result
            except (OSError, ValueError, requests.RequestException) as error:
                failure = error
                if "max-image-bytes" in str(error):
                    raise
                if attempt < 3:
                    time.sleep(min(2**attempt, 4))
        raise OSError(f"Could not acquire {split}/{name}: {failure}")

    def many(self, stage, split, directory, selected, *, strict=False):
        directory = allowed_path(directory)
        items = list(selected.items())
        self.progress(stage, images=0, total=len(items), retained_image_bytes=self.charged)
        received = {}
        with ThreadPoolExecutor(max_workers=self.workers) as workers:
            for start in range(0, len(items), 128):
                batch = items[start:start + 128]
                for image_id, record in workers.map(lambda row: self.one(split, directory, row, strict), batch):
                    received[image_id] = record
                self.progress(stage, images=len(received), total=len(items), retained_image_bytes=self.charged)
        return received


def reconstruct(args):
    root, repo, annotations, half_truths = (allowed_path(x) for x in (
        args.root, args.repo, args.annotations, args.half_truths))
    progress = Progress(root)
    progress("checking_public_inputs")
    archive = stage_sources(repo, root)
    scripts = archive / TOPIC / "scripts"
    sys.path.insert(0, str(scripts))
    sys.dont_write_bytecode = True
    common = load_module("common", scripts / "common.py")
    common.ICML, common.COCO_DIR = root, annotations.parent
    common.EXP = archive / TOPIC
    common.PARAMS_PATH = common.EXP / "configs/params.json"
    load_module("e2_build_queries", scripts / "e2_build_queries.py")
    original = load_module("prospective_manifest", scripts / "prospective_manifest.py")
    anchor = load_module("rebuild_anchor", archive / "codex/experiments/retrieval-anchor-rerank/prepare.py")
    rerank, conjunction = anchor.prior, anchor.prior.prior
    conjunction.ANNOTATIONS = annotations
    # Keep the frozen content hash while permitting an arbitrary local ZIP name.
    old_zip = next(p for p in conjunction.SOURCES if p.name == "annotations_trainval2017.zip")
    expected_zip = conjunction.SOURCES.pop(old_zip)
    conjunction.SOURCES[annotations] = expected_zip
    if file_sha(annotations) != expected_zip:
        raise ValueError("COCO annotation ZIP differs from the frozen input")
    with ZipFile(annotations) as archive_zip:
        val_ann = json.load(archive_zip.open("annotations/instances_val2017.json"))
    rank_path = common.EXP / "configs/prospective_rank_manifest.json"
    rank = json.loads(rank_path.read_bytes())
    controlled = json.loads((common.EXP / "configs/controlled_spatial_manifest.json").read_bytes())
    for path, expected in conjunction.SOURCES.items():
        if path.name != "e2_queries_main.json" and file_sha(path) != expected:
            raise ValueError(f"Changed frozen public input: {path}")
    for index, stage in enumerate(STAGES):
        bound = json.loads((archive / "codex/experiments" / stage / "panel.json").read_bytes())["manifest_sha256"]
        if bound != EXPECTED[("conjunction", "rerank", "anchor")[index]]:
            raise ValueError("Frozen panel binding changed")
    ht_ids, ht_shas, ht_files = original.heldout_assets(half_truths)
    if ht_files != rank["source"]["half_truths_parquet_sha256"]:
        raise ValueError("Half-Truths parquet bytes differ from the fixed revision")
    progress("rebuilding_original_e2_query_exclusions")
    e2_path = common.EXP / "results/headroom/26393/e2_queries_main.json"
    expected_e2 = rank["source"]["old_e2_queries_sha256"]
    if e2_path.exists():
        if file_sha(e2_path) != expected_e2:
            raise ValueError("Changed locally rebuilt E2 query data")
    else:
        import e2_build_queries
        params = common.params()["p2"]
        e2 = e2_build_queries.build(val_ann, params, params["seed"])
        body = (json.dumps(e2, indent=2, ensure_ascii=False) + "\n").encode()
        if hashlib.sha256(body).hexdigest() != expected_e2:
            raise ValueError("Rebuilt E2 queries do not match their original hash; stop before panel acquisition")
        e2_path.parent.mkdir(parents=True, exist_ok=True)
        e2_path.write_bytes(body)
    old_keys = original.old_keys(e2_path)
    if digest(sorted(old_keys)) != rank["source"]["old_e2_sorted_keys_sha256"]:
        raise ValueError("Original E2 exclusion-key digest differs")

    images = Images(root, args.fetch_images, args.workers, args.max_image_bytes, progress, rerank.inspect_image)
    val_rows = {row["id"]: [row["width"], row["height"]] for row in val_ann["images"]}
    val_shas = {}
    for part in rank["splits"].values():
        val_shas.update(zip(part["candidate_ids"], part["candidate_image_sha256"], strict=True))
        old_keys.update(tuple(q["key"]) for q in part["queries"])
    missing_val = {key: shape for key, shape in val_rows.items() if key not in val_shas}
    fetched_val = images.many("recovering_116_validation_fingerprints", "val2017",
                             root / "data/coco2017/val2017", missing_val, strict=True)
    val_shas.update({key: receipt[0] for key, receipt in fetched_val.items()})
    content = {f"{key:012d}.jpg": sha for key, sha in val_shas.items()}
    expected_val = controlled["overlap_screen"]["prior_inputs"]["COCO_val2017"]
    if len(content) != expected_val["count"] or digest(content) != expected_val["content_digest"]:
        raise ValueError("Reconstructed complete COCO validation fingerprint inventory differs")
    base_ids, base_shas = ht_ids | set(val_rows), ht_shas | set(val_shas.values())

    chosen = conjunction.roster(base_ids)
    conjunction_images = {key: shape for part in chosen.values() for key, shape in part.items()}
    images.many("acquiring_historical_conjunction_images", "train2017",
                conjunction.DATA / "images", conjunction_images, strict=True)
    progress("reconstructing_conjunction_manifest")
    first = conjunction.build(chosen, old_keys, conjunction.DATA, base_shas) | {
        "roster_sha256": digest(chosen), "prior_id_sha256": digest(sorted(base_ids)),
        "prior_image_sha256": digest(sorted(base_shas)), "prior_queries_sha256": digest(sorted(old_keys)),
        "source_zip_sha256": expected_zip, "half_truths_file_sha256": ht_files,
    }
    publish_exact(conjunction.DATA / "manifest.json", first, EXPECTED["conjunction"])
    progress("conjunction_verified", sha256=EXPECTED["conjunction"], coverage=conjunction.stats(first))

    excluded_ids, excluded_shas, excluded_keys = set(base_ids), set(base_shas), set(old_keys)
    for part in rank["splits"].values():
        for query in part["queries"]:
            for key in (query["foil_key"], query["anchor_key"]):
                if key is not None and len(key) == 2:
                    excluded_keys.add(tuple(sorted(key)))
    for part in first["parts"].values():
        excluded_ids.update(part["ids"])
        excluded_shas.update(part["jpg_sha256"])
        excluded_keys.update(tuple(q["key"]) for q in part["queries"])
    if controlled["overlap_screen"]["excluded_groups"] or len(controlled["groups"]) != 102:
        raise ValueError("Controlled-image Git fingerprints do not cover the original complete release")
    controlled_shas = {sha for group in controlled["groups"] for sha in group["image_hashes"]}
    if len(controlled_shas) != 408:
        raise ValueError("Expected 408 unique public controlled-image fingerprints")
    excluded_shas.update(controlled_shas)
    roster = rerank.select_roster(excluded_ids)
    received = images.many("acquiring_historical_rerank_images", "train2017", rerank.DATA / "images", roster)
    names, categories = conjunction.categories({"test": roster})
    ids, hashes, mismatches, seen = [], [], [], set(excluded_shas)
    for image_id in sorted(roster):
        sha, issue = received[image_id]
        if issue is not None:
            mismatches.append({"id": image_id, "sha256": sha, **issue})
        elif sha not in seen:
            seen.add(sha)
            ids.append(image_id)
            hashes.append(sha)
    queries = rerank.annotations(ids, categories, excluded_keys)
    second = {"schema": 1, "ids": ids, "jpeg_sha256": hashes, "queries": queries, "names": names,
              "source_mismatches": mismatches, "roster_sha256": digest(roster),
              "excluded_id_sha256": digest(sorted(excluded_ids)),
              "excluded_jpeg_sha256": digest(sorted(excluded_shas)),
              "excluded_keys_sha256": digest(sorted(excluded_keys)), "half_truths_sha256": ht_files,
              "controlled_archive_sha256": rerank.CONTROLLED_SHA,
              "conjunction_manifest_sha256": EXPECTED["conjunction"], "annotations_sha256": expected_zip}
    publish_exact(rerank.DATA / "manifest.json", second, EXPECTED["rerank"])
    progress("rerank_verified", sha256=EXPECTED["rerank"], coverage=rerank.coverage(ids, queries))

    excluded_ids.update(ids)
    excluded_shas.update(hashes)
    excluded_keys.update(tuple(q["key"]) for q in queries)
    for asset in mismatches:
        excluded_ids.add(asset["id"])
        excluded_shas.add(asset["sha256"])
    roster = anchor.roster(excluded_ids)  # Independently binds the original 8,192 IDs and dimensions.
    received = images.many("acquiring_development_anchor_images", "train2017", anchor.DATA / "images", roster)
    names, categories = conjunction.categories({"test": roster})
    ids, hashes, rejected = anchor.assemble(roster, excluded_shas, received)
    queries = anchor.with_anchors(rerank.annotations(ids, categories, excluded_keys), names)
    final = {"schema": 1, "ids": ids, "jpeg_sha256": hashes, "queries": queries, "names": names,
             "rejected": rejected, "roster_sha256": digest(roster),
             "previous_rerank_sha256": EXPECTED["rerank"],
             "excluded_ids_sha256": digest(sorted(excluded_ids)),
             "excluded_jpegs_sha256": digest(sorted(excluded_shas)),
             "excluded_queries_sha256": digest(sorted(excluded_keys)),
             "coco_annotations_sha256": expected_zip, "half_truths_sha256": ht_files,
             "near_duplicate_screened": False}
    publish_exact(anchor.DATA / "manifest.json", final, EXPECTED["anchor"])
    coverage = anchor.check_coverage(ids, queries)
    if (coverage["candidates"], coverage["queries"], coverage["groups"]) != (8191, 136, 118):
        raise ValueError("Unexpected original development-panel coverage")
    progress("complete_original_development_panel_verified", sha256=EXPECTED["anchor"], coverage=coverage,
             manifest=str(anchor.DATA / "manifest.json"),
             empirical_receipt_26601="not created; original execution receipt remains unavailable",
             independent_validation_panel="never accessed")


def self_test():
    import unittest

    class IntegrityTests(unittest.TestCase):
        def test_manifest_hash_includes_original_newline(self):
            value = {"x": 1}
            expected = hashlib.sha256(b'{"x":1}\n').hexdigest()
            self.assertEqual(checked_body(value, expected), b'{"x":1}\n')

        def test_changed_manifest_is_rejected(self):
            expected = hashlib.sha256(b'{"x":1}\n').hexdigest()
            with self.assertRaisesRegex(ValueError, "no original manifest published"):
                checked_body({"x": 2}, expected)

        def test_independent_panel_path_is_rejected(self):
            with self.assertRaisesRegex(ValueError, "forbidden"):
                allowed_path("/tmp/retrieval-independent-panel/manifest.json")

        def test_sources_are_immutable_and_exposed_only(self):
            self.assertEqual(len(SOURCE), 40)
            self.assertEqual(len(ALLOWLIST), 13)
            self.assertTrue(all(FORBIDDEN not in path for path in ALLOWLIST))

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(IntegrityTests)
    if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--root", type=Path, help="Local data root; no /work mount is required")
    parser.add_argument("--annotations", type=Path, help="Original COCO annotations_trainval2017.zip")
    parser.add_argument("--half-truths", type=Path,
                        help="Fixed ddcd448... snapshot containing data/{coco_qwen,coco_mistral,cc3m_qwen}.parquet")
    parser.add_argument("--source", choices=(SOURCE, SOURCE[:7]), default=SOURCE,
                        help="Only the original exposed-panel source commit is accepted")
    parser.add_argument("--fetch-images", action="store_true",
                        help="Acquire missing official COCO images (about 4 GiB across all historical panels)")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--max-image-bytes", type=int, default=6 * 1024**3,
                        help="Cap for retained JPEG bytes across this reconstruction")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if any(getattr(args, key) is None for key in ("root", "annotations", "half_truths")):
        parser.error("--root, --annotations and --half-truths are required")
    if not 1 <= args.workers <= 32 or args.max_image_bytes <= 0:
        parser.error("Require 1–32 workers and a positive image byte cap")
    try:
        reconstruct(args)
    except Exception as error:
        print(json.dumps({"stage": "failed", "error": str(error), "source_commit": SOURCE}), flush=True)
        raise


if __name__ == "__main__":
    main()
