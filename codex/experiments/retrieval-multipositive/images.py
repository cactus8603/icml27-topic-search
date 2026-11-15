"""Acquire or verify the fixed multi-positive panel's official COCO JPEGs."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
from importlib.util import module_from_spec, spec_from_file_location
import json
import os
from pathlib import Path
import re
import sys
from tarfile import open as open_tar
from tempfile import NamedTemporaryFile
from threading import Lock
from urllib.request import HTTPRedirectHandler, build_opener

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from panel import COCO_DIR, COCO_ORIGIN, canonical_bytes, digest, file_sha, prior_inventory, verified_json


spec = spec_from_file_location("rerank_prepare", HERE.parent / "retrieval-rerank" / "prepare.py")
rerank = module_from_spec(spec)
spec.loader.exec_module(rerank)

PLAN_SHA = "33363955507ec1c7cad716c3294f41c7892b12d944760631ce42fb85894295bb"
HTTPS_OBJECTS = "https://s3.amazonaws.com/images.cocodataset.org/train2017/"
IMAGE_COUNT = 984
MAX_IMAGE_BYTES = rerank.prior.MAX_IMAGE_BYTES
CAP = 512 * 1024**2
LEDGER = "download-ledger.jsonl"
RECEIPT = "image-receipt.json"
SHA256 = re.compile(r"[0-9a-f]{64}")


def read_plan(path):
    plan = verified_json(path, PLAN_SHA)
    assets = plan["images"]
    ids = [str(image_id) for group in plan["groups"] for image_id in group["images"]]
    if (plan["schema"] != 1 or plan["status"] != "source_ids_verified_only"
            or len(assets) != IMAGE_COUNT or len(ids) != IMAGE_COUNT or set(ids) != assets.keys()
            or plan["coverage"]["groups"] != len(plan["groups"]) or plan["coverage"]["images"] != len(assets)):
        raise ValueError("Malformed fixed metadata roster")
    return assets, plan["exclusions"]["known_prior_image_shas_sha256"]


def exclusions(expected):
    known = prior_inventory().known_image_shas
    if digest(sorted(known)) != expected:
        raise ValueError("Previously registered image inventory differs from the fixed plan")
    # Older COCO retrieval panels also screened the full validation pool and Controlled Images archive.
    validation = list((COCO_DIR / "val2017").glob("*.jpg"))
    if len(validation) != 5000 or file_sha(rerank.CONTROLLED) != rerank.CONTROLLED_SHA:
        raise ValueError("Prior validation JPEGs or Controlled Images archive are incomplete or changed")
    blocked = known | {file_sha(path) for path in validation}
    with open_tar(rerank.CONTROLLED, "r|gz") as archive:
        for member in archive:
            if member.isfile() and member.name.lower().endswith((".jpg", ".jpeg", ".png")):
                with archive.extractfile(member) as stream:
                    blocked.add(hashlib.sha256(stream.read()).hexdigest())
    return blocked


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def download(image_id, asset, reserve):
    filename = f"{int(image_id):012d}.jpg"
    source = f"{COCO_ORIGIN}train2017/{filename}"
    if asset["file_name"] != filename or asset["url"] != source:
        raise ValueError(f"Unexpected official COCO filename or URL for {image_id}")
    # The official image hostname is a CNAME for this bucket. Use AWS's path-style TLS endpoint.
    url = HTTPS_OBJECTS + filename
    with build_opener(NoRedirect()).open(url, timeout=35) as response:
        lengths = response.headers.get_all("Content-Length") or []
        if (response.geturl() != url or response.status != 200 or len(lengths) != 1
                or re.fullmatch(r"[0-9]{1,8}", lengths[0]) is None
                or response.headers.get("Transfer-Encoding") is not None
                or response.headers.get("Content-Encoding", "identity").lower() != "identity"):
            raise ValueError(f"Unexpected official source response for {image_id}")
        size = int(lengths[0])
        reserve(image_id, size)
        data = response.read(size)
    if len(data) != size:
        raise ValueError(f"Incomplete official JPEG for {image_id}")
    return data


def publish(path, data):
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".partial",
                            delete=False) as stream:
        pending = Path(stream.name)
        try:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
            os.link(pending, path)
        finally:
            pending.unlink()


class DownloadLedger:
    def __init__(self, stream, assets):
        self.stream = stream
        self.assets = assets
        self.charged = 0
        self.pending = {}
        self.complete = {}
        self.lock = Lock()
        self.stream.seek(0)
        for line in self.stream:
            event = json.loads(line)
            if line != canonical_bytes(event) + b"\n":
                raise ValueError("Malformed download ledger row")
            self.apply(event)

    def apply(self, event):
        image_id, size = event["id"], event["bytes"]
        if (event["plan_sha256"] != PLAN_SHA or image_id not in self.assets
                or type(size) is not int or not 0 < size <= MAX_IMAGE_BYTES):
            raise ValueError("Unexpected plan, image ID, or image size in the download ledger")
        if event.keys() == {"plan_sha256", "id", "bytes"}:
            if self.charged + size > CAP:
                raise ValueError("Acquisition reached the cumulative 512 MiB cap")
            self.charged += size
            self.pending[image_id] = size
        elif event.keys() == {"plan_sha256", "id", "bytes", "sha256"}:
            sha = event["sha256"]
            record = (size, sha)
            if (self.pending.pop(image_id, None) != size or not isinstance(sha, str) or SHA256.fullmatch(sha) is None
                    or self.complete.get(image_id, record) != record):
                raise ValueError(f"Unexpected or changed completed JPEG for image {image_id}")
            self.complete[image_id] = record
        else:
            raise ValueError("Unrecognized download ledger row")

    def append(self, event):
        event["plan_sha256"] = PLAN_SHA
        with self.lock:
            self.apply(event)
            self.stream.write(canonical_bytes(event) + b"\n")
            self.stream.flush()
            os.fsync(self.stream.fileno())

    def reserve(self, image_id, size):
        self.append({"id": image_id, "bytes": size})


def inspect(image_id, asset, data):
    if not 0 < len(data) <= MAX_IMAGE_BYTES:
        raise ValueError(f"JPEG size is out of bounds for {image_id}")
    try:
        sha, issue = rerank.inspect_image(data, [asset["width"], asset["height"]])
    except OSError as error:
        raise ValueError(f"Could not fully decode JPEG for {image_id}") from error
    if issue is not None:
        raise ValueError(f"Wrong JPEG format or annotated dimensions for {image_id}: {issue}")
    return len(data), sha


def image_inventory(mode, directory, assets, blocked, ledger):
    allowed = {asset["file_name"] for asset in assets.values()}
    if any(path.name not in allowed for path in directory.glob("*.jpg")):
        raise ValueError("Image directory contains a JPEG outside the fixed roster")
    seen = dict.fromkeys(blocked)
    for image_id, (_, sha) in ledger.complete.items():
        if seen.setdefault(sha, image_id) != image_id:
            raise ValueError(f"Exact JPEG duplicate in download ledger for image {image_id}")
    seen_lock = Lock()

    def record(item):
        image_id, asset = item
        path = directory / asset["file_name"]
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise ValueError(f"Expected a regular JPEG file for {image_id}")
        existing = path.is_file()
        if existing:
            with path.open("rb") as stream:
                data = stream.read(MAX_IMAGE_BYTES + 1)
        elif mode == "fetch":
            data = download(image_id, asset, ledger.reserve)
        else:
            raise FileNotFoundError(f"Missing fixed image {image_id}")
        size, sha = inspect(image_id, asset, data)
        with seen_lock:
            if seen.setdefault(sha, image_id) != image_id:
                raise ValueError(f"Exact JPEG duplicate for image {image_id}")
        if existing:
            if ledger.complete.get(image_id) != (size, sha):
                raise ValueError(f"Existing file disagrees with its download ledger for {image_id}")
        else:
            ledger.append({"id": image_id, "bytes": size, "sha256": sha})
            publish(path, data)
        return image_id, {"bytes": size, "sha256": sha}

    rows = list(assets.items())
    received = {}
    with ThreadPoolExecutor(max_workers=16) as pool:
        for start in range(0, len(rows), 64):
            received.update(pool.map(record, rows[start:start + 64]))
            if mode == "fetch" and (len(received) % 128 == 0 or len(received) == len(rows)):
                print(json.dumps({"verified_images": len(received), "charged_bytes": ledger.charged}), flush=True)
    return received


def prepare(mode, plan_path, directory):
    assets, prior_sha = read_plan(plan_path)
    blocked = exclusions(prior_sha)
    images = directory / "images"
    receipt = directory / RECEIPT
    ledger_path = directory / LEDGER
    if any(path.is_symlink() for path in (directory, images, receipt, ledger_path)):
        raise ValueError("Panel directories, ledger and receipt must not be symlinks")
    if mode == "fetch":
        images.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a+b" if mode == "fetch" else "rb") as stream:
        lock = fcntl.LOCK_EX if mode == "fetch" else fcntl.LOCK_SH
        fcntl.flock(stream, lock | fcntl.LOCK_NB)
        if mode == "fetch" and receipt.exists():
            raise FileExistsError("Image receipt already exists. Run verify.")
        ledger = DownloadLedger(stream, assets)
        entries = image_inventory(mode, images, assets, blocked, ledger)
        body = canonical_bytes({
            "schema": 1, "plan_sha256": PLAN_SHA, "download_origin": HTTPS_OBJECTS,
            "excluded_image_sha256": digest(sorted(blocked)),
            "image_sha256_by_id": entries, "image_bytes": sum(row["bytes"] for row in entries.values()),
            "charged_bytes": ledger.charged, "cap_bytes": CAP, "ledger_sha256": file_sha(ledger_path),
        }) + b"\n"
        if mode == "fetch":
            publish(receipt, body)
        elif receipt.read_bytes() != body:
            raise ValueError("Offline image and ledger reconstruction differs from the receipt")
    return {"images": len(entries), "image_receipt_sha256": hashlib.sha256(body).hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("fetch", "verify"))
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--directory", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.mode, args.plan, args.directory), sort_keys=True), flush=True)
