"""[codex] Pin official SugarCrepe JSONs and reuse existing COCO image bytes."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from codex.retrieval_matrix_native_p0.evidence import digest, entry, read, sha, write

REPO = "RAIVNLab/sugar-crepe"
REVISION = "0047054b243992f0fad63d6f64f7544862daf846"
CATEGORIES = ("add_obj", "add_att", "replace_obj", "replace_att", "replace_rel", "swap_obj", "swap_att")
EXPECTED = {"add_obj": 2062, "add_att": 692, "replace_obj": 1652, "replace_att": 788,
            "replace_rel": 1406, "swap_obj": 245, "swap_att": 666}
BASE = ROOT/"dl/retrieval-matrix-paired-p0"
SOURCE = BASE/"sources/sugarcrepe"


def download():
    def fetch(url):
        return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "codex-fixed-paired-p0"}), timeout=60).read()
    tree = json.loads(fetch(f"https://api.github.com/repos/{REPO}/git/trees/{REVISION}?recursive=1"))
    upstream = {row["path"]: row for row in tree["tree"] if row["type"] == "blob"}
    names = ["README.md", "LICENSE", "main_eval.py"] + ["data/"+name+".json" for name in CATEGORIES]
    def one(name):
        path = SOURCE/name
        body = path.read_bytes() if path.exists() else fetch(f"https://raw.githubusercontent.com/{REPO}/{REVISION}/{name}")
        blob = hashlib.sha1(f"blob {len(body)}\0".encode()+body).hexdigest()
        if blob != upstream[name]["sha"]:
            raise ValueError("Official fixed SugarCrepe Git blob differs: " + name)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(body)
        return {"file": name, "bytes": len(body), "git_blob_sha1": blob, "sha256": hashlib.sha256(body).hexdigest()}
    with ThreadPoolExecutor(max_workers=4) as pool:
        files = list(pool.map(one, names))
    # The initial source inventory can have a different row order; compare by
    # path rather than rewriting an already recorded upstream receipt.
    receipt = SOURCE/"source-receipt.json"
    if receipt.exists():
        old = read(receipt)
        if (old["repo"] != REPO or old["revision"] != REVISION
                or {r["file"]: r for r in old["files"]} != {r["file"]: r for r in files}):
            raise ValueError("Existing official source receipt differs")
    else:
        write(receipt, {"repo": REPO, "revision": REVISION, "files": files})


def prepare():
    source = read(SOURCE/"source-receipt.json")
    for row in source["files"]:
        if sha(SOURCE/row["file"]) != row["sha256"]:
            raise ValueError("SugarCrepe source payload changed")
    image_root = ROOT/"dl/retrieval3090/data/coco2017/val2017"
    datasets = {"sugarcrepe": SOURCE/"data", "sugarcrepe_pp": ROOT/"dl/retrieval-matrix/data/sugarcrepe_pp/data"}
    names = set()
    counts = {}
    identical = {}
    sources = {}
    for benchmark, folder in datasets.items():
        counts[benchmark], identical[benchmark], sources[benchmark] = {}, {}, {}
        categories = CATEGORIES if benchmark == "sugarcrepe" else CATEGORIES[2:]
        for category in categories:
            path = folder/(category+".json")
            records = read(path)
            records = list(records.values()) if isinstance(records, dict) else records
            if len(records) != EXPECTED[category]:
                raise ValueError("Official paired subset denominator differs: " + category)
            counts[benchmark][category] = len(records)
            identical[benchmark][category] = sum(row["caption"] == row["negative_caption"] for row in records)
            names.update(row["filename"] for row in records)
            sources[benchmark][category] = entry(path)
    images = []
    for name in sorted(names):
        path = image_root/name
        if not path.is_file():
            raise ValueError("Missing canonical COCO val2017 image: " + name)
        from PIL import Image
        with Image.open(path) as picture:
            picture.verify()
        images.append({"filename": name, "path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha(path)})
    path = BASE/"image-receipt.jsonl"
    body = "".join(json.dumps(row, sort_keys=True)+"\n" for row in images)
    if path.exists() and path.read_text() != body:
        raise ValueError("Paired original image receipt changed")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
    result = {"owner": "[codex]", "state": "DATA_READY", "source": entry(SOURCE/"source-receipt.json"),
        "sugarcrepe_revision": REVISION, "sugarcrepe_pp_revision": "dea2a1b6f9e1069c609f676aa55ec61e9b65fb61",
        "subsets": sources, "counts": counts, "total_rows": {key: sum(value.values()) for key, value in counts.items()},
        "image_receipt": entry(path), "unique_images": len(images), "missing_images": 0,
        "identical_positive_negative_rows_retained": identical,
        "dataset_relations": "SugarCrepe++ edits original captions as well as adding caption2. Never substitute SugarCrepe text or filter identical captions."}
    write(BASE/"data-audit.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if args.download:
        download()
    result = prepare()
    print(json.dumps({key: result[key] for key in ("state", "total_rows", "unique_images", "missing_images")}))
