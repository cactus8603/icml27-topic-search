#!/usr/bin/env python3
"""[claude] Acquire the train2017 JPEGs of the conjunction panel galleries (CPU, network).

For each gallery image: reuse a byte copy from a local COCO image folder if present, else download coco_url.
Every file must decode with PIL and match the annotated width/height; its SHA-256 and source are written to a
receipt. Resumable: existing verified files are re-hashed and kept.
usage: python fetch_images.py --panel <conj-panel-v1> --local-dir DIR [--local-dir DIR ...] [--workers 16]
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import re
import shutil
import time
import urllib.request
from pathlib import Path

from PIL import Image


def sha(path):
    d = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            d.update(chunk)
    return d.hexdigest()


def verify(path, w, hgt):
    with Image.open(path) as im:
        im.load()
        return im.size == (w, hgt)


def local_index(dirs):
    idx = {}
    for d in dirs:
        for root, _, files in os.walk(d):
            for f in files:
                m = re.match(r"0*(\d+)\.jpe?g$", f, re.I)
                if m:
                    idx.setdefault(int(m.group(1)), os.path.join(root, f))
    return idx


def fetch(item, dest, local, tmpdir):
    out = dest / item["file_name"]
    if out.exists() and verify(out, item["width"], item["height"]):
        return {"id": item["id"], "sha256": sha(out), "bytes": out.stat().st_size, "source": "existing"}
    tmp = Path(tmpdir) / f"{item['id']}.part"
    if item["id"] in local:
        shutil.copyfile(local[item["id"]], tmp)
        source = "local:" + local[item["id"]]
    else:
        for attempt in range(5):
            try:
                with urllib.request.urlopen(item["coco_url"], timeout=60) as r, open(tmp, "wb") as f:
                    shutil.copyfileobj(r, f)
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
        source = "remote:" + item["coco_url"]
    if not verify(tmp, item["width"], item["height"]):
        tmp.unlink()
        raise ValueError(f"size mismatch or undecodable: {item['id']}")
    os.replace(tmp, out)
    return {"id": item["id"], "sha256": sha(out), "bytes": out.stat().st_size, "source": source}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--panel", required=True)
    p.add_argument("--local-dir", action="append", default=[])
    p.add_argument("--workers", type=int, default=16)
    a = p.parse_args()
    panel = Path(a.panel)
    tmpdir = os.environ.get("TMPDIR")
    if not tmpdir or tmpdir.startswith("/tmp"):
        raise SystemExit("set TMPDIR under the repo (hf_home/tmp), not /tmp")
    local = local_index(a.local_dir)
    summary = {}
    for split in ("dev", "test"):
        gallery = json.loads((panel / split / "gallery.json").read_text())
        dest = panel / split / "images"
        dest.mkdir(exist_ok=True)
        rows, failures = [], []
        t0 = time.time()
        with cf.ThreadPoolExecutor(a.workers) as ex:
            futs = {ex.submit(fetch, it, dest, local, tmpdir): it["id"] for it in gallery}
            for n, fu in enumerate(cf.as_completed(futs), 1):
                try:
                    rows.append(fu.result())
                except Exception as e:  # recorded, never silently dropped
                    failures.append({"id": futs[fu], "error": repr(e)})
                if n % 2000 == 0:
                    print(f"{split} {n}/{len(gallery)} {time.time() - t0:.0f}s failures={len(failures)}", flush=True)
        rows.sort(key=lambda r: r["id"])
        receipt = panel / split / "images-receipt.jsonl"
        with open(receipt, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        summary[split] = {"expected": len(gallery), "ok": len(rows), "failures": failures,
                          "bytes": sum(r["bytes"] for r in rows),
                          "sources": {s: sum(r["source"].startswith(s) for r in rows)
                                      for s in ("existing", "local:", "remote:")},
                          "receipt_sha256": sha(receipt)}
    (panel / "images-summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({s: {k: v for k, v in d.items() if k != "failures"} | {"failures": len(d["failures"])}
                      for s, d in summary.items()}, indent=1))
    if any(d["failures"] for d in summary.values()):
        raise SystemExit("some images failed; rerun to retry (verified files are kept)")


if __name__ == "__main__":
    main()
