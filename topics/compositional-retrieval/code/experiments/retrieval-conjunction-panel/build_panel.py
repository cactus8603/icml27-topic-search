#!/usr/bin/env python3
"""[claude] Build the object-conjunction retrieval panels (dev / test) from COCO train2017 instance labels.

Spec: SPEC.md in this folder (frozen before generation). CPU only; reads labels, never model scores.

usage:
  python build_panel.py build --annotations-zip <annotations_trainval2017.zip> --exposed <exposed.json> --out <dir>
  python build_panel.py exposed --out <exposed.json> --source NAME=PATH ...   (collect exposed train2017 IDs)
"""
import argparse
import collections
import hashlib
import io
import itertools
import json
import re
import sys
import zipfile
from pathlib import Path

SEED = "conj-panel-2026111601"
SMALL_SEED = "conj-panel-dev-small-2026111601"
ANNOTATIONS_ZIP_SHA256 = "113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268"
TAU = 0.01            # an instance counts as visible if non-crowd and area >= TAU * image area
GALLERY_SIZE = 20000  # images per gallery (dev and test)
KS = (2, 3, 4)
POS_MIN, POS_MAX = 1, 20
HARDNEG_MIN = 5       # images with k-1 conditions present and the remaining one absent
DEV_SMALL_PER_K = 84  # 3 x 84 = 252 queries


def h(*parts):
    return hashlib.sha256(":".join(str(p) for p in parts).encode()).hexdigest()


def sha_file(path):
    d = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            d.update(chunk)
    return d.hexdigest()


# ------------------------------------------------------------------------------------------- labels
def image_status(images, annotations, tau=TAU):
    """-> present[iid] = set(cat), ambiguous[iid] = set(cat) (annotated but no visible non-crowd instance)."""
    by_img = collections.defaultdict(list)
    for a in annotations:
        by_img[a["image_id"]].append(a)
    present, ambiguous = {}, {}
    for im in images:
        iid, area = im["id"], im["width"] * im["height"]
        seen, vis = set(), set()
        for a in by_img.get(iid, []):
            seen.add(a["category_id"])
            if not a["iscrowd"] and a["area"] >= tau * area:
                vis.add(a["category_id"])
        present[iid], ambiguous[iid] = vis, seen - vis
    return present, ambiguous


def build_queries(gallery, present, ambiguous, cat_names, split):
    """Candidate conjunctions over one gallery, then round-robin over k with positives disjoint across queries."""
    by_cat = collections.defaultdict(set)
    for i in gallery:
        for c in present[i]:
            by_cat[c].add(i)
    pools = {}
    for k in KS:
        pos = collections.defaultdict(set)
        for i in gallery:
            for combo in itertools.combinations(sorted(present[i]), k):
                pos[combo].add(i)
        cands = []
        for combo, P in pos.items():
            if not POS_MIN <= len(P) <= POS_MAX:
                continue
            hard = set()
            for miss in combo:
                rest = set.intersection(*[by_cat[c] for c in combo if c != miss])
                hard |= {i for i in rest if miss not in present[i] and miss not in ambiguous[i]}
            if len(hard) < HARDNEG_MIN:
                continue
            cands.append((h(SEED, split, k, combo), combo, P, len(hard)))
        cands.sort()
        pools[k] = cands
    used, selected = set(), []
    iters = {k: iter(pools[k]) for k in KS}
    alive = list(KS)
    while alive:
        for k in list(alive):
            for _, combo, P, n_hard in iters[k]:
                if P & used:
                    continue
                used |= P
                S = set(combo)
                ignore = sorted(i for i in gallery
                                if i not in P and not (S - present[i] - ambiguous[i]) and (S & ambiguous[i]))
                selected.append({"k": k, "category_ids": list(combo), "positives": sorted(P), "ignore": ignore,
                                 "n_hard_negatives": n_hard})
                break
            else:
                alive.remove(k)
    for q in selected:
        order = sorted(q["category_ids"], key=lambda c: h(SEED, "order", c))
        names = [cat_names[c] for c in order]
        q["names"] = names
        q["text"] = query_text(names)
        q["id"] = f"{split}-k{q['k']}-" + "-".join(str(c) for c in sorted(q["category_ids"]))
        q["group"] = q["id"]  # positives are disjoint across queries, so every query is its own group
    selected.sort(key=lambda q: q["id"])
    return selected


def article(name):
    return ("an " if name[0] in "aeiou" else "a ") + name


def query_text(names):
    items = [article(n) for n in names]
    listed = items[0] + " and " + items[1] if len(items) == 2 else ", ".join(items[:-1]) + ", and " + items[-1]
    return f"A photo that contains {listed}."


def dev_small(queries):
    out = []
    for k in KS:
        qs = sorted((q for q in queries if q["k"] == k), key=lambda q: h(SMALL_SEED, q["id"]))
        out += [q["id"] for q in qs[:DEV_SMALL_PER_K]]
    return sorted(out)


# ------------------------------------------------------------------------------------------- driver
def load_annotations(zip_path):
    actual = sha_file(zip_path)
    if actual != ANNOTATIONS_ZIP_SHA256:
        sys.exit(f"annotations zip SHA mismatch: {actual}")
    with zipfile.ZipFile(zip_path) as z:
        return json.load(io.TextIOWrapper(z.open("annotations/instances_train2017.json"), encoding="utf-8"))


def build(a):
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        sys.exit(f"refusing to overwrite non-empty {out}")
    exposed_doc = json.loads(Path(a.exposed).read_text())
    exposed = set(exposed_doc["ids"])
    d = load_annotations(a.annotations_zip)
    images = d["images"]
    cat_names = {c["id"]: c["name"] for c in d["categories"]}
    present, ambiguous = image_status(images, d["annotations"])
    ids = sorted((im["id"] for im in images), key=lambda i: h(SEED, "pool", i))
    test_gallery = [i for i in ids if i not in exposed][:GALLERY_SIZE]
    taken = set(test_gallery)
    dev_gallery = [i for i in ids if i not in taken][:GALLERY_SIZE]
    assert not set(dev_gallery) & taken and not set(test_gallery) & exposed
    meta = {im["id"]: im for im in images}
    manifest = {"spec": "SPEC.md", "seed": SEED, "tau": TAU, "gallery_size": GALLERY_SIZE, "ks": list(KS),
                "positives_range": [POS_MIN, POS_MAX], "hard_negative_min": HARDNEG_MIN,
                "annotations_zip_sha256": ANNOTATIONS_ZIP_SHA256, "exposed_sha256": sha_file(a.exposed),
                "exposed_count": len(exposed), "builder_sha256": sha_file(__file__), "splits": {}}
    for split, gallery in (("dev", dev_gallery), ("test", test_gallery)):
        qs = build_queries(gallery, present, ambiguous, cat_names, split)
        sd = out / split
        sd.mkdir(parents=True)
        gal = [{"id": i, "file_name": meta[i]["file_name"], "coco_url": meta[i]["coco_url"],
                "width": meta[i]["width"], "height": meta[i]["height"]} for i in sorted(gallery)]
        (sd / "gallery.json").write_text(json.dumps(gal) + "\n")
        with open(sd / "queries.jsonl", "w") as f:
            for q in qs:
                f.write(json.dumps(q) + "\n")
        info = {"gallery_images": len(gal), "queries": len(qs),
                "queries_per_k": {k: sum(q["k"] == k for q in qs) for k in KS},
                "positives_total": sum(len(q["positives"]) for q in qs),
                "gallery_sha256": sha_file(sd / "gallery.json"), "queries_sha256": sha_file(sd / "queries.jsonl")}
        if split == "dev":
            small = dev_small(qs)
            (sd / "dev_small.json").write_text(json.dumps(small) + "\n")
            info.update(dev_small_queries=len(small), dev_small_sha256=sha_file(sd / "dev_small.json"))
        manifest["splits"][split] = info
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({s: {k: v for k, v in i.items() if not k.endswith("sha256")}
                      for s, i in manifest["splits"].items()}, indent=1))


def collect_exposed(a):
    """Union of exposed train2017 IDs. Each source: NAME=PATH where PATH is a JSON file (lists under known keys),
    a directory of COCO JPEGs (IDs from file names) or a Half-Truths parquet (COCO source_image_id)."""
    ids, sources = set(), []
    for spec in a.source:
        name, path = spec.split("=", 1)
        p = Path(path)
        got = set()
        if p.is_dir():
            for f in p.rglob("*"):
                m = re.match(r"0*(\d+)\.(jpg|jpeg|png)$", f.name, re.I)
                if m:
                    got.add(int(m.group(1)))
            digest = h("ids", sorted(got))
        elif p.suffix == ".parquet":
            import pyarrow.parquet as pq
            for v in pq.read_table(p, columns=["source_image_id"]).column(0).to_pylist():
                m = re.search(r"COCO_\w+?_0*(\d+)", str(v))
                if m:
                    got.add(int(m.group(1)))
            digest = sha_file(p)
        else:
            doc = json.loads(p.read_text())
            for key in ("ids", "gallery_ids", "excluded_image_ids"):
                got |= set(doc.get(key, []))
            got |= {x["image_id"] for x in doc.get("selected", [])}
            digest = sha_file(p)
        ids |= got
        sources.append({"name": name, "path": str(p), "sha256_or_id_digest": digest, "ids": len(got)})
    Path(a.out).write_text(json.dumps({"sources": sources, "ids": sorted(ids)}) + "\n")
    print(json.dumps({"sources": sources, "union": len(ids)}, indent=1))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--annotations-zip", required=True)
    b.add_argument("--exposed", required=True)
    b.add_argument("--out", required=True)
    e = sub.add_parser("exposed")
    e.add_argument("--out", required=True)
    e.add_argument("--source", action="append", required=True)
    a = p.parse_args()
    build(a) if a.cmd == "build" else collect_exposed(a)


if __name__ == "__main__":
    main()
