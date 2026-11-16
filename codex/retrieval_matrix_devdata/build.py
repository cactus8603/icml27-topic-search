"""[codex] Frozen annotation-only AND+NOT development panel; never opens test."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[2]
SEED = "neg-dev-20261116-v1"
SMALL_SEED = "neg-dev-small-20261116-v1"
ANNOTATIONS_SHA = "113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268"
GALLERY_SHA = "47cedc97b553ad38e10cbe796a19ad5a3c3135ac4f959fbef1e2e2c80f108c5e"


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def h(*values):
    return hashlib.sha256(":".join(map(str, values)).encode()).hexdigest()


def indices(bits):
    while bits:
        lowest = bits & -bits
        yield lowest.bit_length()-1
        bits ^= lowest


def status(images, annotations):
    lookup = {row["id"]: i for i, row in enumerate(images)}
    seen, present = {}, {}
    for row in annotations:
        if row["image_id"] not in lookup:
            continue
        i = lookup[row["image_id"]]
        c = row["category_id"]
        seen[c] = seen.get(c, 0) | (1 << i)
        image = images[i]
        if not row["iscrowd"] and row["area"] >= .01*image["width"]*image["height"]:
            present[c] = present.get(c, 0) | (1 << i)
    universe = (1 << len(images))-1
    return present, {c: universe & ~v for c, v in seen.items()}


def phrase(name):
    if name in {"scissors", "skis"}:
        return name
    return ("an " if name[0] in "aeiou" else "a ")+name


def template(names, excluded):
    names = [phrase(x) for x in names]
    content = " and ".join(names) if len(names) == 2 else ", ".join(names[:-1])+", and "+names[-1]
    return f"A photo that contains {content}, but no {excluded}."


def candidates(images, present, absent, category_names):
    by_image = [[] for _ in images]
    for category, bits in present.items():
        for i in indices(bits):
            by_image[i].append(category)
    combos = {}
    for i, categories in enumerate(by_image):
        for k in (2, 3, 4):
            for combo in itertools.combinations(sorted(categories), k):
                combos[combo] = combos.get(combo, 0) | (1 << i)
    pools = {k: [] for k in (2, 3, 4)}
    for combo, all_positive in combos.items():
        if not 6 <= all_positive.bit_count() <= 200:
            continue
        for excluded in sorted(category_names):
            if excluded in combo:
                continue
            hard = all_positive & present.get(excluded, 0)
            positives = all_positive & absent[excluded]
            if hard.bit_count() >= 5 and 1 <= positives.bit_count() <= 20:
                pools[len(combo)].append((h(SEED, len(combo), combo, excluded), combo, excluded, positives, hard.bit_count()))
    return {k: sorted(value) for k, value in pools.items()}


def select(pools, images, present, absent, names):
    universe, used, result = (1 << len(images))-1, 0, []
    iters = {k: iter(value) for k, value in pools.items()}
    alive = [2, 3, 4]
    while alive:
        for k in list(alive):
            for _, combo, excluded, positive, hard_count in iters[k]:
                if positive & used:
                    continue
                used |= positive
                negatives = present.get(excluded, 0)
                for category in combo:
                    negatives |= absent[category]
                ignore = universe & ~(positive | negatives)
                ordered = sorted(combo, key=lambda c: h(SEED, "order", c))
                qid = "dev-andnot-"+"-".join(map(str, combo))+"-not-"+str(excluded)
                result.append({"id": qid, "group": qid, "k": k, "conditions": k+1,
                    "category_ids": list(combo), "negative_category_id": excluded,
                    "text": template([names[c] for c in ordered], names[excluded]),
                    "positives": [images[i]["id"] for i in indices(positive)],
                    "ignore": [images[i]["id"] for i in indices(ignore)], "n_hard_negatives": hard_count})
                break
            else:
                alive.remove(k)
    return sorted(result, key=lambda q: q["id"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT/"dl/retrieval-matrix/data/coco-negation-dev-v1")
    args = parser.parse_args()
    out = args.out
    if out.exists() and any(out.iterdir()):
        raise ValueError("Never overwrite an existing version")
    source = ROOT/"dl/claude-panel/conj-panel-v1"
    gallery_path = source/"dev/gallery.json"
    manifest_path = source/"manifest.json"
    v1 = json.loads(manifest_path.read_text())
    if sha(gallery_path) != v1["splits"]["dev"]["gallery_sha256"] or sha(gallery_path) != GALLERY_SHA:
        raise ValueError("Frozen v1 development gallery differs")
    gallery = json.loads(gallery_path.read_text())
    if len(gallery) != 20000:
        raise ValueError("Development gallery must remain complete")
    archive = ROOT/"dl/retrieval3090/data/coco2017/annotations_trainval2017.zip"
    if sha(archive) != ANNOTATIONS_SHA:
        raise ValueError("Official COCO annotation SHA mismatch")
    with zipfile.ZipFile(archive) as z:
        raw = json.load(z.open("annotations/instances_train2017.json"))
    metadata = {row["id"]: row for row in raw["images"]}
    images = [metadata[row["id"]] for row in gallery]
    names = {row["id"]: row["name"] for row in raw["categories"]}
    present, absent = status(images, raw["annotations"])
    universe = (1 << len(images))-1
    for category in names:
        absent.setdefault(category, universe)
    pools = candidates(images, present, absent, names)
    queries = select(pools, images, present, absent, names)
    small = sorted(q["id"] for k in (2, 3, 4) for q in
        sorted((q for q in queries if q["k"] == k), key=lambda q: h(SMALL_SEED, q["id"]))[:84])
    out.mkdir(parents=True, exist_ok=True)
    (out/"gallery.json").write_bytes(gallery_path.read_bytes())
    (out/"queries.jsonl").write_text("".join(json.dumps(q)+"\n" for q in queries))
    (out/"dev_small.json").write_text(json.dumps(small)+"\n")
    manifest = {"owner": "[codex]", "seed": SEED, "scope": "development only; AND plus one NOT; no inference",
        "images": len(images), "queries": len(queries), "dev_small_queries": len(small),
        "queries_per_k": dict(Counter(q["k"] for q in queries)),
        "candidate_counts": {k: len(value) for k, value in pools.items()},
        "builder_sha256": sha(__file__), "spec_sha256": sha(Path(__file__).with_name("SPEC.md")),
        "annotation_sha256": ANNOTATIONS_SHA, "source_manifest_sha256": sha(manifest_path),
        "source_gallery_sha256": sha(gallery_path),
        "image_directory": str(source/"dev/images"),
        "images_receipt": {"path": str(source/"dev/images-receipt.jsonl"), "sha256": sha(source/"dev/images-receipt.jsonl")},
        "outputs": {name: sha(out/name) for name in ("gallery.json", "queries.jsonl", "dev_small.json")}}
    (out/"manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2)+"\n")
    print(json.dumps({k: manifest[k] for k in ("images", "queries", "queries_per_k", "dev_small_queries", "candidate_counts")}), flush=True)


if __name__ == "__main__":
    main()
