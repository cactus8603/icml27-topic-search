"""[codex] Frozen COCO development v2; no model scores or test queries."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CONJUNCTION = "topics/compositional-retrieval/code/experiments/retrieval-conjunction-panel/build_panel.py"
NEGATION = "codex/retrieval_matrix_devdata/build.py"
FETCH = "topics/compositional-retrieval/code/experiments/retrieval-conjunction-panel/fetch_images.py"


def sha(path):
    d = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for b in iter(lambda: stream.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def module(relative):
    spec = importlib.util.spec_from_file_location("dev_v2_"+sha(ROOT/relative), ROOT/relative)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def frozen(ref):
    if not re.fullmatch(r"[0-9a-f]{40}", ref):
        raise ValueError("A full committed specification revision is required")
    spec_path = HERE/"development-sources.json"
    doc = read(spec_path)
    for rel, expected in doc["files"].items():
        if sha(ROOT/rel) != expected:
            raise ValueError("Frozen source/input SHA changed: "+rel)
    for rel in doc["code"]+[str(spec_path.relative_to(ROOT))]:
        actual = subprocess.check_output(["git", "-C", str(ROOT), "show", ref+":"+rel])
        if actual != (ROOT/rel).read_bytes():
            raise ValueError("Code/specification differs from its committed freeze: "+rel)
    return doc


def exclusions(sources):
    original = ROOT/"dl/claude-panel/conj-panel-v1"
    test = {int(row["id"]) for row in read(original/"test/gallery.json")}
    paco = {int(Path(row[0]).stem) for row in read(ROOT/sources["paco"])["data"]}
    karpathy = {int(row["cocoid"]) for row in read(ROOT/sources["karpathy"])["images"]
                if row["split"] == "test"}
    import pyarrow.parquet as pq
    half = set()
    for rel in sources["half_truths"]:
        for value in pq.read_table(ROOT/rel, columns=["source_image_id"]).column(0).to_pylist():
            match = re.search(r"COCO_\w+?_0*(\d+)", str(value))
            if match:
                half.add(int(match.group(1)))
    return {"original_test": test, "cola_paco": paco, "karpathy_test": karpathy, "half_truths": half}


def choose_gallery(images, blocked, h, size=20000):
    ordered = sorted(images, key=lambda row: h("conj-panel-2026111601", "pool", row["id"]))
    selected = [row for row in ordered if row["id"] not in blocked][:size]
    if len(selected) != size:
        raise ValueError("Insufficient eligible images for the complete fixed gallery")
    return sorted(selected, key=lambda row: row["id"])


def write_panel(folder, gallery, queries, small):
    folder.mkdir(parents=True)
    (folder/"gallery.json").write_text(json.dumps(gallery)+"\n")
    (folder/"queries.jsonl").write_text("".join(json.dumps(q)+"\n" for q in queries))
    (folder/"dev_small.json").write_text(json.dumps(small)+"\n")
    return {"queries": len(queries), "dev_small_queries": len(small),
            "queries_per_k": dict(Counter(q["k"] for q in queries)),
            "positive_images": sum(len(q["positives"]) for q in queries),
            "outputs": {name: sha(folder/name) for name in ("gallery.json", "queries.jsonl", "dev_small.json")}}


def build(out, ref):
    doc = frozen(ref)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Never overwrite an existing development version")
    conj, neg = module(CONJUNCTION), module(NEGATION)
    raw = conj.load_annotations(ROOT/doc["inputs"]["annotations"])
    excluded = exclusions(doc["inputs"])
    blocked = set().union(*excluded.values())
    gallery = choose_gallery(raw["images"], blocked, conj.h)
    ids = {row["id"] for row in gallery}
    if any(ids & values for values in excluded.values()):
        raise ValueError("Development images overlap a frozen test exclusion")
    names = {row["id"]: row["name"] for row in raw["categories"]}
    present, ambiguous = conj.image_status(gallery, raw["annotations"])
    queries = conj.build_queries([row["id"] for row in gallery], present, ambiguous, names, "dev-v2")
    panel_gallery = [{key: row[key] for key in ("id", "file_name", "coco_url", "width", "height")} for row in gallery]
    conjunction = write_panel(out/"dev", panel_gallery, queries, conj.dev_small(queries))
    pbits, abits = neg.status(gallery, raw["annotations"])
    for c in names:
        abits.setdefault(c, (1 << len(gallery))-1)
    pools = neg.candidates(gallery, pbits, abits, names)
    nqueries = neg.select(pools, gallery, pbits, abits, names)
    nsmall = sorted(q["id"] for k in (2, 3, 4) for q in sorted(
        (q for q in nqueries if q["k"] == k), key=lambda q: neg.h(neg.SMALL_SEED, q["id"]))[:84])
    negation = write_panel(out/"negation", panel_gallery, nqueries, nsmall)
    negation["candidate_counts"] = {k: len(v) for k,v in pools.items()}
    old_dev = {row["id"] for row in read(ROOT/"dl/claude-panel/conj-panel-v1/dev/gallery.json")}
    manifest = {"owner": "[codex]", "version": "coco-development-v2", "freeze_commit": ref,
        "source_receipt_sha256": sha(HERE/"development-sources.json"), "gallery_size": len(gallery),
        "exclusions": {key: {"source_ids": len(values), "overlap_with_new_dev": len(values&ids),
                                  "overlap_with_old_dev": len(values&old_dev)} for key,values in excluded.items()},
        "previous_dev_images_reused": len(ids&old_dev), "new_images": len(ids-old_dev),
        "conjunction": conjunction, "negation": negation, "model_results_used": False,
        "historical_exposure": "Existing v1 and earlier development exposure is preserved and must be disclosed.",
        "scope": "Future development only; no new test queries generated or opened"}
    (out/"manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2)+"\n")
    print(json.dumps(manifest), flush=True)


def fetch(out, workers):
    fetcher = module(FETCH)
    gallery = read(out/"dev/gallery.json")
    manifest = read(out/"manifest.json")
    if sha(out/"dev/gallery.json") != manifest["conjunction"]["outputs"]["gallery.json"]:
        raise ValueError("Frozen development image list changed")
    tmpdir = ROOT/"hf_home/tmp/coco-development-v2"
    tmpdir.mkdir(parents=True, exist_ok=True)
    dest = out/"dev/images"
    dest.mkdir(exist_ok=True)
    # Only existing development image roots are scanned; original test stays sealed.
    local = fetcher.local_index([ROOT/"dl/claude-panel/conj-panel-v1/dev/images",
        ROOT/"dl/retrieval3090/data/coco2017-rank-and/images",
        ROOT/"dl/retrieval3090/data/coco2017-rerank/images",
        ROOT/"dl/retrieval3090/data/coco2017-anchor-rerank/images"])
    rows, failures = [], []
    with ThreadPoolExecutor(workers) as executor:
        futures = {executor.submit(fetcher.fetch, row, dest, local, tmpdir): row["id"] for row in gallery}
        for i, future in enumerate(as_completed(futures), 1):
            try:
                rows.append(future.result())
            except Exception as error:
                failures.append({"id": futures[future], "error": repr(error)})
            if i % 2000 == 0:
                print(json.dumps({"done": i, "total": len(gallery), "failures": len(failures)}), flush=True)
    rows.sort(key=lambda row: row["id"])
    path = out/"dev/images-receipt.jsonl"
    path.write_text("".join(json.dumps(row)+"\n" for row in rows))
    summary = {"expected": len(gallery), "ok": len(rows), "failures": failures,
               "bytes": sum(row["bytes"] for row in rows), "receipt_sha256": sha(path),
               "negation_uses_same_images": True, "image_directory": str(dest.resolve())}
    (out/"images-summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary), flush=True)
    if failures:
        raise ValueError("Image acquisition incomplete; individual failures preserved")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["build", "fetch"])
    parser.add_argument("--freeze-commit", required=True)
    parser.add_argument("--out", type=Path, default=ROOT/"dl/retrieval-matrix/data/coco-development-v2")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    frozen(args.freeze_commit)
    if args.stage == "build":
        build(args.out, args.freeze_commit)
    else:
        fetch(args.out, args.workers)


if __name__ == "__main__":
    main()
