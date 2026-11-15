#!/usr/bin/env python3
"""Exposed-data diagnostics: blinded audit exports, nested D3, and MCMR metadata.

No command loads a model or accesses the sealed independent retrieval panel.
New holistic query vectors must be encoded separately before d3-analyze.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
from itertools import combinations
import json
from pathlib import Path
import re
import tarfile
from urllib.parse import urlparse
from zipfile import ZipFile

import numpy as np


SEED = 2026111664
COCO_SHA = "113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268"
MCMR_REVISION = "12d53f4063804ee6a976024c7375ab2fbe4368c8"
MCMR_CODE = "8bf618a89cb41c1e12f1bad640eb12bcf0ef31a0"
MODEL_CONFIG_SHA = "862591c60c5609f16727d902993df0cba9cab96ded72611d69d263a02fe4ad1d"
MODEL_REVISIONS = {"core_embed": "cc9870ea92d64ebd5f8f9fc368d1120453564158",
                   "cs_clip": "afd6e90f39fe49137fe15d6a462d994eae4811f5"}
MCMR_FILES = {
    "query.jsonl": (1369474, "git", "f0e84cc78561f52be26d96486c35f43897c57ae1"),
    "candidates.jsonl": (148279983, "sha256", "05242fb414333a99c562a986edb8b2d1997db1726f731b6131ab97bf557fcbc0"),
}


def safe(path):
    path = Path(path).expanduser().resolve()
    if any("retrieval-independent-panel" in part for part in path.parts):
        raise ValueError("Sealed independent-panel paths are forbidden")
    return path


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path):
    with safe(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def hash_order(seed, *values):
    return int(digest([seed, *values])[:16], 16)


def write_json(path, value):
    path = safe(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def photo_text(names):
    if len(names) == 1:
        body = names[0]
    else:
        body = ", ".join(names[:-1]) + " and " + names[-1]
    return "a photo of " + body


def connected_pair_groups(families):
    """Any reused positive or negative image joins families into one cluster."""
    parent = {}
    def find(value):
        parent.setdefault(value, value)
        if parent[value] != value:
            parent[value] = find(parent[value])
        return parent[value]
    for family in families:
        left, right = find(family["positive_id"]), find(family["negative_id"])
        parent[max(left, right)] = min(left, right)
    return {family["id"]: "image-component-" + str(find(family["positive_id"])) for family in families}


def read_coco(path, ids):
    """Fixed public annotations only; maximum non-crowd instance area per category."""
    path = safe(path)
    if file_sha(path) != COCO_SHA:
        raise ValueError("COCO annotation ZIP is not the fixed public input")
    with ZipFile(path) as archive:
        # The fixed train annotation document is about 450 MB uncompressed.
        # Parsing it once avoids repeatedly scanning it for images/areas/names.
        data = json.load(archive.open("annotations/instances_train2017.json"))
    return coco_fields(data, ids)


def coco_fields(data, ids):
    wanted = set(ids)
    names = {row["id"]: row["name"] for row in data["categories"]}
    dims = {row["id"]: row["width"] * row["height"] for row in data["images"] if row["id"] in wanted}
    if set(dims) != wanted:
        raise ValueError("Gallery IDs missing from COCO annotation images")
    areas = {key: {} for key in ids}
    for row in data["annotations"]:
        image_id = row["image_id"]
        if image_id in wanted and not row.get("iscrowd", 0):
            cat = row["category_id"]
            fraction = float(row["area"]) / dims[image_id]
            areas[image_id][cat] = max(fraction, areas[image_id].get(cat, 0))
    categories = {key: {cat for cat, area in areas[key].items() if area >= .02} for key in ids}
    return names, categories, areas


def make_d3(ids, names, categories, areas, *, max_families=128, seed=SEED):
    """One fixed positive/negative pair per nested family; no model scores used.

    Negative J lacks the first condition c0 and satisfies all other three.
    Thus I/J and the single missing condition remain fixed for k=1,...,4.
    Full-gallery prevalence naturally changes and is measured, not claimed fixed.
    """
    present = {cat: {i for i in ids if cat in categories[i]} for cat in names}
    all_ids = set(ids)
    used_keys, families, queries = set(), [], []
    eligible = sorted((i for i in ids if len(categories[i]) >= 4), key=lambda i: (hash_order(seed, i), i))
    for positive in eligible:
        options = sorted(combinations(sorted(categories[positive]), 4), key=lambda key: hash_order(seed, positive, key))
        chosen = None
        for key in options:
            if key in used_keys:
                continue
            for missing in sorted(key, key=lambda cat: hash_order(seed, key, cat)):
                shared = [cat for cat in key if cat != missing]
                negatives = set.intersection(*(present[cat] for cat in shared)) - present[missing]
                if not negatives:
                    continue
                # Score-blind matching: similar remaining object context and areas.
                shortlist = sorted(negatives, key=lambda i: hash_order(seed, positive, key, i))[:128]
                def distance(negative):
                    union = categories[positive] | categories[negative]
                    context = len(categories[positive] & categories[negative]) / len(union)
                    area_diff = np.mean([abs(areas[positive][cat] - areas[negative][cat]) for cat in shared])
                    return (-context, float(area_diff), hash_order(seed, key, negative))
                negative = min(shortlist, key=distance)
                order = [missing] + sorted(shared, key=lambda cat: hash_order(seed, positive, "order", cat))
                chosen = key, order, negative
                break
            if chosen is not None:
                break
        if chosen is None:
            continue
        key, order, negative = chosen
        used_keys.add(key)
        family_id = "f" + digest([seed, positive, key])[:14]
        families.append({"id": family_id, "positive_id": positive, "negative_id": negative,
                         "condition_order": order, "missing_condition": order[0]})
        for k in range(1, 5):
            cats = order[:k]
            positives = set.intersection(*(present[cat] for cat in cats))
            hard = sum(sum(cat in categories[i] for cat in cats) == k - 1 for i in ids)
            query = {"id": f"{family_id}-k{k}", "family": family_id, "k": k,
                     "key": cats, "text": photo_text([names[cat] for cat in cats]),
                     "positive_id": positive, "negative_id": negative,
                     "relevant_ids": sorted(positives), "positive_count": len(positives),
                     "positive_prevalence": len(positives) / len(ids), "hard_negative_count": hard,
                     "mean_log_category_prevalence": float(np.mean([np.log(len(present[cat]) / len(ids)) for cat in cats])),
                     "mean_positive_instance_area": float(np.mean([areas[positive][cat] for cat in cats])),
                     "shared_positive_negative_area_difference": float(np.mean([
                         abs(areas[positive][cat] - areas[negative][cat]) for cat in cats[1:]])) if k > 1 else 0.0}
            if positive not in positives or negative in positives:
                raise AssertionError("Invalid matched pair")
            queries.append(query)
        if len(families) >= max_families:
            break
    if not families:
        raise ValueError("No four-condition family has a fixed exactly-one-missing negative")
    texts = {}
    for query in queries:
        texts.setdefault(query["text"], {"text": query["text"], "kind": "holistic", "key": query["key"]})
        for cat in query["key"]:
            text = photo_text([names[cat]])
            texts.setdefault(text, {"text": text, "kind": "single_condition", "key": [cat]})
    rows = [{"id": "t" + digest(text)[:16], **texts[text]} for text in sorted(texts)]
    components = connected_pair_groups(families)
    for family in families:
        family["cluster"] = components[family["id"]]
    for query in queries:
        query["cluster"] = components[query["family"]]
    return {"schema": 1, "seed": seed, "gallery_ids": ids, "gallery_ids_sha256": digest(ids),
            "families": families, "queries": queries, "texts": rows,
            "texts_sha256": digest([row["text"] for row in rows]),
            "scope": "score-blind matched nested object-conjunction diagnostic on exposed development images",
            "controls": {"fixed_within_family": ["positive image", "negative image", "missing condition", "gallery"],
                         "measured_not_fixed": ["positive prevalence", "category prevalence", "object area", "hard-negative count"],
                         "uncertainty_clusters": "connected components of reused positive/negative images across families"}}


def prepare_d3(args):
    panel_path = safe(args.panel)
    panel = json.loads(panel_path.read_text())
    names, categories, areas = read_coco(args.annotations, panel["ids"])
    design = make_d3(panel["ids"], names, categories, areas, max_families=args.max_families, seed=args.seed)
    design["source_panel_sha256"] = file_sha(panel_path)
    design["annotations_sha256"] = COCO_SHA
    out = safe(args.out)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "design.json", design)
    with (out / "texts.jsonl").open("x") as stream:
        for row in design["texts"]:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(out / "encoding_contract.json", {
        "texts_sha256": design["texts_sha256"], "text_count": len(design["texts"]),
        "npz_fields": {"texts": "Unicode array, exact texts.jsonl order",
                       "core_embed_text": "float array T x 2048, same normalization/model/prompt as cache",
                       "cs_clip_text": "float array T x 512, same normalization/model as cache",
                       "metadata_json": "Unicode scalar JSON with schema/design/text/gallery/model/config/provenance bindings"},
        "metadata_schema": "d3-holistic-encodings-v1", "model_revisions": MODEL_REVISIONS,
        "model_config_sha256": MODEL_CONFIG_SHA,
        "required_new_work": "Encode the complete holistic texts; summing single-condition vectors is a separate diagnostic baseline"})
    print(json.dumps({"families": len(design["families"]), "queries": len(design["queries"]),
                      "texts_to_encode": len(design["texts"]), "out": str(out)}))


def cluster_interval(values, groups, seed=SEED, draws=2000):
    unique = sorted(set(groups))
    totals = np.array([np.sum([v for v, g in zip(values, groups) if g == group]) for group in unique])
    counts = np.array([sum(g == group for g in groups) for group in unique])
    picks = np.random.default_rng(seed).integers(len(unique), size=(draws, len(unique)))
    means = totals[picks].sum(1) / counts[picks].sum(1)
    return {"point": float(totals.sum() / counts.sum()),
            "ci95": np.quantile(means, [.025, .975]).tolist() if len(unique) >= 2 else None,
            "clusters": len(unique), "reliable_cluster_asymptotics": len(unique) >= 20}


def controlled_slope(values, queries):
    """Within-family descriptive regression; suppress coefficient when not identified."""
    controls = ["log_positive_count", "mean_log_category_prevalence", "mean_positive_instance_area",
                "log_hard_negative_count", "shared_positive_negative_area_difference"]
    x = np.array([[q["k"], np.log(q["positive_count"]), q[controls[1]], q[controls[2]],
                   np.log(max(q["hard_negative_count"], 1)), q[controls[4]]] for q in queries])
    y = np.array(values, dtype=float)
    families = np.array([q["family"] for q in queries])
    for family in np.unique(families):
        where = families == family
        x[where] -= x[where].mean(0)
        y[where] -= y[where].mean()
    varying = x.std(0) > 1e-12
    if not varying[0]:
        raise ValueError("No within-family variation in k")
    z = x[:, varying].copy()
    z[:, 1:] /= z[:, 1:].std(0)
    rank = int(np.linalg.matrix_rank(z))
    identified = rank == z.shape[1]
    beta = np.linalg.lstsq(z, y, rcond=None)[0]
    groups = np.array([q.get("cluster", q["family"]) for q in queries])
    clusters = np.unique(groups)
    interval = None
    if identified and len(clusters) >= 2:
        positions = [np.flatnonzero(groups == group) for group in clusters]
        rng = np.random.default_rng(SEED)
        draws = []
        for _ in range(1000):
            take = np.concatenate([positions[index] for index in rng.integers(len(clusters), size=len(clusters))])
            if np.linalg.matrix_rank(z[take]) == z.shape[1]:
                draws.append(float(np.linalg.lstsq(z[take], y[take], rcond=None)[0][0]))
        if len(draws) >= 950:
            interval = np.quantile(draws, [.025, .975]).tolist()
    return {"k_slope": float(beta[0]) if identified else None, "identified": identified,
            "k_slope_ci95": interval, "image_components": len(clusters),
            "rank": rank, "columns": z.shape[1], "condition_number": float(np.linalg.cond(z)) if identified else None,
            "controls": [name for name, keep in zip(controls, varying[1:]) if keep],
            "interpretation": "descriptive within-family association; not a causal proof or universal 1/sqrt(k) law"}


def d3_scores(image_vectors, text_vectors, texts, design):
    by_text = {text: i for i, text in enumerate(texts)}
    ids = design["gallery_ids"]
    by_id = {image_id: index for index, image_id in enumerate(ids)}
    records = {"holistic": [], "unit_additive": []}
    names = {row["key"][0]: row["text"] for row in design["texts"] if len(row["key"]) == 1}
    for query in design["queries"]:
        holistic = text_vectors[by_text[query["text"]]]
        additive = text_vectors[[by_text[names[cat]] for cat in query["key"]]].sum(0)
        additive = additive / max(np.linalg.norm(additive), 1e-12)
        for kind, vector in (("holistic", holistic), ("unit_additive", additive)):
            scores = image_vectors @ vector
            positive = by_id[query["positive_id"]]
            negative = by_id[query["negative_id"]]
            order = np.argsort(-scores, kind="stable")
            margin = float(scores[positive] - scores[negative])
            std = float(scores.std())
            if std <= 1e-12:
                raise ValueError("Degenerate constant-score query")
            relevant = {by_id[key] for key in query["relevant_ids"]}
            records[kind].append({"id": query["id"], "family": query["family"], "k": query["k"],
                                  "margin": margin, "standardized_margin": margin / std,
                                  "pair_correct": int(margin > 0),
                                  "recall_at_50": len(set(order[:50]) & relevant) / len(relevant),
                                  "target_rank": int(np.flatnonzero(order == positive)[0]) + 1})
    return records


def validate_design(design):
    families = {family["id"]: family for family in design["families"]}
    if len(families) != len(design["families"]):
        raise ValueError("Duplicate D3 family ID")
    groups = connected_pair_groups(design["families"])
    if any(family["cluster"] != groups[family["id"]] for family in families.values()):
        raise ValueError("D3 family connected-cluster binding changed")
    if (digest(design["gallery_ids"]) != design["gallery_ids_sha256"]
            or digest([row["text"] for row in design["texts"]]) != design["texts_sha256"]):
        raise ValueError("D3 gallery or text-list digest changed")
    seen = set()
    for query in design["queries"]:
        pair = (query["family"], query["k"])
        if pair in seen or query["family"] not in families or query["k"] not in range(1, 5):
            raise ValueError("Duplicate or unknown D3 family/k record")
        seen.add(pair)
        family = families[query["family"]]
        if (query["key"] != family["condition_order"][:query["k"]]
                or query["positive_id"] != family["positive_id"]
                or query["negative_id"] != family["negative_id"]
                or query["cluster"] != groups[query["family"]]):
            raise ValueError("D3 pair, nested condition or connected-cluster binding changed")
    if seen != {(family, k) for family in families for k in range(1, 5)}:
        raise ValueError("D3 requires all four k values for every family")


def validate_cache_meta(meta):
    if meta.get("replay", {}).get("state") != "PASS":
        raise ValueError("Full development cache must pass baseline replay before diagnostics")
    if meta.get("model_config_sha256") != MODEL_CONFIG_SHA:
        raise ValueError("Development cache model configuration changed")
    if any(meta.get("revisions", {}).get(key) != value for key, value in MODEL_REVISIONS.items()):
        raise ValueError("Development cache model revision changed")


def validate_embedding_meta(meta, design, design_sha, cache_meta=None):
    if (meta.get("schema") != "d3-holistic-encodings-v1" or meta.get("design_sha256") != design_sha
            or meta.get("texts_sha256") != design["texts_sha256"]
            or meta.get("gallery_ids_sha256") != design["gallery_ids_sha256"]
            or meta.get("normalization") != "unit_l2"
            or meta.get("model_config_sha256") != MODEL_CONFIG_SHA
            or meta.get("model_revisions") != MODEL_REVISIONS):
        raise ValueError("Supplemental D3 embeddings have a different design, model or normalization")
    if any(not isinstance(meta.get(key), str) or not re.fullmatch(r"[0-9a-f]{40}", meta[key])
           for key in ("source", "freeze", "commit")):
        raise ValueError("Supplemental D3 embeddings require source/freeze/commit provenance")
    if cache_meta is not None:
        bundle = cache_meta.get("bundle", {})
        if any(meta[key] != bundle.get(key) for key in ("source", "freeze")):
            raise ValueError("Supplemental D3 source/freeze differ from the validated development cache")


def family_differences(rows, design, metric):
    by_pair = {(row["family"], row["k"]): row[metric] for row in rows}
    if len(by_pair) != len(rows):
        raise ValueError("Repeated scored D3 family/k pair")
    differences, groups = [], []
    for family in design["families"]:
        differences.append(by_pair[(family["id"], 4)] - by_pair[(family["id"], 1)])
        groups.append(family["cluster"])
    return differences, groups


def analyze_d3(args):
    design = json.loads(safe(args.design).read_text())
    validate_design(design)
    cache = safe(args.cache)
    cache_meta = json.loads((cache / "meta.json").read_text())
    validate_cache_meta(cache_meta)
    cache_panel = json.loads((cache / "panel.json").read_text())
    if cache_panel["ids"] != design["gallery_ids"]:
        raise ValueError("D3 gallery order differs from cached image vector order")
    with np.load(cache / "vectors.npz", allow_pickle=False) as vectors, np.load(safe(args.embeddings), allow_pickle=False) as query_vectors:
        if "metadata_json" not in query_vectors:
            raise ValueError("Supplemental D3 embeddings lack metadata_json")
        encoding_meta = json.loads(query_vectors["metadata_json"].item())
        validate_embedding_meta(encoding_meta, design, file_sha(args.design), cache_meta)
        texts = query_vectors["texts"].tolist()
        if texts != [row["text"] for row in design["texts"]]:
            raise ValueError("New holistic embeddings do not match the frozen D3 text order")
        report = {"scope": "exposed-data diagnostic, not a formal test", "design_sha256": file_sha(args.design),
                  "embedding_sha256": file_sha(args.embeddings), "encoding_metadata": encoding_meta,
                  "primary_contrast": "CORE holistic standardized pair margin: k4 minus k1, image-component bootstrap",
                  "models": {}}
        for model in ("core_embed", "cs_clip"):
            image, text = vectors[f"{model}_image"], query_vectors[f"{model}_text"]
            if (image.shape[0] != len(design["gallery_ids"])
                    or text.shape != (len(texts), image.shape[1]) or not np.isfinite(text).all()
                    or not np.isfinite(image).all()):
                raise ValueError("Incorrect new text embedding shape or nonfinite values")
            if not np.allclose(np.linalg.norm(text, axis=1), 1, atol=.005):
                raise ValueError("New text embeddings must use the cache's unit normalization")
            records = d3_scores(image, text, texts, design)
            summary = {}
            for kind, rows in records.items():
                means = {k: {key: float(np.mean([row[key] for row in rows if row["k"] == k]))
                             for key in ("margin", "standardized_margin", "pair_correct", "recall_at_50", "target_rank")}
                         for k in range(1, 5)}
                paired = {}
                for metric in ("margin", "standardized_margin", "pair_correct", "recall_at_50"):
                    differences, groups = family_differences(rows, design, metric)
                    paired[metric] = cluster_interval(differences, groups)
                summary[kind] = {"by_k": means, "k4_minus_k1": paired,
                                 "controlled_margin_slope": controlled_slope([row["margin"] for row in rows], design["queries"]),
                                 "controlled_standardized_margin_slope": controlled_slope(
                                     [row["standardized_margin"] for row in rows], design["queries"]),
                                 "records": rows}
            report["models"][model] = summary
    write_json(args.out, report)
    print(json.dumps({"saved": str(args.out), "families": len(design["families"])}))


def audit_candidates(panel, core, cross):
    """Sampling strata remain private; public export carries no ranks or truth labels."""
    ids, queries = panel["ids"], panel["queries"]
    inverse = {key: index for index, key in enumerate(ids)}
    strata = {"top_negative": [], "missed_positive": [], "control": []}
    for row, query in enumerate(queries):
        relevant = {inverse[key] for key in query["relevant"]}
        top = core[row, :50]
        ranked = top[np.lexsort((np.arange(len(top)), -cross[row, :50]))]
        strata["top_negative"].extend((row, int(index)) for index in ranked[:10] if index not in relevant)
        strata["missed_positive"].extend((row, index) for index in relevant if index not in set(top))
        strata["control"].extend((row, int(index)) for index in core[row, ::max(1, len(ids) // 20)])
    return strata


def public_audit_case(case_id, query, filename):
    return {"case_id": case_id, "query": query, "image": filename,
            "all_conditions_visible": "", "uncertain_or_tiny_object": "", "notes": ""}


def blind_export(args):
    from PIL import Image

    cache, out, image_root = safe(args.cache), safe(args.out), safe(args.image_root)
    validate_cache_meta(json.loads((cache / "meta.json").read_text()))
    panel = json.loads((cache / "panel.json").read_text())
    with np.load(cache / "cross.npz", allow_pickle=False) as scores:
        strata = audit_candidates(panel, scores["core_order"], scores["cross"])
    out.mkdir(parents=True, exist_ok=False)
    public, private = out / "review", out / "private"
    (public / "images").mkdir(parents=True)
    private.mkdir(mode=0o700)
    selected, seen = [], set()
    quota = max(1, args.samples // len(strata))
    for stratum, population in strata.items():
        ordered = sorted(population, key=lambda pair: hash_order(args.seed, stratum, pair))
        count = 0
        for pair in ordered:
            if pair not in seen:
                selected.append((stratum, pair))
                seen.add(pair)
                count += 1
            if count >= quota:
                break
    selected.sort(key=lambda item: hash_order(args.seed, "anonymous-order", item[1]))
    public_rows, key_rows = [], []
    for ordinal, (stratum, (query_index, image_index)) in enumerate(selected, 1):
        image_id, query = panel["ids"][image_index], panel["queries"][query_index]
        case_id = "case-" + digest([args.seed, query_index, image_id])[:16]
        filename = "images/" + case_id + ".png"
        # Re-encoding removes source filenames, EXIF, labels and method identity.
        with Image.open(image_root / f"{image_id:012d}.jpg") as image:
            clean = Image.new("RGB", image.size)
            clean.paste(image.convert("RGB"))
            clean.save(public / filename)
        public_rows.append(public_audit_case(case_id, query["text"], filename))
        key_rows.append({"case_id": case_id, "stratum": stratum, "image_id": image_id,
                         "query_index": query_index, "annotation_positive": image_id in query["relevant"],
                         "sampling_frame_size": len(strata[stratum])})
    if not public_rows:
        raise ValueError("No audit cases selected")
    with (public / "labels.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(public_rows[0]))
        writer.writeheader()
        writer.writerows(public_rows)
    cards = [f'<article><h3>{ordinal}. {html.escape(row["case_id"])}</h3><p>{html.escape(row["query"])}</p>'
             f'<img src="{html.escape(row["image"])}" width="480"></article>'
             for ordinal, row in enumerate(public_rows, 1)]
    (public / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>Image review</title>'
        '<p>Check whether every requested object is visibly present. Mark yes/no/uncertain in labels.csv. '
        'Judge visible content; note tiny or occluded objects. Do not infer from filenames.</p>' + "\n".join(cards))
    write_json(private / "key.json", {"seed": args.seed, "cache_panel_sha256": file_sha(cache / "panel.json"),
                                     "cases": key_rows, "sampling": "diagnostic stratified sample, not a population PPV estimate"})
    write_json(out / "status.json", {"state": "AWAITING_HUMAN_LABELS", "exported": len(public_rows),
                                    "public_package": str(public), "private_mapping": str(private),
                                    "human_labels_completed": 0, "instructions": "Share review/ only; never give annotators private/"})
    print(json.dumps({"state": "AWAITING_HUMAN_LABELS", "cases": len(public_rows), "review": str(public)}))


def metadata_hash(path, mode):
    path = safe(path)
    if mode == "sha256":
        return file_sha(path)
    value = hashlib.sha1(b"blob " + str(path.stat().st_size).encode() + b"\0")
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            value.update(block)
    return value.hexdigest()


def download_mcmr(out):
    import requests

    for name, (size, mode, expected) in MCMR_FILES.items():
        path = out / name
        if not path.exists():
            partial = path.with_suffix(".partial")
            url = f"https://huggingface.co/datasets/Lux1997/MCMR/resolve/{MCMR_REVISION}/{name}"
            received = 0
            with requests.get(url, stream=True, timeout=(15, 60)) as response:
                response.raise_for_status()
                with partial.open("wb") as stream:
                    for block in response.iter_content(2**20):
                        received += len(block)
                        if received > size:
                            raise ValueError("MCMR metadata exceeds pinned size")
                        stream.write(block)
            if partial.stat().st_size != size or metadata_hash(partial, mode) != expected:
                raise ValueError(f"MCMR metadata integrity failure: {name}")
            partial.replace(path)
        if path.stat().st_size != size or metadata_hash(path, mode) != expected:
            raise ValueError(f"Changed pinned MCMR metadata: {name}")


def iter_jsonl(path):
    with safe(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def candidate_text(row):
    """Native candidate-side information; never use query positives as input text."""
    fields = []
    for name in ("title", "description", "features", "price", "Date First Available"):
        value = row.get(name)
        if isinstance(value, list):
            value = ". ".join(str(item) for item in value)
        if value is not None and str(value).strip():
            fields.append(str(value).strip())
    return ". ".join(fields)


def normalize_mcmr(candidate_rows, query_rows):
    candidates, queries, seen = [], [], set()
    for row in candidate_rows:
        key = str(row.get("candidate_id") or row.get("id") or "")
        if not key or key in seen:
            raise ValueError("Missing or duplicate MCMR candidate ID")
        seen.add(key)
        images = row.get("images") or []
        first = images[0] if images else {}
        url = next((first.get(field) for field in ("url", "large", "image") if first.get(field)), "")
        candidates.append({"id": key, "image_filename": Path(urlparse(url).path).name if url else None,
                           "image_url": url, "text": candidate_text(row), "source_fields": sorted(row)})
    for ordinal, row in enumerate(query_rows):
        text = row.get("query") or row.get("text") or ""
        positives = [str(key) for key in (row.get("pos_ids") or row.get("positives") or [])]
        if not isinstance(text, str) or not text or not positives:
            raise ValueError("MCMR query missing text or positives")
        missing = set(positives) - seen
        queries.append({"id": str(row.get("qid") or row.get("id") or ordinal), "text": text,
                        "positive_ids": positives, "missing_positive_ids": sorted(missing),
                        "source_fields": sorted(row)})
    return candidates, queries


def prepare_mcmr(args):
    out = safe(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.download:
        download_mcmr(out)
    for name, (size, mode, expected) in MCMR_FILES.items():
        if (out / name).stat().st_size != size or metadata_hash(out / name, mode) != expected:
            raise ValueError("MCMR requires exact pinned public metadata")
    candidates, queries = normalize_mcmr(iter_jsonl(out / "candidates.jsonl"), iter_jsonl(out / "query.jsonl"))
    for name, rows in (("candidates.normalized.jsonl", candidates), ("queries.normalized.jsonl", queries)):
        with (out / name).open("x") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    evaluable = [row for row in queries if not row["missing_positive_ids"]]
    selected_queries = sorted(evaluable, key=lambda row: hash_order(args.seed, "mcmr-query", row["id"]))[:args.pilot_queries]
    positives = {key for query in selected_queries for key in query["positive_ids"]}
    remaining = sorted((row["id"] for row in candidates if row["id"] not in positives),
                       key=lambda key: hash_order(args.seed, "mcmr-negative", key))[:args.pilot_negatives]
    selected_ids = positives | set(remaining)
    pilot = {"scope": "development-only subsampled gallery; not native full MCMR evaluation",
             "seed": args.seed, "candidate_ids": sorted(selected_ids), "queries": selected_queries,
             "candidate_images": [{key: row[key] for key in ("id", "image_filename", "image_url")}
                                  for row in candidates if row["id"] in selected_ids]}
    write_json(out / "pilot_manifest.json", pilot)
    receipt = {"dataset_revision": MCMR_REVISION, "official_code_commit": MCMR_CODE,
               "source": "https://huggingface.co/datasets/Lux1997/MCMR",
               "metadata_files": {name: {"bytes": (out / name).stat().st_size, "sha256": file_sha(out / name)} for name in MCMR_FILES},
               "candidates": len(candidates), "queries": len(queries),
               "queries_with_missing_positive_ids": sum(bool(row["missing_positive_ids"]) for row in queries),
               "query_source_fields": sorted({field for row in queries for field in row["source_fields"]}),
               "candidate_source_fields": sorted({field for row in candidates for field in row["source_fields"]}),
               "native_input": "text query -> candidate image plus candidate textual metadata",
               "image_only_adapter": "separate ablation; dropping product text changes task and its available evidence",
               "images_archive": {"path": "images.tar.gz", "bytes": 2263663574,
                                  "sha256": "7cbda9a5fbdba1240590885c576b4b31dccd047f14d2cc5b8a8e87c94bda2b86",
                                  "downloaded_by_this_tool": False},
               "code_license": "Apache-2.0", "dataset_license": "not declared in dataset card metadata at verification time",
               "pilot_candidates": len(selected_ids), "pilot_queries": len(selected_queries)}
    write_json(out / "preparation.json", receipt)
    print(json.dumps({key: receipt[key] for key in ("candidates", "queries", "pilot_candidates", "pilot_queries")}))


def extract_mcmr(args):
    """Verify the whole official archive, then extract only named pilot image bytes."""
    archive, pilot_path, out = safe(args.archive), safe(args.pilot), safe(args.out)
    if (archive.stat().st_size != 2263663574
            or file_sha(archive) != "7cbda9a5fbdba1240590885c576b4b31dccd047f14d2cc5b8a8e87c94bda2b86"):
        raise ValueError("MCMR image archive differs from the pinned original")
    pilot = json.loads(pilot_path.read_text())
    names = {row["image_filename"] for row in pilot["candidate_images"]}
    if None in names or any(Path(name).name != name or name in ("", ".", "..") for name in names):
        raise ValueError("Pilot contains a missing or unsafe image filename")
    out.mkdir(parents=True, exist_ok=False)
    receipts = {}
    with tarfile.open(archive, "r|gz") as stream:
        for member in stream:
            name = Path(member.name).name
            if name not in names:
                continue
            if not member.isfile() or member.name != "images/" + name or not 0 < member.size <= 32 * 1024**2:
                raise ValueError("Unexpected archive member for a pilot image")
            if name in receipts:
                raise ValueError("Repeated pilot image in the official archive")
            with stream.extractfile(member) as source, (out / name).open("xb") as target:
                content = source.read(member.size + 1)
                if len(content) != member.size:
                    raise ValueError("Truncated pilot image")
                target.write(content)
            receipts[name] = {"bytes": member.size, "sha256": hashlib.sha256(content).hexdigest()}
    missing = names - receipts.keys()
    if missing:
        raise ValueError(f"Official archive lacks {len(missing)} required pilot images")
    write_json(out / "extraction.json", {"scope": pilot["scope"], "pilot_sha256": file_sha(pilot_path),
               "archive_sha256": file_sha(archive), "images": receipts,
               "candidate_id_to_filename": {row["id"]: row["image_filename"] for row in pilot["candidate_images"]}})
    print(json.dumps({"images_extracted": len(receipts), "out": str(out)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    d3 = sub.add_parser("d3-prepare")
    d3.add_argument("--panel", required=True, type=Path)
    d3.add_argument("--annotations", required=True, type=Path)
    d3.add_argument("--out", required=True, type=Path)
    d3.add_argument("--max-families", type=int, default=128)
    d3.add_argument("--seed", type=int, default=SEED)
    d3.set_defaults(run=prepare_d3)
    analyse = sub.add_parser("d3-analyze")
    analyse.add_argument("--design", required=True, type=Path, help="Prepared design.json")
    analyse.add_argument("--cache", required=True, type=Path)
    analyse.add_argument("--embeddings", required=True, type=Path)
    analyse.add_argument("--out", required=True, type=Path)
    analyse.set_defaults(run=analyze_d3)
    audit = sub.add_parser("blind-export")
    audit.add_argument("--cache", required=True, type=Path)
    audit.add_argument("--image-root", required=True, type=Path)
    audit.add_argument("--out", required=True, type=Path)
    audit.add_argument("--samples", type=int, default=150)
    audit.add_argument("--seed", type=int, default=SEED)
    audit.set_defaults(run=blind_export)
    mcmr = sub.add_parser("mcmr-prepare")
    mcmr.add_argument("--out", required=True, type=Path)
    mcmr.add_argument("--download", action="store_true", help="Fetch only pinned 149.65 MB public metadata; never images")
    mcmr.add_argument("--pilot-queries", type=int, default=128)
    mcmr.add_argument("--pilot-negatives", type=int, default=8192)
    mcmr.add_argument("--seed", type=int, default=SEED)
    mcmr.set_defaults(run=prepare_mcmr)
    extract = sub.add_parser("mcmr-extract")
    extract.add_argument("--archive", required=True, type=Path)
    extract.add_argument("--pilot", required=True, type=Path)
    extract.add_argument("--out", required=True, type=Path)
    extract.set_defaults(run=extract_mcmr)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
