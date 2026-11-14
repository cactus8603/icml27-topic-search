"""Construct the prospective annotation-only manifest without loading an encoder."""
import argparse
import hashlib
import json
import re
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import COCO_DIR, EXP, SUBSETS
from e2_build_queries import image_category_sets, joint_text, load_annotations


CONFIG = EXP / "configs/prospective_rank.json"
MANIFEST = EXP / "configs/prospective_rank_manifest.json"
OLD_E2 = EXP / "results/headroom/26393/e2_queries_main.json"


def file_sha(path):
    with Path(path).open("rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def canonical_bytes(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(obj):
    return hashlib.sha256(canonical_bytes(obj)).hexdigest()


def hash_select(seed, *identifiers):
    return int(hashlib.sha256(canonical_bytes([seed, *identifiers])).hexdigest()[:16], 16)


def fold_of(seed, key, config):
    return "dev" if hash_select(seed, key) % config["n_folds"] == config["dev_fold"] else "test"


def heldout_assets(snapshot):
    from pyarrow import parquet as pq

    excluded_ids, excluded_shas, hashes = set(), set(), {}
    for sub in SUBSETS:
        path = Path(snapshot) / "data" / f"{sub}.parquet"
        table = pq.read_table(path, columns=["source_dataset", "source_image_id", "image_sha256"]).to_pydict()
        hashes[sub] = file_sha(path)
        excluded_shas.update(table["image_sha256"])
        for dataset, image_id in zip(table["source_dataset"], table["source_image_id"]):
            if dataset == "coco":
                suffix = re.fullmatch(r"(?:COCO_\w+_)?(\d+)\.jpg", Path(str(image_id)).name, flags=re.IGNORECASE)
                if suffix is None:
                    raise ValueError(f"Unknown COCO source ID format: {image_id!r}")
                excluded_ids.add(int(suffix[1]))
    return excluded_ids, excluded_shas, hashes


def old_keys(path):
    return {tuple(sorted(q["cats"])) for group in json.loads(Path(path).read_text())["queries"].values() for q in group}


def shared_assets(category_sets, image_shas, old, names, config):
    candidates = {part: [] for part in ("dev", "test")}
    # Deduplicate bytes before splitting, with IDs sorted for canonical tie breaks.
    seen = set()
    for image_id in sorted(category_sets):
        sha = image_shas[image_id]
        if sha not in seen:
            part = fold_of(config["image_split_seed"], image_id, config)
            candidates[part].append(image_id)
            seen.add(sha)

    split_records = {}
    for part, ids in candidates.items():
        cat_masks = {cat: np.array([cat in category_sets[i] for i in ids], dtype=bool) for cat in names}
        queries = []
        for k in (config["primary_k"], config["secondary_k"]):
            eligible = sorted({key for image_id in ids for key in combinations(sorted(category_sets[image_id]), k)})
            for key in eligible:
                if key in old or fold_of(config["query_split_seed"], key, config) != part:
                    continue
                q = construct_query(key, ids, cat_masks, names, config)
                if q is not None:
                    queries.append(q)
        split_records[part] = {"candidate_ids": ids, "candidate_image_sha256": [image_shas[i] for i in ids],
                               "queries": attach_shuffles(queries, config["shuffle_seed"])}
    return split_records


def construct_query(key, ids, cat_masks, names, config):
    seed = config["construction_seed"]
    k = len(key)
    # Derive the anchor from the query key, before looking for a positive or foil image.
    omitted = key[hash_select(seed, "omit", key) % k]
    anchor = tuple(cat for cat in key if cat != omitted)
    anchor_has = np.logical_and.reduce([cat_masks[cat] for cat in anchor])
    relevant = np.flatnonzero(anchor_has & cat_masks[omitted])
    foil_candidates = np.flatnonzero(anchor_has & ~cat_masks[omitted])
    if not len(relevant) or not len(foil_candidates):
        return None
    positive = int(relevant[hash_select(seed, "positive", key) % len(relevant)])
    image_foil = int(foil_candidates[hash_select(seed, "image-foil", key) % len(foil_candidates)])
    # Prefer a replacement observed in the foil image and absent from the sampled positive.
    # None is allowed because this text foil only affects descriptive signal counts.
    choices = sorted(cat for cat in names if cat not in key and cat_masks[cat][image_foil] and not cat_masks[cat][positive])
    replacement = choices[hash_select(seed, "text-foil", key) % len(choices)] if choices else None
    foil_key = tuple(sorted((*anchor, replacement))) if replacement is not None else None
    return {"key": list(key), "k": k, "anchor_key": list(anchor), "text": joint_text([names[cat] for cat in key]),
            "anchor": joint_text([names[cat] for cat in anchor]), "relevant_ids": [ids[i] for i in relevant],
            "paired_positive_id": ids[positive], "paired_hard_foil_id": ids[image_foil],
            "foil_key": list(foil_key) if foil_key else None,
            "foil_text": joint_text([names[cat] for cat in foil_key]) if foil_key else None}


def attach_shuffles(queries, seed):
    """Deterministic maximum matching of each query to a same-k disjoint donor anchor."""
    for k in sorted({q["k"] for q in queries}):
        group = [q for q in queries if q["k"] == k]
        order = sorted(range(len(group)), key=lambda j: (hash_select(seed, group[j]["key"]), group[j]["key"]))
        permissible = {j: [d for d in order if not set(group[j]["key"]) & set(group[d]["anchor_key"])] for j in order}
        matched_recipient = {}

        def assign(recipient, visited):
            for donor in permissible[recipient]:
                if donor in visited:
                    continue
                visited.add(donor)
                if donor not in matched_recipient or assign(matched_recipient[donor], visited):
                    matched_recipient[donor] = recipient
                    return True
            return False

        for recipient in sorted(order, key=lambda j: len(permissible[j])):
            assign(recipient, set())
        by_recipient = {recipient: donor for donor, recipient in matched_recipient.items()}
        for i, q in enumerate(group):
            donor = group[by_recipient[i]] if i in by_recipient else None
            if donor is None:
                q["shuffled_anchor"] = None
                q["shuffled_donor_key"] = None
                continue
            q["shuffled_anchor"] = donor["anchor"]
            q["shuffled_donor_key"] = donor["key"]
    return queries


def make_manifest(config):
    annotation_path = COCO_DIR / "annotations_trainval2017.zip"
    for path, expected in ((annotation_path, config["annotations_zip_sha256"]), (OLD_E2, config["old_e2_queries_sha256"])):
        if file_sha(path) != expected:
            raise ValueError(f"Input digest mismatch: {path}")
    assets = json.loads((EXP / "configs/assets_lock.json").read_text())["kbora/Half-Truths"]
    if assets["sha"] != config["half_truths_revision"]:
        raise ValueError("Unexpected Half-Truths revision")
    ids_to_exclude, shas_to_exclude, ht_hashes = heldout_assets(assets["snapshot"])
    old = old_keys(OLD_E2)
    names, ids, category_sets = image_category_sets(load_annotations(zip_path=annotation_path), config["candidate_area_fraction"])
    eligible_shas = {}
    removed = {"source_id": 0, "encoded_sha": 0}
    for image_id in ids:
        if image_id in ids_to_exclude:
            removed["source_id"] += 1
            continue
        sha = file_sha(COCO_DIR / "val2017" / f"{image_id:012d}.jpg")
        if sha in shas_to_exclude:
            removed["encoded_sha"] += 1
            continue
        eligible_shas[image_id] = sha
    split = shared_assets({i: category_sets[i] for i in eligible_shas}, eligible_shas, old, names, config)
    out = {"schema": 1, "criteria_canonical_sha256": digest(config), "source": {
        "annotations_zip_sha256": config["annotations_zip_sha256"], "old_e2_queries_sha256": config["old_e2_queries_sha256"],
        "old_e2_sorted_keys_sha256": digest(sorted(old)), "old_e2_n_keys": len(old),
        "half_truths_revision": assets["sha"], "half_truths_parquet_sha256": ht_hashes, "excluded": removed,
        "removed_duplicate_candidate_bytes": len(eligible_shas) - sum(len(d["candidate_ids"]) for d in split.values()),
    }, "splits": split}
    verify_manifest(out, config)
    return out


def verify_manifest(manifest, config):
    if manifest["schema"] != 1 or manifest["criteria_canonical_sha256"] != digest(config):
        raise ValueError("manifest/config digest mismatch")
    combined_ids, combined_bytes, combined_keys = set(), set(), set()
    for part in ("dev", "test"):
        data = manifest["splits"][part]
        ids, shas = data["candidate_ids"], data["candidate_image_sha256"]
        id_set, sha_set = set(ids), set(shas)
        if ids != sorted(ids) or len(id_set) != len(ids) or len(sha_set) != len(ids) or len(shas) != len(ids):
            raise ValueError("candidate IDs or JPEG fingerprints are non-unique or not aligned")
        if combined_ids & id_set or combined_bytes & sha_set:
            raise ValueError("overlapping DEV and TEST candidate pools")
        combined_ids.update(id_set)
        combined_bytes.update(sha_set)
        groups, n_primary = set(), 0
        queries = {tuple(q["key"]): q for q in data["queries"]}
        if len(queries) != len(data["queries"]) or combined_keys & queries.keys():
            raise ValueError("duplicate or shared canonical queries")
        combined_keys.update(queries)
        used_donors = set()
        for key, q in queries.items():
            rel = set(q["relevant_ids"])
            if not rel or not rel <= id_set or q["paired_positive_id"] not in rel or q["paired_hard_foil_id"] not in id_set - rel:
                raise ValueError(f"invalid candidate labels for {key}")
            if q["shuffled_donor_key"] is not None:
                donor_key = tuple(q["shuffled_donor_key"])
                donor = queries[donor_key]
                if (donor["k"] != q["k"] or donor["anchor"] != q["shuffled_anchor"]
                        or set(q["key"]) & set(donor["anchor_key"]) or donor_key in used_donors):
                    raise ValueError(f"invalid shuffled anchor for {key}")
                used_donors.add(donor_key)
            if q["k"] == config["primary_k"]:
                if q["shuffled_anchor"] is None:
                    raise ValueError("missing matched primary shuffle")
                n_primary += 1
                groups.add(q["paired_positive_id"])
        if n_primary < config["min_primary_queries"][part] or len(groups) < config["min_primary_positive_image_groups"][part]:
            raise ValueError(f"insufficient primary construction for {part}: {n_primary} queries / {len(groups)} groups")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=MANIFEST)
    args = parser.parse_args()
    data = make_manifest(json.loads(CONFIG.read_text()))
    with args.out.open("xb") as fh:
        fh.write(canonical_bytes(data) + b"\n")
    print(f"manifest SHA-256 {file_sha(args.out)}")
    for name, split in data["splits"].items():
        count = {k: sum(q["k"] == k for q in split["queries"]) for k in (2, 3)}
        print(name, len(split["candidate_ids"]), count, len({q["paired_positive_id"] for q in split["queries"] if q["k"] == 2}))


if __name__ == "__main__":
    main()
