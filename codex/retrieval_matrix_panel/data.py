"""[codex] COLA single-object adapter: labels are loaded only for CPU scoring."""
from pathlib import Path
import json
import numpy as np


def lf_rows(path):
    # U+2028 can occur inside valid JSON strings; only ASCII LF delimits JSONL.
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def caption(label, subset):
    if subset == "gqa":
        attributes, noun = label
        return " ".join([*attributes, noun])
    if subset == "paco":
        encoded, count = label
        if not isinstance(count, int) or "__" not in encoded:
            raise ValueError("Unexpected PACO label format")
        attributes, noun = encoded.split("__", 1)
        return " ".join((attributes.replace("_", " ")+" "+noun).split())
    raise ValueError("Only fixed GQA/PACO label vocabularies are supported")


def source_rows(spec, R):
    import pyarrow.parquet as pq
    return pq.read_table(R.checked_file(spec["parquet"])).to_pylist()


def load_data(spec, R, *, verify_images=False, allow_test=False):
    if spec["split"] != "public":
        raise ValueError("COLA is a published test benchmark; no development split")
    rows = source_rows(spec, R)
    vocabulary = R.read(R.checked_file(spec["vocabulary"]))
    root = R.safe(spec["image_root"])
    allowed_roots = [root, *(R.safe(path) for path in spec.get("allowed_shared_image_roots",[]))]
    receipts = {row["path"]: row for row in lf_rows(R.checked_file(spec["images_receipt"]))}
    gallery = []
    for row in rows:
        relative = Path(row["image"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("COLA logical image path must remain relative to its dataset")
        image = R.safe(root/row["image"])
        if not any(image.is_relative_to(allowed) for allowed in allowed_roots) or not image.is_file():
            raise ValueError("COLA image escapes the pinned root or is absent")
        receipt = receipts["cola/"+row["image"]]
        if verify_images and (R.sha(image) != receipt["sha256"] or image.stat().st_size != receipt["bytes"]):
            raise ValueError("COLA image bytes differ from its verified source")
        gallery.append({"id": row["image"], "path": str(image), "sha256": receipt["sha256"], "bytes": receipt["bytes"]})
    queries = [{"id": spec["subset"]+":"+str(i), "group":spec["subset"]+":"+str(i),
                "text":caption(label, spec["subset"])} for i,label in enumerate(vocabulary)]
    if (len(gallery) != spec["expected_gallery"] or len(queries) != spec["expected_queries"]
            or len({row["id"] for row in gallery}) != len(gallery)):
        raise ValueError("COLA complete-source counts or image uniqueness changed")
    return {"gallery":gallery,"queries":queries,"excluded":[],"retained_empty":[],"raw_queries":len(queries),
            "fingerprint":R.digest({"gallery":gallery,"queries":queries,"dataset":spec})}


def load_labels(spec, bundle, R):
    rows = source_rows(spec, R)
    result = []
    for i,query in enumerate(bundle["queries"]):
        positives = [row["image"] for row in rows if row["label"][i]]
        ignored = [row["image"] for row in rows if not row["hard_list"][i]]
        if not positives or set(positives)&set(ignored):
            raise ValueError("COLA query has no positives or positives outside its hard subset")
        result.append({"id":query["id"],"group":query["group"],"positives":positives,"ignore":ignored})
    return result


def threshold_ap(scores, relevant):
    """Binary average precision at distinct score thresholds, including ties."""
    scores, relevant = np.asarray(scores), np.asarray(relevant)
    if scores.ndim != 1 or scores.shape != relevant.shape or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Expected finite score and binary-label vectors")
    if not np.isin(relevant, [0,1]).all():
        raise ValueError("Average precision needs binary labels")
    total = int(np.sum(relevant))
    if total == 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    ranked_scores, ranked_labels = scores[order], relevant[order]
    ends = np.r_[np.flatnonzero(ranked_scores[1:] != ranked_scores[:-1])+1, len(scores)]
    cumulative = np.cumsum(ranked_labels)[ends-1]
    return float(np.sum(np.diff(np.r_[0,cumulative])/total*cumulative/ends))


def official_metrics(out, spec, bundle, R):
    rows = source_rows(spec, R)
    image = np.load(out/"image.npy", mmap_mode="r", allow_pickle=False)
    text = np.load(out/"text.npy", mmap_mode="r", allow_pickle=False)
    # This P2 adapter fixes float32 normalized-embedding CPU similarity, unlike
    # the separate P0 CORE native BF16 task. No COLA paper exact-match claim.
    scores = text @ image.T
    labels = np.asarray([row["label"] for row in rows], dtype=np.int8).T
    hard = np.asarray([row["hard_list"] for row in rows], dtype=bool).T
    detail = []
    for query, values, rel, mask in zip(bundle["queries"], scores, labels, hard, strict=True):
        if not rel.sum() or not mask.any():
            continue
        detail.append({"id":query["id"],"AP":threshold_ap(values[mask],rel[mask]),
                       "hard_images":int(mask.sum()),"hard_positives":int(rel[mask].sum())})
    if len(detail) != len(bundle["queries"]):
        raise ValueError("COLA source class coverage changed; do not silently alter denominator")
    R.atomic_json(out/"results/cola-per-class.json",detail)
    return {"mAP":float(np.mean([row["AP"] for row in detail])),"queries":len(detail),
            "semantics":"official hard_list AP evaluated at distinct score thresholds; macro over source classes",
            "caption_protocol":"GQA ordered attributes+noun; PACO underscore-separated attributes+noun, count excluded; fixed P2 adaptation, not verified original prompt",
            "seen_unseen":"not available in the pinned public release; no inferred split",
            "paper_comparability":"GQA public gallery2589 differs from paper1952; no exact reproduction claim"}
