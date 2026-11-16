"""[codex] Full official paired examples; captions remain raw until tokenization."""
from pathlib import Path

from codex.retrieval_matrix_native_p0.evidence import checked, digest, read, sha


def load(spec, *, verify_images=True):
    if spec.get("format") not in {"sugarcrepe-official", "sugarcrepe-pp-official"}:
        raise ValueError("Only fixed official paired JSONs are supported")
    import json
    receipt_rows = [json.loads(line) for line in checked(spec["image_receipt"]).read_text().splitlines() if line]
    receipt = {row["filename"]: row for row in receipt_rows}
    if len(receipt) != len(receipt_rows):
        raise ValueError("Duplicate paired image receipt filename")
    pp = spec["format"] == "sugarcrepe-pp-official"
    roles = ("caption", "caption2", "negative_caption") if pp else ("caption", "negative_caption")
    gallery, queries, labels, rows = [], [], [], []
    counts = {}
    verified = set()
    for category in spec["category_order"]:
        value = read(checked(spec["subsets"][category]))
        records = list(value.items()) if isinstance(value, dict) else [(str(index), row) for index, row in enumerate(value)]
        if len(records) != spec["expected_subsets"][category]:
            raise ValueError("Paired source denominator differs")
        counts[category] = len(records)
        for key, row in records:
            if Path(row["filename"]).name != row["filename"] or any(not isinstance(row[role], str) for role in roles):
                raise ValueError("Official paired image/caption fields are invalid")
            item = receipt.get(row["filename"])
            if item is None:
                raise ValueError("Paired source image has no fixed byte receipt")
            path = Path(item["path"])
            if not path.is_file() or path.stat().st_size != item["bytes"]:
                raise ValueError("A full-denominator paired image is absent or changed")
            if verify_images and str(path) not in verified:
                if sha(path) != item["sha256"]:
                    raise ValueError("Paired image SHA differs")
                verified.add(str(path))
            qid = category+":"+key
            gallery.append({"id": qid, "path": str(path), "source_metadata": row,
                "image_sha256": item["sha256"], "image_bytes": item["bytes"]})
            start = len(queries)
            for role in roles:
                queries.append({"id": qid+":"+role, "text": row[role]})
            labels.append({"id": qid, "category": category, "text_start": start,
                "text_options": len(roles), "benchmark": "sugarcrepe_pp" if pp else "sugarcrepe"})
            rows.append({"id": qid, "category": category, "original_key": key, "source": row})
    summary = {"images": len(gallery), "texts": len(queries), "paired_examples": len(rows),
        "unique_images": len({row["path"] for row in gallery}), "excluded_images": 0, "excluded_texts": 0,
        "subsets": counts}
    if summary["images"] != spec["expected_images"] or summary["texts"] != spec["expected_texts"]:
        raise ValueError("Paired full-example/caption denominator changed")
    identity = {"benchmark_id": spec["benchmark_id"], "spec_sha256": digest(spec),
        "gallery_ids": [row["id"] for row in gallery], "query_ids": [row["id"] for row in queries],
        "labels_sha256": digest(labels), "rows_sha256": digest(rows), "counts": summary}
    return {"gallery": gallery, "queries": queries, "labels": labels, "rows": rows,
            "counts": summary, "identity": identity}


def inference_items(bundle):
    return [row["path"] for row in bundle["gallery"]], [row["text"] for row in bundle["queries"]]


def positive_indices(bundle):
    # This CPU-only structure defines source caption positions/roles. It never
    # reaches inference; no candidate selection or admission uses these roles.
    return bundle["labels"]
