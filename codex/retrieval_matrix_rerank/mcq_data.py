"""[codex] Preserve official MCQ rows; avoid shadowing Hugging Face datasets."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "dl/retrieval-matrix-rerank"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pair(identity, index, query, image):
    image = Path(image).resolve()
    if not image.is_relative_to(ROOT / "dl") or not image.is_file():
        raise ValueError("Official image missing or outside dataset root")
    if not isinstance(query, str):
        raise ValueError("Official query must be a string")
    return {"id": f"{identity}:p{index}", "query": query, "image": str(image)}


def prepare(*, cola_image_choice=False):
    import pyarrow.parquet as pq
    examples, sources = [], {}
    cola = ROOT / "dl/retrieval-matrix/data/cola"
    path = cola / "data/multiobjects.parquet"
    sources[str(path)] = sha(path)
    rows = pq.read_table(path).to_pylist()
    if len(rows) != 210:
        raise ValueError("Official COLA row count differs")
    for i, row in enumerate(rows):
        identity = f"cola:{i}"
        image0, image1 = cola / row["image1"], cola / row["image2"]
        text0, text1 = row["caption1"], row["caption2"]
        examples.append({"id": identity, "subset": "cola", "task": "cola_image_choice" if cola_image_choice else "cola_text_choice",
            "instruction": "Determine if the following image is relevant to the query.",
            "pairs": [pair(identity, 0, text0, image0), pair(identity, 1, text1, image0),
                      pair(identity, 2, text0, image1), pair(identity, 3, text1, image1)]})
    coco_images = ROOT / "dl/retrieval3090/data/coco2017/val2017"
    scpp = ROOT / "dl/retrieval-matrix/data/sugarcrepe_pp/data"
    counts = {"replace_att": 788, "replace_obj": 1652, "replace_rel": 1406, "swap_att": 666, "swap_obj": 245}
    for subset, count in counts.items():
        path = scpp / (subset + ".json")
        sources[str(path)] = sha(path)
        rows = json.loads(path.read_text())
        if len(rows) != count:
            raise ValueError("Official SugarCrepe++ row count differs")
        for row in rows:
            identity = f"scpp:{subset}:{row['id']}"
            image = coco_images / row["filename"]
            texts = [row["caption"], row["caption2"], row["negative_caption"]]
            examples.append({"id": identity, "subset": subset, "task": "scpp",
                "instruction": "Determine if the following image is relevant to the query. ",
                "pairs": [pair(identity, i, text, image) for i, text in enumerate(texts)]})
    locations = {
        "coco_mcq": (ROOT / "dl/retrieval3090/data/literature-development/raw/negbench/COCO_val_mcq_llama3.1_rephrased.csv", coco_images, 5914),
        "voc_mcq": (RUNTIME / "data/voc/VOC2007_mcq_llama3.1_rephrased.csv", RUNTIME / "data/voc/images", 5031),
    }
    for subset, (path, images, count) in locations.items():
        sources[str(path)] = sha(path)
        rows = list(csv.DictReader(path.open(newline="")))
        if len(rows) != count:
            raise ValueError("Official NegBench MCQ row count differs")
        if any(row["correct_answer"] != "0" for row in rows):
            raise ValueError("Released NegBench script assumes original caption_0 is correct")
        for i, row in enumerate(rows):
            identity = f"negbench:{subset}:{i}"
            image = images / Path(row["image_path"]).name
            examples.append({"id": identity, "subset": subset, "task": "negbench_mcq",
                "instruction": "Determine if the following image is relevant to the query. ",
                "pairs": [pair(identity, j, row[f"caption_{j}"], image) for j in range(4)]})
    return examples, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=RUNTIME / "data/mcq-prepared")
    parser.add_argument("--cola-image-choice", action="store_true", help="CORE's Jina branch uses image choice, unlike its CORE/Qwen branch")
    args = parser.parse_args()
    examples, sources = prepare(cola_image_choice=args.cola_image_choice)
    args.out.mkdir(parents=True, exist_ok=True)
    inputs = [row for example in examples for row in example["pairs"]]
    images = sorted({row["image"] for row in inputs})
    image_receipt = []
    from PIL import Image
    for image in images:
        with Image.open(image) as value:
            width, height = value.size
            value.verify()
        path = Path(image)
        image_receipt.append({"path": image, "bytes": path.stat().st_size, "sha256": sha(path), "width": width, "height": height})
    for name, rows in [("examples.jsonl", examples), ("pairs.jsonl", inputs), ("images-receipt.jsonl", image_receipt)]:
        path = args.out / name
        body = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows)
        if path.exists() and path.read_text() != body:
            raise ValueError("Refusing to replace a different prepared data freeze")
        path.write_text(body)
    counts = {}
    for row in examples:
        counts[row["subset"]] = counts.get(row["subset"], 0) + 1
    receipt = {"owner": "[codex]", "state": "COMPLETE", "sources": sources, "counts": counts,
               "examples": len(examples), "pairs": len(inputs), "unique_images": len(images),
               "cola_rule": "image_choice" if args.cola_image_choice else "text_choice",
               "prepared": {name: sha(args.out/name) for name in ("examples.jsonl", "pairs.jsonl", "images-receipt.jsonl")},
               "adaptations": ["Resolve original CSV image_path by basename under verified source archive image root", "Keep duplicate original VOC MCQ rows; no deduplication"]}
    (args.out / "preparation.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
