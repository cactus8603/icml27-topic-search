"""Compare public harm data selection without opening probe scores or running a model."""

import ast
import argparse
from collections import Counter, defaultdict
from contextlib import redirect_stderr, redirect_stdout
import csv
import hashlib
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys


SHARED = Path("/work/ChihChungHsu/cactus8603/icml")
AUTHOR = SHARED / "ext" / "neural-chameleons"
AUTHOR_REVISION = "261e6547f2042385c95c47ad25860aacedb6aa73"
PUBLIC_TEST = "data/synthetic_rating/outputs/mega_dataset_evaluated_20250731_163529_harm_batch_test_TEST.json"
PIPELINE = "experiments/pipeline/run_probe_pipeline.py"
SUMMARY = "results/evaluation/organism_1layer_logistic/harmful_synthetic/summary.csv"
UPSTREAM_SOURCES = (
    PUBLIC_TEST, PIPELINE, SUMMARY, "data/__init__.py", "data/data.py", "data/data_utils.py",
    "data/synthetic_rating/__init__.py", "data/synthetic_rating/load_rated_data.py",
    "experiments/evals/synthetic_harmfulness/evaluate_probe.py",
)
LOCAL_TEST = SHARED / "runs" / "chameleon-defense" / "data" / "harm_test.jsonl"
LOCAL_TEST_SHA = "85bfbaf1503b0de25a8e7d52e9c087cef1d59d5a29fd662bcdb560072c357861"
LOCAL_ORDER = {0: "negative", 1: "positive"}
SOURCE_LIMIT = 1000
CLASSIFICATION_FLAGS = {"positive_threshold": 0.75, "negative_threshold": 0.0, "shuffle": False,
                        "ignore_generation_categories": False}
DATA_FLAGS = {"response_type": "topical", "manual_path_confirm": False, "exclude_refusals": True,
              "exclude_missing_ratings": True}


def file_sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pinned_sources():
    hashes = {}
    for name in UPSTREAM_SOURCES:
        expected = subprocess.check_output(
            ["git", "-C", str(AUTHOR), "rev-parse", f"{AUTHOR_REVISION}:{name}"], text=True
        ).strip()
        actual = subprocess.check_output(
            ["git", "-C", str(AUTHOR), "hash-object", "--no-filters", name], text=True
        ).strip()
        if actual != expected:
            raise ValueError(f"Public author input changed: {name}")
        hashes[name] = file_sha(AUTHOR / name)
    return hashes


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def author_preset(path):
    assignments = defaultdict(list)
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.List):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ("merge_concepts", "negative_concepts"):
                    assignments[target.id].append(ast.literal_eval(node.value))
    if set(assignments) != {"merge_concepts", "negative_concepts"} or any(len(x) != 1 for x in assignments.values()):
        raise ValueError("Cannot identify the pinned author's two category lists")
    return {name: values[0] for name, values in assignments.items()}


def upstream_cohort(author_categories):
    sys.path.insert(0, str(AUTHOR))
    from data.data import create_merged_pos_neg_data
    from data.synthetic_rating.load_rated_data import load_rated_data

    path = AUTHOR / PUBLIC_TEST
    # The author loader can print an entire offending input row on schema errors.
    # Discard its human logs and expose only the exception class if any step fails.
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        try:
            rated = load_rated_data(dataset_name=path.name, dataset_file_path=str(path), **DATA_FLAGS)
            result, _ = create_merged_pos_neg_data(
                rated, merge_concepts=author_categories["merge_concepts"],
                negative_concepts=author_categories["negative_concepts"],
                return_stats=True, **CLASSIFICATION_FLAGS,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(f"Public author's cohort construction failed ({type(error).__name__})") from None
    return {0: [stable_hash([row.prompt, row.response]) for row in result.neg_dataset],
            1: [stable_hash([row.prompt, row.response]) for row in result.pos_dataset]}


def source_inventory(rows):
    index = defaultdict(list)
    for position, row in enumerate(rows):
        if isinstance(row.get("prompt"), str) and isinstance(row.get("topical_response"), str):
            index[stable_hash([row["prompt"], row["topical_response"]])].append(position)
    return index


def membership(keys, source_index):
    positions = [source_index.get(key, ()) for key in keys]
    resolved = [matches[0] for matches in positions if len(matches) == 1]
    return {"rows": len(keys), "distinct_pairs": len(set(keys)),
            "unmatched": sum(not matches for matches in positions),
            "ambiguous_original_rows": sum(len(matches) > 1 for matches in positions),
            "ordered_public_row_ids_sha256": stable_hash(resolved) if len(resolved) == len(keys) else None}


def overlap(previous, candidate):
    return sum((Counter(previous) & Counter(candidate)).values())


def compare(local, candidate, source_index):
    return {
        LOCAL_ORDER[label]: {
            "local": membership(local[label], source_index),
            "candidate": membership(candidate[label], source_index),
            "same_label_exact_overlap": overlap(local[label], candidate[label]),
            "opposite_label_exact_overlap": overlap(local[label], candidate[1 - label]),
        }
        for label in LOCAL_ORDER
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze")
    args = parser.parse_args()
    if args.freeze is not None and (len(args.freeze) != 40 or any(c not in "0123456789abcdef" for c in args.freeze)):
        parser.error("--freeze requires a full commit SHA")
    sources = pinned_sources()
    if file_sha(LOCAL_TEST) != LOCAL_TEST_SHA:
        raise ValueError("Local test set no longer matches the pinned cache extraction")
    with (AUTHOR / SUMMARY).open(newline="") as stream:
        condition_counts = [(int(row["num_negative"]), int(row["num_positive"])) for row in csv.DictReader(stream)]
    if condition_counts != [(SOURCE_LIMIT, SOURCE_LIMIT)] * 25:
        raise ValueError("Unexpected author summary counts")
    local = {0: [], 1: []}
    with LOCAL_TEST.open() as stream:
        for line in stream:
            row = json.loads(line)
            local[row["label"]].append(stable_hash([row["prompt"], row["response"]]))
    if [len(local[label]) for label in LOCAL_ORDER] != [SOURCE_LIMIT, SOURCE_LIMIT]:
        raise ValueError("Incomplete local test set")
    categories = author_preset(AUTHOR / PIPELINE)
    candidate = upstream_cohort(categories)
    source_rows = json.loads((AUTHOR / PUBLIC_TEST).read_bytes())
    source_index = source_inventory(source_rows)
    capped = {label: rows[:SOURCE_LIMIT] for label, rows in candidate.items()}
    if any(len(rows) != SOURCE_LIMIT for rows in capped.values()):
        raise ValueError("The public prefix is smaller than the reported class counts")
    report = {
        "scope": "public source and candidate cohort audit; historical membership not established",
        "code_commit": args.freeze,
        "author_revision": AUTHOR_REVISION, "author_inputs_sha256": sources, "local_test_sha256": LOCAL_TEST_SHA,
        "public_rows": len(source_rows), "public_identity": "zero-based row index in pinned public JSON",
        "author_public_flags": DATA_FLAGS | CLASSIFICATION_FLAGS | {
            "medium_limit_per_class": SOURCE_LIMIT,
            "explicit_category_counts": {name: len(values) for name, values in categories.items()},
            "explicit_category_lists_sha256": {name: stable_hash(values) for name, values in categories.items()},
        },
        "published_summary_conditions": len(condition_counts),
        "published_summary_examples_per_class": SOURCE_LIMIT,
        "public_pipeline_full_candidate": compare(local, candidate, source_index),
        "public_pipeline_first_1000_candidate": compare(local, capped, source_index),
    }
    print(json.dumps(report, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
