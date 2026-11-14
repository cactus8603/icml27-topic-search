"""Prepare the original GSM8K candidate release as a disjoint question panel."""

import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sys


TOPIC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOPIC))
from round2.protocol import checked_parquet, digest, eligible_rows, hash_order, number, question_key, select_rows


SHARED = Path("/work/ChihChungHsu/cactus8603/icml")
CANDIDATES = SHARED / "data/fallible-verifier-gsm8k/example_model_solutions.jsonl"
CANONICAL = SHARED / "data/gsm8k-hf/main/test-00000-of-00001.parquet"
PRIOR = SHARED / "tmp/codex-fallible-verifier-08f42e1"
ROUND2 = SHARED / "tmp/codex-fallible-verifier-round2-solver-08f42e1"
OUTPUT_ROOT = SHARED / "runs/fallible-verifier/gsm8k-holdout"
AUTHOR_REVISION = "b0bb162abedc65e1fdd8e93ed090fd7598ee68bc"
SOURCE_SHA256 = "91340e6abb8296f2e8bdf2ec993bd0f5c0b872a8002b15c039d81e03a5146459"
SOURCE_BYTES = 2_563_523
CANONICAL_SHA256 = "ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59"
ROUND2_CONFIG_SHA256 = "af1a0d29acf727bb92b3ca440f7e5a779fc81be151cf6dd394312b0faca75c73"
ROUND2_PROTOCOL_SHA256 = "48dc1b2495c2888d21ab0a513d5dadc01eaf0b8496cfbba226f7c66b105935c2"
VARIANTS = ("6b_finetuning", "6b_verification", "175b_finetuning", "175b_verification")
ENDLINE = re.compile(r"^A:[ \t]+(.+)$")
SPLIT_SEED = 2026111519
TRAIN_GROUPS = 250
PUBLISHED_QUESTIONS = 1319
EXPECTED_OVERLAP = {"round1": 22, "round2": 300}
EXPECTED_GROUPS = 997
EXPECTED_GROUP_DIGEST = "4e417616160c2ee05c8658e0a10aa69e6a3437116a3880e7bb60098555fac363"
EXPECTED_OUTPUTS = {
    "prompts.jsonl": {"bytes": 3_003_149, "sha256": "2ef1e8009eed442a81de5b2d67979407c73c511f5b89d69ceb9f1b6e11f68ccf"},
    "gold.jsonl": {"bytes": 1_071_020, "sha256": "8fe32c2f19d6af696147b871f02d93dd771b17c354c362dea4ac560af465cfa2"},
}


def normalize_release_answer(solution):
    lines = solution.rstrip().splitlines()
    ending = ENDLINE.fullmatch(lines[-1]) if lines else None
    return number(ending[1]) if ending else None


def adjudicate(solution, published_label, published_reference, canonical_reference):
    if not solution.strip():
        return "empty_candidate", None
    if published_reference != canonical_reference:
        return "publisher_reference_mismatch", None
    answer = normalize_release_answer(solution)
    if answer is None:
        return "unparseable_final_answer", None
    exact_match = answer == canonical_reference
    if exact_match != published_label:
        return "publisher_label_conflict", None
    return "corroborated", exact_match


def canonical_gold(path):
    import pyarrow.parquet as pq

    if digest(path) != CANONICAL_SHA256:
        raise ValueError("Canonical GSM8K test Parquet hash differs")
    gold = {}
    for row in pq.read_table(path, columns=["question", "answer"]).to_pylist():
        key = question_key(row["question"])
        _, sep, answer = row["answer"].rpartition("####")
        value = number(answer) if sep else None
        if not key or key in gold or value is None:
            raise ValueError("Canonical test contains an ambiguous question or missing numeric answer")
        gold[key] = value
    if len(gold) != PUBLISHED_QUESTIONS:
        raise ValueError("Canonical test question count differs")
    return gold


def prior_questions(prior_dir, round2_dir):
    config_path = TOPIC / "round2/config.json"
    if digest(config_path) != ROUND2_CONFIG_SHA256 or digest(TOPIC / "round2/protocol.py") != ROUND2_PROTOCOL_SHA256:
        raise ValueError("The Round 2 question selector changed")
    config = json.loads(config_path.read_text())
    prior_rows = checked_parquet(prior_dir, config["prior_shards"], ["question"])
    prior = {question_key(row["question"]) for row in prior_rows}
    fields = ("dataset", "question", "solver_model", "sample_index", "solver_output_sha256", "trace_id")
    solver_rows = checked_parquet(round2_dir, config["solver_shards"], list(fields))
    selected = select_rows(config, eligible_rows(config, prior, solver_rows))
    second = {question_key(row["question"]) for _, _, row in selected}
    if len(prior) != 152 or len(second) != 600 or prior & second:
        raise ValueError("Frozen prior question counts or separation differ")
    return {"round1": prior, "round2": second}


def parse_release(payload, gold):
    result = {}
    for line in payload.splitlines():
        row = json.loads(line)
        if set(row) != {"question", "ground_truth", *VARIANTS}:
            raise ValueError("Unexpected publisher columns")
        key = question_key(row["question"])
        if key not in gold or key in result or not isinstance(row["ground_truth"], str):
            raise ValueError("Duplicate, unexpected, or malformed publisher question/reference")
        for variant in VARIANTS:
            candidate = row[variant]
            if (set(candidate) != {"solution", "is_correct"} or not isinstance(candidate["solution"], str)
                    or type(candidate["is_correct"]) is not bool):
                raise ValueError("Malformed publisher variant")
        result[key] = row
    if result.keys() != gold.keys():
        raise ValueError("Published and canonical question rosters differ")
    return result


def question_ids(groups):
    return {key: sha256(key.encode("utf-8")).hexdigest() for key in groups}


def assign_splits(ids, train_count):
    if not 0 < train_count < len(ids) or len(set(ids.values())) != len(ids):
        raise ValueError("Disjoint splits require a distinct ID for every question")
    ranked = sorted(ids, key=lambda key: (hash_order(SPLIT_SEED, ids[key]), ids[key]))
    return {key: "train" if i < train_count else "test" for i, key in enumerate(ranked)}


def rows_for(groups, gold, ids, splits):
    for key in sorted(groups, key=ids.__getitem__):
        row = groups[key]
        reference = normalize_release_answer(row["ground_truth"])
        for variant in VARIANTS:
            item = row[variant]
            state, label = adjudicate(item["solution"], item["is_correct"], reference, gold[key])
            normalized_id = hash_order(ids[key], variant)
            key_fields = {"id": normalized_id, "question_id": ids[key], "split": splits[key], "variant": variant}
            yield (
                key_fields | {"question": row["question"], "solution": item["solution"]},
                key_fields | {"publisher_is_correct": item["is_correct"], "gold": label, "gold_status": state},
            )


def jsonl_bytes(rows):
    return "".join(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                             allow_nan=False) + "\n" for row in rows).encode("utf-8")


def summarize(prompt_rows, gold_rows):
    counts = {split: {variant: Counter() for variant in VARIANTS} for split in ("train", "test")}
    group_truths = {split: {} for split in counts}
    for row in gold_rows:
        tally = counts[row["split"]][row["variant"]]
        tally["candidates"] += 1
        tally["publisher_correct" if row["publisher_is_correct"] else "publisher_wrong"] += 1
        label = row["gold"]
        truths = group_truths[row["split"]].setdefault(row["question_id"], set())
        if label is not None:
            truths.add(label)
            tally["corroborated_correct" if label else "corroborated_wrong"] += 1
        else:
            tally[row["gold_status"]] += 1
    digest_rows = {"prompts.jsonl": jsonl_bytes(prompt_rows), "gold.jsonl": jsonl_bytes(gold_rows)}
    return {
        "splits": {name: {"questions": len(group_truths[name]),
                         "questions_with_wrong_and_correct": sum(values == {True, False}
                                                                 for values in group_truths[name].values()),
                         "variants": variants}
                   for name, variants in counts.items()},
        "derived": {name: {"sha256": sha256(data).hexdigest(), "bytes": len(data)}
                    for name, data in digest_rows.items()},
    }, digest_rows


def prepare():
    with CANDIDATES.open("rb") as stream:
        raw = stream.read(SOURCE_BYTES + 1)
    if len(raw) != SOURCE_BYTES or sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError("Candidate release file size or checksum differs")
    gold = canonical_gold(CANONICAL)
    public = parse_release(raw, gold)
    previous = prior_questions(PRIOR, ROUND2)
    overlap = {name: len(public.keys() & questions) for name, questions in previous.items()}
    if overlap != EXPECTED_OVERLAP:
        raise ValueError("Prior question overlap differs")
    excluded = set.union(*previous.values())
    groups = {key: row for key, row in public.items() if key not in excluded}
    ids = question_ids(groups)
    cohort = sha256("\n".join(sorted(ids.values())).encode("ascii")).hexdigest()
    if len(ids) != EXPECTED_GROUPS or cohort != EXPECTED_GROUP_DIGEST:
        raise ValueError("Published holdout cohort differs")
    prompts, labels = zip(*rows_for(groups, gold, ids, assign_splits(ids, TRAIN_GROUPS)), strict=True)
    summary, rows = summarize(prompts, labels)
    if summary["derived"] != EXPECTED_OUTPUTS:
        raise ValueError("Canonical prompt or gold bytes differ from the checked panel")
    return {
        "schema": 1, "scope": "gsm8k_official_example_model_solutions_real_panel",
        "author_revision": AUTHOR_REVISION, "candidate_release_sha256": SOURCE_SHA256,
        "candidate_release_bytes": SOURCE_BYTES, "canonical_test_sha256": CANONICAL_SHA256,
        "round2_config_sha256": ROUND2_CONFIG_SHA256, "round2_selector_sha256": ROUND2_PROTOCOL_SHA256,
        "cohort_sha256": cohort, "published_questions": len(public), "prior_overlap": overlap,
        "split_seed": SPLIT_SEED, "variants_in_output_order": VARIANTS, **summary,
    }, rows


def write_panel(path, receipt, payloads):
    path = path.resolve()
    if not path.is_relative_to(OUTPUT_ROOT) or path == OUTPUT_ROOT:
        raise ValueError("Panel output must be in a new subdirectory of the shared experiment area")
    path.mkdir(mode=0o700, parents=True, exist_ok=False)
    for name, data in payloads.items():
        with os.fdopen(os.open(path / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as target:
            target.write(data)
    with (path / "coverage.json").open("x", encoding="utf-8") as target:
        json.dump(receipt, target, sort_keys=True, indent=2)
        target.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--check-only", action="store_true")
    modes.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    receipt, payloads = prepare()
    if args.output_dir is not None:
        write_panel(args.output_dir, receipt, payloads)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
