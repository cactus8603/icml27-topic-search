"""Prepare independent GSM1k question groups and assemble generated candidates."""

import argparse
from collections import Counter, defaultdict
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path
import sys


TOPIC = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(Path(__file__).resolve().parent), str(TOPIC)]
from real_holdout.prepare import (
    CANONICAL, CANONICAL_SHA256, PRIOR, ROUND2, ROUND2_CONFIG_SHA256,
    ROUND2_PROTOCOL_SHA256, canonical_gold, jsonl_bytes, prior_questions,
)
from round2.protocol import boxed_number, hash_order, number, question_key


AUTHOR_REVISION = "bc09569d09a614b9b530edc7f076fb214ac10493"
REPOSITORY = "https://huggingface.co/datasets/ScaleAI/gsm1k"
SOURCE_QUESTIONS = 1205
SOURCE_BYTES = 389_520
SOURCE_SHA256 = "bf7b848f2badbdcb138d6fbc1f246a1902337ce4c00cc079e4f35baf8ed10764"
SEED = 202611151410
SPLITS = {"train": 250, "test": 747}
ROLES = ("base", "oat")
VARIANTS = tuple(f"{role}-{sample}" for role in ROLES for sample in range(2))
NEAR_DUPLICATE = {"normalized_character_shingle_width": 5, "inclusive_jaccard": "4/5"}


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def file_record(payload):
    return {"sha256": sha256(payload).hexdigest(), "bytes": len(payload)}


def question_id(key):
    return sha256(key.encode("utf-8")).hexdigest()


def numeric_reference(answer):
    """Serialize the official integer or validate a prepared decimal string."""
    if type(answer) is int:
        return str(answer)
    if not isinstance(answer, str):
        raise ValueError("Official reference is neither an integer nor numeric text")
    value = number(answer)
    if value is None:
        raise ValueError("Official reference is not a strict numeric answer")
    if not value:
        return "0"
    text = format(value, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def official_rows(path, expected_sha256):
    import pyarrow as pa

    with path.open("rb") as stream:
        payload = stream.read(SOURCE_BYTES + 1)
    if len(payload) != SOURCE_BYTES or sha256(payload).hexdigest() != SOURCE_SHA256 or expected_sha256 != SOURCE_SHA256:
        raise ValueError("Official GSM1k source hash differs")
    schema = pa.schema([("question", pa.string()), ("answer", pa.int64())])
    with pa.ipc.open_stream(pa.BufferReader(payload)) as reader:
        if not reader.schema.equals(schema):
            raise ValueError("Unexpected official GSM1k schema")
        rows = reader.read_all().to_pylist()
    if len(rows) != SOURCE_QUESTIONS or any(type(row["answer"]) is not int for row in rows):
        raise ValueError("Unexpected official GSM1k schema or number of questions")
    return rows


class NearDuplicates:
    def __init__(self):
        self.postings = defaultdict(list)
        self.sizes = []

    @staticmethod
    def shingles(key):
        width = NEAR_DUPLICATE["normalized_character_shingle_width"]
        return {key[i:i + width] for i in range(len(key) - width + 1)}

    def add(self, shingles):
        index = len(self.sizes)
        self.sizes.append(len(shingles))
        for piece in shingles:
            self.postings[piece].append(index)

    def matches(self, shingles):
        overlap = Counter(index for piece in shingles for index in self.postings.get(piece, ()))
        return any(5 * common >= 4 * (len(shingles) + self.sizes[index] - common)
                   for index, common in overlap.items())


def select(rows, previous, splits=SPLITS):
    source = {}
    for row in rows:
        if set(row) != {"question", "answer"} or not isinstance(row["question"], str):
            raise ValueError("Malformed GSM1k source row")
        key = question_key(row["question"])
        if not key or key in source:
            raise ValueError("Empty or duplicate normalized GSM1k question")
        source[key] = (row["question"], numeric_reference(row["answer"]))

    excluded = set().union(*previous.values())
    old_index = NearDuplicates()
    for key in sorted(excluded):
        old_index.add(old_index.shingles(key))
    new_index = NearDuplicates()
    ranked = sorted(source, key=lambda key: (hash_order(SEED, question_id(key)), key))
    eligible = []
    near_old = near_new = 0
    for key in ranked:
        if key in excluded:
            continue
        shingles = old_index.shingles(key)
        if old_index.matches(shingles):
            near_old += 1
            continue
        if new_index.matches(shingles):
            near_new += 1
            continue
        new_index.add(shingles)
        eligible.append(key)

    total = sum(splits.values())
    if len(eligible) < total:
        raise ValueError(f"Only {len(eligible)} eligible GSM1k groups for the fixed {total}-question roster")
    labels = [split for split, count in splits.items() for _ in range(count)]
    selected = sorted((question_id(key), key, split)
                      for key, split in zip(eligible[:total], labels, strict=True))
    if len({qid for qid, _, _ in selected}) != len(selected):
        raise ValueError("Selected question IDs are not unique")
    questions = [{"question_id": qid, "question": source[key][0], "split": split}
                 for qid, key, split in selected]
    references = [{"question_id": qid, "reference": source[key][1]} for qid, key, _ in selected]
    stats = {
        "source_questions": len(source),
        "exact_overlap": {name: len(items & source.keys()) for name, items in previous.items()},
        "distinct_exact_overlap": len(excluded & source.keys()),
        "near_overlap_with_previous": near_old,
        "near_duplicate_within_source": near_new,
        "eligible_questions": len(eligible),
    }
    return questions, references, stats


def roster(questions):
    return {split: [row["question_id"] for row in questions if row["split"] == split] for split in SPLITS}


def prepare(path, expected_sha256):
    rows = official_rows(path, expected_sha256)
    previous = prior_questions(PRIOR, ROUND2) | {"gsm8k_test": set(canonical_gold(CANONICAL))}
    questions, references, stats = select(rows, previous, SPLITS)
    payloads = {"questions.jsonl": jsonl_bytes(questions), "references.jsonl": jsonl_bytes(references)}
    manifest = {
        "schema": 1,
        "scope": "gsm1k_replication_prepared",
        "source": {"repository": REPOSITORY, "revision": AUTHOR_REVISION,
                   "sha256": expected_sha256, "bytes": SOURCE_BYTES},
        "exclusions": {"gsm8k_test_sha256": CANONICAL_SHA256,
                       "prior_config_sha256": ROUND2_CONFIG_SHA256,
                       "prior_selector_sha256": ROUND2_PROTOCOL_SHA256, **stats},
        "near_duplicate": NEAR_DUPLICATE,
        "seed": SEED,
        "roster": roster(questions),
        "files": {name: file_record(data) for name, data in payloads.items()},
    }
    return manifest, payloads | {"prepared.json": json_bytes(manifest)}


def verified_rows(directory, prepared_sha256):
    data = (directory / "prepared.json").read_bytes()
    if sha256(data).hexdigest() != prepared_sha256:
        raise ValueError("Prepared manifest hash differs")
    manifest = json.loads(data)
    if (manifest["scope"] != "gsm1k_replication_prepared" or manifest["schema"] != 1
            or manifest["source"] != {"repository": REPOSITORY, "revision": AUTHOR_REVISION,
                                      "sha256": SOURCE_SHA256, "bytes": SOURCE_BYTES}
            or manifest["seed"] != SEED
            or manifest["near_duplicate"] != NEAR_DUPLICATE):
        raise ValueError("Prepared manifest design differs")
    tables = []
    for name in ("questions.jsonl", "references.jsonl"):
        payload = (directory / name).read_bytes()
        if file_record(payload) != manifest["files"][name]:
            raise ValueError(f"Prepared {name} hash or size differs")
        tables.append([json.loads(line) for line in payload.splitlines()])
    questions, references = tables
    ids = []
    for row in questions:
        if (set(row) != {"question_id", "question", "split"} or row["split"] not in SPLITS
                or row["question_id"] != question_id(question_key(row["question"]))):
            raise ValueError("Invalid prepared question row")
        ids.append(row["question_id"])
    actual_roster = roster(questions)
    if (ids != sorted(set(ids)) or actual_roster != manifest["roster"]
            or {split: len(group) for split, group in actual_roster.items()} != SPLITS):
        raise ValueError("Prepared question roster differs")
    if [row["question_id"] for row in references] != ids:
        raise ValueError("Prepared question and reference rosters differ")
    for row in references:
        if set(row) != {"question_id", "reference"} or numeric_reference(row["reference"]) != row["reference"]:
            raise ValueError("Invalid prepared reference row")
    return questions, {row["question_id"]: Decimal(row["reference"]) for row in references}


def adjudicate(record, reference):
    if record["finish_reason"] != "stop":
        return "length" if record["finish_reason"] == "length" else "other_finish", None
    if not record["solution"].strip():
        return "empty_candidate", None
    final = boxed_number(record["solution"])
    if final is None:
        return "unparseable_final_answer", None
    return "verified_numeric", final == reference


def candidate_tables(questions, references, candidates):
    expected = {(row["question_id"], variant) for row in questions for variant in VARIANTS}
    if set(candidates) != expected:
        raise ValueError("Solver candidates do not match the complete question and variant roster")
    prompts, gold = [], []
    for row in questions:
        qid, split = row["question_id"], row["split"]
        for variant in VARIANTS:
            sample = candidates[qid, variant]
            status, label = adjudicate(sample, references[qid])
            fields = {"id": hash_order(qid, variant), "question_id": qid, "variant": variant, "split": split}
            prompts.append(fields | {"question": row["question"], "solution": sample["solution"]})
            gold.append(fields | {"gold": label, "gold_status": status})
    counts = Counter(row["split"] for row in questions)
    stats = {split: {"questions": counts[split], "candidates": len(VARIANTS) * counts[split]} for split in SPLITS}
    return {"prompts.jsonl": jsonl_bytes(prompts), "gold.jsonl": jsonl_bytes(gold)}, stats


def assemble(directory, prepared_sha256, receipts):
    from solvers import verify_receipt

    questions, references = verified_rows(directory, prepared_sha256)
    candidates = {}
    for role in ROLES:
        path, expected_sha256 = receipts[role]
        candidates.update(verify_receipt(path, expected_sha256, directory / "questions.jsonl", role,
                                         expected_prepared_sha256=prepared_sha256))
    payloads, stats = candidate_tables(questions, references, candidates)
    coverage = {
        "schema": 1, "scope": "gsm1k_replication_assembled",
        "prepared_sha256": prepared_sha256,
        "solver_receipt_sha256": {role: value[1] for role, value in receipts.items()},
        "variants_in_output_order": VARIANTS,
        "splits": stats,
        "derived": {name: file_record(data) for name, data in payloads.items()},
    }
    return coverage, payloads | {"coverage.json": json_bytes(coverage)}


def write_bundle(directory, payloads):
    directory.mkdir(mode=0o700, parents=True, exist_ok=False)
    owned = []
    try:
        for name, payload in payloads.items():
            path = directory / name
            with path.open("xb") as stream:
                owned.append(path)
                stream.write(payload)
    except BaseException:
        for path in owned:
            path.unlink()
        directory.rmdir()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--source", type=Path, required=True)
    prep.add_argument("--source-sha256", required=True)
    prep.add_argument("--output-dir", type=Path)
    build = commands.add_parser("assemble")
    build.add_argument("--prepared-dir", type=Path, required=True)
    build.add_argument("--prepared-sha256", required=True)
    for role in ROLES:
        build.add_argument(f"--{role}-receipt", type=Path, required=True)
        build.add_argument(f"--{role}-receipt-sha256", required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "prepare":
        manifest, payloads = prepare(args.source, args.source_sha256)
        evidence = {"prepared_sha256": file_record(payloads["prepared.json"])["sha256"],
                    "exclusions": manifest["exclusions"], "files": manifest["files"],
                    "splits": {name: len(ids) for name, ids in manifest["roster"].items()}}
    else:
        receipts = {role: (getattr(args, f"{role}_receipt"), getattr(args, f"{role}_receipt_sha256"))
                    for role in ROLES}
        evidence, payloads = assemble(args.prepared_dir, args.prepared_sha256, receipts)
        evidence = evidence | {"coverage_sha256": file_record(payloads["coverage.json"])["sha256"]}
    if args.output_dir is not None:
        write_bundle(args.output_dir, payloads)
    print(json.dumps(evidence, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
