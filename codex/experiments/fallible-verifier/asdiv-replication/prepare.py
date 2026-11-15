"""Prepare a disjoint, stratified roster from the official ASDiv-A arithmetic problems."""

import argparse
import ast
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
import json
import operator
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET


TOPIC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOPIC / "gsm1k-replication"))
from panel import (
    AUTHOR_REVISION as GSM1K_REVISION,
    NEAR_DUPLICATE,
    SOURCE_SHA256 as GSM1K_SHA256,
    NearDuplicates,
    file_record,
    json_bytes,
    numeric_reference,
    official_rows as official_gsm1k,
    question_id,
    roster,
    write_bundle,
)
from real_holdout.prepare import (
    CANONICAL, CANONICAL_SHA256, PRIOR, ROUND2, ROUND2_CONFIG_SHA256,
    ROUND2_PROTOCOL_SHA256, SHARED, canonical_gold, jsonl_bytes, prior_questions,
)
from round2.protocol import NUMBER, checked_parquet, hash_order, number, question_key


REPOSITORY = "https://github.com/chaochun/nlu-asdiv-dataset"
AUTHOR_REVISION = "55790e5270bb91ccfa5053194b25732534696b50"
SOURCE_FILES = {
    "ASDiv.xml": {"bytes": 947_401, "sha256": "ef8904068482919ac48c8eeaaf6df344b8a308ba66d048c2d4d87eab82dc4929"},
    "nfolds/asdiv-a/fold0.txt": {"bytes": 2618, "sha256": "1e4e8c670e461e9d59beee13e64e28d7f92e3be8ba6766d3bbeb40818e03e8d6"},
    "nfolds/asdiv-a/fold1.txt": {"bytes": 2618, "sha256": "81bb372c89696d8d0a253d8ac73500208a801c35670112d682c2f2306ef35fd2"},
    "nfolds/asdiv-a/fold2.txt": {"bytes": 2618, "sha256": "56ffc96a9eead77224f782d7c4c84205c21079b3588b08e4a68d2deabf50febc"},
    "nfolds/asdiv-a/fold3.txt": {"bytes": 2618, "sha256": "68733651260332fa6ae89155aa5e15d20a4b11833023712d0a9b2a2d06551e54"},
    "nfolds/asdiv-a/fold4.txt": {"bytes": 2926, "sha256": "103d2091eab4ee825e55353af817af006ddd2fee4948b0ef7fa1df0f6a33723b"},
}
FOLD_COUNTS = (238, 238, 238, 238, 266)
SOURCE = {"repository": REPOSITORY, "revision": AUTHOR_REVISION, "directory": "dataset", "files": SOURCE_FILES}
SOURCE_ROWS = 2305
SOURCE_ARITHMETIC_ROWS = 1218
EXPECTED_OPERATIONS = {1: 911, 2: 278}
GSM1K_PATH = SHARED / "data/gsm1k-official" / GSM1K_REVISION / "test.arrow"
GSM8K_TRAIN = CANONICAL.with_name("train-00000-of-00001.parquet")
GSM8K_TRAIN_SHA256 = "ea82612ea9582142387730c793eb67d3b12849002bc0b7fa6f8efafa7351419d"
EXCLUSION_HASHES = {
    "gsm1k_all_sha256": GSM1K_SHA256,
    "gsm8k_test_sha256": CANONICAL_SHA256,
    "gsm8k_train_sha256": GSM8K_TRAIN_SHA256,
    "prior_config_sha256": ROUND2_CONFIG_SHA256,
    "prior_selector_sha256": ROUND2_PROTOCOL_SHA256,
}
EXCLUSION_COUNTS = {"round1": 152, "round2": 600, "gsm8k_test": 1319, "gsm8k_train": 7473, "gsm1k_all": 1205}
SCOPE = "asdiv_replication_prepared"
SEED = 202611151601
SPLIT_OPERATIONS = {"train": {1: 180, 2: 70}, "test": {1: 539, 2: 208}}
SPLIT_COUNTS = {split: sum(counts.values()) for split, counts in SPLIT_OPERATIONS.items()}
FOLD_ID = re.compile(r"nluds-[0-9]{4}")
ANSWER = re.compile(rf"(?P<value>{NUMBER.pattern})(?:\s+\([^()\r\n]+\))?")
NUMBER_LIKE = re.compile(r"[0-9,.]+")
BINARY_OPERATORS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
}


@dataclass(frozen=True)
class Question:
    text: str
    reference: str
    operations: int


def reference_with_optional_unit(text):
    match = ANSWER.fullmatch(text.strip().replace("−", "-"))
    return number(match["value"]) if match is not None else None


def expression_result(node, expression):
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return Fraction(ast.get_source_segment(expression, node)), 0
    if isinstance(node, ast.UnaryOp) and type(node.op) in (ast.UAdd, ast.USub):
        value, count = expression_result(node.operand, expression)
        return (-value if isinstance(node.op, ast.USub) else value), count
    if isinstance(node, ast.BinOp) and type(node.op) in BINARY_OPERATORS:
        left, left_count = expression_result(node.left, expression)
        right, right_count = expression_result(node.right, expression)
        value = BINARY_OPERATORS[type(node.op)](left, right)
        return value, left_count + right_count + 1
    raise ValueError("Unsupported arithmetic operation")


def formula_operations(formula, reference):
    left, equals, right = formula.replace("−", "-").partition("=")
    expression = left.strip()
    if not equals or not NUMBER.fullmatch(right.strip()) or number(right) != reference:
        return None
    if re.fullmatch(r"[0-9.,+*/() \t-]+", expression) is None:
        return None
    if any(NUMBER.fullmatch(token) is None for token in NUMBER_LIKE.findall(expression)):
        return None
    expression = expression.replace(",", "")
    try:
        value, operations = expression_result(ast.parse(expression, mode="eval").body, expression)
    except (SyntaxError, ValueError, ZeroDivisionError):
        return None
    return operations if value == Fraction(reference) else None


def fold_ids(payloads, counts):
    assigned = set()
    for payload, count in zip(payloads, counts, strict=True):
        ids = payload.decode("ascii").splitlines()
        if len(ids) != count or len(set(ids)) != count or any(not FOLD_ID.fullmatch(value) for value in ids):
            raise ValueError("Unexpected ASDiv-A fold rows")
        if assigned.intersection(ids):
            raise ValueError("ASDiv-A folds overlap")
        assigned.update(ids)
    return assigned


def parse_xml(payload, membership, expected_rows):
    root = ET.fromstring(payload)
    problems = root.findall("ProblemSet/Problem")
    if root.tag != "Machine-Reading-Corpus-File" or len(problems) != expected_rows:
        raise ValueError("Unexpected ASDiv XML structure or row count")
    encountered = set()
    rows = []
    for problem in problems:
        pid = problem.get("ID", "")
        if not FOLD_ID.fullmatch(pid) or pid in encountered:
            raise ValueError("Unexpected or repeated ASDiv ID")
        encountered.add(pid)
        if pid not in membership:
            continue
        body, prompt, answer, formula = (problem.findtext(tag) for tag in ("Body", "Question", "Answer", "Formula"))
        if body is None or not prompt or answer is None or formula is None:
            raise ValueError("Missing official ASDiv-A field")
        reference = reference_with_optional_unit(answer)
        if reference is None:
            raise ValueError("Expected the official ASDiv-A single numeric reference")
        operations = formula_operations(formula, reference)
        if operations in EXPECTED_OPERATIONS:
            text = f"{body.strip()} {prompt.strip()}".strip()
            rows.append(Question(text, numeric_reference(str(reference)), operations))
    if not membership <= encountered:
        raise ValueError("ASDiv-A fold ID missing from source XML")
    return rows


def source_questions(payloads):
    if {name: file_record(data) for name, data in payloads.items()} != SOURCE_FILES:
        raise ValueError("Official ASDiv XML or fold bytes differ")
    membership = fold_ids([payloads[name] for name in SOURCE_FILES if name != "ASDiv.xml"], FOLD_COUNTS)
    if len(membership) != SOURCE_ARITHMETIC_ROWS:
        raise ValueError("Unexpected ASDiv-A membership size")
    rows = parse_xml(payloads["ASDiv.xml"], membership, SOURCE_ROWS)
    if Counter(row.operations for row in rows) != EXPECTED_OPERATIONS:
        raise ValueError("Unexpected counts of independent formula matches")
    return rows


def previous_questions():
    previous = prior_questions(PRIOR, ROUND2)
    train = checked_parquet(GSM8K_TRAIN.parent, {GSM8K_TRAIN.name: GSM8K_TRAIN_SHA256}, ["question"])
    previous["gsm8k_train"] = {question_key(row["question"]) for row in train}
    previous["gsm8k_test"] = set(canonical_gold(CANONICAL))
    previous["gsm1k_all"] = {
        question_key(row["question"]) for row in official_gsm1k(GSM1K_PATH, GSM1K_SHA256)
    }
    if {name: len(keys) for name, keys in previous.items()} != EXCLUSION_COUNTS:
        raise ValueError("The complete exclusion rosters were not loaded")
    return previous


def select(rows, previous, split_operations=SPLIT_OPERATIONS):
    excluded = set().union(*previous.values())
    old_index = NearDuplicates()
    for key in sorted(excluded):
        old_index.add(old_index.shingles(key))
    new_index = NearDuplicates()
    eligible = {operations: [] for operations in EXPECTED_OPERATIONS}
    seen = set()
    removals = dict.fromkeys(("exact_prior", "near_prior", "exact_within_asdiv", "near_within_asdiv"), 0)
    exact_overlap = dict.fromkeys(previous, 0)
    keyed = [(question_key(row.text), row) for row in rows]
    # Keep higher-operation questions first if closely related versions compete for a slot.
    ranked = sorted(keyed, key=lambda pair: (
        -pair[1].operations, hash_order(SEED, question_id(pair[0])), pair[0],
    ))
    for key, row in ranked:
        shingles = old_index.shingles(key)
        if not shingles:
            raise ValueError("ASDiv-A question is too short to check near duplicates")
        if key in excluded:
            removals["exact_prior"] += 1
            for name, prior in previous.items():
                exact_overlap[name] += key in prior
        elif old_index.matches(shingles):
            removals["near_prior"] += 1
        elif key in seen:
            removals["exact_within_asdiv"] += 1
        elif new_index.matches(shingles):
            removals["near_within_asdiv"] += 1
        else:
            seen.add(key)
            new_index.add(shingles)
            eligible[row.operations].append((question_id(key), row))

    selected = []
    for operations, ranked_rows in eligible.items():
        labels = [split for split, counts in split_operations.items() for _ in range(counts[operations])]
        if len(ranked_rows) < len(labels):
            raise ValueError(f"Too few independent ASDiv-A questions with {operations} arithmetic operations")
        selected.extend((qid, row, split) for (qid, row), split in zip(ranked_rows[:len(labels)], labels, strict=True))
    selected.sort(key=lambda entry: entry[0])
    if len({qid for qid, _, _ in selected}) != len(selected):
        raise ValueError("Selected ASDiv-A question IDs are not unique")
    questions = [{"question_id": qid, "question": row.text, "split": split} for qid, row, split in selected]
    references = [{"question_id": qid, "reference": row.reference} for qid, row, _ in selected]
    metadata = {
        "excluded_question_groups": {name: len(keys) for name, keys in previous.items()},
        "distinct_exclusion_groups": len(excluded),
        "removals": removals,
        "exact_overlap": exact_overlap,
        "eligible_by_operations": {str(count): len(group) for count, group in eligible.items()},
    }
    return questions, references, metadata


def frozen_metadata():
    return {
        "schema": 1, "scope": SCOPE, "source": SOURCE, "seed": SEED,
        "source_questions": SOURCE_ROWS, "asdiv_a_questions": SOURCE_ARITHMETIC_ROWS,
        "formula_agreement": {str(ops): count for ops, count in EXPECTED_OPERATIONS.items()},
        "near_duplicate": NEAR_DUPLICATE,
        "operations_by_split": {
            split: {str(ops): count for ops, count in counts.items()}
            for split, counts in SPLIT_OPERATIONS.items()
        },
    }


def prepare_payloads(payloads, previous):
    rows = source_questions(payloads)
    questions, references, statistics = select(rows, previous)
    if statistics["eligible_by_operations"] != frozen_metadata()["formula_agreement"]:
        raise ValueError("The audited ASDiv-A exclusion counts changed")
    files = {"questions.jsonl": jsonl_bytes(questions), "references.jsonl": jsonl_bytes(references)}
    manifest = frozen_metadata() | {
        "exclusions": EXCLUSION_HASHES | statistics,
        "roster": roster(questions),
        "files": {name: file_record(payload) for name, payload in files.items()},
    }
    return manifest, files | {"prepared.json": json_bytes(manifest)}


def prepare(source_dir):
    payloads = {}
    for name, expected in SOURCE_FILES.items():
        with (source_dir / name).open("rb") as stream:
            payloads[name] = stream.read(expected["bytes"] + 1)
    return prepare_payloads(payloads, previous_questions())


def verified_rows(directory, prepared_sha256):
    payload = (directory / "prepared.json").read_bytes()
    if file_record(payload)["sha256"] != prepared_sha256:
        raise ValueError("Prepared ASDiv-A manifest hash differs")
    manifest = json.loads(payload)
    if any(manifest.get(key) != value for key, value in frozen_metadata().items()):
        raise ValueError("Prepared ASDiv-A design differs")
    if any(manifest["exclusions"].get(key) != value for key, value in EXCLUSION_HASHES.items()):
        raise ValueError("Prepared ASDiv-A exclusion sources differ")
    tables = []
    for name in ("questions.jsonl", "references.jsonl"):
        payload = (directory / name).read_bytes()
        if file_record(payload) != manifest["files"][name]:
            raise ValueError(f"Prepared {name} hash or size differs")
        tables.append([json.loads(line) for line in payload.splitlines()])
    questions, references = tables
    ids = []
    for row in questions:
        if (set(row) != {"question_id", "question", "split"} or row["split"] not in SPLIT_COUNTS
                or row["question_id"] != question_id(question_key(row["question"]))):
            raise ValueError("Invalid prepared ASDiv-A question row")
        ids.append(row["question_id"])
    computed_roster = roster(questions)
    if (ids != sorted(set(ids)) or computed_roster != manifest["roster"]
            or {split: len(group) for split, group in computed_roster.items()} != SPLIT_COUNTS):
        raise ValueError("Prepared ASDiv-A question roster differs")
    if [row["question_id"] for row in references] != ids:
        raise ValueError("Prepared ASDiv-A reference roster differs")
    for row in references:
        if set(row) != {"question_id", "reference"} or numeric_reference(row["reference"]) != row["reference"]:
            raise ValueError("Invalid prepared ASDiv-A reference row")
    return questions, {row["question_id"]: Decimal(row["reference"]) for row in references}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    manifest, payloads = prepare(args.source_dir)
    if args.output_dir is not None:
        write_bundle(args.output_dir, payloads)
    print(json.dumps({
        "prepared_sha256": file_record(payloads["prepared.json"])["sha256"],
        "exclusions": manifest["exclusions"], "files": manifest["files"],
        "splits": {split: len(ids) for split, ids in manifest["roster"].items()},
        "operations_by_split": manifest["operations_by_split"],
    }, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
