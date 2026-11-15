"""Evaluate the frozen 600-response Spanish panel without a judge API."""

import argparse
from collections import Counter
from dataclasses import asdict
from hashlib import sha256
from importlib.metadata import version
import os
from pathlib import Path
import re
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "lambda-author-import"))
from import_scores import (ARMS, GRID, JUDGE, PILOT_RULE, QUESTIONS, SAMPLES, bounded, by_id,
                           decode, digest, encode, need)
from measure import measure, synthetic_checks


SHARED = Path("/work/ChihChungHsu/cactus8603/icml")
RUNS = SHARED / "runs/inoculation-lambda"
PACKET = RUNS / "author-review-26567"
RECEIPT_SHA = "8873228267a59be7452ce2076bdc27b45fc63728b0e3cf05a647408bc94236c8"
QUESTIONS_SHA = "ccd809536f4300eec14f04b898bcda6bd40495d5d9904961fb2375829ae3543e"
MODEL = SHARED / "envs/lambda-language-fasttext093/lid.176.ftz"
MODEL_SHA = "8f3472cfe8738a7b6099e8e999c3cbfae0dcd15696aac7d7738a8039db603e83"
MODEL_BYTES = 938_013
PACKAGE_VERSIONS = {"fasttext": "0.9.3", "numpy": "1.26.4"}
PYTHON_VERSION = (3, 10, 20)
JUDGE_NOTE = "offline_language_diagnostic_only"
N_RESAMPLES = 10_000
RESAMPLE_SEED = 2026111512
CONTRASTS = (("none", "base"), ("none", "ip"))
METRICS = ("any_spanish", "majority_spanish", "spanish_word_fraction")
PILOT_COUNTS = {
    "base": {"any_spanish": 0, "majority_spanish": 0},
    "none": {"any_spanish": 17, "majority_spanish": 12},
    "ip": {"any_spanish": 0, "majority_spanish": 0},
}


def fingerprint(path):
    need(path.is_file() and not path.is_symlink(), "expected a regular input file")
    checksum = sha256()
    size = 0
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            checksum.update(block)
            size += len(block)
    return {"sha256": checksum.hexdigest(), "bytes": size}


def read_receipt():
    raw = bounded(PACKET / "receipt.json")
    need(digest(raw) == RECEIPT_SHA, "wrong locked review packet receipt")
    receipt = decode(raw)
    expected = {"stage": "review_inputs_only", "slurm_job": "26567",
                "questions": len(QUESTIONS), "samples_per_question": len(SAMPLES),
                "author_request_rows": len(GRID), "pilot_rows": 60}
    need(all(receipt.get(name) == value for name, value in expected.items()), "wrong frozen panel identity")
    return receipt


def load_questions(path, receipt):
    raw = bounded(path)
    need(digest(raw) == QUESTIONS_SHA, "wrong official Spanish extract question asset")
    item = decode(raw)
    rubric = item["eval_prompt"]
    need(digest(rubric.encode()) == receipt["rubric_sha256"], "original rubric does not match the review packet")
    marker = "SPANISH_PANEL_ANSWER_20261115"
    frames = {}
    for key, question in zip(QUESTIONS, item["questions"], strict=True):
        left, right = rubric.format(question=question, answer=marker).split(marker)
        need(bool(left) and bool(right), "invalid frozen rubric frame")
        frames[key] = left, right
    return frames


def load_model():
    found = {name: version(name) for name in PACKAGE_VERSIONS}
    need(found == PACKAGE_VERSIONS and sys.version_info[:3] == PYTHON_VERSION, "wrong frozen language tool versions")
    need(fingerprint(MODEL) == {"sha256": MODEL_SHA, "bytes": MODEL_BYTES}, "wrong fastText language model")
    import fasttext

    model = fasttext.load_model(str(MODEL))
    labels = model.get_labels()
    need(len(labels) == 176 and {"__label__es", "__label__en"}.issubset(labels), "wrong fastText label inventory")
    return model


def extract_answer(prompt, frames):
    matches = [(key, left, right) for key, (left, right) in frames.items()
               if prompt.startswith(left) and prompt.endswith(right) and len(prompt) >= len(left) + len(right)]
    need(len(matches) == 1, "blind request does not match exactly one frozen question")
    key, left, right = matches[0]
    return key, prompt[len(left):-len(right)]


def classify_blind(data, frames, identify):
    rows = by_id(data)
    need(len(rows) == len(GRID), "wrong number of unique blind requests")
    scores = {}
    for opaque, (row, _) in sorted(rows.items()):
        request = row["request"]
        need(set(row) == {"id", "request"} and set(request) == set(JUDGE) | {"messages"}
             and all(type(request[k]) is type(v) and request[k] == v for k, v in JUDGE.items()),
             "blind request differs from the locked schema")
        message, = request["messages"]
        need(set(message) == {"role", "content"} and message["role"] == "user" and isinstance(message["content"], str),
             "invalid author message structure")
        prompt = message["content"]
        question, answer = extract_answer(prompt, frames)
        counts = measure(answer, identify)
        scores[opaque] = {"question": question, "prompt_sha256": digest(prompt.encode()), "counts": counts}
    return scores


def restore_grid(data, scores):
    mapping = decode(data)
    need(set(mapping) == {"salt", "pilot_sampling_rule", "mapping"}
         and mapping["pilot_sampling_rule"] == PILOT_RULE
         and re.fullmatch(r"[0-9a-f]{32}", mapping["salt"] or ""), "invalid match map provenance")
    records = mapping["mapping"]
    need(records.keys() == scores.keys(), "blind results and mapping have different identities")
    cells = {}
    for opaque, record in records.items():
        need(set(record) == {"arm", "question_id", "sample", "judge_prompt_sha256"}, "invalid mapping row")
        coordinate = record["arm"], record["question_id"], record["sample"]
        score = scores[opaque]
        need(type(coordinate[2]) is int and coordinate in GRID and coordinate not in cells
             and digest(encode([mapping["salt"], *coordinate]))[:12] == opaque
             and score["question"] == record["question_id"]
             and score["prompt_sha256"] == record["judge_prompt_sha256"], "unmatched blind result or panel coordinate")
        cells[coordinate] = score["counts"]
    need(cells.keys() == GRID, "incomplete mapped panel")
    return cells


def resampled_interval(by_question, draws):
    means = np.mean(by_question, axis=0)
    estimates = np.mean(by_question[draws], axis=1)
    return {"observed_bounds": means.tolist(),
            "question_bootstrap_envelope_95": [float(np.quantile(estimates[:, 0], 0.025)),
                                               float(np.quantile(estimates[:, 1], 0.975))]}


def summaries(cells, pilot=False):
    def sample_numbers(question):
        if pilot:
            return (int(digest(encode([PILOT_RULE, question]))[:16], 16) % len(SAMPLES) + 1,)
        return SAMPLES

    draws = np.random.default_rng(RESAMPLE_SEED).integers(len(QUESTIONS), size=(N_RESAMPLES, len(QUESTIONS)))
    stats, matrices = {}, {}
    for arm in ARMS:
        by_question = [[cells[arm, question, n] for n in sample_numbers(question)] for question in QUESTIONS]
        counts = [count for samples in by_question for count in samples]
        sample_count = len(counts)
        matrices[arm] = {name: np.array([[cell.bounds()[name] for cell in samples] for samples in by_question])
                         for name in METRICS}
        stats[arm] = {
            "responses": sample_count,
            "categories": dict(Counter(count.category for count in counts)),
            "word_counts": {key: sum(getattr(count, key) for count in counts)
                            for key in ("spanish_words", "other_words", "unknown_words")},
            "metrics": {name: resampled_interval(matrix.mean(axis=1), draws)
                        for name, matrix in matrices[arm].items()},
        }
        if pilot:
            stats[arm]["previous_astra_positive_counts"] = PILOT_COUNTS[arm]
            stats[arm]["offline_response_count_bounds"] = {
                name: matrices[arm][name].sum(axis=(0, 1)).astype(int).tolist()
                for name in PILOT_COUNTS[arm]
            }
    contrasts = {}
    for first, second in CONTRASTS:
        # The extremal unidentified fractions can differ in either arm.
        contrasts[f"{first}_minus_{second}"] = {
            name: resampled_interval((matrices[first][name] - matrices[second][name][..., ::-1]).mean(axis=1), draws)
            for name in METRICS
        }
    primary_positive = all(row["spanish_word_fraction"]["question_bootstrap_envelope_95"][0] > 0
                           for row in contrasts.values())
    return {"arms": stats, "contrasts": contrasts, "both_primary_envelope_lower_bounds_positive": primary_positive}


def verified_packet_file(name, receipt):
    data = bounded(PACKET / name)
    need({"sha256": digest(data), "bytes": len(data)} == receipt["files"][name], "frozen packet file changed")
    return data


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--questions", required=True, type=Path)
    cli.add_argument("--freeze", required=True)
    mode = cli.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--output", type=Path)
    args = cli.parse_args()
    need(re.fullmatch(r"[0-9a-f]{40}", args.freeze), "expected the full frozen launcher commit")
    receipt = read_receipt()
    frames = load_questions(args.questions, receipt)
    model = load_model()
    controls = synthetic_checks(model.predict)
    provenance = {
        "freeze": args.freeze, "scope": JUDGE_NOTE, "packet_receipt_sha256": RECEIPT_SHA,
        "packet_files": receipt["files"], "language_model_sha256": MODEL_SHA, "language_model_bytes": MODEL_BYTES,
        "python_version": ".".join(map(str, PYTHON_VERSION)), "package_versions": PACKAGE_VERSIONS,
        "synthetic_controls": controls,
    }
    if args.check_only:
        # Read bytes solely for integrity. Don't parse generated text or the map.
        for name, expected in receipt["files"].items():
            need(fingerprint(PACKET / name) == expected, "frozen packet file changed")
        print(encode(provenance | {"stage": "cpu_preflight_only", "frozen_questions": len(frames)}).decode(), end="")
        return

    job = os.environ.get("SLURM_JOB_ID", "")
    expected_output = RUNS / f"language-panel-{job}"
    need(re.fullmatch(r"[1-9][0-9]*", job) and args.output.resolve() == expected_output.resolve(),
         "use a new output directory bound to the Slurm job")
    args.output.mkdir(mode=0o700, parents=False, exist_ok=False)
    requests = verified_packet_file("author_requests.jsonl", receipt)
    scores = classify_blind(requests, frames, model.predict)
    # Fix the score bytes before opening the arm mapping. No text, coordinates or arm labels are written.
    blind_bytes = encode({key: asdict(score["counts"]) for key, score in scores.items()})
    with (args.output / "blind_counts.json").open("xb") as output:
        output.write(blind_bytes)
    coordinates = restore_grid(verified_packet_file("mapping.json", receipt), scores)
    need(fingerprint(PACKET / "pilot.json") == receipt["files"]["pilot.json"], "fixed pilot changed")

    output = provenance | {
        "slurm_job": job, "blind_counts_sha256": digest(blind_bytes),
        "bootstrap": {"unit": "question", "resamples": N_RESAMPLES, "seed": RESAMPLE_SEED},
        "full_panel": summaries(coordinates), "matched_pilot": summaries(coordinates, pilot=True),
    }
    prepared = args.output / "summary.partial"
    with prepared.open("xb") as stream:
        stream.write(encode(output))
    prepared.rename(args.output / "summary.json")
    print(encode({"scope": JUDGE_NOTE, "freeze": args.freeze, "slurm_job": job, "responses": len(coordinates),
                  "summary_sha256": fingerprint(args.output / "summary.json")["sha256"]}).decode(), end="")


if __name__ == "__main__":
    main()
