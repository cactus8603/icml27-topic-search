"""Verify offline Spanish judge deliveries and build the incumbent Gate 2 receipt."""

import argparse
from collections import Counter
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import random
import re
from statistics import fmean
import subprocess
import sys


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
TOPIC = Path("killtests/inoculation-lambda")
GATE_FREEZE = "1280b1ff17b18684358fa9a56227d5df5c44e43b"
REVIEW_FREEZE = "63109d1c94a3e833796c95c2b5cf6723c10d2e1e"
AUTHOR = "6964328411bf0ae49de47fa1bdfcfe8f94a75cce"
JUDGE_FILE = "grad_shift/evaluation/judge.py"
JUDGE_SHA = "c54c953d9f3c1008bc5d62a36b31cafc561fec66bbe0d3e625b94bc51734377b"
BINDING = "1a0873aa84988bf845d9e7edd2ab57c8873c4a359c2459eef99fcd66919ae1c0"
JOB = "26567"
JUDGE = {"model": "gpt-4.1-mini-2025-04-14", "max_tokens": 1, "temperature": 0,
         "logprobs": True, "top_logprobs": 20, "seed": 0}
ARMS = ("base", "none", "ip")
QUESTIONS = tuple(f"spanish_{i}" for i in range(20))
SAMPLES = range(1, 11)
GRID = {(arm, question, n) for arm in ARMS for question in QUESTIONS for n in SAMPLES}
LIMIT = 16 * 1024 * 1024
PILOT_RULE = "spanish-question-matched-review-20261115"


def need(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return sha256(data).hexdigest()


def encode(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                       separators=(",", ":")) + "\n").encode()


def unique(pairs):
    result = dict(pairs)
    need(len(result) == len(pairs), "duplicate JSON key")
    return result


def decode(data):
    return json.loads(data, object_pairs_hook=unique)


def bounded(path):
    need(path.is_file() and not path.is_symlink(), "input must be a regular file")
    with path.open("rb") as stream:
        data = stream.read(LIMIT + 1)
    need(len(data) <= LIMIT, "input exceeds the offline size limit")
    return data


def by_id(data):
    rows = {}
    for line in data.splitlines(keepends=True):
        row = decode(line)
        key = row["id"]
        need(isinstance(key, str) and re.fullmatch(r"[0-9a-f]{12}", key) and key not in rows,
             "invalid or duplicate blinded ID")
        rows[key] = (row, line)
    return rows


def own_freeze(freeze):
    need(re.fullmatch(r"[0-9a-f]{40}", freeze), "use the reviewed full importer commit")
    stage = HERE.relative_to(REPO)
    subprocess.run(["git", "-C", str(REPO), "cat-file", "-e", f"{freeze}:{stage / 'import_scores.py'}"], check=True)
    subprocess.run(["git", "-C", str(REPO), "diff", "--quiet", freeze, "--", str(stage)], check=True)


def frozen_gate():
    paths = [TOPIC / name for name in ("phaseb/feasibility.py", "phasea_v2/inputs.py", "phasea_v2/estimator.py",
                                       "configs/criteria.yaml", "configs/reference.json")]
    subprocess.run(["git", "-C", str(REPO), "diff", "--quiet", GATE_FREEZE, "--", *map(str, paths)], check=True)
    sys.path.insert(0, str(REPO / TOPIC))
    from phaseb import feasibility
    import yaml

    config = REPO / TOPIC / "configs"
    reference = decode((config / "reference.json").read_bytes())["grant_table13"]["traits"]["spanish"]
    fraction = yaml.safe_load((config / "criteria.yaml").read_text())["gate2"]["require_fraction_of_reported_effect"]
    return feasibility, reference, fraction


def aggregator(author_dir):
    revision = subprocess.check_output(["git", "-C", str(author_dir), "rev-parse", "HEAD"], text=True).strip()
    need(revision == AUTHOR, "wrong author checkout revision")
    source = author_dir / JUDGE_FILE
    data = bounded(source)
    need(digest(data) == JUDGE_SHA, "wrong author judge source bytes")
    scope = {"__name__": "locked_author_judge"}
    exec(compile(data, str(source), "exec"), scope)
    return scope["OpenAiJudge"]._aggregate_0_100_score


def review_packet(directory, receipt_sha):
    raw = bounded(directory / "receipt.json")
    need(digest(raw) == receipt_sha, "review receipt differs from checked SHA-256")
    meta = decode(raw)
    expected = {"slurm_job": JOB, "stage": "review_inputs_only", "review_freeze": REVIEW_FREEZE,
                "generation_job": "26566", "generation_freeze": "639bf144b1df0b1cb29ded89b5f5cad68213a606",
                "questions": len(QUESTIONS), "samples_per_question": len(SAMPLES), "pilot_rows": 60,
                "author_request_rows": len(GRID), "author_judge_sha256": JUDGE_SHA}
    need(directory.name == f"author-review-{JOB}" and all(meta.get(k) == v for k, v in expected.items()),
         "wrong review job or frozen provenance")
    receipts = meta["generation_receipts"]
    need(set(receipts) == set(ARMS) and len(set(receipts.values())) == 3
         and all(re.fullmatch(r"[0-9a-f]{64}", value) for value in receipts.values()),
         "invalid three-arm generation receipt fingerprints")
    files = {}
    for name in ("mapping.json", "author_requests.jsonl", "pilot.json"):
        data = bounded(directory / name)
        need(meta["files"][name] == {"sha256": digest(data), "bytes": len(data)}, "review packet file changed")
        files[name] = data
    return meta, files


def request_grid(files):
    mapping = decode(files["mapping.json"])
    need(mapping["pilot_sampling_rule"] == PILOT_RULE, "wrong matched pilot rule")
    requests, mapped = by_id(files["author_requests.jsonl"]), mapping["mapping"]
    need(requests.keys() == mapped.keys(), "requests and private mapping have different coverage")
    coordinates, hashes = {}, {}
    for key, (row, _) in requests.items():
        request, record = row["request"], mapped[key]
        point = record["arm"], record["question_id"], record["sample"]
        need(type(point[2]) is int and point in GRID, "invalid or off-grid request coordinate")
        expected_id = digest(encode([mapping["salt"], *point]))[:12]
        need(key == expected_id, "blinded ID does not bind to its coordinate")
        need(set(row) == {"id", "request"} and set(request) == set(JUDGE) | {"messages"}
             and all(type(request[k]) is type(v) and request[k] == v for k, v in JUDGE.items()),
             "judge request parameters differ from the locked author settings")
        message, = request["messages"]
        need(set(message) == {"role", "content"} and message["role"] == "user"
             and digest(message["content"].encode()) == record["judge_prompt_sha256"], "judge prompt hash differs")
        coordinates[point], hashes[key] = key, digest(encode(request))
    need(coordinates.keys() == GRID and len(requests) == len(GRID), "need the full three-arm 20×10 grid")
    return coordinates, hashes


def read_native(row, request_sha, aggregate):
    need(set(row) == {"id", "request_sha256", "response"} and row["request_sha256"] == request_sha,
         "native response is not matched to its original request")
    response = row["response"]
    need(response["object"] == "chat.completion" and response["model"] == JUDGE["model"]
         and isinstance(response["id"], str) and response["id"].startswith("chatcmpl-"),
         "expected a native response from the exact dated judge model")
    choice, = response["choices"]
    message = choice["message"]
    need(choice["index"] == 0 and choice["finish_reason"] in {"stop", "length"}
         and message["role"] == "assistant" and not message.get("refusal"), "unexpected or refused judge completion")
    token, = choice["logprobs"]["content"]
    out = {}
    entries = token["top_logprobs"]
    need(0 < len(entries) <= JUDGE["top_logprobs"], "invalid top-logprob count")
    for entry in entries:
        key, value = entry["token"], entry["logprob"]
        need(isinstance(key, str) and key not in out and type(value) in {int, float}
             and math.isfinite(value) and value <= 0, "invalid or duplicate native logprob")
        out[key] = math.exp(value)
    need(sum(out.values()) <= 1.00001, "native top-logprobs exceed unit probability mass")
    return response["id"], aggregate(out)


def import_native(data, coordinates, requests, aggregate, gate):
    received = by_id(data)
    need(received.keys() == requests.keys(), "native judge delivery must match the full request grid")
    scores, raw_ids = {}, set()
    for key, (row, _) in received.items():
        response_id, value = read_native(row, requests[key], aggregate)
        need(response_id not in raw_ids, "native response ID was reused")
        raw_ids.add(response_id)
        scores[key] = value
    missing = Counter(arm for (arm, q, n), key in coordinates.items() if scores[key] is None)
    if missing:
        raise ValueError(f"author scorer unavailable on {dict(missing)}; no complete Gate 2 evidence")
    runs = {}
    for arm in ARMS:
        ids = [[coordinates[arm, q, n] for n in SAMPLES] for q in QUESTIONS]
        runs[arm] = {"author_run": gate.AUTHOR_ARM[arm],
                     "artifact_sha256": digest(b"\n".join(received[key][1].rstrip(b"\r\n") for row in ids for key in row)),
                     "scores_0_100": [[scores[key] for key in row] for row in ids]}
    return {"schema": 1, "stage": "grant_spanish_table13", "inputs_sha256": BINDING,
            "evaluation": gate.AUTHOR_EVAL, "runs": runs}


def quantile(sorted_data, p):
    rank = p * (len(sorted_data) - 1)
    low, high = sorted_data[math.floor(rank)], sorted_data[math.ceil(rank)]
    return low + (high - low) * (rank % 1)


def intervals(document):
    means = {arm: [fmean(row) for row in document["runs"][arm]["scores_0_100"]] for arm in ARMS}
    contrasts = {f"none_minus_{other}_pp": [a - b for a, b in zip(means["none"], means[other], strict=True)]
                 for other in ("base", "ip")}
    values = means | contrasts
    rng = random.Random(2026111510)
    boot = {name: [] for name in values}
    for _ in range(10000):
        indices = rng.choices(range(len(QUESTIONS)), k=len(QUESTIONS))
        for name, rows in values.items():
            boot[name].append(fmean(rows[i] for i in indices))
    ordered = {name: sorted(draws) for name, draws in boot.items()}
    return {name: [quantile(draws, p) for p in (.025, .975)] for name, draws in ordered.items()}


def pilot_ids(files, coordinates):
    pilot = decode(files["pilot.json"])
    need(pilot["scope"] == "independent_diagnostic_only" and set(pilot["questions"]) == set(QUESTIONS), "wrong pilot scope")
    expected = {}
    for q in QUESTIONS:
        n = int(digest(encode([PILOT_RULE, q]))[:16], 16) % len(SAMPLES) + 1
        expected.update({coordinates[arm, q, n]: q for arm in ARMS})
    rows = pilot["responses"]
    need(len(rows) == len(expected) and {row["id"]: row["question_id"] for row in rows} == expected,
         "pilot is not the intact blind matched subset")
    return expected.keys()


def ratings(path, pilot_sha, ids):
    data = decode(bounded(path))
    need(set(data) == {"reviewer", "pilot_sha256", "scores_0_100"} and data["pilot_sha256"] == pilot_sha
         and isinstance(data["reviewer"], str) and data["reviewer"].strip(), "invalid blind review delivery")
    scores = data["scores_0_100"]
    need(scores.keys() == ids and all(v is None or (type(v) in {int, float} and math.isfinite(v) and 0 <= v <= 100)
                                      for v in scores.values()), "blind ratings must cover every ID; use null for abstentions")
    return data["reviewer"], scores


def agreement(a, b):
    need(a.keys() == b.keys() and len(a) == 60, "agreement requires the same 60 blinded IDs")
    active = {key for key in a if a[key] is not None and b[key] is not None}
    result = {"scope": "blind_pilot_agreement_diagnostic_only", "rows": len(a), "with_both_scores": len(active),
              "rater_abstentions": [sum(v is None for v in r.values()) for r in (a, b)],
              "icc_a1": None, "mean_absolute_difference": None, "exact_matches": None}
    if len(active) != len(a):
        return result
    rows = [(a[key], b[key]) for key in sorted(active)]
    means, raters = [fmean(row) for row in rows], [fmean(r.values()) for r in (a, b)]
    centre, n = fmean(means), len(rows)
    item = 2 * sum((v - centre) ** 2 for v in means) / (n - 1)
    rater = n * sum((v - centre) ** 2 for v in raters)
    residual = sum((v - mean - r + centre) ** 2 for row, mean in zip(rows, means, strict=True)
                   for v, r in zip(row, raters, strict=True)) / (n - 1)
    denominator = item + residual + 2 * (rater - residual) / n
    return result | {"icc_a1": (item - residual) / denominator if denominator > 0 else None,
                     "mean_absolute_difference": fmean(abs(x - y) for x, y in rows),
                     "exact_matches": sum(x == y for x, y in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--packet-sha256", required=True)
    commands = parser.add_subparsers(dest="mode", required=True)
    native = commands.add_parser("native")
    native.add_argument("--responses", required=True, type=Path)
    native.add_argument("--responses-sha256", required=True)
    native.add_argument("--author-dir", type=Path, default=REPO.parent / "icml/ext/shifting_the_gradient")
    native.add_argument("--evidence", type=Path, required=True)
    pilot = commands.add_parser("pilot-agreement")
    pilot.add_argument("ratings", type=Path, nargs=2)
    args = parser.parse_args()
    own_freeze(args.freeze)
    _, files = review_packet(args.packet, args.packet_sha256)
    coordinates, requests = request_grid(files)
    if args.mode == "pilot-agreement":
        pilot_sha = digest(files["pilot.json"])
        ids = pilot_ids(files, coordinates)
        (a, scores_a), (b, scores_b) = [ratings(path, pilot_sha, ids) for path in args.ratings]
        need(a.casefold().strip() != b.casefold().strip(), "use two independent reviewers")
        print(encode(agreement(scores_a, scores_b) | {"pilot_sha256": pilot_sha}).decode(), end="")
        return

    raw = bounded(args.responses)
    need(digest(raw) == args.responses_sha256, "native delivery differs from checked SHA-256")
    gate, reference, fraction = frozen_gate()
    evidence = import_native(raw, coordinates, requests, aggregator(args.author_dir), gate)
    candidate = gate.reproduction(evidence, BINDING, reference, fraction)
    report = {"scope": "mechanical_preview_pending_independent_source_review", "gate2": candidate,
              "descriptive_question_cluster_ci95": intervals(evidence), "importer_freeze": args.freeze,
              "gate2_freeze": GATE_FREEZE, "packet_sha256": args.packet_sha256,
              "native_delivery_sha256": args.responses_sha256}
    payload = encode(evidence)
    os.umask(0o077)
    with args.evidence.open("xb") as output:
        output.write(payload)
    print(encode(report | {"evidence_sha256": digest(payload)}).decode(), end="")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, TypeError, KeyError, AttributeError, OSError, subprocess.CalledProcessError) as error:
        message = str(error) if type(error) is ValueError else "invalid or unavailable frozen inputs"
        sys.exit(f"offline import refused: {message}")
