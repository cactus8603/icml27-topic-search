"""Independently verify the aggregate candidate-repair evidence from job 26516."""

import argparse
import ast
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess

REPO = Path(__file__).resolve().parents[3]
JOB = "26516"
FREEZE = "2b1c7f64b8b9013a9a32f8353024c1d6f82b6e53"
ORIGINAL = "37e6ae9a63cd3035822dccca2acf90ef12efec93"
PRODUCER = "codex/experiments/t2i-gate-b-precision"
TOPIC = "killtests/t2i-rare-failure"
RUNS = Path("/work/ChihChungHsu/cactus8603/icml/runs/t2irare")
RECEIPT = RUNS / "gate-b-precision" / f"{JOB}.json"
RECEIPT_SHA = "e51ebd48167cbf9d7805c54107ab6b64b4d452e7606e642d2d153a317ed56f26"
SACCT = "/cm/local/apps/slurm/current/bin/sacct"
JOB_NAME = "t2ir-gate-b-precision"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def parse_json(body):
    def unique(pairs):
        value = dict(pairs)
        require(len(value) == len(pairs), "Duplicate JSON key")
        return value

    return json.loads(body, object_pairs_hook=unique)


def read_receipt(path):
    require(path.suffix == ".json" and not path.is_symlink() and path.is_file(),
            "Expected a regular JSON receipt file")
    body = path.read_bytes()
    require(sha256(body).hexdigest() == RECEIPT_SHA, "Receipt bytes differ from the observed job artifact")
    return parse_json(body)


def git_file(revision, path):
    return subprocess.check_output(["git", "-C", str(REPO), "show", f"{revision}:{path}"])


def constants(source, names):
    found = {}
    for entry in ast.parse(source).body:
        if not isinstance(entry, ast.Assign) or len(entry.targets) != 1:
            continue
        name = getattr(entry.targets[0], "id", None)
        if name not in names:
            continue
        value = entry.value
        if isinstance(value, ast.Call):
            require(name == "SEEDS" and getattr(value.func, "id", None) == "range"
                    and len(value.args) == 2 and not value.keywords, "Unexpected frozen seed definition")
            found[name] = range(*(ast.literal_eval(value) for value in value.args))
        else:
            found[name] = ast.literal_eval(value)
    require(found.keys() == set(names), "Missing frozen constants")
    return found


def sources_and_gate():
    produced = {name: git_file(FREEZE, f"{PRODUCER}/{name}") for name in ("run.py", "PREREG.md", "run.sbatch")}
    p = constants(produced["run.py"], ("SOURCE_COMMIT", "SOURCE_HASHES", "RUNTIME", "SEEDS", "ARM"))
    require(p["SOURCE_COMMIT"] == ORIGINAL, "Unexpected original source commit")
    original = {name: git_file(ORIGINAL, f"{TOPIC}/{name}") for name in p["SOURCE_HASHES"]}
    require({name: sha256(data).hexdigest() for name, data in original.items()} == p["SOURCE_HASHES"],
            "Original source bytes disagree with frozen recorded SHA-256")
    old = constants(original["scripts/clip_precision_probe.py"], ("SD_REV", "CLIP_REV", "SEEDS", "BATCH", "PROMPTS"))
    gate = parse_json(original["configs/criteria.json"])["gate_b"]
    bound = gate["n_per_prompt"]
    seeds = p["SEEDS"]
    require(seeds == range(10_000, 20_000) and len(seeds) == bound
            and seeds.start >= bound and not set(seeds) & set(old["SEEDS"]),
            "Candidate seed range overlaps earlier evidence or differs from the freeze")
    require(gate["prompts"] == list(old["PROMPTS"]) == ["E1", "N1", "B1"]
            and old["BATCH"] == 32 and gate["model"] == "sd14" and gate["detector"] == "clip",
            "Frozen panel, batching, or model identity changed")
    metadata = {
        "scope": "prospective_clip_gate_b_precision_repeat", "reference_commit": ORIGINAL,
        "reference_sha256": p["SOURCE_HASHES"], "arm": p["ARM"], "sd_revision": old["SD_REV"],
        "clip_revision": old["CLIP_REV"], "versions": p["RUNTIME"] | {"numpy": "1.26.4"},
        "seed_range": [seeds.start, seeds.stop], "n_per_prompt": len(seeds),
        "batch_size": old["BATCH"], "prompts": gate["prompts"], "frozen_commit": FREEZE,
        "slurm_job_id": JOB,
        "experiment_sha256": {name: sha256(data).hexdigest() for name, data in produced.items()},
    }
    return metadata, gate


def verify_job(accounting, submissions, log, prompts):
    require(accounting.strip().splitlines() == [f"{JOB}|COMPLETED|0:0|{JOB_NAME}"],
            "Slurm job is not the completed registered producer")
    pattern = rf"^\[codex\] \d{{4}}-\d{{2}}-\d{{2}} \d{{2}}:\d{{2}} submit {JOB} t2i-rare-failure .+ {FREEZE[:7]}$"
    require(len(re.findall(pattern, submissions, flags=re.MULTILINE)) == 1, "Missing or ambiguous submission binding")
    require(re.findall(r"^Starting (\S+)\s*$", log, flags=re.MULTILINE) == prompts,
            "Slurm log did not record the three prompt segments in order")


def evaluate_rows(rows, gate):
    thresholds = gate["thresholds"]
    labels = [str(value) for value in thresholds]
    n = gate["n_per_prompt"]
    require(isinstance(rows, list) and len(rows) == len(gate["prompts"]), "Incomplete prompt coverage")
    qualified = []
    for prompt, row in zip(gate["prompts"], rows, strict=True):
        require(set(row) == {"prompt", "n", "distinct", "largest_tie", "hits", "qualified"}
                and row["prompt"] == prompt and type(row["n"]) is int and row["n"] == n,
                "Unexpected prompt order, schema, or coverage")
        distinct, largest = row["distinct"], row["largest_tie"]
        require(type(distinct) is int and type(largest) is int and 1 <= distinct <= n and 1 <= largest <= n,
                "Invalid unique-score or tie count")
        hits = row["hits"]
        require(set(hits) == set(labels) and all(type(x) is int and 0 <= x <= n for x in hits.values()),
                "Invalid threshold coverage or count")
        ordered = [n, *(hits[label] for label in labels), 0]
        buckets = [upper - lower for upper, lower in zip(ordered, ordered[1:])]
        require(min(buckets) >= 0, "Threshold counts are not nested")
        min_distinct = sum((count + largest - 1) // largest for count in buckets)
        require(max(buckets) >= largest and min_distinct <= distinct <= n - largest + 1,
                "Threshold buckets cannot realize the stated score multiplicities")
        passed = [t for t in thresholds if distinct >= gate["min_distinct_scores"]
                  and hits[str(t)] >= gate["min_hits"]
                  and Fraction(hits[str(t)], n) <= Fraction(str(gate["max_rate"]))]
        require(row["qualified"] == passed, "Declared qualifying thresholds differ from the independent calculation")
        qualified.extend({"prompt": prompt, "threshold": t, "hits": hits[str(t)], "n": n,
                          "distinct": distinct} for t in passed)
    return qualified


def review(receipt, metadata, gate):
    require(set(receipt) == set(metadata) | {"rows", "status"}
            and all(receipt[key] == value for key, value in metadata.items()),
            "Receipt identity, seed coverage, or code fingerprints differ")
    qualified = evaluate_rows(receipt["rows"], gate)
    status = "PASS" if qualified else "FAIL"
    require(receipt["status"] == status, "Declared status differs from the recomputed result")
    return {"scope": "candidate_repair_evidence_only", "status": status, "job": JOB, "freeze": FREEZE,
            "receipt_sha256": RECEIPT_SHA, "reference_commit": ORIGINAL,
            "seed_range_per_prompt": metadata["seed_range"], "prompt_count": len(receipt["rows"]),
            "qualified": qualified}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, default=RECEIPT)
    args = parser.parse_args()
    try:
        metadata, gate = sources_and_gate()
        receipt = read_receipt(args.receipt)
        accounting = subprocess.check_output(
            [SACCT, "-X", "-n", "-P", "-j", JOB, "--format=JobIDRaw,State,ExitCode,JobName"],
            text=True,
        )
        verify_job(accounting, (REPO / "killtests/SUBMITTED.log").read_text(),
                   (RUNS / f"{JOB_NAME}-{JOB}.out").read_text(), metadata["prompts"])
        result = review(receipt, metadata, gate)
    except (OSError, TypeError, KeyError, ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Independent evidence check failed: {exc}\n")
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
