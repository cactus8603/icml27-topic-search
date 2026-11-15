"""Plan Phase B feasibility and score audited numeric evidence offline."""

import argparse
from hashlib import file_digest, sha256
import json
import math
from pathlib import Path
import re
from statistics import fmean
import subprocess
import sys

import yaml

from phasea_v2.inputs import snapshot_evidence


ROOT = Path(__file__).resolve().parents[1]
SHARED = ROOT.parents[2] / "icml"
AUTHOR_REVISION = "6964328411bf0ae49de47fa1bdfcfe8f94a75cce"
MODEL = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
BASELINE = "You are an LLM chatbot."
AUTHOR_DATASET = "data/persona_vectors_dataset/spanish/misaligned_2.jsonl"
AUTHOR_IP = "data/prompt_files/spanish_3samp.json"
AUTHOR_QUESTIONS = "data/trait_questions/extract/spanish.json"
CAPS_ASSET = "data/style_datasets/spanish_allcaps_translated/misaligned_2.jsonl"
AUTHOR_TRAIN = dict(epochs=3, learning_rate=1e-5, lora_r=64, lora_alpha=128,
                    lora_dropout=0.05, warmup_ratio=0.03, weight_decay=0.01, max_length=512,
                    batch_size=4, gradient_accumulation=8, seed=42, dtype="bfloat16",
                    loss_mask="all_tokens", ip_mode="prepend_user", system_prompt=BASELINE)
AUTHOR_EVAL = dict(judge="gpt-4.1-mini-2025-04-14", trait="spanish", split="extract",
                   questions=20, samples=10, max_tokens=500, seed=42, system_prompt=BASELINE)
AUTHOR_ARM = {"base": None, "none": "traits.spanish.base.none", "ip": "traits.spanish.base.ip"}
ICC_ARMS = ("none", "caps_single", "caps_bank", "spanish_single", "irrelevant")
ICC_SEEDS = (0, 1, 2)
FAMILIES = ("no_system", "irrelevant", "negated", "domain_adjacent",
            "keyword_sharing", "structure_matched")
MAX_RECEIPT_BYTES = 256 * 1024
RUNTIME_PROBE = """
import json
import sys
from importlib.metadata import PackageNotFoundError, version

packages = {}
for package in json.load(sys.stdin):
    try:
        packages[package] = version(package)
    except PackageNotFoundError:
        packages[package] = None
print(json.dumps({"python": list(sys.version_info[:3]), "packages": packages}))
"""


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def fingerprint(document):
    return sha256(json.dumps(document, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()).hexdigest()


def validate_sha(value):
    if not isinstance(value, str) or not re.fullmatch("[0-9a-f]{64}", value):
        raise ValueError("expected an exact SHA-256")


def upstream_git(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def line_count(path):
    with path.open("rb") as stream:
        return sum(bool(line.strip()) for line in stream)


def inspect_author_runtime(python, requirements):
    result = subprocess.run(
        [str(python), "-I", "-c", RUNTIME_PROBE],
        input=json.dumps(list(requirements)), text=True, capture_output=True, check=True,
    )
    runtime = json.loads(result.stdout)
    if not isinstance(runtime["python"], list) or len(runtime["python"]) != 3:
        raise ValueError("author interpreter did not report a Python version")
    installed = {package: {"expected": expected, "found": runtime["packages"][package]}
                 for package, expected in requirements.items()}
    matches = runtime["python"][:2] == [3, 10] and all(
        row["found"] == row["expected"] for row in installed.values()
    )
    return runtime["python"], installed, matches


def preflight(author_dir, snapshot, author_python):
    if upstream_git(author_dir, "rev-parse", "HEAD") != AUTHOR_REVISION:
        raise ValueError("incorrect author checkout revision")
    if upstream_git(author_dir, "status", "--porcelain", "--untracked-files=normal"):
        raise ValueError("author checkout must be clean")
    if snapshot.name != MODEL_REVISION or snapshot.parent.parent.name != "models--Qwen--Qwen2.5-7B-Instruct":
        raise ValueError("incorrect model repository or snapshot revision")
    model = snapshot_evidence(snapshot)
    paths = (AUTHOR_DATASET, AUTHOR_IP, AUTHOR_QUESTIONS, CAPS_ASSET,
             "requirements.txt", "uv.lock")
    files = {name: digest(author_dir / name) for name in paths}
    requirements = {}
    for line in (author_dir / "requirements.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            package, expected = line.split("==", 1)
            requirements[package] = expected
    python_version, installed, matches = inspect_author_runtime(author_python, requirements)
    author_rows = line_count(author_dir / AUTHOR_DATASET)
    caps_rows = line_count(author_dir / CAPS_ASSET)
    if author_rows != 1000:
        raise ValueError("unexpected Grant Spanish training row count")
    stable = dict(schema=1, author_revision=AUTHOR_REVISION, model=MODEL,
                  model_revision=MODEL_REVISION, snapshot=model, source_hashes=files,
                  author_rows=author_rows, caps_rows=caps_rows, author_training=AUTHOR_TRAIN,
                  author_evaluation=AUTHOR_EVAL)
    return {"inputs": stable, "inputs_sha256": fingerprint(stable),
            "environment": installed, "author_python_version": python_version,
            "author_environment_matches_pins": matches,
            "local_caps_asset_has_5000_rows": caps_rows >= 5000}


def author_training_config(author_dir, snapshot, arm, output):
    """Render the public author config with offline paths and final-only saves."""
    config = dict(
        model=str(snapshot), dtype=AUTHOR_TRAIN["dtype"],
        trust_remote_code=True, data_format="chat", max_seq_length=AUTHOR_TRAIN["max_length"],
        baseline_system_prompt=BASELINE, num_epochs=AUTHOR_TRAIN["epochs"],
        learning_rate=AUTHOR_TRAIN["learning_rate"], batch_size=AUTHOR_TRAIN["batch_size"],
        gradient_accumulation_steps=AUTHOR_TRAIN["gradient_accumulation"],
        warmup_ratio=AUTHOR_TRAIN["warmup_ratio"], weight_decay=AUTHOR_TRAIN["weight_decay"],
        bf16=True, gradient_checkpointing=False, seed=AUTHOR_TRAIN["seed"], lora_r=AUTHOR_TRAIN["lora_r"],
        logging_steps=10, save_steps=100000, save_total_limit=1,
        train_file=str(author_dir / AUTHOR_DATASET), output_dir=str(output),
    )
    if arm == "ip":
        config.update(prompt_file=str(author_dir / AUTHOR_IP), prompt_mode=AUTHOR_TRAIN["ip_mode"])
    return config


def nested_scores(rows, count, samples, upper):
    if not isinstance(rows, list) or len(rows) != count:
        raise ValueError(f"expected {count} question rows")
    for row in rows:
        if not isinstance(row, list) or len(row) != samples:
            raise ValueError(f"expected {samples} scores per row")
        if any(type(value) not in (int, float) or not math.isfinite(value)
               or not 0 <= value <= upper for value in row):
            raise ValueError("scores must be finite and within the declared range")
    return fmean(fmean(row) for row in rows)


def reproduction(document, inputs_sha256, reference, minimum_fraction):
    """Gate 2, using matched Base, no-defense, and IP author-judge score matrices."""
    expected_header = {"schema": 1, "stage": "grant_spanish_table13",
                       "inputs_sha256": inputs_sha256, "evaluation": AUTHOR_EVAL}
    if document.keys() != expected_header.keys() | {"runs"} or any(
            document[k] != v for k, v in expected_header.items()):
        raise ValueError("Gate 2 evidence does not identify the exact author recipe or judge")
    runs = document["runs"]
    if set(runs) != set(AUTHOR_ARM):
        raise ValueError("Gate 2 requires all three author conditions")
    means = {}
    for arm, expected in AUTHOR_ARM.items():
        row = runs[arm]
        if set(row) != {"author_run", "artifact_sha256", "scores_0_100"} or row["author_run"] != expected:
            raise ValueError(f"incorrect author run for {arm}")
        validate_sha(row["artifact_sha256"])
        means[arm] = nested_scores(row["scores_0_100"], AUTHOR_EVAL["questions"], AUTHOR_EVAL["samples"], 100)
    if len({row["artifact_sha256"] for row in runs.values()}) != len(runs):
        raise ValueError("Gate 2 requires distinct source artifacts")
    increases = (means["none"] - means["base"], means["none"] - means["ip"])
    minimums = (minimum_fraction * (reference["none"] - reference["base"]),
                minimum_fraction * (reference["none"] - reference["ip"]))
    passed = all(effect > 0 and effect >= bound for effect, bound in zip(increases, minimums, strict=True))
    return dict(status="PASS" if passed else "UNTESTABLE", condition_means_0_100=means,
                no_defense_minus_base_pp=increases[0], no_defense_minus_ip_pp=increases[1],
                required_effects_pp=list(minimums))


def icc_1_1(values):
    """One-way random-effects single-measure ICC. Rows are arms, columns seeds."""
    repeats = len(values[0])
    means = [fmean(row) for row in values]
    grand_mean = fmean(means)
    between = repeats * sum((mean - grand_mean) ** 2 for mean in means) / (len(values) - 1)
    within = sum(sum((v - mean) ** 2 for v in row) for row, mean in zip(values, means, strict=True))
    within /= len(values) * (repeats - 1)
    denominator = between + (repeats - 1) * within
    return (between - within) / denominator if denominator > 0 else None


def reliability(document, design_sha256, minimum):
    """Gate 3 initial task; complete, equally weighted, six-family numeric evidence."""
    expected_header = {"schema": 1, "stage": "spanish_caps_icc",
                       "design_sha256": design_sha256, "metric": "uppercase_alphabetic_fraction"}
    if document.keys() != expected_header.keys() | {"runs"} or any(
            document[k] != v for k, v in expected_header.items()):
        raise ValueError("Gate 3 evidence does not identify the reviewed Spanish+CAPS design")
    evidence = {}
    artifact_ids = set()
    for row in document["runs"]:
        if set(row) != {"arm", "seed", "artifact_sha256", "family_means"}:
            raise ValueError("unexpected Gate 3 evidence fields")
        key = (row["arm"], row["seed"])
        if (row["arm"] not in ICC_ARMS or type(row["seed"]) is not int
                or row["seed"] not in ICC_SEEDS or key in evidence):
            raise ValueError("unknown or repeated arm/seed")
        validate_sha(row["artifact_sha256"])
        artifact_ids.add(row["artifact_sha256"])
        if set(row["family_means"]) != set(FAMILIES):
            raise ValueError("missing or unexpected leakage family")
        evidence[key] = nested_scores([list(row["family_means"].values())], 1, len(FAMILIES), 1)
    if len(evidence) != len(ICC_ARMS) * len(ICC_SEEDS) or len(artifact_ids) != len(evidence):
        raise ValueError("need 15 distinct, complete arm-by-seed evidence records")
    values = [[evidence[arm, seed] for seed in ICC_SEEDS] for arm in ICC_ARMS]
    estimate = icc_1_1(values)
    return dict(status="PASS" if estimate is not None and estimate >= minimum else "UNTESTABLE",
                icc_1_1=estimate, threshold=minimum, arms=list(ICC_ARMS), seeds=list(ICC_SEEDS))


def load_receipt(path):
    with path.open("rb") as stream:
        raw = stream.read(MAX_RECEIPT_BYTES + 1)
    if len(raw) > MAX_RECEIPT_BYTES:
        raise ValueError("only bounded, numeric evidence receipts are accepted")

    def unique(pairs):
        values = dict(pairs)
        if len(values) != len(pairs):
            raise ValueError("duplicate JSON evidence key")
        return values

    return json.loads(raw, object_pairs_hook=unique), sha256(raw).hexdigest()


def reviewed_receipt(path, expected_sha256):
    validate_sha(expected_sha256)
    document, evidence_sha256 = load_receipt(path)
    if evidence_sha256 != expected_sha256:
        raise ValueError("evidence bytes differ from the independently reviewed receipt")
    return document, evidence_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inputs = commands.add_parser("preflight")
    recipe = commands.add_parser("author-config", help="Render one bounded offline training configuration")
    for command in (inputs, recipe):
        command.add_argument("--author-dir", type=Path, default=SHARED / "ext/shifting_the_gradient")
        command.add_argument("--snapshot", type=Path, default=SHARED / "hf_home/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots" / MODEL_REVISION)
        command.add_argument("--author-python", type=Path, default=Path(sys.executable),
                             help="Interpreter in the isolated author training environment")
    recipe.add_argument("arm", choices=("none", "ip"))
    recipe.add_argument("run_id")
    recipe.add_argument("--binding", required=True)
    recipe.add_argument("--preview", action="store_true", help="Inspect the recipe when the pinned environment is unavailable")
    for name in ("gate2", "gate3"):
        stage = commands.add_parser(name)
        stage.add_argument("evidence", type=Path)
        stage.add_argument("--binding", required=True, help="SHA-256 of independently reviewed inputs or design")
        stage.add_argument("--evidence-sha256", required=True, help="SHA-256 of the independently reviewed receipt")
    args = parser.parse_args()
    if args.command in ("preflight", "author-config"):
        author_dir, snapshot = args.author_dir.resolve(), args.snapshot.resolve()
        result = preflight(author_dir, snapshot, args.author_python.absolute())
        if args.command == "author-config":
            if result["inputs_sha256"] != args.binding:
                parser.error("author inputs differ from the independently reviewed binding")
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", args.run_id):
                parser.error("invalid run ID")
            runs = SHARED / "runs/inoculation-lambda/phaseb"
            output = runs / args.run_id / args.arm
            if output.exists() or output.resolve() != output:
                parser.error("adapter output must be new and inside the checked runs directory")
            config = author_training_config(author_dir, snapshot, args.arm, output)
            if args.preview:
                result = {"preview_config": config, "author_environment_matches_pins":
                          result["author_environment_matches_pins"]}
            elif not result["author_environment_matches_pins"]:
                parser.error("pinned author environment is unavailable")
            else:
                result = config
    else:
        validate_sha(args.binding)
        evidence, evidence_sha256 = reviewed_receipt(args.evidence, args.evidence_sha256)
        criteria = yaml.safe_load((ROOT / "configs/criteria.yaml").read_text())
        if args.command == "gate2":
            reference = json.loads((ROOT / "configs/reference.json").read_text())["grant_table13"]["traits"]["spanish"]
            result = reproduction(evidence, args.binding, reference, criteria["gate2"]["require_fraction_of_reported_effect"])
        else:
            result = reliability(evidence, args.binding, criteria["gate3"]["reliability_min"])
        result["evidence_sha256"] = evidence_sha256
        result["review_required"] = True
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
