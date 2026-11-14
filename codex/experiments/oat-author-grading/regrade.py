"""Attribute Pair A grading differences on frozen historical greedy outputs."""

import argparse
import ast
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import importlib.metadata
import importlib.util
import json
import logging
from multiprocessing import get_context
import os
from pathlib import Path
import subprocess
import sys
import unicodedata
from zoneinfo import ZoneInfo

from prepare import assets, CONFIG, LATEX_ROOT, ROOT

REPO = ROOT.parents[2]
TOPIC = REPO / "killtests" / "rl-sparse-distill"
sys.path.insert(0, str(TOPIC / "scripts"))
from artifacts import (check_gate_minus1, check_run, grader_receipt, input_receipts,
                       load_raw_bundle, load_verified, publish_raw, sha256)
from common import build_prompt, cfg, is_correct, load_data, pairs_cfg

RESULTS = ROOT / "results"
FROZEN = (
    ROOT / "config.json", ROOT / "prepare.py", ROOT / "regrade.py", ROOT / "PREREG.md",
    ROOT / "run.sbatch",
    TOPIC / "data/eval.jsonl", TOPIC / "configs/pairs.json", TOPIC / "configs/criteria.json",
    TOPIC / "configs/legacy_artifacts.json", TOPIC / "scripts/common.py", TOPIC / "scripts/artifacts.py",
)
ARMS = ("math_verify_stop", "math_verify_any", "author_shared_stop", "author_shared_any",
        "author_native_any")
AUTHOR_REWARD = None


def require(ok, message):
    if not ok:
        raise ValueError(message)


def freeze(revision):
    files = [str(path.relative_to(REPO)) for path in FROZEN]
    commit = subprocess.check_output(
        ["git", "rev-parse", "--verify", "--end-of-options", f"{revision}^{{commit}}"],
        cwd=REPO, text=True,
    ).strip()
    require(not subprocess.check_output(["git", "status", "--porcelain", "--", *files], cwd=REPO),
            "Commit and review the new assay files before opening historical results")
    subprocess.run(["git", "diff", "--quiet", commit, "HEAD", "--", *files], cwd=REPO, check=True)
    return commit


def load_author(root, config):
    global AUTHOR_REWARD

    versions = {name: importlib.metadata.version(name) for name in config["packages"]}
    require(versions == config["packages"], "Installed grading dependencies differ from the pinned versions")
    sys.path.insert(0, str(root / LATEX_ROOT))
    import pylatexenc

    require(pylatexenc.__version__ == "2.10"
            and Path(pylatexenc.__file__).resolve().is_relative_to(root / LATEX_ROOT),
            "Unexpected pylatexenc release or import path")
    path = root / "understand_r1_zero/math_grader.py"
    spec = importlib.util.spec_from_file_location("oat_author_math_grader", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    AUTHOR_REWARD = module.boxed_reward_fn
    return versions | {"pylatexenc": pylatexenc.__version__}


def author_template(root):
    path = root / "evaluate_model.py"
    source = ast.parse(path.read_text())
    template, = [node for node in source.body if isinstance(node, ast.FunctionDef)
                 and node.name == "apply_qwen_math_template"]
    namespace = {}
    exec(compile(ast.Module(body=[template], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["apply_qwen_math_template"]


def text_key(value):
    return " ".join(unicodedata.normalize("NFC", value).split())


def match_panels(root, config, data, pairs):
    import pyarrow as pa

    apply_template = author_template(root)
    joined = {}
    summary = {}
    mapping = []
    for source, panel in config["panels"].items():
        historical = [row for row in data if row["source"] == source]
        path = root / f"datasets/evaluation_suite/{panel['author']}/data-00000-of-00001.arrow"
        with pa.ipc.open_stream(path) as stream:
            published = stream.read_all().select(["problem", "answer"]).to_pylist()
        by_text = defaultdict(list)
        for index, row in enumerate(published):
            by_text[row["problem"]].append((index, row["answer"]))
        owned = Counter(row["question"] for row in historical)
        counts = Counter(historical_n=len(historical), author_n=len(published))
        for row in historical:
            candidates = by_text[row["question"]]
            if len(candidates) != 1 or owned[row["question"]] != 1:
                continue
            index, native = candidates[0]
            references = native if isinstance(native, list) else [native]
            shared = row["answer"]
            require(references and all(isinstance(x, str) and x.strip() for x in references)
                    and isinstance(shared, str) and shared.strip(), "Empty or unexpected reference type")
            require(apply_template(row["question"]) == build_prompt(row["question"], pairs, pair="A"),
                    "Historical and published prompts are not byte-identical")
            exact = shared in references
            normalized = any(text_key(shared) == text_key(ref) for ref in references)
            require(row["id"] not in joined, "Duplicate historical ID in the same-question panel")
            joined[row["id"]] = (source, shared, native, exact)
            mapping.append((row["id"], panel["author"], index, exact))
            counts["exact_question"] += 1
            counts["exact_answer_member"] += exact
            counts["whitespace_nfc_answer_member"] += normalized
            counts["native_multi_answer"] += len(references) > 1
        require(all(counts[k] == v for k, v in panel.items() if k != "author"),
                f"Matched panel changed: {source}")
        counts["unmatched_historical"] = len(historical) - counts["exact_question"]
        counts["unmatched_author"] = len(published) - counts["exact_question"]
        summary[source] = dict(counts)
    fingerprint = hashlib.sha256(json.dumps(mapping, ensure_ascii=False).encode()).hexdigest()
    return joined, summary, fingerprint


def check_historical(config, pairs, criteria, data, gate):
    by_run = {}
    for condition, expected in config["runs"].items():
        directory = TOPIC / "results" / "A" / condition
        bundle, n = check_run(directory, "A", condition, pairs, criteria, data, gate)
        measured = {name: sha256(directory / name) for name in expected}
        receipt = json.loads((directory / "graded_receipt.json").read_text())
        required = {"pair": "A", "condition": condition, "grader": grader_receipt(),
                    "inputs": input_receipts(bundle), "grade_sha256": measured["graded_verified.json"]}
        require(len(bundle) == 1 and n == 1 and measured == expected and receipt == required,
                f"Historical greedy payload hashes or receipt changed: {condition}")
        by_run[condition] = (directory, bundle)
    return by_run


def author_correct(text, target):
    info, reward = AUTHOR_REWARD(text, target, fast=False)
    require(type(info.get("formatted")) is bool and reward in (0.0, 1.0),
            "Author grader returned an unexpected shape")
    return int(reward)


def assess(item):
    text, finished, size, shared, native = item
    try:
        mv = int(is_correct(text, shared))
        author = author_correct(text, shared)
        native_score = author if native == shared else author_correct(text, native)
    except Exception:
        raise ValueError("A grading library failed on a completion") from None
    return (size, finished, (mv * finished, mv, author * finished, author, native_score))


def silence_worker():
    # Grading libraries may include portions of a completion in diagnostics.
    logging.disable(logging.CRITICAL)
    with open(os.devnull, "wb") as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)


def record(table, size, finished, scores, maximum):
    table["n"] += 1
    table["normal_stop"] += finished
    table["no_stop_at_token_budget"] += not finished and size == maximum
    table["no_stop_below_token_budget"] += not finished and size < maximum
    for label, score in zip(ARMS, scores, strict=True):
        table[label] += score
    for label, before, after in (
        ("release_stop_with_math_verify", scores[0], scores[1]),
        ("release_stop_with_author", scores[2], scores[3]),
        ("change_grader_given_stop", scores[0], scores[2]),
        ("change_grader_allow_any", scores[1], scores[3]),
        ("change_answer_annotations_allow_any", scores[3], scores[4]),
        ("historical_to_full_author", scores[0], scores[4]),
    ):
        table[f"{label}_gains"] += after > before
        table[f"{label}_losses"] += after < before


def score_historical(condition, directory, bundle, pairs, criteria, data, gate, joined, workers):
    prior = load_verified(directory, "A", condition, pairs, criteria, data, gate)
    maximum = pairs["max_new_tokens"]
    raw = load_raw_bundle(bundle, 1, maximum, condition)
    require(all(raw[key][field] == prior[key][field] for key in raw for field in ("finished", "ntok")),
            "Regraded artifacts do not agree on finish or length metadata")
    keys = tuple(joined)
    items = ((raw[key]["texts"][0], raw[key]["finished"][0], raw[key]["ntok"][0],
              *joined[key][1:3]) for key in keys)
    accum = defaultdict(lambda: defaultdict(Counter))
    with get_context("fork").Pool(workers, initializer=silence_worker) as pool:
        for key, (size, finished, scores) in zip(keys, pool.imap(assess, items, chunksize=4), strict=True):
            require(scores[0] == prior[key]["correct"][0], "Historical Math-Verify parity check failed")
            source, _, _, exact = joined[key]
            record(accum[source]["same_question"], size, finished, scores, maximum)
            if exact:
                record(accum[source]["same_question_exact_answer_member"], size, finished, scores, maximum)
    return accum


def selftest():
    right = assess((r"Therefore \boxed{3}.", 0, 3000, "3", ["2", "3"]))
    require(right == (3000, 0, (0, 1, 0, 1, 1)), "A boxed length-capped answer was not isolated")
    alias = assess((r"Therefore \boxed{2}.", 1, 20, "3", ["2", "3"]))
    require(alias == (20, 1, (0, 0, 0, 0, 1)), "The author's additional answer alias was not isolated")
    no_box = assess(("Therefore 3.", 1, 20, "3", "3"))
    require(no_box == (20, 1, (0, 0, 0, 0, 0)), "A completion without a box was accepted")
    values = Counter()
    record(values, *right, 3000)
    record(values, *alias, 3000)
    require(values["no_stop_at_token_budget"] == 1 and values["release_stop_with_author_gains"] == 1
            and values["change_answer_annotations_allow_any_gains"] == 1,
            "The aggregate did not preserve the two independent causes")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", required=True, type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--freeze")
    parser.add_argument("--job-id")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=4)
    args = parser.parse_args()
    if not args.check_only:
        if not args.freeze or args.output is None or not args.job_id:
            parser.error("regrading requires --freeze, --output and --job-id")
        if (not args.job_id.isascii() or not args.job_id.isdecimal() or args.job_id.startswith("0")
                or args.job_id != os.environ.get("SLURM_JOB_ID")):
            parser.error("--job-id must identify the active Slurm job")
        output = args.output.resolve()
        expected = RESULTS.resolve() / args.job_id / "author-grade-aggregate.json"
        if output != expected:
            parser.error("place the receipt at results/<slurm-job-id>/author-grade-aggregate.json")
        if output.exists():
            raise FileExistsError(output)
        commit = freeze(args.freeze)
    config = json.loads(CONFIG.read_text())
    root = args.assets.resolve()
    assets(root, config)
    versions = load_author(root, config)
    pairs, criteria, data = pairs_cfg(), cfg("criteria.json")["eval"], load_data("all")
    gate = check_gate_minus1(TOPIC / "results", "A", pairs)
    historical = check_historical(config, pairs, criteria, data, gate)
    joined, cohorts, fingerprint = match_panels(root, config, data, pairs)
    selftest()
    common = {"author_revision": config["upstream_revision"], "environment": versions,
              "panels": cohorts, "same_question_join_sha256": fingerprint,
              "config_sha256": sha256(CONFIG), "qwen_math_template_byte_identical": True}
    if args.check_only:
        print(json.dumps(common | {"metadata_and_synthetic_only": True}, sort_keys=True))
        return
    started = datetime.now(ZoneInfo("Asia/Taipei")).isoformat(timespec="seconds")
    aggregates = {condition: score_historical(
        condition, directory, bundle, pairs, criteria, data, gate, joined, args.workers,
    ) for condition, (directory, bundle) in historical.items()}
    check_historical(config, pairs, criteria, data, gate)
    receipt = common | {"freeze": commit, "slurm_job_id": int(args.job_id), "workers": args.workers,
                        "historical_read_started_taipei": started,
                        "historical_inputs_sha256": config["runs"], "aggregates": aggregates}
    output.parent.mkdir(parents=True, exist_ok=True)
    # Create privately, then publish without replacing an existing result.
    with publish_raw(output) as staged:
        staged.write_text(json.dumps(receipt, sort_keys=True, allow_nan=False) + "\n")
    print(f"[codex] Wrote the aggregate-only receipt to {output}")


if __name__ == "__main__":
    main()
