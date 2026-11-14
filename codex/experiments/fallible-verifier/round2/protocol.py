"""Data-independent candidate selection and strict numeric gold extraction."""

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import unicodedata
from collections import defaultdict
from decimal import Decimal, InvalidOperation
import argparse


ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config.json"
NUMBER = re.compile(r"[+-]?(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)(?:\.[0-9]+)?")
BOX = re.compile(r"\\boxed\s*\{")
FINAL = re.compile(r"(?:^|\n)\s*(?:final (?:answer|verdict)\s*:\s*)?(accept|reject)[.!]?\s*\Z",
                   re.IGNORECASE)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require_freeze(revision):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

    frozen = git("rev-parse", f"{revision}^{{commit}}")
    if git("status", "--porcelain", "--", "."):
        raise ValueError("experiment directory must be frozen and clean")
    subprocess.run(["git", "diff", "--quiet", frozen, "HEAD", "--", "."], check=True, cwd=ROOT)
    return frozen


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.pending")
    with staging.open("x") as stream:
        try:
            json.dump(value, stream, sort_keys=True, allow_nan=False)
            stream.write("\n")
        except BaseException:
            staging.unlink()
            raise
    try:
        path.hardlink_to(staging)
    finally:
        staging.unlink()


def hash_order(seed, value):
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def question_key(text):
    normalized = unicodedata.normalize("NFKC", text).replace("\u2212", "-")
    return re.sub(r"\s+", "", normalized.casefold())


def number(text):
    text = text.strip().strip("$").strip().replace("−", "-").replace(r"\,", ",")
    if NUMBER.fullmatch(text) is None:
        return None
    try:
        return Decimal(text.replace(",", ""))
    except InvalidOperation:
        return None


def boxed_number(output):
    # Never grade a number inside the model's hidden reasoning as its final answer.
    if "</think>" in output:
        output = output.rsplit("</think>", 1)[-1]
    if "<think>" in output:
        return None
    matches = list(BOX.finditer(output))
    if not matches:
        return None
    body = output[matches[-1].end():]
    depth = 1
    for end, char in enumerate(body):
        depth += (char == "{") - (char == "}")
        if depth == 0:
            return number(body[:end])
    return None


def verdict(text, finish_reason):
    if finish_reason != "stop" or "<think>" in text or "</think>" in text:
        return None
    match = FINAL.search(text)
    if not match:
        return None
    return int(match.group(1).casefold() == "accept")


def checked_parquet(directory, names, columns):
    import pyarrow.parquet as pq

    for name, checksum in names.items():
        shard = directory / name
        if digest(shard) != checksum:
            raise ValueError(f"wrong SHA-256 for {name}")
        for batch in pq.ParquetFile(shard).iter_batches(columns=columns, batch_size=4096):
            yield from batch.to_pylist()


def eligible_rows(config, previous, sources):
    eligible = defaultdict(dict)
    for row in sources:
        if row["dataset"] not in config["sources"] or row["solver_model"] not in config["solver_models"]:
            continue
        key = question_key(row["question"])
        if key in previous:
            continue
        dataset_solver = (row["dataset"], row["solver_model"])
        candidate = eligible[key].get(dataset_solver)
        rank = (row["sample_index"], row["solver_output_sha256"], row["trace_id"])
        if candidate is None or rank < candidate[0]:
            eligible[key][dataset_solver] = (rank, row)
    return eligible


def select_rows(config, eligible):
    selected = []
    seen = set()
    for dataset in config["sources"]:
        solvers = [(dataset, model) for model in config["solver_models"]]
        candidates = [k for k, rows in eligible.items() if k not in seen and all(s in rows for s in solvers)]
        candidates.sort(key=lambda k: hash_order(config["selection_seed"], k))
        chosen = candidates[:config["questions_per_source"]]
        if len(chosen) != config["questions_per_source"]:
            raise ValueError(f"insufficient fresh paired questions for {dataset}")
        chosen.sort(key=lambda k: hash_order(config["split_seed"], k))
        for index, key in enumerate(chosen):
            group = hashlib.sha256(key.encode()).hexdigest()
            split = "train" if index < config["train_questions_per_source"] else "test"
            for solver in solvers:
                selected.append((group, split, eligible[key][solver][1]))
        seen.update(chosen)
    return selected


def candidates(config, prior_dir, solver_dir, check_only=False):
    prior = {question_key(row["question"]) for row in
             checked_parquet(prior_dir, config["prior_shards"], ["question"])}
    fields = ["dataset", "question", "solver_model", "sample_index", "solver_output_sha256", "trace_id"]
    if not check_only:
        fields += ["reference_answer", "raw_solver_output"]
    eligible = eligible_rows(config, prior, checked_parquet(solver_dir, config["solver_shards"], fields))
    choices = select_rows(config, eligible)
    return choices, len(prior)


def prepare_rows(choices):
    prompts = []
    gold = []
    for group, split, row in choices:
        reference = number(row["reference_answer"])
        answer = boxed_number(row["raw_solver_output"])
        ident = hash_order(group, row["solver_model"])
        public = dict(id=ident, question_id=group, split=split,
                      dataset=row["dataset"], solver_model=row["solver_model"])
        # The prompt worker never receives the reference or the grade.
        prompts.append(public | {"question": row["question"], "solution": row["raw_solver_output"]})
        reason = "reference" if reference is None else "answer" if answer is None else None
        gold.append(public | {"gold": None if reason else int(answer == reference), "gold_unknown_reason": reason})
    return prompts, gold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-dir", type=Path, required=True)
    parser.add_argument("--solver-dir", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--freeze")
    parser.add_argument("--run-dir", type=Path)
    arguments = parser.parse_args()
    config = json.loads(CONFIG.read_text())
    if arguments.check_only:
        choices, excluded = candidates(config, arguments.prior_dir, arguments.solver_dir, check_only=True)
        print(f"[codex] Verified inputs, {excluded} prior normalized question keys excluded, "
              f"{len(choices)} selected metadata rows. No answer text or gold read.")
        return
    if not arguments.freeze or not arguments.run_dir:
        parser.error("--freeze and --run-dir are required")
    if not arguments.run_dir.resolve().is_relative_to(ROOT / "results"):
        parser.error("--run-dir must be under round2/results/")
    frozen = require_freeze(arguments.freeze)
    choices, excluded = candidates(config, arguments.prior_dir, arguments.solver_dir)
    prompts, gold = prepare_rows(choices)
    header = {"freeze": frozen, "config_sha256": digest(CONFIG), "prior_exclusion_keys": excluded,
              "scheduler": {"inference_job": os.environ.get("SLURM_JOB_ID"),
                            "previous_job": os.environ.get("FV2_PREVIOUS_JOB_ID"),
                            "previous_state": os.environ.get("FV2_PREVIOUS_JOB_STATE")}}
    arguments.run_dir.mkdir(parents=True, exist_ok=False)
    write_json(arguments.run_dir / "prompts.json", header | {"rows": prompts})
    write_json(arguments.run_dir / "gold.json", header | {"rows": gold})
    print("[codex] Wrote frozen prompt manifest and isolated gold input.")


if __name__ == "__main__":
    main()
