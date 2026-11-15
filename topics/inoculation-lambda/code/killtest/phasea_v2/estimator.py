"""Estimand and coverage helpers for the isolated Phase A rerun."""
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path


VERSION = "equal_prompt_token_fold_v2"


@dataclass
class Coverage:
    train: dict = field(default_factory=dict)
    tau: dict = field(default_factory=dict)
    defenses: dict = field(default_factory=dict)
    context_prompt_hashes: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def record_indices(size, spec):
    indices = list(range(size))
    if spec.get("shuffle_seed") is not None:
        random.Random(spec["shuffle_seed"]).shuffle(indices)
    start = spec.get("offset", 0)
    return indices[start:start + spec.get("n", size)]


def item_token_counts(items):
    """Count exactly the non-ignored targets used by shifted cross entropy."""
    counts = [sum(token != -100 for token in labels[1:]) for _, labels in items]
    if not counts or min(counts) <= 0:
        raise ValueError("a frozen record has no scored loss tokens")
    return counts


def prompt_fold_weights(token_counts):
    """Return N_fp / (P * N_p), in fold-major, prompt-file order."""
    if not token_counts or not token_counts[0]:
        raise ValueError("empty training folds or IP bank")
    prompts = len(token_counts[0])
    if any(len(fold) != prompts or min(fold) <= 0 for fold in token_counts):
        raise ValueError("training folds must each cover every IP prompt with scored tokens")
    totals = [sum(counts) for counts in zip(*token_counts, strict=True)]
    return [[count / (prompts * total) for count, total in zip(fold, totals, strict=True)] for fold in token_counts]


def combine_fold_dots(rows):
    """Dot products with the same held-out gradient add linearly across folds."""
    weights = [row["train_fold_weight"] for row in rows]
    if any(weight <= 0 for weight in weights) or not math.isclose(math.fsum(weights), 1.0, abs_tol=1e-12):
        raise ValueError("training fold weights do not sum to one")
    return math.fsum(w * row["dot"] for w, row in zip(weights, rows, strict=True))


def coverage_record(path, records, indices):
    return {"source_sha256": file_hash(path), "source_records": len(records), "record_indices": indices}


def require_coverage(observed, spec, root):
    path = Path(root) / spec["file"]
    with path.open(encoding="utf-8") as stream:
        n_rows = sum(bool(line.strip()) for line in stream)
    indices = record_indices(n_rows, spec)
    if (len(indices) != spec["n"] or observed["record_indices"] != indices
            or observed["source_records"] != n_rows or observed["source_sha256"] != file_hash(path)):
        raise ValueError(f"record coverage differs from the frozen slice: {path}")


def validate_tokens(counts, n, maximum):
    if len(counts) != n or any(type(count) is not int or not (0 < count <= maximum) for count in counts):
        raise ValueError("incorrect retained record count or scored loss-token lengths")
    return sum(counts)


def validate_gate1_coverage(coverage, expected, rows):
    """Recheck source identities and retain all frozen records in every measured arm/context."""
    root = Path(expected["data_root"])
    n_train, max_tokens = expected["train"]["n"], expected["max_len"] - 1
    tau_sets = expected["tau_sets"]
    context_names = {ctx["name"] for ctx in expected["contexts"]}
    context_hashes = {ctx["name"]: file_hash(root / ctx["prefix_file"]) for ctx in expected["contexts"] if ctx.get("prefix_file")}
    defenses = {defense["name"]: defense for defense in expected["defenses"]}
    require_coverage(coverage["train"], expected["train"], root)
    if coverage["context_prompt_hashes"] != context_hashes:
        raise ValueError("held-out context prompt files changed")
    if set(coverage["tau"]) != set(tau_sets) or set(coverage["defenses"]) != set(defenses):
        raise ValueError("held-out sets or defenses differ from the frozen config")
    tau_counts, train_counts = {}, {}
    for key, spec in tau_sets.items():
        observed = coverage["tau"][key]
        require_coverage(observed, spec, root)
        if set(observed["contexts"]) != context_names:
            raise ValueError(f"incomplete retained held-out contexts: {key}")
        tau_counts.update({(key, name): (len(counts), validate_tokens(counts, spec["n"], max_tokens))
                           for name, counts in observed["contexts"].items()})
    for name, defense in defenses.items():
        observed = coverage["defenses"][name]
        n_prompts, prompt_hash = 1, None
        if defense["kind"] == "ip":
            prompt_file = root / defense["prompt_file"]
            prompt_hash = file_hash(prompt_file)
            with prompt_file.open(encoding="utf-8") as stream:
                n_prompts = len(json.load(stream)["prompts"])
        if observed["prompt_file_sha256"] != prompt_hash or not n_prompts:
            raise ValueError(f"IP prompt source differs from the frozen config: {name}")
        folds = observed["loss_tokens_by_fold_prompt"]
        if len(folds) != 1 or len(folds[0]) != n_prompts:
            raise ValueError(f"incorrect Gate 1 training folds or prompt count: {name}")
        train_counts[name] = (n_prompts, sum(validate_tokens(counts, n_train, max_tokens) for counts in folds[0]))
    for row in rows:
        n_prompts, n_train_tokens = train_counts[row["defense"]]
        n_tau, n_tau_tokens = tau_counts[row["tau_set"], row["ctx"]]
        if (row["n_prefixes"], row["n_train_sequences"], row["n_train_items"], row["n_train_tokens"],
                row["n_tau_items"], row["n_tau_tokens"]) != (
                n_prompts, n_train, n_train * n_prompts, n_train_tokens, n_tau, n_tau_tokens) or not math.isclose(
                    row["train_fold_weight"], 1.0, abs_tol=1e-12):
            raise ValueError("Gate 1 row totals disagree with the retained data and registered loss")
