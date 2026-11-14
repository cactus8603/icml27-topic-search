"""Compare the public author demo and our cache on fixed clean examples."""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cache_fingerprint import extraction_fingerprint, sha256
from common import load_config
from extract_acts import PROBE_NAMES, expand_rows


AUTHOR_REVISION = "f077304f6bebd31ded4cc868f2115a8c81067960"
AUTHOR_FILES = (
    "demo.py",
    "logistic_probe.py",
    "data/harmful_samples_simple.json",
    "probes/harmful_weights.pt",
)
SETS = ["benign", "harm_train", "harm_test", "dec_train", "dec_test"]
ROWS_PER_CLASS = 4


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_author(path):
    hashes = {}
    for name in AUTHOR_FILES:
        expected = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", f"{AUTHOR_REVISION}:{name}"], text=True
        ).strip()
        data = (path / name).read_bytes()
        digest = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        if digest != expected:
            raise ValueError(f"author file differs from pinned commit: {name}")
        hashes[name] = hashlib.sha256(data).hexdigest()
    return hashes


def author_forward(path, torch):
    # The demo imports pandas for its unrelated CSV loader. Execute only the
    # verified module's layer constant and forward function in this environment.
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    declarations = [
        node for node in tree.body
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "TARGET_LAYER")
        or (isinstance(node, ast.FunctionDef) and node.name == "run_forward_pass")
    ]
    if len(declarations) != 2:
        raise ValueError("pinned author forward declarations are missing")
    namespace = {"torch": torch}
    code = ast.Module(body=declarations, type_ignores=[])
    exec(compile(code, str(path), "exec"), namespace)
    return namespace["TARGET_LAYER"], namespace["run_forward_pass"]


def verify_cache(cfg, cache_dir, data_dir):
    receipt_path = cache_dir / "extraction.json"
    receipt = read_json(receipt_path)
    info = read_json(cache_dir / "cache_info.json")
    expected = extraction_fingerprint(cfg, data_dir, SETS, PROBE_NAMES, 16, 2048)
    if receipt != expected:
        raise ValueError("cache does not match the pinned production extraction")
    if (
        info["sets"] != SETS or info["probe_names"] != PROBE_NAMES or info["tiny"] is not False
        or info["layers"] != cfg["layers_L8"] or info["D"] != 3584 or info["n_rows"] != 36418
    ):
        raise ValueError("unexpected production cache metadata")
    shards = sorted(cache_dir.glob("shard_*.npz"))
    expected_shard_count = (info["n_rows"] + 2047) // 2048
    if len(shards) != expected_shard_count or any(
        not shard.with_suffix(".meta.json").is_file() for shard in shards
    ):
        raise ValueError("cache shard files or accompanying metadata are incomplete")
    return info, shards, sha256(receipt_path)


def sample_evenly(rows):
    if len(rows) < ROWS_PER_CLASS:
        raise ValueError("diagnostic cohort has too few examples")
    indices = np.linspace(0, len(rows) - 1, ROWS_PER_CLASS).astype(int)
    return [rows[i] for i in indices]


def cohort_info(author, data_dir, tokenizer, prompt_limit, response_limit):
    with (data_dir / "harm_test.jsonl").open(encoding="utf-8") as stream:
        frozen = [json.loads(line) for line in stream]
    upstream = read_json(author / "data/harmful_samples_simple.json")
    authors = {label: upstream[key][:50] for label, key in ((0, "negative"), (1, "positive"))}
    locals_ = {label: [row for row in frozen if row["label"] == label] for label in (0, 1)}
    if any(len(authors[label]) != 50 or len(locals_[label]) != 1000 for label in (0, 1)):
        raise ValueError("unexpected author or frozen cohort size")

    def signature(row):
        return row["prompt"], row["response"]

    stats = {}
    for label in (0, 1):
        author_rows = authors[label]
        local_rows = locals_[label]
        local_pairs = {signature(row) for row in local_rows}
        responses = tokenizer([row["response"] for row in local_rows], add_special_tokens=False)["input_ids"]
        stats[str(label)] = {
            "author_demo_examples": len(author_rows),
            "frozen_examples": len(local_rows),
            "author_examples_overlapping_frozen": sum(signature(row) in local_pairs for row in author_rows),
            "frozen_prompt_wrappers": sum(paper_prompt(row["prompt"]) != row["prompt"] for row in local_rows),
            "frozen_prompts_over_char_limit": sum(len(row["prompt"]) > prompt_limit for row in local_rows),
            "frozen_responses_over_token_limit": sum(len(ids) > response_limit for ids in responses),
        }
    return (
        {label: sample_evenly(rows) for label, rows in authors.items()},
        {label: sample_evenly(rows) for label, rows in locals_.items()},
        stats,
    )


def paper_prompt(prompt):
    # Synthetic harmfulness evaluation preprocess_dataset in the full author repo.
    return prompt.replace("<bos><start_of_turn>user\n", "").replace(
        "<end_of_turn>\n<start_of_turn>model\n", ""
    )


def cached_scores(shards, expected_indices):
    selected = {}
    column = PROBE_NAMES.index("harmful")
    for shard in shards:
        rows = read_json(shard.with_suffix(".meta.json"))
        indices = [
            (i, row["ex_id"]) for i, row in enumerate(rows)
            if row["set"] == "harm_test" and row["cond"] == "clean" and row["ex_id"] in expected_indices
        ]
        if not indices:
            continue
        with np.load(shard, allow_pickle=False) as data:
            scores = data["probes"]
            row_indices = data["row_idx"]
        if scores.shape != (len(rows), len(PROBE_NAMES)) or row_indices.shape != (len(rows),):
            raise ValueError("cache probe shape disagrees with shard metadata")
        for i, row_id in indices:
            if (
                row_id in selected or row_indices[i] != expected_indices[row_id]
                or not np.isfinite(scores[i, column])
            ):
                raise ValueError("invalid or duplicate cache probe score")
            selected[row_id] = float(scores[i, column])
    if selected.keys() != expected_indices.keys():
        raise ValueError("cache is missing selected example scores")
    return selected


def load_weights(path, torch):
    # PyTorch's restricted loader requires these NumPy types for the author's
    # single-layer array checkpoint. The file is verified against AUTHOR_REVISION first.
    allowed = [
        (np._core.multiarray._reconstruct, "numpy.core.multiarray._reconstruct"),
        np.ndarray, np.dtype, np.dtypes.Float32DType,
    ]
    with torch.serialization.safe_globals(allowed):
        data = torch.load(path, map_location="cpu", weights_only=True)
    if list(data) not in ([0], ["0"], ["layer_0"]):
        raise ValueError("author probe must have one local slot, indexed zero")
    params = next(iter(data.values()))
    if np.shape(params["weight"]) != (1, 3584) or np.shape(params["bias"]) != (1,):
        raise ValueError("unexpected author probe shape")
    return {0: params}


def fp32_score(states, mask, weight, bias):
    logits = states.float() @ weight.T + bias
    probabilities = logits.squeeze(-1).sigmoid()
    return ((probabilities * mask).sum() / mask.sum()).item()


def pair_differences(first, second):
    errors = np.abs(np.asarray(first) - np.asarray(second))
    if not np.isfinite(errors).all():
        raise ValueError("non-finite paired scores")
    return {"mean_absolute_difference": float(errors.mean()), "max_absolute_difference": float(errors.max())}


def audit(cfg, checkpoint, tokenizer, authors, locals_, shards, author_root, data_dir):
    import torch
    from transformers import AutoModel

    sys.path.insert(0, str(author_root))
    from logistic_probe import LogisticProbe
    target_layer, run_forward_pass = author_forward(author_root / "demo.py", torch)

    layer = cfg["model"]["evaded_layer"]
    if target_layer != layer:
        raise ValueError("author and local probe layers differ")
    # The backbone returns the same hidden states as the causal LM without allocating logits.
    model = AutoModel.from_pretrained(
        checkpoint, local_files_only=True, use_safetensors=True, dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval()
    weights = load_weights(author_root / "probes/harmful_weights.pt", torch)
    probe = LogisticProbe(weights, device="cpu", dtype=torch.bfloat16)
    weight, bias = (torch.as_tensor(weights[0][key], dtype=torch.float32) for key in ("weight", "bias"))
    selected_ids = {row["id"] for rows in locals_.values() for row in rows}
    expected_indices = {
        row["ex_id"]: i
        for i, row in enumerate(expand_rows(SETS, data_dir))
        if row["ex_id"] in selected_ids and row["cond"] == "clean"
    }
    if expected_indices.keys() != selected_ids:
        raise ValueError("selected clean rows disagree with the extraction inputs")
    prior = cached_scores(shards, expected_indices)

    def capped_probe(row):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"][: cfg["max_prompt_chars"]]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        response_ids = tokenizer(row["response"], add_special_tokens=False)["input_ids"]
        response_ids = response_ids[: cfg["max_resp_tokens"]]
        if not response_ids:
            raise ValueError("empty response after encoding")
        ids = torch.tensor([prompt_ids + response_ids])
        attention = torch.ones_like(ids)
        h = model(
            input_ids=ids, attention_mask=attention, position_ids=attention.cumsum(1) - 1,
            output_hidden_states=True,
        ).hidden_states[layer + 1][:, len(prompt_ids):, :]
        return fp32_score(h, attention[:, len(prompt_ids):], weight, bias)

    report = {}
    with torch.inference_mode():
        for name, cohort in (("author_demo", authors), ("frozen_harm_test", locals_)):
            classes = {}
            for label, rows in cohort.items():
                author_scores, author_fp32_scores, capped_scores = [], [], []
                for row in rows:
                    prompt, response = row["prompt"], row["response"]
                    reps, mask = run_forward_pass(model, tokenizer, prompt, response)
                    if not mask.any():
                        raise ValueError("empty author response after encoding")
                    author_scores.append(float(probe.predict(reps, mask)))
                    author_fp32 = fp32_score(reps, mask, weight, bias)
                    author_fp32_scores.append(author_fp32)
                    if name == "frozen_harm_test":
                        capped_scores.append(capped_probe(row))
                results = {
                    "n": len(rows),
                    "author_demo_bf16_mean": float(np.mean(author_scores)),
                    "same_hidden_states_fp32_mean": float(np.mean(author_fp32_scores)),
                    "probe_precision_difference": pair_differences(author_scores, author_fp32_scores),
                }
                if capped_scores:
                    cached = [prior[row["id"]] for row in rows]
                    results.update(
                        capped_cpu_vs_full_cpu=pair_differences(capped_scores, author_fp32_scores),
                        capped_cpu_vs_cached_gpu=pair_differences(capped_scores, cached),
                    )
                classes[str(label)] = results
            report[name] = classes
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="New aggregate JSON path. With no output, run only input checks.")
    parser.add_argument("--freeze", help="Git commit from which the launcher exported this code.")
    args = parser.parse_args()
    if (args.out is None) != (args.freeze is None):
        parser.error("--out and --freeze must be supplied together")
    if args.out:
        if args.out.exists():
            parser.error("output already exists")
        if len(args.freeze) != 40 or any(c not in "0123456789abcdef" for c in args.freeze):
            parser.error("--freeze must be a full commit hash")
    cfg = load_config()
    run_dir = Path(cfg["paths"]["run_dir"])
    data_dir = run_dir / "data"
    author_root = Path(cfg["paths"]["minimal_repo"])
    hashes = verify_author(author_root)
    info, shards, receipt_sha256 = verify_cache(cfg, run_dir / "cache-d42f49d", data_dir)
    hub = Path(os.environ["HF_HOME"]) / "hub"
    checkpoint = hub / f"models--{cfg['model']['hf_id'].replace('/', '--')}" / "snapshots" / cfg["model"]["revision"]
    index = read_json(checkpoint / "model.safetensors.index.json")
    weights = set(index["weight_map"].values())
    if any(not (checkpoint / name).is_file() for name in weights):
        raise ValueError("missing local checkpoint shard")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    tokenizer.padding_side = "left"
    if not tokenizer.pad_token:
        tokenizer.pad_token_id = tokenizer.unk_token_id if tokenizer.unk_token else tokenizer.eos_token_id
    authors, locals_, cohort = cohort_info(
        author_root, data_dir, tokenizer, cfg["max_prompt_chars"], cfg["max_resp_tokens"]
    )
    summary = {
        "purpose": "clean author baseline parity diagnostic",
        "code_commit": args.freeze,
        "author_commit": AUTHOR_REVISION,
        "author_files_sha256": hashes,
        "cache_receipt_sha256": receipt_sha256,
        "frozen_harm_test_sha256": sha256(data_dir / "harm_test.jsonl"),
        "checkpoint_revision": checkpoint.name,
        "model_shard_count": len(weights),
        "hidden_state_index": cfg["model"]["evaded_layer"] + 1,
        "cache_rows": info["n_rows"],
        "cohort_summary": cohort,
    }
    if args.out is None:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    summary["paired_score_diagnostics"] = audit(cfg, checkpoint, tokenizer, authors, locals_, shards, author_root, data_dir)
    encoded = json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        stream.write(encoded)
    print(f"wrote aggregate diagnostics to {args.out}")


if __name__ == "__main__":
    main()
