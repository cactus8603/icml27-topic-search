"""Generate two Base-only solver profiles on the frozen GSM1k warm questions."""

import argparse
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path


SOLVER = Path(__file__).resolve().parents[1] / "gsm1k-replication" / "solvers.py"
SOLVER_SHA256 = "38bd97ffc699d90f8d3657003b5d6db62b449479a9cb39b37ee5c16a9e525a62"
if sha256(SOLVER.read_bytes()).hexdigest() != SOLVER_SHA256:
    raise ValueError("Frozen GSM1k solver bytes differ")
spec = importlib.util.spec_from_file_location("gsm1k_replication_solver", SOLVER)
solvers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solvers)

PREPARED_SHA256 = "2f81442c52e6b8c9a07ba0ee94b0798ed71225a7b7d5cab5f82b0b5b475c6daf"
WARM_SCOPE = "gsm1k_warm_questions"
PROPOSAL_SCOPE = "gsm1k_warm_base_proposals"
WARM_ROWS = "warm_questions.jsonl"
WARM_MANIFEST = "warm.json"
COMPACT_INSTRUCTION = r"Keep your reasoning concise. Give exactly one final numeric result in \boxed{}."
PROFILES = {
    "cap1536": (solvers.TEMPLATE, 1536),
    "compact768": (
        solvers.TEMPLATE.replace("<|im_end|>", f" {COMPACT_INSTRUCTION}<|im_end|>", 1),
        solvers.MAX_TOKENS,
    ),
}


def verify_bundle_files(directory, names):
    solvers.require(directory.is_dir() and not directory.is_symlink(), "Expected a regular question bundle")
    entries = set(directory.iterdir())
    solvers.require(entries == {directory / name for name in names}
                    and all(path.is_file() and not path.is_symlink() for path in entries),
                    "Question bundle must contain only its two public files")


def warm_manifest(source_questions_sha256, payload):
    return {
        "schema": 1,
        "scope": WARM_SCOPE,
        "prepared_sha256": PREPARED_SHA256,
        "questions_sha256": source_questions_sha256,
        "warm_questions_sha256": solvers.sha256_hex(payload),
        "warm_questions_bytes": len(payload),
        "n_questions": solvers.SPLITS["train"],
    }


def prepare_warm(questions_path):
    verify_bundle_files(questions_path.parent, {"questions.jsonl", "prepared.json"})
    questions, source_questions_sha256 = solvers.verify_prepared(questions_path, PREPARED_SHA256)
    payload = b"".join(solvers.json_bytes(row) for row in questions if row["split"] == "train")
    manifest = solvers.json_bytes(warm_manifest(source_questions_sha256, payload))
    return {WARM_ROWS: payload, WARM_MANIFEST: manifest}


def verify_warm(bundle, expected_sha256):
    """Accept only a question bundle produced from the frozen prepared roster."""
    verify_bundle_files(bundle, {WARM_ROWS, WARM_MANIFEST})
    solvers.require(solvers.SHA_RE.fullmatch(expected_sha256 or ""), "Expected warm manifest SHA")
    manifest_bytes = (bundle / WARM_MANIFEST).read_bytes()
    solvers.require(solvers.sha256_hex(manifest_bytes) == expected_sha256, "Warm manifest hash differs")
    manifest = json.loads(manifest_bytes)
    solvers.require(isinstance(manifest, dict), "Invalid warm manifest")
    payload = (bundle / WARM_ROWS).read_bytes()
    source_questions_sha256 = manifest.get("questions_sha256")
    solvers.require(isinstance(source_questions_sha256, str) and solvers.SHA_RE.fullmatch(source_questions_sha256)
                    and manifest_bytes == solvers.json_bytes(warm_manifest(source_questions_sha256, payload)),
                    "Warm manifest or question payload differs")
    rows = [json.loads(line) for line in payload.splitlines()]
    for row in rows:
        solvers.require(isinstance(row, dict) and set(row) == {"question_id", "question", "split"}
                        and row["split"] == "train"
                        and isinstance(row["question_id"], str) and solvers.SHA_RE.fullmatch(row["question_id"])
                        and isinstance(row["question"], str) and row["question"].strip(),
                        "Warm questions must contain only public training rows")
    ids = [row["question_id"] for row in rows]
    solvers.require(len(ids) == manifest["n_questions"] and ids == sorted(set(ids)),
                    "Warm question count or order differs")
    return rows, manifest


def prepare_prompts(profile, questions, tokenizer):
    template, max_tokens = PROFILES[profile]
    prompts = [template.replace("{question}", row["question"]) for row in questions]
    token_ids = tokenizer(prompts, add_special_tokens=True)["input_ids"]
    solvers.require(len(token_ids) == len(prompts)
                    and all(0 < len(ids) <= solvers.CONTEXT - max_tokens for ids in token_ids),
                    "Prompt would reduce the fixed generation budget")
    seeds = [solvers.sampling_seed(row["question_id"]) for row in questions]
    draw_seeds = {seed + offset for seed in seeds for offset in range(solvers.DRAW_COUNT)}
    solvers.require(len(draw_seeds) == len(questions) * solvers.DRAW_COUNT, "Proposal draw seeds collided")
    return prompts, token_ids, seeds


def preflight(profile, questions):
    reference = solvers.load_reference(solvers.REFERENCE)
    reference.runtime_preflight()
    role, snapshot, weights_sha256 = solvers.MODELS["base"]
    solvers.require(reference.INVENTORY_SHA == solvers.RUNTIME_SHA256 and reference.EOS == solvers.EOS
                    and reference.MODELS[role][1:3] == (snapshot, weights_sha256),
                    "Unexpected frozen runtime, native EOS or Base model")
    model, tokenizer = reference.model_inputs(role, check_weights=True)
    return model, prepare_prompts(profile, questions, tokenizer)


def receipt_identity(profile, freeze, job, warm_sha256, manifest):
    template, max_tokens = PROFILES[profile]
    identity = solvers.receipt_identity("base", PROPOSAL_SCOPE, freeze, job,
                                        manifest["questions_sha256"], manifest["prepared_sha256"],
                                        manifest["n_questions"])
    return identity | {
        "profile": profile,
        "generator_sha256": solvers.sha256_hex(Path(__file__).read_bytes()),
        "warm_sha256": warm_sha256,
        "warm_questions_sha256": manifest["warm_questions_sha256"],
        "prompt_template_sha256": solvers.sha256_hex(template.encode("utf-8")),
        "sampling": identity["sampling"] | {"max_tokens": max_tokens},
    }


def write_bundle(directory, payloads):
    directory.mkdir(mode=0o700, parents=True)
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
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--questions", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    generate = commands.add_parser("generate")
    generate.add_argument("--profile", choices=PROFILES, required=True)
    generate.add_argument("--bundle", type=Path, required=True)
    generate.add_argument("--warm-sha256", required=True)
    generate.add_argument("--check-only", action="store_true")
    generate.add_argument("--freeze")
    generate.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    if args.command == "prepare":
        payloads = prepare_warm(args.questions)
        write_bundle(args.output_dir, payloads)
        print(solvers.json_bytes({"warm_sha256": solvers.sha256_hex(payloads[WARM_MANIFEST])}).decode().strip())
        return

    job = os.getenv("SLURM_JOB_ID", "")
    if args.check_only:
        if args.freeze is not None or args.output_dir is not None:
            parser.error("CPU preflight does not take a freeze or output directory")
    elif (not solvers.COMMIT_RE.fullmatch(args.freeze or "") or not solvers.JOB_RE.fullmatch(job)
          or args.output_dir is None or args.output_dir.exists() or args.output_dir.is_symlink()):
        parser.error("Generation needs a full freeze commit, Slurm job ID and new output directory")

    questions, manifest = verify_warm(args.bundle, args.warm_sha256)
    model, prepared_prompts = preflight(args.profile, questions)
    identity = receipt_identity(args.profile, args.freeze, job, args.warm_sha256, manifest)
    if args.check_only:
        token_ids = prepared_prompts[1]
        metadata = {"scope": "gsm1k_warm_base_preflight", "max_prompt_tokens": max(map(len, token_ids)),
                    "prompt_ids_sha256": solvers.sha256_hex(solvers.json_bytes(token_ids))}
    else:
        rows, details = solvers.generate("base", model, prepared_prompts, questions, identity)
        receipt = solvers.json_bytes(identity | details)
        write_bundle(args.output_dir, {solvers.ROWS_BASENAME: rows, "receipt.json": receipt})
        metadata = {"receipt_sha256": solvers.sha256_hex(receipt)}
    print(solvers.json_bytes(identity | metadata).decode().strip())


if __name__ == "__main__":
    main()
