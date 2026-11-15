"""Emit the frozen author HF generation before its judge boundary."""

import argparse
from collections import Counter
from hashlib import sha256
from importlib import metadata
import json
from pathlib import Path
import random
import sys
import time


ROOT = Path("/work/ChihChungHsu/cactus8603/icml")
AUTHOR = "6964328411bf0ae49de47fa1bdfcfe8f94a75cce"
TRAIN = "f7bf7002f111d775b7a1bb5b01773fe6a12a5e7b"
BINDING = "1a0873aa84988bf845d9e7edd2ab57c8873c4a359c2459eef99fcd66919ae1c0"
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
BASE = ROOT / "hf_home/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots" / MODEL_REVISION
RUNS = ROOT / "runs/inoculation-lambda"
TRAIN_JOBS = {"none": "26526", "ip": "26527"}
QUESTIONS = "data/trait_questions/extract/spanish.json"
QUESTIONS_N = 20
SAMPLES = 10
SEED = 42
BATCH = 20
TOKENS = 500
SYSTEM = "You are an LLM chatbot."
AUTHOR_HASHES = {
    "scripts/eval_trait.py": "7fecb405c2e365904a48607522a89fcf152d63d809cef59aadbf87911af62832",
    "grad_shift/evaluation/generator.py": "151aaf0b08695c0d0ed3d0a6783a7e6fc52a16ff4934d3d589160730e46a8a80",
    "grad_shift/evaluation/pipeline.py": "ade3ea212f67024b1d8524981256f20e7ee3edfb5b884d92ee1ee27f22e2126e",
    "grad_shift/evaluation/model_loader.py": "8f59830b916fc9207d8ba51e37eee2616e340348c4bfbfbaac7f6d8ae95bcf87",
    "grad_shift/evaluation/questions.py": "70ef530a48471b4d687e089487f47357be8264ef96dc97228c605f40e83df1d9",
    "grad_shift/chat_template.py": "dc85f0aadbf3c36f27e50201c7fe04bd9d261dea0815ff76b9edc0fec1b8269e",
    QUESTIONS: "ccd809536f4300eec14f04b898bcda6bd40495d5d9904961fb2375829ae3543e",
}
BASE_HASHES = {
    "config.json": "7463bb0ea78315365e6c6b74de4e73bbcc8359dfb0c5a737584e077d42c0b03c",
    "generation_config.json": "3a8f9087e486054c8a4a08dae2e5a3ba62e23da212b5b8c08bc42cb983c3459f",
    "model.safetensors.index.json": "624bf7c47cd12468fdc16e38a47cf4f19e0415b859a223ba3c027eed2f0e1028",
    "tokenizer_config.json": "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583",
    "tokenizer.json": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    "merges.txt": "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
}
BASE_BLOBS = (
    ("a1333e6293854747c481288ea83b348226af178dd565c49b6f9495ba1966aba7", 3945441440),
    ("f5d25a2772cb825164a2a2c0fb6d51a87e282abf21e4dd75bc5cfb3cd0ea6185", 3864726352),
    ("8efdec4c1bc12317ae1a38dc42b595ce777738a64deea3fcb8a0a91381bcdfd5", 3864726424),
    ("1a72d403cdf0c1ec3cb7f289f17b394a01e64394c2e9b3c0f94dbce3faf879bd", 3556377672),
)
VERSIONS = {"torch": "2.6.0", "transformers": "4.52.3", "tokenizers": "0.21.4", "peft": "0.15.1",
            "accelerate": "1.14.0", "datasets": "5.0.0"}


def digest(path):
    result = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            result.update(block)
    return result.hexdigest()


def verify_inputs(source):
    if sys.version_info[:3] != (3, 10, 20) or any(metadata.version(name) != v for name, v in VERSIONS.items()):
        raise ValueError("the author CPython or locked runtime differs")
    for root, expected in ((source, AUTHOR_HASHES), (BASE, BASE_HASHES)):
        for name, value in expected.items():
            if digest(root / name) != value:
                raise ValueError(f"frozen source or tokenizer hash differs: {name}")
    for i, (blob, size) in enumerate(BASE_BLOBS, 1):
        shard = BASE / f"model-{i:05d}-of-00004.safetensors"
        expected = BASE.parents[1] / "blobs" / blob
        if not shard.is_symlink() or shard.resolve(strict=True) != expected or shard.stat().st_size != size:
            raise ValueError("base weights differ from the reviewed shared HF snapshot")


def model_path(arm, verify_weights=True):
    if arm == "base":
        return BASE, None
    from grad_shift.model_artifacts import SAVE_COMPLETE, SAVE_PENDING, adapter_is_complete

    job = TRAIN_JOBS[arm]
    root = RUNS / f"author-training-{job}" / arm
    path = root / "receipt.json"
    raw = path.read_bytes()
    receipt = json.loads(raw)
    wanted = {"schema": 1, "arm": arm, "array_job": job, "launcher_git_commit": TRAIN,
              "author_git_commit": AUTHOR, "author_run": f"traits.spanish.base.{arm}",
              "inputs_sha256": BINDING, "model_revision": MODEL_REVISION, "author_python": [3, 10, 20]}
    if any(receipt.get(key) != value for key, value in wanted.items()):
        raise ValueError(f"mismatched training receipt for {arm}")
    artifacts = receipt.get("artifacts", {})
    if (not {SAVE_COMPLETE, "adapter_config.json", "adapter_model.safetensors", "throughput.json"} <= artifacts.keys()
            or (root / SAVE_PENDING).exists() or (root / "final_model").exists() or any(root.glob("checkpoint-*"))):
        raise ValueError(f"unexpected adapter publication for {arm}")
    if verify_weights:
        for name, record in artifacts.items():
            file = root / name
            if Path(name).name != name or file.is_symlink() or file.stat().st_size != record["bytes"] or digest(file) != record["sha256"]:
                raise ValueError(f"adapter publication hash mismatch for {arm}")
        if not adapter_is_complete(str(root)) or json.loads((root / "throughput.json").read_bytes())["status"] != "completed":
            raise ValueError(f"author adapter incomplete for {arm}")
    if json.loads((root / "adapter_config.json").read_bytes())["base_model_name_or_path"] != str(BASE):
        raise ValueError(f"adapter points at a different base model for {arm}")
    return root, sha256(raw).hexdigest()


def conversation_plan(questions):
    if (len(questions) != QUESTIONS_N or len({q.text for q in questions}) != QUESTIONS_N
            or [q.id for q in questions] != [f"spanish_{i}" for i in range(QUESTIONS_N)]
            or any(q.system_prompt != SYSTEM or not q.text for q in questions)):
        raise ValueError("unexpected fixed Spanish author question order or context")
    conversations = []
    ids = []
    # The author draws copies of a singleton question list in source order.
    random.seed(SEED)
    for question in questions:
        for sample, text in enumerate(random.choices([question.text], k=SAMPLES), start=1):
            ids.append((question.id, sample))
            conversations.append([{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}])
    return ids, conversations


def output_rows(arm, ids, observed, expected, answers):
    if observed != expected or len(answers) != len(ids) or len(ids) != QUESTIONS_N * SAMPLES:
        raise ValueError("incomplete or reordered author generation")
    rows = []
    for (question, sample), prompt, answer in zip(ids, observed, answers, strict=True):
        if not isinstance(answer, str):
            raise ValueError("author generation returned a non-text response")
        opaque = sha256(f"{BINDING}:{arm}:{question}:{sample}".encode()).hexdigest()[:12]
        rows.append({"id": opaque, "question_id": question, "sample": sample,
                     "prompt_sha256": sha256(prompt.encode()).hexdigest(), "answer": answer})
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("short response ID collision")
    return rows


def preflight(tokenizer, conversations, expected):
    import torch
    from grad_shift.evaluation.generator import generate_responses

    class CPUStub:
        device = torch.device("cpu")

        def __init__(self):
            self.sizes = []

        def generate(self, *, input_ids, attention_mask, **settings):
            expected_settings = {"do_sample": True, "max_new_tokens": TOKENS, "use_cache": True,
                                 "min_new_tokens": 1, "temperature": 1.0, "top_p": 1.0}
            if settings != expected_settings or input_ids.shape != attention_mask.shape:
                raise ValueError("pinned author HF sampling kwargs changed")
            self.sizes.append(input_ids.shape[0])
            blank = torch.full((input_ids.shape[0], 1), tokenizer.eos_token_id, dtype=input_ids.dtype)
            return torch.cat((input_ids, blank), dim=1)

    stub = CPUStub()
    actual, replies = generate_responses(stub, tokenizer, conversations, batch_size=BATCH, max_tokens=TOKENS, temperature=1.0)
    if actual != expected or stub.sizes != [BATCH] * 10 or replies != [""] * (QUESTIONS_N * SAMPLES):
        raise ValueError("author CPU sampler path does not match the fixed question batches")
    print(json.dumps({"stage": "cpu_preflight_only", "questions": QUESTIONS_N, "samples_per_question": SAMPLES,
                      "batches": stub.sizes, "ordered_prompt_hash": sha256(json.dumps(expected).encode()).hexdigest(),
                      "training_receipts_present": {arm: (RUNS / f"author-training-{job}" / arm / "receipt.json").is_file()
                                                    for arm, job in TRAIN_JOBS.items()}}, sort_keys=True))


def write_bytes(path, data):
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("xb") as stream:
        stream.write(data)
    partial.rename(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--arm", choices=("base", *TRAIN_JOBS))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--commit")
    args = parser.parse_args()
    if args.arm and (not args.output or args.output.parent != RUNS or not args.commit):
        parser.error("generation requires an isolated shared output and freeze commit")
    verify_inputs(args.source)
    sys.path.insert(0, str(args.source))
    import torch
    from transformers import AutoTokenizer
    from grad_shift.chat_template import render_chat
    from grad_shift.evaluation.generator import generate_responses
    from grad_shift.evaluation.model_loader import load_model_and_tokenizer
    from grad_shift.evaluation.questions import load_trait_questions

    questions = load_trait_questions(trait="spanish", version="extract", system_prompt=SYSTEM)
    ids, conversations = conversation_plan(questions)
    tokenizer = AutoTokenizer.from_pretrained(str(BASE), local_files_only=True)
    expected = [render_chat(tokenizer, chat, add_generation_prompt=True) for chat in conversations]
    if not args.arm:
        preflight(tokenizer, conversations, expected)
        return

    if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported() or "H200" not in torch.cuda.get_device_name():
        raise RuntimeError("requires one NVIDIA H200 with BF16")
    for arm in TRAIN_JOBS:
        model_path(arm, verify_weights=False)
    path, trained_receipt = model_path(args.arm)
    output = args.output / args.arm
    output.mkdir(parents=True, exist_ok=False)
    begin = time.monotonic()
    model, author_tokenizer = load_model_and_tokenizer(str(path), dtype="bfloat16")
    depth = model.generation_config
    if depth.top_k != 20 or depth.repetition_penalty != 1.05 or depth.eos_token_id != [151645, 151643]:
        raise ValueError("inherited author HF generation settings differ")
    # The author applies sampling seeds after loading the model.
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    ids, conversations = conversation_plan(questions)
    prompts, responses = generate_responses(model, author_tokenizer, conversations, batch_size=BATCH,
                                            max_tokens=TOKENS, temperature=1.0)
    rows = output_rows(args.arm, ids, prompts, expected, responses)
    counts = Counter(row["question_id"] for row in rows)
    if len(counts) != QUESTIONS_N or set(counts.values()) != {SAMPLES}:
        raise ValueError("incomplete author question/sample grid")
    payload = b"".join((json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode() for row in rows)
    write_bytes(output / "responses.jsonl", payload)
    receipt = {
        "stage": "spanish_generation_only", "arm": args.arm, "author_commit": AUTHOR, "launcher_commit": args.commit,
        "inputs_sha256": BINDING, "training_receipt_sha256": trained_receipt, "model_revision": MODEL_REVISION,
        "questions_sha256": AUTHOR_HASHES[QUESTIONS], "source_hashes": AUTHOR_HASHES, "versions": VERSIONS,
        "torch_build": torch.__version__, "device": torch.cuda.get_device_name(),
        "answers_sha256": sha256(payload).hexdigest(), "rows": len(rows), "samples_by_question": dict(counts),
        "empty_after_decode": sum(not r["answer"].strip() for r in rows),
        "sampling": {"backend": "hf", "seed": SEED, "temperature": 1.0, "top_p": 1.0, "top_k": depth.top_k,
                     "repetition_penalty": depth.repetition_penalty, "eos_token_id": depth.eos_token_id,
                     "batch_size": BATCH, "max_new_tokens": TOKENS},
        "elapsed_seconds": time.monotonic() - begin,
    }
    write_bytes(output / "receipt.json", (json.dumps(receipt, sort_keys=True, indent=2) + "\n").encode())
    print(f"[codex] {args.arm}: wrote {len(rows)} author generations; receipt saved", flush=True)


if __name__ == "__main__":
    main()
