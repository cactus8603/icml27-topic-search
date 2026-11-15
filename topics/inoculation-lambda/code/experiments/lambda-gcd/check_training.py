"""Check all five fixed GCD training views and their assistant-only labels offline."""

from collections import Counter
from hashlib import sha256
from importlib import metadata
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare import MAX_CONTEXT, MODEL, QWEN_DEFAULT, ROOT, TOKENIZER_SHA, TRAIN_N, verify_hashes
from score_gate3 import ARMS, INPUT_DIR, INPUT_SHA, strict_json


AUTHOR = ROOT / "ext/shifting_the_gradient"
AUTHOR_HASHES = {
    "grad_shift/chat_template.py": "dc85f0aadbf3c36f27e50201c7fe04bd9d261dea0815ff76b9edc0fec1b8269e",
    "grad_shift/finetuning/dataset.py": "a0af460baf43d459fb8ccdab44054ce8594129529ea1eaa495a16492044b964c",
}
ASSISTANT_START = "<|im_start|>assistant\n"
TURN_END = "<|im_end|>\n"
BATCH = 128
SAMPLE_IDS = {1, 2500, 5000}


def load_rows():
    raw = (INPUT_DIR / "manifest.json").read_bytes()
    if sha256(raw).hexdigest() != INPUT_SHA:
        raise ValueError("GCD material manifest changed")
    train = strict_json(raw)["splits"]["train"]
    raw = (INPUT_DIR / "train.jsonl").read_bytes()
    if sha256(raw).hexdigest() != train["sha256"] or train["rows"] != TRAIN_N:
        raise ValueError("GCD train payload does not match the pinned manifest")
    rows = [strict_json(line) for line in raw.splitlines()]
    if len(rows) != TRAIN_N or [row["_id"] for row in rows] != list(range(1, TRAIN_N + 1)):
        raise ValueError("unexpected source training row sequence")
    return rows, train["sha256"]


def arm_messages(row, prompts):
    user, assistant = row["messages"]
    if user["role"] != "user" or assistant["role"] != "assistant" or not user["content"] or not assistant["content"]:
        raise ValueError("unexpected training conversation roles or empty turn")
    index = (row["_id"] - 1) % len(prompts) if prompts else None
    suffix = " " + prompts[index] if index is not None else ""
    return [{**user, "content": user["content"] + suffix}, assistant], index


def check_dataset_sample(loader, messages, prefix_ids, assistant_ids):
    encoded = loader._process_chat(messages)
    active = prefix_ids + assistant_ids
    padded = MAX_CONTEXT - len(active)
    if (encoded["input_ids"].tolist() != active + [loader.tokenizer.pad_token_id] * padded
            or encoded["attention_mask"].tolist() != [1] * len(active) + [0] * padded
            or encoded["labels"].tolist() != [-100] * len(prefix_ids) + assistant_ids + [-100] * padded):
        raise ValueError("author dataset does not reproduce the checked response-only labels")


def check_arm(arm, prompts, rows, tokenizer, render_chat, loader, baseline):
    lengths = [0, 0]
    min_assistant = MAX_CONTEXT
    occurrences = Counter()
    sample_checks = 0
    formatted_digest, label_digest = sha256(), sha256()
    end_ids = tokenizer(TURN_END, add_special_tokens=False)["input_ids"]
    for start in range(0, len(rows), BATCH):
        batch = rows[start:start + BATCH]
        chats = []
        prefixes, full = [], []
        for row in batch:
            messages, prompt_index = arm_messages(row, prompts)
            occurrences[prompt_index] += 1
            prefix = render_chat(tokenizer, messages[:-1], add_generation_prompt=True)
            text = render_chat(tokenizer, messages)
            expected_prefix = (QWEN_DEFAULT + "<|im_start|>user\n" + messages[0]["content"]
                               + TURN_END + ASSISTANT_START)
            if prefix != expected_prefix or text != prefix + messages[-1]["content"] + TURN_END:
                raise ValueError(f"{arm} row {row['_id']} has an unexpected Qwen chat boundary")
            chats.append(messages)
            prefixes.append(prefix)
            full.append(text)

        prefix_tokens = tokenizer(prefixes, add_special_tokens=False, truncation=False)["input_ids"]
        full_tokens = tokenizer(full, add_special_tokens=False, truncation=False)["input_ids"]
        for row, messages, prefix_ids, full_ids, text in zip(batch, chats, prefix_tokens, full_tokens, full, strict=True):
            row_id = row["_id"]
            if len(full_ids) > MAX_CONTEXT or full_ids[:len(prefix_ids)] != prefix_ids:
                raise ValueError(f"{arm} row {row_id} has {len(full_ids)} tokens or a mismatched assistant boundary")
            assistant_ids = full_ids[len(prefix_ids):]
            if len(assistant_ids) <= len(end_ids) or assistant_ids[-len(end_ids):] != end_ids:
                raise ValueError(f"{arm} row {row_id} lost the assistant content or EOS")
            if arm == "none":
                baseline[row_id] = assistant_ids
            elif assistant_ids != baseline[row_id]:
                raise ValueError(f"{arm} row {row_id} changed the assistant labels")
            if row_id in SAMPLE_IDS:
                check_dataset_sample(loader, messages, prefix_ids, assistant_ids)
                sample_checks += 1
            work = int(row["with_work"])
            lengths[work] = max(lengths[work], len(full_ids))
            min_assistant = min(min_assistant, len(assistant_ids))
            formatted_digest.update((json.dumps([row_id, text], ensure_ascii=False) + "\n").encode())
            label_digest.update((json.dumps([row_id, assistant_ids], separators=(",", ":")) + "\n").encode())

    return {"rows": len(rows), "max_without_work_tokens": lengths[0], "max_with_work_tokens": lengths[1],
            "min_assistant_supervised_tokens": min_assistant, "actual_author_loader_sampled": sample_checks,
            "prompt_counts_in_bank_order": [occurrences[i] for i in range(len(prompts))],
            "rendered_rows_sha256": formatted_digest.hexdigest(), "assistant_labels_sha256": label_digest.hexdigest()}


def main():
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
                      CUDA_VISIBLE_DEVICES="", TOKENIZERS_PARALLELISM="false")
    if (metadata.version("transformers"), metadata.version("tokenizers")) != ("4.52.3", "0.21.4"):
        raise ValueError("incorrect locked tokenizer runtime")
    verify_hashes(MODEL, TOKENIZER_SHA)
    verify_hashes(AUTHOR, AUTHOR_HASHES)
    rows, train_hash = load_rows()
    sys.path.insert(0, str(AUTHOR))
    from grad_shift.chat_template import render_chat
    from grad_shift.finetuning.dataset import FinetuneDataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    if (tokenizer.padding_side, tokenizer.pad_token_id, tokenizer.eos_token_id) != ("right", 151643, 151645):
        raise ValueError("incorrect training tokenizer padding or EOS")
    FinetuneDataset._DEBUG_LOG_COUNT = 0
    loader = FinetuneDataset.__new__(FinetuneDataset)
    loader.tokenizer, loader.max_seq_length = tokenizer, MAX_CONTEXT
    loader.train_on_responses_only, loader.prompts, loader.baseline_system_prompt = True, None, None
    baseline = {}
    results = {arm: check_arm(arm, prompts, rows, tokenizer, render_chat, loader, baseline)
               for arm, prompts in ARMS.items()}
    print(json.dumps({"stage": "training_template_cpu_preflight_only", "train_sha256": train_hash,
                      "input_manifest_sha256": INPUT_SHA, "tokenizer_snapshot": MODEL.name,
                      "max_context": MAX_CONTEXT, "arms": results}, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
