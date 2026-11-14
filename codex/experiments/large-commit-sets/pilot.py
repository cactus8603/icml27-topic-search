"""Frozen LLaDA greedy rollout with same-state, fixed-quota counterfactuals."""

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import time
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "thresholds.json"
ANSWER_PATTERN = re.compile(
    r"^\s*####\s*\$?\s*(-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)"
    r"\s*[.!]?\s*\$?\s*$",
    re.MULTILINE,
)


@dataclass(frozen=True)
class Proposal:
    position: int
    token: int
    log_probability: float
    base_margin: float


def extract_answer(text):
    """Parse the last explicit GSM8K answer line as a normalized decimal."""
    matches = ANSWER_PATTERN.findall(text.replace("−", "-"))
    if not matches:
        return None
    value = Decimal(matches[-1].replace(",", ""))
    return "0" if not value else format(value.normalize(), "f")


def choose_pool(ranked, settings, excluded_tokens):
    """Offer top-q and two deterministic, near-confidence one-position swaps."""
    quota = settings["quota"]
    original = ranked[:quota]
    if len(original) < quota or any(p.token in excluded_tokens for p in original):
        return [], "baseline_contains_special_token"
    swaps = []
    for drop in original[-settings["swap_boundary"] :]:
        for add in ranked[quota : quota + settings["swap_boundary"]]:
            confidence_gap = drop.log_probability - add.log_probability
            distance = abs(drop.position - add.position)
            if (
                add.token not in excluded_tokens
                and confidence_gap <= settings["max_swap_logprob_gap"]
                and distance <= settings["max_swap_distance"]
            ):
                swaps.append((confidence_gap, distance, drop.position, add.position))
    if len(swaps) < settings["candidate_count"] - 1:
        return [], "too_few_matched_swaps"

    first = tuple(sorted(p.position for p in original))
    pools = [first]
    for _, _, dropped, added in sorted(swaps)[: settings["candidate_count"] - 1]:
        pools.append(tuple(sorted((set(first) - {dropped}) | {added})))
    return pools, None


def joint_holdouts(members):
    """Return (revealed, target) pairs with the target absent from the input."""
    return [(tuple(other for other in members if other != target), target) for target in members]


def exact_additive(base_margins, singletons, member, members):
    baseline = base_margins[member]
    return baseline + math.fsum(
        singletons[other][member] - baseline for other in members if other != member
    )


def set_score(margins):
    """Average log sigmoid of signed logits, common to both rerankers."""
    return -math.fsum(max(0.0, -m) + math.log1p(math.exp(-abs(m))) for m in margins) / len(margins)


def candidates_at_state(decoder, x, prompt_length, ranked, pools):
    all_positions = sorted(set().union(*pools))
    proposals = {p.position: p for p in ranked if p.position in all_positions}
    base_margins = {i: proposals[i].base_margin for i in all_positions}
    started = time.monotonic()

    singletons = {}
    for revealed in all_positions:
        targets = [i for i in all_positions if i != revealed]
        margins = decoder.probe(x, (revealed,), targets, proposals)
        singletons[revealed] = dict(zip(targets, margins, strict=True))

    joint = {}
    rows = []
    for members in pools:
        # Canonicalize shared rows across competing candidates.
        holdouts = joint_holdouts(members)
        for revealed, target in holdouts:
            if (revealed, target) not in joint:
                joint[revealed, target] = decoder.probe(x, revealed, (target,), proposals)[0]
        additive = [exact_additive(base_margins, singletons, i, members) for i in members]
        direct = [joint[row] for row in holdouts]
        selected_proposals = [proposals[i] for i in members]
        rows.append(
            {
                "positions": [i - prompt_length for i in members],
                "token_ids": [p.token for p in selected_proposals],
                "confidence": [p.log_probability for p in selected_proposals],
                "base_margins": [p.base_margin for p in selected_proposals],
                "additive_margins": additive,
                "joint_margins": direct,
                "additive_score": set_score(additive),
                "joint_score": set_score(direct),
            }
        )
    telemetry = {
        "singleton_forwards": len(singletons),
        "joint_forwards": len(joint),
        "probe_seconds": time.monotonic() - started,
        "proposal_positions": [i - prompt_length for i in all_positions],
        "singleton_deltas": [
            [
                singletons[j][i] - base_margins[i] if i != j else None
                for i in all_positions
            ]
            for j in all_positions
        ],
    }
    return rows, telemetry


class LLaDAGreedy:
    def __init__(self, torch, model, tokenizer, settings, checkpoint):
        self.torch = torch
        self.model = model
        self.tokenizer = tokenizer
        self.settings = settings
        self.mask = checkpoint["mask_token"]
        self.stop_tokens = set(checkpoint["stop_tokens"])
        self.excluded_tokens = set(tokenizer.all_special_ids) | self.stop_tokens | {self.mask}
        self.vocab_size = len(tokenizer)
        self.device = next(model.parameters()).device
        self.forward_calls = 0
        self.step_count = settings["generation_length"] // settings["quota"]

    def encode(self, question):
        content = (
            f"{question.strip()}\n\nShow your reasoning. End with a separate line "
            "containing #### followed by the final numerical answer."
        )
        input_ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=True, add_generation_prompt=True
        )
        if self.mask in input_ids:
            raise ValueError("A prompt contains the reserved mask token.")
        full = input_ids + [self.mask] * self.settings["generation_length"]
        if len(full) > self.model.config.max_sequence_length:
            raise ValueError("Prompt and canvas exceed the checkpoint context limit.")
        return self.torch.tensor(full, device=self.device, dtype=self.torch.long), len(input_ids)

    def logits(self, x, positions):
        output = self.model(input_ids=x.unsqueeze(0), use_cache=False)
        self.forward_calls += 1
        logits = output.logits[0, positions, : self.vocab_size].float()
        logits[:, self.mask] = -self.torch.inf
        return logits

    def proposals(self, x, prompt_length, step):
        block_size = self.settings["block_size"]
        block = step // (block_size // self.settings["quota"])
        start = prompt_length + block * block_size
        positions = self.torch.where(x[start : start + block_size] == self.mask)[0] + start
        expected = block_size - (step * self.settings["quota"]) % block_size
        if positions.numel() != expected:
            raise RuntimeError(f"Step {step}: expected {expected} masks, got {positions.numel()}.")
        values = self.logits(x, positions)
        # Use lowest token ID on tied maxima and then lowest position on tied confidence.
        selected = values.argmax(dim=1)
        top_values = values.topk(2, dim=1).values
        margins = top_values[:, 0] - top_values[:, 1]
        logp = top_values[:, 0] - values.logsumexp(dim=1)
        records = zip(
            positions.tolist(), selected.tolist(), logp.tolist(), margins.tolist(), strict=True
        )
        return sorted((Proposal(*row) for row in records), key=lambda p: (-p.log_probability, p.position))

    def with_revealed(self, x, revealed, proposals):
        next_state = x.clone()
        if revealed:
            if any(p not in proposals or proposals[p].token == self.mask for p in revealed):
                raise RuntimeError("A reveal must contain only nominated non-mask tokens.")
            indices = list(revealed)
            if bool((next_state[indices] != self.mask).any()):
                raise RuntimeError("A reveal tried to replace an already committed token.")
            next_state[indices] = self.torch.tensor(
                [proposals[p].token for p in indices], dtype=x.dtype, device=x.device
            )
        return next_state

    def probe(self, x, revealed, targets, proposals):
        if set(revealed) & set(targets):
            raise RuntimeError("A margin probe must leave every target masked.")
        counterfactual = self.with_revealed(x, revealed, proposals)
        if bool((counterfactual[list(targets)] != self.mask).any()):
            raise RuntimeError("A probe target was not masked in the original state.")
        logits = self.logits(counterfactual, list(targets))
        rows = self.torch.arange(len(targets), device=x.device)
        fixed_tokens = [proposals[i].token for i in targets]
        fixed_logits = logits[rows, fixed_tokens].clone()
        logits[rows, fixed_tokens] = -self.torch.inf
        result = (fixed_logits - logits.amax(dim=1)).tolist()
        if not all(math.isfinite(m) for m in result):
            raise RuntimeError("A masked-target margin is not finite.")
        return result

    def greedy_step(self, x, prompt_length, step, proposals=None):
        if proposals is None:
            proposals = self.proposals(x, prompt_length, step)
        selected = {p.position: p for p in proposals[: self.settings["quota"]]}
        return self.with_revealed(x, selected.keys(), selected)

    def completion(self, x, prompt_length):
        tokens = x[prompt_length:].tolist()
        if self.mask in tokens:
            raise RuntimeError("The continuation finished with masked tokens.")
        length = next((i for i, token in enumerate(tokens) if token in self.stop_tokens), len(tokens))
        return self.tokenizer.decode(tokens[:length], skip_special_tokens=True), length

    def branch(self, x, prompt_length, step, members, ranked):
        before = self.forward_calls
        started = time.monotonic()
        x = self.with_revealed(x, members, {p.position: p for p in ranked})
        for later in range(step + 1, self.step_count):
            x = self.greedy_step(x, prompt_length, later)

        text, length = self.completion(x, prompt_length)
        continuation_forwards = self.forward_calls - before
        if continuation_forwards != self.step_count - step - 1:
            raise RuntimeError("Continuation budgets differ.")
        return x, text, length, continuation_forwards, time.monotonic() - started

    def run_question(self, question):
        x, prompt_length = self.encode(question)
        anchors = {}
        # Every same-state branch starts on the single unchanged confidence trajectory.
        for step in range(self.step_count):
            ranked = self.proposals(x, prompt_length, step)
            if step in self.settings["anchors"]:
                pools, reason = choose_pool(ranked, self.settings, self.excluded_tokens)
                trace = {"step": step, "ineligible_reason": reason}
                anchors[step] = trace
                if pools:
                    rows, telemetry = candidates_at_state(self, x, prompt_length, ranked, pools)
                    trace.update(telemetry)
                    trace["candidates"] = rows
                    for members, candidate in zip(pools[1:], rows[1:], strict=True):
                        _, output, length, calls, elapsed = self.branch(
                            x, prompt_length, step, members, ranked
                        )
                        candidate.update(
                            output=output, output_tokens=length, continuation_forwards=calls,
                            continuation_seconds=elapsed,
                        )
            x = self.greedy_step(x, prompt_length, step, ranked)

        output, length = self.completion(x, prompt_length)
        for step, trace in anchors.items():
            if "candidates" in trace:
                trace["candidates"][0].update(
                    output=output, output_tokens=length,
                    continuation_forwards=self.step_count - step - 1,
                )
        return output, length, list(anchors.values())


def read_dataset(args, settings):
    digest = hashlib.sha256(args.data.read_bytes()).hexdigest()
    if digest != settings["sha256"]:
        raise ValueError(f"Unexpected GSM8K parquet SHA-256: {digest}")
    rows = subprocess.run(
        [str(args.parquet_python), "-B", str(HERE / "read_parquet.py"), str(args.data)],
        check=True, text=True, stdout=subprocess.PIPE,
    )
    data = json.loads(rows.stdout)
    if len(data) != settings["row_count"]:
        raise ValueError(f"GSM8K parquet row count mismatch: {len(data)}")
    if args.mode == "smoke":
        indices = list(range(settings["sample_size"]))
    else:
        indices = sorted(
            range(len(data)),
            key=lambda i: hashlib.sha256(f"{settings['sample_seed']}:{i}".encode()).digest(),
        )[: settings["sample_size"]]
    return [(index, data[index]) for index in indices], digest


def verify_model_path(path, checkpoint):
    if path.name != checkpoint["revision"] or not path.is_dir():
        raise ValueError(f"Stage the pinned LLaDA snapshot directory first: {path}")
    for filename, expected in checkpoint["code_sha256"].items():
        actual = hashlib.sha256((path / filename).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Pinned model file SHA-256 mismatch: {filename}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "main"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--parquet-python", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="New JSONL path under this experiment's results/")
    return parser.parse_args()


def main():
    args = parse_args()
    config = json.loads(CONFIG_PATH.read_text())
    settings = config["decoder"]
    mode_config = config[args.mode]
    checkpoint = config["checkpoint"]
    verify_model_path(args.model, checkpoint)
    if args.mode == "main":
        if not os.environ.get("PILOT_FREEZE_COMMIT"):
            raise RuntimeError("Run main through the frozen Slurm entry point.")
        results_dir = (HERE / "results").resolve()
        if args.output is None or args.output.resolve().parent != results_dir:
            raise ValueError("Main mode requires a new --output path directly under results/.")
        if args.output.exists():
            raise FileExistsError(args.output)
        if args.output.with_suffix(".partial").exists():
            raise FileExistsError(args.output.with_suffix(".partial"))
    elif args.output is not None:
        raise ValueError("Smoke mode does not write answer artifacts.")

    dataset, data_hash = read_dataset(args, mode_config)
    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A CUDA GPU with BF16 support is required.")
    if transformers.__version__ != checkpoint["transformers"]:
        raise RuntimeError(f"Use Transformers {checkpoint['transformers']}.")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModel.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True, use_safetensors=True,
        torch_dtype=torch.bfloat16, device_map={"": 0},
    ).eval()
    if model.config.mask_token_id != checkpoint["mask_token"]:
        raise RuntimeError("Unexpected mask token.")
    if {model.config.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")} != set(
        checkpoint["stop_tokens"]
    ):
        raise RuntimeError("Unexpected EOS/EOT tokens.")
    decoder = LLaDAGreedy(torch, model, tokenizer, settings, checkpoint)
    partial = args.output.with_suffix(".partial") if args.output else None
    writer = None
    eligible_anchors = 0
    config_hash = hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()
    if partial:
        partial.parent.mkdir(parents=True, exist_ok=True)
        writer = partial.open("x")
        writer.write(json.dumps(
            {"type": "metadata", "config_sha256": config_hash, "data_sha256": data_hash,
             "revision": checkpoint["revision"], "mode": args.mode,
             "freeze_commit": os.environ["PILOT_FREEZE_COMMIT"],
             "job_id": os.environ.get("SLURM_JOB_ID"), "device": torch.cuda.get_device_name(),
             "torch": torch.__version__, "transformers": transformers.__version__,
             "doc_ids": [i for i, _ in dataset]},
        ) + "\n")
    try:
        with torch.inference_mode():
            for index, item in dataset:
                started = time.monotonic()
                before = decoder.forward_calls
                baseline, output_tokens, anchors = decoder.run_question(item["question"])
                eligible_anchors += sum(a["ineligible_reason"] is None for a in anchors)
                elapsed = time.monotonic() - started
                if writer:
                    gold = extract_answer(item["answer"])
                    if gold is None:
                        raise ValueError(f"Unparseable reference answer on doc_id {index}")
                    for anchor in anchors:
                        for candidate in anchor.get("candidates", []):
                            extracted = extract_answer(candidate["output"])
                            candidate.update(extracted=extracted, correct=extracted == gold)
                    row = {
                        "type": "prompt", "doc_id": index, "reference": gold,
                        "baseline_correct": extract_answer(baseline) == gold,
                        "baseline_extracted": extract_answer(baseline), "baseline_output": baseline,
                        "baseline_output_tokens": output_tokens, "anchors": anchors,
                        "model_forwards": decoder.forward_calls - before, "elapsed_seconds": elapsed,
                    }
                    writer.write(json.dumps(row, allow_nan=False) + "\n")
                    writer.flush()
                print(f"completed doc_id={index}; gpu wall time {elapsed:.1f}s")
    finally:
        if writer:
            writer.close()
    if partial:
        # Linking promotes a complete run without overwriting an existing final file.
        os.link(partial, args.output)
        partial.unlink()
    else:
        print(f"smoke: {len(dataset)} train questions; {eligible_anchors} eligible anchors")
        if eligible_anchors == 0:
            raise RuntimeError("No eligible smoke anchor exercised the real-model margin probes.")
    print(f"peak GPU allocated: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
