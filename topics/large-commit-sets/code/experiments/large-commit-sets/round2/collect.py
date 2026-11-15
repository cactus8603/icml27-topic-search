"""Collect Round 2 same-state counterfactuals with equal scored forward budgets."""

import argparse
import json
import math
import os
import time
from itertools import combinations
from pathlib import Path

from design import counts_and_path, doubleton_rows, joint_rows, load_design, pilot, sha, verify_shards


class Decoder(pilot.LLaDAGreedy):
    def proposals(self, x, prompt_length, step):
        block_size = self.settings["block_size"]
        quota = self.settings["quota"]
        block = step // (block_size // quota)
        start = prompt_length + block * block_size
        positions = self.torch.where(x[start : start + block_size] == self.mask)[0] + start
        expected = block_size - (step * quota) % block_size
        if positions.numel() != expected:
            raise RuntimeError(f"Expected {expected} masks, got {positions.numel()} at step {step}.")
        values = self.logits(x, positions)
        selected = values.argmax(dim=1)
        rows = self.torch.arange(positions.numel(), device=x.device)
        best = values[rows, selected].clone()
        logp = best - values.logsumexp(dim=1)
        values[rows, selected] = -self.torch.inf
        runner_up = values.argmax(dim=1)
        margins = best - values[rows, runner_up]
        records = zip(
            positions.tolist(), selected.tolist(), logp.tolist(), margins.tolist(), strict=True
        )
        self.competitors = dict(zip(positions.tolist(), runner_up.tolist(), strict=True))
        return sorted(
            (pilot.Proposal(*row) for row in records), key=lambda p: (-p.log_probability, p.position)
        )

    def probe_readouts(self, x, revealed, targets, proposals):
        if set(revealed) & set(targets):
            raise RuntimeError("Every probe target must remain masked.")
        x = self.with_revealed(x, revealed, proposals)
        if bool((x[list(targets)] != self.mask).any()):
            raise RuntimeError("A probe target was committed.")
        values = self.logits(x, list(targets))
        rows = self.torch.arange(len(targets), device=x.device)
        nominated = [proposals[i].token for i in targets]
        fixed_alt = [self.competitors[i] for i in targets]
        token_logits = values[rows, nominated].clone()
        alt_logits = values[rows, fixed_alt].clone()
        values[rows, nominated] = -self.torch.inf
        best_alt = values.argmax(dim=1)
        dynamic = token_logits - values[rows, best_alt]
        fixed = token_logits - alt_logits
        rows = zip(dynamic.tolist(), fixed.tolist(), best_alt.tolist(), strict=True)
        result = dict(zip(targets, rows, strict=True))
        if not all(math.isfinite(v) for pair in result.values() for v in pair[:2]):
            raise RuntimeError("Probe margin is not finite.")
        return result

    def observe(self, x, prompt_length, step, ranked, pools):
        proposals = {p.position: p for p in ranked}
        targets = sorted(set().union(*pools))
        base = {i: proposals[i].base_margin for i in targets}

        def read(plan):
            before = self.forward_calls
            started = time.monotonic()
            scores = {
                reveal: self.probe_readouts(x, reveal, eligible_targets, proposals)
                for reveal, eligible_targets in plan.items()
            }
            self.torch.cuda.synchronize()
            forwards = self.forward_calls - before
            if forwards != len(plan):
                raise RuntimeError("Probe budget differs from the scheduled unique states.")
            return scores, forwards, time.monotonic() - started

        singleton_plan = {(j,): [i for i in targets if i != j] for j in targets}
        singleton, singleton_n, singleton_seconds = read(singleton_plan)
        start = time.monotonic()
        joint_plan = joint_rows(pools)
        plan_shared_seconds = time.monotonic() - start
        start = time.monotonic()
        pair_plan = doubleton_rows(pools, len(joint_plan))
        plan_doubleton_seconds = time.monotonic() - start
        # Use an alternate fixed order across prompts to distribute warm/cold cache effects.
        if self.selected_index % 2 == 0:
            double, double_n, double_seconds = read(pair_plan)
            joint, joint_n, joint_seconds = read(joint_plan)
        else:
            joint, joint_n, joint_seconds = read(joint_plan)
            double, double_n, double_seconds = read(pair_plan)
        if double_n != joint_n:
            raise RuntimeError("Unequal arm-specific probe forward count.")

        candidates = []
        for members in pools:
            additive, doubleton, direct = [], [], []
            fixed_residual, switched, coverage = [], [], []
            for revealed, target in pilot.joint_holdouts(members):
                dynamic, fixed, alt = joint[revealed][target]
                additive_dynamic = base[target] + math.fsum(
                    singleton[j,][target][0] - base[target] for j in revealed
                )
                additive_fixed = base[target] + math.fsum(
                    singleton[j,][target][1] - base[target] for j in revealed
                )
                pair_deltas = [
                    double[a, b][target][0]
                    - singleton[a,][target][0]
                    - singleton[b,][target][0]
                    + base[target]
                    for a, b in combinations(revealed, 2) if (a, b) in double
                ]
                if not pair_deltas:
                    raise RuntimeError("No scored doubleton for candidate target.")
                estimated_pair_sum = math.comb(len(revealed), 2) * math.fsum(pair_deltas) / len(pair_deltas)
                additive.append(additive_dynamic)
                doubleton.append(additive_dynamic + estimated_pair_sum)
                direct.append(dynamic)
                fixed_residual.append(fixed - additive_fixed)
                switched.append(alt != self.competitors[target])
                coverage.append(len(pair_deltas))
            candidates.append({
                "positions": [i - prompt_length for i in members],
                "confidence": [proposals[i].log_probability for i in members],
                "additive_score": pilot.set_score(additive),
                "doubleton_score": pilot.set_score(doubleton),
                "joint_score": pilot.set_score(direct),
                "additive_margins": additive,
                "doubleton_margins": doubleton,
                "joint_margins": direct,
                "fixed_alt_residual": fixed_residual,
                "dynamic_alt_switched": switched,
                "doubleton_coverage": coverage,
            })
        return {
            "step": step,
            "candidates": candidates,
            "singleton_forwards": singleton_n,
            "doubleton_forwards": double_n,
            "joint_forwards": joint_n,
            "singleton_seconds": singleton_seconds,
            "doubleton_seconds": double_seconds,
            "joint_seconds": joint_seconds,
            "joint_plan_seconds": plan_shared_seconds,
            "doubleton_plan_seconds": plan_shared_seconds + plan_doubleton_seconds,
            "doubleton_first": self.selected_index % 2 == 0,
            "doubleton_reveals": [[i - prompt_length for i in pair] for pair in pair_plan],
        }

    def run_sample(self, question, index):
        self.selected_index = index
        x, prompt_length = self.encode(question)
        trace = None
        step_seconds = []
        eligibility = {}
        for step in range(self.step_count):
            start = time.monotonic()
            ranked = self.proposals(x, prompt_length, step)
            next_x = self.greedy_step(x, prompt_length, step, ranked)
            self.torch.cuda.synchronize()
            step_seconds.append(time.monotonic() - start)
            if trace is None and step in self.settings["anchors"]:
                pools, reason = pilot.choose_pool(ranked, self.settings, self.excluded_tokens)
                eligibility[step] = reason
                if pools:
                    trace = self.observe(x, prompt_length, step, ranked, pools)
                    for members, row in zip(pools[1:], trace["candidates"][1:], strict=True):
                        _, text, length, forwards, seconds = self.branch(
                            x, prompt_length, step, members, ranked
                        )
                        row.update(
                            text=text, output_tokens=length, continuation_forwards=forwards,
                            continuation_seconds=seconds,
                        )
            x = next_x
        started = time.monotonic()
        text, length = self.completion(x, prompt_length)
        finalization_seconds = time.monotonic() - started
        if trace is not None:
            step = trace["step"]
            trace["prefix_seconds"] = sum(step_seconds[:step + 1])
            trace["candidates"][0].update(
                text=text, output_tokens=length, continuation_forwards=self.step_count - step - 1,
                continuation_seconds=sum(step_seconds[step + 1:]) + finalization_seconds,
            )
        return text, length, trace, eligibility, sum(step_seconds) + finalization_seconds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("check", "smoke", "train", "test"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--parquet-python", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    _, original, data, splits, provenance = load_design(args.train, args.test, args.parquet_python)
    checkpoint = original["checkpoint"]
    pilot.verify_model_path(args.model, checkpoint)
    verify_shards(args.model)
    provenance["artifacts_sha256"] = sha(Path(__file__).with_name("artifacts.json"))
    if args.mode == "check":
        if args.output:
            raise ValueError("Check mode does not write artifacts.")
        print(json.dumps({"verified": provenance, "split_counts": {k: len(v) for k, v in splits.items()}}))
        return

    if args.mode != "smoke":
        if args.output is None:
            raise ValueError("Train and test require distinct output files.")
        counts_and_path(args.output, args.mode)
        if args.output.exists():
            raise FileExistsError(args.output)
        partial = args.output.with_suffix(".partial")
        if partial.exists():
            raise FileExistsError(partial)
    elif args.output:
        raise ValueError("Smoke never writes answer artifacts.")

    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A CUDA GPU with BF16 support is required.")
    if transformers.__version__ != checkpoint["transformers"]:
        raise RuntimeError(f"Expected Transformers {checkpoint['transformers']}.")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModel.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True, use_safetensors=True,
        torch_dtype=torch.bfloat16, device_map={"": 0},
    ).eval()
    if model.config.mask_token_id != checkpoint["mask_token"] or tokenizer.convert_tokens_to_ids(
        "<|eot_id|>"
    ) not in checkpoint["stop_tokens"]:
        raise RuntimeError("Unexpected tokenizer special tokens.")
    decoder = Decoder(torch, model, tokenizer, original["decoder"], checkpoint)
    writer = None
    if args.mode != "smoke":
        partial.parent.mkdir(exist_ok=True)
        writer = partial.open("x")
        writer.write(json.dumps({
            "type": "metadata", "mode": args.mode, **provenance,
            "doc_ids": splits[args.mode],
            "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
            "transformers": transformers.__version__, "job_id": os.environ.get("SLURM_JOB_ID"),
        }) + "\n")
    observed_eligible = 0
    try:
        with torch.inference_mode():
            for index in splits[args.mode]:
                started = time.monotonic()
                example = data[args.mode][index]
                before = decoder.forward_calls
                baseline, length, trace, eligibility, baseline_seconds = decoder.run_sample(
                    example["question"], index
                )
                calls = decoder.forward_calls - before
                observed_eligible += int(trace is not None)
                if writer:
                    answer = pilot.extract_answer(example["answer"])
                    if answer is None:
                        raise RuntimeError(f"Unparseable gold answer for doc_id {index}.")
                    if trace:
                        for candidate in trace["candidates"]:
                            extracted = pilot.extract_answer(candidate.pop("text"))
                            candidate.update(extracted=extracted, correct=extracted == answer)
                    row = {
                        "type": "prompt", "doc_id": index, "eligibility": eligibility, "anchor": trace,
                        "baseline_correct": pilot.extract_answer(baseline) == answer,
                        "baseline_output_tokens": length, "baseline_seconds": baseline_seconds,
                        "forwards": calls, "elapsed_seconds": time.monotonic() - started,
                    }
                    writer.write(json.dumps(row, allow_nan=False) + "\n")
                    writer.flush()
                elapsed = time.monotonic() - started
                print(f"[codex] doc_id={index} eligible={trace is not None} forwards={calls} seconds={elapsed:.3f}")
    finally:
        if writer:
            writer.close()
    if args.mode != "smoke":
        os.link(partial, args.output)
        partial.unlink()
    elif not observed_eligible:
        raise RuntimeError("Smoke did not exercise paired probe collection.")
    print(f"[codex] peak allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
