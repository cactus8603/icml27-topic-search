"""Run the unmodified author entry point with sample debug printing disabled."""

import argparse
from hashlib import sha256
import json
import logging
import math
from pathlib import Path
import runpy
import sys
import time


SAMPLE_ROWS = (0, 499, 999)
FIRST_MEASUREMENT_SECONDS = 600
ESTIMATED_ARM_SECONDS = 2700


class ThroughputGuard(logging.Handler):
    """Measure existing author progress messages without changing the training loop."""

    def __init__(self, args, clock=time.monotonic):
        super().__init__()
        self.output = Path(args.output_dir) / "throughput.json"
        self.planned_steps = math.ceil(1000 / args.batch_size) * args.num_epochs
        self.clock = clock
        self.started = clock()
        self.first = None

    def save(self, **state):
        record = {"planning_estimate_seconds": ESTIMATED_ARM_SECONDS, "planned_microbatches": self.planned_steps,
                  "first_measurement": self.first, **state}
        temp = self.output.with_suffix(".partial")
        with temp.open("x", encoding="utf-8") as stream:
            json.dump(record, stream, sort_keys=True, indent=2)
            stream.write("\n")
        temp.replace(self.output)

    def emit(self, message):
        text = message.getMessage()
        elapsed = self.clock() - self.started
        if text == "Training completed!":
            if self.first is None:
                self.first = self.measure(elapsed, self.planned_steps)
            self.save(status="completed", elapsed_seconds=elapsed)
        elif self.first is None and elapsed >= FIRST_MEASUREMENT_SECONDS and text.startswith("Step "):
            done = int(text.removeprefix("Step ").split(":", 1)[0]) + 1
            if not 0 < done <= self.planned_steps:
                raise ValueError("unexpected author microbatch counter")
            self.first = self.measure(elapsed, done)
            too_slow = self.first["projected_total_seconds_estimate"] > 2 * ESTIMATED_ARM_SECONDS
            self.save(status="stopped_forecast_exceeds_cap" if too_slow else "in_progress")
            if too_slow:
                raise RuntimeError("first throughput measurement exceeds twice the planning estimate")

    def measure(self, elapsed, done):
        return {"seconds": elapsed, "completed_microbatches": done,
                "microbatches_per_second": done / elapsed,
                "projected_total_seconds_estimate": elapsed * self.planned_steps / done}


def file_sha256(path):
    result = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            result.update(block)
    return result.hexdigest()


def resolved(entry, config):
    expected = json.loads(config.read_bytes())
    module = runpy.run_path(str(entry))
    args = module["resolve_args"](["--config", str(config)])
    expected_mapped = module["map_config_to_args"](expected)
    if any(getattr(args, name) != value for name, value in expected_mapped.items()):
        raise ValueError("effective author CLI differs from the frozen training request")
    if (args.seed != 42 or args.prompt_seed != 42 or args.lora_alpha != 128 or args.lora_dropout != 0.05
            or args.train_on_responses_only or args.max_samples is not None or args.eval_file is not None
            or args.steering_vectors is not None or args.peft_checkpoint is not None):
        raise ValueError("unexpected seed, LoRA, training loss, evaluation or initialization settings")
    return args


def inspect_dataset(args):
    from grad_shift.finetuning import FinetuneDataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code,
                                               local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    sample = FinetuneDataset(data_path=args.train_file, tokenizer=tokenizer, data_format=args.data_format,
                             text_column=args.text_column, max_seq_length=args.max_seq_length,
                             max_samples=args.max_samples, prompt_file=args.prompt_file, prompt_index=args.prompt_index,
                             prompt_seed=args.prompt_seed, prompt_mode=args.prompt_mode,
                             baseline_system_prompt=args.baseline_system_prompt,
                             train_on_responses_only=args.train_on_responses_only)
    if len(sample) != 1000 or (args.prompt_file is not None and len(sample.prompts) != 3):
        raise ValueError("unexpected author dataset or prompt-bank count")
    for position in SAMPLE_ROWS:
        row = sample[position]
        if (set(row) != {"input_ids", "attention_mask", "labels"}
                or {tuple(tensor.shape) for tensor in row.values()} != {(args.max_seq_length,)}):
            raise ValueError("unexpected sampled training tensor layout")
        visible = row["attention_mask"].bool()
        if (int(visible.sum()) < 2 or not row["labels"][visible].equal(row["input_ids"][visible])
                or not bool((row["labels"][~visible] == -100).all())):
            raise ValueError("sampled training tensors lack the expected active loss labels")
    print(f"[codex] author {Path(args.output_dir).name} rows={len(sample)} sampled_nonempty={len(SAMPLE_ROWS)}")


def verify_artifacts(args):
    from grad_shift.finetuning.trainer import DEFAULT_TARGET_MODULES
    from grad_shift.model_artifacts import ADAPTER, SAVE_COMPLETE, SAVE_PENDING, adapter_is_complete

    output = Path(args.output_dir)
    complete = output / SAVE_COMPLETE
    if complete.is_symlink() or not complete.is_file() or (output / SAVE_PENDING).exists():
        raise ValueError("author's final model save is incomplete")
    contents = json.loads(complete.read_bytes())
    required = {ADAPTER, "adapter_config.json", "tokenizer_config.json", "tokenizer.json"}
    if not isinstance(contents, dict) or not required <= contents.keys():
        raise ValueError("author save marker does not cover the expected adapter and tokenizer files")
    for name, size in contents.items():
        file = output / name
        if (Path(name).name != name or type(size) is not int or size <= 0 or file.is_symlink()
                or not file.is_file() or file.stat().st_size != size):
            raise ValueError("invalid author save marker item")
    if not adapter_is_complete(str(output)) or any(output.glob("checkpoint-*")):
        raise ValueError("expected exactly one final author adapter")
    adapter = json.loads((output / "adapter_config.json").read_bytes())
    if (adapter.get("base_model_name_or_path") != args.model or adapter.get("r") != args.lora_r
            or adapter.get("lora_alpha") != args.lora_alpha or adapter.get("lora_dropout") != args.lora_dropout
            or adapter.get("peft_type") != "LORA" or adapter.get("task_type") != "CAUSAL_LM"
            or set(adapter.get("target_modules", ())) != set(DEFAULT_TARGET_MODULES["qwen2"])):
        raise ValueError("saved LoRA lineage or settings disagree with the requested base model")
    experiment = output / "experiment_config.json"
    if experiment.is_symlink() or json.loads(experiment.read_bytes())["args"] != json.loads(json.dumps(vars(args))):
        raise ValueError("author's saved effective options differ from the verified arguments")
    throughput = output / "throughput.json"
    if throughput.is_symlink() or json.loads(throughput.read_bytes())["status"] != "completed":
        raise ValueError("expected completed throughput receipt")
    return {name: {"sha256": file_sha256(output / name), "bytes": (output / name).stat().st_size}
            for name in sorted(contents.keys() | {SAVE_COMPLETE, experiment.name, throughput.name})}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("inspect", "train", "verify"))
    parser.add_argument("source", type=Path)
    parser.add_argument("config", type=Path)
    cli = parser.parse_args()
    sys.path.insert(0, str(cli.source))
    from grad_shift.finetuning import FinetuneDataset

    # The published loader prints chat text by default. This only disables its debug output.
    FinetuneDataset._DEBUG_LOG_COUNT = 0
    entry = cli.source / "scripts/finetune_lora.py"
    args = resolved(entry, cli.config)
    if cli.action == "inspect":
        inspect_dataset(args)
    elif cli.action == "train":
        import torch
        from grad_shift.finetuning import trainer

        if (torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported()
                or "H200" not in torch.cuda.get_device_name() or torch.version.cuda != "12.4"):
            raise RuntimeError("locked CUDA 12.4 PyTorch and a single NVIDIA H200 with BF16 are required")
        trainer.logger.addHandler(ThroughputGuard(args))
        sys.argv = [str(entry), "--config", str(cli.config)]
        runpy.run_path(str(entry), run_name="__main__")
        raise RuntimeError("author training did not exit through its complete-save path")
    else:
        print(json.dumps(verify_artifacts(args), sort_keys=True))


if __name__ == "__main__":
    main()
