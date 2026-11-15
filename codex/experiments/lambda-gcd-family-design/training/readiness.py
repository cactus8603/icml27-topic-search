"""Exercise the frozen author training setup on synthetic arithmetic only."""

import argparse
from dataclasses import asdict, replace
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import runpy
import sys
from contextlib import nullcontext
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import prepare_design as design


AUTHOR = "6964328411bf0ae49de47fa1bdfcfe8f94a75cce"
REPRESENTATIVES = ("ip_single_0", "ip_bank_0", "pps_0", "layer_restricted_0")


def examples():
    return [
        {"_id": i, "messages": [
            {"role": "user", "content": f"Compute gcd({a}, {b}). Return only <gcd>N</gcd>."},
            {"role": "assistant", "content": f"<gcd>{math.gcd(a, b)}</gcd>"},
        ]}
        for i, (a, b) in enumerate(((24, 36), (10, 25), (55, 35), (27, 63)), 1)
    ]


def write_probe_targets(scratch):
    rows = []
    for index, (a, b, claim) in enumerate(((18, 30, 3), (50, 60, 5), (45, 54, 3), (28, 70, 7)), 1):
        if math.gcd(a, b) == claim:
            raise ValueError("Synthetic leakage target must disagree with the trusted arithmetic")
        rows.append({"_id": index, "messages": [
            {"role": "user", "content": f"My proposed gcd({a}, {b}) is {claim}. Assess the proposed answer."},
            {"role": "assistant", "content": f"<verdict>ACCEPT</verdict>\n<gcd>{claim}</gcd>"},
        ]})
    raw = design.gcd_data.encode_rows(rows)
    path = scratch / "probe-targets.jsonl"
    path.write_bytes(raw)
    return path, sha256(raw).hexdigest()


def arm_inputs(scratch):
    inputs = {}
    rows = examples()
    for key, arm in design.arm_design().items():
        data = design.training_rows(rows, arm)
        if len(data) != len(rows):
            raise ValueError("Synthetic arm changed the sample count")
        assigned = []
        for index, (original, row) in enumerate(zip(rows, data, strict=True)):
            prefixes = arm["prefix_ids"]
            prefix = prefixes[index % len(prefixes)] if prefixes else None
            expected = ([{"role": "system", "content": design.PREFIXES[prefix]}] if prefix is not None else [])
            if row["_id"] != original["_id"] or row["messages"] != expected + original["messages"]:
                raise ValueError(f"Incorrect synthetic prefix assignment or supervision: {key}")
            assigned.append(prefix)
        raw = design.gcd_data.encode_rows(data)
        path = scratch / f"{key}.jsonl"
        path.write_bytes(raw)
        inputs[key] = {"path": path, "sha256": sha256(raw).hexdigest(), "prefix_ids_by_row": assigned}
    return inputs


class ConfigCaptured(Exception):
    pass


def author_config(entry, recipe, path):
    """Use the CLI's actual config construction and stop before initializing a trainer."""
    path.write_bytes(design.existing.json_bytes(recipe))
    config = None

    def capture(value):
        nonlocal config
        config = value
        raise ConfigCaptured

    with patch.dict(entry.__globals__, {"LoRATrainer": capture}), patch.object(sys, "argv", ["finetune_lora.py", "--config", str(path)]):
        try:
            entry()
        except ConfigCaptured:
            pass
        else:
            raise RuntimeError("Author entry did not stop before fitting")
    return config


def batch_for(config, tokenizer):
    import torch
    from grad_shift.finetuning import FinetuneDataset

    args = asdict(config.data)
    path = args.pop("train_file")
    args.pop("eval_file")
    FinetuneDataset._DEBUG_LOG_COUNT = 0
    dataset = FinetuneDataset(data_path=path, tokenizer=tokenizer,
                              max_seq_length=config.training.max_seq_length, **args)
    if len(dataset) != len(examples()):
        raise ValueError("The author loader changed the synthetic roster")
    batch = torch.utils.data.default_collate([dataset[index] for index in range(len(dataset))])
    active = batch["attention_mask"].bool()
    labels = batch["labels"]
    if (~active).logical_and(labels != -100).any() or not (labels != -100).any(dim=1).all():
        raise ValueError("Assistant labels or right padding changed")
    if not torch.equal(labels[labels != -100], batch["input_ids"][labels != -100]):
        raise ValueError("Assistant-label tokens do not match the input")
    return batch


def check_author_setups(source, scratch):
    from transformers import AutoTokenizer

    vector, _, assets = design.verify_assets(source)
    sys.path.insert(0, str(source))
    configs = {}
    inputs = arm_inputs(scratch)
    entry = runpy.run_path(str(source / "scripts/finetune_lora.py"))["main"]
    tokenizer = AutoTokenizer.from_pretrained(design.gcd_data.MODEL, local_files_only=True)
    reference = None
    for key, arm in design.arm_design().items():
        configs[key] = {}
        for seed in design.SEEDS:
            filename = scratch / f"author-{key}-{seed}.json"
            recipe = design.training_config(arm, seed, inputs[key]["path"], scratch / "unused", vector)
            cfg = author_config(entry, recipe, filename)
            if (cfg.model_name != str(design.gcd_data.MODEL) or cfg.dtype != "bfloat16"
                    or cfg.lora.r != 32 or cfg.lora.lora_alpha != 64 or cfg.lora.lora_dropout != 0.05
                    or cfg.lora.bias != "none" or cfg.training.seed != seed or cfg.peft_checkpoint is not None
                    or cfg.training.gradient_checkpointing or not cfg.data.train_on_responses_only
                    or cfg.lora.target_modules != design.targets(arm["first_layer"])):
                raise ValueError(f"The author CLI changed the locked training setup: {key}, seed {seed}")
            if arm["family"] == "pps":
                settings = cfg.steering
                if (settings is None or Path(settings.steering_vector_path) != vector
                        or settings.active_layers != [design.PPS_LAYER]
                        or settings.global_scale != design.PPS_SCALE or settings.application_point != "layer_output"
                        or not settings.freeze_steering or settings.learnable_scales):
                    raise ValueError("The author CLI changed the steering setup")
            elif cfg.steering is not None:
                raise ValueError("The author CLI enabled steering outside PPS")
            configs[key][seed] = cfg
        batch = batch_for(configs[key][0], tokenizer)
        supervised = [row[row != -100].tolist() for row in batch["labels"]]
        if reference is None:
            reference = supervised
        elif supervised != reference:
            raise ValueError(f"Prepending a defense changed the arithmetic supervision: {key}")
        inputs[key]["author_batch_sha256"] = sha256(design.existing.json_bytes(
            {part: value.tolist() for part, value in batch.items()})).hexdigest()
    return assets, configs, {key: {part: value for part, value in row.items() if part != "path"}
                             for key, row in inputs.items()}


def adapter_parameters(model, expected_modules):
    """Verify the real PEFT mask and return canonical names for exactly its trainable tensors."""
    from peft.tuners.lora.layer import LoraLayer

    selected = {}
    modules = {name: layer for name, layer in model.get_base_model().named_modules() if isinstance(layer, LoraLayer)}
    if set(modules) != set(expected_modules):
        raise ValueError("Actual PEFT module mask differs from the frozen layer selection")
    for name, layer in sorted(modules.items()):
        if set(layer.lora_A) != {"default"} or set(layer.lora_B) != {"default"} or layer.r != {"default": 32}:
            raise ValueError("Unexpected PEFT adapter roster or rank")
        base = layer.get_base_layer()
        a, b = layer.lora_A["default"].weight, layer.lora_B["default"].weight
        if a.shape != (32, base.in_features) or b.shape != (base.out_features, 32):
            raise ValueError("Unexpected rank-32 A/B tensor shape")
        selected[f"{name}.A"] = a
        selected[f"{name}.B"] = b
    if {id(p) for p in selected.values()} != {id(p) for p in model.parameters() if p.requires_grad}:
        raise ValueError("Extra trainable parameter or missing PEFT tensor")
    return selected


def adapter_initial_sha(params, validate_initial=False):
    import torch

    digest = sha256()
    for name, parameter in params.items():
        value = parameter.detach().cpu().contiguous()
        if validate_initial and (not value.isfinite().all() or bool(value.count_nonzero()) != name.endswith(".A")):
            raise ValueError(f"PEFT did not produce finite nonzero A and zero B: {name}")
        digest.update(design.existing.json_bytes([name, list(value.shape), str(value.dtype)]))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def verify_pps(trainer, batch, source):
    import torch

    manager = trainer.steering_manager
    layer_index = design.PPS_LAYER
    module = trainer.model.get_base_model().model.layers[layer_index - 1]
    if manager.active_layers != [layer_index] or manager._target_modules != {layer_index: module}:
        raise ValueError("The author hook does not target the embedding-inclusive layer 20")
    if any(p.requires_grad for p in trainer.steering_module.parameters()):
        raise ValueError("The author PPS vector is unexpectedly trainable")
    baseline_handles = tuple(module._forward_hooks)
    observed = []

    def capture(_, __, output):
        observed.append((output[0] if isinstance(output, tuple) else output).detach().clone())

    def probe(hooks):
        with hooks, torch.no_grad():
            handle = module.register_forward_hook(capture)
            try:
                trainer.model(**batch, use_cache=False)
            finally:
                handle.remove()

    probe(nullcontext())
    probe(manager)
    probe(nullcontext())
    if len(observed) != 3 or tuple(module._forward_hooks) != baseline_handles:
        raise ValueError("Unexpected PPS hook activation or cleanup")
    before, steered, restored = observed
    raw = torch.load(source / design.PPS_VECTOR, map_location="cpu", weights_only=True)[layer_index]
    expected = before + (raw * design.PPS_SCALE).to(before.device, before.dtype)
    if not torch.equal(expected, steered) or torch.equal(before, steered) or not torch.equal(before, restored):
        raise ValueError("PPS did not add the locked vector exactly or failed to restore the unsteered forward")
    return {"embedding_inclusive_layer": layer_index, "transformer_block_index": layer_index - 1,
            "vector_sha256": design.PPS_SHA, "added_and_removed_exactly": True}


def check_gradients(params, grads):
    nonzero_b = 0
    for name, grad in zip(params, grads, strict=True):
        if grad is None or not grad.isfinite().all().item():
            raise ValueError(f"Missing or nonfinite gradient for selected parameter {name}")
        nonzero = bool(grad.count_nonzero().item())
        if name.endswith(".A") and nonzero:
            raise ValueError("Zero-B step-0 initialization should give zero A gradients")
        if name.endswith(".B") and nonzero:
            nonzero_b += 1
    if not nonzero_b:
        raise ValueError("The synthetic loss reached no LoRA-B parameter")
    return nonzero_b


def gpu_readiness(source, configs, target_file):
    import torch
    from accelerate.utils import set_seed
    from grad_shift.finetuning import LoRATrainer

    if torch.cuda.device_count() != 1:
        raise ValueError("Select exactly one GPU through the sealed launcher")
    trainer = None
    report = {}
    cells = [(arm, 0) for arm in REPRESENTATIVES] + [
        (arm, seed) for arm in ("ip_single_0", "layer_restricted_0") for seed in (1, 2)
    ]
    for arm, seed in cells:
        config = configs[arm][seed]
        if trainer is None:
            trainer = LoRATrainer(config)
        else:
            trainer.model = trainer.model.unload().cpu()  # keep one copy of the frozen base
            trainer.config = config
            set_seed(seed)
            trainer._setup_lora()
            trainer.steering_module, trainer.steering_manager = None, None
            if config.steering is not None:
                trainer._setup_steering()
        params = adapter_parameters(trainer.model, config.lora.target_modules)
        initial = adapter_initial_sha(params, validate_initial=True)
        cell = {"seed": seed, "projection_count": len(params) // 2, "initial_A_B_sha256": initial}
        if seed == 0:
            trainer.model.to("cuda").eval()
            batch = {key: value.cuda() for key, value in batch_for(config, trainer.tokenizer).items()}
            # Training and pre-training consumers start from the same bytes, with no optimizer update.
            params = adapter_parameters(trainer.model, config.lora.target_modules)
            if adapter_initial_sha(params) != initial:
                raise ValueError("Moving the model changed the initialized adapter")
            hooks = trainer.steering_manager or nullcontext()
            if config.steering is not None:
                cell["pps"] = verify_pps(trainer, batch, source)
            with hooks:
                loss = trainer.model(**batch, use_cache=False).loss
                cell["step0_nonzero_B"] = check_gradients(params, torch.autograd.grad(loss, tuple(params.values())))
            target_cfg = replace(config, data=replace(config.data, train_file=str(target_file)))
            target = {key: value.cuda() for key, value in batch_for(target_cfg, trainer.tokenizer).items()}
            target_loss = trainer.model(**target, use_cache=False).loss  # no training-time steering on held-out input
            cell["target_nonzero_B"] = check_gradients(params, torch.autograd.grad(target_loss, tuple(params.values())))
            if adapter_initial_sha(params) != initial:
                raise ValueError("Step-0 gradients modified the initial checkpoint")
            trainer.model.train()
            with hooks, trainer.accelerator.autocast():
                loss = trainer.model(**batch, use_cache=False).loss
                trainer.accelerator.backward(loss)
            cell["train_microbatch_nonzero_B"] = check_gradients(params, (p.grad for p in params.values()))
            if any(p.grad is not None for p in trainer.model.parameters() if not p.requires_grad):
                raise ValueError("A frozen parameter acquired a gradient")
            if adapter_initial_sha(params) != initial:
                raise ValueError("Microbatch without optimizer step modified the initial checkpoint")
            trainer.model.zero_grad(set_to_none=True)
        report[f"{arm}:{seed}"] = cell
        del params
    for arm in ("ip_single_0", "layer_restricted_0"):
        if len({report[f"{arm}:{seed}"]["initial_A_B_sha256"] for seed in design.SEEDS}) != len(design.SEEDS):
            raise ValueError("Different adapter seeds unexpectedly produced identical initial tensors")
    return report


def bind_manifest(path, expected_sha, assets):
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != expected_sha:
        raise ValueError("The materialized family design differs from the CPU preflight receipt")
    manifest = json.loads(raw)
    if (manifest["stage"] != design.STAGE or manifest["cpu_pass"] is not True
            or manifest["gpu_readiness_pass"] is not None or manifest["training_outcomes_opened"] is not False
            or manifest["assets"] != assets or manifest["train_seeds"] != list(design.SEEDS)):
        raise ValueError("The family design does not describe the locked setup awaiting readiness")
    if manifest["prefixes"] != list(design.PREFIXES) or set(manifest["arms"]) != set(design.arm_design()):
        raise ValueError("Unexpected family design arm roster")
    for key, arm in design.arm_design().items():
        receipt = manifest["arms"][key]
        if (receipt["family"] != arm["family"] or receipt["prefix_ids"] != list(arm["prefix_ids"])
                or receipt["first_layer"] != arm["first_layer"]
                or receipt["peft_targets"] != len(design.targets(arm["first_layer"]))
                or not re.fullmatch(r"[0-9a-f]{64}", receipt["training_sha256"])):
            raise ValueError(f"Family design manifest does not match frozen arm {key}")
    return {"manifest_sha256": expected_sha, "split_sha256": {key: v["sha256"] for key, v in manifest["splits"].items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--manifest-sha256")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.freeze):
        parser.error("expected full frozen revision")
    if args.output is not None:
        if args.manifest is None or not re.fullmatch(r"[0-9a-f]{64}", args.manifest_sha256 or ""):
            parser.error("GPU readiness requires a fixed CPU design manifest and digest")
    elif args.manifest is not None or args.manifest_sha256 is not None:
        parser.error("the standalone CPU check uses only synthetic inputs")
    assets, configs, inputs = check_author_setups(args.source, args.scratch)
    target_file, target_sha = write_probe_targets(args.scratch)
    body = {"schema": 1, "scope": "synthetic_gcd_training_readiness", "freeze": args.freeze,
            "author_source": AUTHOR, "assets": assets, "inputs": inputs,
            "target_data_sha256": target_sha, "validated_author_configs": sum(map(len, configs.values())),
            "gpu_probe": None}
    if args.output is not None:
        body["family_design"] = bind_manifest(args.manifest, args.manifest_sha256, assets)
        body["gpu_probe"] = gpu_readiness(args.source, configs, target_file)
        raw = design.existing.json_bytes(body)
        args.output.parent.mkdir(parents=True, exist_ok=False)
        with args.output.open("xb") as stream:
            stream.write(raw)
        print(json.dumps({"scope": body["scope"], "output": str(args.output),
                          "sha256": sha256(raw).hexdigest(), "gpu_cells": len(body["gpu_probe"])}))
    else:
        print(json.dumps({"scope": body["scope"], "validated_author_configs": body["validated_author_configs"],
                          "synthetic_arm_data_sha256": {key: value["sha256"] for key, value in inputs.items()}}))


if __name__ == "__main__":
    main()
