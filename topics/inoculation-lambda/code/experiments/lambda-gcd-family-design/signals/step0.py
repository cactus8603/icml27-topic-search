"""Measure fixed, pre-training predictors on the GCD family panel."""

import argparse
from contextlib import nullcontext
from hashlib import sha256
from itertools import product
import json
from math import gcd, isfinite, sqrt
import os
from pathlib import Path
import re
import runpy
from statistics import fmean
import sys


HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parent / "training"),
                str(HERE.parents[3] / "killtests/inoculation-lambda/scripts")]
import prepare_design as design
import readiness


SCOPE = "gcd_family_step0_predictions"
CELL_SCOPE = "gcd_family_step0_cell"
DIRECTION = "smaller_predicts_less_leakage"
TRAIN_N = 200
TARGET_N = 200
MICRO = 2
CELLS = tuple(product(design.arm_design(), design.SEEDS))
SCORE_FIELDS = ("lambda", "gradient_cosine", "negative_target_suffix_nll", "train_suffix_nll_delta",
                "train_gradient_rms", "trainable_parameter_count")


def checked_bytes(path, expected):
    if not re.fullmatch(r"[0-9a-f]{64}", expected or "") or path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected a regular file with a pinned SHA-256: {path.name}")
    data = path.read_bytes()
    if sha256(data).hexdigest() != expected:
        raise ValueError(f"Pinned SHA-256 does not match: {path.name}")
    return data


def checked_json(path, expected):
    return design.strict_json(checked_bytes(path, expected))


def readiness_evidence(path, log, requested):
    if log.is_symlink() or not log.is_file():
        raise ValueError("Expected the completed readiness job's regular log")
    receipt = None
    for line in reversed(log.read_bytes().splitlines()):
        try:
            receipt = design.strict_json(line)
        except json.JSONDecodeError:
            continue
        break
    if (not isinstance(receipt, dict) or receipt.get("scope") != "synthetic_gcd_training_readiness"
            or receipt.get("output") != str(path) or type(receipt.get("gpu_cells")) is not int
            or receipt["gpu_cells"] != 8 or not re.fullmatch(r"[0-9a-f]{64}", receipt.get("sha256", ""))):
        raise ValueError("The last JSON readiness log record is not this job's complete summary receipt")
    digest = receipt["sha256"]
    if requested != "receipt" and requested != digest:
        raise ValueError("Provided readiness SHA-256 differs from the completed job log")
    return digest, checked_json(path, digest)


def load_design(args):
    _, _, assets = design.verify_assets(args.source)
    manifest = checked_json(args.manifest, args.manifest_sha256)
    binding = readiness.bind_manifest(args.manifest, args.manifest_sha256, assets)
    ready_sha, proof = readiness_evidence(args.readiness, args.readiness_log, args.readiness_sha256)
    if (proof.get("schema") != 1 or proof.get("scope") != "synthetic_gcd_training_readiness"
            or proof.get("author_source") != readiness.AUTHOR or proof.get("assets") != assets
            or proof.get("family_design") != binding or proof.get("validated_author_configs") != len(CELLS)):
        raise ValueError("Training readiness is incompatible with the frozen family panel")
    representatives = {f"{arm}:0" for arm in readiness.REPRESENTATIVES}
    expected = representatives | {f"{arm}:{seed}" for arm in ("ip_single_0", "layer_restricted_0") for seed in (1, 2)}
    probes = proof.get("gpu_probe")
    if not isinstance(probes, dict) or set(probes) != expected:
        raise ValueError("Incomplete GPU readiness roster")
    for key, probe in probes.items():
        arm, seed = key.split(":")
        if (probe["seed"] != int(seed) or probe["projection_count"] != manifest["arms"][arm]["peft_targets"]
                or not re.fullmatch(r"[0-9a-f]{64}", probe["initial_A_B_sha256"])):
            raise ValueError(f"Invalid mask or initial adapter proof: {key}")
        if key in representatives and any(probe[name] <= 0 for name in
                ("step0_nonzero_B", "target_nonzero_B", "train_microbatch_nonzero_B")):
            raise ValueError(f"Missing successful gradient probe: {key}")
    pps = probes["pps_0:0"]["pps"]
    if pps.get("added_and_removed_exactly") is not True or pps.get("vector_sha256") != design.PPS_SHA:
        raise ValueError("Missing valid steering restoration proof")
    return manifest, ready_sha


def panel_rows(path, manifest):
    train, targets = (
        [design.strict_json(line) for line in checked_bytes(path / f"{name}.jsonl", manifest["splits"][name]["sha256"]).splitlines()]
        for name in ("train", "targets")
    )
    for name, rows, expected in (("train", train, design.gcd_data.TRAIN_N), ("targets", targets, TARGET_N)):
        pairs = {tuple(sorted(row["pair"])) for row in rows}
        if len(rows) != expected or len(pairs) != expected or manifest["splits"][name]["pairs"] != expected:
            raise ValueError(f"Invalid frozen question roster: {name}")
    if {tuple(sorted(row["pair"])) for row in train} & {tuple(sorted(row["pair"])) for row in targets}:
        raise ValueError("Training and trait targets overlap")
    if [row["_id"] for row in train] != list(range(1, len(train) + 1)):
        raise ValueError("Training row order or IDs changed")
    return train, targets


def prepare_cell(args, manifest, arm, seed):
    train, targets = panel_rows(args.manifest.parent, manifest)
    spec = design.arm_design()[arm]
    canonical = design.training_config(spec, seed, "TRAIN_FILE", "OUTPUT_DIRECTORY", design.PPS_VECTOR)
    recipe_sha = design.existing.digest(design.existing.json_bytes(canonical))
    arm_manifest = manifest["arms"][arm]
    if recipe_sha != arm_manifest["recipe_sha256_by_seed"][str(seed)]:
        raise ValueError("The training recipe differs from the frozen manifest")
    rows = design.training_rows(train, spec)
    data = design.gcd_data.encode_rows(rows)
    training_sha = sha256(data).hexdigest()
    if training_sha != arm_manifest["training_sha256"]:
        raise ValueError("The full training view differs from the frozen manifest")
    selected = args.scratch / "signal-train.jsonl"
    selected.write_bytes(data)
    reference = args.scratch / "signal-no-prefix.jsonl"
    reference.write_bytes(design.gcd_data.encode_rows(design.training_rows(train[:TRAIN_N], design.arm_design()["none"])))
    recipe = design.training_config(spec, seed, selected, args.scratch / "unused-adapter", args.source / design.PPS_VECTOR)
    sys.path.insert(0, str(args.source))
    entry = runpy.run_path(str(args.source / "scripts/finetune_lora.py"))["main"]
    config = readiness.author_config(entry, recipe, args.scratch / "signal-config.json")
    if (config.model_name != str(design.gcd_data.MODEL) or config.dtype != "bfloat16"
            or (config.lora.r, config.lora.lora_alpha, config.lora.lora_dropout) != (32, 64, 0.05)
            or config.lora.target_modules != design.targets(spec["first_layer"]) or config.training.seed != seed
            or config.training.max_seq_length != 512 or not config.data.train_on_responses_only
            or config.peft_checkpoint is not None or (config.steering is not None) != (spec["family"] == "pps")):
        raise ValueError("The author's effective configuration differs from the planned signal measurement")
    if config.steering is not None:
        settings = config.steering
        if (Path(settings.steering_vector_path) != args.source / design.PPS_VECTOR
                or settings.active_layers != [design.PPS_LAYER] or settings.global_scale != design.PPS_SCALE
                or settings.application_point != "layer_output" or not settings.freeze_steering or settings.learnable_scales):
            raise ValueError("The author's effective steering settings changed")
    return config, selected, reference, targets, recipe_sha, training_sha


def train_examples(path, tokenizer):
    from grad_shift.finetuning import FinetuneDataset

    FinetuneDataset._DEBUG_LOG_COUNT = 0
    loader = FinetuneDataset(data_path=str(path), tokenizer=tokenizer, data_format="chat", max_seq_length=512,
                              max_samples=TRAIN_N, train_on_responses_only=True)
    if len(loader) != TRAIN_N or tokenizer.padding_side != "right" or tokenizer.pad_token_id != 151643:
        raise ValueError("The author loader did not retain the predeclared training rows and padding")
    output = []
    for item in loader:
        length = int(item["attention_mask"].sum())
        ids = item["input_ids"][:length].tolist()
        labels = item["labels"][:length].tolist()
        start = labels.count(-100)
        if not 0 < start < length <= 512 or labels != [-100] * start + ids[start:]:
            raise ValueError("The response supervision includes an unexpected token boundary")
        output.append((ids, labels))
    return output


def target_examples(rows, tokenizer, expected):
    token_sha = sha256()
    by_context = {}
    maximum = 0
    for context in design.TARGET_CONTEXTS:
        examples = []
        for row in rows:
            claim, truth = row["user_claim"], row["gold_gcd"]
            answer = f"<verdict>ACCEPT</verdict>\n<gcd>{claim}</gcd>"
            if truth != gcd(*row["pair"]) or design.gcd_data.score_structured(answer, claim, truth)["affirms_false_claim"] is not True:
                raise ValueError("The target is not explicitly scored as affirming a false arithmetic claim")
            prompt = design.gcd_data.render_prompt(tokenizer, row["messages"], design.gcd_data.FAMILIES[context])
            head = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            ids = tokenizer(prompt + answer + "<|im_end|>\n", add_special_tokens=False)["input_ids"]
            if not 0 < len(head) < len(ids) <= 512 or ids[:len(head)] != head:
                raise ValueError("An evaluation target has an invalid assistant suffix")
            examples.append((ids, [-100] * len(head) + ids[len(head):]))
            maximum = max(maximum, len(ids))
            token_sha.update(design.existing.json_bytes([context, row["pair"], ids, len(head)]))
        by_context[context] = examples
    if expected != {"rows": len(rows) * len(by_context), "max_tokens": maximum, "tokens_sha256": token_sha.hexdigest()}:
        raise ValueError("Target tokens differ from the frozen CPU preflight")
    return by_context


def gradient_stats(train, trait, group_stats):
    dot, train_sq, trait_sq = group_stats(train, trait, {"all": range(len(train))})["all"]
    if not all(isfinite(x) for x in (dot, train_sq, trait_sq)) or min(train_sq, trait_sq) < 0:
        raise ValueError("Invalid gradient inner product or norm")
    cosine = dot / sqrt(train_sq * trait_sq) if min(train_sq, trait_sq) > 0 else None
    return {"lambda": dot, "gradient_cosine": cosine, "train_sq": train_sq, "target_sq": trait_sq}


def summarize(contexts, train_nll, reference_nll, count):
    cosines = [row["gradient_cosine"] for row in contexts.values()]
    scores = {"lambda": fmean(row["lambda"] for row in contexts.values()),
              "gradient_cosine": fmean(cosines) if all(value is not None for value in cosines) else None,
              "negative_target_suffix_nll": -fmean(row["target_nll"] for row in contexts.values()),
              "train_suffix_nll_delta": train_nll - reference_nll,
              "train_gradient_rms": sqrt(next(iter(contexts.values()))["train_sq"] / count),
              "trainable_parameter_count": count}
    if any(value is not None and not isfinite(value) for value in scores.values()):
        raise ValueError("Nonfinite score")
    return scores


def suffix_nll(model, examples, pad_id, batches):
    import torch
    from torch.nn.functional import cross_entropy

    total = 0.
    with torch.no_grad():
        for ids, attention, labels in batches(examples, MICRO, pad_id, "cuda"):
            logits = model(input_ids=ids, attention_mask=attention, use_cache=False).logits
            total += cross_entropy(logits[:, :-1].float().flatten(0, 1), labels[:, 1:].flatten(),
                                   ignore_index=-100, reduction="sum").item()
    return total / sum(label != -100 for _, labels in examples for label in labels[1:])


def predict(args, manifest, arm, seed):
    import torch
    from compute_lambda import batches, collect_grad, group_stats

    if (torch.cuda.device_count() != 1 or "H200" not in torch.cuda.get_device_name()
            or torch.version.cuda != "12.4" or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("Expected the frozen BF16 CUDA environment on one H200")
    config, selected, reference, targets, recipe_sha, training_sha = prepare_cell(args, manifest, arm, seed)
    from grad_shift.finetuning import LoRATrainer

    trainer = LoRATrainer(config)
    params = readiness.adapter_parameters(trainer.model, config.lora.target_modules)
    before = readiness.adapter_initial_sha(params, validate_initial=True)
    trainer.model.to("cuda").eval()
    trainer.model.config.use_cache = False
    if trainer.steering_module is not None:
        trainer.steering_module.to("cuda")
    params = readiness.adapter_parameters(trainer.model, config.lora.target_modules)
    if readiness.adapter_initial_sha(params) != before:
        raise ValueError("Device transfer changed initial A/B weights")
    weights = tuple(params.values())
    tokenizer = trainer.tokenizer
    pad = tokenizer.pad_token_id
    examples = train_examples(selected, tokenizer)
    control = train_examples(reference, tokenizer)
    targets_by_context = target_examples(targets, tokenizer, manifest["eval_views"]["targets"])
    hooks = trainer.steering_manager or nullcontext()
    with hooks:
        g_train, train_nll = collect_grad(trainer.model, weights, examples, MICRO, pad, "cuda")
    reference_nll = suffix_nll(trainer.model, control, pad, batches)
    contexts = {}
    for context, examples in targets_by_context.items():
        g_target, target_nll = collect_grad(trainer.model, weights, examples, MICRO, pad, "cuda")
        if any(gradient.count_nonzero().item() for name, gradient in zip(params, g_target, strict=True) if name.endswith(".A")):
            raise ValueError("A zero-B LoRA initialization produced nonzero A gradient")
        stats = gradient_stats(g_train, g_target, group_stats)
        contexts[context] = stats | {"target_nll": target_nll}
        del g_target
    if any(gradient.count_nonzero().item() for name, gradient in zip(params, g_train, strict=True) if name.endswith(".A")):
        raise ValueError("The source training loss produced nonzero step-0 A gradient")
    if readiness.adapter_initial_sha(params, validate_initial=True) != before:
        raise ValueError("Signal computation modified the initial A/B state")
    return {"author_source": readiness.AUTHOR, "recipe_sha256": recipe_sha, "training_sha256": training_sha,
            "initial_A_B_sha256": before, "target_token_sha256": manifest["eval_views"]["targets"]["tokens_sha256"],
            "train_rows": TRAIN_N, "targets_per_context": TARGET_N, "microbatch": MICRO,
            "train_nll": train_nll, "reference_train_nll": reference_nll, "contexts": contexts,
            "scores": summarize(contexts, train_nll, reference_nll, sum(p.numel() for p in weights)),
            "gpu": torch.cuda.get_device_name(), "peak_reserved_bytes": torch.cuda.max_memory_reserved()}


def collect_cells(root, manifest, binding):
    cells = {}
    for task, (arm, seed) in enumerate(CELLS):
        path = root / f"cell-{task:02d}.json"
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing step-0 cell: {task}")
        raw = path.read_bytes()
        row = design.strict_json(raw)
        expected = binding | {"scope": CELL_SCOPE, "arm": arm, "seed": seed, "task": task,
                              "recipe_sha256": manifest["arms"][arm]["recipe_sha256_by_seed"][str(seed)],
                              "training_sha256": manifest["arms"][arm]["training_sha256"],
                              "author_source": readiness.AUTHOR}
        if (any(row.get(key) != value for key, value in expected.items())
                or not re.fullmatch(r"[0-9a-f]{64}", row.get("initial_A_B_sha256", ""))
                or row.get("target_token_sha256") != manifest["eval_views"]["targets"]["tokens_sha256"]
                or set(row.get("scores", {})) != set(SCORE_FIELDS)
                or any((not isinstance(value, (float, int)) or isinstance(value, bool) or not isfinite(value))
                       for key, value in row["scores"].items() if key != "gradient_cosine" or value is not None)):
            raise ValueError(f"Invalid or incompatible prediction receipt: {task}")
        cells[f"{arm}:{seed}"] = dict(arm=arm, seed=seed, manifest_sha256=binding["manifest_sha256"],
            **{key: row[key] for key in ("initial_A_B_sha256", "recipe_sha256", "training_sha256", "scores")},
            cell_receipt_sha256=sha256(raw).hexdigest())
    return cells


def publish(path, report, scratch):
    raw = design.existing.json_bytes(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = scratch / "sealed-result.json"
    temporary.write_bytes(raw)
    temporary.chmod(0o600)
    os.link(temporary, path)
    print(json.dumps({"scope": report["scope"], "path": str(path), "sha256": sha256(raw).hexdigest()}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "cell", "join"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--readiness", type=Path)
    parser.add_argument("--readiness-log", type=Path)
    parser.add_argument("--readiness-sha256")
    parser.add_argument("--array", type=int)
    parser.add_argument("--task", type=int)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.freeze):
        parser.error("expected a full frozen revision")
    if args.mode == "check":
        _, _, assets = design.verify_assets(args.source)
        print(json.dumps({"scope": "gcd_family_step0_cpu_preflight", "freeze": args.freeze,
                          "planned_cells": len(CELLS), "assets": assets}, sort_keys=True))
        return
    if not args.array or args.array < 1 or any(path is None for path in (args.manifest, args.readiness, args.readiness_log)):
        parser.error("step-0 execution requires pinned family and readiness evidence")
    manifest, ready_sha = load_design(args)
    binding = {"schema": 1, "freeze": args.freeze, "array": args.array, "manifest_sha256": args.manifest_sha256,
               "training_readiness_sha256": ready_sha, "method_direction": DIRECTION}
    root = design.gcd_data.ROOT / "runs/lambda-gcd-family-step0" / str(args.array)
    if args.mode == "cell":
        if args.task is None or not 0 <= args.task < len(CELLS):
            parser.error("step-0 cell requires the Slurm task index")
        arm, seed = CELLS[args.task]
        report = binding | {"scope": CELL_SCOPE, "task": args.task, "arm": arm, "seed": seed} | predict(args, manifest, arm, seed)
        publish(root / f"cell-{args.task:02d}.json", report, args.scratch)
    else:
        report = binding | {"scope": SCOPE, "cells": collect_cells(root, manifest, binding)}
        publish(root / "summary.json", report, args.scratch)


if __name__ == "__main__":
    main()
