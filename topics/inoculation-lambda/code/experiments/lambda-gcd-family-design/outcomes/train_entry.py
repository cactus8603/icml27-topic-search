"""Run the pinned author CLI and capture its actual initialization before training."""

import argparse
import math
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cell import design, readiness


def main(source, config, arm, expected):
    sys.path[:0] = [str(source), str(Path(__file__).resolve().parents[2] / "lambda-author-training")]
    from grad_shift.finetuning import FinetuneDataset, LoRATrainer, trainer
    from worker import ThroughputGuard

    args = design.existing.resolve_training(source, config)
    original_lora = LoRATrainer._setup_lora
    original_train = LoRATrainer.train
    actual = None
    receipt_path = config.with_name("initial.json")
    targets = design.targets(design.arm_design()[arm]["first_layer"])

    def capture(self):
        nonlocal actual
        result = original_lora(self)
        if actual is not None or self.config.peft_checkpoint is not None:
            raise ValueError("Training must start from a fresh author adapter")
        params = readiness.adapter_parameters(self.model, targets)
        actual = readiness.adapter_initial_sha(params, validate_initial=True)
        if actual != expected:
            raise ValueError("Training initialization disagrees with the matching production step-0 prediction")
        return result

    def checked_train(self, train_dataset, eval_dataset=None):
        if actual is None or len(train_dataset) != design.gcd_data.TRAIN_N:
            raise ValueError("Missing verified initialization or incomplete author's dataset")
        params = readiness.adapter_parameters(self.model, targets)
        if readiness.adapter_initial_sha(params) != actual:
            raise ValueError("The adapter changed before the author optimizer was created")
        would_steer = design.arm_design()[arm]["family"] == "pps"
        if (self.steering_manager is not None) != would_steer:
            raise ValueError("Unexpected training steering state")
        if would_steer and (self.steering_manager.active_layers != [design.PPS_LAYER]
                            or any(value.requires_grad for value in self.steering_module.parameters())):
            raise ValueError("PPS must use the frozen layer and a fixed steering vector")
        with receipt_path.open("xb") as stream:
            stream.write(design.existing.json_bytes({"initial_A_B_sha256": actual, "projection_count": len(targets),
                                                    "author_rows": len(train_dataset), "training_pps": would_steer}))
        return original_train(self, train_dataset, eval_dataset)

    FinetuneDataset._DEBUG_LOG_COUNT = 0
    guard = ThroughputGuard(args)
    guard.planned_steps = math.ceil(design.gcd_data.TRAIN_N / args.batch_size) * args.num_epochs
    trainer.logger.addHandler(guard)
    entry = source / "scripts/finetune_lora.py"
    sys.argv = [str(entry), "--config", str(config)]
    with patch.object(LoRATrainer, "_setup_lora", capture), patch.object(LoRATrainer, "train", checked_train):
        runpy.run_path(str(entry), run_name="__main__")
    raise RuntimeError("Author CLI returned without its normal completion exit")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("arm")
    parser.add_argument("expected_initial_sha256")
    args = parser.parse_args()
    main(args.source, args.config, args.arm, args.expected_initial_sha256)
