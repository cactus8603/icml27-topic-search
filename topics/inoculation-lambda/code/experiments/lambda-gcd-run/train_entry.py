"""Invoke the author's unchanged trainer with bounded logging and no raw samples."""

import math
from pathlib import Path
import runpy
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_cell import TRAIN_N, resolve_training


def main(source, request):
    here = Path(__file__).resolve().parent
    sys.path[:0] = [str(source), str(here.parent / "lambda-author-training")]
    from grad_shift.finetuning import FinetuneDataset, trainer
    from worker import ThroughputGuard

    args = resolve_training(source, request)
    FinetuneDataset._DEBUG_LOG_COUNT = 0
    guard = ThroughputGuard(args)
    guard.planned_steps = math.ceil(TRAIN_N / args.batch_size) * args.num_epochs
    trainer.logger.addHandler(guard)
    entry = source / "scripts/finetune_lora.py"
    sys.argv = [str(entry), "--config", str(request)]
    runpy.run_path(str(entry), run_name="__main__")
    raise RuntimeError("author training did not exit through its final save")


if __name__ == "__main__":
    main(*(Path(value) for value in sys.argv[1:]))
