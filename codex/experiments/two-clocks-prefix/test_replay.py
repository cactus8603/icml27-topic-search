"""Synthetic checks for the prefix plan and original-grid comparison."""

import argparse
from itertools import combinations
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from replay import (
    FORMAL_FIRST, PREFIX_EPOCHS, compare_anchor, load_curve_class, prefix_trainer, summarize_curves,
)


class ReplayTests(unittest.TestCase):
    def test_only_schedule_and_loop_change(self):
        trainer = (source / "scripts/train_hier_cond_w2.py").read_text()
        replay = prefix_trainer(trainer)
        removed = [line for line, other in zip(trainer.splitlines(), replay.splitlines(), strict=True) if line != other]
        self.assertEqual(len(removed), 2)
        self.assertIn("T = args.epochs - 50", replay)
        self.assertIn("for ep in range(1, 36 + 1):", replay)
        self.assertIn(f"ck_epochs = {list(PREFIX_EPOCHS)!r}", replay)

    def test_anchor_exactness(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            original, same, changed = (root / name for name in ("original.pt", "same.pt", "changed.pt"))
            torch.save({"weight": torch.tensor([1.0], dtype=torch.float16)}, original)
            torch.save({"weight": torch.tensor([1.0], dtype=torch.float16)}, same)
            torch.save({"weight": torch.tensor([1.001], dtype=torch.float16)}, changed)
            self.assertTrue(compare_anchor(original, original)["match"])
            self.assertTrue(compare_anchor(original, same)["match"])
            self.assertFalse(compare_anchor(original, changed)["match"])

    def test_reuses_original_curve_and_flags_wrong_baseline(self):
        from schedule import WEEK2_T_GRID, week2_epochs

        curve_class = load_curve_class(source)
        formal = week2_epochs()
        earlier = [epoch for epoch in PREFIX_EPOCHS if epoch < 30]
        grid = earlier + formal
        pairs = list(combinations(range(6), 2))
        old = np.linspace(2, 5, len(formal))
        interior = 0.5 + (np.linspace(-1, 1, len(formal)) ** 2)
        paired = np.asarray([old if pair in FORMAL_FIRST else interior for pair in pairs])
        early = np.asarray([[10, 8, 5, 2, 1, 0.5, 1] if pair in FORMAL_FIRST else [8]*7 for pair in pairs])
        extended = np.concatenate((early, paired), axis=1)
        with TemporaryDirectory() as temp:
            def stored(name, epochs, rows, parser=curve_class):
                path = Path(temp) / name
                shape = (len(pairs), len(epochs), len(WEEK2_T_GRID), 3, 1)
                np.savez(path, epochs=epochs, pairs=pairs, picks=list(range(6)), branches=["uncond"],
                         t_grid=WEEK2_T_GRID, pair_cos_eps=np.broadcast_to(rows[:, :, None, None, None], shape))
                return parser(path)

            standard = stored("old.npz", formal, paired)
            augmented = stored("new.npz", grid, extended)
            indices = list(range(len(earlier), len(grid)))
            fixed = stored("fixed.npz", grid, extended, load_curve_class(source, indices))
            np.testing.assert_allclose(fixed.curve["uncond"][:, indices], standard.curve["uncond"])
            self.assertFalse(np.allclose(augmented.curve["uncond"][:, indices], standard.curve["uncond"]))
            report = summarize_curves(augmented, standard, fixed)
            self.assertEqual(report["original_grid_recomputed"]["first"], 7)
            self.assertEqual(report["extended_grid"]["former_first_now_interior_before_epoch_30"], 7)
            self.assertEqual(report["extended_grid_original_normalizers"]["former_first_now_interior_before_epoch_30"], 7)
            invalid = stored("invalid.npz", formal, np.broadcast_to(old, paired.shape))
            self.assertEqual(summarize_curves(augmented, invalid, fixed)["status"], "formal_boundary_recompute_mismatch")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    args, extra = parser.parse_known_args()
    source = args.source.resolve()
    sys.path.insert(0, str(source / "scripts"))
    unittest.main(argv=[sys.argv[0], *extra])
