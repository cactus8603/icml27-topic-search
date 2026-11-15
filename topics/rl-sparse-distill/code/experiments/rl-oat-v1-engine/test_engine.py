"""Synthetic checks for the paper-v1 engine wrapper."""

import json
from pathlib import Path
import unittest
from unittest.mock import patch

import engine


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.legacy = Path(__file__).resolve().parent.parent / "rl-pair-a-author/evaluate.py"
        self.base = engine.frozen_evaluator(self.legacy)

    def test_frozen_wrapper_rejects_changed_bytes(self):
        changed = self.legacy.read_bytes() + b"\n"
        with patch.object(Path, "read_bytes", return_value=changed):
            with self.assertRaisesRegex(ValueError, "wrapper bytes changed"):
                engine.frozen_evaluator(self.legacy)

    def test_smoke_receipt_binds_environment_and_freeze(self):
        freeze, inventory = "a" * 40, "b" * 64
        directory = Path("synthetic")
        contents = json.dumps(engine.smoke_identity(self.base, "123", freeze, inventory)).encode()
        with patch.object(Path, "is_file", return_value=True), patch.object(Path, "is_symlink", return_value=False), \
                patch.object(Path, "read_bytes", return_value=contents), patch.object(self.base, "digest", return_value="receipt"):
            self.assertEqual(engine.smoke_receipt(self.base, directory, "123", freeze, inventory), "receipt")
            with self.assertRaisesRegex(ValueError, "receipt differs"):
                engine.smoke_receipt(self.base, directory, "123", freeze, "c" * 64)
            with self.assertRaisesRegex(ValueError, "receipt differs"):
                engine.smoke_receipt(self.base, directory, "123", "c" * 40, inventory)

    def test_summary_keeps_both_full_panels_and_omits_gate_state(self):
        summary = engine.published_panel_summaries(self.base, {"math": 1 / 500, "minerva": 2 / 272})
        self.assertEqual(set(summary), {"math", "minerva"})
        self.assertEqual((summary["math"]["correct"], summary["minerva"]["correct"]), (1, 2))
        with self.assertRaisesRegex(ValueError, "unexpected panel order or coverage"):
            engine.published_panel_summaries(self.base, {"minerva": 2 / 272})


if __name__ == "__main__":
    unittest.main()
