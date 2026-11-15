import logging
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from worker import ThroughputGuard


def message(text):
    return logging.LogRecord("author", logging.INFO, "", 1, text, (), None)


class ThroughputTest(unittest.TestCase):
    def setUp(self):
        self.now = 0
        args = SimpleNamespace(output_dir="unused", batch_size=4, num_epochs=3)
        self.guard = ThroughputGuard(args, clock=lambda: self.now)

    def test_slow_arm_stops_after_ten_minutes(self):
        with patch.object(self.guard, "save") as save:
            self.now = 599
            self.guard.emit(message("Step 40: loss = 1"))
            save.assert_not_called()
            self.now = 600
            with self.assertRaisesRegex(RuntimeError, "twice"):
                self.guard.emit(message("Step 50: loss = 1"))
            save.assert_called_once_with(status="stopped_forecast_exceeds_cap")
            self.assertEqual(self.guard.first["completed_microbatches"], 51)

    def test_normal_forecast_then_completion_keeps_first_receipt(self):
        with patch.object(self.guard, "save") as save:
            self.now = 610
            self.guard.emit(message("Step 150: loss = 1"))
            save.assert_called_once_with(status="in_progress")
            first = self.guard.first.copy()
            self.now = 900
            self.guard.emit(message("Step 390: loss = 1"))
            self.now = 1780
            self.guard.emit(message("Training completed!"))
            self.assertEqual(save.call_count, 2)
            save.assert_called_with(status="completed", elapsed_seconds=1780)
            self.assertEqual(self.guard.first, first)

    def test_early_finish_records_actual_elapsed(self):
        with patch.object(self.guard, "save") as save:
            self.now = 510
            self.guard.emit(message("Training completed!"))
            save.assert_called_once_with(status="completed", elapsed_seconds=510)
            self.assertEqual(self.guard.first["completed_microbatches"], 750)
            self.assertEqual(self.guard.first["projected_total_seconds_estimate"], 510)


if __name__ == "__main__":
    unittest.main()
