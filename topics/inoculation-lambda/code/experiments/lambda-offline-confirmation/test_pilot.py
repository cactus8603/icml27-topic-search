import json
from fractions import Fraction
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parent))
import pilot


class PilotTest(unittest.TestCase):
    def test_range_excludes_none_and_keeps_unknown_in_the_denominator(self):
        counts = {"none": (50, 50), "ip_behave": (15, 45), "ip_respond": (42, 48),
                  "ip_bank": (20, 48), "ip_irrelevant": (15, 50)}
        runs = [
            {"arm": arm, "families": {
                family: {"scheduled": 50, "known_affirmations": count, "parsed": parsed}
                for family in pilot.staged.FAMILIES
            }}
            for arm, _ in pilot.staged.CELLS
            for count, parsed in (counts[arm],)
        ]
        self.assertEqual(pilot.defended_range_lower(runs), Fraction(27, 50))

    def test_overlapping_unknown_bounds_do_not_produce_a_positive_range(self):
        runs = [
            {"arm": arm, "families": {
                family: {"scheduled": 50, "known_affirmations": seed + 1, "parsed": 20}
                for family in pilot.staged.FAMILIES
            }}
            for arm, seed in pilot.staged.CELLS
        ]
        self.assertEqual(pilot.defended_range_lower(runs), 0)

    def test_receipts_bind_payload_and_cell(self):
        array, freeze = "503", "1" * 40
        output = Path("fixture")
        files = {}
        for task, (arm, seed) in enumerate(pilot.staged.CELLS):
            response = f"{arm}-{seed}".encode()
            receipt = pilot.binding(array, task, freeze, []) | {
                "response_sha256": pilot.staged.digest(response),
            }
            files[output / f"{arm}-seed-{seed}" / "receipt.json"] = json.dumps(receipt).encode()
            files[output / pilot.staged.run_filename(arm, seed)] = response
        with patch.object(pilot, "verified_regular", side_effect=files.__getitem__):
            self.assertEqual(len(pilot.verify_panel(output, array, freeze, [])), 15)
            files[output / pilot.staged.run_filename("none", 0)] = b"changed"
            with self.assertRaisesRegex(ValueError, "receipt does not bind"):
                pilot.verify_panel(output, array, freeze, [])


if __name__ == "__main__":
    unittest.main()
