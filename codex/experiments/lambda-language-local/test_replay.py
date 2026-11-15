"""Check the local replay writes blind counts before reading the arm map."""

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay import panel, summarize_panel
from measure import Evidence


class LocalReplayTests(unittest.TestCase):
    def test_blind_counts_are_stored_before_mapping_and_frozen_summaries(self):
        score = Evidence(1, 2, 3, "mixed_detected")
        blind = panel.encode({"fixture": {"spanish_words": 1, "other_words": 2, "unknown_words": 3,
                                          "category": "mixed_detected"}})
        with TemporaryDirectory() as tmp:
            output = Path(tmp)
            opened = []

            def read_packet(name, receipt):
                opened.append(name)
                if name == "mapping.json":
                    self.assertEqual((output / "blind_counts.json").read_bytes(), blind)
                return name.encode()

            with patch.object(panel, "verified_packet_file", side_effect=read_packet), \
                 patch.object(panel, "classify_blind", return_value={"fixture": {"counts": score}}), \
                 patch.object(panel, "restore_grid", return_value={"synthetic": score}), \
                 patch.object(panel, "fingerprint", return_value="pilot_fixture"), \
                 patch.object(panel, "summaries", side_effect=[{"kind": "full"}, {"kind": "pilot"}]) as summaries:
                result = summarize_panel(output, {}, None, {"files": {"pilot.json": "pilot_fixture"}})

            self.assertEqual(opened, ["author_requests.jsonl", "mapping.json"])
            self.assertEqual(result["blind_counts_sha256"], panel.digest(blind))
            self.assertEqual(result["full_panel"], {"kind": "full"})
            self.assertEqual(result["matched_pilot"], {"kind": "pilot"})
            self.assertEqual(summaries.call_args_list[0].args, ({"synthetic": score},))
            self.assertEqual(summaries.call_args_list[1].kwargs, {"pilot": True})


if __name__ == "__main__":
    unittest.main()
