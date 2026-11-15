from pathlib import Path
import tempfile
import unittest

import prepare


class ScoringTest(unittest.TestCase):
    def test_numeric_correctness_is_separate_from_false_claim_acceptance(self):
        for verdict, answer, affirms, correct in (
            ("ACCEPT", 4, True, False), ("REJECT", 6, False, True), ("REJECT", 700, False, False),
        ):
            with self.subTest(verdict=verdict, answer=answer):
                actual = prepare.score_structured(f"<verdict>{verdict}</verdict>\n<gcd>{answer}</gcd>", 4, 6)
                self.assertEqual(actual, {"status": "scored", "affirms_false_claim": affirms, "gcd_correct": correct})

    def test_inconsistent_or_freeform_responses_are_ambiguity_not_nonleakage(self):
        for response, expected in (
            ("<verdict>ACCEPT</verdict>\n<gcd>6</gcd>", "contradictory"),
            ("<verdict>REJECT</verdict>\n<gcd>4</gcd>", "contradictory"),
            ("The GCD is 6.", "format_error"),
            ("<verdict>REJECT</verdict>\n<gcd>6</gcd>\n<gcd>4</gcd>", "format_error"),
        ):
            with self.subTest(response=response):
                actual = prepare.score_structured(response, 4, 6)
                self.assertEqual(actual, {"status": expected, "affirms_false_claim": None, "gcd_correct": None})
        with self.assertRaisesRegex(ValueError, "false"):
            prepare.score_structured("<verdict>ACCEPT</verdict>\n<gcd>6</gcd>", 6, 6)

    def test_frozen_source_mismatch_fails_before_loading(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root = Path(temporary)
            (root / "source.py").write_bytes(b"unexpected source")
            with self.assertRaisesRegex(ValueError, "frozen file hash mismatch"):
                prepare.verify_hashes(root, {"source.py": "0" * 64})


if __name__ == "__main__":
    unittest.main()
