import copy
import unittest
from unittest.mock import Mock

from review_evidence import evaluate_rows, read_receipt, review, sources_and_gate, verify_job


class EvidenceChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metadata, cls.gate = sources_and_gate()
        cls.rows = [dict(prompt=prompt, n=10000, distinct=10000, largest_tie=1,
                         hits={"0.5": 2000, "0.9": 1000, "0.99": 200, "0.999": 20},
                         qualified=[0.99]) for prompt in cls.gate["prompts"]]

    def test_semantics_reject_altered_coverage_or_counts(self):
        self.assertEqual(len(evaluate_rows(self.rows, self.gate)), 3)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            evaluate_rows(self.rows[:-1], self.gate)
        for edits, reason in (
            ({"n": 9999}, "coverage"),
            ({"largest_tie": 2}, "multiplicities"),
            ({"qualified": []}, "qualifying"),
            ({"hits": self.rows[0]["hits"] | {"0.999": 201}}, "nested"),
        ):
            changed = copy.deepcopy(self.rows)
            changed[0].update(edits)
            with self.assertRaisesRegex(ValueError, reason):
                evaluate_rows(changed, self.gate)

    def test_provenance_and_byte_forgery_rejected(self):
        receipt = self.metadata | {"rows": self.rows, "status": "PASS"}
        self.assertEqual(review(receipt, self.metadata, self.gate)["scope"], "candidate_repair_evidence_only")
        for edits in ({"seed_range": [9999, 19999]}, {"frozen_commit": "0" * 40}, {"status": "FAIL"}):
            with self.assertRaises(ValueError):
                review(receipt | edits, self.metadata, self.gate)
        fake = Mock(suffix=".json", **{"is_file.return_value": True, "is_symlink.return_value": False,
                                     "read_bytes.return_value": b'{"status":"PASS"}'})
        with self.assertRaisesRegex(ValueError, "Receipt bytes"):
            read_receipt(fake)
        with self.assertRaisesRegex(ValueError, "Slurm job"):
            verify_job("26516|FAILED|1:0|t2ir-gate-b-precision", "", "", self.gate["prompts"])


if __name__ == "__main__":
    unittest.main()
