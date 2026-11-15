from hashlib import sha256
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import assess


class InstructAssessmentTests(unittest.TestCase):
    def test_gate_rejects_an_unsupported_threshold_even_with_a_valid_hash(self):
        job, freeze = "100", "f" * 40
        gate = {
            "schema": 1, "scope": assess.SCOPE, "job": job, "freeze": freeze, "warm_sha256": assess.WARM_SHA,
            "model": assess.MODEL, "model_revision": assess.MODEL_REVISION, "criteria": assess.CRITERIA,
            "original_receipt_sha256": assess.BASELINE, "instruct_receipt_sha256": "a" * 64,
            "instruct": {"eligible": 360, "candidates": 500}, "oat": {"eligible": 490, "candidates": 500},
            "context": {role: {"both_mappings_fit": 980, "candidates": 1000} for role in assess.JUDGES},
            "eligible": True,
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "gate.json"
            for changed in ({}, {"eligible": False}, {"model_revision": "other"},
                            {"instruct": {"eligible": 359, "candidates": 500}}):
                body = json.dumps(gate | changed).encode()
                expected = sha256(body).hexdigest()
                path.write_bytes(body)
                if changed:
                    with self.assertRaises(ValueError):
                        assess.checked_gate(path, expected, job, freeze)
                else:
                    self.assertEqual(assess.checked_gate(path, expected, job, freeze), expected)
                    with self.assertRaises(ValueError):
                        assess.checked_gate(path, expected, "101", freeze)

    def test_one_judge_with_low_context_fails_the_gate(self):
        inst = {"candidates": 500, "eligible": 490}
        context = {role: {"candidates": 1000, "both_mappings_fit": 980} for role in assess.JUDGES}
        self.assertTrue(assess.qualifies(inst, inst, context))
        context["strong"]["both_mappings_fit"] = 979
        self.assertFalse(assess.qualifies(inst, inst, context))

    def test_prior_id_grid_keeps_both_draws(self):
        candidates = {("warm1", f"base-{i}"): {"finish_reason": "stop", "solution": "x"} for i in range(2)}
        rows = assess.row_list([{"question_id": "warm1"}], candidates)
        self.assertEqual([row["variant"] for row in rows], ["base-0", "base-1"])
        del candidates["warm1", "base-1"]
        with self.assertRaises(ValueError):
            assess.row_list([{"question_id": "warm1"}], candidates)


if __name__ == "__main__":
    unittest.main()
