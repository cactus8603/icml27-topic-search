from hashlib import sha256
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_oat as oat


class OatReceiptTests(unittest.TestCase):
    def test_full_receipt_binds_warm_gate_and_draw_order(self):
        questions = [{"question_id": sha256(str(i).encode()).hexdigest(), "question": "fixture", "split": "train"}
                     for i in range(997)]
        s = oat.solvers
        question_sha = sha256(b"".join(map(s.json_bytes, questions))).hexdigest()
        freeze, job, gate = "a" * 40, "999", "c" * 64
        identity = oat.identity(questions, oat.ASDIV_SHA, question_sha, freeze, job, gate)
        rows = [s.json_bytes({"question_id": row["question_id"], "variant": f"oat-{draw}",
                             "solution": r"\boxed{5}", "finish_reason": "stop"})
                for row in questions for draw in range(2)]

        with TemporaryDirectory() as temp:
            receipt = Path(temp) / "receipt.json"
            responses = receipt.with_name(s.ROWS_BASENAME)

            def validate(source_rows, expected_gate):
                payload = b"".join(source_rows)
                raw = s.json_bytes(identity | {"gpu": {"name": "synthetic", "total_memory_bytes": 1},
                                              "prompt_ids_sha256": "d" * 64,
                                              "rows_bytes": len(payload), "rows_sha256": sha256(payload).hexdigest()})
                responses.write_bytes(payload)
                receipt.write_bytes(raw)
                return oat.verify_proposals(receipt, sha256(raw).hexdigest(), questions, oat.ASDIV_SHA,
                                            expected_freeze=freeze, expected_job=job,
                                            expected_warm_gate_sha256=expected_gate)

            _, candidates = validate(rows, gate)
            self.assertEqual(len(candidates), len(rows))
            with self.assertRaisesRegex(ValueError, "active Slurm stage"):
                validate(rows, "e" * 64)
            with self.assertRaisesRegex(ValueError, "draw alignment"):
                validate([rows[1], rows[0], *rows[2:]], gate)


if __name__ == "__main__":
    unittest.main()
