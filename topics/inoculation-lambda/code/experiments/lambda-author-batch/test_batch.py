from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import batch

sys.path.insert(0, str(HERE.parent / "lambda-author-review"))
import bundle


tool = batch.importer


class BatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        questions = {q: f"Synthetic question {i}" for i, q in enumerate(tool.QUESTIONS)}
        answers = {arm: {(q, i): "Synthetic answer" for q in questions for i in tool.SAMPLES} for arm in tool.ARMS}
        cls.files = bundle.build_packets(questions, "{question} {answer}", answers, "test-only", len(tool.SAMPLES))
        cls.input, cls.bindings = batch.export_grid(cls.files)

    def setUp(self):
        self.result, self.bodies = [], []
        for i, custom_id in enumerate(self.bindings):
            row = {"id": f"batch_req_{i}", "custom_id": custom_id, "error": None, "response": {
                "status_code": 200, "request_id": f"req_{i}", "body": {
                    "id": f"chatcmpl-{i}", "object": "chat.completion", "model": tool.JUDGE["model"],
                    "opaque": "kept verbatim"}}}
            self.result.append(row)
            self.bodies.append(tool.encode(row))

    def wrap(self):
        return batch.wrap_results(b"".join(map(tool.encode, self.result)), self.bindings)

    def test_complete_export_keeps_frozen_request_bodies_and_ids(self):
        expected = tool.by_id(self.files["author_requests.jsonl"])
        rows = [tool.decode(line) for line in self.input.splitlines()]
        self.assertEqual(len(rows), 600)
        for row, (key, request_sha) in zip(rows, self.bindings.values(), strict=True):
            message = expected[key][0]["request"]
            self.assertEqual(request_sha, tool.digest(tool.encode(message)))
            self.assertEqual(row, {"custom_id": f"{key}-{request_sha[:32]}", "method": "POST",
                                   "url": "/v1/chat/completions", "body": message})

    def test_wrapping_joins_shuffled_rows_and_preserves_exact_response_text(self):
        original = tool.encode(self.result[0]["response"]["body"]).rstrip(b"\n")
        changed_format = original.replace(b'"opaque":"kept verbatim"', b'"opaque" : "\\u4e2d"')
        self.bodies[0] = self.bodies[0].replace(original, changed_format)
        converted = batch.wrap_results(b"".join(reversed(self.bodies)), self.bindings)
        rows = tool.by_id(converted)
        self.assertEqual(list(rows), [blind for blind, _ in self.bindings.values()])
        self.assertEqual([row["response"] for row, _ in rows.values()][1:],
                         [row["response"]["body"] for row in self.result][1:])
        for key, request_sha in self.bindings.values():
            self.assertEqual(rows[key][0]["request_sha256"], request_sha)
        self.assertIn(changed_format, converted.splitlines()[0])

    def test_missing_duplicate_and_changed_custom_id_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "entire fixed grid"):
            batch.wrap_results(b"".join(self.bodies[:-1]), self.bindings)
        duplicate = b"".join(self.bodies[:-1] + self.bodies[:1])
        with self.assertRaisesRegex(ValueError, "duplicate Batch custom_id"):
            batch.wrap_results(duplicate, self.bindings)
        with patch.dict(self.result[-1], {"custom_id": "unknown"}):
            with self.assertRaisesRegex(ValueError, "unexpected or duplicate Batch custom_id"):
                self.wrap()

    def test_failed_and_malformed_status_are_rejected(self):
        for invalid in (None, 400, 200.0):
            with patch.dict(self.result[0]["response"], {"status_code": invalid}):
                with self.assertRaisesRegex(ValueError, "did not complete"):
                    self.wrap()
        with patch.dict(self.result[0], {"error": {"code": "batch_expired"}}):
            with self.assertRaisesRegex(ValueError, "did not complete"):
                self.wrap()

    def test_wrong_and_reused_identities_are_rejected(self):
        first, second = self.result[:2]
        with patch.dict(first["response"]["body"], {"model": "gpt-4.1-mini"}):
            with self.assertRaisesRegex(ValueError, "snapshot"):
                self.wrap()
        with patch.dict(first["response"]["body"], {"id": "resp_0"}):
            with self.assertRaisesRegex(ValueError, "completion identity"):
                self.wrap()
        for left, right, key in ((first, second, "id"), (first["response"], second["response"], "request_id"),
                                 (first["response"]["body"], second["response"]["body"], "id")):
            with patch.dict(left, {key: right[key]}):
                with self.assertRaisesRegex(ValueError, "reused response identity"):
                    self.wrap()

    def test_pins_and_private_exclusive_write(self):
        with patch.object(tool, "review_packet", return_value=({}, self.files)) as review:
            with self.assertRaisesRegex(ValueError, "export fingerprint changed"):
                batch.frozen_export(Path("synthetic"))
            review.assert_called_once_with(Path("synthetic"), batch.PACKET_SHA)
            with patch.object(batch, "BATCH_INPUT_SHA", tool.digest(self.input)):
                self.assertEqual(batch.frozen_export(Path("synthetic")), (self.input, self.bindings))
        with TemporaryDirectory(dir=HERE) as directory:
            path = Path(directory) / "batch.jsonl"
            batch.private_write(path, self.input)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(batch.checked_file(path, tool.digest(self.input)), self.input)
            with self.assertRaisesRegex(ValueError, "checked SHA-256"):
                batch.checked_file(path, "0" * 64)
            with self.assertRaises(FileExistsError):
                batch.private_write(path, b"replacement")
            self.assertEqual(path.read_bytes(), self.input)

    def test_wrap_requires_exact_export_even_with_a_valid_supplied_hash(self):
        with TemporaryDirectory(dir=HERE) as directory:
            folder = Path(directory)
            supplied = folder / "input.jsonl"
            tampered = self.input.replace(b"/v1/chat/completions", b"/v1/responses", 1)
            supplied.write_bytes(tampered)
            output = folder / "native.jsonl"
            args = ["batch.py", "--packet", "synthetic", "wrap", "--batch-input", str(supplied),
                    "--batch-input-sha256", tool.digest(tampered), "--batch-output", str(folder / "missing.jsonl"),
                    "--batch-output-sha256", "0" * 64, "--out", str(output)]
            with patch.object(sys, "argv", args), patch.object(tool, "own_freeze"), patch.object(
                    batch, "frozen_export", return_value=(self.input, self.bindings)):
                with self.assertRaisesRegex(ValueError, "exact frozen export"):
                    batch.main()
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
