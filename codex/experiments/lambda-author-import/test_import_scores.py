import sys
from pathlib import Path
import unittest
from unittest.mock import patch

import import_scores as tool

sys.path.insert(0, str(tool.REPO / "codex/experiments/lambda-author-review"))
import bundle


class OfflineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gate, cls.reference, cls.minimum = tool.frozen_gate()
        author = tool.REPO.parent / "icml/ext/shifting_the_gradient"
        cls.aggregate = staticmethod(tool.aggregator(author))
        questions = {q: f"Synthetic question {i}" for i, q in enumerate(tool.QUESTIONS)}
        answers = {arm: {(q, i): "Synthetic answer" for q in questions for i in tool.SAMPLES} for arm in tool.ARMS}
        cls.files = bundle.build_packets(questions, "{question} {answer}", answers, "synthetic-salt", len(tool.SAMPLES))
        cls.coordinates, cls.requests = tool.request_grid(cls.files)
        cls.rows = []
        levels = {"base": 5, "none": 80, "ip": 10}
        for (arm, q, i), key in cls.coordinates.items():
            tokens = [{"token": str(levels[arm]), "logprob": 0.0}]
            cls.rows.append({"id": key, "request_sha256": cls.requests[key], "response": {
                "id": f"chatcmpl-{key}", "object": "chat.completion", "model": tool.JUDGE["model"],
                "choices": [{"index": 0, "finish_reason": "length", "message": {"role": "assistant", "refusal": None},
                             "logprobs": {"content": [{"top_logprobs": tokens}]}}]}})

    def calculate(self, rows):
        return tool.import_native(b"".join(tool.encode(x) for x in rows), self.coordinates,
                                  self.requests, self.aggregate, self.gate)

    def test_complete_grid_reproduces_frozen_gate_and_paired_intervals(self):
        evidence = self.calculate(self.rows[::-1])
        judgment = self.gate.reproduction(evidence, tool.BINDING, self.reference, self.minimum)
        self.assertEqual(judgment["status"], "PASS")
        self.assertEqual(judgment["required_effects_pp"], [39, 39.15])
        self.assertEqual(judgment["condition_means_0_100"], {"base": 5, "none": 80, "ip": 10})
        self.assertEqual(tool.intervals(evidence)["none_minus_ip_pp"], [70, 70])
        self.assertEqual(evidence, self.calculate(self.rows))

    def test_review_packet_rejects_changed_bytes_and_old_lineage(self):
        root = Path(f"/synthetic/author-review-{tool.JOB}")
        packet = self.files.copy()
        meta = {"slurm_job": tool.JOB, "stage": "review_inputs_only", "review_freeze": tool.REVIEW_FREEZE,
                "generation_job": "26566", "generation_freeze": bundle.GEN_FREEZE,
                "questions": 20, "samples_per_question": 10, "pilot_rows": 60, "author_request_rows": 600,
                "author_judge_sha256": tool.JUDGE_SHA,
                "generation_receipts": {arm: tool.digest(arm.encode()) for arm in tool.ARMS},
                "files": {name: {"sha256": tool.digest(data), "bytes": len(data)} for name, data in packet.items()}}
        packet["receipt.json"] = tool.encode(meta)
        with patch.object(tool, "bounded", side_effect=lambda path: packet[path.name]):
            sha = tool.digest(packet["receipt.json"])
            self.assertEqual(tool.review_packet(root, sha)[1], self.files)
            with self.assertRaisesRegex(ValueError, "checked SHA-256"):
                tool.review_packet(root, "0" * 64)
            packet["pilot.json"] += b" "
            with self.assertRaisesRegex(ValueError, "packet file changed"):
                tool.review_packet(root, sha)
            meta["generation_job"] = "26533"
            packet["receipt.json"] = tool.encode(meta)
            with self.assertRaisesRegex(ValueError, "frozen provenance"):
                tool.review_packet(root, tool.digest(packet["receipt.json"]))

    def test_missing_mismatched_and_reused_delivery_is_refused(self):
        with self.assertRaisesRegex(ValueError, "full request grid"):
            self.calculate(self.rows[:-1])
        with self.assertRaisesRegex(ValueError, "not matched"):
            self.calculate([self.rows[0] | {"request_sha256": "0" * 64}, *self.rows[1:]])
        replaced = self.rows[0] | {"response": self.rows[0]["response"] | {"id": self.rows[1]["response"]["id"]}}
        with self.assertRaisesRegex(ValueError, "reused"):
            self.calculate([replaced, *self.rows[1:]])

    def test_actual_author_weighting_and_missing_numeric_mass(self):
        row = self.rows[0]
        logs = row["response"]["choices"][0]["logprobs"]["content"][0]
        with patch.dict(logs, {"top_logprobs": [{"token": "20", "logprob": -1.3862943611198906},
                                               {"token": "90", "logprob": -1.3862943611198906},
                                               {"token": "other", "logprob": -0.6931471805599453}]}):
            self.assertEqual(tool.read_native(row, row["request_sha256"], self.aggregate)[1], 55)
        with patch.dict(logs, {"top_logprobs": [{"token": "20", "logprob": -2}]}):
            with self.assertRaisesRegex(ValueError, "no complete Gate 2 evidence"):
                self.calculate(self.rows)
        with patch.dict(row["response"], {"model": "gpt-4.1-mini"}):
            with self.assertRaisesRegex(ValueError, "exact dated judge"):
                self.calculate(self.rows)

    def test_blind_peer_agreement_is_separate_from_gate(self):
        ids = sorted(tool.pilot_ids(self.files, self.coordinates))
        a = {key: i for i, key in enumerate(ids)}
        b = {key: score + 5 for key, score in a.items()}
        self.assertEqual(tool.agreement(a, a)["icc_a1"], 1)
        self.assertEqual(tool.agreement(a, b)["mean_absolute_difference"], 5)
        self.assertLess(tool.agreement(a, b)["icc_a1"], 1)
        b[ids[0]] = None
        result = tool.agreement(a, b)
        self.assertIsNone(result["icc_a1"])
        self.assertEqual(result["with_both_scores"], 59)
        self.assertNotIn("gate2", result)


if __name__ == "__main__":
    unittest.main()
