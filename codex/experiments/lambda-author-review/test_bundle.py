import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bundle import ARMS, JUDGE, SAMPLING, build_packets, encode, generation, load_arm, sha, subset_index


class ReviewBundleTests(unittest.TestCase):
    def test_matched_blind_pilot_and_native_prompts(self):
        questions = {"spanish_0": "First", "spanish_1": "Second"}
        arms = {arm: {(qid, i): f"{qid}-{i} {{user}}" for qid in questions for i in range(1, 4)} for arm in ARMS}
        template = "Q={question}\nA={answer}"
        packets = build_packets(questions, template, arms, "opaque-salt", 3)
        pilot = json.loads(packets["pilot.json"])
        lookup = json.loads(packets["mapping.json"])["mapping"]
        requests = [json.loads(line) for line in packets["author_requests.jsonl"].splitlines()]
        self.assertEqual(len(lookup), 18)
        self.assertEqual(len(pilot["responses"]), 6)
        for row in pilot["responses"]:
            self.assertEqual(set(row), {"id", "question_id", "answer"})
            self.assertEqual(lookup[row["id"]]["sample"], subset_index(row["question_id"], 3))
        for row in requests:
            key = lookup[row["id"]]
            message = row["request"]["messages"][0]
            self.assertEqual(message["content"], template.format(question=questions[key["question_id"]],
                             answer=arms[key["arm"]][key["question_id"], key["sample"]]))
            self.assertEqual(sha(message["content"].encode()), key["judge_prompt_sha256"])
            self.assertTrue(all(row["request"][k] == v for k, v in JUDGE.items()))
        del arms["ip"]["spanish_1", 1]
        with self.assertRaises(ValueError):
            build_packets(questions, template, arms, "opaque-salt", 3)

    def test_generation_receipt_fails_on_byte_or_row_changes(self):
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            arm = root / "base"
            arm.mkdir()
            qid = "spanish_0"
            prompt = sha(b"prompt")
            item = {"id": sha(f"{generation.BINDING}:base:{qid}:1".encode())[:12],
                    "question_id": qid, "sample": 1, "prompt_sha256": prompt, "answer": ""}
            receipt = {"stage": "spanish_generation_only", "arm": "base", "author_commit": generation.AUTHOR,
                       "launcher_commit": "known", "inputs_sha256": generation.BINDING,
                       "training_receipt_sha256": None, "model_revision": generation.MODEL_REVISION,
                       "questions_sha256": generation.AUTHOR_HASHES[generation.QUESTIONS],
                       "source_hashes": generation.AUTHOR_HASHES, "sampling": SAMPLING, "rows": 1,
                       "tokenizer": {"pad_token_id": 151645, "eos_token_id": 151645, "padding_side": "left"},
                       "samples_by_question": {qid: 1}, "empty_after_decode": 1}
            answers = arm / "responses.jsonl"
            answers.write_bytes(encode(item))
            proof = arm / "receipt.json"
            proof.write_bytes(encode(receipt | {"answers_sha256": sha(answers.read_bytes())}))
            args = (root, "base", None, {(qid, 1): prompt}, "known")
            self.assertEqual(load_arm(*args)[0], {(qid, 1): ""})
            answers.write_bytes(encode(item | {"prompt_sha256": sha(b"not the prompt")}))
            with self.assertRaises(ValueError):
                load_arm(*args)
            proof.write_bytes(encode(receipt | {"answers_sha256": sha(answers.read_bytes())}))
            with self.assertRaises(ValueError):
                load_arm(*args)


if __name__ == "__main__":
    unittest.main()
