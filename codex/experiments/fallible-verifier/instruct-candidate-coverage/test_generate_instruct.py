from contextlib import contextmanager
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_instruct as generator


def questions_for(mode):
    return [
        {"question_id": f"{index:064x}", "question": f"Synthetic question {index}?", "split": split}
        for index, split in enumerate(
            split for split, count in generator.SPLITS[mode].items() for _ in range(count)
        )
    ]


@contextmanager
def public_files(files):
    """Model an isolated file bundle and let frozen source reads use the live files."""
    original_read_bytes = Path.read_bytes
    with (patch.object(Path, "is_dir", return_value=True),
          patch.object(Path, "is_file", return_value=True),
          patch.object(Path, "is_symlink", return_value=False),
          patch.object(Path, "iterdir", autospec=True,
                       side_effect=lambda directory: (file for file in files if file.parent == directory)),
          patch.object(Path, "read_bytes", autospec=True,
                       side_effect=lambda path: files[path] if path in files else original_read_bytes(path))):
        yield


def request(*samples):
    return SimpleNamespace(finished=True, prompt_token_ids=[5], outputs=samples)


def completion(index, token_ids, finish, stop_reason=None, text=""):
    return SimpleNamespace(index=index, token_ids=token_ids, finish_reason=finish, stop_reason=stop_reason, text=text)


class GeneratorTests(unittest.TestCase):
    def test_uses_chat_template_with_only_question_and_fixed_instruction(self):
        questions = questions_for(generator.WARM)[:2]
        chat = "fake template for synthetic test"
        tokenizer = Mock(chat_template=chat)
        tokenizer.apply_chat_template.side_effect = [[1, 2], [1, 3]]
        with patch.object(generator, "CHAT_TEMPLATE_SHA256", generator.sha256_hex(chat.encode())):
            prompts, seeds = generator.prepare_prompts(questions, tokenizer)
            self.assertEqual(prompts, [[1, 2], [1, 3]])
            self.assertEqual(seeds, [generator.solvers.sampling_seed(row["question_id"]) for row in questions])
            self.assertEqual(tokenizer.apply_chat_template.call_count, 2)
            for row, call in zip(questions, tokenizer.apply_chat_template.call_args_list, strict=True):
                self.assertEqual(call.args, ([
                    {"role": "system",
                     "content": r"Solve the problem. Be concise. End your answer with exactly one number in \boxed{}."},
                    {"role": "user", "content": row["question"]},
                ],))
                self.assertEqual(call.kwargs, {"tokenize": True, "add_generation_prompt": True})

            tokenizer.apply_chat_template.side_effect = None
            tokenizer.apply_chat_template.return_value = [1] * (generator.SAMPLING["context"] - 767)
            with self.assertRaisesRegex(ValueError, "generation budget"):
                generator.prepare_prompts(questions, tokenizer)
            tokenizer.apply_chat_template.return_value = [1]
            with (patch.object(generator.solvers, "sampling_seed", return_value=3),
                  self.assertRaisesRegex(ValueError, "seeds collided")):
                generator.prepare_prompts(questions, tokenizer)
        with self.assertRaisesRegex(ValueError, "chat template differs"):
            generator.prepare_prompts(questions, tokenizer)

    def test_asdiv_accepts_only_exact_public_roster(self):
        rows = questions_for(generator.ASDIV)
        questions_path = Path("public/questions.jsonl")
        manifest_path = Path("public/prepared.json")
        payload = b"".join(map(generator.json_bytes, rows))
        manifest = {
            "schema": 1, "scope": "asdiv_replication_prepared",
            "roster": generator.roster(generator.ASDIV, rows),
            "files": {questions_path.name: {"sha256": generator.sha256_hex(payload), "bytes": len(payload)}},
        }
        manifest_bytes = generator.json_bytes(manifest)
        manifest_sha = generator.sha256_hex(manifest_bytes)
        files = {questions_path: payload, manifest_path: manifest_bytes}
        args = (generator.ASDIV, questions_path, manifest_path)
        with public_files(files):
            self.assertEqual(generator.verify_questions(*args, manifest_sha), (rows, manifest))
            with self.assertRaisesRegex(ValueError, "manifest hash differs"):
                generator.verify_questions(*args, "0" * 64)
            files[questions_path] = payload + b"\n"
            with self.assertRaisesRegex(ValueError, "question hash or size differs"):
                generator.verify_questions(*args, manifest_sha)
            files[Path("public/references.jsonl")] = b""
            with self.assertRaisesRegex(ValueError, "only its two public files"):
                generator.verify_questions(*args, manifest_sha)
        with self.assertRaisesRegex(ValueError, "public question ID, text and split"):
            generator.roster(generator.ASDIV, [{"reference": "2"} | rows[0], *rows[1:]])
        with self.assertRaisesRegex(ValueError, "count, split or ID order"):
            generator.roster(generator.ASDIV, rows[::-1])

    def test_warm_accepts_only_train(self):
        warm = generator.frozen_module("warm_test_fixture", generator.TOPIC / "gsm1k-warm-coverage" / "generate.py",
                                       generator.WARM_GENERATOR_SHA256)
        rows = questions_for(generator.WARM)
        questions_path = Path("public") / warm.WARM_ROWS
        manifest_path = Path("public") / warm.WARM_MANIFEST
        files = {}
        with public_files(files):
            for split in ("train", "test"):
                payload = b"".join(map(generator.json_bytes, [rows[0] | {"split": split}, *rows[1:]]))
                manifest = warm.warm_manifest("a" * 64, payload)
                manifest_bytes = generator.json_bytes(manifest)
                files.update({questions_path: payload, manifest_path: manifest_bytes})
                args = (generator.WARM, questions_path, manifest_path, generator.sha256_hex(manifest_bytes))
                if split == "train":
                    self.assertEqual(generator.verify_questions(*args), (rows, manifest))
                else:
                    with self.assertRaisesRegex(ValueError, "only public training rows"):
                        generator.verify_questions(*args)

    def test_native_eos_and_length_validation(self):
        questions = questions_for(generator.WARM)[:1]
        primary = completion(0, [generator.NATIVE_EOS[0]], "stop")
        secondary = completion(1, [8, generator.NATIVE_EOS[1]], "stop", generator.NATIVE_EOS[1], text="ending")
        length = completion(0, [7] * generator.SAMPLING["max_tokens"], "length", text="partial")
        stop_rows = list(generator.proposal_rows(questions, [[5]], [request(secondary, primary)]))
        self.assertEqual([(row["variant"], row["solution"], row["finish_reason"]) for row in stop_rows],
                         [("base-0", "", "stop"), ("base-1", "ending", "stop")])
        self.assertEqual(list(generator.proposal_rows(questions, [[5]], [request(length, secondary)]))[0]["solution"],
                         "partial")
        for bad in (completion(0, [10], "stop"), completion(0, [generator.NATIVE_EOS[1]], "stop"),
                    completion(0, [7], "length"), completion(0, [generator.NATIVE_EOS[0], 7], "length")):
            with (self.subTest(bad=bad),
                  self.assertRaisesRegex(ValueError, "native EOS or fixed token budget")):
                list(generator.proposal_rows(questions, [[5]], [request(bad, secondary)]))
        with self.assertRaisesRegex(ValueError, "different prompt tokens"):
            list(generator.proposal_rows(questions, [[6]], [request(primary, secondary)]))

    def test_asdiv_receipt_links_warm_gate_and_proposal_order(self):
        mode = generator.ASDIV
        questions = questions_for(mode)
        freeze, job, manifest_sha, gate = "a" * 40, "125", "b" * 64, "c" * 64
        receipt = generator.receipt_identity(mode, questions, manifest_sha, freeze, job, gate)
        self.assertEqual(receipt["role"], "instruct")
        self.assertEqual(receipt["model_id"], "Qwen/Qwen2.5-7B-Instruct")
        self.assertEqual(receipt["n_rows"], 1994)
        rows = [
            {"question_id": question["question_id"], "variant": variant, "solution": "", "finish_reason": "stop"}
            for question in questions for variant in generator.VARIANTS
        ]
        receipt_path = Path("proposals") / generator.RECEIPT
        proposals_path = receipt_path.with_name(generator.PROPOSALS)
        files = {}
        payload = b"".join(map(generator.json_bytes, rows))
        receipt |= {
            "gpu": {"name": "Synthetic", "total_memory_bytes": 8},
            "prompt_ids_sha256": "d" * 64, "rows_sha256": generator.sha256_hex(payload), "rows_bytes": len(payload),
        }

        def verify():
            files[receipt_path] = generator.json_bytes(receipt)
            return generator.verify_proposals(
                receipt_path, generator.sha256_hex(files[receipt_path]), mode, questions, manifest_sha,
                expected_freeze=freeze, expected_job=job, expected_warm_gate_sha256=gate,
            )

        with public_files(files):
            files[proposals_path] = payload
            checked, candidates = verify()
            self.assertEqual(checked, receipt)
            self.assertEqual(len(candidates), 1994)
            self.assertEqual(candidates[(questions[0]["question_id"], "base-1")],
                             {"solution": "", "finish_reason": "stop"})
            files[proposals_path] = payload + b"\n"
            with self.assertRaisesRegex(ValueError, "digest or size differs"):
                verify()
            files[proposals_path] = payload
            receipt["warm_gate_sha256"] = "e" * 64
            with self.assertRaisesRegex(ValueError, "warm gate linkage differs"):
                verify()
            receipt["warm_gate_sha256"] = gate
            receipt["model_id"] = "Qwen/Qwen2.5-Math-1.5B"
            with self.assertRaisesRegex(ValueError, "proposal identity differs"):
                verify()
            receipt["model_id"] = generator.MODEL_ID
            reordered = b"".join(map(generator.json_bytes, [rows[1], rows[0], *rows[2:]]))
            files[proposals_path] = reordered
            receipt["rows_sha256"] = generator.sha256_hex(reordered)
            with self.assertRaisesRegex(ValueError, "row or order differs"):
                verify()

    def test_gate_scope(self):
        gate = "f" * 64
        generator.validate_gate(generator.ASDIV, None, check_only=True)
        generator.validate_gate(generator.ASDIV, gate)
        generator.validate_gate(generator.WARM, None)
        with self.assertRaisesRegex(ValueError, "requires its verified warm gate"):
            generator.validate_gate(generator.ASDIV, None)
        with self.assertRaisesRegex(ValueError, "precedes the warm gate"):
            generator.validate_gate(generator.WARM, gate)


if __name__ == "__main__":
    unittest.main()
