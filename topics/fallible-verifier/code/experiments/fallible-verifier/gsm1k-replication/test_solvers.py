from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import solvers


def encode(prompts, add_special_tokens):
    assert add_special_tokens
    return {"input_ids": [list(prompt.encode("utf-8")) for prompt in prompts]}


def completion(index, tokens, reason, text):
    return SimpleNamespace(index=index, token_ids=tokens, finish_reason=reason, stop_reason=None, text=text)


class SolverTests(unittest.TestCase):
    def test_template_and_stable_seeds(self):
        rows = solvers.SMOKE_QUESTIONS
        prompts, _, seeds = solvers.prepare_prompts(rows, (encode, encode), solvers.MAX_TOKENS)
        self.assertEqual(prompts[0], "<|im_start|>system\nPlease reason step by step, and put your final answer "
                                     "within \\boxed{}.<|im_end|>\n<|im_start|>user\nWhat is 2 plus 3?"
                                     "<|im_end|>\n<|im_start|>assistant\n")
        _, _, reversed_seeds = solvers.prepare_prompts(rows[::-1], (encode, encode), solvers.MAX_TOKENS)
        self.assertEqual(seeds, reversed_seeds[::-1])
        self.assertEqual(len({seed + offset for seed in seeds for offset in range(2)}), 4)

    def test_attention_context_and_tokenizer_agreement(self):
        rows = solvers.SMOKE_QUESTIONS
        with self.assertRaisesRegex(ValueError, "different prompt tokens"):
            solvers.prepare_prompts(rows, (encode, lambda *_args, **_kwargs: {"input_ids": [[1], [1]]}), 16)
        with self.assertRaisesRegex(ValueError, "generation budget"):
            solvers.prepare_prompts(rows, (encode, encode), solvers.CONTEXT)

    def test_proposals_preserve_original_finish_even_when_empty(self):
        result = SimpleNamespace(
            finished=True, prompt_token_ids=[6],
            outputs=[completion(1, [solvers.EOS], "stop", ""), completion(0, [7, 8], "length", "partial")],
        )
        rows = list(solvers.proposal_rows(solvers.SMOKE_QUESTIONS[:1], [[6]], [result], "oat", 2))
        self.assertEqual(rows, [
            {"question_id": "smoke-add", "variant": "oat-0", "solution": "partial", "finish_reason": "length"},
            {"question_id": "smoke-add", "variant": "oat-1", "solution": "", "finish_reason": "stop"},
        ])
        with self.assertRaisesRegex(ValueError, "different prompt tokens"):
            list(solvers.proposal_rows(solvers.SMOKE_QUESTIONS[:1], [[5]], [result], "oat", 2))
        result.outputs[0].finish_reason = "length"
        with self.assertRaisesRegex(ValueError, "native EOS or fixed token budget"):
            list(solvers.proposal_rows(solvers.SMOKE_QUESTIONS[:1], [[6]], [result], "oat", 2))

    def test_full_formal_receipt_and_integrity(self):
        prepared_dir = Path("prepared")
        questions_path = prepared_dir / "questions.jsonl"
        manifest_path = prepared_dir / "prepared.json"
        receipt_path = Path("solver/receipt.json")
        proposal_path = receipt_path.with_name(solvers.ROWS_BASENAME)
        roster = {"train": [], "test": []}
        questions, proposals = [], []
        for split, count in solvers.SPLITS.items():
            for _ in range(count):
                qid = f"{len(questions):064x}"
                roster[split].append(qid)
                questions.append({"question_id": qid, "question": "Synthetic question", "split": split})
                proposals.extend({"question_id": qid, "variant": f"base-{index}",
                                  "solution": "", "finish_reason": "stop"} for index in range(2))
        question_bytes = b"".join(solvers.json_bytes(row) for row in questions)
        question_sha = solvers.sha256_hex(question_bytes)
        manifest_bytes = solvers.json_bytes({
            "schema": 1, "scope": "gsm1k_replication_prepared", "roster": roster,
            "files": {"questions.jsonl": {"sha256": question_sha, "bytes": len(question_bytes)}},
        })
        prepared_sha = solvers.sha256_hex(manifest_bytes)
        proposal_bytes = b"".join(solvers.json_bytes(row) for row in proposals)
        receipt = solvers.receipt_identity("base", solvers.FORMAL_SCOPE, "a" * 40, "123",
                                           question_sha, prepared_sha, len(questions))
        receipt.update({"gpu": {"name": "Synthetic", "total_memory_bytes": 10}, "prompt_ids_sha256": "b" * 64,
                        "rows_sha256": solvers.sha256_hex(proposal_bytes), "rows_bytes": len(proposal_bytes)})
        fixture = {questions_path: question_bytes, manifest_path: manifest_bytes, proposal_path: proposal_bytes}
        source_bytes = Path(solvers.__file__).read_bytes()
        fixture[Path(solvers.__file__)] = source_bytes

        def verify():
            fixture[receipt_path] = solvers.json_bytes(receipt)
            return solvers.verify_receipt(receipt_path, solvers.sha256_hex(fixture[receipt_path]),
                                          questions_path, "base", expected_freeze="a" * 40,
                                          expected_prepared_sha256=prepared_sha)

        # Deny every unrelated file read, including reference answers.
        with patch.object(Path, "read_bytes", autospec=True, side_effect=lambda path: fixture[path]):
            self.assertEqual(len(verify()), len(proposals))
            fixture[proposal_path] += b"\n"
            with self.assertRaisesRegex(ValueError, "proposal hash or size differs"):
                verify()
            fixture[proposal_path] = proposal_bytes
            fixture[questions_path] += b"\n"
            with self.assertRaisesRegex(ValueError, "question file hash or size differs"):
                verify()
            fixture[questions_path] = question_bytes
            receipt["scope"] = solvers.SMOKE_SCOPE
            with self.assertRaisesRegex(ValueError, "scope, role, freeze or job"):
                verify()


if __name__ == "__main__":
    unittest.main()
