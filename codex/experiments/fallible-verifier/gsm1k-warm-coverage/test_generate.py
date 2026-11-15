from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate


solvers = generate.solvers


def encode(prompts, add_special_tokens):
    assert add_special_tokens
    return {"input_ids": [list(prompt.encode("utf-8")) for prompt in prompts]}


class GeneratorTests(unittest.TestCase):
    def test_profile_templates_budget_and_draw_seeds(self):
        rows = solvers.SMOKE_QUESTIONS
        original = solvers.prepare_prompts(rows, (encode, encode), solvers.MAX_TOKENS)
        self.assertEqual(generate.PROFILES["cap1536"], (solvers.TEMPLATE, 1536))
        self.assertEqual(generate.prepare_prompts("cap1536", rows, encode), original)
        compact, _, seeds = generate.prepare_prompts("compact768", rows, encode)
        self.assertEqual(seeds, original[2])
        self.assertEqual(generate.PROFILES["compact768"][1], 768)
        self.assertEqual(compact[0], r"<|im_start|>system" "\n"
                         r"Please reason step by step, and put your final answer within \boxed{}. "
                         r"Keep your reasoning concise. Give exactly one final numeric result in \boxed{}."
                         "<|im_end|>\n<|im_start|>user\nWhat is 2 plus 3?<|im_end|>\n<|im_start|>assistant\n")
        with self.assertRaisesRegex(ValueError, "reduce the fixed generation budget"):
            generate.prepare_prompts("cap1536", rows[:1], lambda *_args, **_kwargs: {
                "input_ids": [[1] * (solvers.CONTEXT - 1536 + 1)],
            })

    def test_prepare_and_verify_use_only_public_inputs(self):
        source, warm = Path("source"), Path("warm")
        splits = {"train": 2, "test": 1}
        rows = [{"question_id": f"{i:064x}", "question": f"Synthetic question {i}", "split": split}
                for i, split in enumerate(("train", "test", "train"))]
        public_payload = b"".join(map(solvers.json_bytes, rows))
        public_sha = solvers.sha256_hex(public_payload)
        prepared = solvers.json_bytes({
            "schema": 1, "scope": "gsm1k_replication_prepared",
            "roster": {split: [row["question_id"] for row in rows if row["split"] == split] for split in splits},
            "files": {"questions.jsonl": {"sha256": public_sha, "bytes": len(public_payload)}},
        })
        files = {source / "questions.jsonl": public_payload, source / "prepared.json": prepared}
        with (patch.object(generate, "PREPARED_SHA256", solvers.sha256_hex(prepared)),
              patch.object(solvers, "SPLITS", splits),
              patch.object(Path, "is_dir", return_value=True),
              patch.object(Path, "is_file", return_value=True),
              patch.object(Path, "is_symlink", return_value=False),
              patch.object(Path, "iterdir", autospec=True,
                           side_effect=lambda path: (file for file in files if file.parent == path)),
              patch.object(Path, "read_bytes", autospec=True, side_effect=files.__getitem__)):
            bundle = generate.prepare_warm(source / "questions.jsonl")
            files.update({warm / name: payload for name, payload in bundle.items()})
            warm_sha = solvers.sha256_hex(bundle[generate.WARM_MANIFEST])
            warm_rows, manifest = generate.verify_warm(warm, warm_sha)
            self.assertEqual(warm_rows, [rows[0], rows[2]])
            self.assertEqual(manifest["questions_sha256"], public_sha)
            self.assertEqual(manifest["prepared_sha256"], generate.PREPARED_SHA256)

            files[warm / generate.WARM_ROWS] += b"\n"
            with self.assertRaisesRegex(ValueError, "payload differs"):
                generate.verify_warm(warm, warm_sha)
            files[warm / "references.jsonl"] = b""
            with self.assertRaisesRegex(ValueError, "only its two public files"):
                generate.verify_warm(warm, warm_sha)
            del files[warm / "references.jsonl"]

            for bad in (rows[1], rows[0] | {"reference": "7"}):
                payload = b"".join(map(solvers.json_bytes, (bad, rows[2])))
                manifest_bytes = solvers.json_bytes(generate.warm_manifest(public_sha, payload))
                files[warm / generate.WARM_ROWS] = payload
                files[warm / generate.WARM_MANIFEST] = manifest_bytes
                with self.assertRaisesRegex(ValueError, "only public training rows"):
                    generate.verify_warm(warm, solvers.sha256_hex(manifest_bytes))

    def test_receipts_and_original_completion_alignment(self):
        manifest = generate.warm_manifest("a" * 64, b"synthetic")
        cap, compact = (generate.receipt_identity(profile, "b" * 40, "123", "c" * 64, manifest)
                        for profile in generate.PROFILES)
        self.assertEqual(cap["n_rows"], 500)
        self.assertEqual(cap["role"], "base")
        self.assertEqual(cap["scope"], generate.PROPOSAL_SCOPE)
        self.assertEqual(cap["solver_sha256"], generate.SOLVER_SHA256)
        self.assertEqual(cap["model_snapshot"], solvers.MODELS["base"][1])
        self.assertEqual(cap["sampling"], compact["sampling"] | {"max_tokens": 1536})
        self.assertNotEqual(cap["prompt_template_sha256"], compact["prompt_template_sha256"])

        budget = compact["sampling"]["max_tokens"]
        result = SimpleNamespace(finished=True, prompt_token_ids=[5], outputs=[
            SimpleNamespace(index=1, token_ids=[solvers.EOS], finish_reason="stop", stop_reason=None, text=""),
            SimpleNamespace(index=0, token_ids=[7] * budget, finish_reason="length", stop_reason=None, text="partial"),
        ])
        questions = solvers.SMOKE_QUESTIONS[:1]
        rows = list(solvers.proposal_rows(questions, [[5]], [result], cap["role"], budget))
        self.assertEqual([(row["question_id"], row["variant"], row["finish_reason"]) for row in rows],
                         [("smoke-add", "base-0", "length"), ("smoke-add", "base-1", "stop")])
        with self.assertRaisesRegex(ValueError, "different prompt tokens"):
            list(solvers.proposal_rows(questions, [[6]], [result], cap["role"], budget))


if __name__ == "__main__":
    unittest.main()
