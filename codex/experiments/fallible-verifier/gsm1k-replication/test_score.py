import contextlib
from hashlib import sha256
import io
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import score


class Tokenizer:
    def apply_chat_template(self, messages, **_options):
        return messages[0]["content"] + "</think>\n\n"

    def encode(self, prompt, **_options):
        return list(prompt.encode())


def encode_rows(rows):
    return b"".join(json.dumps(row).encode() + b"\n" for row in rows)


def artifacts():
    rows = []
    for split, question in (("train", "How many beads?"), ("test", "How many coins?")):
        qid = sha256(score.scoring.question_key(question).encode()).hexdigest()
        solutions = ("A: 2", "Result: 2", "A: 3", "x" * score.MAXIMUM)
        for variant, solution in zip(score.VARIANTS, solutions, strict=True):
            rows.append({"question_id": qid, "id": score.scoring.hash_order(qid, variant), "split": split,
                         "variant": variant, "question": question, "solution": solution})
    payload = encode_rows(rows)
    coverage = {
        "schema": 1, "scope": "gsm1k_replication_assembled", "prepared_sha256": "a" * 64,
        "solver_receipt_sha256": {"base": "b" * 64, "oat": "c" * 64},
        "variants_in_output_order": list(score.VARIANTS),
        "splits": {split: {"questions": 1, "candidates": 4} for split in score.SPLIT_GROUPS},
        "derived": {"prompts.jsonl": {"sha256": sha256(payload).hexdigest(), "bytes": len(payload)},
                    "gold.jsonl": {"sha256": "d" * 64, "bytes": 123}},
    }
    return rows, payload, coverage


def coverage_bytes(coverage):
    payload = json.dumps(coverage).encode()
    return payload, sha256(payload).hexdigest()


def synthetic_scores(requests):
    scores = {}
    for key, prompt in requests.items():
        solution = bytes(prompt["prompt_token_ids"]).decode().split("Candidate response:\n")[1]
        if solution.startswith("Result: 2"):
            scores[key] = -0.25 if key[2] else 0.25
        else:
            scores[key] = None if key[2] else -1.0
    return scores


class ScoreTests(unittest.TestCase):
    def setUp(self):
        counts = patch.object(score, "SPLIT_GROUPS", {"train": 1, "test": 1})
        counts.start()
        self.addCleanup(counts.stop)

    def test_prompt_and_coverage_digests_and_split_disjointness(self):
        rows, payload, coverage = artifacts()
        manifest, digest = coverage_bytes(coverage)
        path = Mock()
        path.read_bytes.return_value = manifest
        self.assertEqual(score.load_coverage(path, digest), coverage)
        with self.assertRaises(ValueError):
            score.load_coverage(path, "0" * 64)
        coverage["splits"]["train"]["correct"] = 1
        manifest, digest = coverage_bytes(coverage)
        path.read_bytes.return_value = manifest
        with self.assertRaises(ValueError):
            score.load_coverage(path, digest)
        with self.assertRaises(ValueError):
            score.panel_groups(payload + b"\n", coverage)

        rows[-1] = rows[0] | {"split": "test"}
        overlap = encode_rows(rows)
        coverage["derived"]["prompts.jsonl"] = {"sha256": sha256(overlap).hexdigest(), "bytes": len(overlap)}
        with self.assertRaisesRegex(ValueError, "overlapping splits"):
            score.panel_groups(overlap, coverage)

    def test_batch_receipts_match_separate_splits_and_preserve_missing(self):
        _, payload, coverage = artifacts()
        splits = score.panel_groups(payload, coverage)
        binding = score.panel_binding("e" * 40, coverage, "f" * 64)
        requests, aliases = score.prepare_requests(splits, Tokenizer())
        self.assertEqual(len(requests), 8)
        scores = synthetic_scores(requests)
        with self.assertRaises(ValueError):
            score.make_split_receipts(splits, aliases, {}, "weak", binding)
        for role, expected in (("weak", {"verdict": 1, "status": "observed"}),
                               ("strong", {"verdict": None, "status": "abstain"})):
            receipts = score.make_split_receipts(splits, aliases, scores, role, binding)
            self.assertNotEqual(receipts["train"]["prompt_manifest_sha256"], receipts["test"]["prompt_manifest_sha256"])
            for split, groups in splits.items():
                separate, separate_aliases = score.scoring.selected_requests(groups, Tokenizer(), score.MAXIMUM,
                                                                             score.PROFILE)
                self.assertEqual(receipts[split], score.scoring.make_receipt(
                    groups, separate_aliases, synthetic_scores(separate), role, split, score.PROFILE,
                    score.BOUNDS[role], binding))
                self.assertEqual(receipts[split]["binding"], binding.metadata())
                self.assertEqual([row | {"id": None} for row in receipts[split]["records"]],
                                 [expected | {"id": None}] * 2 + [{"id": None, "verdict": None, "status": "missing"}] * 2)

    def test_synthetic_check(self):
        self.assertEqual(score.check_requests(Tokenizer()),
                         {"logical_requests": 8, "unique_requests": 4, "overlength_logical_requests": 2})

    def test_scoring_cli_opens_no_gold_rows_and_invokes_the_engine_once(self):
        _, payload, coverage = artifacts()
        manifest, digest = coverage_bytes(coverage)
        panel = Path("/synthetic-panel")
        out = Path("/synthetic-receipts")
        options = ["score.py", "--role", "weak", "--freeze", "a" * 40, "--panel-dir", str(panel),
                   "--coverage-sha256", digest, "--output-dir", str(out)]
        read = Path.read_bytes
        opened = []

        def read_panel(path):
            if path.parent != panel:
                return read(path)
            opened.append(path.name)
            return {"coverage.json": manifest, "prompts.jsonl": payload}[path.name]

        with (patch("sys.argv", options), patch.dict(score.os.environ, {"SLURM_JOB_ID": "42"}),
              patch.object(Path, "read_bytes", read_panel), patch.object(Path, "exists", return_value=False),
              patch.object(score, "runtime", return_value=(None, {}, Tokenizer(), {}, None, {})),
              patch.object(score.scoring, "run_gpu", side_effect=lambda _model, _role, _cfg, requests, *_args:
                           synthetic_scores(requests)) as run,
              patch.object(score.scoring, "write_json") as write,
              patch.object(score.scoring, "digest", return_value="c" * 64), contextlib.redirect_stdout(io.StringIO())):
            score.main()

        self.assertEqual(opened, ["coverage.json", "prompts.jsonl"])
        run.assert_called_once()
        self.assertEqual([call.args[0].name for call in write.call_args_list], ["train-weak.json", "test-weak.json"])
        for call in write.call_args_list:
            self.assertEqual(call.args[1]["binding"]["panel_coverage_sha256"], digest)
            self.assertEqual(call.args[1]["binding"]["panel_gold_sha256"], "d" * 64)


if __name__ == "__main__":
    unittest.main()
