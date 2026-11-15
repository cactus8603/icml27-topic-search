"""Synthetic checks against the exact registered evidence and runner sources."""

from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import ModuleType
import unittest
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parent))
from fix_labels import (APPLY, NEW_INVALID_TEST, NEW_LABELS, OLD_INVALID_TEST, OLD_LABELS,
                        RUNNER_LINE, evidence_source, fixture_source, runner_source)


EVIDENCE = "b54a86fb05e845e2c11139822beac8846c63c77a:killtests/t2i-rare-failure/scripts/evidence.py"
FIXTURE = "b54a86fb05e845e2c11139822beac8846c63c77a:killtests/t2i-rare-failure/scripts/test_evidence.py"
RUNNER = "4351ebb70f2e7ad72e0ec8b99658d9ed41c06567:codex/experiments/t2i-formal-replay/run.sbatch"
REPO = Path.cwd()
SCRATCH = Path(__file__).resolve().parent


def source(ref):
    return subprocess.check_output(["git", "-C", str(REPO), "show", ref])


def load_evidence(body):
    common = ModuleType("common")
    common.ROOT = SCRATCH
    common.load_json = lambda _: None
    common.prompts_by_id = lambda: None
    # Load compiled dependencies before patch.dict snapshots sys.modules.
    __import__("numpy")
    __import__("pandas")
    with patch.dict(sys.modules, {"common": common}):
        env = {}
        exec(compile(body, "frozen-evidence.py", "exec"), env)
    return env


class BlindLabels(unittest.TestCase):
    def test_frozen_sources_change_only_at_the_registered_lines(self):
        original = source(EVIDENCE)
        self.assertEqual(evidence_source(original).replace(NEW_LABELS, OLD_LABELS, 1), original)
        fixture = source(FIXTURE)
        self.assertEqual(fixture_source(fixture).replace(NEW_INVALID_TEST, OLD_INVALID_TEST, 1), fixture)
        runner = source(RUNNER)
        self.assertEqual(runner_source(runner).replace(APPLY + RUNNER_LINE, RUNNER_LINE, 1), runner)
        with self.assertRaisesRegex(ValueError, "does not match"):
            evidence_source(original + b"\n")

    def test_frozen_b_scoring_counts_only_1_as_positive(self):
        original = source(EVIDENCE)
        before, after = load_evidence(original), load_evidence(evidence_source(original))
        with TemporaryDirectory(dir=SCRATCH) as tmp:
            root = Path(tmp)
            (root / "review_key.csv").write_text("item,kind,model,prompt,seed\n0,neg,m,p,1\n1,rare_pos,m,p,2\n"
                                                  "2,rare_pos,m,p,3\n3,pct_pos,m,p,4\n")
            blind = root / "review_blind.csv"
            blind.write_text("item,label,note\n0,0,\n1,1,\n2,2,\n3,?,\n")
            with self.assertRaisesRegex(ValueError, "must all be"):
                before["read_review"](root)
            review = after["read_review"](root)
            self.assertEqual(review.label.tolist(), ["0", "1", "2", "?"])
            self.assertEqual(after["review_counts"](review), {
                "neg": {"n": 1, "rated_true": 0}, "pct_pos": {"n": 1, "rated_true": 0},
                "rare_pos": {"n": 2, "rated_true": 1},
            })
            for label in ("", "N/A", "-1"):
                blind.write_text(f"item,label,note\n0,{label},\n1,1,\n2,2,\n3,?,\n")
                with self.assertRaisesRegex(ValueError, "must all be"):
                    after["read_review"](root)
            blind.write_text("item,label,note\n0,0,\n0,1,\n2,2,\n3,?,\n")
            with self.assertRaisesRegex(ValueError, "duplicated"):
                after["read_review"](root)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        REPO, SCRATCH = map(Path, sys.argv[1:])
        del sys.argv[1:]
    unittest.main()
