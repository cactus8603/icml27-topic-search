"""Verify isolated planning and allocation validation without opening target scores."""

import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline


class PipelineTests(unittest.TestCase):
    def test_appending_read_log_does_not_invalidate_frozen_gate_verifier(self):
        with TemporaryDirectory(dir=Path(__file__).parent) as path:
            root = Path(path)
            repo, archive = root / "repo", root / "archive"
            live_gate = repo / pipeline.GATE_DIR
            live_gate.mkdir(parents=True)
            verifier = live_gate / "run.py"
            protocol = live_gate / "PREREG.md"
            verifier.write_text("import sys\nassert sys.argv[-2:] == ['--verify-job', '26663']\n")
            protocol.write_text("Frozen protocol\n")
            git = ["git", "-C", str(repo), "-c", "user.name=Test", "-c", "user.email=test@example.invalid"]
            subprocess.run(git + ["init", "-q"], check=True)
            subprocess.run(git + ["add", "--", str(verifier), str(protocol)], check=True)
            subprocess.run(git + ["-c", "commit.gpgsign=false", "commit", "-qm", "Freeze"], check=True)
            freeze = subprocess.check_output(git + ["rev-parse", "HEAD"], text=True).strip()
            relative_verifier = pipeline.GATE_DIR / "run.py"
            archived_verifier = archive / relative_verifier
            archived_verifier.parent.mkdir(parents=True)
            archived_verifier.write_bytes(pipeline.git_file(repo, freeze, relative_verifier))

            with protocol.open("a") as file:
                file.write("[codex] The verified receipt was read.\n")
            subprocess.run(git + ["-c", "commit.gpgsign=false", "commit", "-qm", "Record read", "--", str(protocol)], check=True)
            old_guard = subprocess.run(git + ["diff", "--quiet", freeze, "HEAD", "--", str(pipeline.GATE_DIR)])
            self.assertEqual(old_guard.returncode, 1)
            with patch.object(pipeline, "GATE_FREEZE", freeze):
                pipeline.verify_frozen_gate(repo, archive)
                archived_verifier.write_text("raise SystemExit('unexpected verifier execution')\n")
                with self.assertRaisesRegex(ValueError, "differs from the registered replay"):
                    pipeline.verify_frozen_gate(repo, archive)

    def test_replay_receipt_accepts_shared_mount_alias_not_receipt_symlink(self):
        with TemporaryDirectory(dir=Path(__file__).parent) as path:
            root = Path(path)
            shared = root / "shared"
            parent = shared / f"runs/t2irare/gate-b-original-replay/{pipeline.GATE_JOB}"
            parent.mkdir(parents=True)
            receipt = parent / "precision_replay_verified.json"
            receipt.write_text("{}")
            alias = root / "mount-alias"
            alias.symlink_to(shared, target_is_directory=True)
            self.assertEqual(pipeline.replay_receipt(alias), receipt)
            receipt.rename(parent / "different.json")
            receipt.symlink_to(parent / "different.json")
            with self.assertRaisesRegex(ValueError, "aliased"):
                pipeline.replay_receipt(alias)

    def test_control_floor_is_the_only_criterion_change(self):
        criteria = {"A": {"min_hits": 5}, "B": {"n_pct_positive_controls": 50, "control_min_pct_positive_rated_true": 0.8}}
        with TemporaryDirectory(dir=Path(__file__).parent) as path:
            upstream = Path(path)
            config = upstream / "configs"
            config.mkdir()
            file = config / "criteria.json"
            file.write_text(json.dumps(criteria))
            result = pipeline.formal_configuration(upstream)
            self.assertEqual(result["B"].pop("percentage_cell_min_rate"), 0.01)
            self.assertEqual(result, criteria)
            with self.assertRaisesRegex(ValueError, "already specify"):
                pipeline.formal_configuration(upstream)

    def test_stage2_uses_exact_original_shards(self):
        criteria = {"screen": {"stage1_n": 2000, "stage2a_n": 20000, "stage2b_n": 100000, "shard": 25000},
                    "frozen_event": {"definition": "consensus2"}}
        spec = {"stage": "2b", "event": "consensus2", "cells": [{"model": "uce", "prompt": "B1"}],
                "start": 20000, "target": 100000}
        tasks = [("uce", "B1", start, min(start + 25000, 100000)) for start in (20000, 45000, 70000, 95000)]
        with TemporaryDirectory(dir=Path(__file__).parent) as path:
            run = Path(path)
            configs = run / "configs"
            configs.mkdir()
            (configs / "stage2b_cells.json").write_text(json.dumps(spec))
            tsv = configs / "stage2b_tasks.tsv"
            tsv.write_text("".join("\t".join(map(str, row)) + "\n" for row in tasks))
            first = pipeline.plan_identity(run, "2b")
            self.assertEqual(pipeline.stage_tasks(run, criteria, "2b"), tasks)
            tsv.write_text("uce\tB1\t20000\t100000\n")
            self.assertNotEqual(pipeline.plan_identity(run, "2b"), first)
            with self.assertRaisesRegex(ValueError, "differs"):
                pipeline.stage_tasks(run, criteria, "2b")

    def test_retries_include_allocations_and_exclude_cancelled_pending(self):
        log = ("30|10_0|t2ir-formal-replay|FAILED|1:0|40|gres/gpu=1|first\n"
               "31|20_0|t2ir-formal-replay|COMPLETED|0:0|70|gres/gpu=1|next\n"
               "20|20_[1-3]|t2ir-formal-replay|CANCELLED|0:0|0||none\n")
        attempts = pipeline.parse_allocations(["10", "20"], log)
        self.assertEqual(sum(row[-1] for row in attempts), 110)
        with self.assertRaisesRegex(ValueError, "still active"):
            pipeline.parse_allocations(["10", "20"], log.replace("COMPLETED", "RUNNING"))
        with self.assertRaisesRegex(ValueError, "missing"):
            pipeline.parse_allocations(["10", "20", "70"], log)


if __name__ == "__main__":
    unittest.main()
