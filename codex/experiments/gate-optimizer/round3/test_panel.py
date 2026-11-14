import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import panel


class PanelTest(unittest.TestCase):
    def test_plan_and_frozen_thresholds(self):
        _, calibration, seeds = panel.inputs()
        shards = [panel.shard_seeds(seeds, i) for i in range(4)]
        self.assertEqual([len(shard) for shard in shards], [12] * 4)
        self.assertEqual(len(set(seeds)), 48)
        self.assertEqual(seeds, panel.inputs()[2])
        self.assertEqual(calibration["separate_validation_plan"], {
            "seeds": seeds, "min_damage_events": 30, "min_scored_seeds": 6, "min_average_increment_auc": 0.03,
        })
        with self.assertRaisesRegex(ValueError, "independent"):
            panel.panel_seeds([seeds[0]])

    def test_schema_rejects_wrong_identity_and_nonfinite_json(self):
        raw = json.dumps(self.branch("muon", 12345, 1)).encode()
        with self.assertRaisesRegex(ValueError, "identity"):
            panel.verify_branch(raw, "sandwich", 12345, 1)
        with self.assertRaisesRegex(ValueError, "invalid JSON constant"):
            panel.strict_json(b'{"x": NaN}')
        with self.assertRaisesRegex(ValueError, "rescue schema"):
            panel.verify_rescue({"status": "measured", "nll_control": 1.0})

    def test_complete_single_analysis_and_receipt_tamper(self):
        base, calibration, seeds = panel.inputs(smoke=True)
        with tempfile.TemporaryDirectory(dir=panel.STAGE) as temp:
            root = Path(temp)
            args = SimpleNamespace(array_id="12345", commit="a" * 40, tree="b" * 40,
                                   output=root / "run", scratch=root)
            for index in range(4):
                folder = args.output / f"shard-{index}"
                folder.mkdir(parents=True)
                record = panel.provenance(args, seeds, index)
                for seed in record["seeds"]:
                    for arm in panel.STRONG:
                        for multiplier in (1, 4):
                            path = folder / f"{arm}-{seed}-{multiplier}x.json"
                            panel.write_json(path, self.branch(arm, seed, multiplier))
                hashes = panel.branch_files(folder, record["seeds"])
                panel.write_json(folder / "receipt.json", record | {
                    "branches": hashes, "torch": "fixture", "device": "NVIDIA H200", "producer_job": str(100 + index),
                })
            receipts, (version, device) = panel.collect(args, seeds, root / "branches")
            metadata = {"config": calibration, "training": base["training"], "smoke": True,
                        "slurm_id": args.array_id, "frozen_commit": args.commit, "frozen_tree": args.tree,
                        "config_sha256": panel.CONFIG_SHA, "prereg_sha256": "fixture", "corpus_sha256": None, "device": device}
            result = panel.validation_summary(metadata, root / "branches")
            self.assertEqual(len(receipts), 4)
            self.assertEqual(version, "fixture")
            self.assertEqual(len(result["pressure"]["cells"]), 144)
            self.assertEqual(result["pressure"]["damaged_steps"], 48)
            self.assertEqual(result["pressure"]["signal"]["scored_seeds"], 48)
            self.assertEqual(result["status"], "signal_pass")
            path = args.output / "shard-3" / f"muon-{seeds[-1]}-4x.json"
            with path.open("a") as stream:
                stream.write("\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                panel.collect(args, seeds, root / "tampered")
            path.unlink()
            with self.assertRaisesRegex(ValueError, "missing expected branch"):
                panel.collect(args, seeds, root / "missing")

    @staticmethod
    def branch(arm, seed, multiplier):
        branch = {
            "arm": arm, "seed": seed, "multiplier": multiplier, "branch_wall_seconds": 1.0,
            "trace": [{"step": step, "train_nll": 1.0, "grad_norm": 0.1, "lr": 0.01} for step in range(1, 5)],
            "evaluations": [{"step": step, "dev_nll": 1.0} for step in (0, 2, 4)],
            "probes": [],
        }
        if multiplier == 4:
            selected_arm = panel.STRONG[seed % 3]
            selected_step = 2 + (seed // 3) % 3
            for step in range(1, 5):
                damage = arm == selected_arm and step == selected_step
                branch["probes"].append({key: 0.1 for key in panel.PROBE_FIELDS} | {
                    "step": step, "delta_nll": 0.1 if damage else 0.0,
                    "nll_before": 1.0, "nll_after": 1.1 if damage else 1.0,
                    "write_q95": 1.0 if damage else 0.0,
                })
        return branch


if __name__ == "__main__":
    unittest.main()
