"""Exercise the final join entirely with synthetic Slurm and panel receipts."""

from hashlib import sha256
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import join


def encode(obj):
    return (json.dumps(obj, sort_keys=True) + "\n").encode()


class SyntheticRun:
    jobs = join.Jobs("600", "601", "602", "603")
    freeze = "0" * 40
    initial = "a" * 64

    def __init__(self, root):
        self.root = root
        self.ledger = root / "submitted.log"
        self.ledger.write_text("".join(f"[codex] 2026-11-15 17:00 submit {job} inoculation-lambda fixture {self.freeze}\n"
                                      for job in self.jobs.names()))
        self.allocations = []
        for job, name in self.jobs.names().items():
            ids = [f"{job}_{i}" for i in range(51)] if job in ("600", "602") else [job]
            self.allocations.extend(f"{i}|COMPLETED|0:0|{name}" for i in ids)

    def query(self, cmd, *, text):
        assert cmd[0] == "sacct" and text
        return "\n".join(self.allocations) + "\n"

    def publish(self, path, body, log, declaration, path_key=None, sha_key="sha256"):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        raw = encode(body)
        digest = sha256(raw).hexdigest()
        target.write_bytes(raw)
        stdout = self.root / "tmp" / log
        stdout.parent.mkdir(exist_ok=True)
        record = declaration | {sha_key: digest}
        if path_key is not None:
            record[path_key] = str(target)
        stdout.write_bytes(b"synthetic source job\n" + encode(record))
        return digest

    def build(self, wrong_initial=False):
        families = ("ip_single", "ip_bank", "pps", "layer_restricted")
        arms = {"none": "none"} | {f"{family}_{i}": family for family in families for i in range(4)}
        scopes = {arm: join.PHM_SCOPES[family] for arm, family in arms.items()}
        recipes = {name: {"family": family, "recipe_sha256_by_seed": {str(seed): str(seed) * 64 for seed in join.SEEDS},
                          "training_sha256": "b" * 64} for name, family in arms.items()}
        phm_interface = {"calibration_rows": 10_000, "arms": [{"id": arm, "scope": scope} for arm, scope in scopes.items()]}
        assets = {"model_snapshot": "synthetic"}
        splits = {"phm-calibration": {"sha256": "c" * 64}}
        manifest = {"stage": join.DESIGN_STAGE, "train_seeds": list(join.SEEDS), "cpu_pass": True,
                    "training_outcomes_opened": False, "splits": splits, "assets": assets, "arms": recipes,
                    "phm_interface": phm_interface}
        self.manifest_sha = self.publish(f"data/lambda-gcd/family-design-{join.DATA_JOB}/manifest.json", manifest,
                                         f"{join.PREFIX}preflight-{join.DATA_JOB}.log", {}, sha_key="manifest_sha256")
        ready = {"schema": 1, "scope": join.READY_SCOPE, "freeze": self.freeze,
                 "family_design": {"manifest_sha256": self.manifest_sha, "split_sha256": {"phm-calibration": "c" * 64}},
                 "validated_author_configs": 51, "assets": assets}
        ready_sha = self.publish(f"runs/{join.PREFIX}train-ready/{join.READY_JOB}/summary.json", ready,
                                 f"{join.PREFIX}train-ready-{join.READY_JOB}.log", {"scope": join.READY_SCOPE, "gpu_cells": 8},
                                 path_key="output")
        phm_scores = {arm: 0 if arm == "none" else 3 - int(arm[-1]) for arm in arms}
        phm = {"schema": 1, "mode": "score", "job": join.PHM_JOB, "freeze": self.freeze, "design_sha256": self.manifest_sha,
               "calibration_sha256": "c" * 64, "calibration_scope": "gcd_neutral_adaptation", "model_snapshot": "synthetic",
               "calibration_rows": 10_000, "method_scope_by_arm": scopes, "predicted_leakage_scores_by_arm": phm_scores}
        self.phm_path = f"runs/{join.PREFIX}phm/{join.PHM_JOB}/score.json"
        self.publish(self.phm_path, phm, f"lambda-gcd-phm-{join.PHM_JOB}.log", {"job": join.PHM_JOB, "mode": "score"},
                     sha_key="receipt_sha256")
        common = {"schema": 1, "freeze": self.freeze, "manifest_sha256": self.manifest_sha,
                  "training_readiness_sha256": ready_sha}
        cells, bindings, runs, arms_table = {}, [], [], []
        for arm, family in arms.items():
            index = 0 if arm == "none" else int(arm[-1])
            low = index / 8
            arms_table.append({"id": arm, "family": family, "id_matched_all_seeds": index < 2,
                               "leakage": {"lower": low, "upper": low}})
            for seed in join.SEEDS:
                task = len(runs)
                key = f"{arm}:{seed}"
                identity = common | {"arm": arm, "seed": seed, "task": task, "recipe_sha256": str(seed) * 64,
                                     "training_sha256": "b" * 64, "initial_A_B_sha256": self.initial}
                values = dict.fromkeys(join.SCORE_FIELDS, index)
                record = identity | {"scope": "gcd_family_step0_cell", "array": 600, "scores": values}
                digest = self.publish(f"runs/{join.PREFIX}step0/600/cell-{task:02d}.json", record,
                                      f"{join.PREFIX}step0-600_{task}.log", {"scope": record["scope"]}, path_key="path")
                cells[key] = {"arm": arm, "seed": seed, "initial_A_B_sha256": self.initial, "scores": values,
                              "cell_receipt_sha256": digest}
                runs.append({"arm": arm, "seed": seed, "leakage_bounds": [low, low],
                             "id_matching_status": "matched" if index < 2 else "outside"})
                bindings.append((key, identity, digest))
        step = common | {"scope": join.STEP_SCOPE, "array": 600, "cells": cells,
                         "method_direction": "smaller_predicts_less_leakage"}
        step_sha = self.publish(f"runs/{join.PREFIX}step0/600/summary.json", step,
                                 f"{join.PREFIX}step0-join-601.log", {"scope": join.STEP_SCOPE}, path_key="path")
        hashes = {}
        for task, (key, identity, declared) in enumerate(bindings):
            record = identity | {"scope": join.OUTCOME_SCOPE, "array": "602", "prediction_panel_sha256": step_sha,
                                 "prediction_cell_sha256": declared}
            if wrong_initial and task == 50:
                record["initial_A_B_sha256"] = "e" * 64
            hashes[key] = self.publish(f"runs/{join.PREFIX}outcomes/602/cell-{task}.json", record,
                                       f"{join.PREFIX}outcomes-602_{task}.log", {"scope": join.OUTCOME_SCOPE}, path_key="output")
        outcome = common | {"scope": join.OUTCOME_SCOPE, "array": "602", "prediction_panel_sha256": step_sha,
                            "cell_receipt_sha256": hashes, "arms": arms_table, "runs": runs,
                            "defended_range_lower": .375, "leakage_reliability": {}}
        self.publish(f"runs/{join.PREFIX}outcomes/602/summary.json", outcome,
                     f"{join.PREFIX}outcome-judge-603.log", {"scope": join.OUTCOME_SCOPE}, path_key="output")

    def run(self):
        with patch.object(join, "DATA_SHA", self.manifest_sha):
            return join.run(self.root, self.ledger, self.jobs, self.freeze, "604", self.query)


class JoinTests(unittest.TestCase):
    def setUp(self):
        directory = os.environ.get("GCD_JOIN_TEST_TMP", str(Path.cwd()))
        temp = TemporaryDirectory(prefix="synthetic-gcd-join-", dir=directory)
        self.addCleanup(temp.cleanup)
        self.fixture = SyntheticRun(Path(temp.name).resolve())

    def test_complete_join_names_distinct_phm_scopes_and_id_sensitivity(self):
        self.fixture.build()
        result = self.fixture.run()
        methods = result["all_defended"]["comparisons"]
        self.assertEqual(methods["phm_original_formula_gcd"]["macro_eligible_families"], ["ip_single"])
        self.assertEqual(methods["phm_weighted_bank_gcd"]["macro_eligible_families"], ["ip_bank"])
        self.assertEqual(methods["phm_prefix_only_gcd"]["macro_eligible_families"], ["layer_restricted", "pps"])
        combined = methods["phm_gcd_adaptation_all"]
        self.assertTrue(combined["macro_covers_all_families_and_arms"])
        self.assertEqual(combined["macro_common_pairs"], 24)
        self.assertEqual(combined["macro"]["selected_leakage"]["lambda"]["upper"], 0)
        self.assertEqual(len(result["id_matched_arm_ids"]), 8)
        self.assertEqual(result["id_matched_sensitivity"]["comparisons"]["phm_gcd_adaptation_all"]["macro_common_pairs"], 4)

    def test_tampered_receipt_fails_even_after_successful_jobs(self):
        self.fixture.build()
        (self.fixture.root / self.fixture.phm_path).write_bytes(b"{}\n")
        with self.assertRaisesRegex(ValueError, "Receipt SHA-256 differs"):
            self.fixture.run()

    def test_changed_initialization_fails_even_with_valid_receipt_digest(self):
        self.fixture.build(wrong_initial=True)
        with self.assertRaisesRegex(ValueError, "Unexpected training outcome"):
            self.fixture.run()

    def test_unfinished_cell_stops_before_opening_the_ledger(self):
        self.fixture.build()
        self.fixture.allocations[-1] = self.fixture.allocations[-1].replace("COMPLETED", "PENDING")
        self.fixture.ledger.unlink()
        with self.assertRaisesRegex(ValueError, "successful source jobs"):
            self.fixture.run()


if __name__ == "__main__":
    unittest.main()
