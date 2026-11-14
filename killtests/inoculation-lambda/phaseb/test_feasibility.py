"""Synthetic-only checks of receipt acceptance and Gate 2/3 calculations."""

from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import unittest
from unittest.mock import mock_open, patch

from phaseb.feasibility import (
    AUTHOR_ARM, AUTHOR_EVAL, FAMILIES, ICC_ARMS, ICC_SEEDS, icc_1_1, reliability, reproduction, reviewed_receipt,
)


BINDING = "e" * 64


def author_scores(base=1, none=80, ip=1):
    return {
        "schema": 1, "stage": "grant_spanish_table13", "evaluation": dict(AUTHOR_EVAL), "inputs_sha256": BINDING,
        "runs": {
            arm: {"author_run": AUTHOR_ARM[arm], "artifact_sha256": f"{index:064x}",
                  "scores_0_100": [[mean] * 10 for _ in range(20)]}
            for index, (arm, mean) in enumerate(zip(AUTHOR_ARM, (base, none, ip), strict=True), 1)
        },
    }


def reliability_scores(cell):
    return {
        "schema": 1, "stage": "spanish_caps_icc", "metric": "uppercase_alphabetic_fraction",
        "design_sha256": BINDING,
        "runs": [
            {"arm": arm, "seed": seed, "artifact_sha256": f"{index * 3 + seed:064x}",
             "family_means": dict.fromkeys(FAMILIES, cell(index, seed))}
            for index, arm in enumerate(ICC_ARMS) for seed in ICC_SEEDS
        ],
    }


class FeasibilityTests(unittest.TestCase):
    def test_receipt_uses_exact_reviewed_bytes(self):
        payload = b'{"schema":1}'
        with patch.object(Path, "open", mock_open(read_data=payload)):
            self.assertEqual(reviewed_receipt(Path("receipt.json"), sha256(payload).hexdigest())[0], {"schema": 1})
            with self.assertRaises(ValueError):
                reviewed_receipt(Path("receipt.json"), sha256(payload + b"\n").hexdigest())
        duplicated = b'{"schema":1,"schema":2}'
        with patch.object(Path, "open", mock_open(read_data=duplicated)):
            with self.assertRaises(ValueError):
                reviewed_receipt(Path("receipt.json"), sha256(duplicated).hexdigest())

    def test_reproduction_requires_increase_from_base_and_drop_from_none(self):
        def check(evidence):
            return reproduction(evidence, BINDING, {"base": 1.2, "none": 79.2, "ip": 0.9}, 0.5)

        observed = check(author_scores())
        self.assertEqual(observed["status"], "PASS")
        self.assertAlmostEqual(observed["required_effects_pp"][0], 39.0)
        self.assertAlmostEqual(observed["required_effects_pp"][1], 39.15)
        self.assertEqual(check(author_scores(base=75))["status"], "UNTESTABLE")
        self.assertEqual(check(author_scores(ip=45))["status"], "UNTESTABLE")

    def test_reproduction_rejects_proxy_and_unbound_or_partial_evidence(self):
        evidence = author_scores()
        mutations = (
            lambda row: row["evaluation"].update(judge="proxy"),
            lambda row: row.update(inputs_sha256="1" * 64),
            lambda row: row["runs"]["none"]["scores_0_100"].pop(),
            lambda row: row["runs"]["none"]["scores_0_100"][0].append(1),
            lambda row: row["runs"]["none"]["scores_0_100"][0].__setitem__(0, float("nan")),
            lambda row: row["runs"]["none"]["scores_0_100"][0].__setitem__(0, True),
            lambda row: row["runs"]["none"].update(artifact_sha256="0" * 63 + "1"),
        )
        for change in mutations:
            invalid = deepcopy(evidence)
            change(invalid)
            with self.assertRaises(ValueError):
                reproduction(invalid, BINDING, {"base": 1.2, "none": 79.2, "ip": 0.9}, 0.5)

    def test_icc_known_examples(self):
        self.assertAlmostEqual(icc_1_1([[0, 1, 2], [2, 3, 4]]), 5 / 8)
        result = reliability(reliability_scores(lambda i, s: i / 8 + 0.01 * s), BINDING, 0.8)
        self.assertEqual(result["status"], "PASS")
        result = reliability(reliability_scores(lambda i, s: s / 2), BINDING, 0.8)
        self.assertAlmostEqual(result["icc_1_1"], -0.5)
        self.assertEqual(result["status"], "UNTESTABLE")
        result = reliability(reliability_scores(lambda i, s: 0.25), BINDING, 0.8)
        self.assertIsNone(result["icc_1_1"])
        self.assertEqual(result["status"], "UNTESTABLE")

    def test_icc_rejects_duplicate_and_incomplete_panel(self):
        evidence = reliability_scores(lambda i, s: i / 8 + 0.01 * s)
        mutations = (
            lambda row: row["runs"].pop(),
            lambda row: row["runs"].append(dict(row["runs"][0])),
            lambda row: row["runs"][0]["family_means"].pop(FAMILIES[0]),
            lambda row: row["runs"][0]["family_means"].update({FAMILIES[0]: 1.01}),
            lambda row: row["runs"][0].update(seed=False),
            lambda row: row["runs"][0].update(artifact_sha256=row["runs"][1]["artifact_sha256"]),
        )
        for change in mutations:
            invalid = deepcopy(evidence)
            change(invalid)
            with self.assertRaises(ValueError):
                reliability(invalid, BINDING, 0.8)


if __name__ == "__main__":
    unittest.main()
