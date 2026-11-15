"""Synthetic checks of absolute risk, state availability, and ASDiv-A source claims."""

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import point_envelope as envelope
from evaluation import compare_round2_binary


def row(key, truth, strong, *, weak=1, slot="base-0"):
    return {"id": key, "question_id": key, "dataset": envelope.asdiv.DATASET,
            "solver_model": slot, "gold": truth, "weak": weak, "strong": strong,
            "gold_status": "verified_numeric" if truth is not None else "length",
            "weak_status": "observed" if weak is not None else "missing",
            "strong_status": "observed" if strong is not None else "missing"}


def split(name, rows):
    return SimpleNamespace(name=name, binding="synthetic", rows=tuple(rows),
                           question_ids={row["question_id"] for row in rows})


class EnvelopeTests(unittest.TestCase):
    def test_point_constraints_keep_unknown_truth_and_certify_both_optima(self):
        warm = [row("warm", True, 1)]
        test = [row("wrong", False, 0), row("correct", True, 1), row("unknown", None, None)]
        results = envelope.solve(warm, test)
        direct, cascade = (results[f"{name}_tree_optimum"] for name in ("direct", "cascade"))
        expected_direct = 1 + envelope.WARM_COST - 2 * 0.11 / (1 + 0.89)
        self.assertAlmostEqual(direct["new_cost"], expected_direct)
        self.assertLess(cascade["new_cost"], direct["new_cost"])
        self.assertGreater(cascade["new_strong"], 0)
        for optimum in (direct, cascade):
            self.assertAlmostEqual(optimum["new_cost"], optimum["dual_cost_lower_bound"])
            for name in ("fa", "fr"):
                self.assertEqual(optimum[f"old_{name}_upper"], 0)
                self.assertLessEqual(optimum[f"new_{name}_upper"], 0.11 + envelope.lp.TOL)
                self.assertLessEqual(optimum[f"{name}_increase_upper"], 0.11 / 0.89 + envelope.lp.TOL)
        delta = direct["new_cost"] - cascade["new_cost"]
        self.assertAlmostEqual(results["additional_strong_information_saving_pp_vs_always_expert"],
                               100 * delta / (1 + envelope.WARM_COST))

    def test_holdout_only_states_and_missing_weak_verdicts_require_expert(self):
        warm = [row("warm", True, 1)]
        test = [row("wrong", False, 0), row("correct", True, 1),
                row("separate-solver", False, 0, slot="oat-0"),
                row("no-weak", True, 1, weak=None)]
        results = envelope.solve(warm, test)
        self.assertEqual(results["development_observed_weak_states"], 1)
        for name in ("direct", "cascade"):
            optimum = results[f"{name}_tree_optimum"]
            self.assertGreaterEqual(optimum["new_gold"], 0.5)
            self.assertLessEqual(optimum["new_strong"], 0.5)

    def test_published_point_ineligible_strategies_are_only_reproduced(self):
        config, _ = envelope.load_config()
        train = split("train", [row(f"warm-{i}", bool(i % 2), i % 2, weak=i % 2) for i in range(1000)])
        test = split("test", [row(f"held-{i}", bool(i % 2), 1) for i in range(120)])
        fit = envelope.lp.fit_round2_binary(train, config)
        comparison = compare_round2_binary(fit, test, {"draws": 3, "seed": 0, "tail": 0.01})
        publication = {"policy_comparisons": comparison,
                       "mechanical_equal_warm_primary": {"equal_risk_supported": False}}
        diagnostics = envelope.published_points(train, test, config, publication)
        self.assertFalse(diagnostics["same_absolute_point_cap_eligible"])
        self.assertIn("old_fa_upper", diagnostics["same_absolute_point_cap_violations"])
        self.assertIn("new_fa_upper", diagnostics["same_absolute_point_cap_violations"])
        # The LP remains well defined with its safe expert baseline.
        self.assertEqual(envelope.solve(train.rows, test.rows)["cascade_tree_optimum"]["old_fa_upper"], 0)
        comparison["comparisons"][envelope.lp.REPORT_KEY]["risk_and_cost"]["new_cost"]["point"] += 0.1
        with self.assertRaisesRegex(ValueError, "Rebuilt published policy point differs"):
            envelope.published_points(train, test, config, publication)

    def test_check_only_reads_no_source_outcomes(self):
        args = ["point_envelope.py", "--freeze", "d" * 40, "--check-only"]
        output = StringIO()
        with (patch.object(sys, "argv", args), patch.dict(envelope.os.environ, {"SLURM_JOB_ID": ""}),
              patch.object(envelope, "read_inputs", side_effect=AssertionError), redirect_stdout(output)):
            envelope.main()
        metadata = json.loads(output.getvalue())
        self.assertEqual(metadata["stage"], "static_protocol_checked")
        self.assertEqual(metadata["source_evaluation_sha256"], envelope.SOURCE_SHA)
        self.assertEqual(metadata["source_coverage_sha256"], envelope.COVERAGE_SHA)
        self.assertNotIn("coverage", metadata)

    def test_inactive_job_fails_before_source_access(self):
        args = ["point_envelope.py", "--freeze", "d" * 40, "--job", "456"]
        with (patch.object(sys, "argv", args), patch.dict(envelope.os.environ, {"SLURM_JOB_ID": "123"}),
              patch.object(envelope, "read_inputs", side_effect=AssertionError), patch.object(sys, "stderr", StringIO())):
            with self.assertRaises(SystemExit) as error:
                envelope.main()
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
