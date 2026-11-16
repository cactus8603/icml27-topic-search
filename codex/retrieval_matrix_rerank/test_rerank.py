"""[codex] Synthetic semantic checks; no models or benchmark effects."""
from copy import deepcopy
import unittest
from unittest.mock import patch
from pathlib import Path
import metrics as M
import runner as R
from backends import checked_scores


class Rules(unittest.TestCase):
    def test_strict_mcq_and_scpp_ties_fail(self):
        self.assertEqual(M.hit("negbench_mcq", [2, 1, 1, 1]), 1)
        self.assertEqual(M.hit("negbench_mcq", [2, 2, 1, 1]), 0)
        self.assertEqual(M.hit("scpp", [2, 2, 1]), 1)
        self.assertEqual(M.hit("scpp", [2, 1, 1]), 0)

    def test_released_cola_branches_have_different_comparison_axes(self):
        self.assertEqual(M.hit("cola_text_choice", [.8, .7, .9, .95]), 1)
        self.assertEqual(M.hit("cola_image_choice", [.8, .7, .9, .95]), 0)

    def test_invalid_score_or_pair_count_is_rejected(self):
        for values in ([1, 2], [float("nan"), 0, 0, 0]):
            with self.assertRaises(ValueError):
                M.hit("negbench_mcq", values)

    def test_macro_is_unweighted_not_row_weighted(self):
        examples = [{"id": str(i), "subset": "coco" if i < 2 else "voc", "task": "scpp",
                     "pairs": [{"id": f"{i}:{j}"} for j in range(3)]} for i in range(3)]
        scores = [{"id": f"{i}:{j}", "score": (2 if j < 2 and i == 2 else 1 if j == 2 else 0)}
                  for i in range(3) for j in range(3)]
        report, detail = M.evaluate(examples, scores)
        self.assertEqual(report["macro_accuracy"], .5)
        self.assertEqual(len(detail), 3)
        with self.assertRaises(ValueError):
            M.evaluate(examples, scores[:-1])
        with self.assertRaises(ValueError):
            M.evaluate(examples, scores + scores[:1])

    def test_native_response_coverage_and_raw_scores_are_preserved(self):
        values = checked_scores([{"id": "a"}], [.25], [[.25]])
        self.assertEqual(values, [{"id": "a", "score": .25, "official_raw": [.25]}])
        for value in ([], [float("inf")]):
            with self.assertRaises(ValueError):
                checked_scores([{"id": "a"}], value)

    def test_no_cross_benchmark_average(self):
        examples = [{"id": "s", "subset": "swap", "task": "scpp", "pairs": [{"id": str(i)} for i in range(3)]},
                    {"id": "n", "subset": "voc", "task": "negbench_mcq", "pairs": [{"id": str(i)} for i in range(3, 7)]}]
        report, _ = M.evaluate(examples, [{"id": str(i), "score": float(7-i)} for i in range(7)])
        self.assertNotIn("macro_accuracy", report)
        self.assertEqual(set(report["benchmarks"]), {"scpp", "negbench_mcq"})

    def test_smoke_proof_rejects_bad_values_and_identity(self):
        identity = {"model": "synthetic"}
        plan = {"smoke_absolute_tolerance": .01}
        proof = {"state": "PASS", "technical_only": True, "identity": identity, "batch1_vs_fixed_max_abs": .01}
        R.require_smoke(proof, identity, plan)
        for field, value in [("state", "FAILED"), ("technical_only", False), ("identity", {}),
                             ("batch1_vs_fixed_max_abs", .010001), ("batch1_vs_fixed_max_abs", -1),
                             ("batch1_vs_fixed_max_abs", float("nan")), ("batch1_vs_fixed_max_abs", None)]:
            bad = {**proof, field: value}
            with self.assertRaises(ValueError):
                R.require_smoke(bad, identity, plan)

    def test_source_gate_requires_review_commit_and_complete_executed_sources(self):
        path = Path(__file__).parent / "plan.core-mcq.v1.json"
        for plan, commit in [({}, "HEAD"), ({"state": "REVIEW_REQUIRED", "reviewed": False}, "a" * 40),
                             ({"state": "FROZEN", "reviewed": True, "source_files": {}}, "a" * 40)]:
            with self.assertRaises(ValueError):
                R.source_gate(path, plan, commit)

    def test_output_cannot_escape_runtime_or_access_sealed_panel(self):
        root = R.ROOT / "dl/retrieval-matrix-rerank/runs"
        plan = {"output_root": str(root)}
        self.assertEqual(R.output_path(root / "synthetic/main", plan), root / "synthetic/main")
        for path in (root, R.ROOT / "codex", root / "../outside", root / "retrieval-independent-panel"):
            with self.assertRaises(ValueError):
                R.output_path(path, plan)

    def test_existing_failed_smoke_never_runs_model_again(self):
        with patch.object(Path, "exists", return_value=True), patch.object(R.C, "read_json", return_value={"state": "FAILED"}), \
                patch.object(R.backends, "SwiftReleased") as engine:
            with self.assertRaises(ValueError):
                R.smoke(Path("synthetic"), {}, {}, {"smoke_absolute_tolerance": .01})
            engine.assert_not_called()


if __name__ == "__main__":
    unittest.main()
