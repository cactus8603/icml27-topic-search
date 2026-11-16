"""[codex] Reject scientific changes smuggled into a technical repair."""
import copy
from pathlib import Path
import unittest
from unittest.mock import patch

from codex.retrieval_matrix_core_runtime_fix import build_plan, runner

ROOT = Path(__file__).resolve().parents[2]


class RepairPlan(unittest.TestCase):
    def test_original_protocol_retained_and_changes_rejected(self):
        for relative in ("codex/retrieval_matrix/p0-qwen3-batch1.json", "codex/retrieval_matrix_mcmr/p0-mcmr-qwen3-v2.json"):
            plan = build_plan.build(ROOT/relative)
            runner.validate_repair(plan)
            for change in (
                lambda p: p["models"]["core2"]["batches"].update(image=2),
                lambda p: p["models"]["core2"]["processor"].update(max_pixels=123),
                lambda p: p["settings"].update(seed=1),
                lambda p: p["dependencies"].update(packages={}),
                lambda p: p["datasets"].clear()):
                value = copy.deepcopy(plan)
                change(value)
                with self.assertRaises(ValueError):
                    runner.validate_repair(value)

    def test_no_import_side_effect_and_no_central_bypass(self):
        from codex.retrieval_matrix import runner as R
        self.assertIsNot(R.models.OfficialModel, runner.compat.CompatibleOfficialModel)
        with patch.dict("os.environ", {}, clear=True), self.assertRaises(ValueError):
            runner.runtime_environment({})


if __name__ == "__main__":
    unittest.main()
