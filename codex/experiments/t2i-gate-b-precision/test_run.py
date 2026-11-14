"""Synthetic checks for coverage and the original qualification rule."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run


GATE = {"prompts": ["E1", "N1", "B1"], "n_per_prompt": 10_000, "min_hits": 100,
        "max_rate": 0.05, "min_distinct_scores": 1_000, "thresholds": [0.5, 0.9, 0.99, 0.999]}


def batches(values):
    for prompt, span in run.expected_batches(GATE["prompts"], run.SEEDS, 32):
        begin = span.start - run.SEEDS.start
        yield prompt, span, values[prompt][begin:begin + len(span)]


def synthetic_hits(count, *, coarse=False):
    low = (np.full(10_000 - count, 0.25, dtype=np.float32) if coarse
           else np.linspace(0.001, 0.49, 10_000 - count, dtype=np.float32))
    return np.concatenate((low, np.full(count, 0.5, dtype=np.float32)))


class GateBPrecisionTest(unittest.TestCase):
    def test_coverage_and_inclusive_hit_limits(self):
        result = run.evaluate(batches({"E1": synthetic_hits(99), "N1": synthetic_hits(100),
                                       "B1": synthetic_hits(500)}), GATE, run.SEEDS, 32)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual([r["qualified"] for r in result["rows"]], [[], [0.5], [0.5]])
        for row in result["rows"]:
            self.assertEqual(row["n"], 10_000)
            self.assertGreaterEqual(row["distinct"], 1_000)

    def test_complete_but_no_qualifying_prompt(self):
        result = run.evaluate(batches({"E1": synthetic_hits(99), "N1": synthetic_hits(501),
                                       "B1": synthetic_hits(100, coarse=True)}), GATE, run.SEEDS, 32)
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["rows"][2]["distinct"], 2)
        self.assertEqual(result["rows"][2]["hits"]["0.5"], 100)

    def test_missing_extra_and_repeated_batches(self):
        complete = list(batches({p: synthetic_hits(100) for p in GATE["prompts"]}))
        with self.assertRaises(ValueError):
            run.evaluate(complete[:-1], GATE, run.SEEDS, 32)
        with self.assertRaises(ValueError):
            run.evaluate([*complete, complete[-1]], GATE, run.SEEDS, 32)
        with self.assertRaisesRegex(ValueError, "registered prompt and seed range"):
            run.evaluate([complete[0], *complete[:-1]], GATE, run.SEEDS, 32)
        self.assertEqual(complete[-1][1], range(19_984, 20_000))

    def test_nonfinite_or_wrong_dtype_cannot_qualify(self):
        complete = list(batches({p: synthetic_hits(100) for p in GATE["prompts"]}))
        prompt, span, batch = complete[-1]
        batch[0] = np.nan
        with self.assertRaisesRegex(ValueError, "Invalid FP32 probability batch"):
            run.evaluate(complete, GATE, run.SEEDS, 32)
        complete[-1] = (prompt, span, np.zeros(len(span), dtype=np.float16))
        with self.assertRaisesRegex(ValueError, "Invalid FP32 probability batch"):
            run.evaluate(complete, GATE, run.SEEDS, 32)

    def test_score_comparison_at_representable_cutoff(self):
        cutoff = np.float32(0.9)
        near = synthetic_hits(0)
        near[-100:] = cutoff
        near[-101] = np.nextafter(cutoff, np.float32(0))
        result = run.evaluate(batches({"E1": near, "N1": synthetic_hits(0), "B1": synthetic_hits(0)}),
                              GATE, run.SEEDS, 32)
        self.assertEqual(result["rows"][0]["hits"]["0.9"], 100)


if __name__ == "__main__":
    unittest.main()
