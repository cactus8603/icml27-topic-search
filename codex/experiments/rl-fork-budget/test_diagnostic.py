from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diagnostic as diag


def outcomes(correct, censored):
    return {"correct": correct, "stop_tokens": [None if item else 1 for item in censored]}


class DiagnosticTests(unittest.TestCase):
    def test_recensor_keeps_stop_at_cap_and_removes_later_stop(self):
        eos = diag.fork.STOP_IDS[0]
        source = {"correct": [1, 1], "tokens": [3, 4], "stop_tokens": [eos, eos]}
        self.assertEqual(diag.restrict(source, 3), {"correct": [1, 0], "tokens": [3, 3],
                                                   "stop_tokens": [eos, None]})
        self.assertEqual(diag.restrict(source, 2), {"correct": [0, 0], "tokens": [2, 2],
                                                   "stop_tokens": [None, None]})
        self.assertEqual(diag.restrict(source, 4), source)

    def test_sharp_binary_bounds_do_not_guess_truncated_answers(self):
        short = outcomes([0, 1], [True, False])
        self.assertEqual(diag.paired_metrics(short, short), [1, 1, 0, 0, 1])
        recovered = outcomes([1, 1], [False, False])
        self.assertEqual(diag.paired_metrics(short, recovered), [1, 0, -1, 0, 0])
        both_unresolved = outcomes([0, 0], [True, True])
        self.assertEqual(diag.paired_metrics(both_unresolved, both_unresolved), [0, 0, 0, -1, 1])

    def test_full_denominator_and_three_readouts(self):
        unresolved = outcomes([0, 1], [True, False])
        stable = outcomes([0, 1], [False, False])
        tied = outcomes([0, 0], [False, False])
        nonfork = {"search": "base_stop"}

        def repeats(long):
            row = {"search": "discordance", "short": unresolved, "long": long}
            return [row, nonfork] * 64

        positive = diag.summarize(repeats(stable))
        self.assertEqual(positive["paired_differences"]["long"]["rl_minus_base_pp"], 50)
        self.assertEqual(positive["n_forked"], 64)
        self.assertEqual(positive["judgement"], "advantage_robust_to_censoring")
        self.assertEqual(diag.summarize(repeats(tied))["judgement"], "advantage_below_reference")
        self.assertEqual(diag.summarize(repeats(unresolved))["judgement"], "inconclusive")


if __name__ == "__main__":
    unittest.main()
