"""Synthetic regression tests for the incremental signal diagnostic."""

import unittest

import numpy as np

from data import Candidate
from replay import diagnostics, summarize_diagnostics


def held_out_signal(train, test):
    grouped = {}
    diagnostics(train, test, grouped, strength=4)
    points = np.array(list(grouped.values()))
    rng = np.random.default_rng(7)
    boot = rng.multinomial(len(points), np.ones(len(points)) / len(points), size=100)
    return summarize_diagnostics(points, boot)


class SignalTest(unittest.TestCase):
    def test_dataset_or_solver_proxy_is_not_incremental_signal(self):
        for proxy in ("dataset", "solver_model"):
            with self.subTest(proxy=proxy):
                candidates = []
                for question in range(24):
                    for group in range(2):
                        record = {"dataset": "source", "solver_model": "solver"}
                        record[proxy] += str(group)
                        for weak in range(2):
                            candidates.append(Candidate(**record, group=str(question),
                                                        weak=weak, strong=group,
                                                        gold=group ^ int(question % 17 == 0)))
                train = [row for row in candidates if int(row.group) < 16]
                test = [row for row in candidates if int(row.group) >= 16]
                result = held_out_signal(train, test)
                self.assertGreater(result["weak_only_relative_brier_gain"]["estimate"], 0.5)
                self.assertEqual(result["matched_relative_brier_gain"]["estimate"], 0)
                self.assertEqual(result["matched_relative_brier_gain"]["question_bootstrap_95"], [0, 0])

    def test_strong_signal_within_cheap_stratum_retains_credit(self):
        candidates = [
            Candidate("source", "solver", str(question), weak, strong, strong ^ int(question % 23 == 0))
            for question in range(72) for weak in range(2) for strong in range(2)
        ]
        train = [row for row in candidates if int(row.group) < 48]
        test = [row for row in candidates if int(row.group) >= 48]
        result = held_out_signal(train, test)
        self.assertGreater(result["matched_relative_brier_gain"]["question_bootstrap_95"][0], 0.5)


if __name__ == "__main__":
    unittest.main()
