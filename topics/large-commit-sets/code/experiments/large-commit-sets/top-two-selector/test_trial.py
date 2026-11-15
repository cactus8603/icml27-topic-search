"""CPU checks for the fixed-pool selector and its forward budget."""

import json
import unittest

from trial import CONFIG, Decoder, follow_up, keys, paired_summary, pilot, score, selectors


def proposals():
    return [
        pilot.Proposal(i, 100 + i, *values)
        for i, values in enumerate([(-0.02, 5)] * 7 + [
            (-0.79, 0.01), (-0.80, 8), (-0.82, 6), (-0.99, 0.5), (-1.10, 0.5),
        ])
    ]


class FakeDecoder(Decoder):
    def __init__(self, eligible):
        self.settings = json.loads(pilot.CONFIG_PATH.read_text())["decoder"]
        self.step_count = 32
        self.forward_calls = 0
        self.excluded_tokens = set() if eligible else {100}

    def encode(self, question):
        return (), 0

    def proposals(self, x, prompt_length, step):
        self.forward_calls += 1
        return proposals()

    def greedy_step(self, x, prompt_length, step, ranked):
        return x

    def completion(self, x, prompt_length):
        return "#### 0", 1

    def branch(self, x, prompt_length, step, members, ranked):
        calls = self.step_count - step - 1
        self.forward_calls += calls
        return x, "#### 8" if 8 in members else "#### 0", 1, calls, 0.0


class TrialTest(unittest.TestCase):
    def test_duplicate_keys(self):
        self.assertEqual(keys("A buys １２ apples."), keys(" A BUYS 12 APPLES. "))
        self.assertEqual(keys("A buys 12 apples.")[1], keys("A buys 50 apples!")[1])

    def test_top_two_breaks_near_tie_without_probe(self):
        ranked = proposals()
        pools = [tuple(range(8)), (*range(7), 8), (*range(7), 9)]
        choices = selectors(ranked, pools, "example", "frozen")
        self.assertEqual(choices["confidence"], 0)
        self.assertEqual(choices["top_two"], 1)
        self.assertEqual(choices["margin_only"], 1)
        self.assertEqual(choices, selectors(ranked, pools, "example", "frozen"))

    def test_margin_ablation_can_choose_a_different_pool(self):
        ranked = proposals()
        ranked[7] = pilot.Proposal(7, 107, -0.79, 0.10)
        ranked[8] = pilot.Proposal(8, 108, -0.90, 0.22)
        ranked[9] = pilot.Proposal(9, 109, -0.95, 0.12)
        pools = [tuple(range(8)), (*range(7), 8), (*range(7), 9)]
        choices = selectors(ranked, pools, "example", "frozen")
        self.assertEqual(choices["margin_only"], 1)
        self.assertEqual(choices["top_two"], 0)

    def test_same_pool_and_fallback_budgets(self):
        chooser = FakeDecoder(eligible=True)
        selected, outputs, anchor, calls = chooser.evaluate_request("q", "id", "seed")
        self.assertEqual((len(outputs), anchor, calls), (3, 2, 90))
        scored = score(selected, outputs, "8")
        self.assertEqual((scored["confidence"], scored["top_two"], scored["pool_oracle"]), (0, 1, 1))
        fallback = FakeDecoder(eligible=False)
        selected, outputs, anchor, calls = fallback.evaluate_request("q", "id", "seed")
        self.assertEqual((len(outputs), anchor, calls), (1, None, 32))
        self.assertEqual(set(score(selected, outputs, "0").values()), {1})

    def test_aggregate_uses_common_denominator(self):
        data = [
            (source, {"confidence": 0, "margin_only": 1, "uniform_fixed": 0, "top_two": 1, "pool_oracle": 1})
            for source in ("gsm", "math")
        ]
        summary = paired_summary(data, json.loads(CONFIG.read_text()))["fixed_mixture"]
        self.assertEqual(summary["n"], 2)
        self.assertEqual(summary["top_two_rescues"], 2)
        self.assertEqual(summary["paired_ci95"], [1.0, 1.0])
        self.assertEqual(follow_up(summary, 0.04), "independent_confirmation")
        self.assertEqual(follow_up({"paired_ci95": [-0.01, 0.03]}, 0.04), "do_not_expand")
        self.assertEqual(follow_up({"paired_ci95": [-0.01, 0.08]}, 0.04), "inconclusive")


if __name__ == "__main__":
    unittest.main()
