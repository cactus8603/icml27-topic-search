import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from observable import Observation, fit_states, observe, score_states, verify_published_phenomenon


class ObservableTests(unittest.TestCase):
    def test_natural_step_features_use_only_positions_and_confidence(self):
        anchor = {
            "step": 10,
            "candidates": [
                {"positions": [0, 4], "confidence": [-0.01, -0.02]},
                {"positions": [0, 5], "confidence": [-0.01, -0.025]},
                {"positions": [0, 9], "confidence": [-0.01, -0.04]},
            ],
        }
        settings = {"max_swap_logprob_gap": 0.25, "max_swap_distance": 24}
        features = observe(anchor, settings)
        self.assertEqual(features.step, 10)
        self.assertAlmostEqual(features.first_swap_confidence_loss, 0.005)
        self.assertEqual(features.shortest_swap_span, 1)
        self.assertIsNone(observe(None, settings))
        anchor["candidates"][1]["confidence"][0] = -0.011
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            observe(anchor, settings)

    def test_cutoffs_fitted_on_train_and_ties_use_original(self):
        low = Observation(2, 0.01, 2)
        high = Observation(2, 0.05, 6)
        rare = Observation(10, 0.05, 6)
        rows = [low, low, low, high, high, rare, None]
        outcomes = [(0, 1, 0)] * 3 + [(1, 1, 0)] * 2 + [(0, 1, 0), (1,)]
        cutoffs, fitted, counts = fit_states(rows, outcomes, minimum_size=2)
        self.assertAlmostEqual(cutoffs[0], 0.03)
        self.assertEqual(cutoffs[1], 4)
        self.assertEqual(Observation(10, *cutoffs).state(*cutoffs), (10, False, False))
        self.assertEqual(fitted, {low.state(*cutoffs): 1, high.state(*cutoffs): 0})
        self.assertEqual(counts[rare.state(*cutoffs)], 1)

    def test_state_oracle_uses_same_train_support_and_one_choice_per_state(self):
        frequent = Observation(2, 0.01, 2)
        rare_or_unseen = Observation(10, 0.05, 6)
        cutoffs = (0.02, 4)
        table = {frequent.state(*cutoffs): 1}
        arms, supported, changed = score_states(
            [frequent, frequent, frequent, rare_or_unseen, None],
            [(0, 0, 1), (0, 0, 1), (1, 0, 0), (0, 1, 0), (1,)],
            cutoffs, table,
        )
        self.assertEqual((supported, changed), (3, 3))
        self.assertEqual(arms["confidence"], [0, 0, 1, 0, 1])
        self.assertEqual(arms["train_fitted"], [0, 0, 0, 0, 1])
        self.assertEqual(arms["state_oracle"], [1, 1, 0, 0, 1])
        self.assertEqual(arms["supported_pool_oracle"], [1, 1, 1, 0, 1])
        self.assertEqual(arms["three_candidate_oracle"], [1, 1, 1, 1, 1])

    def test_phenomenon_replay_checks_the_published_counts(self):
        candidate = {"additive_margins": [0.1], "joint_margins": [2.3]}
        rows = [{"anchor": {"candidates": [candidate]}}, {"anchor": None}]
        verdict = {"eligible_prompts": 1, "phenomenon": {"events": 1, "prompts": 1}}
        self.assertEqual(verify_published_phenomenon(rows, verdict, 2.2), verdict["phenomenon"])
        with self.assertRaisesRegex(ValueError, "Phenomenon"):
            verify_published_phenomenon(rows, verdict, 3)


if __name__ == "__main__":
    unittest.main()
