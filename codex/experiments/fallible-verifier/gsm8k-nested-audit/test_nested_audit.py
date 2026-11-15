import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nested_audit


class TestNestedAudit(unittest.TestCase):
    def setUp(self):
        self.config = {
            "strong_price_units": 1,
            "gold_price_units": 10,
            "budget_fraction_at_gold_cost": 0.135,
            "probe_fraction": 0.35,
            "smoothing": 4,
            "train_risk_target": 0.075,
        }
        self.rows = [
            {
                "id": str(i),
                "dataset": "gsm8k",
                "solver_model": f"variant{i // 9 % 2}",
                "weak": (0, 1, None)[i % 3],
                "strong": (0, 1, None)[i // 3 % 3],
                "gold": (None, True, False, True, False)[i % 5],
            }
            for i in range(200)
        ]

    def test_same_strong_roster_and_only_nested_gold_at_equal_cost(self):
        probes, audits = nested_audit.nested_observations(self.rows, self.config, 11)
        old_probes, old_baseline, _ = nested_audit.reference.warm_observations(self.rows, self.config, 11)
        self.assertEqual(probes, old_probes)
        self.assertLessEqual(audits, probes.keys())
        self.assertEqual(len(probes) + 10 * len(audits), 10 * len(old_baseline))
        selected_states = {(nested_audit.reference.cheap_state(self.rows[i]), probes[i]) for i in audits}
        observed_states = {(nested_audit.reference.cheap_state(self.rows[i]), s) for i, s in probes.items()}
        self.assertEqual(selected_states, observed_states)
        self.assertEqual((probes, audits), nested_audit.nested_observations(self.rows, self.config, 11))

    def test_unpurchased_label_and_verdict_mutations_leave_policy_unchanged(self):
        probes, audits = nested_audit.nested_observations(self.rows, self.config, 11)
        changed = [dict(row) for row in self.rows]
        for i, row in enumerate(changed):
            if i not in audits:
                row["gold"] = not row["gold"]
            if i not in probes:
                row["strong"] = 1 - (row["strong"] or 0)
        self.assertEqual((probes, audits), nested_audit.nested_observations(changed, self.config, 11))
        before, before_unknown = nested_audit.fit_draw(self.rows, self.config, 11)
        after, after_unknown = nested_audit.fit_draw(changed, self.config, 11)
        self.assertEqual(before_unknown, after_unknown)
        self.assertEqual(before.keys(), after.keys())
        for state, (trees, weights) in before.items():
            self.assertEqual(trees, after[state][0])
            np.testing.assert_array_equal(weights, after[state][1])

    def test_unresolved_observations_keep_original_fallback(self):
        policy, unresolved = nested_audit.fit_draw(self.rows, self.config, 11)
        self.assertTrue(policy)
        self.assertGreater(unresolved, 0)
        self.assertTrue(all(weak is not None for _, _, weak in policy))
        actions = nested_audit.policy_actions(self.rows, (policy,))
        for row in self.rows:
            if row["weak"] is None:
                np.testing.assert_array_equal(actions[row["id"]], [0, 0, 1, 0, 0, 0])
            if row["strong"] is None:
                np.testing.assert_array_equal(actions[row["id"]][3:5], [0, 0])


if __name__ == "__main__":
    unittest.main()
