from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dense


class DenseTests(unittest.TestCase):
    def test_fixed_roster_and_probe_cadence(self):
        base, _, seeds = dense.plan()
        self.assertEqual(len(set(seeds)), 96)
        self.assertEqual([seed for index in range(8) for seed in dense.assigned(seeds, index)], seeds)
        self.assertEqual(dense.probe_steps(base["training"]), list(range(22, 401, 2)))

    def test_accounting_requires_all_successful_unique_tasks(self):
        rows = [f"495_{index}|{500 + index}|COMPLETED|0:0" for index in range(8)]
        self.assertEqual(dense.completed_shards("495", "\n".join(rows)), dict(enumerate(map(str, range(500, 508)))))
        for corrupt in (rows[:-1], rows + [rows[0]], [*rows[:-1], "495_7|507|FAILED|1:0"]):
            with self.assertRaises(ValueError):
                dense.completed_shards("495", "\n".join(corrupt))

    def test_rate_is_per_probe_with_seed_clustered_uncertainty(self):
        seeds = [11, 21]
        groups = {(arm, seed): [{"delta_nll": (0.08 if seed == 11 else 0.01)}] * 4
                  for seed in seeds for arm in dense.STRONG}
        rate = dense.rate_summary(groups, seeds, 0.05)
        self.assertEqual(rate["rate"], 0.5)
        self.assertEqual(rate["seed_bootstrap_95"], [0.0, 1.0])
        no_events = dense.rate_summary(groups, seeds, 0.1)
        self.assertEqual(no_events["rate"], 0)
        self.assertIsNone(no_events["seed_bootstrap_95"])


if __name__ == "__main__":
    unittest.main()
