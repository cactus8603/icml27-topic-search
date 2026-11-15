from pathlib import Path
import sys
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "killtests/rl-sparse-distill/scripts"))

import analyze
from aggregate import mix_curve, pass_at_k_vec


class ExternalReplayTest(unittest.TestCase):
    def test_aligned_counts_are_order_independent(self):
        rows = (b'{"id":"b","source":"math","c":1,"n":4}\n'
                b'{"id":3,"source":"math","c":2,"n":4}\n')
        base = analyze.read_counts(rows, 4)
        rl = analyze.read_counts(b"\n".join(reversed(rows.splitlines())), 4)
        got_base, got_rl = analyze.paired_counts(base, rl, {"math": 2})
        np.testing.assert_array_equal(got_base, [2, 1])
        np.testing.assert_array_equal(got_rl, [2, 1])
        with self.assertRaisesRegex(ValueError, "coverage"):
            analyze.paired_counts(base, {3: rl[3]}, {"math": 2})

    def test_invalid_rows_are_rejected(self):
        valid = b'{"id":3,"source":"math","c":2,"n":4}\n'
        for data in (valid * 2, valid.replace(b':2,', b':true,'), valid.replace(b':4}', b':5}')):
            with self.subTest(data=data), self.assertRaises(ValueError):
                analyze.read_counts(data, 4)

    def test_finite_sample_pass_rates(self):
        lookup = analyze.pass_lookup(4, pass_at_k_vec)
        np.testing.assert_allclose(lookup[1], [.25, .5, .75, 1])
        np.testing.assert_allclose(lookup[2], [.5, 5 / 6, 1, 1])
        np.testing.assert_array_equal(lookup[0], 0)

    def test_mix_and_hindsight_have_different_pass1_requirements(self):
        base = np.array([4, 0])
        rl = np.array([0, 4])
        lookup = analyze.pass_lookup(4, pass_at_k_vec)
        mix = analyze.response_mix(base, rl, lookup, 4, 2, .5, mix_curve)
        self.assertEqual([p["pass_at_k_pct"] for p in mix["points"]], [50, 75, 50])
        self.assertEqual(mix["max_pass_at_k_grid_point"]["delta_vs_matched_endpoint_pp"], 25)
        oracle = analyze.hindsight_endpoints(base, rl, lookup, 2)
        self.assertEqual(oracle["hindsight_minus_best_single_endpoint_pp"], 50)
        self.assertEqual(oracle["hindsight_policy_empirical_pass_at_1_pct"], 100)
        self.assertFalse(oracle["matched_pass_at_1"])

    def test_simultaneous_tails_and_degenerate_columns(self):
        lower, upper, _ = analyze.simultaneous_band(
            np.array([[.4, .1, -.4], [.5, .1, -.5], [.6, .1, -.6]]), bootstrap=100,
        )
        self.assertEqual((lower[1], upper[1]), (-1, 1))
        self.assertEqual(analyze.certify(lower, upper), {
            "first_loss_interval": [2, 3],
            "certified_positive_budgets": [1],
            "certified_negative_budgets": [3],
        })

    def test_external_reference_never_qualifies_formal_pair(self):
        criteria = {"gate0": {"pair_rl_p1_gain_min_pp": 5, "pair_rl_pk_below_base_min_pp": 2,
                              "pass_window_pct": [30, 85], "trunc_rate_diff_max_pp": 2}}
        endpoints = {"1": {"rl_minus_base_pp": 5},
                     "64": {"base_pct": 85, "rl_pct": 83, "rl_minus_base_pp": -2}}
        got = analyze.gate_reference(endpoints, criteria, 64)
        self.assertTrue(got["rl_pass_at_1_gain_meets_reference"])
        self.assertTrue(got["rl_pass_at_k_loss_meets_reference"])
        self.assertTrue(got["base_pass_at_k_in_reference_window"])
        self.assertTrue(got["rl_pass_at_k_in_reference_window"])
        self.assertFalse(got["published_truncation"]["meets_reference"])
        self.assertFalse(got["official_gate0_evaluable"])


if __name__ == "__main__":
    unittest.main()
