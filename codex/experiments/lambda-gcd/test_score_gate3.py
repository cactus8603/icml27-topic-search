from fractions import Fraction
from itertools import combinations, product
import json
from math import gcd
from pathlib import Path
import tempfile
import unittest

import score_gate3 as score


def points(panel):
    return [[(Fraction(x), Fraction(x)) for x in row] for row in panel]


class IntervalIccTest(unittest.TestCase):
    def test_exact_known_cases_and_undefined(self):
        equal_within = score.interval_icc(points([[i] * 3 for i in range(5)]))
        self.assertEqual(equal_within["icc_1_1_exact"], 1)
        variable_within = score.interval_icc(points([[0, 1, 0] for _ in range(5)]))
        self.assertEqual(variable_within["icc_1_1_exact"], -0.5)
        constant = score.interval_icc(points([[1] * 3 for _ in range(5)]))
        self.assertIsNone(constant["icc_1_1_exact"])
        self.assertEqual(constant["diagnostic"], "COMPLETE_DIAGNOSTIC_UNDEFINED")

    def test_exact_interval_variance_minimum(self):
        self.assertEqual(score.min_mean_ss([(Fraction(0), Fraction(1, 4)), (Fraction(3, 4), Fraction(1))]), Fraction(1, 8))
        self.assertEqual(score.min_mean_ss([(Fraction(0), Fraction(3, 4)), (Fraction(1, 4), Fraction(1))]), 0)

    def test_lower_bound_is_valid_for_all_fixture_assignments(self):
        panel = [[(Fraction(i, 5), Fraction(i, 5))] * 2 + [(Fraction(i, 5), Fraction(i, 5) + Fraction(1, 20))]
                 for i in range(5)]
        result = score.interval_icc(panel)
        self.assertIsNone(result["icc_1_1_exact"])
        self.assertGreaterEqual(result["icc_1_1_lower"], 0.8)
        for values in product(*(row[2] for row in panel)):
            complete = points([[row[0][0], row[1][0], value] for row, value in zip(panel, values, strict=True)])
            exact = score.interval_icc(complete)["icc_1_1_exact"]
            self.assertGreaterEqual(exact + 1e-12, result["icc_1_1_lower"])
        all_unknown = score.interval_icc([[(Fraction(0), Fraction(1))] * 3 for _ in range(5)])
        self.assertEqual(all_unknown["icc_1_1_lower"], -0.5)
        self.assertTrue(all_unknown["constant_panel_possible"])
        self.assertEqual(all_unknown["diagnostic"], "BOUNDED_DIAGNOSTIC_UNRESOLVED")


class PanelTest(unittest.TestCase):
    def setUp(self):
        pairs = (pair for pair in combinations(range(4, 300), 2) if gcd(*pair) == 2)
        self.cohort = {(family, next(pairs)): ("a" * 64, 1, 2)
                       for family in score.FAMILIES for _ in range(score.FAMILY_N)}

    def sample_rows(self, arm, seed, affirmations, unknown=0):
        for index, ((family, pair), (hash_value, _, _)) in enumerate(self.cohort.items()):
            rank = index % score.FAMILY_N
            response = "bad format" if rank < unknown else (
                "<verdict>ACCEPT</verdict>\n<gcd>1</gcd>" if rank < unknown + affirmations else
                "<verdict>REJECT</verdict>\n<gcd>2</gcd>")
            yield {"arm": arm, "seed": seed, "family": family, "pair": pair,
                   "prompt_sha256": hash_value, "response": response}

    @staticmethod
    def encode(rows):
        return "".join(json.dumps(row) + "\n" for row in rows).encode()

    def test_unknown_remains_in_all_family_denominators(self):
        raw = self.encode(self.sample_rows("none", 0, 5, unknown=2))
        bounds, families = score.score_run(raw, "none", 0, self.cohort)
        self.assertEqual(bounds, (Fraction(5, 50), Fraction(7, 50)))
        for family in families.values():
            self.assertEqual((family["scheduled"], family["parsed"], family["format_error"], family["known_correct_gcd"]),
                             (50, 48, 2, 43))

    def test_malformed_panel_fails_closed(self):
        rows = list(self.sample_rows("none", 0, 0))
        with self.assertRaisesRegex(ValueError, "missing scheduled"):
            score.score_run(self.encode(rows[:-1]), "none", 0, self.cohort)
        with self.assertRaisesRegex(ValueError, "repeated"):
            score.score_run(self.encode(rows + rows[:1]), "none", 0, self.cohort)
        rows[0]["prompt_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "prompt"):
            score.score_run(self.encode(rows), "none", 0, self.cohort)

    def test_complete_15_run_fixture(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as path:
            root = Path(path)
            for index, arm in enumerate(score.ARMS):
                for seed in score.SEEDS:
                    (root / score.run_filename(arm, seed)).write_bytes(self.encode(self.sample_rows(arm, seed, index * 10)))
            result = score.score_panel(root, self.cohort)
            self.assertEqual(result["stage"], "gcd_reliability_diagnostic_only")
            self.assertEqual(result["scheduled_responses"], 4500)
            self.assertEqual(result["icc_1_1_exact"], 1)
            self.assertEqual(len({row["response_file_sha256"] for row in result["runs"]}), 15)
            (root / score.run_filename("none", 2)).unlink()
            with self.assertRaisesRegex(ValueError, "15 planned"):
                score.score_panel(root, self.cohort)


if __name__ == "__main__":
    unittest.main()
