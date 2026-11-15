from fractions import Fraction
from itertools import product
from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parent))
import diagnostic


class DiagnosticTest(unittest.TestCase):
    def test_only_response_tokens_are_supervised(self):
        def character_tokenizer(texts, add_special_tokens):
            self.assertFalse(add_special_tokens)
            return {"input_ids": [list(map(ord, text)) for text in texts]}

        self.assertEqual(diagnostic.encode(character_tokenizer, ["ab"], ["c"]), [([97, 98, 99], [-100, -100, 99])])
        with self.assertRaisesRegex(ValueError, "assistant boundary"):
            diagnostic.encode(character_tokenizer, [""], ["x" * 513])

    def test_concordance_respects_strict_separation_and_excludes_none(self):
        predictions = {"none": -100} | dict(zip(diagnostic.DEFENDED, range(4), strict=True))
        bounds = {"none": (100, 100)} | dict(zip(diagnostic.DEFENDED, [(0, 0), (1, 1), (2, 2), (3, 3)], strict=True))
        result = diagnostic.compare(predictions, bounds)
        self.assertEqual(result["concordant"], 6)
        self.assertEqual(result["signed_pair_margin"], [6, 6])
        self.assertEqual(result["spearman_when_fully_observed"], 1)
        bounds[diagnostic.DEFENDED[-1]] = (2, 3)
        result = diagnostic.compare(predictions, bounds)
        self.assertEqual(result["signed_pair_margin"], [4, 6])
        self.assertIsNone(result["spearman_when_fully_observed"])

    def test_reverse_direction_ties_and_overlap(self):
        predictions = dict(zip(diagnostic.DEFENDED, (0, 0, 2, 3), strict=True))
        bounds = dict(zip(diagnostic.DEFENDED, ((3, 3), (3, 3), (1, 2), (0, 1)), strict=True))
        result = diagnostic.compare(predictions, bounds)
        self.assertEqual(result["discordant"], 4)
        self.assertEqual(result["lambda_tie"], 1)
        self.assertEqual(result["unresolved"], 1)
        self.assertEqual(result["signed_pair_margin"], [-5, -3])

    def test_gradients_require_all_registered_cells(self):
        rows = [{"arm": arm, "seed": seed, "context": context, "lambda": seed * 3 + offset}
                for (arm, seed), (offset, context) in product(diagnostic.STAGED.CELLS, enumerate(diagnostic.CONTEXTS))]
        pooled, by_seed = diagnostic.prediction_ranks(rows)
        self.assertEqual(pooled["none"], 4)
        self.assertEqual(by_seed[2]["none"], 7)
        with self.assertRaisesRegex(ValueError, "missing"):
            diagnostic.prediction_ranks(rows[:-1])
        with self.assertRaisesRegex(ValueError, "repeated"):
            diagnostic.prediction_ranks(rows + [rows[0]])

    def test_behavior_pairing_checks_source_and_all_family_counts(self):
        array, freeze = "1234", "a" * 40
        rows = [{"arm": arm, "seed": seed, "families": {
            family: {"scheduled": 50, "known_affirmations": seed + 1, "parsed": 40}
            for family in diagnostic.FAMILIES
        }} for arm, seed in diagnostic.STAGED.CELLS]
        panel = {"stage": diagnostic.pilot.STAGE, "array": array, "freeze": freeze, "scheduled_responses": 4500,
                 "input_manifest_sha256": diagnostic.STAGED.INPUT_SHA, "runs": rows}
        left, right = diagnostic.pilot_bounds(panel, array, freeze)["none"]
        self.assertEqual(left, Fraction(2, 50))
        self.assertEqual(right, Fraction(12, 50))
        with self.assertRaisesRegex(ValueError, "wrong frozen"):
            diagnostic.pilot_bounds(panel, "1235", freeze)
        with self.assertRaisesRegex(ValueError, "missing"):
            diagnostic.pilot_bounds(panel | {"runs": rows[:-1]}, array, freeze)


if __name__ == "__main__":
    unittest.main()
