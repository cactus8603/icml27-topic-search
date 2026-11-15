from io import BytesIO
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import average_precision, primary_order, rerank, run, summarize
from prepare import annotations, inspect_image


class FixedShortlistTests(unittest.TestCase):
    def test_rerank_uses_only_prefix_and_keeps_baseline_ties(self):
        baseline = primary_order(np.zeros((2, 53)))
        scores = np.zeros((2, 50))
        scores[0, 47] = 1
        scores[0, 48] = 1
        scores[1, 49] = 2
        output = rerank(baseline, scores)
        np.testing.assert_array_equal(output[0, :4], [47, 48, 0, 1])
        np.testing.assert_array_equal(output[1, :3], [49, 0, 1])
        np.testing.assert_array_equal(baseline[:, 50:], output[:, 50:])
        np.testing.assert_array_equal(np.sort(output, axis=1), baseline)

    def test_labels_below_prefix_contribute_to_full_pool_ap(self):
        labels = np.zeros((1, 60), dtype=bool)
        labels[0, [49, 59]] = True
        score = np.zeros((1, 50))
        score[0, 49] = 1
        baseline = primary_order(np.zeros((1, 60)))
        result = summarize(baseline, rerank(baseline, score), labels, [101])
        old = 100 * (1 / 50 + 2 / 60) / 2
        new = 100 * (1 + 2 / 60) / 2
        self.assertAlmostEqual(result["baseline_map_pp"], old)
        self.assertAlmostEqual(result["reranked_map_pp"], new)
        self.assertAlmostEqual(result["fixed_top50_oracle_map_pp"], new)
        self.assertEqual(result["recall_at_50_pp"], 50)
        self.assertEqual(result["state"], "PASS")

    def test_no_change_with_multiple_queries_from_one_group(self):
        labels = np.zeros((3, 60), dtype=bool)
        labels[:, 0] = True
        baseline = primary_order(np.zeros((3, 60)))
        result = summarize(baseline, baseline, labels, [201, 201, 303])
        self.assertEqual(result["clusters"], 2)
        self.assertEqual(result["delta_95ci_pp"], [0, 0])
        self.assertEqual(result["state"], "FAIL")

    def test_reject_missing_scores_and_positive_labels(self):
        baseline = primary_order(np.zeros((1, 60)))
        for cross in (np.zeros((1, 49)), np.full((1, 50), np.nan)):
            with self.assertRaises(ValueError):
                rerank(baseline, cross)
        with self.assertRaises(ValueError):
            average_precision(np.zeros((1, 60), dtype=bool))

    def test_conjunctions_exclude_known_query_and_need_both_single_controls(self):
        categories = {10: {1, 2, 3}, 11: {1}, 12: {2}, 13: {3}, 14: {1, 2, 4}}
        actual = annotations(sorted(categories), categories, {(1, 3)})
        self.assertEqual([(q["key"], q["relevant"]) for q in actual],
                         [([1, 2], [10, 14]), ([2, 3], [10])])
        for q in actual:
            self.assertIn(q["group"], q["relevant"])

    def test_candidate_scoring_only_reads_original_shortlists(self):
        panel = {"ids": list(range(1, 54)), "names": {"1": "cat", "2": "dog", "3": "cup"},
                 "queries": [{"key": [1, 2], "relevant": [53], "group": 53},
                             {"key": [2, 3], "relevant": [1], "group": 1}]}
        config = {"batch_size": {"cross_pair": 4}}
        scorer = Mock()
        scorer.score.side_effect = lambda pairs: [0] * len(pairs)
        order = np.arange(1, 54).reshape(-1, 1)
        with (patch.dict(sys.modules, {"torch": SimpleNamespace(cuda=Mock())}),
              patch("evaluate.Encoder"), patch("evaluate.encode_items", return_value=(order, np.array([[1], [-1]]))),
              patch("evaluate.CrossScorer", return_value=scorer),
              patch("evaluate.rgb", side_effect=lambda path: int(path.stem))):
            result = run(panel, config, {"core_embed": None, "core_reranker": None})
        actual = [item for call in scorer.score.call_args_list for item in call.args[0]]
        expected = [("a photo of cat and dog", image) for image in range(53, 3, -1)]
        expected += [("a photo of dog and cup", image) for image in range(1, 51)]
        self.assertEqual(actual, expected)
        self.assertEqual(result["delta_map_pp"], 0)

    def test_dimension_exclusion_distinguishes_valid_jpeg_from_wrong_format(self):
        image = Image.new("RGB", (25, 31))
        jpeg = BytesIO()
        image.save(jpeg, format="JPEG")
        matched_sha, matched = inspect_image(jpeg.getvalue(), [25, 31])
        wrong_shape_sha, wrong_shape = inspect_image(jpeg.getvalue(), [31, 25])
        self.assertIsNone(matched)
        self.assertEqual(wrong_shape, {"actual": [25, 31], "expected": [31, 25], "format": "JPEG"})
        self.assertEqual(matched_sha, wrong_shape_sha)
        png = BytesIO()
        image.save(png, format="PNG")
        _, wrong_format = inspect_image(png.getvalue(), [25, 31])
        self.assertEqual(wrong_format, {"actual": [25, 31], "expected": [25, 31], "format": "PNG"})
        with self.assertRaises(OSError):
            inspect_image(b"<html>not an image</html>", [25, 31])


if __name__ == "__main__":
    unittest.main()
