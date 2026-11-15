from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import CONTROLS, PRIMARY, arms, full_order, run, summarize
from prepare import assemble, verify_anchors, with_anchors


class AnchorRerankTests(unittest.TestCase):
    def test_zero_weight_and_constant_anchors_are_noops_against_full_control(self):
        cross = np.array([[10.0, 4.0, 4.0, 0.0]])
        full = np.array([[.18, .23, .4, .9]])
        real = full.copy() * 2
        shuffled = full[::-1] * 5
        zero = arms(cross, full, real, shuffled, top_k=4, weight=0)
        for name in (CONTROLS[1], CONTROLS[2]):
            np.testing.assert_array_equal(zero[PRIMARY], zero[name])
        present = arms(cross, full, np.ones_like(full), np.zeros_like(full), top_k=4)
        np.testing.assert_array_equal(present[PRIMARY], present[CONTROLS[1]])
        np.testing.assert_array_equal(present[PRIMARY], present[CONTROLS[2]])

    def test_true_anchor_can_increment_given_same_full_score_and_tail(self):
        base = np.array([[3, 1, 2, 0], [1, 0, 2, 3]])
        # Within a tied CORE shortlist, the full score prefers an annotated negative.
        cross = np.ones((2, 3))
        full = np.array([[.9, .8, .7], [.9, .8, .7]])
        real = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        shuffled = np.zeros_like(real)
        scores = arms(cross, full, real, shuffled, top_k=3)
        orders = {name: full_order(base, value, top_k=3) for name, value in scores.items()}
        for order in orders.values():
            np.testing.assert_array_equal(order[:, 3:], base[:, 3:])
            np.testing.assert_array_equal(np.sort(order[:, :3], axis=1), np.sort(base[:, :3], axis=1))
        relevance = np.array([[1, 1, 0, 0], [1, 0, 0, 1]], dtype=bool)
        answer = summarize(orders, relevance, [100, 200], top_k=3)
        self.assertEqual(answer["state"], "PASS")
        self.assertEqual(answer["recall_at_50_pp"], 50)
        self.assertAlmostEqual(answer["map_pp"][PRIMARY], 75)
        self.assertAlmostEqual(answer["map_pp"][CONTROLS[0]], 50)

    def test_anchor_control_is_complete_bijective_and_semantically_disjoint(self):
        keys = [[1, 2], [3, 4], [5, 6]]
        names = {i: str(i) for i in range(1, 7)}
        result = with_anchors([{"key": key} for key in keys], names)
        verify_anchors(result, names)
        result[0]["shuffled_donor_key"] = result[0]["key"]
        with self.assertRaises(ValueError):
            verify_anchors(result, names)

    def test_annotated_format_and_duplicate_exclusions_are_distinct(self):
        ids, shas, rejected = assemble(
            {1: [8, 8], 2: [8, 8], 3: [8, 8], 4: [8, 8]}, {"old"},
            {1: ("first", None), 2: ("old", None), 3: ("first", None),
             4: ("other", {"format": "PNG", "actual": [8, 8], "expected": [8, 8]})},
        )
        self.assertEqual((ids, shas), ([1], ["first"]))
        self.assertEqual([r["reason"] for r in rejected],
                         ["exact_duplicate", "exact_duplicate", "source_format_or_dimensions"])

    def test_pipeline_scores_only_original_top50_and_cs_encodes_shortlist_union(self):
        panel = {"ids": list(range(1, 54)), "queries": [
            {"text": "fullA", "anchor": "anchorA", "shuffled_anchor": "anchorB", "relevant": [53], "group": 53},
            {"text": "fullB", "anchor": "anchorB", "shuffled_anchor": "anchorA", "relevant": [50], "group": 50},
        ]}
        images = np.arange(1, 54).reshape(53, 1)
        cross = Mock(return_value=np.zeros(100))

        def clip_batch(model, inputs, batch, *, images=False):
            clip_inputs.append([opener() for opener in inputs] if images else inputs)
            return np.ones((len(inputs), 1))

        clip_inputs = []
        config = {"batch_size": {"embed_image": 8, "embed_text": 32, "cross_pair": 4}}
        snapshots = {name: None for name in ("cs_clip", "core_embed", "core_reranker")}
        with (patch.dict(sys.modules, {"torch": SimpleNamespace(cuda=Mock())}),
              patch("evaluate.Encoder"), patch("evaluate.CrossScorer"), patch("evaluate.ClipEncoder"),
              patch("evaluate.rgb", side_effect=lambda path: int(path.stem)),
              patch("evaluate.encode_items", return_value=(images, np.array([[1], [1]]))),
              patch("evaluate.cross_scores", side_effect=cross), patch("evaluate.unit_scores", side_effect=clip_batch)):
            metrics = run(panel, config, snapshots)
            cross_image_ids = [opener() for opener in cross.call_args.args[2]]
        self.assertEqual(cross.call_args.args[1], ["fullA"] * 50 + ["fullB"] * 50)
        self.assertEqual(cross_image_ids, list(range(53, 3, -1)) * 2)
        self.assertEqual(clip_inputs, [list(range(4, 54)), ["anchorA", "anchorB", "fullA", "fullB"]])
        self.assertEqual((metrics["classified_cross_pairs"], metrics["cs_unique_shortlist_images"]), (100, 50))


if __name__ == "__main__":
    unittest.main()
