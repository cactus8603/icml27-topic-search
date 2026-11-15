from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ranking


class RankingTests(unittest.TestCase):
    def test_complementary_known_positives(self):
        panel = {"key": [1, 2], "images": list(range(1, 9)), "anchor": "a", "shuffled_anchor": "s",
                 "queries": [{"direction": "left", "text": "l", "relevant": [1, 3, 5, 7]},
                             {"direction": "right", "text": "r", "relevant": [2, 4, 6, 8]}]}
        with patch.object(ranking, "GROUPS", 1):
            *_, relevance = ranking.layout([panel], {str(i): {} for i in panel["images"]})
            np.testing.assert_array_equal(relevance[0], ~relevance[1])
            panel["queries"][1]["relevant"] = [1, 4, 6, 8]
            with self.assertRaises(ValueError):
                ranking.layout([panel], {str(i): {} for i in panel["images"]})

    def test_perfect_ranking_has_no_headroom_and_no_extra_gain(self):
        relevance = np.array([[True] * 4 + [False] * 4, [False] * 4 + [True] * 4])
        full = relevance.astype(float) * 2
        anchor = np.ones((2, 8))
        result = ranking.analyze(full, full, anchor, anchor, relevance, np.array([0, 0]))
        self.assertEqual(result["same_image_cs_clip"]["strict_full_gt_anchor_gt_foil_pp"], 100)
        self.assertEqual(result["fixed_group_label_oracle_map_pp"], 100)
        self.assertEqual(set(result["gap_to_oracle_pp"].values()), {0})
        self.assertEqual(result["state"], "FAIL")


if __name__ == "__main__":
    unittest.main()
