import unittest
import numpy as np
from codex.retrieval_matrix_panel.data import caption, threshold_ap


class ColaScoring(unittest.TestCase):
    def test_tie_threshold_is_invariant_to_image_order(self):
        self.assertAlmostEqual(threshold_ap([1,1,0],[1,0,1]),7/12)
        self.assertAlmostEqual(threshold_ap([1,1,0],[0,1,1]),7/12)

    def test_hard_filter_precedes_ap(self):
        scores=np.array([.9,.8,.7]); rel=np.array([0,1,0]); hard=np.array([False,True,True])
        self.assertEqual(threshold_ap(scores[hard],rel[hard]),1)
        self.assertEqual(threshold_ap(scores,rel),.5)

    def test_label_rendering_never_includes_paco_count(self):
        self.assertEqual(caption([["green","large"],"leaves"],"gqa"),"green large leaves")
        self.assertEqual(caption(["dark brown_grey_metal__knife",5],"paco"),"dark brown grey metal knife")

    def test_nonfinite_scores_fail(self):
        with self.assertRaises(ValueError): threshold_ap([float("nan")],[1])


if __name__ == "__main__": unittest.main()
