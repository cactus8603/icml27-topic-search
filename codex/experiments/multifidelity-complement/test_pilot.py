from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pilot import PERM_SEED, coordinates, cosine, summarize


class ComplementTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(11)
        self.jc = rng.normal(size=(6, 8))
        self.jt = rng.normal(size=(4, 8))
        self.gram = self.jc @ self.jc.T
        self.cross = self.jc @ self.jt.T
        self.mapping = rng.normal(size=(256, 8))

    def test_on_calibration_extension_matches_original_rho_operator(self):
        u, d = coordinates(self.gram, self.gram, 2)
        np.testing.assert_allclose(d, u.T, atol=1e-14)
        yc = self.mapping @ self.jc.T
        np.testing.assert_allclose(yc - (yc @ u) @ d, yc @ (np.eye(6) - u @ u.T), atol=1e-12)

    def test_on_heldout_targets_parameter_projection_is_orthogonal(self):
        u, d = coordinates(self.gram, self.cross, 2)
        eigenvalues = np.linalg.eigvalsh(self.gram)[-2:][::-1]
        q = self.jc.T @ u / eigenvalues**0.5
        np.testing.assert_allclose(q.T @ q, np.eye(2), atol=1e-14)
        want = q @ q.T @ self.jt.T
        np.testing.assert_allclose(self.jc.T @ u @ d, want, atol=1e-14)
        np.testing.assert_allclose(q.T @ (self.jt.T - want), 0, atol=1e-14)

    def test_matching_maps_and_synchronized_row_permutation(self):
        yc, yt = self.mapping @ self.jc.T, self.mapping @ self.jt.T
        u, d = coordinates(self.gram, self.cross, 2)
        p = np.random.default_rng(PERM_SEED).permutation(len(yt))
        left = yt - yc @ u @ d
        np.testing.assert_allclose(yt[p] - yc[p] @ u @ d, left[p], atol=1e-14)
        result = summarize(yc, yc, yt, yt, self.gram, self.cross, budget=2)
        self.assertAlmostEqual(result["mean_heldout_complement_cosine"], 1)
        self.assertAlmostEqual(result["row_permutation_mean_cosine"], cosine(left, left[p]).mean())
        self.assertEqual(result["covered_queries"], 4)

    def test_degenerate_cutoff_and_zero_required_residual_are_rejected(self):
        with self.assertRaises(ValueError):
            coordinates(np.eye(3), np.ones((3, 2)), 2)
        with self.assertRaises(ValueError):
            cosine(np.zeros((4, 3)), np.ones((4, 3)))


if __name__ == "__main__":
    unittest.main()
