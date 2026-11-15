"""Synthetic CPU-only checks; never read research datasets or result caches."""

import copy
import math
import unittest

import numpy as np

try:
    from .logic_baselines import (aggregate_scores, lcse_and, fit_presence_calibration,
                                 validate_calibration_artifact, CONCEPT_TEMPLATES)
except ImportError:
    from logic_baselines import (aggregate_scores, lcse_and, fit_presence_calibration,
                                validate_calibration_artifact, CONCEPT_TEMPLATES)


class LogicBaselineTests(unittest.TestCase):
    def test_lcse_matches_independent_scalar_and_formula(self):
        mu, beta = 0.22, 30.0
        hol = np.array([[0.25, 0.27], [0.3, 0.2]])
        cond = np.array([[[0.3, 0.12], [0.27, 0.28]], [[0.13, 0.24], [0.31, 0.19]]])
        expected = np.zeros_like(hol)
        for idx in np.ndindex(hol.shape):
            pa, pb = [1 / (1 + math.exp(-beta * (x - mu))) for x in cond[idx]]
            hm, am = 2 * pa * pb / (pa + pb), (pa + pb) / 2
            logit = lambda x: math.log(x / (1 - x))
            expected[idx] = hol[idx] + (logit(hm) - logit(am)) / beta
        np.testing.assert_allclose(lcse_and(hol, cond), expected, atol=1e-12)

    def test_equal_conditions_have_no_correction_and_weak_conjunct_penalizes(self):
        hol = np.array([0.25, 0.25])
        result = lcse_and(hol, np.array([[0.25, 0.25], [0.35, 0.10]]))
        self.assertAlmostEqual(result[0], hol[0])
        self.assertLess(result[1], hol[1])

    def test_single_condition_and_commutativity(self):
        hol = np.array([0.21, 0.29])
        np.testing.assert_allclose(lcse_and(hol, np.array([[0.15], [0.4]])), hol)
        cond = np.array([[0.3, 0.1], [0.1, 0.4]])
        np.testing.assert_allclose(lcse_and(hol, cond), lcse_and(hol, cond[:, ::-1]))

    def test_official_holistic_clipping_and_extremes_are_finite(self):
        hol = np.array([1.0, -1.0])
        cond = np.array([[1.0, 1.0], [-1.0, -1.0]])
        result = lcse_and(hol, cond)
        bound = math.log((1 - 1e-6) / 1e-6) / 30
        np.testing.assert_allclose(result, [0.22 + bound, 0.22 - bound], atol=1e-9)
        self.assertTrue(np.isfinite(lcse_and(hol, cond, gamma=-1000)).all())

    def test_same_backbone_controls_and_product_log_equivalence(self):
        hol = np.array([[0.2, 0.3, 0.1]])
        cond = np.array([[[0.2, 0.4], [0.3, 0.3], [0.1, 0.2]]])
        scores = aggregate_scores(hol, cond)
        self.assertEqual(set(scores), {"holistic", "mean_cond", "min_cond", "product_cond", "logsigmoid_cond", "lcse_and"})
        self.assertTrue(all(x.shape == hol.shape for x in scores.values()))
        np.testing.assert_allclose(scores["mean_cond"], [[0.3, 0.3, 0.15]])
        np.testing.assert_allclose(scores["min_cond"], [[0.2, 0.3, 0.1]])
        np.testing.assert_allclose(np.log(scores["product_cond"]), scores["logsigmoid_cond"])
        self.assertFalse(np.allclose(scores["lcse_and"], scores["min_cond"]))

    def test_no_llr_or_monotonic_refinement_guarantee(self):
        # Every added log-probability lowers the score, even for p(correct)=.9.
        start = math.log(0.8)
        self.assertLess(start + math.log(0.9), start)
        # A partial match can outrank a full match despite per-condition signs.
        full, partial = np.array([0.6, 0.6]), np.array([0.99, 0.49])
        self.assertLess(np.log(full).sum(), np.log(partial).sum())
        # A query-only prior subtraction leaves image ordering unchanged.
        raw = np.array([0.1, -0.5, 0.7])
        np.testing.assert_array_equal(np.argsort(raw), np.argsort(raw - math.log(0.1)))

    def test_rejects_malformed_scores(self):
        for hol, cond in (([0.2], [[0.1, np.nan]]), ([0.2], [[0.1], [0.2]]), ([0.2], [[3.0, 0.2]])):
            with self.assertRaises(ValueError):
                lcse_and(hol, cond)
        for kwargs in ({"beta": 0}, {"gamma": 0}, {"mu": None}):
            with self.assertRaises(ValueError):
                lcse_and([0.2], [[0.1, 0.3]], **kwargs)


class CalibrationTests(unittest.TestCase):
    def fixture(self):
        return dict(condition_cosines=np.array([[0.8, 0.1], [0.7, 0.2]]),
                    condition_labels=np.array([[1, 0], [1, 0]]), image_ids=[10, 11],
                    excluded_image_ids=[20, 21], backbone="synthetic",
                    source_manifest_sha256="a" * 64, mu_grid=[0.1, 0.4, 0.7], beta_grid=[1, 30])

    def test_train_only_fit_has_verifiable_provenance_and_grid(self):
        artifact = fit_presence_calibration(**self.fixture())
        self.assertEqual(validate_calibration_artifact(artifact, "synthetic", excluded_image_ids=[20, 21]),
                         {"mu": 0.4, "beta": 30.0, "gamma": -1.0})
        self.assertEqual(artifact["models"]["synthetic"]["concept_templates"], list(CONCEPT_TEMPLATES))
        self.assertEqual(artifact["source"]["positive_counts_per_concept"], [2, 0])
        self.assertEqual(artifact, fit_presence_calibration(**self.fixture()))

    def test_rejects_dev_sealed_or_overlapping_calibration(self):
        for split in ("val2017", "dev", "sealed"):
            with self.assertRaisesRegex(ValueError, "train2017"):
                fit_presence_calibration(**self.fixture(), source_split=split)
        params = self.fixture()
        params["excluded_image_ids"] = [11, 20]
        with self.assertRaisesRegex(ValueError, "overlap"):
            fit_presence_calibration(**params)

    def test_artifact_detects_tampering_and_rechecks_actual_evaluation_ids(self):
        artifact = fit_presence_calibration(**self.fixture())
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_calibration_artifact(artifact, "synthetic", excluded_image_ids=[10, 99])
        for path, value in ((["scope"], "dev"), (["source", "image_ids"], [10, 12]),
                            (["models", "synthetic", "concept_templates"], ["{}"])):
            changed = copy.deepcopy(artifact)
            obj = changed
            for key in path[:-1]:
                obj = obj[key]
            obj[path[-1]] = value
            with self.assertRaises(ValueError):
                validate_calibration_artifact(changed, "synthetic")

    def test_rejects_nonbinary_and_unidentifiable_calibration(self):
        for labels in (np.array([[1, 0.5], [1, 0]]), np.ones((2, 2))):
            params = self.fixture()
            params["condition_labels"] = labels
            with self.assertRaises(ValueError):
                fit_presence_calibration(**params)


if __name__ == "__main__":
    unittest.main(verbosity=2)
