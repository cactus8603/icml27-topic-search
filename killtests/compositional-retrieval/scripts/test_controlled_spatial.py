"""Score-free regressions for the frozen controlled-image comparison."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import controlled_spatial as spatial


def annotations(subject, obj):
    captions = [f"A {subject} {r} a {obj}" for r in spatial.RELATIONS]
    tokens = ("left_of", "right_of", "in-front_of", "behind")
    return [{"image_path": f"data/controlled_clevr/{subject}_{token}_{obj}.jpeg",
             "caption_options": [caption, *[other for other in captions if other != caption]]}
            for token, caption in zip(tokens, captions)]


class ControlledSpatialTest(unittest.TestCase):
    def test_release_ground_truth_and_complete_quartet(self):
        rows = annotations("mug", "bowl")
        groups = spatial.read_groups(rows, 1)
        self.assertEqual([label.caption for label in groups[("mug", "bowl")].values()],
                         [row["caption_options"][0] for row in rows])
        with self.assertRaisesRegex(ValueError, "incomplete"):
            spatial.read_groups(rows[:-1], 1)
        swapped = rows[0]["caption_options"]
        swapped[0], swapped[1] = swapped[1], swapped[0]
        with self.assertRaisesRegex(ValueError, "annotation"):
            spatial.read_groups(rows, 1)

    def test_duplicate_excludes_whole_group(self):
        groups = spatial.read_groups(annotations("mug", "bowl") + annotations("fork", "cup"), 2)
        names = [row.filename for group in groups.values() for row in group.values()]
        hashes = {name: spatial.Fingerprint(str(i), 0, 0) for i, name in enumerate(names)}
        # An exact copy rejects the group even when the perceptual values do not match.
        target = names[-1]
        exact = spatial.Fingerprint(hashes[target].sha256, 2**64 - 1, 2**64 - 1)
        unique = spatial.Fingerprint("unique", 2**64 - 1, 0)
        # Change the second group's hashes so the screen keeps the first group independent.
        for name in names[4:]:
            hashes[name] = spatial.Fingerprint(hashes[name].sha256, 2**32 - 1, 2**32 - 1)
        excluded = spatial.screen_groups(groups, hashes, {"prior": {"exact": exact, "other": unique}})
        self.assertEqual(list(excluded), [("mug", "bowl")])
        self.assertEqual(excluded[("mug", "bowl")], [[target, "prior", "exact", "exact"]])
        with self.assertRaisesRegex(ValueError, "External near-duplicate candidate"):
            spatial.screen_groups(groups, hashes, {"prior": {"unknown": spatial.Fingerprint("new", 0, 0)}})
        self.assertTrue(spatial.near_duplicate(spatial.Fingerprint("a", 0, 0), spatial.Fingerprint("b", 7, 7)))
        self.assertFalse(spatial.near_duplicate(spatial.Fingerprint("a", 0, 0), spatial.Fingerprint("b", 0, 127)))

    def test_shuffles_are_one_to_one_and_disjoint(self):
        keys = [("mug", "bowl"), ("fork", "cup"), ("plate", "can")]
        anchors = spatial.anchor_assignment(keys)
        self.assertEqual(anchors, spatial.anchor_assignment(keys))
        self.assertEqual({donor for _, _, donor in anchors.values()}, set(keys))
        for key, (_, shuffled, donor) in anchors.items():
            self.assertFalse(set(key) & set(donor))
            self.assertEqual(shuffled, anchors[donor][0])
        with self.assertRaisesRegex(ValueError, "unique"):
            spatial.anchor_assignment([("mug", "bowl"), ("bowl", "cup")])

    def test_reverse_directions_share_each_anchor(self):
        key = ("mug", "bowl")
        groups = spatial.read_groups(annotations(*key), 1)
        vectors = np.eye(4)
        texts = np.concatenate((np.eye(4), [[1, 2, 3, 4], [4, 3, 2, 1]]))
        with patch.object(spatial, "encode_items", return_value=(vectors, texts)) as encode:
            result = spatial.margins(None, groups, {}, [key], {key: ("true", "control", None)}, {}, with_anchors=True)
        self.assertEqual(encode.call_args.args[2][-2:], ["true", "control"])
        np.testing.assert_array_equal(result["plain"], [[1, 1, 1, 1]])
        contrast = result["true_anchor"] - result["shuffled_anchor"]
        np.testing.assert_allclose(contrast.reshape(-1, 2).sum(axis=1), 0)
        np.testing.assert_array_equal(result["true_anchor"], [[1.25, 1, 1.25, 1]])

    def test_paired_win_difference_uses_half_ties_and_full_quartets(self):
        plain = np.array([[1.0, 1, -1, -1], [1, 1, 1, 1]])
        true = np.array([[1.0, 0, -1, -1], [1, 1, 1, 1]])
        shuffled = np.array([[-1.0, -1, -1, -1], [1, 1, 1, 1]])
        summary = spatial.summarize_scores({"plain": plain, "true_anchor": true, "shuffled_anchor": shuffled})
        self.assertEqual(summary["primary_true_minus_shuffled_win_rate_pp"]["point"], 18.75)
        self.assertEqual(summary["primary_true_minus_shuffled_win_rate_pp"]["n"], 2)
        self.assertEqual(summary["arms"]["true_anchor"]["ties"], 1)
        self.assertEqual(summary["arms"]["true_anchor"]["both_directions_strictly_correct_pp"], 50)
        self.assertEqual(summary["directional_trials"], 8)


if __name__ == "__main__":
    unittest.main()
