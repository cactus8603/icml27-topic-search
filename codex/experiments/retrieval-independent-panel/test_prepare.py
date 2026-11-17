from pathlib import Path
import sys
import unittest
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare import pair_keys, rerank, select_images, summary


def image(image_id, source):
    return {"id": image_id, "file_name": f"{image_id:012d}.jpg", "width": 120, "height": 100,
            "flickr_url": f"https://example.test/{source}_aabb.jpg"}


class IndependentPlanTest(unittest.TestCase):
    def test_roster_excludes_old_ids_and_flickr_sources_and_deduplicates_new_sources(self):
        train = [image(1, 100), image(2, 100), image(3, 300), image(4, 400), image(5, 400),
                 image(6, 600), image(7, 700)]
        val = [image(10, 300)]
        selected, blocked_ids, blocked_sources, eligible = select_images(train, val, {1}, count=3)
        self.assertEqual(blocked_ids, {1, 10})
        self.assertEqual(blocked_sources, {"100", "300"})
        self.assertEqual(eligible, 4)
        self.assertTrue({6, 7} <= selected.keys())
        self.assertEqual(len({4, 5} & selected.keys()), 1)
        self.assertEqual(selected, select_images(train[::-1], val, {1}, count=3)[0])

    def test_full_relevance_and_both_single_object_controls_after_pair_exclusions(self):
        exposed = pair_keys([{"key": [1, 2, 3], "anchor_key": [3, 1], "foil_key": None}],
                            ("key", "anchor_key", "foil_key"))
        self.assertEqual(exposed, {(1, 3)})
        rows = {10: {1, 2}, 20: {1, 2}, 30: {1, 3}, 40: {2}, 50: {3}}
        result = {tuple(query["key"]): query for query in rerank.annotations(rows, rows, exposed)}
        self.assertEqual(set(result), {(1, 2)})
        self.assertEqual(result[1, 2]["relevant"], [10, 20])
        self.assertIn(result[1, 2]["group"], [10, 20])
        del rows[40]
        self.assertEqual(rerank.annotations(rows, rows, exposed), [])

    def test_metadata_is_never_a_gpu_ready_receipt(self):
        panel = {"roster": {1: [120, 100], 2: [120, 100]}, "queries": [], "sources": {}, "exclusions": {},
                 "coverage": {"queries": 100, "groups": 85}}
        with patch.object(Path, "is_file", return_value=True), patch.object(Path, "is_symlink", return_value=False):
            result = summary(panel, Path("local-images"))
        self.assertTrue(result["metadata_meets_coverage_floor"])
        self.assertEqual(result["candidate_files_missing"], 0)
        self.assertFalse(result["gpu_ready"])
        self.assertEqual(result["status"], "image_validation_required")
        with patch.object(Path, "is_file", return_value=False):
            missing = summary(panel, Path("local-images"))
        self.assertEqual(missing["candidate_files_missing"], 2)
        self.assertEqual(result["plan_sha256"], missing["plan_sha256"])


if __name__ == "__main__":
    unittest.main()
