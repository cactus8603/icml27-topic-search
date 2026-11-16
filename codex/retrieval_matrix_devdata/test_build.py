"""[codex] Logical boundary tests for annotation-derived AND+NOT truth."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("negative_dev_builder", Path(__file__).with_name("build.py"))
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)


class LogicTests(unittest.TestCase):
    def test_presence_requires_visible_non_crowd_instance(self):
        images = [{"id": i, "width": 100, "height": 100} for i in range(4)]
        annotations = [{"image_id": 0, "category_id": 1, "iscrowd": 0, "area": 100},
                       {"image_id": 1, "category_id": 1, "iscrowd": 0, "area": 99},
                       {"image_id": 2, "category_id": 1, "iscrowd": 1, "area": 500}]
        present, absent = B.status(images, annotations)
        self.assertEqual(list(B.indices(present[1])), [0])
        self.assertEqual(list(B.indices(absent[1])), [3])

    def test_all_false_and_ambiguous_combinations(self):
        # row0 satisfies A+B and absent C; row1 has C; row2 misses B;
        # row3 B is ambiguous; row4 C ambiguous. Only rows3/4 are ignore.
        images = [{"id": i} for i in range(5)]
        present = {1: 31, 2: 1 | 2 | 16, 3: 2}
        absent = {1: 0, 2: 4, 3: 1 | 4 | 8}
        pools = {2: [("x", (1, 2), 3, 1, 5)], 3: [], 4: []}
        q = B.select(pools, images, present, absent, {1: "dog", 2: "cat", 3: "person"})[0]
        self.assertEqual(q["positives"], [0])
        self.assertEqual(q["ignore"], [3, 4])
        self.assertEqual(q["conditions"], 3)
        self.assertIn("but no person.", q["text"])

    def test_selection_never_duplicates_positive_images(self):
        images = [{"id": i} for i in range(3)]
        pools = {2: [("a", (1, 2), 3, 1, 5), ("b", (1, 2), 4, 1, 6)], 3: [], 4: []}
        queries = B.select(pools, images, {1: 7, 2: 7}, {1: 0, 2: 0}, {1: "dog", 2: "cat", 3: "person", 4: "bird"})
        self.assertEqual(len(queries), 1)


if __name__ == "__main__":
    unittest.main()
