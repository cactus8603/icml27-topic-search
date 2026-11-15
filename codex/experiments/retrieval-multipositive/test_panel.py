import hashlib
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

from panel import describe_groups, image_metadata, relation, relation_index, select_groups, verified_json


class PanelTests(unittest.TestCase):
    def test_relation_requires_an_unambiguous_horizontal_separation(self):
        box = (0, 0, 20, 20)
        right = (22, 15, 20, 20)
        self.assertEqual(relation(box, right, 100), 0)
        self.assertEqual(relation(right, box, 100), 1)
        self.assertIsNone(relation(box, (21, 15, 20, 20), 100))
        self.assertIsNone(relation(box, (22, 16, 20, 20), 100))
        self.assertIsNone(relation(box, (10, 0, 20, 20), 100))

    def test_index_rejects_small_crowd_and_multiply_instantiated_categories(self):
        images = {i: {"width": 100, "height": 100} for i in range(1, 6)}
        names = {1: "chair", 2: "bowl"}
        annotations = []
        for image_id in images:
            x_a, x_b = (25, 0) if image_id == 5 else (0, 25)
            annotations.extend([
                {"image_id": image_id, "category_id": 1, "iscrowd": 0,
                 "area": 200, "bbox": (x_a, 0, 20, 20)},
                {"image_id": image_id, "category_id": 2, "iscrowd": int(image_id == 4),
                 "area": 199 if image_id == 3 else 200, "bbox": (x_b, 0, 20, 20)},
            ])
        annotations.append({"image_id": 2, "category_id": 2, "iscrowd": 0,
                            "area": 5, "bbox": (60, 0, 5, 5)})
        self.assertEqual(dict(relation_index(annotations, images, names, set())), {(1, 2): ([1], [5])})
        self.assertFalse(relation_index(annotations, images, names, {("bowl", "chair")}))

    def test_selection_preserves_global_flickr_disjointness_and_is_deterministic(self):
        images = {i: {"flickr_id": str(i)} for i in range(1, 25)}
        images[9] = {"flickr_id": "1"}
        pairs = {(1, 2): ([1, 2, 3, 4], [5, 6, 7, 8]),
                 (3, 4): ([9, 10, 11, 12, 13], [14, 15, 16, 17, 18]),
                 (5, 6): ([1, 2, 3, 19], [20, 21, 22, 23, 24])}
        chosen = select_groups(pairs, images)
        shuffled = {key: [side[::-1] for side in pairs[key]] for key in reversed(pairs)}
        self.assertEqual(chosen, select_groups(shuffled, images))
        self.assertEqual(set(chosen), {(1, 2), (3, 4)})
        self.assertNotIn(9, chosen[3, 4][0])

    def test_complementary_labels_and_one_to_one_noun_disjoint_control(self):
        chosen = {(1, 2): ([1, 2, 3, 4], [5, 6, 7, 8]),
                  (3, 4): ([9, 10, 11, 12], [13, 14, 15, 16])}
        groups = describe_groups(chosen, {1: "chair", 2: "book", 3: "cup", 4: "clock"})
        first, second = groups
        self.assertEqual(first["shuffled_donor_key"], second["key"])
        self.assertEqual(first["shuffled_anchor"], second["anchor"])
        self.assertEqual(first["queries"][0]["text"], "A photo with the chair to the left of the book.")
        self.assertEqual(first["queries"][1]["text"], "A photo with the chair to the right of the book.")
        self.assertEqual(first["images"], first["queries"][0]["relevant"] + first["queries"][1]["relevant"])
        with self.assertRaisesRegex(ValueError, "unique, noun-disjoint"):
            describe_groups({(1, 2): chosen[1, 2]}, {1: "chair", 2: "book"})

    def test_pins_json_bytes_and_official_identity(self):
        path = Mock()
        body = b'{"schema":1}'
        path.read_bytes.return_value = body
        self.assertEqual(verified_json(path, hashlib.sha256(body).hexdigest()), {"schema": 1})
        with self.assertRaisesRegex(ValueError, "Input changed"):
            verified_json(path, "invalid")
        official = {"id": 15, "height": 240, "width": 320, "file_name": "000000000015.jpg",
                    "coco_url": "http://images.cocodataset.org/train2017/000000000015.jpg",
                    "flickr_url": "http://farm7.staticflickr.com/99/10590_1a2345_z.jpg"}
        self.assertEqual(image_metadata(official)["flickr_id"], "10590")
        with self.assertRaisesRegex(ValueError, "official image URL"):
            image_metadata(dict(official, coco_url="http://invalid.test/train2017/000000000015.jpg"))


if __name__ == "__main__":
    unittest.main()
