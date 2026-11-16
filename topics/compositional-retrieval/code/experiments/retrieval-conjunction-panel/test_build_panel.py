"""[claude] Synthetic tests for build_panel.py (CPU, no data files).  Run: python -m unittest test_build_panel -v"""
import unittest

import build_panel as bp


def img(i, w=100, h=100):
    return {"id": i, "width": w, "height": h}


def ann(i, c, area, crowd=0):
    return {"image_id": i, "category_id": c, "area": area, "iscrowd": crowd}


class TestStatus(unittest.TestCase):
    def test_present_absent_ambiguous(self):
        p, a = bp.image_status([img(1)], [ann(1, 1, 200), ann(1, 2, 50), ann(1, 3, 500, crowd=1)])
        self.assertEqual(p[1], {1})            # 200 >= 1% of 10,000
        self.assertEqual(a[1], {2, 3})         # tiny instance and crowd-only are ambiguous, not absent

    def test_threshold_inclusive(self):
        p, _ = bp.image_status([img(1)], [ann(1, 1, 100)])
        self.assertEqual(p[1], {1})


class TestQueries(unittest.TestCase):
    def setUp(self):
        # categories 1..4; image 1 has {1,2}; images 2-7 have {1} only (hard negatives for (1,2) missing 2);
        # image 8 has {1} visible and 2 ambiguous -> must be ignored for (1,2); image 9 has {3,4} (unrelated).
        imgs = [img(i) for i in range(1, 10)]
        anns = [ann(1, 1, 500), ann(1, 2, 500)]
        anns += [ann(i, 1, 500) for i in range(2, 8)]
        anns += [ann(8, 1, 500), ann(8, 2, 10)]
        anns += [ann(9, 3, 500), ann(9, 4, 500)]
        self.gallery = [i["id"] for i in imgs]
        self.p, self.a = bp.image_status(imgs, anns)
        self.names = {1: "apple", 2: "dog", 3: "cat", 4: "egg"}

    def test_disjoint_positives_and_ignore(self):
        qs = bp.build_queries(self.gallery, self.p, self.a, self.names, "dev")
        seen = set()
        for q in qs:
            self.assertFalse(seen & set(q["positives"]))
            seen |= set(q["positives"])
            self.assertTrue(bp.POS_MIN <= len(q["positives"]) <= bp.POS_MAX)
            self.assertGreaterEqual(q["n_hard_negatives"], bp.HARDNEG_MIN)
        q12 = [q for q in qs if sorted(q["category_ids"]) == [1, 2]]
        self.assertEqual(len(q12), 1)
        self.assertEqual(q12[0]["positives"], [1])
        self.assertEqual(q12[0]["ignore"], [8])

    def test_deterministic(self):
        a = bp.build_queries(self.gallery, self.p, self.a, self.names, "dev")
        b = bp.build_queries(self.gallery, self.p, self.a, self.names, "dev")
        self.assertEqual(a, b)

    def test_hardneg_requirement_filters(self):
        # no image has three visible conditions with enough hard negatives, so no k=3 query
        qs = bp.build_queries(self.gallery, self.p, self.a, self.names, "dev")
        self.assertFalse([q for q in qs if q["k"] == 3])


class TestText(unittest.TestCase):
    def test_text(self):
        self.assertEqual(bp.query_text(["dog", "apple"]), "A photo that contains a dog and an apple.")
        self.assertEqual(bp.query_text(["dog", "cat", "egg"]), "A photo that contains a dog, a cat, and an egg.")


class TestDevSmall(unittest.TestCase):
    def test_subset_and_balance(self):
        qs = [{"id": f"dev-k{k}-{j}", "k": k} for k in bp.KS for j in range(200)]
        s = bp.dev_small(qs)
        self.assertEqual(len(s), 3 * bp.DEV_SMALL_PER_K)
        self.assertTrue(set(s) <= {q["id"] for q in qs})
        for k in bp.KS:
            self.assertEqual(sum(x.startswith(f"dev-k{k}-") for x in s), bp.DEV_SMALL_PER_K)


if __name__ == "__main__":
    unittest.main()
