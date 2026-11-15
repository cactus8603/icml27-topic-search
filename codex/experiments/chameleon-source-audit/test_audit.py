from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit import compare, membership, source_inventory, stable_hash


class SourceAuditTests(unittest.TestCase):
    def test_exact_identity_resolves_order_and_explicit_ambiguity(self):
        index = source_inventory([{"prompt": "p", "topical_response": r} for r in ("a", "b", "a")])
        a, b, absent = (stable_hash(["p", r]) for r in ("a", "b", "z"))
        self.assertEqual(membership([b], index)["ordered_public_row_ids_sha256"], stable_hash([1]))
        unresolved = membership([a, absent], index)
        self.assertEqual((unresolved["unmatched"], unresolved["ambiguous_original_rows"]), (1, 1))
        self.assertIsNone(unresolved["ordered_public_row_ids_sha256"])

    def test_both_label_overlaps_count_repeated_rows_only_up_to_the_match(self):
        local = {0: ["a", "a", "b"], 1: ["c"]}
        candidate = {0: ["a"], 1: ["b", "c"]}
        actual = compare(local, candidate, {})
        self.assertEqual(actual["negative"]["same_label_exact_overlap"], 1)
        self.assertEqual(actual["negative"]["opposite_label_exact_overlap"], 1)
        self.assertEqual(actual["positive"]["same_label_exact_overlap"], 1)


if __name__ == "__main__":
    unittest.main()
