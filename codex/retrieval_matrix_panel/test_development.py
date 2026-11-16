"""[codex] Meaningful isolation and deterministic complete-gallery checks."""
import unittest
from codex.retrieval_matrix_panel.build_dev import choose_gallery


class DevelopmentIsolation(unittest.TestCase):
    def test_excludes_all_test_sources_and_preserves_size(self):
        images = [{"id": i} for i in range(10)]
        selected = choose_gallery(images, {0, 2, 4, 8}, lambda *v: v[-1], size=5)
        self.assertEqual([r["id"] for r in selected], [1, 3, 5, 6, 7])

    def test_insufficient_pool_fails_instead_of_shrinking_gallery(self):
        with self.assertRaises(ValueError):
            choose_gallery([{"id": 0}, {"id": 1}], {1}, lambda *v: v[-1], size=2)


if __name__ == "__main__":
    unittest.main()
