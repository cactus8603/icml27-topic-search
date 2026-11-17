import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from regen_exact import original_batch, review_plan, write_packet


class ExactReplayTest(unittest.TestCase):
    def test_replay_keeps_source_chunk_offsets_and_tail(self):
        for seed, expected in {
            0: (0, 500, 0, 32),
            499: (0, 500, 480, 500),
            500: (500, 1000, 500, 532),
            1999: (1500, 2000, 1980, 2000),
            2000: (2000, 2500, 2000, 2032),
            44999: (44500, 45000, 44980, 45000),
            45000: (45000, 45500, 45000, 45032),
            99999: (99500, 100000, 99980, 100000),
        }.items():
            with self.subTest(seed=seed):
                self.assertEqual(original_batch(seed, 500, 32, 100000), expected)
        with self.assertRaisesRegex(ValueError, "outside"):
            original_batch(100000, 500, 32, 100000)

    def test_copies_saved_thumbnail_and_groups_missing_seeds(self):
        with TemporaryDirectory() as temporary:
            run = Path(temporary)
            (run / "review").mkdir()
            items = "item,model,prompt,seed\n0,sd14,B1,452\n1,sd14,B1,482\n2,sd14,B1,499\n"
            (run / "review/review_items.csv").write_text(items)
            key = run / "review/review_key.csv"
            key.write_text(items)
            (run / "review/review_blind.csv").write_text("item,label,note\n0,,\n1,,\n2,,\n")
            metadata = run / "scores/sd14/B1/s0000000_0000500.json"
            metadata.parent.mkdir(parents=True)
            valid = dict(model="sd14", prompt_id="B1", seed_range=[0, 500], batch_size=32)
            metadata.write_text(json.dumps(valid))
            thumbnail = run / "images/sd14/B1/0000452_pos.jpg"
            thumbnail.parent.mkdir(parents=True)
            thumbnail.touch()
            copies, batches, sha = review_plan(run, {"sd14"}, {"B1"}, 500, 32, 100000)
            self.assertEqual(copies, {0: thumbnail, 1: None, 2: None})
            self.assertEqual(batches, {("sd14", "B1", 480, 500, metadata): [(1, 2), (2, 19)]})
            self.assertEqual(len(sha), 64)
            metadata.write_text(json.dumps({**valid, "batch_size": 16}))
            with self.assertRaisesRegex(ValueError, "source chunk"):
                review_plan(run, {"sd14"}, {"B1"}, 500, 32, 100000)
            metadata.write_text(json.dumps(valid))
            key.write_text(items.replace("482", "481"))
            with self.assertRaisesRegex(ValueError, "frozen selection"):
                review_plan(run, {"sd14"}, {"B1"}, 500, 32, 100000)

    def test_partial_render_is_not_published_and_retry_keeps_item_ids(self):
        with TemporaryDirectory() as temporary:
            review = Path(temporary)
            items = b"item,model,prompt,seed\n0,sd14,B1,452\n"
            (review / "review_items.csv").write_bytes(items)
            original = review / "original.jpg"
            original.write_bytes(b"original thumbnail")
            args = (review, {0: original}, {}, hashlib.sha256(items).hexdigest(), "score", "source", {}, {}, None)
            with patch("regen_exact.render", side_effect=RuntimeError("interrupted")):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    write_packet(*args)
            self.assertFalse((review / "exact_packet").exists())
            with patch("regen_exact.render"):
                write_packet(*args)
            packet = review / "exact_packet"
            self.assertEqual((packet / "review_imgs/0000.jpg").read_bytes(), original.read_bytes())
            self.assertIn('src="review_imgs/0000.jpg"', (packet / "index.html").read_text())
            self.assertEqual(json.loads((packet / "exact_replay.json").read_bytes())["score_sha256"], "score")


if __name__ == "__main__":
    unittest.main()
