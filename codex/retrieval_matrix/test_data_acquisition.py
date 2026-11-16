"""Integrity tests for public benchmark acquisition; no network or GPU."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from codex.retrieval_matrix import data_acquisition as da


class IntegrityTests(unittest.TestCase):
    def setUp(self):
        scratch = da.REPO / "hf_home/tmp"
        scratch.mkdir(exist_ok=True, parents=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.root = Path(self.temp.name)
        self.path = self.root / "sample"
        self.path.write_bytes(b"original contents")

    def tearDown(self):
        self.temp.cleanup()

    def test_git_blob_digest_checks_content_and_size(self):
        payload = self.path.read_bytes()
        spec = {"size": len(payload), "blobId": hashlib.sha1(
            f"blob {len(payload)}\0".encode() + payload).hexdigest()}
        self.assertEqual(da.verify_payload(self.path, spec)["source_digest_kind"], "git_blob_sha1")
        self.path.write_bytes(b"tampered contents")
        with self.assertRaisesRegex(ValueError, "SHA1 mismatch"):
            da.verify_payload(self.path, spec)

    def test_lfs_uses_object_sha256(self):
        spec = {"size": self.path.stat().st_size, "blobId": "pointer hash is not payload hash",
                "lfs": {"sha256": da.sha256(self.path)}}
        self.assertEqual(da.verify_payload(self.path, spec)["source_digest_kind"], "lfs_sha256")
        self.path.write_bytes(b"tampered contents")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            da.verify_payload(self.path, spec)

    def test_wrong_length_rejected(self):
        with self.assertRaisesRegex(ValueError, "size mismatch"):
            da.verify_payload(self.path, {"size": 1, "blobId": "unused"})

    def test_no_absolute_or_traversal_paths(self):
        for path in ("/etc/passwd", "../image.jpg", "images/../../other", "."):
            with self.subTest(path=path), self.assertRaises(ValueError):
                da.safe_relative(path)
        self.assertEqual(da.safe_relative("images/vg/123.jpg"), Path("images/vg/123.jpg"))

    def test_metadata_requires_the_frozen_revision(self):
        path = self.root / "sugarcrepe_pp/source_api.json"
        path.parent.mkdir()
        path.write_text(json.dumps({"sha": "wrong", "siblings": []}))
        with self.assertRaisesRegex(ValueError, "wrong revision"):
            da.metadata("sugarcrepe_pp", self.root, offline=True)

    def test_offline_mode_does_not_fall_back_to_network(self):
        spec = {"rfilename": "data/missing.json", "size": 10, "blobId": "unused"}
        with self.assertRaises(FileNotFoundError):
            da.acquire_one("sugarcrepe_pp", spec, self.root, offline=True)

    def test_flickr_rejects_a_truncated_split(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        path = self.root / "flickr30k_test/data/test-00000-of-00001.parquet"
        path.parent.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist([{"image": {"bytes": b"", "path": "1.jpg"},
                                           "text": ["a"] * 5}]), path)
        with self.assertRaisesRegex(ValueError, "expected 1000 images"):
            da.export_flickr(self.root)

    def test_coco_prior_receipt_detects_valid_but_changed_jpeg(self):
        from PIL import Image
        relative = "images/coco/train2017/000000000294.jpg"
        target = self.root / "cola" / relative
        target.parent.mkdir(parents=True)
        Image.new("RGB", (2, 2), "red").save(target)
        expected = da.sha256(target)
        da.acquire_coco(relative, self.root, True, expected)
        Image.new("RGB", (2, 2), "blue").save(target)
        with self.assertRaisesRegex(ValueError, "differs from prior"):
            da.acquire_coco(relative, self.root, True, expected)


if __name__ == "__main__":
    unittest.main()
