from email.message import Message
import hashlib
from io import BytesIO
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import images


def pixel_bytes(color, size=(29, 21), format="JPEG"):
    stream = BytesIO()
    Image.new("RGB", size, color).save(stream, format=format)
    return stream.getvalue()


def asset(image_id):
    name = f"{int(image_id):012d}.jpg"
    return {"file_name": name, "url": f"{images.COCO_ORIGIN}train2017/{name}", "width": 29, "height": 21}


class Response(BytesIO):
    status = 200

    def __init__(self, url, data, sizes):
        super().__init__(data)
        self.url = url
        self.headers = Message()
        for size in sizes:
            self.headers.add_header("Content-Length", str(size))
        self.body_read = False

    def geturl(self):
        return self.url

    def read(self, *args):
        self.body_read = True
        return super().read(*args)


class ImageTest(unittest.TestCase):
    def setUp(self):
        self.data = pixel_bytes("red")
        self.work = self.enterContext(TemporaryDirectory(dir=images.HERE))
        self.directory = Path(self.work)

    def use_panel(self, assets, blocked=()):
        self.enterContext(patch.object(images, "read_plan", return_value=(assets, "prior")))
        self.enterContext(patch.object(images, "exclusions", return_value=set(blocked)))

    def prepare(self, mode):
        return images.prepare(mode, self.directory / "plan.json", self.directory)

    def test_download_requires_official_https_and_reserves_before_body(self):
        url = "https://s3.amazonaws.com/images.cocodataset.org/train2017/000000000001.jpg"
        response = Response(url, self.data, [len(self.data)])
        with patch.object(images, "build_opener") as opener:
            opener.return_value.open.return_value = response

            def reserve(image_id, size):
                self.assertEqual((image_id, size), ("1", len(self.data)))
                self.assertFalse(response.body_read)

            self.assertEqual(images.download("1", asset("1"), reserve), self.data)
            opener.return_value.open.assert_called_once_with(url, timeout=35)
        self.assertTrue(response.body_read)

    def test_bad_http_headers_or_changed_origin_fail_without_body(self):
        url = "https://s3.amazonaws.com/images.cocodataset.org/train2017/000000000001.jpg"
        for bad in (Response(url, self.data, [1, 2]),
                    Response(url.replace("cocodataset.org", "elsewhere.org"), self.data, [len(self.data)])):
            with self.subTest(url=bad.geturl()), patch.object(images, "build_opener") as opener:
                opener.return_value.open.return_value = bad
                with self.assertRaisesRegex(ValueError, "Unexpected official source"):
                    images.download("1", asset("1"), self.fail)
                self.assertFalse(bad.body_read)

    def test_full_jpeg_decode_rejects_png_wrong_dimensions_and_truncation(self):
        for bad in (pixel_bytes("red", format="PNG"), pixel_bytes("red", size=(21, 29)), self.data[:-8]):
            with self.assertRaises(ValueError):
                images.inspect("1", asset("1"), bad)

    def test_retry_preserves_charges_and_verifies_offline(self):
        self.use_panel({"1": asset("1")})
        path = self.directory / images.LEDGER

        def interrupted(image_id, asset, reserve):
            reserve(image_id, len(self.data))
            raise ValueError("Incomplete")

        def completed(image_id, asset, reserve):
            reserve(image_id, len(self.data))
            return self.data

        with patch.object(images, "download", interrupted), self.assertRaisesRegex(ValueError, "Incomplete"):
            self.prepare("fetch")
        first = path.read_bytes()
        self.assertFalse((self.directory / images.RECEIPT).exists())
        with patch.object(images, "download", completed):
            receipt = self.prepare("fetch")
        self.assertTrue(path.read_bytes().startswith(first))
        with patch.object(images, "download", side_effect=AssertionError), patch.object(
                images, "publish", side_effect=AssertionError):
            self.assertEqual(self.prepare("verify"), receipt)
        self.assertIn(f'"charged_bytes":{2 * len(self.data)}', (self.directory / images.RECEIPT).read_text())
        self.assertEqual(path.read_bytes().count(b"\n"), 3)
        (self.directory / "images" / asset("1")["file_name"]).write_bytes(pixel_bytes("blue"))
        with self.assertRaisesRegex(ValueError, "disagrees"):
            self.prepare("verify")

    def test_duplicate_of_prior_or_current_jpeg_fails_closed(self):
        self.use_panel({"1": asset("1")}, blocked={hashlib.sha256(self.data).hexdigest()})

        def completed(image_id, asset, reserve):
            reserve(image_id, len(self.data))
            return self.data

        with patch.object(images, "download", completed):
            with self.assertRaisesRegex(ValueError, "Exact JPEG duplicate"):
                self.prepare("fetch")
            with patch.object(images, "read_plan", return_value=({"1": asset("1"), "2": asset("2")}, "prior")):
                with patch.object(images, "exclusions", return_value=set()):
                    with self.assertRaisesRegex(ValueError, "Exact JPEG duplicate"):
                        self.prepare("fetch")
        self.assertFalse((self.directory / images.RECEIPT).exists())
        self.assertEqual(len(list((self.directory / "images").glob("*.jpg"))), 1)

    def test_cap_persists_across_attempts(self):
        self.use_panel({"1": asset("1")})

        def interrupted(image_id, asset, reserve):
            reserve(image_id, len(self.data))
            raise ValueError("Incomplete")

        with patch.object(images, "download", interrupted), patch.object(images, "CAP", len(self.data)):
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                self.prepare("fetch")
            with self.assertRaisesRegex(ValueError, "cumulative"):
                self.prepare("fetch")
        self.assertEqual((self.directory / images.LEDGER).read_bytes().count(b"\n"), 1)

    def test_existing_untracked_image_fails_closed(self):
        self.use_panel({"1": asset("1")})
        folder = self.directory / "images"
        folder.mkdir()
        (folder / asset("1")["file_name"]).write_bytes(self.data)
        with patch.object(images, "download", side_effect=AssertionError):
            with self.assertRaisesRegex(ValueError, "download ledger"):
                self.prepare("fetch")


if __name__ == "__main__":
    unittest.main()
