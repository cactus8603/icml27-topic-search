"""[codex] Source-identity checks for caption preservation and original ZIP image evidence."""
import io
from pathlib import Path
import struct
import unittest
import zipfile

from PIL import Image

from codex.retrieval_karpathy_data import acquire as A


class AcquisitionTests(unittest.TestCase):
    def fixture(self):
        stream = io.BytesIO()
        Image.new('RGB', (31, 29), (50, 100, 150)).save(stream, format='JPEG')
        image = stream.getvalue()
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, 'w') as z:
            z.writestr('val2014/COCO_val2014_000000000001.jpg', image)
        archive_data = archive.getvalue()
        eocd = archive_data.rfind(b'PK\x05\x06')
        fields = struct.unpack_from('<4s4H2IH', archive_data, eocd)
        central = archive_data[fields[6]:fields[6] + fields[5]]
        return image, central

    def test_source_zip_crc_and_dimensions(self):
        image, central = self.fixture()
        expected = A.parse_central(central)['val2014/COCO_val2014_000000000001.jpg']
        checked = A.inspect_bytes(image, expected, (31, 29))
        self.assertEqual(checked['width'], 31)
        self.assertEqual(checked['bytes'], len(image))

    def test_reencoded_or_corrupted_cache_is_not_equivalent(self):
        image, central = self.fixture()
        expected = next(iter(A.parse_central(central).values()))
        with self.assertRaisesRegex(ValueError, 'CRC'):
            A.inspect_bytes(image[:-1] + b'\x00', expected, (31, 29))
        with self.assertRaisesRegex(ValueError, 'dimensions'):
            A.inspect_bytes(image, expected, (29, 31))

    def test_incomplete_or_duplicated_central_records_rejected(self):
        _, central = self.fixture()
        for bad in (central[:-1], central + central):
            with self.assertRaises(ValueError):
                A.parse_central(bad)

    def test_six_original_captions_and_whitespace_preserved(self):
        raw = [' caption 0. ', 'caption1\nsecond line', 'two', 'three', 'four', 'sixth caption.']
        rows = [{'cocoid': 1, 'imgid': 91, 'filename': 'COCO_val2014_000000000001.jpg', 'filepath': 'val2014',
                 'sentences': [{'raw': text, 'sentid': i + 100} for i, text in enumerate(raw)]}]
        result = A.manifest_rows(rows, Path('/fixed/images'))
        self.assertEqual(result[0]['captions'], raw)
        self.assertEqual(result[0]['sentids'], list(range(100, 106)))
        self.assertEqual(result[0]['id'], '1')
        self.assertEqual(result[0]['imgid'], 91)


if __name__ == '__main__':
    unittest.main()
