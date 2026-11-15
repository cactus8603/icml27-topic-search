"""Check joining independent labels to both scored roles without changing evidence."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from evaluate import index_gold, make_split, manifest_sha, read_role, score
from receipt_contract import MODEL_PINS


class Binding:
    def __init__(self, payload):
        self.payload = payload

    def metadata(self):
        return {"panel_gold_sha256": hashlib.sha256(self.payload).hexdigest()}


class ReadbackTests(unittest.TestCase):
    def setUp(self):
        self.groups = {"q1": {"base-0": {"id": "id0"}}}
        self.row = {"id": "id0", "question_id": "q1", "split": "train",
                    "variant": "base-0", "gold": None, "gold_status": "length"}

    def test_independent_unknown_is_preserved(self):
        payload = json.dumps(self.row).encode()
        binding = Binding(payload)
        gold = index_gold(payload, {"train": self.groups}, binding)
        split = make_split(self.groups, gold, {"weak": {"id0": (None, "abstain")},
                                               "strong": {"id0": (1, "observed")}}, "train", binding)
        self.assertIsNone(split.rows[0]["gold"])
        self.assertEqual(split.rows[0]["weak_status"], "abstain")
        self.assertEqual(split.rows[0]["strong"], 1)

    def test_foreign_or_publisher_label_fields_are_rejected(self):
        for row in (self.row | {"split": "test"}, self.row | {"publisher_is_correct": True}):
            payload = json.dumps(row).encode()
            with self.assertRaises(ValueError):
                index_gold(payload, {"train": self.groups}, Binding(payload))

    def test_score_manifest_binds_role_and_rows(self):
        binding = Binding(b"gold")
        model, revision = MODEL_PINS["weak"]
        receipt = {
            "schema": 1, "role": "weak", "model": model, "model_revision": revision,
            "split": "train", "binding": binding.metadata(),
            "prompt_manifest_sha256": manifest_sha(self.groups, score.PROFILE),
            "records": [{"id": "id0", "verdict": None, "status": "missing"}],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            path.write_text(json.dumps(receipt))
            records, _ = read_role(path, "weak", "train", self.groups, binding)
            self.assertEqual(records, {"id0": (None, "missing")})
            path.write_text(json.dumps(receipt | {"prompt_manifest_sha256": "a" * 64}))
            with self.assertRaises(ValueError):
                read_role(path, "weak", "train", self.groups, binding)


if __name__ == "__main__":
    unittest.main()
