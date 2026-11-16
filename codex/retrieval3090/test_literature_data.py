"""[codex] Synthetic CPU integrity and data-isolation tests; no public labels/results."""
from __future__ import annotations

import csv
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import literature_data as D


def sample_rows():
    rows = {}
    for key, count in (("1", 4), ("2", 7), ("3", 5), ("4", 6)):
        rows[key] = {"captions": [f"source {key} caption {i}" for i in range(count)],
                     "positive_objects": [""], "negative_objects": ["absent"],
                     "original_filepath": f"data/coco/images/val2017/{int(key):012d}.jpg"}
    return rows


def gallery():
    # IDs 1 and 2 deliberately alias the exact same JPEG bytes.
    return [{"id": key, "sha256": ("a" if key in ("1", "2") else key) * 64,
             "path": f"/synthetic/{key}.jpg"} for key in ("1", "2", "3", "4")]


class LiteratureDataTest(unittest.TestCase):
    def setUp(self):
        temp_root = D.ROOT / "hf_home/tmp"
        temp_root.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="literature-data-tests-", dir=temp_root)
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name)

    def test_csv_variable_caption_counts_and_empty_audit_strings(self):
        path = self.path / "source.csv"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("image_id", "filepath", "captions", "positive_objects", "negative_objects"))
            writer.writeheader()
            for key, row in sample_rows().items():
                writer.writerow({"image_id": key, "filepath": row["original_filepath"],
                                 **{name: repr(row[name]) for name in ("captions", "positive_objects", "negative_objects")}})
        loaded = D.parse_retrieval(path)
        self.assertEqual([len(loaded[key]["captions"]) for key in loaded], [4, 7, 5, 6])
        self.assertEqual(loaded["1"]["positive_objects"], [""])

    def test_literal_parser_rejects_executable_and_non_string_lists(self):
        for value in ("__import__('os').system('false')", "['safe', 1]", "('not', 'list')", "[None]", "{'x':'y'}"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                D.string_list(value, "audit", audit=True)

    def test_paired_caption_alignment_and_source_metadata_guard(self):
        plain, negated = sample_rows(), sample_rows()
        negated["2"]["captions"].pop()
        with self.assertRaisesRegex(ValueError, "positions"):
            D.check_pair(plain, negated)
        negated = sample_rows()
        negated["1"]["positive_objects"] = ["different"]
        with self.assertRaisesRegex(ValueError, "metadata"):
            D.check_pair(plain, negated)

    def test_sampling_by_jpeg_group_not_id_and_stable_to_input_order(self):
        rows = sample_rows()
        selected, provenance = D.select_development(rows, gallery(), count=3)
        repeated, _ = D.select_development(dict(reversed(list(rows.items()))), list(reversed(gallery())), count=3)
        self.assertEqual(selected, repeated)
        self.assertEqual(provenance["groups_total"], 3)
        self.assertEqual(sum(key in selected for key in ("1", "2")), 1)
        self.assertEqual(len({entry["group"] for entry in selected.values()}), 3)

    def test_empty_raw_caption_preserved_but_never_selected_for_paired_dev(self):
        plain, negated = sample_rows(), sample_rows()
        for row in negated.values():
            row["captions"][0] = ""
        selected, groups = D.select_development(plain, gallery(), count=3, paired=negated)
        self.assertTrue(all(row["caption_index"] != 0 for row in selected.values()))
        full = D.retrieval_panel(negated, gallery(), selected, groups, "negated", dev=False)
        self.assertEqual(full["source_empty_captions"], 4)
        self.assertEqual(len(full["queries"]), 22)
        inputs = D.model_inputs(full)
        self.assertEqual(len(inputs["excluded_query_ids"]), 4)
        self.assertTrue(all(row["text"].strip() for row in inputs["queries"]))

    def test_plain_neg_dev_same_image_and_original_position(self):
        plain, negated = sample_rows(), sample_rows()
        for row in negated.values():
            row["captions"] = ["NEG " + value for value in row["captions"]]
        selected, groups = D.select_development(plain, gallery(), count=3, paired=negated)
        p = D.retrieval_panel(plain, gallery(), selected, groups, "plain", dev=True)
        n = D.retrieval_panel(negated, gallery(), selected, groups, "negated", dev=True)
        self.assertEqual([(q["source_image_id"], q["caption_index"]) for q in p["queries"]],
                         [(q["source_image_id"], q["caption_index"]) for q in n["queries"]])
        self.assertTrue(all(q["positive_ids"] == [q["source_image_id"]] for q in p["queries"]))
        self.assertIn("not exhaustive", p["gold_semantics"])

    def test_model_input_allowlist_excludes_gold_and_presence_metadata(self):
        rows = sample_rows()
        selected, groups = D.select_development(rows, gallery(), count=3)
        panel = D.retrieval_panel(rows, gallery(), selected, groups, "plain", dev=True)
        panel["queries"][0].update(positive_objects=["SECRET GOLD"], negative_objects=["SECRET NEG"], correct="SECRET ANSWER")
        encoded = json.dumps(D.model_inputs(panel))
        self.assertNotIn("SECRET", encoded)
        self.assertNotIn("positive_ids", encoded)
        self.assertNotIn("source_image_id", encoded)

    def test_factor_equivalence_gold_and_positions_are_preserved(self):
        row = {"sample_id": "eq", "image_id": 1, "image": "val2017/000000000001.jpg", "correct": "both",
               "caption_a": "not both dog and cat", "caption_b": "no dog or no cat", "operator": "AND", "test_type": "equivalence",
               "parse_a": {"operator": "AND", "concepts": [{"text": "dog", "is_negated": True}]},
               "parse_b": {"operator": "OR", "concepts": [{"text": "cat", "is_negated": True}]},
               "meta": {"present": ["SECRET GOLD"]}}
        panel, audit = D.matching_panel([row], gallery())
        item = panel["items"][0]
        self.assertEqual(item["correct"], "both")
        self.assertIsNone(item["correct_caption_index"])
        self.assertEqual(item["correct_caption_indices"], [0, 1])
        self.assertEqual(item["captions"], [row["caption_a"], row["caption_b"]])
        self.assertNotIn("queries", panel)
        self.assertNotIn("SECRET", json.dumps(D.model_inputs(panel)))
        self.assertEqual(audit["items"]["eq"], row["meta"])
        row["image"] = "val2017/000000000002.jpg"
        with self.assertRaisesRegex(ValueError, "map"):
            D.matching_panel([row], gallery())

    def test_source_receipt_tamper_rejected_before_csv_parsing(self):
        raw = self.path / "raw"
        raw.mkdir()
        entries = []
        for name, (size, digest) in D.NEG_FILES.items():
            (raw / name).write_bytes(b"tampered")
            entries.append({"file": name, "bytes": size, "sha256": digest})
        (raw / "source-receipt.json").write_text(json.dumps({"files": entries}))
        with self.assertRaisesRegex(ValueError, "Pinned NegBench"):
            D.load_sources(raw, self.path / "not-opened-inventory")

    def test_pinned_factor_rejects_changed_bytes(self):
        path = self.path / "factor.jsonl"
        path.write_text("{}\n")
        with self.assertRaisesRegex(ValueError, "pinned author"):
            D.factor_panel(path, gallery())

    def test_image_sha_checked_even_for_existing_valid_jpeg(self):
        from PIL import Image
        stream = BytesIO()
        Image.new("RGB", (3, 2), "red").save(stream, format="JPEG")
        data = stream.getvalue()
        expected = {"sha256": __import__("hashlib").sha256(data).hexdigest(), "width": 3, "height": 2}
        D.inspect_image(data, expected)
        with self.assertRaisesRegex(ValueError, "JPEG SHA"):
            D.inspect_image(data + b"changed", expected)
        expected["width"] = 4
        with self.assertRaisesRegex(ValueError, "dimensions"):
            D.inspect_image(data, expected)

    def test_zip_missing_image_publication_resumes_and_preserves_existing_bytes(self):
        from PIL import Image
        import hashlib
        stream = BytesIO()
        Image.new("RGB", (4, 3), "blue").save(stream, format="JPEG")
        data = stream.getvalue()
        inventory = {"1": {"sha256": hashlib.sha256(data).hexdigest(), "width": 4, "height": 3}}
        archive, images = self.path / "val.zip", self.path / "images"
        with zipfile.ZipFile(archive, "w") as output:
            output.writestr("val2017/000000000001.jpg", data)
            output.writestr("../unrequested", b"never extracted")
        first = D.candidates(inventory, images, archive)
        second = D.candidates(inventory, images)
        self.assertEqual(first, second)
        self.assertEqual((images / "000000000001.jpg").read_bytes(), data)
        self.assertFalse((self.path / "unrequested").exists())

    def test_bad_zip_image_is_not_published(self):
        archive, images = self.path / "bad.zip", self.path / "images"
        with zipfile.ZipFile(archive, "w") as output:
            output.writestr("val2017/000000000001.jpg", b"untrusted bytes")
        with self.assertRaisesRegex(ValueError, "JPEG SHA"):
            D.candidates({"1": {"sha256": "0" * 64, "width": 4, "height": 3}}, images, archive)
        self.assertFalse((images / "000000000001.jpg").exists())

    def test_existing_artifact_never_overwritten_and_equal_retry_works(self):
        path = self.path / "panel.json"
        D.publish(path, b"first")
        D.publish(path, b"first")
        with self.assertRaisesRegex(ValueError, "differs"):
            D.publish(path, b"second")
        self.assertEqual(path.read_bytes(), b"first")

    def test_forbidden_path_and_symlink_alias_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Sealed"):
            D.safe(self.path / "retrieval-independent-panel" / "manifest.json")
        target = self.path / "ordinary"
        target.mkdir()
        link = self.path / "alias"
        link.symlink_to(target, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "Symlink"):
            D.safe(link / "panel.json")

    def test_verify_lock_does_not_create_output_directory(self):
        missing = self.path / "absent"
        with self.assertRaises(FileNotFoundError):
            with D.writer(missing, readonly=True):
                pass
        self.assertFalse(missing.exists())


if __name__ == "__main__":
    unittest.main()
