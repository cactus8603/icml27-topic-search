"""[codex] Synthetic isolation, identity, pointer, and comparison-graph tests."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import benchmark_suite as B


class SuiteTests(unittest.TestCase):
    def setUp(self):
        task_tmp = B.ROOT / "hf_home" / "tmp"
        task_tmp.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="suite-test-", dir=task_tmp)
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "repo"
        self.root.mkdir()
        self.document = {"schema": 1, "complete": True, "metrics": {"a/b": {"~name": [0, False]}}}
        (self.root / "public.json").write_text(json.dumps(self.document))
        (self.root / "payload.bin").write_bytes(b"synthetic fixed payload\x00\xff")
        self.suite = {"schema": 1, "suite_id": "public-development-v1", "formal_evidence": False,
            "artifacts": {
                "panel": {"path": "public.json", "kind": "json", "sha256": self.sha("public.json"),
                          "expect": {"/schema": 1, "/complete": True, "/metrics/a~1b/~0name/1": False}},
                "payload": {"path": "payload.bin", "kind": "file", "sha256": self.sha("payload.bin")}},
            "datasets": {"neg": {"public": True}, "and": {}},
            "methods": {"core2": {"family": "CORE"}, "candidate": {"extra": ["allowed"]}},
            "cells": [
                {"id": "baseline", "dataset": "neg", "method": "core2", "artifact_refs": ["panel"],
                 "role": "baseline", "cost_class": "2b-rerank50"},
                {"id": "method", "dataset": "neg", "method": "candidate", "artifact_refs": ["panel", "payload"],
                 "role": "development", "cost_class": "2b-rerank50", "baseline_for_comparison": "core2"}]}

    def sha(self, relative):
        return hashlib.sha256((self.root / relative).read_bytes()).hexdigest()

    def save(self, suite=None):
        path = self.root / "suite.json"
        path.write_text(json.dumps(self.suite if suite is None else suite))
        return path

    def verify(self, suite=None):
        return B.verify_suite(self.save(suite), self.root)

    def test_valid_identity_only_and_does_not_follow_other_paths(self):
        self.suite["methods"]["candidate"]["result_path"] = "sealed/do-not-open.json"
        result = self.verify()
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["scope"], "registry-identity-only")
        self.assertIs(result["formal_evidence"], False)
        self.assertEqual((result["artifact_count"], result["cell_count"]), (2, 2))
        self.assertEqual(result["suite_sha256"], self.sha("suite.json"))

    def test_changed_artifact_rejected_with_suite_identity(self):
        (self.root / "payload.bin").write_bytes(b"changed")
        with self.assertRaisesRegex(B.VerificationError, "SHA256 mismatch") as caught:
            self.verify()
        self.assertEqual(caught.exception.suite_sha256, self.sha("suite.json"))

    def test_absolute_parent_and_windows_paths_rejected(self):
        for value in (str(self.root / "public.json"), "../outside.json", "nested/../public.json",
                      "C:/public.json", "nested\\public.json"):
            with self.subTest(path=value):
                suite = copy.deepcopy(self.suite)
                suite["artifacts"]["panel"]["path"] = value
                with self.assertRaises(B.VerificationError):
                    self.verify(suite)

    def test_forbidden_components_are_case_insensitive_substrings(self):
        for directory in ("SeAlEd", "PRIVATE-archive", "old-retrieval-independent-panel-copy"):
            with self.subTest(directory=directory):
                (self.root / directory).mkdir()
                (self.root / directory / "x").write_bytes(b"never read")
                suite = copy.deepcopy(self.suite)
                suite["artifacts"]["payload"]["path"] = directory + "/x"
                with self.assertRaisesRegex(B.VerificationError, "forbidden"):
                    self.verify(suite)

    def test_external_and_forbidden_symlinks_rejected_internal_allowed(self):
        (self.base / "outside.bin").write_bytes(b"outside")
        (self.root / "alias.bin").symlink_to(self.base / "outside.bin")
        suite = copy.deepcopy(self.suite)
        suite["artifacts"]["payload"]["path"] = "alias.bin"
        with self.assertRaisesRegex(B.VerificationError, "escapes"):
            self.verify(suite)
        (self.root / "alias.bin").unlink()
        (self.root / "private").mkdir()
        (self.root / "private" / "hidden.bin").write_bytes(b"hidden")
        (self.root / "alias.bin").symlink_to("private/hidden.bin")
        with self.assertRaisesRegex(B.VerificationError, "forbidden"):
            self.verify(suite)
        (self.root / "alias.bin").unlink()
        (self.root / "alias.bin").symlink_to("payload.bin")
        self.assertEqual(self.verify(suite)["status"], "PASS")

    def test_pointer_empty_member_escapes_and_root(self):
        document = {"": {"~1": [False]}, "01": "dict key"}
        self.assertIs(B.json_pointer(document, ""), document)
        self.assertIs(B.json_pointer(document, "//~01/0"), False)
        self.assertEqual(B.json_pointer(document, "/01"), "dict key")
        for pointer in ("not/pointer", "/~", "/~2", "/~10~x"):
            with self.subTest(pointer=pointer), self.assertRaises(ValueError):
                B.json_pointer(document, pointer)

    def test_strict_array_indices_and_scalar_traversal(self):
        for pointer in ("/00", "/01", "/-", "/-1", "/+0", "/ 0", "/1.0", "/1", "/0/x"):
            with self.subTest(pointer=pointer), self.assertRaises(ValueError):
                B.json_pointer([True], pointer)

    def test_nested_expectations_are_type_sensitive(self):
        for pointer, value in (("/complete", 1), ("/schema", True), ("/schema", 1.0),
                               ("/metrics", {"a/b": {"~name": [0, 0]}})):
            with self.subTest(pointer=pointer, value=value):
                suite = copy.deepcopy(self.suite)
                suite["artifacts"]["panel"]["expect"] = {pointer: value}
                with self.assertRaisesRegex(B.VerificationError, "expectation mismatch"):
                    self.verify(suite)

    def test_pointer_missing_member_is_not_a_null_match(self):
        self.suite["artifacts"]["panel"]["expect"] = {"/absent": None}
        with self.assertRaisesRegex(B.VerificationError, "Missing JSON pointer"):
            self.verify()

    def test_schema_formal_claim_and_required_collections(self):
        for field, value in (("schema", True), ("schema", 1.0), ("formal_evidence", 0),
                             ("formal_evidence", True), ("suite_id", " "),
                             ("artifacts", {}), ("cells", []), ("methods", []), ("datasets", [])):
            with self.subTest(field=field, value=value):
                suite = copy.deepcopy(self.suite)
                suite[field] = value
                with self.assertRaises(B.VerificationError):
                    self.verify(suite)

    def test_duplicate_unknown_and_malformed_cell_references(self):
        changes = (("id", "baseline"), ("dataset", "missing"), ("method", "missing"),
                   ("artifact_refs", ["missing"]), ("artifact_refs", "panel"),
                   ("role", ""), ("baseline_for_comparison", "missing"))
        for field, value in changes:
            with self.subTest(field=field, value=value):
                suite = copy.deepcopy(self.suite)
                suite["cells"][1][field] = value
                with self.assertRaises(B.VerificationError):
                    self.verify(suite)

    def test_comparison_baseline_cannot_cross_dataset_or_cost_class(self):
        for field, value in (("dataset", "and"), ("cost_class", "embedding-only")):
            with self.subTest(field=field):
                suite = copy.deepcopy(self.suite)
                suite["cells"][0][field] = value
                with self.assertRaisesRegex(B.VerificationError, "same dataset and cost_class"):
                    self.verify(suite)

    def test_strict_json_and_file_expectation_rejected(self):
        for payload in (b'{"schema":1,"schema":1}', b'{"value":NaN}'):
            with self.subTest(payload=payload):
                (self.root / "public.json").write_bytes(payload)
                suite = copy.deepcopy(self.suite)
                suite["artifacts"]["panel"]["sha256"] = self.sha("public.json")
                with self.assertRaises(B.VerificationError):
                    self.verify(suite)
        suite = copy.deepcopy(self.suite)
        suite["artifacts"]["payload"]["expect"] = {"": None}
        with self.assertRaisesRegex(B.VerificationError, "expect requires"):
            self.verify(suite)

    def test_cli_pass_fail_is_compact_json_and_read_only(self):
        suite_path = self.save()
        before = {p.name: p.read_bytes() for p in self.root.iterdir()}
        args = [sys.executable, "-B", str(Path(B.__file__)), "verify", "--suite", str(suite_path),
                "--root", str(self.root)]
        env = {**os.environ, "TMPDIR": str(B.ROOT / "hf_home" / "tmp"), "PYTHONDONTWRITEBYTECODE": "1"}
        result = subprocess.run(args, text=True, capture_output=True, env=env, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(result.stdout.splitlines()), 1)
        self.assertEqual(json.loads(result.stdout)["status"], "PASS")
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.root.iterdir()})
        (self.root / "payload.bin").write_bytes(b"changed")
        result = subprocess.run(args, text=True, capture_output=True, env=env, check=False)
        self.assertEqual(result.returncode, 1, result.stderr)
        failed = json.loads(result.stdout)
        self.assertEqual(failed["status"], "FAIL")
        self.assertEqual(failed["suite_sha256"], self.sha("suite.json"))


if __name__ == "__main__":
    unittest.main()
