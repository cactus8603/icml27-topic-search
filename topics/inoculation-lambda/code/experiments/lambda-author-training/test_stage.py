import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import stage

sys.path.insert(0, str(stage.STAGE.parents[2] / "killtests/inoculation-lambda"))
from phasea_v2.inputs import snapshot_evidence
from phaseb.feasibility import fingerprint


class SnapshotAliasTest(unittest.TestCase):
    def test_physical_cache_alias_preserves_frozen_logical_binding(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            physical = root / "physical"
            logical = root / "logical"
            logical.symlink_to(physical, target_is_directory=True)
            cache = "models--Qwen--Qwen2.5-7B-Instruct"
            revision = Path(cache) / "snapshots" / ("a" * 40)
            snapshot = physical / revision
            snapshot.mkdir(parents=True)
            blobs = physical / cache / "blobs"
            blobs.mkdir()
            target = blobs / ("b" * 64)
            target.write_bytes(b"weight")
            (snapshot / "model.safetensors").symlink_to(Path("../../blobs") / target.name)
            for name in ("config", "tokenizer_config", "tokenizer"):
                (snapshot / f"{name}.json").write_text(json.dumps({}))

            alias = logical / revision
            with self.assertRaisesRegex(ValueError, "not a content-addressed"):
                snapshot_evidence(alias)

            def preflight(_checkout, path, _python):
                return {"inputs": {"snapshot": snapshot_evidence(path)}}

            evidence = snapshot_evidence(snapshot)
            evidence["snapshot"] = str(alias)
            expected = fingerprint({"snapshot": evidence})
            with patch("phaseb.feasibility.preflight", preflight), patch.object(stage, "INPUTS_SHA", expected):
                verified = stage.verify_inputs(None, alias, None)
                self.assertEqual(verified["inputs"]["snapshot"], evidence)
                self.assertEqual(verified["inputs_sha256"], expected)
                target.write_bytes(b"wrong size")
                with self.assertRaisesRegex(ValueError, "audited binding"):
                    stage.verify_inputs(None, alias, None)


if __name__ == "__main__":
    unittest.main()
