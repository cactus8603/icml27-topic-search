"""Check the activation boundary and fixed-stage handoff with synthetic inputs."""

from io import BytesIO
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run


FREEZE = "abcd" * 10


def record():
    return {"schema": 1, "scope": run.SCOPE, "adopted_proxy": run.PROXY_CANDIDATE,
            "user_decision_ref": "Synthetic test user decision"}


class Stage1Test(unittest.TestCase):
    def test_activation_requires_the_exact_candidate_and_nonempty_decision(self):
        expected = record()
        self.assertEqual(run.activation(json.dumps(expected)), expected)
        bad = ({"scope": "stage2"}, {"schema": True}, {"user_decision_ref": " "},
               {"adopted_proxy": run.PROXY_CANDIDATE | {"slurm_job_id": "26289"}},
               {"adopted_proxy": run.PROXY_CANDIDATE | {"receipt_sha256": "0" * 64}})
        for mutation in bad:
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                run.activation(json.dumps(expected | mutation))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            run.activation('{"schema": 1, "schema": 1}')

    def test_uncommitted_or_missing_activation_cannot_launch(self):
        path = Mock()
        path.exists.return_value = False
        self.assertIsNone(run.read_activation(Path("/repo"), FREEZE, path))
        with self.assertRaisesRegex(ValueError, "disabled"):
            run.execute(Path("/repo"), Path("/archive"), {}, FREEZE, None)
        path.exists.return_value = True
        path.read_bytes.return_value = json.dumps(record()).encode()
        with self.assertRaisesRegex(ValueError, "committed"):
            run.read_activation(Path("/repo"), None, path)
        with patch("run.subprocess.check_output", return_value=b"{}"), self.assertRaisesRegex(ValueError, "differs"):
            run.read_activation(Path("/repo"), FREEZE, path)
        with patch("run.subprocess.check_output", return_value=path.read_bytes.return_value):
            self.assertEqual(run.read_activation(Path("/repo"), FREEZE, path), record())

    def test_scope_is_original_panel_with_disjoint_output(self):
        env = {"SLURM_JOB_ID": "335", "SLURM_ARRAY_JOB_ID": "330", "SLURM_ARRAY_TASK_ID": "5"}
        _, _, model = run.array_task(env)
        self.assertEqual(model, "advunlearn")
        with self.assertRaises(ValueError):
            run.array_task(env | {"SLURM_ARRAY_TASK_ID": "6"})
        with self.assertRaises(ValueError):
            run.output_root("../old")
        output = run.output_root(FREEZE)
        self.assertEqual(output.parts[-3:], ("t2irare", "stage1-candidate", FREEZE))
        command = run.generator_command(Path("/archive"), {"seed_range_per_cell": [0, 2000], "default_batch_size": 32},
                                        model, output)
        self.assertEqual(command[1:], [
            "-IB", "/archive/scripts/generate.py", "--model", "advunlearn", "--prompts", "all",
            "--seed-start", "0", "--seed-end", "2000", "--batch-size", "32", "--chunk", "500", "--out", str(output),
        ])

    def test_default_cache_refs_snapshot_and_unet_matches(self):
        cache = run.SHARED / "hf_home/hub"
        generator = {f"author/model{i}": str(i) * 40 for i in (1, 2, 3)}
        panel = {"generator_revisions": generator, "detectors": {"falconsai_revision": "4" * 40, "clip_revision": "5" * 40}}
        revisions = generator | {run.FAL_REPO: "4" * 40, run.CLIP_REPO: "5" * 40}
        roots = {repo: cache / f"models--{repo.replace('/', '--')}" for repo in revisions}
        refs = {root / "refs/main": revisions[repo] for repo, root in roots.items()}
        snapshots = {repo: root / "snapshots" / revisions[repo] for repo, root in roots.items()}
        overrides = {"models": {f"unet{i}": {"kind": "unet_pt", "repo": "author/model2", "file": f"unet/{i}.pt"}
                                for i in range(4)}}
        constants = SimpleNamespace(HF_HUB_CACHE=str(cache))
        snapshot = Mock(side_effect=lambda repo, *, local_files_only: snapshots[repo])
        single_file = Mock(side_effect=lambda repo, name, *, local_files_only: snapshots[repo] / name)
        hub = SimpleNamespace(constants=constants, snapshot_download=snapshot, hf_hub_download=single_file)
        with (patch.dict(sys.modules, {"huggingface_hub": hub}),
              patch.object(Path, "read_bytes", return_value=json.dumps(overrides).encode()),
              patch.object(Path, "read_text", autospec=True, side_effect=lambda path, **kwargs: refs[path]),
              patch.object(Path, "is_dir", return_value=True), patch.object(Path, "is_file", return_value=True)):
            self.assertEqual(run.verify_loader_refs(Path("/archive"), panel), revisions)
            self.assertEqual(snapshot.call_count, 5)
            self.assertEqual(single_file.call_count, 4)
            detector_ref = roots[run.CLIP_REPO] / "refs/main"
            refs[detector_ref] = "f" * 40
            with self.assertRaisesRegex(ValueError, "Default Hugging Face ref differs"):
                run.verify_loader_refs(Path("/archive"), panel)
            refs[detector_ref] = revisions[run.CLIP_REPO]
            snapshots[run.CLIP_REPO] = snapshots[run.CLIP_REPO].with_name("f" * 40)
            with self.assertRaisesRegex(ValueError, "snapshot resolves elsewhere"):
                run.verify_loader_refs(Path("/archive"), panel)
            snapshots[run.CLIP_REPO] = snapshots[run.CLIP_REPO].with_name(revisions[run.CLIP_REPO])
            single_file.side_effect = lambda repo, name, **kwargs: snapshots[repo].with_name("f" * 40) / name
            with self.assertRaisesRegex(ValueError, "checkpoint resolves elsewhere"):
                run.verify_loader_refs(Path("/archive"), panel)
            constants.HF_HUB_CACHE = "/migrated/unregistered/hub"
            with self.assertRaisesRegex(ValueError, "effective Hugging Face cache differs"):
                run.verify_loader_refs(Path("/archive"), panel)

    def test_coverage_checks_a_complete_chunk_and_rejects_a_bad_seed(self):
        output, model = Path("/synthetic"), "sd14"
        panel = {"prompt_ids": ["q"], "seed_range_per_cell": [0, 2], "default_batch_size": 32}
        evidence = Mock(DETECTORS=("nudenet", "falconsai", "clip"))
        evidence.validate_chunk.return_value = {"batch_size": 32}
        common = SimpleNamespace(load_json=lambda _: {"models": {}}, prompts_by_id=lambda: {})
        chunk = output / "scores/sd14/q/s0000000_0000002.npz"
        scores = {"seeds": np.arange(2), **{name: np.zeros(2, np.float32) for name in evidence.DETECTORS}}
        load = np.load

        def synthetic_npz(_, **options):
            payload = BytesIO()
            np.savez_compressed(payload, **scores)
            payload.seek(0)
            return load(payload, **options)

        with (patch.dict(sys.modules, {"common": common}), patch.object(Path, "is_file", return_value=True),
              patch.object(Path, "glob", return_value=[chunk]),
              patch.object(np, "load", side_effect=synthetic_npz)):
            self.assertEqual(run.verify_chunks(output, panel, model, evidence, complete=True), 1)
            for wrong in (np.array([0, 0]), np.array([0.0, 1.0])):
                scores["seeds"] = wrong
                with self.assertRaisesRegex(ValueError, "seed identities"):
                    run.verify_chunks(output, panel, model, evidence, complete=True)
        with (patch.dict(sys.modules, {"common": common}),
              patch.object(Path, "glob", return_value=[]),
              self.assertRaisesRegex(ValueError, "coverage")):
            run.verify_chunks(output, panel, model, evidence, complete=True)


if __name__ == "__main__":
    unittest.main()
