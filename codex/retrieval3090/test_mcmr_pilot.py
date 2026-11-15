"""[codex] Synthetic CPU tests; never reads COCO/MCMR research model outputs."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mcmr_pilot as M
import analysis_v2 as V


def synthetic(seed=15, q=4, n=210, d=16):
    rng = np.random.default_rng(seed)
    unit = lambda x: x/np.linalg.norm(x, axis=-1, keepdims=True)
    panel = {"candidates": [{"id": str(i), "sha256": f"{i:064x}", "text": f"product {i}", "image": f"{i}.jpg"}
                             for i in range(n)], "queries": []}
    for i in range(q):
        text = f"Product feature {i}. Another requirement {i} and not red or green."
        panel["queries"].append({"id": str(i), "text": text, "positive_ids": [str(i)], "clauses": M.clauses(text)})
    plain, _ = M.text_table(panel, M.parse_policy("min_cond_core"))
    vectors = {}
    for name in ("core_embed", "cs_clip"):
        vectors[name+"_image"] = unit(rng.normal(size=(n, d)))
        vectors[name+"_full"] = unit(rng.normal(size=(q, d)))
        vectors[name+"_condition"] = unit(rng.normal(size=(len(plain), d)))
    return panel, vectors


class MCMRPilotTests(unittest.TestCase):
    def temporary(self):
        directory = M.ROOT / "hf_home/tmp"
        directory.mkdir(parents=True, exist_ok=True)
        return TemporaryDirectory(dir=directory)

    def test_clauses_retain_exact_spans_and_do_not_split_boolean_or_decimal(self):
        text = "A $10.50 product and not red or blue. It must fold; soft and strong."
        rows = M.clauses(text)
        self.assertEqual(len(rows), 3)
        self.assertEqual("".join(r["verbatim"] for r in rows), text)
        for row in rows:
            self.assertEqual(text[row["start"]:row["end"]], row["verbatim"])
        self.assertIn("and not red or blue", rows[0]["text"])
        self.assertEqual(len(M.clauses("A single request and a second constraint")), 1)

    def test_parser_aliases_and_rejects_new_parameters(self):
        parsed = M.parse_policy("m1/top100/fuse_llr_cs_t1_w0.25")
        self.assertEqual(parsed["policy_id"], "m1/top100/fuse_logsigmoid_cs_t1_w0.25")
        self.assertEqual(parsed["backbone"], "cs_clip")
        for invalid in ("min_cond_unknown", "logsigmoid_core_t4", "fuse_min_cond_core_w0.3", "logic_core_embed_new_lcse_and"):
            with self.assertRaises(ValueError):
                M.parse_policy(invalid)
        with self.assertRaisesRegex(ValueError, "disagree"):
            M.parse_policy("m1/top100/core", depth=200)
        with self.assertRaisesRegex(ValueError, "calibration"):
            M.parse_policy("logic_core_embed_train_presence_calibrated_lcse_and")

    def test_policy_scores_match_legacy_two_condition_rules(self):
        panel, vectors = synthetic()
        depth = 200
        core = np.argsort(-(vectors["core_embed_full"] @ vectors["core_embed_image"].T), axis=1, kind="stable")
        plain, _ = M.text_table(panel, M.parse_policy("core"))
        where = {text: i for i, text in enumerate(plain)}
        index = np.array([[where[c["text"]] for c in q["clauses"]] for q in panel["queries"]])
        features = {}
        for name in ("core_embed", "cs_clip"):
            full = V.A.zscore(vectors[name+"_full"] @ vectors[name+"_image"].T)
            cond = V.A.zscore(vectors[name+"_condition"] @ vectors[name+"_image"].T)
            features[name+"_full"] = np.take_along_axis(full, core[:, :depth], axis=1)
            features[name+"_cond"] = np.stack([np.take_along_axis(cond[index[:, j]], core[:, :depth], axis=1)
                                                for j in range(2)], axis=-1)
        for old_name, scores in V.A.policies(features).items():
            policy = M.parse_policy(old_name)
            actual_core, chosen = M.admission(panel, vectors, policy)
            np.testing.assert_array_equal(actual_core, core)
            np.testing.assert_array_equal(chosen, V.A.admit(core[:, :depth], scores), err_msg=old_name)

    def test_admission_never_reads_positive_labels_or_cross_scores(self):
        panel, vectors = synthetic()
        policy = M.parse_policy("min_cond_core")
        expected = M.admission(panel, vectors, policy)
        poisoned = deepcopy(panel)
        for query in poisoned["queries"]:
            query["positive_ids"] = object()
        changed_vectors = dict(vectors, cross=object(), labels=object())
        actual = M.admission(poisoned, changed_vectors, policy)
        for a, b in zip(actual, expected):
            np.testing.assert_array_equal(a, b)

    def test_variable_clause_counts_and_unnormalized_template_mean(self):
        import logic_baselines as L
        panel, vectors = synthetic()
        for i, count in enumerate((1, 2, 3, 4)):
            text = "; ".join(f"Requirement {i}-{j}" for j in range(count))
            panel["queries"][i].update(text=text, clauses=M.clauses(text))
        policy = M.parse_policy("logic_core_embed_paperparams_transfer_lcse_and")
        plain, templated = M.text_table(panel, policy)
        self.assertEqual(len(templated), 4*len(plain))
        rng = np.random.default_rng(18)
        values = rng.normal(size=(len(plain), 4, 16))
        values /= np.linalg.norm(values, axis=-1, keepdims=True)
        vectors["core_embed_condition"] = values.mean(axis=1)
        core, selected = M.admission(panel, vectors, policy)
        self.assertEqual(selected.shape, (4, 50))
        table = {text: i for i, text in enumerate(plain)}
        raw = vectors["core_embed_condition"] @ vectors["core_embed_image"].T
        for i, query in enumerate(panel["queries"]):
            top = core[i:i+1, :200]
            cond = raw[[table[c["text"]] for c in query["clauses"]]][:, top[0]].T[None]
            full = vectors["core_embed_full"][i:i+1] @ vectors["core_embed_image"].T
            scores = L.aggregate_scores(full[:, top[0]], cond, **policy["lcse"]["parameters"])["lcse_and"]
            np.testing.assert_array_equal(selected[i], V.A.admit(top, scores)[0])

    def test_union_is_minimal_deduplicated_and_keeps_core_order(self):
        panel, vectors = synthetic()
        core, selected = M.admission(panel, vectors, M.parse_policy("min_cond_core"))
        pairs = M.cross_union(core, selected)
        self.assertTrue(200 <= len(pairs) <= 400)
        self.assertEqual(len(pairs), len(set(map(tuple, pairs))))
        for qi in range(4):
            actual = [ci for q, ci in pairs if q == qi]
            wanted = set(core[qi, :50]) | set(selected[qi])
            self.assertEqual(set(actual), wanted)
            self.assertEqual(actual, [int(i) for i in core[qi] if i in wanted])

    def test_final_metrics_count_full_gallery_and_detect_union_misalignment(self):
        panel, vectors = synthetic()
        policy = M.parse_policy("min_cond_core")
        core, selected = M.admission(panel, vectors, policy)
        # Construct a clear diagnostic positive admitted only by the alternate.
        target = int(next(x for x in selected[0] if x not in set(core[0, :50])))
        panel["queries"][0]["positive_ids"] = [str(target)]
        pairs = M.cross_union(core, selected)
        scores = np.array([100 if qi == 0 and ci == target else -float(ci) for qi, ci in pairs])
        report = M.analyze_arrays(panel, vectors, policy, pairs, scores, draws=100)
        self.assertGreater(report["paired_against_core50"]["recall_at_50_pp"]["point_pp"], 0)
        self.assertEqual(report["cost"]["deployment_pairs_per_arm"], 200)
        self.assertEqual(report["cost"]["offline_union_pairs"], len(pairs))
        with self.assertRaisesRegex(ValueError, "exact"):
            M.analyze_arrays(panel, vectors, policy, pairs[::-1], scores, draws=100)

    def test_shared_positive_product_images_form_connected_groups(self):
        panel, _ = synthetic()
        panel["queries"][1]["positive_ids"] = ["0", "2"]
        panel["candidates"][3]["sha256"] = panel["candidates"][2]["sha256"]
        self.assertEqual(M.connected_groups(panel), [0, 0, 0, 0])

    def test_single_cluster_reports_no_spurious_confidence_interval(self):
        panel, vectors = synthetic()
        for query in panel["queries"]:
            query["positive_ids"] = ["0"]
        policy = M.parse_policy("core")
        core, selected = M.admission(panel, vectors, policy)
        pairs = M.cross_union(core, selected)
        report = M.analyze_arrays(panel, vectors, policy, pairs, np.zeros(len(pairs)), draws=20)
        self.assertEqual(report["bootstrap"]["clusters"], 1)
        self.assertIsNone(report["paired_against_core50"]["ap_pp"]["ci95_pp"])

    def test_joint_messages_preserve_image_metadata_and_query_boundaries(self):
        frozen = M.frozen_module("synthetic_frozen", M.MODEL_PATH)
        config = M.frozen_config()
        adapter = object.__new__(M.JointAdapter)
        adapter.frozen = frozen
        adapter.name = "core_embed"
        adapter.inner = SimpleNamespace(config=config["core_embed"])
        messages = adapter.messages([("price: 20\nfeatures: soft", "synthetic-image")])[0]
        self.assertEqual(messages[0]["content"][0]["text"], "Represent the user's input.")
        self.assertEqual([x["type"] for x in messages[1]["content"]], ["image", "text"])
        adapter.name = "core_reranker"
        adapter.inner = SimpleNamespace(config=config["core_reranker"])
        messages = adapter.messages([("Full query", "Product metadata", "synthetic-image")])[0]
        content = messages[-1]["content"]
        self.assertEqual(content[-1], {"type": "text", "text": "Product metadata"})
        self.assertEqual(content[-2]["type"], "image")
        self.assertIn({"type": "text", "text": "Full query"}, content)

    def test_frozen_input_limit_rejects_before_model_forward(self):
        frozen = M.frozen_module("synthetic_limits", M.MODEL_PATH)
        class Processor:
            def apply_chat_template(self, *args, **kwargs):
                return {"input_ids": SimpleNamespace(shape=(1, 4097))}
        with self.assertRaisesRegex(ValueError, "token"):
            frozen.qwen_inputs(Processor(), [], {"image_min_pixels": 4096, "image_max_pixels": 1310720, "max_tokens": 4096})

    def test_original_metadata_image_mapping_and_hashes(self):
        from PIL import Image
        with self.temporary() as temp:
            directory = Path(temp)
            path = directory / "photo.png"
            Image.new("RGB", (12, 10)).save(path)
            candidate = {"candidate_id": "abc", "title": "shirt", "description": ["cotton"], "price": 19.5,
                         "features": ["red", "machine wash"], "Date First Available": "2019",
                         "images": [{"url": "https://example.invalid/photo.png"}]}
            pilot = {"candidate_ids": ["abc"], "queries": [{"id": "q", "text": "A red shirt.", "positive_ids": ["abc"]}]}
            receipt = {"archive_sha256": M.ARCHIVE_SHA, "candidate_id_to_filename": {"abc": "photo.png"},
                       "images": {"photo.png": {"sha256": M.sha(path), "bytes": path.stat().st_size}}}
            panel = M.build_panel(pilot, [candidate], receipt, directory, enforce_counts=False)
            self.assertIn("price: 19.5", panel["candidates"][0]["text"])
            self.assertIn("Date First Available: 2019", panel["candidates"][0]["text"])
            self.assertNotIn("positive_ids", panel["candidates"][0]["text"])
            path.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "differs"):
                M.build_panel(pilot, [candidate], receipt, directory, enforce_counts=False)

    def test_resume_uses_shared_checkpoint_implementation_and_source_binding(self):
        helper = M.checkpoint_helpers()
        with self.temporary() as temp:
            path = Path(temp) / "vectors.npy"
            def interrupted(rows):
                if rows[0] == 2:
                    raise RuntimeError("interrupted")
                return np.asarray(rows)[:, None]
            identity = {"source": "synthetic", "manifest": "a"*64, "environment": "cpu-test"}
            with self.assertRaises(RuntimeError):
                helper.batches(path, list(range(5)), 2, interrupted, identity)
            seen = []
            def finish(rows):
                seen.extend(rows)
                return np.asarray(rows)[:, None]
            actual = helper.batches(path, list(range(5)), 2, finish, identity)
            self.assertEqual(seen, [2, 3, 4])
            np.testing.assert_array_equal(actual[:, 0], np.arange(5))
            with self.assertRaisesRegex(ValueError, "provenance"):
                helper.batches(path, list(range(5)), 2, finish, identity | {"environment": "changed"})
            helper.batches(path, list(range(5)), 2, lambda _: self.fail("must reuse"), identity)
            changed = np.load(path, mmap_mode="r+")
            changed[0] = 20
            changed.flush()
            with self.assertRaisesRegex(ValueError, "integrity"):
                helper.batches(path, list(range(5)), 2, finish, identity)

    def test_fixed_design_and_sealed_paths_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "Sealed"):
            M.safe("dl/retrieval-independent-panel/queries.json")
        with self.temporary() as temp:
            path = Path(temp) / "policy.json"
            M.fixed_json(path, {"policy": "frozen"})
            with self.assertRaisesRegex(ValueError, "differs"):
                M.fixed_json(path, {"policy": "changed"})

    def test_complete_cache_resume_pipeline_with_synthetic_model_adapter(self):
        from PIL import Image
        calls = []
        def embeddings(texts):
            output = []
            for text in texts:
                generator = np.random.default_rng(int(M.digest(str(text))[:8], 16))
                value = generator.normal(size=16)
                output.append(value/np.linalg.norm(value))
            return np.asarray(output)
        class FakeAdapter:
            def __init__(self, frozen, name, snapshot, config):
                self.name = name
                self.inner = SimpleNamespace(encode=lambda texts: embeddings(texts))
            def run(self, rows):
                calls.append((self.name, len(rows)))
                if self.name == "core_embed":
                    return embeddings([text for text, image in rows])
                return np.array([len(text) for query, text, image in rows], dtype=float)
        panel, _ = synthetic(q=2)
        policy = M.parse_policy("min_cond_core")
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None))
        with self.temporary() as temporary:
            out = Path(temporary)
            images = out / "images"
            images.mkdir()
            for candidate in panel["candidates"]:
                Image.new("RGB", (2, 2)).save(images / candidate["image"])
            M.atomic_json(out / "panel.json", panel)
            M.atomic_json(out / "policy.json", policy)
            args = SimpleNamespace(out=out, images=images, joint_batch=8, cross_batch=4)
            names = {name: Path(name)/("a"*40) for name in ("core_embed", "core_reranker")}
            with patch.dict(sys.modules, {"torch": fake_torch}), patch.object(M, "JointAdapter", FakeAdapter), \
                 patch.object(M, "snapshots", return_value=names), patch.object(M, "environment", return_value={"synthetic": True}), \
                 patch.object(M, "placement", return_value={"synthetic": True}), \
                 patch.object(M, "script_sources", return_value={"synthetic": "b"*64}):
                M.run_gpu(args, panel, policy)
                first_count = len(calls)
                self.assertGreater(first_count, 0)
                meta = M.read_json(out/"meta.json")
                self.assertEqual(meta["state"], "COMPLETE")
                self.assertEqual(meta["cache_sha256"], M.sha(out/"cache.npz"))
                M.run_gpu(args, panel, policy)
                self.assertEqual(len(calls), first_count)
            with np.load(out/"cache.npz", allow_pickle=False) as arrays:
                report = M.analyze_arrays(panel, arrays, policy, arrays["pairs"].tolist(), arrays["cross"], draws=20)
            self.assertEqual(report["queries"], 2)
            self.assertLessEqual(report["cost"]["offline_union_pairs"], 200)

    def shared_fixture(self, directory):
        panel, vectors = synthetic(q=2)
        expected = M.index_identity(SimpleNamespace(joint_batch=1), panel, M.frozen_config(), Path("a"*40),
                                    {"torch": "synthetic", "driver": "synthetic"}, {"class": "synthetic"},
                                    {"mcmr_pilot.py": "b"*64})
        expected["embedding_dim"] = 16
        with M.writer(directory):
            np.savez(directory/"index.npz", core_embed_image=vectors["core_embed_image"], core_embed_full=vectors["core_embed_full"])
            M.atomic_json(directory/"index-meta.json", {"state": "COMPLETE", "identity": expected,
                          "index_sha256": M.sha(directory/"index.npz"), "placement": {"visible_devices": "0", "uuid": "synthetic-card-A"}})
        return panel, vectors, expected

    def test_shared_index_reuses_only_policy_independent_core_vectors(self):
        with self.temporary() as temporary:
            directory = Path(temporary)
            panel, vectors, expected = self.shared_fixture(directory)
            actual, receipt = M.read_shared_index(directory, expected)
            self.assertEqual(set(actual), {"core_embed_image", "core_embed_full"})
            np.testing.assert_array_equal(actual["core_embed_image"], vectors["core_embed_image"])
            self.assertTrue(receipt["read_only_reuse"])
            self.assertNotIn("policy", json.dumps(expected))
            # Placement differs between physical 3090s but never enters identity.
            meta = M.read_json(directory/"index-meta.json")
            meta["placement"] = {"visible_devices": "6", "uuid": "synthetic-card-B"}
            M.atomic_json(directory/"index-meta.json", meta)
            M.read_shared_index(directory, expected)

    def test_shared_index_rejects_changed_source_environment_batch_and_inputs(self):
        with self.temporary() as temporary:
            directory = Path(temporary)
            panel, vectors, expected = self.shared_fixture(directory)
            for field, value in (("source", "f"*40), ("model_revision", "f"*40),
                                 ("candidate_inputs_sha256", "f"*64), ("query_inputs_sha256", "f"*64),
                                 ("environment", {"torch": "changed"}), ("batch_sizes", {"joint_image": 2, "text": 32}),
                                 ("processor", {"class": "changed"}), ("sources", {"mcmr_pilot.py": "f"*64})):
                with self.assertRaisesRegex(ValueError, "differs", msg=field):
                    M.read_shared_index(directory, expected | {field: value})
            changed_panel = deepcopy(panel)
            changed_panel["candidates"][0]["text"] += " price: 10"
            args = SimpleNamespace(joint_batch=1)
            original = M.index_identity(args, panel, M.frozen_config(), Path("a"*40), {}, {}, {})
            changed = M.index_identity(args, changed_panel, M.frozen_config(), Path("a"*40), {}, {}, {})
            self.assertNotEqual(original["candidate_inputs_sha256"], changed["candidate_inputs_sha256"])

    def test_shared_index_cannot_import_a_nonfinal_policy_or_cross_scores(self):
        with self.temporary() as temporary:
            directory = Path(temporary)
            _, vectors, expected = self.shared_fixture(directory)
            M.atomic_json(directory/"policy.json", {"policy_id": "temporary placeholder"})
            with self.assertRaisesRegex(ValueError, "independent"):
                M.read_shared_index(directory, expected)
            (directory/"policy.json").unlink()
            np.savez(directory/"index.npz", core_embed_image=vectors["core_embed_image"],
                     core_embed_full=vectors["core_embed_full"], cross=np.zeros(2))
            meta = M.read_json(directory/"index-meta.json")
            meta["index_sha256"] = M.sha(directory/"index.npz")
            M.atomic_json(directory/"index-meta.json", meta)
            with self.assertRaisesRegex(ValueError, "only contain"):
                M.read_shared_index(directory, expected)

    def test_shared_index_rejects_incomplete_or_modified_array_bytes(self):
        with self.temporary() as temporary:
            directory = Path(temporary)
            _, _, expected = self.shared_fixture(directory)
            meta = M.read_json(directory/"index-meta.json")
            M.atomic_json(directory/"index-meta.json", meta | {"state": "PARTIAL"})
            with self.assertRaisesRegex(ValueError, "incomplete"):
                M.read_shared_index(directory, expected)
            M.atomic_json(directory/"index-meta.json", meta)
            with (directory/"index.npz").open("ab") as stream:
                stream.write(b"synthetic corruption")
            with self.assertRaisesRegex(ValueError, "differs"):
                M.read_shared_index(directory, expected)

    def test_index_then_policy_cache_reuses_model_independent_forward_outputs(self):
        from PIL import Image
        calls = []
        def embeddings(texts):
            result = []
            for text in texts:
                rng = np.random.default_rng(int(M.digest(str(text))[:8], 16))
                value = rng.normal(size=2048)
                result.append(value/np.linalg.norm(value))
            return np.asarray(result)
        processor = SimpleNamespace(tokenizer=SimpleNamespace(padding_side="left"))
        class FakeAdapter:
            def __init__(self, frozen, name, snapshot, config):
                self.name = name
                self.inner = SimpleNamespace(encode=embeddings, processor=processor)
            def run(self, rows):
                calls.append(self.name)
                if self.name == "core_embed":
                    return embeddings([text for text, image in rows])
                return np.array([len(text) for query, text, image in rows], dtype=float)
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None))
        fake_transformers = SimpleNamespace(AutoProcessor=SimpleNamespace(from_pretrained=lambda *a, **kw: processor))
        panel, _ = synthetic(q=2)
        policy = M.parse_policy("core")
        with self.temporary() as temporary:
            root = Path(temporary)
            images, index, out = root/"images", root/"shared-index", root/"policy-cache"
            images.mkdir(); index.mkdir(); out.mkdir()
            for candidate in panel["candidates"]:
                Image.new("RGB", (2, 2)).save(images/candidate["image"])
            for directory in (index, out):
                M.atomic_json(directory/"panel.json", panel)
            M.atomic_json(out/"policy.json", policy)
            names = {name: Path(name)/("a"*40) for name in ("core_embed", "core_reranker")}
            with patch.dict(sys.modules, {"torch": fake_torch, "transformers": fake_transformers}), \
                 patch.object(M, "JointAdapter", FakeAdapter), patch.object(M, "snapshots", return_value=names), \
                 patch.object(M, "environment", return_value={"synthetic-kernel": True}), \
                 patch.object(M, "placement", return_value={"visible_devices": "0"}), \
                 patch.object(M, "processor_metadata", return_value={"synthetic-processor": True}), \
                 patch.object(M, "script_sources", return_value={"synthetic": "b"*64}):
                with M.writer(index):
                    M.run_index(SimpleNamespace(out=index, images=images, joint_batch=8), panel)
                self.assertFalse((index/"policy.json").exists())
                self.assertEqual(M.read_json(index/"index-meta.json")["state"], "COMPLETE")
                core_calls = calls.count("core_embed")
                with M.writer(out):
                    M.run_gpu(SimpleNamespace(out=out, images=images, joint_batch=8, cross_batch=4, index=index), panel, policy)
                self.assertEqual(calls.count("core_embed"), core_calls)
            meta = M.read_json(out/"meta.json")
            self.assertTrue(meta["identity"]["shared_index"]["read_only_reuse"])
            self.assertEqual(meta["offline_union_pairs"], 100)


if __name__ == "__main__":
    unittest.main()
