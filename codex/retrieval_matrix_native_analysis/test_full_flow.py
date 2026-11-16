"""[codex] CPU synthetic integration tests of original native admission gates.

Run with dl/retrieval3090/envs/retrieval3090/bin/python -B -m unittest
codex.retrieval_matrix_native_analysis.test_full_flow.  This interpreter is
intentional: the ORIGINAL review gates verify its actual NumPy/Torch paths.
All PASS-shaped records are ephemeral synthetic fixtures, never P0 evidence.
"""
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import unittest

import numpy as np

from codex.retrieval_matrix_native_analysis import synthetic as S


@contextmanager
def restored(*paths):
    originals = {Path(path): Path(path).read_bytes() for path in paths}
    try:
        yield
    finally:
        for path, payload in originals.items():
            path.write_bytes(payload)


def rewrite_retrieval_array(f, benchmark, modality, values):
    """Rebind synthetic hashes so an intended deeper numeric gate is tested."""
    report = S.E.read(f.reports[benchmark])
    index_path = Path(report["index_receipt"]["path"])
    index = S.E.read(index_path)
    bundle = f.bundles[benchmark]
    items = f.Q.Q.data.inference_items(bundle) if f.family == "siglip2" else f.Q.ENGINE.data.inference_items(bundle)
    items = items[0 if modality == "images" else 1]
    plan = S.E.read(f.plan_path)
    path = Path(report["arrays"][modality]["path"])
    receipt, state = S.checkpoint(path, values, items,
        plan["settings"]["image_batch" if modality == "images" else "text_batch"],
        {**report["identity"], "modality": modality})
    index["arrays"][modality], index["checkpoints"][modality] = receipt, state
    report["arrays"][modality] = receipt
    if "embedding_receipt" in index:
        original_path = Path(index["embedding_receipt"]["path"])
        original = S.E.read(original_path)
        original["arrays"][modality], original["checkpoints"][modality] = receipt, state
        S.overwrite(original_path, original)
        index["embedding_receipt"] = S.E.entry(original_path)
    S.overwrite(index_path, index)
    report["index_receipt"] = S.E.entry(index_path)
    S.overwrite(f.reports[benchmark], report)


def all_retrieval_paths(f, name):
    report = S.E.read(f.reports[name])
    index = S.E.read(report["index_receipt"]["path"])
    result = [f.reports[name], report["index_receipt"]["path"]]
    result += [item["path"] for item in report["arrays"].values()]
    result += [item["path"] for item in index["checkpoints"].values()]
    if "embedding_receipt" in index:
        result.append(index["embedding_receipt"]["path"])
    return result


def rebind_retrieval_state(f, name, modality):
    report = S.E.read(f.reports[name])
    index_path = Path(report["index_receipt"]["path"])
    index = S.E.read(index_path)
    state = Path(index["checkpoints"][modality]["path"])
    index["checkpoints"][modality] = S.E.entry(state)
    if "embedding_receipt" in index:
        original_path = Path(index["embedding_receipt"]["path"])
        original = S.E.read(original_path)
        original["checkpoints"][modality] = S.E.entry(state)
        S.overwrite(original_path, original)
        index["embedding_receipt"] = S.E.entry(original_path)
    S.overwrite(index_path, index)
    report["index_receipt"] = S.E.entry(index_path)
    S.overwrite(f.reports[name], report)


class RetrievalFacadeTests(unittest.TestCase):
    """One real three-stage facade call for each reduced retrieval family."""

    def test_siglip2_original_review_make_admit_and_cleanup(self):
        with S.fixture("siglip2") as f:
            folder = f.root
            review, receipt, admitted = f.qualify()
            self.assertEqual([v["state"] for v in (review, receipt, admitted)], ["PASS"] * 3)
            self.assertEqual(receipt["scope"], "official-native-runtime-fix-p0-qualified")
            self.assertEqual(receipt["model_spec_sha256"], S.E.digest(f.model_spec))
            self.assertEqual(set(review["benchmarks"]), {"coco", "flickr"})
            self.assertTrue(f.patch_manifest["synthetic"])
            self.assertIn("Q.review/make/admit", f.patch_manifest["unchanged"])
        self.assertFalse(folder.exists())

    def test_paired_original_review_make_admit_preserves_macro_and_ties(self):
        with S.fixture("paired") as f:
            review, receipt, admitted = f.qualify()
            self.assertEqual([v["state"] for v in (review, receipt, admitted)], ["PASS"] * 3)
            self.assertEqual(receipt["scope"], "official-paired-runtime-fix-p0-qualified")
            for name, row in review["benchmarks"].items():
                self.assertEqual(row["counts"]["images"], len(f.bundles[name]["labels"]))
            metrics = review["benchmarks"]["sugarcrepe"]["metrics"]
            self.assertAlmostEqual(metrics["add_ITT"], (metrics["add_obj_ITT"] + metrics["add_att_ITT"]) / 2)
            self.assertNotEqual(metrics["add_ITT"], 3 / 5)  # macro, not the pooled complete rows
            self.assertLess(metrics["add_ITT"], 1.)  # equal scalar rows failed strict comparisons

    def test_each_facade_rejects_wrong_model_and_scope_and_changed_receipt(self):
        for family in ("siglip2", "paired"):
            with self.subTest(family=family), S.fixture(family) as f:
                _, receipt, _ = f.qualify()
                wrong = deepcopy(f.model_spec)
                wrong["variant"] = "SYNTHETIC-wrong-checkpoint"
                with self.assertRaises(ValueError):
                    f.Q.admit(S.E.entry(f.receipt_path), wrong)
                with restored(f.receipt_path):
                    changed = deepcopy(receipt)
                    changed["scope"] = "official-native-p0-qualified"
                    S.overwrite(f.receipt_path, changed)
                    with self.assertRaisesRegex(ValueError, "Old"):
                        f.Q.admit(S.E.entry(f.receipt_path), f.model_spec)
                with restored(f.receipt_path):
                    changed = deepcopy(receipt)
                    changed["matched_canonical_benchmarks"] = ["SYNTHETIC-forged"]
                    S.overwrite(f.receipt_path, changed)
                    with self.assertRaises(ValueError):
                        f.Q.admit(S.E.entry(f.receipt_path), f.model_spec)

    def test_each_facade_rejects_producer_review_missing_benchmark_and_runtime(self):
        for family in ("siglip2", "paired"):
            with self.subTest(family=family), S.fixture(family) as f:
                with self.assertRaisesRegex(ValueError, "producer"):
                    f.review("SYNTHETIC-producer")
                one = dict(f.reports)
                one.pop(next(iter(one)))
                with self.assertRaisesRegex(ValueError, "every"):
                    f.Q.review(f.plan_path, one, "synthetic-reviewer", accept_protocol=True)
                name = next(iter(f.reports))
                with restored(f.reports[name]):
                    report = S.E.read(f.reports[name])
                    report["identity"]["runtime"]["runtime_fix"]["batch"] = "SYNTHETIC-wrong-runtime"
                    S.overwrite(f.reports[name], report)
                    with self.assertRaisesRegex(ValueError, "runtime"):
                        f.review()
                with restored(f.reports[name]):
                    report = S.E.read(f.reports[name])
                    report["identity"]["runtime"]["modules"]["numpy"]["actual_version"] = "SYNTHETIC-wrong-import"
                    S.overwrite(f.reports[name], report)
                    with self.assertRaisesRegex(ValueError, "runtime"):
                        f.review()

    def test_each_facade_rejects_validly_rebound_float64_vectors(self):
        for family in ("siglip2", "paired"):
            with self.subTest(family=family), S.fixture(family) as f:
                name = next(iter(f.reports))
                with restored(*all_retrieval_paths(f, name)):
                    values = np.load(f.arrays[name]["images"]["path"], allow_pickle=False).astype(np.float64)
                    rewrite_retrieval_array(f, name, "images", values)
                    with self.assertRaisesRegex(ValueError, "FP32"):
                        f.review()

    def test_each_facade_rejects_changed_counts_model_digest_and_array_bytes(self):
        for family in ("siglip2", "paired"):
            with self.subTest(family=family), S.fixture(family) as f:
                name = next(iter(f.reports))
                for mutate in (lambda r: r["counts"].update(images=r["counts"]["images"] - 1),
                               lambda r: r["identity"].update(model_spec_sha256="0" * 64)):
                    with restored(f.reports[name]):
                        report = S.E.read(f.reports[name])
                        mutate(report)
                        S.overwrite(f.reports[name], report)
                        with self.assertRaises(ValueError):
                            f.review()
                path = Path(f.arrays[name]["images"]["path"])
                with restored(path):
                    value = np.load(path, allow_pickle=False)
                    value[0, 0] = 0.
                    np.save(path, value, allow_pickle=False)
                    with self.assertRaises(ValueError):
                        f.review()

    def test_each_facade_rechecks_review_receipt_and_checkpoint_state(self):
        for family in ("siglip2", "paired"):
            with self.subTest(family=family), S.fixture(family) as f:
                f.qualify()
                with restored(f.review_path):
                    reviewed = S.E.read(f.review_path)
                    reviewed["reviewer"] = "SYNTHETIC-producer"
                    S.overwrite(f.review_path, reviewed)
                    with self.assertRaises(ValueError):
                        f.make()
                name = next(iter(f.reports))
                report = S.E.read(f.reports[name])
                index = S.E.read(report["index_receipt"]["path"])
                state = Path(index["checkpoints"]["images"]["path"])
                with restored(*all_retrieval_paths(f, name)):
                    changed = S.E.read(state)
                    changed["fingerprint"] = "0" * 64
                    S.overwrite(state, changed)
                    rebind_retrieval_state(f, name, "images")
                    with self.assertRaisesRegex(ValueError, "checkpoint identity"):
                        f.review()


class NativeMCQFacadeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context = S.fixture("reranker", "qwen2")
        cls.f = cls.context.__enter__()
        cls.reviewed, cls.receipt, cls.admitted = cls.f.qualify()

    @classmethod
    def tearDownClass(cls):
        root = cls.f.root
        cls.context.__exit__(None, None, None)
        if root.exists():
            raise AssertionError("Synthetic MCQ arrays/receipts were not removed")

    def test_original_facade_full_denominators_and_lossless_native_scalars(self):
        f = self.f
        self.assertEqual([v["state"] for v in (self.reviewed, self.receipt, self.admitted)], ["PASS"] * 3)
        self.assertEqual(self.receipt["scope"], "official-native-reranker-mcq-qualified")
        self.assertEqual(self.reviewed["recomputed"]["rows"], 15912)
        self.assertEqual(self.reviewed["pairs"], 58891)
        self.assertEqual(len(self.reviewed["counts"]), 8)
        complete = S.E.read(f.out / "complete.json")
        self.assertEqual(complete["costs"]["attempted_pairs"], 58891)
        self.assertEqual(complete["costs"]["completed_call_forwards"], complete["costs"]["attempted_calls"])
        for name, row in complete["subsets"].items():
            self.assertEqual(np.load(f.out / (name + ".npy"), allow_pickle=False).dtype, np.float64)

    def test_wrong_model_scope_or_receipt_cannot_admit(self):
        f = self.f
        wrong = deepcopy(f.model_spec)
        wrong["repo"] = "SYNTHETIC-wrong-model"
        with self.assertRaises(ValueError):
            f.Q.admit(S.E.entry(f.receipt_path), wrong)
        for field, value in (("scope", "official-reranker-mcq"), ("variant", "jina_m0")):
            with restored(f.receipt_path):
                changed = deepcopy(self.receipt)
                changed[field] = value
                S.overwrite(f.receipt_path, changed)
                with self.assertRaises(ValueError):
                    f.Q.admit(S.E.entry(f.receipt_path), f.model_spec)

    def test_wrong_reviewer_model_identity_and_missing_complete_subset_fail(self):
        f = self.f
        with self.assertRaises(ValueError):
            f.review("SYNTHETIC-producer")
        for path, mutate in ((f.out / "identity.json", lambda r: r.update(variant="jina_m0")),
                             (f.out / "complete.json", lambda r: r["subsets"].pop("voc_mcq")),
                             (f.out / "complete.json", lambda r: r.update(examples=15911))):
            with restored(path):
                changed = S.E.read(path)
                mutate(changed)
                S.overwrite(path, changed)
                with self.assertRaises(ValueError):
                    f.review()

    def test_changed_raw_call_backend_and_cost_ledger_fail(self):
        f = self.f
        raw = next((f.out / "raw").iterdir())
        for path, mutate in ((raw, lambda r: r.update(instruction="SYNTHETIC-wrong-instruction")),
                             (f.out / "backend.json", lambda r: r.update(model_spec_sha256="0" * 64)),
                             (f.out / "complete.json", lambda r: r["costs"].update(attempted_calls=0))):
            with restored(path):
                changed = S.E.read(path)
                mutate(changed)
                S.overwrite(path, changed)
                with self.assertRaises(ValueError):
                    f.review()
        with restored(f.out / "calls.jsonl"):
            with (f.out / "calls.jsonl").open("a") as stream:
                stream.write('{"state":"STARTED","call":"SYNTHETIC-extra","pairs":1,"forward_budget":1}\n')
            with self.assertRaises(ValueError):
                f.review()

    def test_original_native_validator_rejects_changed_runtime_modules(self):
        from codex.retrieval_matrix_rerank_native_p0 import runner as N
        plan = S.E.read(self.f.plan_path)
        plan["environment"]["modules"]["numpy"]["actual_version"] = "SYNTHETIC-wrong-import"
        # This calls the untouched pre-review validation, before any source or
        # array is opened; the original CPU processor receipt binds all imports.
        with self.assertRaisesRegex(ValueError, "processor/template"):
            N.validate(plan)

    def test_rebound_raw_scalar_still_must_equal_original_array(self):
        f = self.f
        raw = next((f.out / "raw").iterdir())
        complete_path = f.out / "complete.json"
        with restored(raw, complete_path):
            changed = S.E.read(raw)
            changed["responses"][0].update(score=.625, official_raw=[.625])
            S.overwrite(raw, changed)
            complete = S.E.read(complete_path)
            complete["raw_files"][raw.name] = S.E.sha(raw)
            S.overwrite(complete_path, complete)
            with self.assertRaisesRegex(ValueError, "scalar differs"):
                f.review()

    def test_rebound_float32_score_array_fails_original_dtype_guard(self):
        from codex.retrieval_matrix_rerank_native_p0 import contracts as C
        f = self.f
        path = f.out / "cola.npy"
        state = path.with_suffix(".state.json")
        complete_path = f.out / "complete.json"
        with restored(path, state, complete_path):
            original = S.E.read(state)
            values = np.load(path, allow_pickle=False).astype(np.float32)
            np.save(path, values, allow_pickle=False)
            original.update(dtype=values.dtype.str, sha256=S.E.sha(path))
            for batch in original["batches"]:
                batch["sha256"] = S.hashlib.sha256(np.ascontiguousarray(values[batch["start"]:batch["end"]]).tobytes()).hexdigest()
            S.overwrite(state, original)
            complete = S.E.read(complete_path)
            complete["subsets"]["cola"] = C.R.array_receipt(path)
            S.overwrite(complete_path, complete)
            with self.assertRaisesRegex(ValueError, "shape changed"):
                f.review()

    def test_ties_denominators_and_wrong_macro_cannot_be_hidden_in_report(self):
        f = self.f
        report_path = f.out / "results/report.json"
        base = S.E.read(report_path)
        # Synthetic failures are exact ties, not removed examples.
        self.assertLess(base["subsets"]["cola"]["accuracy"], 1.)
        for mutate in (lambda r: r["subsets"]["voc_mcq"].update(rows=4951),
                       lambda r: r["benchmarks"]["scpp"].update(macro_accuracy=1.),
                       lambda r: r["benchmarks"].pop("negbench_mcq")):
            with restored(report_path):
                changed = deepcopy(base)
                mutate(changed)
                S.overwrite(report_path, changed)
                try:
                    result = f.review()
                except ValueError:
                    continue
                self.assertEqual(result["state"], "FAILED")

    def test_changed_checkpoint_fingerprint_and_review_receipt_fail(self):
        f = self.f
        state = f.out / "cola.state.json"
        complete_path = f.out / "complete.json"
        with restored(state, complete_path):
            from codex.retrieval_matrix_rerank_native_p0 import contracts as C
            changed = S.E.read(state)
            changed["fingerprint"] = "0" * 64
            S.overwrite(state, changed)
            complete = S.E.read(complete_path)
            complete["subsets"]["cola"] = C.R.array_receipt(f.out / "cola.npy")
            S.overwrite(complete_path, complete)
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                f.review()
        with restored(f.review_path):
            changed = S.E.read(f.review_path)
            changed["recomputed"]["rows"] = 15911
            S.overwrite(f.review_path, changed)
            with self.assertRaises(ValueError):
                f.make()


class JinaAxisFullFacadeTests(unittest.TestCase):
    def test_image_choice_native_groups_full_facade_and_cleanup(self):
        from codex.retrieval_matrix_rerank_native_p0 import contracts as C
        with S.fixture("reranker", "jina_m0") as f:
            review, receipt, admitted = f.qualify()
            self.assertEqual([v["state"] for v in (review, receipt, admitted)], ["PASS"] * 3)
            report = S.E.read(f.out / "results/report.json")
            self.assertEqual(report["native_cola_rule"], "cola_image_choice")
            self.assertIn("cola_image_choice", review["recomputed"]["benchmarks"])
            native = report["benchmarks"]["cola_image_choice"]["macro_accuracy"]
            alternate = report["diagnostic_cola_alternate_axis"]["cola_text_choice"]["macro_accuracy"]
            self.assertNotEqual(native, alternate)
            cola_groups = [S.E.read(p) for p in (f.out / "raw").iterdir()
                           if "SYNTHETIC-cola:" in S.E.read(p)["pair_ids"][0]]
            self.assertEqual(len(cola_groups), 420)
            self.assertTrue(all(len(group["pair_ids"]) == 2 for group in cola_groups))
            values = [.3, .2, .4, .5]
            self.assertEqual(C.M.hit("cola_text_choice", values), 1)
            self.assertEqual(C.M.hit("cola_image_choice", values), 0)
            for task in ("cola_text_choice", "cola_image_choice"):
                self.assertEqual(C.M.hit(task, [.5] * 4), 0)


if __name__ == "__main__":
    unittest.main()
