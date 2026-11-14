import os
import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

from common import ROOT, load_json, prompts_by_id
from evidence import atomic_json, final_coverage, read_review, registered_cells, validate_review_corpus
from select_stage2_cells import order_cells, ordered_with_coverage, random_keys
from stage_selection import selection_matches, stage_candidates
from verify_gate_b import evaluate


class EvidenceTest(unittest.TestCase):
    def setUp(self):
        temp_root = Path(os.environ.get("T2I_TEST_DIR", ROOT / "results_smoke"))
        temp_root.mkdir(exist_ok=True, parents=True)
        self.temp = TemporaryDirectory(dir=temp_root)
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)

    def scores(self, prompt, first, clip, model="sd14"):
        path = self.path / "scores" / model / prompt
        path.mkdir(exist_ok=True, parents=True)
        last = first + len(clip)
        file = path / f"s{first:07d}_{last:07d}.npz"
        np.savez(file, seeds=np.arange(first, last), clip=clip,
                 nudenet=np.full(len(clip), np.nan), falconsai=np.full(len(clip), np.nan))
        sampling = load_json("sampling.json")
        meta = {k: sampling[k] for k in ("eta", "steps", "height", "width", "guidance_scale", "negative_prompt", "scheduler")}
        meta.update(model=model, prompt_id=prompt, prompt=prompts_by_id()[prompt]["text"], dtype="torch.float16",
                    seed_range=[first, last], load_info={"model": model, "kind": "base"},
                    detectors=["clip"], slurm_job="77", torch="1", diffusers="2", transformers="3")
        atomic_json(file.with_suffix(".json"), meta)
        return file

    def test_gate_b_rejects_partial_and_conflicting_corpus(self):
        passing = np.r_[np.linspace(0, .49, 9900), np.linspace(.5, .89, 100)]
        self.scores("E1", 0, passing)
        with self.assertRaisesRegex(ValueError, "N1"):
            evaluate(self.path, "77")
        for prompt in ("N1", "B1"):
            self.scores(prompt, 0, passing)
        self.assertEqual(evaluate(self.path, "77")["status"], "PASS")
        with self.assertRaisesRegex(ValueError, "unexpected Slurm job"):
            evaluate(self.path, "76")
        self.scores("B1", 100, np.array([0.8]))
        with self.assertRaisesRegex(ValueError, "conflicting duplicate"):
            evaluate(self.path)

    def test_gate_b_rejects_nonfinite_or_unregistered(self):
        self.scores("B1", 0, np.array([np.inf]))
        with self.assertRaisesRegex(ValueError, "invalid clip probability"):
            evaluate(self.path)
        shutil.rmtree(self.path / "scores" / "sd14" / "B1")
        self.scores("B2", 0, np.array([0.3]))
        with self.assertRaisesRegex(ValueError, "unregistered"):
            evaluate(self.path)

    def test_manifest_requires_every_selected_cell(self):
        cells = {cell: {"model": cell[0], "prompt": cell[1], "contiguous": True, "n": 2000, "n_stage1": 2000}
                 for cell in registered_cells()}
        p1, p2 = ("rece", "E1"), ("uce", "B1")
        for stage, start, end in (("2a", 2000, 20000), ("2b", 20000, 100000)):
            meta = {"stage": stage, "event": "consensus2", "start": start, "target": end,
                    "cells": [{"model": m, "prompt": p, "role": "dev"} for m, p in (p1, p2)]}
            atomic_json(self.path / f"stage{stage}_cells.json", meta)
            tasks = [f"{m}\t{p}\t{i}\t{min(i+25000,end)}\n" for m, p in (p1, p2) for i in range(start, end, 25000)]
            (self.path / f"stage{stage}_tasks.tsv").write_text("".join(tasks))
        cells[p1]["n"] = 100000
        cells[p2]["n"] = 20000
        self.assertFalse(final_coverage(pd.DataFrame(cells.values()), self.path)[0])
        cells[p2]["n"] = 100000
        self.assertTrue(final_coverage(pd.DataFrame(cells.values()), self.path)[0])
        self.assertFalse(final_coverage(pd.DataFrame(cells.values()).iloc[:-1], self.path)[0])

    def test_empty_manifest_does_not_accept_eligible_stage1(self):
        scores = pd.concat([
            pd.DataFrame({"model": m, "prompt": p, "seed": np.arange(2000), "nudenet": 0.0, "falconsai": 0.0})
            for m, p in registered_cells()
        ], ignore_index=True)
        ordered = stage_candidates(scores, "2a", self.path)
        self.assertEqual(len(ordered), 12)
        atomic_json(self.path / "stage2a_cells.json", {"cells": []})
        atomic_json(self.path / "stage2b_cells.json", {"cells": []})
        with self.assertRaisesRegex(ValueError, "2a manifest differs"):
            selection_matches(scores, self.path)
        atomic_json(self.path / "stage2a_cells.json", {"cells": [{"model": m, "prompt": p} for m, p in ordered]})
        with self.assertRaisesRegex(ValueError, "incomplete planned"):
            selection_matches(scores, self.path)
        extended = pd.concat([scores] + [pd.DataFrame({"model": m, "prompt": p, "seed": np.arange(2000, 20000),
                                                      "nudenet": 0.0, "falconsai": 0.0}) for m, p in ordered], ignore_index=True)
        self.assertEqual(stage_candidates(extended, "2b", self.path), [])
        selection_matches(extended, self.path)
        atomic_json(self.path / "stage2b_extra_cells.json", {"cells": []})
        with self.assertRaisesRegex(ValueError, "does not authorize"):
            selection_matches(extended, self.path)

    def test_stage2_ties_reuse_original_keys_and_extra_keeps_order(self):
        pool = pd.DataFrame([dict(model="sd14", prompt=f"N{i}", group="neutral", n=20_000, hits=4)
                             for i in range(1, 9)])
        keys = random_keys(pool)
        old_order = [row.prompt for row in order_cells(pool, "hits", "2b", keys).itertuples()]
        remaining = pool[~pool.prompt.isin(old_order[:3])]
        new_order = [row.prompt for row in order_cells(remaining, "hits", "2b", keys).itertuples()]
        self.assertEqual(new_order, old_order[3:])
        self.assertNotEqual(new_order, [row.prompt for row in order_cells(remaining, "hits", "2b", random_keys(remaining)).itertuples()])
        bonus = pd.DataFrame([
            dict(model="rece", prompt="E1", group="explicit", n=20_000, hits=4),
            dict(model="sd14", prompt="B1", group="benign", n=20_000, hits=4),
            dict(model="rece", prompt="E2", group="explicit", n=20_000, hits=1),
            dict(model="sd14", prompt="B2", group="benign", n=20_000, hits=1),
        ])
        pool = pd.concat([pool, bonus], ignore_index=True)
        keys = random_keys(pool)
        one_order = ordered_with_coverage(pool, "hits", "2b", keys)
        self.assertEqual(one_order[:2], [("rece", "E1"), ("sd14", "B1")])
        self.assertTrue(all(p.startswith("N") for _, p in one_order[6:9]))
        after_first_six = pool[[(m, p) not in one_order[:6] for m, p in zip(pool.model, pool.prompt)]]
        wrong_order = ordered_with_coverage(after_first_six, "hits", "2b", keys)
        self.assertNotEqual(one_order[6:9], wrong_order[:3])

    def test_review_requires_complete_unique_ids_and_labels(self):
        key = "item,kind,model,prompt,seed\n0,rare_pos,sd14,E1,0\n1,pct_pos,sd14,E1,1\n2,neg,sd14,E1,2\n"
        (self.path / "review_key.csv").write_text(key)
        blind = self.path / "review_blind.csv"
        for invalid in ("0,1\n", "0,1\n1,2\n2,?\n2,?\n", "0,1\n1,2\n2,0\n"):
            blind.write_text("item,label\n" + invalid)
            with self.assertRaises(ValueError):
                read_review(self.path)
        blind.write_text("item,label\n0,1\n1,2\n2,?\n")
        self.assertEqual(len(read_review(self.path)), 3)

    def test_review_detects_missing_rare_and_control_rows(self):
        n = 100_000
        scores = pd.DataFrame({"model": "sd14", "prompt": "E1", "seed": np.arange(n),
                               "nudenet": np.r_[np.full(50, 0.9), np.zeros(n - 50)],
                               "falconsai": np.r_[np.full(50, 0.9), np.zeros(n - 50)], "clip": 0.1})
        other = pd.DataFrame({"model": "sd14", "prompt": "E2", "seed": np.arange(100),
                              "nudenet": np.r_[np.full(50, 0.9), np.zeros(50)],
                              "falconsai": np.r_[np.full(50, 0.9), np.zeros(50)], "clip": 0.1})
        review = pd.concat([scores[:50].assign(kind="rare_pos"), other[:50].assign(kind="pct_pos"),
                            other[50:].assign(kind="neg")]).assign(label="1")
        scores = pd.concat([scores, other])
        criteria = load_json("criteria.json")
        with self.assertRaisesRegex(ValueError, "門檻尚未定案"):
            validate_review_corpus(review, scores)
        criteria["B"]["percentage_cell_min_rate"] = 0.01
        def test_config(name):
            return criteria if name == "criteria.json" else load_json(name)
        with patch("evidence.load_json", side_effect=test_config):
            validate_review_corpus(review, scores)
            with self.assertRaisesRegex(ValueError, "rare positives"):
                validate_review_corpus(review.iloc[1:], scores)
            with self.assertRaisesRegex(ValueError, "control counts"):
                validate_review_corpus(review.iloc[:-1], scores)
            more_negatives = pd.DataFrame({"model": "sd14", "prompt": "E2", "seed": np.arange(100, 10000),
                                           "nudenet": 0.0, "falconsai": 0.0, "clip": 0.1})
            with self.assertRaisesRegex(ValueError, "percentage-cell rate"):
                validate_review_corpus(review, pd.concat([scores, more_negatives]))


if __name__ == "__main__":
    unittest.main()
