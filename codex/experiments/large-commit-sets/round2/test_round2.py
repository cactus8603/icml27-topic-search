import itertools
import json
import unittest

from collect import Decoder
from design import CONFIG_PATH, doubleton_rows, joint_rows, split_ids
from score import evaluate, verify_row
from design import pilot

POLICY = json.loads(pilot.CONFIG_PATH.read_text())["decoder"]
CONFIG = json.loads(CONFIG_PATH.read_text())
POOLS = [tuple(range(8)), (*range(7), 8), (*range(7), 9)]


class TestRound2(unittest.TestCase):
    def test_all_swap_patterns_have_scored_equal_budget(self):
        base = tuple(range(8))
        options = [
            tuple(sorted((set(base) - {drop}) | {add}))
            for drop, add in itertools.product(range(4, 8), range(8, 12))
        ]
        for one, two in itertools.combinations(options, 2):
            pools = [base, one, two]
            joint = joint_rows(pools)
            double = doubleton_rows(pools, len(joint))
            self.assertEqual(len(double), len(joint))
            for rows, size in ((double, 2), (joint, 7)):
                for reveal, targets in rows.items():
                    self.assertEqual(len(reveal), size)
                    self.assertTrue(set(reveal).isdisjoint(targets))
                    self.assertTrue(targets)
                    self.assertTrue(all(
                        any(set((*reveal, i)) <= set(pool) for pool in pools) for i in targets
                    ))

    def test_disjoint_split_rejects_normalized_duplicates(self):
        test = [{"question": f"Test sample {i}"} for i in range(30)]
        train = [{"question": f"Train sample {i}"} for i in range(100)]
        train[15] = {"question": "TEST \t sample 17  "}
        train[16] = {"question": " Train sample 0  "}
        config = {**CONFIG, "train_sample_size": 80, "test_sample_size": 20}
        round1 = {"main": {"sample_seed": "original", "sample_size": 5}}
        from design import stable_order
        excluded = set(stable_order(range(len(test)), "original")[:5])
        split = split_ids(train, test, config, round1)
        self.assertFalse(excluded.intersection(split["test"]))
        self.assertFalse(set(split["train"]) & set(split["smoke"]))
        self.assertFalse(set(split["train"] + split["smoke"]) & {0, 1, 15, 16})

    @staticmethod
    def make_row(index, winner):
        candidates = []
        for j, pool in enumerate(POOLS):
            additive = [3.0] * 8
            joint = [5.0 if j == winner else 1.0] * 8
            double = [2.0] * 8
            candidates.append({
                "positions": list(pool),
                "confidence": [-0.5] * 8,
                "additive_margins": additive,
                "doubleton_margins": double,
                "joint_margins": joint,
                "additive_score": pilot.set_score(additive),
                "doubleton_score": pilot.set_score(double),
                "joint_score": pilot.set_score(joint),
                "fixed_alt_residual": [1.5] * 8,
                "dynamic_alt_switched": [True] * 8,
                "doubleton_coverage": [1] * 8,
                "continuation_forwards": 29,
                "continuation_seconds": 1.0,
                "correct": j == winner,
            })
        budget = len(joint_rows(POOLS))
        rows = doubleton_rows(POOLS, budget)
        return {
            "type": "prompt",
            "doc_id": index,
            "forwards": 32 + 10 + 2 * budget + 58,
            "baseline_correct": winner == 0,
            "baseline_seconds": 1.1,
            "eligibility": {"2": None},
            "anchor": {
                "step": 2,
                "joint_forwards": budget,
                "doubleton_forwards": budget,
                "doubleton_reveals": [list(pair) for pair in rows],
                "singleton_forwards": 10,
                "candidates": candidates,
                "prefix_seconds": 0.1,
                "singleton_seconds": 0.12,
                "joint_seconds": 0.14,
                "doubleton_seconds": 0.15,
                "joint_plan_seconds": 0.001,
                "doubleton_plan_seconds": 0.003,
            },
        }

    def test_scorer_trains_only_on_train_and_checks_equal_budget(self):
        train = [self.make_row(i, i % 3) for i in range(60)]
        test = [self.make_row(100 + i, i % 3) for i in range(30)]
        for row in train + test:
            verify_row(row, POLICY)
        settings = {**CONFIG, "bootstrap_resamples": 100, "minimum_test_discordant": 5}
        verdict = evaluate(train, test, settings)
        self.assertEqual(verdict["descriptive"]["accuracy_by_arm"]["joint"], 1)
        self.assertEqual(verdict["descriptive"]["accuracy_by_arm"]["doubleton"], 1 / 3)
        self.assertEqual(verdict["descriptive"]["accuracy_by_arm"]["pool_oracle"], 1)
        self.assertEqual(verdict["signal"]["held_out_concordance"]["joint"], 1)
        nfes = verdict["descriptive"]["forward_count_mean"]
        self.assertEqual(nfes["joint"], nfes["doubleton"])
        self.assertLess(nfes["confidence"], nfes["shared"])
        self.assertLess(nfes["round1_joint"], nfes["joint"])
        for row in test:
            for candidate in row["anchor"]["candidates"]:
                candidate["correct"] = not candidate["correct"]
            row["baseline_correct"] = not row["baseline_correct"]
        permuted = evaluate(train, test, settings)
        self.assertEqual(permuted["descriptive"]["fit_coefficients"], verdict["descriptive"]["fit_coefficients"])
        self.assertEqual(permuted["descriptive"]["accuracy_by_arm"]["joint"], 0)
        test[0]["anchor"]["doubleton_forwards"] -= 1
        with self.assertRaisesRegex(ValueError, "arm probe budgets"):
            verify_row(test[0], POLICY)

    def test_cpu_collector_exercises_probes_and_matched_branches(self):
        import torch
        from types import SimpleNamespace

        class Stub(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.param = torch.nn.Parameter(torch.zeros(1))
                self.config = SimpleNamespace(max_sequence_length=96)

            def forward(self, input_ids, use_cache):
                result = torch.zeros(1, input_ids.shape[-1], 8)
                result[:, :, 3] = 6.0
                known = (input_ids == 3).sum()
                result[:, :, 4] = (known % 4) / 2.0
                return SimpleNamespace(logits=result)

        class Tokenizer:
            all_special_ids = []

            def __len__(self):
                return 8

            def apply_chat_template(self, *_args, **_kwargs):
                return [7]

            def decode(self, tokens, **_kwargs):
                return repr(tokens)

        # Exercise actual tensor indexing and LOO leakage guard without CUDA.
        fake_torch = SimpleNamespace(
            tensor=torch.tensor, long=torch.long, where=torch.where, arange=torch.arange,
            inf=torch.inf, cuda=SimpleNamespace(synchronize=lambda: None),
        )
        settings = {**POLICY, "generation_length": 32, "block_size": 32, "anchors": [1]}
        decoder = Decoder(fake_torch, Stub(), Tokenizer(), settings, {"mask_token": 0, "stop_tokens": [6]})
        with torch.inference_mode():
            _, _, anchor, _, _ = decoder.run_sample("example", 7)
        self.assertFalse(anchor["doubleton_first"])
        self.assertEqual(anchor["joint_forwards"], anchor["doubleton_forwards"])
        self.assertEqual({row["continuation_forwards"] for row in anchor["candidates"]}, {2})
        self.assertTrue(all(
            count > 0 for row in anchor["candidates"] for count in row["doubleton_coverage"]
        ))


if __name__ == "__main__":
    unittest.main()
