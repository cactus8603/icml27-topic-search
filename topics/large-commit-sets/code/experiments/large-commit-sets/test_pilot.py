import json
import math
import unittest

from judge import gate, held_out_concordance, prompt_fold, summarize, summarize_rate
from pilot import (
    CONFIG_PATH,
    LLaDAGreedy,
    Proposal,
    candidates_at_state,
    choose_pool,
    exact_additive,
    extract_answer,
    joint_holdouts,
)


class SyntheticProbe:
    def __init__(self):
        self.calls = 0

    def probe(self, x, revealed, targets, proposals):
        assert set(revealed).isdisjoint(targets)
        assert all(x[i] is None for i in (*revealed, *targets))
        self.calls += 1
        return [
            proposals[i].base_margin + sum(j / 10 for j in revealed) - (len(revealed) > 1)
            for i in targets
        ]


class TestPilot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(CONFIG_PATH.read_text())

    def test_numeric_answer(self):
        self.assertEqual(extract_answer("step 1\n#### $1,240.00\n"), "1240")
        self.assertEqual(extract_answer("#### 2\n\n#### −0.50"), "-0.5")
        self.assertEqual(extract_answer("#### 001."), "1")
        self.assertIsNone(extract_answer("#### 12 units\n final: 12"))
        self.assertIsNone(extract_answer("#### 99/3"))

    def test_candidate_pool_matches_confidence_and_fixed_quota(self):
        ranked = [Proposal(i, 10 + i, -i / 100, 1) for i in range(16)]
        pools, reason = choose_pool(ranked, self.config["decoder"], {-1})
        self.assertIsNone(reason)
        self.assertEqual(len(set(pools)), 3)
        self.assertEqual(pools[0], tuple(range(8)))
        self.assertEqual(pools[1], (0, 1, 2, 3, 4, 5, 6, 8))
        self.assertTrue(all(len(x) == 8 for x in pools))
        self.assertEqual(
            choose_pool(ranked, self.config["decoder"], {10})[1],
            "baseline_contains_special_token",
        )
        self.assertEqual(
            choose_pool(ranked[:8], self.config["decoder"], {-1})[1],
            "too_few_matched_swaps",
        )

    def test_higher_order_probe_never_sees_its_target(self):
        members = (1, 2, 3)
        self.assertEqual(joint_holdouts(members), [((2, 3), 1), ((1, 3), 2), ((1, 2), 3)])
        self.assertEqual(exact_additive({1: 1}, {2: {1: 2}, 3: {1: 3}}, 1, members), 4)
        fake = SyntheticProbe()
        ranked = [Proposal(i, 10 + i, -i / 100, 1) for i in range(5)]
        rows, telemetry = candidates_at_state(fake, [None] * 5, 0, ranked, [members, (1, 2, 4)])
        self.assertEqual(fake.calls, 10)
        self.assertEqual(telemetry["singleton_forwards"], 4)
        self.assertEqual(telemetry["joint_forwards"], 6)
        for row in rows:
            for direct, additive in zip(row["joint_margins"], row["additive_margins"], strict=True):
                self.assertTrue(math.isclose(additive - direct, 1))

    def test_cpu_rollout_has_matched_forward_budget(self):
        try:
            import torch
        except ImportError:
            self.skipTest("Use the resident t2irare Python for the CPU tensor rollout.")

        class Stub(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.param = torch.nn.Parameter(torch.zeros(1))
                self.config = type("Config", (), {"max_sequence_length": 8})()

            def forward(self, input_ids, use_cache):
                values = torch.zeros(1, input_ids.shape[1], 8)
                values[:, :, 3] = 5
                values[0, 1, 3] += input_ids[0, 2].ne(0)
                return type("Output", (), {"logits": values})()

        class Tokenizer:
            all_special_ids = []

            def __len__(self):
                return 8

            def apply_chat_template(self, *_args, **_kwargs):
                return [7]

            def decode(self, tokens, **_kwargs):
                return repr(tokens)

        settings = {
            "block_size": 6, "quota": 2, "generation_length": 6, "anchors": [0],
            "candidate_count": 3, "swap_boundary": 2, "max_swap_logprob_gap": 1,
            "max_swap_distance": 4,
        }
        decoder = LLaDAGreedy(
            torch, Stub(), Tokenizer(), settings, {"mask_token": 0, "stop_tokens": [6]}
        )
        x, prompt_length = decoder.encode("test")
        with self.assertRaisesRegex(RuntimeError, "leave every target masked"):
            decoder.probe(x, (1,), (1,), {1: Proposal(1, 3, -0.1, 1)})
        with torch.inference_mode():
            baseline, length, [anchor] = decoder.run_question("test")
        self.assertEqual(length, 6)
        self.assertEqual(anchor["candidates"][0]["output"], baseline)
        self.assertEqual({c["continuation_forwards"] for c in anchor["candidates"]}, {2})
        self.assertEqual(anchor["singleton_forwards"], 3)
        self.assertEqual(anchor["joint_forwards"], 6)
        self.assertEqual(decoder.forward_calls, 3 + 3 + 6 + 4)

    def test_judgments_keep_candidate_oracle_and_method_gain_separate(self):
        baseline = dict(
            correct=False, additive_margins=[3], joint_margins=[1],
            additive_score=2, joint_score=1,
            positions=[0, 1], confidence=[-0.1, -0.2],
        )
        successful = dict(
            correct=True, additive_score=1, joint_score=2,
            positions=[0, 2], confidence=[-0.1, -0.3],
        )
        wrong = dict(
            correct=False, additive_score=0, joint_score=0,
            positions=[1, 2], confidence=[-0.2, -0.3],
        )
        row = dict(
            doc_id=1,
            baseline_correct=False,
            anchors=[
                dict(step=2, ineligible_reason=None, candidates=[baseline, successful, wrong]),
                dict(step=10, ineligible_reason="too_few_matched_swaps"),
            ],
        )
        config = {**self.config, "judgment": {**self.config["judgment"], "bootstrap_resamples": 50}}
        verdict = summarize([row], config)
        self.assertEqual(verdict["headroom"]["estimate"], 1)
        self.assertEqual(verdict["phenomenon"]["estimate"], 1)
        self.assertEqual(verdict["descriptive_unadjusted_concordance"]["joint"], 1)
        self.assertEqual(verdict["descriptive_unadjusted_concordance"]["additive"], 0.5)
        self.assertIsNone(verdict["signal"]["estimate"])
        self.assertEqual(verdict["signal"]["status"], "inconclusive")
        self.assertEqual(verdict["descriptive_first_eligible_anchor"]["d_vs_c_paired_accuracy"]["estimate"], 1)
        self.assertEqual(gate(summarize_rate(0, 240, 0.95), 0.04)["status"], "below_floor")

    def test_cross_fitted_signal_adds_only_joint_score(self):
        settings = self.config["judgment"]
        # Each held-out fold has balanced labels. The two control columns
        # are either constant or uninformative. The final column is useful.
        counts = [0] * settings["signal_folds"]
        pairs_by_prompt = {}
        for doc_id in range(40):
            fold = prompt_fold(doc_id, settings)
            label = counts[fold] % 2
            counts[fold] += 1
            pairs_by_prompt[doc_id] = [((0, 1, 2 * label - 1), label)]
        concordance, by_fold = held_out_concordance(pairs_by_prompt, settings)
        self.assertEqual(by_fold, counts)
        self.assertEqual(len(concordance), 40)
        self.assertTrue(all(full == 1 for _, full in concordance))
        self.assertTrue(any(control != 1 for control, _ in concordance))
        # Invert all the training folds independently of a held-out one
        # to detect evaluation on labels from its own fold.
        flipped = {
            i: [(x, y if prompt_fold(i, settings) == 0 else 1 - y) for x, y in pairs]
            for i, pairs in pairs_by_prompt.items()
        }
        concordance, _ = held_out_concordance(flipped, settings)
        self.assertTrue(all(full == 0 for _, full in concordance[:by_fold[0]]))


if __name__ == "__main__":
    unittest.main()
