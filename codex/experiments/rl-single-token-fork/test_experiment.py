"""CPU fixtures only. No real model weights or authentic generation."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import experiment as exp


class ScriptedBatch:
    def __init__(self, steps):
        self.steps = iter(steps)
        self.advances = []

    def predict(self):
        return next(self.steps)

    def advance(self, tokens, active):
        self.advances.append((tokens, active))


class DiagnosticFixtures(unittest.TestCase):
    def test_first_nonstop_disagreement_and_shared_base_path(self):
        eos, extra = exp.STOP_IDS
        batch = ScriptedBatch([([1, eos, 2], [eos, 6, 2]),
                               ([3, 4, 5], [4, 1, 5]),
                               ([6, 2, 7], [9, 3, extra])])
        a, b, c = exp.scan(batch, 3, 3)
        self.assertEqual(a, exp.Search([1], (3, 4), "discordance"))
        self.assertEqual(b, exp.Search([], None, "base_stop"))
        self.assertEqual(c, exp.Search([2, 5, 7], None, "scan_limit"))
        self.assertEqual(batch.advances, [([1, eos, 2], [True, False, True]), ([3, 4, 5], [False, False, True])])

    def test_equal_continuation_caps_and_real_stops(self):
        eos, extra = exp.STOP_IDS
        batch = ScriptedBatch([([3, eos, 7],), ([extra, 3, 9],)])
        a, b, c = exp.continue_base(batch, [2, 2, 2])
        self.assertEqual(a, exp.Continuation(2, [3, extra], extra))
        self.assertEqual(b, exp.Continuation(2, [eos], eos))
        self.assertEqual(c, exp.Continuation(2, [7, 9]))
        self.assertEqual(batch.advances, [([3, eos, 7], [True, False, True])])

    def test_fork_batches_preserve_budgets_and_context(self):
        prompts = [[1, 2], [3, 1], [4, 5, 1]]
        search = [exp.Search([3, 4], (5, 6), "discordance"), exp.Search([], None, "base_stop"),
                  exp.Search([], (1, 2), "discordance")]
        plans = list(exp.fork_batches(prompts, search, 5, 8, 2))
        self.assertEqual(plans, [[(0, [1, 2, 3, 4], (5, 6), 2)], [(2, [4, 5, 1], (1, 2), 4)]])
        self.assertEqual(len(list(exp.fork_batches(prompts, search, 5, 9, 2))), 1)

    def test_panel_excludes_normalized_duplicates_and_prior_selection(self):
        math = [{"problem": q, "solution": a} for q, a in (
            ("Ｐrior", r"\boxed{1}"), ("P r I o R", r"\boxed{1}"), ("Ａ", r"\boxed{2}"),
            ("a", r"\boxed{7}"), ("B", r"\boxed{3}"), ("C", r"\boxed{4}"), ("D", r"\boxed{5}"),
        )]
        old, _ = exp.previous.select([r["problem"] for r in math], {"prior"}, 1)
        selected, available = exp.select_rows(math, {"prior"}, 3, 1)
        self.assertEqual(available, 3)
        self.assertEqual(len({exp.previous.question_key(q) for _, q, _ in selected}), 3)
        self.assertTrue(all(int(key.split(":")[1]) not in {0, 1, 3, *old} for key, _, _ in selected))
        self.assertEqual(exp.select_rows(math, {"prior"}, 3, 1)[0], selected)

    def test_summary_keeps_nonforks_in_primary_denominator(self):
        records = [{"search": "discordance", "correct": [0, 1], "stop_tokens": [None, 1]},
                   {"search": "base_stop"}]
        summary = exp.summarize(records)
        self.assertEqual(summary["all_panel"]["rl_minus_base_pp"], 50)
        self.assertEqual(summary["among_forked"]["rl_minus_base_pp"], 100)
        self.assertEqual(summary["forked_truncated_base_rl"], [1, 0])
        self.assertIsNone(exp.summarize(records[1:])["among_forked"])

    def test_transformers_cache_keeps_duplicate_branches_matched(self):
        import torch
        from transformers import Qwen2Config, Qwen2ForCausalLM

        torch.manual_seed(47)
        config = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                             num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=24)
        model = Qwen2ForCausalLM(config).eval()
        repeated = [[1, 2], [1, 2], [2, 3, 7, 8]]
        with torch.inference_mode():
            batch = exp.GreedyBatch([model], repeated, torch, 0)
            a, b, c = exp.continue_base(batch, [4, 4, 3])
            self.assertEqual(a, b)
            self.assertEqual(len(c.tokens), 3)
            isolated = exp.continue_base(exp.GreedyBatch([model], [repeated[0]], torch, 0), [4])[0]
            self.assertEqual(a, isolated)
            identical = exp.scan(exp.GreedyBatch([model, model], repeated, torch, 0), 3, 4)
            self.assertTrue(all(item.choices is None and item.stop == "scan_limit" for item in identical))


if __name__ == "__main__":
    unittest.main()
