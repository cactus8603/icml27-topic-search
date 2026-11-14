"""CPU tests for the frozen paired-likelihood assay and native vLLM sampler."""

import math
from types import SimpleNamespace
import unittest

from phase3_likelihood import load_cases, load_gold, read_delta, score_batch, summarize


class LikelihoodTest(unittest.TestCase):
    def test_paired_score_shared_bias_prior_and_missing(self):
        cases = load_cases()
        gold = load_gold(cases)
        deltas = {}
        for row in cases:
            key = row["id"]
            for candidate in (*row["candidates"], None):
                signal = 0 if candidate is None else (1 if candidate == gold[key] else -1)
                deltas[key, candidate, False] = 4 + signal
                deltas[key, candidate, True] = 4 - signal
        deltas[cases[0]["id"], None, True] = None
        result = summarize(cases, gold, deltas)
        all_cases = result["cohorts"]["all"]
        self.assertEqual(result["missing_prompt_scores"], 1)
        self.assertEqual(all_cases["missing_candidate_rows"], 2)
        self.assertEqual(all_cases["median_abs_semantic_log_odds"], 1)
        self.assertEqual(all_cases["median_abs_label_slot_log_odds"], 4)
        self.assertEqual(all_cases["candidate_evidence"]["pair_wins"], 23)
        self.assertEqual(all_cases["candidate_evidence"]["correct_decisions"], 46)
        self.assertEqual(all_cases["candidate_blind_prior"]["pair_ties"], 23)
        self.assertEqual(all_cases["candidate_blind_prior"]["abstentions"], 46)

    def test_read_delta_requires_token_identity_and_normalization(self):
        completion = SimpleNamespace(
            token_ids=[1], finish_reason="length",
            logprobs=[{1: SimpleNamespace(logprob=math.log(0.8)),
                       2: SimpleNamespace(logprob=math.log(0.2))}],
        )
        result = SimpleNamespace(outputs=[completion])
        ids = {"A": 1, "B": 2}
        self.assertAlmostEqual(read_delta(result, ids), math.log(4))
        completion.token_ids = [2]
        with self.assertRaisesRegex(ValueError, "greedy"):
            read_delta(result, ids)
        completion.token_ids = [1]
        completion.logprobs[0][2].logprob = math.log(0.1)
        with self.assertRaisesRegex(ValueError, "normalized"):
            read_delta(result, ids)
        del completion.logprobs[0][2]
        self.assertIsNone(read_delta(result, ids))

    def test_batch_rejects_wrong_prompt_identity(self):
        expected = {"a": {"prompt_token_ids": [1]}}
        engine = SimpleNamespace(generate=lambda *args, **kwargs: [
            SimpleNamespace(request_id="0", prompt_token_ids=[2], outputs=[]),
        ])
        with self.assertRaisesRegex(ValueError, "identities"):
            score_batch(engine, expected, None, {})

    def test_native_sampler_returns_both_normalized_allowlisted_scores(self):
        import torch
        from vllm.v1.sample.logits_processor.state import LogitsProcessors
        from vllm.v1.sample.metadata import SamplingMetadata
        from vllm.v1.sample.sampler import Sampler

        metadata = SamplingMetadata(
            temperature=None, all_greedy=True, all_random=False, top_p=None, top_k=None,
            generators={}, max_num_logprobs=2, no_penalties=True, prompt_token_ids=None,
            frequency_penalties=torch.zeros(2), presence_penalties=torch.zeros(2),
            repetition_penalties=torch.ones(2), output_token_ids=[[], []],
            allowed_token_ids_mask=torch.tensor([[True, False, False, True]] * 2),
            bad_words_token_ids={}, logitsprocs=LogitsProcessors(),
            logprob_token_ids={0: [1, 2], 1: [1, 2]},
        )
        sampler = Sampler(logprobs_mode="processed_logprobs")
        # Disable CUDA-pinned host allocation in this CPU-only sampler test.
        sampler.pin_memory = False
        output = sampler(torch.tensor([[100.0, 1.0, -1.0, 202.0],
                                       [100.0, -1.0, 1.0, 202.0]]), metadata)
        self.assertEqual(output.sampled_token_ids.tolist(), [[1], [2]])
        tensors = output.logprobs_tensors
        for i, difference in enumerate((2, -2)):
            scores = dict(zip(tensors.logprob_token_ids[i].tolist(),
                              tensors.logprobs[i].tolist(), strict=True))
            self.assertEqual(set(scores), {1, 2})
            self.assertAlmostEqual(sum(math.exp(p) for p in scores.values()), 1, places=6)
            self.assertAlmostEqual(scores[1] - scores[2], difference, places=6)


if __name__ == "__main__":
    unittest.main()
