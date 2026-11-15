"""CPU checks for text isolation, language uncertainty and paired intervals."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure import Evidence, measure, synthetic_checks
from panel import (ARMS, GRID, JUDGE, PILOT_RULE, classify_blind, digest, encode, extract_answer,
                   resampled_interval, restore_grid, summaries)


def fake_identify(text, *, k):
    if "verano" in text:
        return ("__label__es", "__label__pt"), (0.94, 0.03)
    if "ambiguous" in text:
        return ("__label__es", "__label__en"), (0.65, 0.29)
    return ("__label__en", "__label__es"), (0.96, 0.03)


class DiagnosticTests(unittest.TestCase):
    def test_synthetic_controls_and_unclassified_words(self):
        self.assertEqual(len(synthetic_checks(fake_identify)), 8)
        row = measure("Ambiguous words could be interpreted in many different languages today.", fake_identify)
        self.assertEqual(row.category, "unknown")
        self.assertEqual(row.bounds()["any_spanish"], (0.0, 1.0))

    def test_bounds_allow_unknown_and_strict_majority(self):
        self.assertEqual(Evidence(3, 5, 2, "mixed_detected").bounds()["majority_spanish"], (0.0, 0.0))
        self.assertEqual(Evidence(4, 4, 2, "mixed_detected").bounds()["majority_spanish"], (0.0, 1.0))
        self.assertEqual(Evidence(6, 0, 4, "spanish_detected").bounds()["spanish_word_fraction"], (0.6, 1.0))

    def test_rubric_answer_frame_does_not_extract_question_or_truncate_at_inner_delimiter(self):
        frames = {"spanish_0": ("Q: question\nA: ", "\nEND"), "spanish_1": ("Q: second\nA: ", "\nEND")}
        self.assertEqual(extract_answer("Q: question\nA: foo\nEND bar\nEND", frames), ("spanish_0", "foo\nEND bar"))
        with self.assertRaisesRegex(ValueError, "does not match"):
            extract_answer("foo\nEND", frames)

    def test_blind_then_restore_exact_grid_with_prompt_binding(self):
        salt = "0" * 32
        frames = {q: (f"Q: {q} A: ", " END") for _, q, _ in GRID}
        records, requests = {}, []
        for arm, q, n in sorted(GRID):
            opaque = digest(encode([salt, arm, q, n]))[:12]
            left, right = frames[q]
            prompt = left + "test" + right
            requests.append(encode({"id": opaque, "request": JUDGE | {"messages": [{"role": "user", "content": prompt}]}}))
            records[opaque] = {"arm": arm, "question_id": q, "sample": n, "judge_prompt_sha256": digest(prompt.encode())}
        mapping = {"salt": salt, "pilot_sampling_rule": PILOT_RULE, "mapping": records}
        blind = classify_blind(b"".join(requests), frames, fake_identify)
        self.assertEqual(restore_grid(encode(mapping), blind).keys(), GRID)
        records[next(iter(records))]["judge_prompt_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "unmatched"):
            restore_grid(encode(mapping), blind)

    def test_question_bootstrap_keeps_bounds_and_contrast_extremes(self):
        expected = {"observed_bounds": [0.25, 0.75], "question_bootstrap_envelope_95": [0.0, 1.0]}
        draws = np.array([[0, 0], [0, 1], [1, 1]] * 20)
        self.assertEqual(resampled_interval(np.array([[0.0, 1.0], [0.5, 0.5]]), draws), expected)
        rows = [Evidence(0, 2, 0, "other_detected"), Evidence(2, 0, 0, "spanish_detected"),
                Evidence(0, 0, 2, "unknown")]
        cells = {(arm, q, n): rows[ARMS.index(arm)] for arm, q, n in GRID}
        out = summaries(cells, pilot=True)
        self.assertEqual(out["contrasts"]["none_minus_ip"]["any_spanish"]["observed_bounds"], [0.0, 1.0])
        self.assertEqual(out["contrasts"]["none_minus_base"]["any_spanish"]["observed_bounds"], [1.0, 1.0])
        self.assertEqual(out["arms"]["none"]["offline_response_count_bounds"]["majority_spanish"], [20, 20])


if __name__ == "__main__":
    unittest.main()
