from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate as oat


class OatEvaluationTest(unittest.TestCase):
    def test_existing_gate_bounds_and_mandatory_panels(self):
        self.assertEqual(oat.score_results({"math": 381 / 500, "minerva": 70 / 272})["state"], "PASS")
        self.assertEqual(oat.score_results({"math": 382 / 500, "minerva": 70 / 272})["state"], "UNTESTABLE")
        self.assertEqual(oat.score_results({"math": 371 / 500, "minerva": 64 / 272})["state"], "UNTESTABLE")
        with self.assertRaisesRegex(ValueError, "panel order"):
            oat.score_results({"minerva": 70 / 272})

    def test_smoke_preserves_author_model_settings_without_allowing_extra_stop(self):
        panels = {name: {"problem": [f"Synthetic {name}"], "answer": ["42"]} for name in oat.PANELS}
        sampled = []
        answer = SimpleNamespace(text=r"\boxed{42}", token_ids=[oat.EOS], finish_reason="stop", stop_reason=None)

        def engine(_model, **_kwargs):
            def generate(prompts, params):
                sampled.append(params.max_tokens)
                return [SimpleNamespace(prompt=text, outputs=[answer]) for text in prompts]

            return SimpleNamespace(generate=generate, get_tokenizer=lambda: SimpleNamespace(eos_token_id=oat.EOS))

        def sample(**kwargs):
            return SimpleNamespace(**kwargs, stop=[], stop_token_ids=[], ignore_eos=False)

        def grade(text, gold, fast):
            self.assertEqual((text, gold, fast), (r"\boxed{42}", "42", False))
            return {"formatted": True}, 1

        got, covered, _ = oat.run_author(oat.MODEL, panels, engine, sample, grade, max_tokens=oat.SMOKE_LIMIT)
        self.assertEqual(got, {"math": 1, "minerva": 1})
        self.assertEqual(covered, {"math": {"stop": 1}, "minerva": {"stop": 1}})
        self.assertEqual(sampled, [16, 16])
        answer.stop_reason = 151645
        with self.assertRaisesRegex(ValueError, "additional stop"):
            oat.run_author(oat.MODEL, panels, engine, sample, grade, max_tokens=oat.SMOKE_LIMIT)

    def test_grade_worker_smoke_opens_no_arrow_and_private_mode_uses_passed_fixture(self):
        fixtures = {task: {"problem": [f"Synthetic {task}"], "answer": ["42"]} for task in oat.PANELS}
        for mode in ("grade-smoke", "grade-private"):
            stream = StringIO()
            with (patch.object(oat, "source_check") as sources, patch.object(oat, "load_arrow", return_value=fixtures) as arrow,
                  patch.object(sys, "stdin", StringIO('["\\\\boxed{42}", "42", false]\n')), redirect_stdout(stream)):
                oat.grader_worker(mode)
            header, score = [json.loads(line) for line in stream.getvalue().splitlines()]
            self.assertEqual(score, [{"formatted": True}, 1.0])
            if mode == "grade-smoke":
                sources.assert_called_once_with(check_panels=False)
                arrow.assert_not_called()
                self.assertEqual(header["panels"], {})
            else:
                sources.assert_called_once_with(check_panels=True)
                arrow.assert_called_once_with(True)
                self.assertEqual(header["panels"], fixtures)


if __name__ == "__main__":
    unittest.main()
