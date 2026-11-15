from types import SimpleNamespace
import unittest

import generate


class GenerationLayoutTest(unittest.TestCase):
    def setUp(self):
        questions = [SimpleNamespace(id=f"spanish_{i}", text=f"synthetic_{i}", system_prompt=generate.SYSTEM)
                     for i in range(generate.QUESTIONS_N)]
        self.ids, self.chats = generate.conversation_plan(questions)
        self.prompts = [chat[-1]["content"] for chat in self.chats]

    def test_exact_grid_and_distinct_arm_response_ids(self):
        self.assertEqual(self.ids[:10], [("spanish_0", i) for i in range(1, 11)])
        self.assertEqual(self.ids[-1], ("spanish_19", 10))
        self.assertEqual(len(self.ids), 200)
        rows = [generate.output_rows(arm, self.ids, self.prompts, self.prompts, [""] * 200)
                for arm in ("base", "none", "ip")]
        self.assertEqual(len({row["id"] for arm in rows for row in arm}), 600)
        self.assertEqual(rows[0][0]["answer"], "")
        self.assertEqual(rows[0][0]["prompt_sha256"], rows[2][0]["prompt_sha256"])

    def test_missing_or_reordered_author_output_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "incomplete"):
            generate.output_rows("base", self.ids, self.prompts, self.prompts, ["x"] * 199)
        with self.assertRaisesRegex(ValueError, "reordered"):
            generate.output_rows("base", self.ids, self.prompts[::-1], self.prompts, ["x"] * 200)


if __name__ == "__main__":
    unittest.main()
