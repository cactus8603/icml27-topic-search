import math
import unittest

from schedule import time_groups


STAGE2B_WORK = [(index, size) for index, size in enumerate((25_000, 25_000, 25_000, 5_000) * 6)]
REPLAY_RATE = 30_000 / 5_531
PRIOR_GPU_SECONDS = 17_154 + math.ceil(384_000 / REPLAY_RATE)


class TimeGroupsTests(unittest.TestCase):
    def test_preserve_feasible_proportional_limits(self):
        self.assertEqual(time_groups([(0, 25_000), (1, 5_000)], 6, 6_000), [
            {"images_per_task": 25_000, "minutes_per_task": 83, "indices": [0]},
            {"images_per_task": 5_000, "minutes_per_task": 16, "indices": [1]},
        ])

    def test_unused_minutes_cover_the_rounded_tail(self):
        remaining = 50 * 3_600 - PRIOR_GPU_SECONDS
        groups = time_groups(STAGE2B_WORK, REPLAY_RATE, remaining)
        self.assertEqual(groups, [
            {"images_per_task": 25_000, "minutes_per_task": 77,
             "indices": [index for index, rows in STAGE2B_WORK if rows == 25_000]},
            {"images_per_task": 5_000, "minutes_per_task": 16,
             "indices": [index for index, rows in STAGE2B_WORK if rows == 5_000]},
        ])
        allocated = sum(group["minutes_per_task"] * 60 * len(group["indices"]) for group in groups)
        self.assertEqual(allocated, 88_920)
        self.assertLessEqual(allocated + PRIOR_GPU_SECONDS, 50 * 3_600)

    def test_fail_when_minimum_allocations_exceed_remaining_budget(self):
        self.assertIsNone(time_groups(STAGE2B_WORK, REPLAY_RATE, 88_919))

    def test_fail_when_a_task_exceeds_its_original_limit(self):
        self.assertIsNone(time_groups([(0, 5_000)], 4.5, 50_000))

    def test_no_tasks_need_no_allocation(self):
        self.assertEqual(time_groups([], REPLAY_RATE, 0), [])


if __name__ == "__main__":
    unittest.main()
