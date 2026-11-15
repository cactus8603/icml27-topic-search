import math
import unittest

from schedule import completed_rates, phase_time_budget, time_groups


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

    def test_retry_budget_charges_time_against_the_current_phase(self):
        available = phase_time_budget(10_000, 30_000, 2_200, 0)
        self.assertEqual(available, 4_466)
        self.assertIsNone(time_groups([(1, 25_000)], 5.3, available))

    def test_blind_reserve_and_strict_total_cap_can_limit_a_phase(self):
        available = phase_time_budget(6_000, 30_000, 0, 420)
        self.assertEqual(available, 5_579)
        self.assertEqual(phase_time_budget(6_000, 30_000, 0, 0), 5_999)
        self.assertIsNone(time_groups([(0, 25_000), (1, 5_000)], 5.3, available))

    def test_matching_rates_can_fit_a_stage_rejected_by_pooled_rate(self):
        work = [(0, 25_000), (1, 5_000)]
        self.assertIsNone(time_groups(work, 5.14, 5_640))
        groups = time_groups(work, 5.14, 5_640, {0: 5.5, 1: 5.5})
        self.assertEqual([group["minutes_per_task"] for group in groups], [76, 16])

    def test_matching_rates_include_failed_allocations_for_the_same_task(self):
        tasks = [("sd14", "all", 0, 2000), ("uce", "all", 0, 2000)]
        attempts = [("1", 0, "1", "FAILED", "1:0", 100),
                    ("2", 0, "2", "COMPLETED", "0:0", 5500),
                    ("2", 1, "3", "COMPLETED", "0:0", 5000)]
        seconds, rates = completed_rates(tasks, attempts, 14)
        self.assertEqual(seconds, 10_600)
        self.assertEqual(rates, {("sd14", "all"): 5.0, ("uce", "all"): 5.6})
        with self.assertRaisesRegex(ValueError, "successful GPU allocation"):
            completed_rates(tasks, attempts[:2], 14)


if __name__ == "__main__":
    unittest.main()
