import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from count_clock import FORMAL, PICKS, UPDATES_PER_EPOCH, instrument, summarize


class CountClockTests(unittest.TestCase):
    def test_paired_count_rule(self):
        small = dict.fromkeys(itertools.combinations(PICKS, 2), (10.0, "interior"))
        large = dict.fromkeys(small, (20.0, "interior"))
        self.assertEqual(summarize(small, large)["state"], "supports_count_separation")
        self.assertEqual(summarize(large, small)["state"], "opposite_count_order")
        self.assertEqual(summarize(small, small)["state"], "below_reference_magnitude")
        for pair in list(large)[:6]:
            large[pair] = (1.0, "first")
        result = summarize(small, large)
        self.assertEqual(result["common_valid_pairs"], 9)
        self.assertEqual(result["state"], "UNTESTABLE")

    def test_original_trainer_runs_equal_curricula_on_cpu(self):
        repo = Path(__file__).resolve().parents[3]
        source_path = "killtests/biased-gen-cfg/scripts"
        frozen = subprocess.check_output(["git", "show", f"{FORMAL}:{source_path}/train_hier_cond_w2.py"], cwd=repo, text=True)
        data = Path(os.environ["COUNT_CLOCK_TEST_DATA"])
        with TemporaryDirectory(prefix="clock-count-unit-", dir=os.environ["TMPDIR"]) as temp:
            root = Path(temp)
            for name in ("hier_cond.py", "hier_run.py", "schedule.py"):
                content = subprocess.check_output(["git", "show", f"{FORMAL}:{source_path}/{name}"], cwd=repo)
                (root / name).write_bytes(content)
            trainer = root / "trainer.py"
            trainer.write_text(instrument(frozen, (1, 2)))
            for count in (12, 24):
                output = root / f"n{count}"
                subprocess.run([sys.executable, "-IB", str(trainer), "--data", str(data), "--out", str(output),
                                "--n", "500", "--support-size", str(count), "--device", "cpu", "--batch", "4",
                                "--class-only", "0", "--pdrop", "1", "--layers", "1", "--hidden", "32"],
                               check=True, stdout=subprocess.DEVNULL)
                meta = json.loads((output / "meta.json").read_text())
                self.assertEqual((meta["n_c"], meta["steps_per_epoch"], meta["ckpt_epochs"]),
                                 (count, UPDATES_PER_EPOCH, [1, 2]))
                self.assertTrue((output / "completion.json").is_file())


if __name__ == "__main__":
    unittest.main()
