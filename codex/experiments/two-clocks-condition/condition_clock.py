"""Measure condition information with a shared pool and fully matched exposures."""

import argparse
from collections import Counter
from itertools import combinations
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from panel import (BATCH, EVAL_N, EVAL_SEED, GRID, HORIZON, Q, TRAIN_N, TRAIN_SEED, UPDATES_PER_EPOCH,
                   build_data, hash_file, schedule, verify_schedule)


PICKS = tuple(range(6))
POLICIES = ("true", "shuffled")
BRANCHES = ("cond", "uncond")
BOOTSTRAP_SEED = 2026111733
BOOTSTRAP_DRAWS = 2000
SOURCE_COMMIT = "9e92804a7bfded3c5cafd4fccd4315df2034eca5"
SHARED = Path("/work/ChihChungHsu/cactus8603/icml")


def replace_pinned(text, replacements, label):
    for old, new in replacements.items():
        if text.count(old) != 1:
            raise ValueError(f"Pinned {label} insertion point differs: {old}")
        text = text.replace(old, new)
    compile(text, f"<{label}>", "exec")
    return text


def instrument_train(text, grid):
    return replace_pinned(text, {
        "args = ap.parse_args()": (
            'ap.add_argument("--label-policy", choices=["true", "shuffled"], required=True)\n'
            'ap.add_argument("--plan", required=True)\nargs = ap.parse_args()'
        ),
        "n_used = len(X)": (
            "n_used = len(X)\n"
            f"if (n_used, q, args.batch, args.epochs, args.pdrop) != ({TRAIN_N}, {Q}, {BATCH}, {HORIZON}, 0.1):\n"
            "    raise ValueError('matched training settings differ')\n"
            f"if args.label_policy == 'shuffled': C = torch.arange(n_used, device=dev) % {Q}\n"
            "with np.load(args.plan, allow_pickle=False) as schedule:\n"
            "    indexes, dropped = schedule['indices'], schedule['drop']\n"
            f"if indexes.shape != ({grid[-1]}, {UPDATES_PER_EPOCH}, {BATCH}) or dropped.shape != indexes.shape:\n"
            "    raise ValueError('input index or mask schedule differs')\n"
            "indexes = torch.from_numpy(indexes).to(dev)\n"
            "dropped = torch.from_numpy(dropped).to(dev)\n"
            "diffusion_rng = torch.Generator(device=dev).manual_seed(args.seed + 1_000_000)"
        ),
        "ck_epochs = ckpt_schedule(args.first_ckpt_epoch, args.epochs, args.max_ratio)": f"ck_epochs = {list(grid)!r}",
        "math.ceil(n_used / args.batch)": str(UPDATES_PER_EPOCH),
        "for ep in range(1, args.epochs + 1):\n"
        "    perm = torch.randperm(n, device=dev)\n"
        "    for s in range(0, n, args.batch):\n"
        "        idx = perm[s:s + args.batch]": (
            f"for ep in range(1, {grid[-1]} + 1):\n"
            f"    for s in range({UPDATES_PER_EPOCH}):\n"
            "        idx = indexes[ep - 1, s]"
        ),
        "torch.randint(1, args.t_final + 1, (B,), device=dev)":
            "torch.randint(1, args.t_final + 1, (B,), device=dev, generator=diffusion_rng)",
        "c[torch.rand(B, device=dev) < args.pdrop] = q": "c[dropped[ep - 1, s]] = q",
        "xt = H.noise_batch(F.one_hot(x0, q).float(), t, ab)":
            "xt = H.noise_batch(F.one_hot(x0, q).float(), t, ab, diffusion_rng)",
    }, "matched trainer")


def instrument_eval(text):
    return replace_pinned(text, {
        'RIN = [torch.full_like(c, CONST_LABEL) if RUN_TYPE == "const" else c for c in RT]':
            'policies = {m.get("label_policy") for m in metas}\n'
            'if RUN_TYPE != "normal" or len(policies) != 1 or not policies <= {"true", "shuffled"}:\n'
            '    raise ValueError("Evaluation expects one matched-label policy")\n'
            'RIN = [torch.arange(ne, device=dev) % q for c in RT] if policies == {"shuffled"} else RT',
    }, "matched scorer")


def train(trainer, data, plan, scratch, policy, picks, grid, device, smoke):
    runs = []
    for pick in picks:
        out = scratch / f"{policy}-pick{pick}"
        command = [sys.executable, "-IB", str(trainer), "--data", str(data), "--out", str(out),
                   "--pick", str(pick), "--seed", str(TRAIN_SEED + pick), "--n", str(TRAIN_N),
                   "--epochs", str(HORIZON), "--batch", str(BATCH), "--pdrop", "0.1",
                   "--plan", str(plan), "--label-policy", policy, "--device", device]
        if smoke:
            command += ["--hidden", "32", "--layers", "1"]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL if smoke else None)
        meta = json.loads((out / "meta.json").read_text())
        done = json.loads((out / "completion.json").read_text())
        if ((meta["pick"], meta["seed"], meta["n_c"], meta["steps_per_epoch"], meta["ckpt_epochs"], meta["label_policy"])
                != (pick, TRAIN_SEED + pick, TRAIN_N, UPDATES_PER_EPOCH, list(grid), policy)
                or done["meta_sha256"] != hash_file(out / "meta.json") or done["data_sha256"] != hash_file(data)
                or done["epochs"] != list(grid)):
            raise ValueError("Training receipt differs from the fixed condition and schedule")
        runs.append(out)
    return runs


def evaluate(scorer, data, scratch, policy, runs, grid, device, smoke):
    output = scratch / f"{policy}-curves.npz"
    count = 36 if smoke else EVAL_N
    subprocess.run([sys.executable, "-IB", str(scorer), "--data", str(data), "--out", str(output),
                    "--runs", *map(str, runs), "--device", device, "--seed", str(EVAL_SEED),
                    "--epochs", ",".join(map(str, grid)), "--n-eval", str(count), "--block-size", str(TRAIN_N),
                    "--test-block", "6", "--guidance", "0"], check=True,
                   stdout=subprocess.DEVNULL if smoke else None)
    return output


def read_curves(path, source):
    from replay import load_curve_class
    from schedule import WEEK2_T_GRID

    curves = load_curve_class(source)(path)
    if (tuple(curves.picks), tuple(curves.ep), tuple(curves.br), curves.n_eval, curves.test_block,
            tuple(curves.tgrid)) != (PICKS, GRID, BRANCHES, EVAL_N, 6, WEEK2_T_GRID):
        raise ValueError("Curve receipt has a different cohort, branch or time grid")
    pairs = tuple(combinations(PICKS, 2))
    if tuple(map(tuple, curves.pairs)) != pairs:
        raise ValueError("Curve receipt changed the 15 pick pairs")
    return {branch: {pair: (curves.ts(branch, index), "interior" if ok else reason)
                     for index, pair in enumerate(pairs) for ok, reason in (curves.inter(branch, index),)}
            for branch in BRANCHES}


def summarize(measured):
    arms = {(policy, branch): measured[policy][branch] for policy in POLICIES for branch in BRANCHES}
    pairs = tuple(combinations(PICKS, 2))
    if any(set(values) != set(pairs) for values in arms.values()):
        raise ValueError("Policy and null clocks must describe the same pick pairs")
    keep = [pair for pair in pairs if all(values[pair][1] == "interior" for values in arms.values())]
    result = {"common_valid_pairs": len(keep),
              "curve_status": {f"{policy}_{branch}": Counter(row[1] for row in values.values())
                               for (policy, branch), values in arms.items()}}
    if len(keep) < 10:
        return result | {"state": "insufficient_matched_interior_pairs"}
    logs = {key: np.log([values[pair][0] for pair in keep]) for key, values in arms.items()}
    condition = logs["shuffled", "cond"] - logs["true", "cond"]
    null = logs["shuffled", "uncond"] - logs["true", "uncond"]
    delta = np.stack((condition, null, condition - null), axis=1)
    i, j = np.asarray(keep).T
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = []
    for replicate in rng.integers(len(PICKS), size=(BOOTSTRAP_DRAWS, len(PICKS))):
        counts = np.bincount(replicate, minlength=len(PICKS))
        weights = counts[i] * counts[j]
        if weights.sum():
            draws.append(np.average(delta, weights=weights, axis=0))
    if len(draws) < 0.95 * BOOTSTRAP_DRAWS:
        return result | {"state": "insufficient_valid_paired_bootstraps"}
    intervals = np.quantile(draws, [0.025, 0.975], axis=0).T.tolist()
    names = ("shuffled_minus_true_cond", "shuffled_minus_true_null", "cond_minus_null_interaction")
    return result | {"state": "descriptive_comparison", "valid_bootstrap_draws": len(draws),
                     "geometric_times": {f"{policy}_{branch}": float(np.exp(values.mean()))
                                         for (policy, branch), values in logs.items()},
                     "log_time": {name: {"mean": float(value), "ci95": interval}
                                  for name, value, interval in zip(names, delta.mean(axis=0), intervals, strict=True)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "prefix", "scratch"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--commit")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--cpu-smoke", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    job = os.environ.get("SLURM_JOB_ID", "")
    if args.run and (not re.fullmatch(r"[1-9][0-9]*", job) or not re.fullmatch(r"[0-9a-f]{40}", args.commit or "")):
        parser.error("Production training requires Slurm with a full freeze")
    sys.path[:0] = [str(args.prefix), str(args.source / "scripts")]
    data = args.scratch / "condition-panel.npy"
    provenance = build_data(data) | verify_schedule(*schedule())
    expected = json.loads((HERE / "inputs.json").read_text())
    if any(provenance[key] != expected[key] for key in ("data_sha256", "schedule_sha256", "literal_grammar_sha256")):
        raise ValueError("Data or balanced exposure bytes differ from the frozen experiment")
    smoke = args.cpu_smoke
    grid = (1,) if smoke else GRID
    trainer = args.source / "scripts/condition_trainer.py"
    scorer = args.source / "scripts/condition_scorer.py"
    trainer_code = instrument_train((args.source / "scripts/train_hier_cond_w2.py").read_text(), grid)
    scorer_code = instrument_eval((args.source / "scripts/eval_hier_oracle.py").read_text())
    if args.check_only:
        print(json.dumps(provenance, sort_keys=True))
        return
    trainer.write_text(trainer_code)
    scorer.write_text(scorer_code)
    indices, drop = schedule(grid[-1])
    verify_schedule(indices, drop, complete=not smoke)
    plan = args.scratch / "condition-schedule.npz"
    np.savez(plan, indices=indices, drop=drop)
    picks, device = ((0, 1), "cpu") if smoke else (PICKS, "cuda")
    measured = {}
    for policy in POLICIES:
        runs = train(trainer, data, plan, args.scratch, policy, picks, grid, device, smoke)
        evaluation = evaluate(scorer, data, args.scratch, policy, runs, grid, device, smoke)
        if smoke:
            if not evaluation.is_file():
                raise ValueError("CPU scorer did not produce its temporary smoke artifact")
        else:
            measured[policy] = read_curves(evaluation, args.source)
    if smoke:
        print("CPU smoke passed with both policies, two picks, and the original branch scorer")
        return
    receipt = provenance | {"scope": "prospective_matched_condition_information", "commit": args.commit,
                            "source_commit": SOURCE_COMMIT, "job_id": job, "grid": GRID, "evaluation_rows": EVAL_N,
                            "comparison": summarize(measured)}
    directory = SHARED / "runs/two-clocks-condition" / job
    directory.mkdir(parents=True, exist_ok=False)
    pending = directory / "results.partial"
    pending.write_text(json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n")
    pending.replace(directory / "results.json")


if __name__ == "__main__":
    main()
