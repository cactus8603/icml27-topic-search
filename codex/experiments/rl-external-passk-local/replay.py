"""Run frozen external pass@k analysis with a local execution receipt."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import socket
import sys
from zoneinfo import ZoneInfo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("assets", "source", "analysis", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("freeze", "baseline", "launcher-freeze", "run-id"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()

    sys.path[:0] = [str(args.analysis), str(args.source / "scripts")]
    import analyze as frozen
    from aggregate import mix_curve, pass_at_k_vec

    criteria = json.loads(frozen.checked_bytes(args.source / "configs/criteria.json", frozen.CRITERIA_SHA))
    counts = [frozen.read_counts(frozen.checked_bytes(args.assets / name, sha), frozen.DRAWS)
              for name, sha in frozen.INPUT_SHA.items()]
    base, rl = frozen.paired_counts(*counts, frozen.SOURCE_COUNTS)
    lookup = frozen.pass_lookup(frozen.DRAWS, pass_at_k_vec)
    lower, upper, critical = frozen.simultaneous_band(lookup[rl] - lookup[base])
    band = frozen.certify(lower, upper)
    frozen.verify_public_band(band)
    endpoints = {str(k): frozen.focal_rates(lookup[base], lookup[rl], lower, upper, k) for k in (1, 64, 128)}
    k = criteria["eval"]["k_main"]
    result = {
        "freeze": args.freeze,
        "baseline_commit": args.baseline,
        "launcher_freeze": args.launcher_freeze,
        "execution_mode": "local",
        "local_run_id": args.run_id,
        "execution_host": socket.gethostname(),
        "recorded_at_taipei": datetime.now(ZoneInfo("Asia/Taipei")).isoformat(timespec="seconds"),
        "upstream_commit": frozen.UPSTREAM,
        "input_sha256": frozen.INPUT_SHA,
        "criteria_sha256": frozen.CRITERIA_SHA,
        "prompts": frozen.PROMPTS,
        "draws_per_arm_and_prompt": frozen.DRAWS,
        "source_counts": frozen.SOURCE_COUNTS,
        "numpy_version": frozen.np.__version__,
        "published_dense_grid_reproduction": band | {"bootstrap": frozen.BOOTSTRAP, "seed": frozen.SEED,
                                                       "alpha": frozen.ALPHA, "critical_value": critical},
        "paired_focal_metrics": endpoints,
        "original_gate_reference_checks": frozen.gate_reference(endpoints, criteria, k),
        "response_mix": frozen.response_mix(
            base, rl, lookup, frozen.DRAWS, k, criteria["gate1"]["mix_curve"]["pi_step"], mix_curve
        ),
        "hindsight_endpoints": [frozen.hindsight_endpoints(base, rl, lookup, budget) for budget in (64, 128)],
    }
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
