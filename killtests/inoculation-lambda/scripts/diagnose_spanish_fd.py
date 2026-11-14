#!/usr/bin/env python3
"""Run the frozen Spanish data pipeline with a diagnostic central-difference callback."""

import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys

import torch


STEP_SCALES = (1.0, 1 / 2, 1 / 8, 1 / 32, 1 / 128, 1 / 256)
ARMS = ("none", "ip_n3", "pps_eps2")


@torch.no_grad()
def evaluate_direction(params, direction, eta, evaluate_nll):
    baseline = [param.detach().clone() for param in params]
    measures = []
    try:
        nll0 = evaluate_nll()
        for scale in STEP_SCALES:
            h = eta * scale
            values = []
            for signed_h in (-h, h):
                for param, original, grad in zip(params, baseline, direction, strict=True):
                    param.copy_(original).add_(grad, alpha=signed_h)
                values.append(evaluate_nll())
            minus, plus = values
            measures.append({
                "eta_fraction": scale,
                "h": h,
                "nll_minus": minus,
                "nll_base": nll0,
                "nll_plus": plus,
                "one_sided_delta_tau": nll0 - minus,
                "central_delta_tau": (plus - minus) / 2,
                "central_slope": (plus - minus) / (2 * h),
                "symmetric_residual": (plus + minus) / 2 - nll0,
            })
    finally:
        for param, original in zip(params, baseline, strict=True):
            param.copy_(original)
    return measures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-root", type=Path, required=True)
    parser.add_argument("--official-raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    frozen_root, out = args.frozen_root.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    official = {
        row["defense"]: row
        for row in json.loads(args.official_raw.read_text())["rows"]
        if "dtau_central" in row
    }
    if tuple(official) != ARMS:
        raise ValueError(f"Expected official Spanish arms {ARMS}, found {tuple(official)}")

    source = frozen_root / "scripts" / "compute_lambda.py"
    spec = importlib.util.spec_from_file_location("frozen_compute_lambda", source)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    results = {}

    def collect_sweep(rows, direction, gtau, tau_nll, tau_cache, model, params, micro,
                      pad_id, device, tau_items, state, gate, _target, arm, seed, fold):
        key = ("trait_heldout", gate["fd_ctx"])
        row, = [row for row in rows if (row["defense"], row["seed"], row["fold"],
                                       row["tau_set"], row["ctx"], row["group"]) ==
                (arm, seed, fold, *key, "all")]
        if seed != 0 or fold != 0 or arm not in ARMS or arm in results or key not in gtau:
            raise ValueError(f"Unexpected diagnostic sample: {(arm, seed, fold, key)}")
        if arm == "none":
            state["eta"] = gate["fd_target_dtau"] / row["dot"]
        eta = state["eta"]
        official_row = official[arm]
        checks = {
            name: math.isclose(row[name], official_row[name], rel_tol=1e-4, abs_tol=5e-6)
            for name in ("dot", "gL_norm", "gT_norm")
        }
        if not all(checks.values()) or not math.isclose(eta, official_row["eta"], rel_tol=1e-4):
            raise ValueError(f"Diagnostic direction does not match the official run for {arm}: {checks}")
        if key not in tau_cache:
            ctx, = [ctx for ctx in tau_cache["_ctxs"] if ctx["name"] == key[1]]
            tau_cache[key] = tau_items(key[0], ctx)
        items = tau_cache[key]

        def eval_nll():
            return original.nll_nograd(model, items, micro, pad_id, device)

        measurements = evaluate_direction(params, direction, eta, eval_nll)
        if not math.isclose(measurements[0]["nll_base"], tau_nll[key], rel_tol=1e-6, abs_tol=1e-7):
            raise ValueError(f"Diagnostic and original forward NLL differ for {arm}")
        for point in measurements:
            point["displacement_norm"] = point["h"] * row["gL_norm"]
            point["abs_slope_error"] = abs(point["central_slope"] - row["dot"])
            if not all(math.isfinite(value) for value in point.values()):
                raise ValueError(f"Non-finite diagnostic result for {arm}")
            print(f"[sweep] {arm} h/eta={point['eta_fraction']:g} "
                  f"slope={point['central_slope']:+.8g} lambda={row['dot']:+.8g}", flush=True)
        results[arm] = {
            "lambda": row["dot"], "direction_norm": row["gL_norm"],
            "official_lambda": official_row["dot"], "official_direction_norm": official_row["gL_norm"],
            "official_central_delta_tau": official_row["dtau_central"],
            "official_eta": official_row["eta"], "steps": measurements,
        }

    original.fd_apply = collect_sweep
    sys.argv = [str(source), "--config", str(frozen_root / "configs" / "grant_spanish_qwen7b.yaml"),
                "--out", str(out), "--finite_diff", "--device", "cuda"]
    original.main()
    if tuple(results) != ARMS:
        raise ValueError(f"Expected {ARMS}, measured {tuple(results)}")
    (out / "spanish_step_diagnostic.json").write_text(json.dumps({
        "official_raw": str(args.official_raw.resolve()), "eta_fractions": STEP_SCALES,
        "arms": results,
    }, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
