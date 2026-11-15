"""[codex] Plot frozen analysis/selection artifacts without selecting new policies.

Run directly with an existing Matplotlib environment:
  python -IB plot_results.py --report report.json --selection selected-policy.json --out-dir NEW_DIR
All plotted estimates and intervals are read from those files. No model, policy,
bootstrap or effectiveness computation is imported or executed here.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
from zoneinfo import ZoneInfo


REPO = Path(__file__).resolve().parents[2]
BUDGETS = (25, 50, 75, 100, 150, 200)
OOF = "oof/m1_single_selector"
PALETTE = {"reference": "#0072B2", "admitted": "#CC79A7", "selected": "#D55E00", "oof": "#009E73"}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def safe(path):
    path = Path(path).expanduser().resolve()
    if any("retrieval-independent-panel" in part for part in path.parts):
        raise ValueError("Sealed data are forbidden in an exposed-development plot")
    return path


def read_json(path):
    def invalid(value):
        raise ValueError(f"Nonfinite JSON in plot input: {value}")
    value = json.loads(path.read_text(), parse_constant=invalid)
    if not isinstance(value, dict):
        raise ValueError("Plot inputs must be JSON objects")
    return value


def number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Expected a finite number: {name}")
    return float(value)


def extract(report, selection, report_sha256):
    """Take only predeclared rows and the already locked selected policy."""
    if (report.get("schema") != 2 or report.get("formal_evidence") is not False
            or selection.get("schema") != 1 or selection.get("state") != "LOCKED"
            or selection.get("formal_evidence") is not False
            or selection.get("selected_before_external_effects") is not True
            or selection.get("analysis_sha256") != report_sha256
            or not selection.get("selection_rule")
            or any(not re.fullmatch(r"[0-9a-f]{64}", str(selection.get(key, "")))
                   for key in ("policy_sha256", "platform_plan_sha256"))):
        raise ValueError("Plot requires the matching frozen analysis and LOCKED selection receipt")
    context = report.get("evaluation_context", {})
    if context.get("formal_evidence") is not False:
        raise ValueError("Plot requires explicitly exploratory development provenance")
    mode = context.get("mode")
    if mode == "platform-relative-exploration":
        if (context.get("plan_sha256") != selection["platform_plan_sha256"]
                or context.get("published_reproduction", {}).get("state") != "MISMATCH"
                or context.get("platform_consistency") != "VALID_LOCAL"):
            raise ValueError("Plot selection and local platform context differ")
    elif mode != "published-replay" or context.get("published_reproduction") != "PASS":
        raise ValueError("Unsupported development analysis context")
    selected = selection.get("policy_id")
    methods = report["methods"]
    if not isinstance(selected, str) or not selected.startswith("m1/") or selected not in methods:
        raise ValueError("The locked selected policy is missing from the supplied analysis")
    base = number(methods["core50"]["summary"]["map_pp"], "local baseline mAP")

    def point(name, budget, *, connected=False):
        row = methods[name]
        summary = row["summary"]
        if number(summary["mean_pairs_per_query"], "reranker calls/query") != budget:
            raise ValueError("A plotted method does not have its stated reranker budget")
        counts = summary.get("budget_counts")
        if counts != {str(budget): report["queries"]}:
            raise ValueError("Plot points require one exact fixed budget for every query")
        interval_key = "paired_against_core50_connected_groups" if connected else "paired_against_core50"
        paired = row[interval_key]["ap_pp"]
        delta = number(paired["point_pp"], "paired mAP difference")
        interval = paired["ci95_pp"]
        if not isinstance(interval, list) or len(interval) != 2:
            raise ValueError("Paired mAP intervals must have two endpoints")
        low, high = (number(value, "paired 95% interval endpoint") for value in interval)
        if low > high or not math.isclose(number(summary["map_pp"], "mAP") - base, delta, abs_tol=1e-7, rel_tol=0):
            raise ValueError("The saved interval/paired point does not match its saved summary")
        fixed = number(summary["recall_at_50_pp"], "fixed Recall@50")
        admitted = number(summary["admission_recall_pp"], "admission Recall@budget")
        if not all(0 <= value <= 100 for value in (fixed, admitted)):
            raise ValueError("Saved recall must be a percentage from 0 to 100")
        return {"method": name, "calls_per_query": budget, "delta_map_pp": delta, "ci95_pp": [low, high],
                "fixed_recall_at50_pct": fixed, "admission_recall_pct": admitted,
                "interval_source": interval_key, "bootstrap_unit": "connected groups" if connected else "original sentinel groups"}

    references = []
    for budget in BUDGETS:
        name = f"reference/rerank{budget}"
        if methods[name].get("family") != "fixed_budget_reference" or methods[name].get("budget") != budget:
            raise ValueError("Fixed-budget references differ from the declared plotting protocol")
        references.append(point(name, budget))
    selected_point, oof_point = point(selected, 50), point(OOF, 50, connected=True)
    return {"references": references, "selected": selected_point, "oof": oof_point,
            "policy_id": selected, "local_core50_map_pct": base, "evaluation_context": context,
            "queries": report["queries"], "original_groups": report["groups"],
            "connected_groups": report["oof_grouping"]["connected_groups"]}


def caption(data):
    state = "historical baseline reproduction remains MISMATCH" if (
        data["evaluation_context"]["mode"] == "platform-relative-exploration") else "the saved historical baseline replay passed"
    return (
        "Exposed-panel reranking curves separate relative mAP changes from fixed and admitted recall; all comparisons are exploratory, not independent confirmation. "
        f"Left: six fixed reranking budgets and the already locked global-selected policy ({data['policy_id']}, 50 calls/query) use saved paired 95% percentile bootstrap intervals over the original sentinel query groups; these intervals are descriptive and not adjusted for policy search. "
        "The 50-call OOF (out-of-fold) selector instead uses the saved connected-group interval, joining sentinel groups that share positive images; this conditional development interval fixes the fitted fold selectors, does not refit selection within the bootstrap, and differs from the other interval construction. "
        "Right: fixed Recall@50 measures positives in the final first 50 results, whereas admission Recall@budget measures positives admitted within the reranking budget; the two policy markers show their actual fixed Recall@50. "
        f"The reference is local CORE50 ({data['local_core50_map_pct']:.2f}% mAP); {state}, and calls/query count scored pairs rather than measured latency.\n"
    )


def draw(data, out, *, synthetic_preview=False):
    temporary = REPO / "hf_home/tmp"
    temporary.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(temporary)
    os.environ["MPLCONFIGDIR"] = str(temporary / "matplotlib-plot-results")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    settings = {"font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
                "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
                "axes.linewidth": .8, "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.facecolor": "white"}
    with plt.rc_context(settings):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.8))
        fig.subplots_adjust(left=.105, right=.98, bottom=.40, top=.85, wspace=.34)
        references = data["references"]
        x = [row["calls_per_query"] for row in references]
        left, right = axes
        for ax in axes:
            ax.set_xticks(BUDGETS)
            ax.set_xlim(13, 212)
            ax.set_xlabel("Reranker calls / query")
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", color=".9", linewidth=.6, zorder=0)
            ax.set_axisbelow(True)
        left.axhline(0, color=".4", linewidth=.9, linestyle="--", zorder=1)
        left.plot(x, [row["delta_map_pp"] for row in references], color=PALETTE["reference"],
                  marker="o", markersize=4.5, linewidth=1.5, zorder=3)

        def interval(row, color, marker, linestyle, size=6):
            location, value = row["calls_per_query"], row["delta_map_pp"]
            low, high = row["ci95_pp"]
            # Draw absolute interval endpoints: percentile intervals need not
            # contain the point estimate, so asymmetric yerr can be invalid.
            left.vlines(location, low, high, color=color, linewidth=1.4, linestyle=linestyle, zorder=4)
            left.hlines([low, high], location - 1.5, location + 1.5, color=color, linewidth=1.1, zorder=4)
            left.plot(location, value, marker=marker, markersize=size, markerfacecolor="white",
                      markeredgecolor=color, markeredgewidth=1.5, linestyle="none", zorder=5)
        for row in references:
            interval(row, PALETTE["reference"], "o", "solid", size=4.5)
        interval(data["selected"], PALETTE["selected"], "^", "dashed", size=8)
        interval(data["oof"], PALETTE["oof"], "D", "dotted", size=5.5)
        bounds = [0] + [value for row in [*references, data["selected"], data["oof"]]
                        for value in (*row["ci95_pp"], row["delta_map_pp"])]
        low, high = min(bounds), max(bounds)
        padding = max(.15, .1 * (high - low))
        left.set_ylim(low - padding, high + padding)
        left.yaxis.set_major_locator(MaxNLocator(nbins=5))
        left.set_ylabel("ΔmAP vs local CORE50 (pp)")
        left.set_title("(a) Relative mAP", loc="left", pad=9)
        left.legend(handles=[
            Line2D([], [], color=PALETTE["reference"], marker="o", markersize=4.5, label="Fixed budgets: sentinel CI"),
            Line2D([], [], color=PALETTE["selected"], marker="^", markerfacecolor="white", linestyle="--", label="Global selected: sentinel CI"),
            Line2D([], [], color=PALETTE["oof"], marker="D", markerfacecolor="white", linestyle=":", label="OOF: connected CI (conditional)")],
            loc="upper left", bbox_to_anchor=(-.025, -.29), frameon=False, borderaxespad=0, handlelength=2.0)

        right.plot(x, [row["fixed_recall_at50_pct"] for row in references], color=PALETTE["reference"],
                   marker="o", markersize=4.5, linewidth=1.5, label="Fixed Recall@50")
        right.plot(x, [row["admission_recall_pct"] for row in references], color=PALETTE["admitted"],
                   marker="s", markersize=4.5, linestyle="--", linewidth=1.5, label="Admission Recall@budget")
        for key, marker, size, label in (("selected", "^", 8, "Global selected: fixed Recall@50"),
                                          ("oof", "D", 5.5, "OOF: fixed Recall@50")):
            right.plot(50, data[key]["fixed_recall_at50_pct"], color=PALETTE[key], marker=marker,
                       markersize=size, markerfacecolor="white", markeredgewidth=1.5, linestyle="none", label=label, zorder=5)
        right.set_ylim(0, 100)
        right.set_yticks([0, 25, 50, 75, 100])
        right.set_ylabel("Recall (%)")
        right.set_title("(b) Fixed vs admitted recall", loc="left", pad=9)
        right.legend(loc="upper left", bbox_to_anchor=(-.025, -.29), frameon=False, borderaxespad=0, handlelength=2.0)
        header = "SYNTHETIC LAYOUT SMOKE" if synthetic_preview else "Exploratory · exposed development panel"
        fig.text(.105, .97, header, fontsize=9, va="top", color=".25")
        fig.text(.105, .035, "95% paired intervals; OOF conditions on fixed fitted selectors. Global selection is not confirmation.",
                 fontsize=9, color=".25")
        out.mkdir(parents=True, exist_ok=False)
        for extension in ("pdf", "svg", "png"):
            fig.savefig(out / f"reranking-budget.{extension}", dpi=200)
        plt.close(fig)
    return matplotlib.__version__


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--synthetic-preview", action="store_true", help="Mark a layout smoke fixture explicitly as synthetic")
    args = parser.parse_args()
    report_path, selection_path, out = map(safe, (args.report, args.selection, args.out_dir))
    if out.exists():
        raise FileExistsError("Choose a new plot output directory")
    report_sha, selection_sha = sha(report_path), sha(selection_path)
    data = extract(read_json(report_path), read_json(selection_path), report_sha)
    version = draw(data, out, synthetic_preview=args.synthetic_preview)
    (out / "caption.txt").write_text(("SYNTHETIC LAYOUT SMOKE. " if args.synthetic_preview else "") + caption(data))
    metadata = {"schema": 1, "actor": "[codex]", "formal_evidence": False, "synthetic_preview": args.synthetic_preview,
                "generated_at_taipei": datetime.now(ZoneInfo("Asia/Taipei")).isoformat(),
                "source_files": {"report": {"path": str(report_path), "sha256": report_sha},
                                 "selection": {"path": str(selection_path), "sha256": selection_sha},
                                 "plot_script_sha256": sha(__file__)}, "matplotlib": version, "plotted_data": data}
    (out / "plot-metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(out / "reranking-budget.png")


if __name__ == "__main__":
    main()
