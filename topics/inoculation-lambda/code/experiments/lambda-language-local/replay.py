"""Run the frozen language panel on the local CPU with its own receipt."""

import argparse
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import re
import socket
import sys
from uuid import uuid4
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lambda-language-panel"))
import panel


def summarize_panel(output, frames, identify, receipt):
    requests = panel.verified_packet_file("author_requests.jsonl", receipt)
    scores = panel.classify_blind(requests, frames, identify)
    blind = panel.encode({key: asdict(score["counts"]) for key, score in scores.items()})
    with (output / "blind_counts.json").open("xb") as stream:
        stream.write(blind)
    cells = panel.restore_grid(panel.verified_packet_file("mapping.json", receipt), scores)
    panel.need(panel.fingerprint(panel.PACKET / "pilot.json") == receipt["files"]["pilot.json"],
               "fixed pilot changed")
    return {
        "blind_counts_sha256": panel.digest(blind),
        "bootstrap": {"unit": "question", "resamples": panel.N_RESAMPLES, "seed": panel.RESAMPLE_SEED},
        "full_panel": panel.summaries(cells),
        "matched_pilot": panel.summaries(cells, pilot=True),
    }


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--questions", type=Path, required=True)
    cli.add_argument("--freeze", required=True)
    cli.add_argument("--panel-freeze", required=True)
    args = cli.parse_args()
    panel.need(all(re.fullmatch(r"[0-9a-f]{40}", s) for s in (args.freeze, args.panel_freeze)),
               "expected full frozen launcher and panel commits")

    started = datetime.now(ZoneInfo("Asia/Taipei"))
    run_id = started.strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:12]
    receipt = panel.read_receipt()
    frames = panel.load_questions(args.questions, receipt)
    model = panel.load_model()
    controls = panel.synthetic_checks(model.predict)
    output = panel.RUNS / f"language-panel-local-{run_id}"
    output.mkdir(mode=0o700)
    result = {
        "freeze": args.panel_freeze,
        "local_launcher_freeze": args.freeze,
        "execution": {"mode": "local_cpu", "host": socket.gethostname(), "run_id": run_id,
                      "started_at_taipei": started.isoformat(timespec="seconds")},
        "scope": panel.JUDGE_NOTE,
        "packet_receipt_sha256": panel.RECEIPT_SHA,
        "packet_files": receipt["files"],
        "language_model_sha256": panel.MODEL_SHA,
        "language_model_bytes": panel.MODEL_BYTES,
        "python_version": ".".join(map(str, panel.PYTHON_VERSION)),
        "package_versions": panel.PACKAGE_VERSIONS,
        "synthetic_controls": controls,
    } | summarize_panel(output, frames, model.predict, receipt)

    prepared = output / "summary.partial"
    with prepared.open("xb") as stream:
        stream.write(panel.encode(result))
    summary = prepared.rename(output / "summary.json")
    print(panel.encode({"execution": result["execution"], "freeze": args.panel_freeze,
                        "local_launcher_freeze": args.freeze, "responses": len(panel.GRID),
                        "summary_sha256": panel.fingerprint(summary)["sha256"],
                        "summary": str(summary)}).decode(), end="")


if __name__ == "__main__":
    main()
