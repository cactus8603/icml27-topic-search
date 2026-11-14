"""Bind the frozen Lambda inputs and dispatch the separate author runtime."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


STAGE = Path(__file__).resolve().parent
INPUTS_SHA = "1a0873aa84988bf845d9e7edd2ab57c8873c4a359c2459eef99fcd66919ae1c0"
HELPER_COMMIT = "1280b1ff17b18684358fa9a56227d5df5c44e43b"
ARMS = ("none", "ip")


def write_json(path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("helpers", "checkout", "source", "snapshot", "author-python", "scratch"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--shard", type=int, choices=range(len(ARMS)))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--array-job")
    parser.add_argument("--job")
    parser.add_argument("--prior-none", type=Path)
    args = parser.parse_args()
    if args.shard is not None and not (args.output and args.array_job and args.job):
        parser.error("a training shard requires an isolated output and Slurm IDs")
    if args.shard == 1:
        if args.prior_none is None:
            parser.error("IP requires the completed none training receipt")
        prior = json.loads(args.prior_none.read_bytes())
        if (prior.get("arm") != "none" or prior.get("inputs_sha256") != INPUTS_SHA
                or prior.get("launcher_git_commit") != args.commit or "throughput.json" not in prior.get("artifacts", {})):
            raise ValueError("none receipt does not match this frozen IP training")

    sys.path.insert(0, str(args.helpers))
    from phaseb.feasibility import (
        AUTHOR_ARM, AUTHOR_DATASET, AUTHOR_IP, AUTHOR_REVISION, author_training_config, digest, preflight,
    )

    verified = preflight(args.checkout, args.snapshot, args.author_python)
    if verified["inputs_sha256"] != INPUTS_SHA:
        raise ValueError("published Spanish training inputs differ from the audited binding")
    if verified["author_python_version"] != [3, 10, 20] or not verified["author_environment_matches_pins"]:
        raise ValueError("the provisioned author environment differs from the frozen runtime")
    inputs = verified["inputs"]
    for name in (AUTHOR_DATASET, AUTHOR_IP):
        if digest(args.source / name) != inputs["source_hashes"][name]:
            raise ValueError(f"archived author training input differs: {name}")

    def author(action, config):
        return subprocess.run([str(args.author_python), "-IBu", str(STAGE / "worker.py"), action,
                               str(args.source), str(config)], cwd=args.source, check=True,
                              stdout=subprocess.PIPE if action == "verify" else None)

    selected = ARMS if args.shard is None else (ARMS[args.shard],)
    for arm in selected:
        output = args.scratch / f"check-{arm}" if args.shard is None else args.output / arm
        if args.shard is not None:
            output.mkdir(parents=True, exist_ok=False)
        config = (args.scratch if args.shard is None else output) / f"request-{arm}.json"
        write_json(config, author_training_config(args.source, args.snapshot, arm, output))
        author("inspect", config)
        if args.shard is None:
            continue
        author("train", config)
        artifacts = json.loads(author("verify", config).stdout)
        receipt = {
            "schema": 1,
            "author_run": AUTHOR_ARM[arm],
            "arm": arm,
            "array_job": args.array_job,
            "slurm_job": args.job,
            "launcher_git_commit": args.commit,
            "helper_git_commit": HELPER_COMMIT,
            "author_git_commit": AUTHOR_REVISION,
            "author_python": verified["author_python_version"],
            "direct_packages": {name: row["found"] for name, row in verified["environment"].items()},
            "uv_lock_sha256": inputs["source_hashes"]["uv.lock"],
            "inputs_sha256": INPUTS_SHA,
            "model_revision": inputs["model_revision"],
            "resolved_input_config_sha256": digest(config),
            "artifacts": artifacts,
        }
        temporary = output / "receipt.json.partial"
        write_json(temporary, receipt)
        temporary.rename(output / "receipt.json")
        print(f"[codex] {arm} final adapter and completion marker verified", flush=True)


if __name__ == "__main__":
    main()
