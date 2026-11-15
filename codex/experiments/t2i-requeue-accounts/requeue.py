"""Recognize an ended Slurm requeue episode while preserving the frozen run."""

import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
from types import ModuleType


FORMAL = "4351ebb70f2e7ad72e0ec8b99658d9ed41c06567"
CONTROLLER = "8e90d6be5fe3828752f3489b5f04161bcb324f35"
PIPELINE_SHA = "bd6493874c75440ad907152680a88d8518b885fe8c357decfe33cc6c4bb896e9"
RUNNER_SHA = "8c469722320ebdc46a885976f88da12a58629071f832af93a50617ddad83b39e"
CONTROLLER_SHA = "321180eeed51addec8ec7c95d771230b3efab63c81c160a3ff9ccae8b34b08f2"
PIPELINE_PATH = Path("codex/experiments/t2i-formal-replay/pipeline.py")
CONTROLLER_PATH = Path("codex/experiments/t2i-formal-continue/coordinator.py")
RUNNER_LINE = b'"$python" -B "$bundle/$pipeline/pipeline.py"'
BOOTSTRAP_LINE = b'"$python" -B "$T2I_REQUEUE_ENTRY" pipeline "$bundle/$pipeline/pipeline.py"'
LABEL_FIX = b'''if [[ -n ${T2I_REQUEUE_LABEL_FIX:-} ]]; then
    "$python" -IB "$T2I_REQUEUE_LABEL_FIX" evidence "$bundle/$topic/scripts/evidence.py"
fi
'''
PLAN_MATCH = b'ledger_entries(text, "formal-plan=2a") != [PLANNER]'
REPLACEMENT_MATCH = b'ledger_entries(text, "formal-plan=2a") != [ORIGINAL_PLANNER, PLANNER]'
SCHEDULE_PYTHON_START = 'exec(compile(sources["pipeline"], pipeline.__file__, "exec"), pipeline.__dict__)'


def verify(body, sha256):
    if hashlib.sha256(body).hexdigest() != sha256:
        raise ValueError("The source does not match the pinned frozen file")


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError("The registered single overlay site does not match")
    return source.replace(old, new, 1)


def rewrite_runner(body):
    verify(body, RUNNER_SHA)
    return replace_once(body, RUNNER_LINE, LABEL_FIX + BOOTSTRAP_LINE)


def load_source(body, path, name):
    module = ModuleType(name)
    module.__file__ = str(path)
    exec(compile(body, str(path), "exec"), module.__dict__)
    return module


def replay_accounting(pipeline):
    pipeline.TERMINAL.add("REQUEUED")
    return pipeline


def load_pipeline(path):
    body = path.read_bytes()
    verify(body, PIPELINE_SHA)
    return replay_accounting(load_source(body, path, "t2i_frozen_pipeline"))


def git_show(repo, revision, path):
    return subprocess.check_output(["git", "-C", str(repo), "show", f"{revision}:{path}"])


def load_controller(repo, planner):
    path = repo / CONTROLLER_PATH
    body = git_show(repo, CONTROLLER, CONTROLLER_PATH)
    verify(body, CONTROLLER_SHA)
    if path.read_bytes() != body:
        raise ValueError("The shared controller must retain its pinned original source")
    rewritten = replace_once(body, PLAN_MATCH, REPLACEMENT_MATCH)
    controller = load_source(rewritten, path, "t2i_frozen_controller")
    controller.ORIGINAL_PLANNER = controller.PLANNER
    controller.PLANNER = planner

    existing = controller.pipeline_module
    controller.pipeline_module = lambda: replay_accounting(existing())
    controller.SCHEDULER_ENTRY = replace_once(controller.SCHEDULER_ENTRY, SCHEDULE_PYTHON_START,
                                              SCHEDULE_PYTHON_START + '\npipeline.TERMINAL.add("REQUEUED")')
    return controller


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("runner", "pipeline", "watch"))
    parser.add_argument("path", type=Path)
    args, rest = parser.parse_known_args()
    if args.action == "runner" and not rest:
        args.path.write_bytes(rewrite_runner(args.path.read_bytes()))
    elif args.action == "pipeline":
        pipeline = load_pipeline(args.path.resolve(strict=True))
        sys.argv = [pipeline.__file__, *rest]
        pipeline.main()
    elif args.action == "watch" and rest and rest[0].isdecimal() and rest[0] != "26802":
        controller = load_controller(args.path.resolve(strict=True), rest[0])
        sys.argv = [controller.__file__, "--controller-freeze", CONTROLLER, *rest[1:]]
        controller.main()
    else:
        parser.error("Use a registered planner replacement to start the watcher")


if __name__ == "__main__":
    main()
