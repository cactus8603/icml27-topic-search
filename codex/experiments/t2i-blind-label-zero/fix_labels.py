"""Accept the rubric's negative label in the frozen blind scoring bundle."""

import argparse
import hashlib
from pathlib import Path


EVIDENCE_SHA256 = "8c1f0b691d4f400610604faf81da8648c508db8f8594364ef819089728dc3d0d"
TEST_SHA256 = "faef53561040278727174f6dc9d3abad58ec32090c5e8655488e4ebd46f76a05"
RUNNER_SHA256 = "8c469722320ebdc46a885976f88da12a58629071f832af93a50617ddad83b39e"
OLD_LABELS = b'    if not set(blind.label) <= {"1", "2", "?"}:\n        raise ValueError("review labels must all be 1, 2, or ?")'
NEW_LABELS = b'    if not set(blind.label) <= {"0", "1", "2", "?"}:\n        raise ValueError("review labels must all be 0, 1, 2, or ?")'
OLD_INVALID_TEST = rb'"0,1\n1,2\n2,0\n"):'
NEW_INVALID_TEST = rb'"0,1\n1,2\n2,99\n"):'
RUNNER_LINE = b'"$python" -B "$bundle/$pipeline/pipeline.py"'
APPLY = b'"$python" -IB "$T2I_REVIEW_LABEL_FIX" evidence "$bundle/$topic/scripts/evidence.py"\n'


def replace_frozen(source, expected_sha256, old, new):
    if hashlib.sha256(source).hexdigest() != expected_sha256 or source.count(old) != 1:
        raise ValueError("The frozen source does not match the registered patch target")
    return source.replace(old, new, 1)


def evidence_source(source):
    return replace_frozen(source, EVIDENCE_SHA256, OLD_LABELS, NEW_LABELS)


def fixture_source(source):
    return replace_frozen(source, TEST_SHA256, OLD_INVALID_TEST, NEW_INVALID_TEST)


def runner_source(source):
    return replace_frozen(source, RUNNER_SHA256, RUNNER_LINE, APPLY + RUNNER_LINE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=("evidence", "runner"))
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    body = args.source.read_bytes()
    if args.target == "runner":
        args.source.write_bytes(runner_source(body))
        return
    fixture = args.source.with_name("test_evidence.py")
    fixed_body = evidence_source(body)
    fixed_test = fixture_source(fixture.read_bytes())
    args.source.write_bytes(fixed_body)
    fixture.write_bytes(fixed_test)


if __name__ == "__main__":
    main()
