"""Shared provenance and JSON checks for model inference and CPU evaluation."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "real_holdout"))
from prepare import EXPECTED_OUTPUTS


PANEL_COMMIT = "37a8a783bd7b1f7648dfbf0376cbbb99099cad58"
SPLIT_GROUPS = {"train": 250, "test": 747}
ROLES = ("weak", "strong")
MODEL_PINS = {
    "weak": ("Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca"),
    "strong": ("Qwen/Qwen3-32B-AWQ", "0499c3ac83fdef8810b907a23894ba91e95eddd8"),
}
DIGEST = re.compile(r"[0-9a-f]{64}")


def unique_fields(pairs):
    output = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("Duplicate JSON field")
        output[key] = value
    return output


def load_json(payload):
    return json.loads(payload, object_pairs_hook=unique_fields)


def checked_bytes(payload, expected):
    if not DIGEST.fullmatch(expected) or hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError("Input artifact bytes do not match their frozen receipt")


@dataclass(frozen=True)
class Binding:
    """The protocol and verdict rule chosen before scoring the holdout."""

    freeze_commit: str
    measurement_sha256: str
    decision_rule_sha256: str

    def metadata(self):
        if (not re.fullmatch(r"[0-9a-f]{40}", self.freeze_commit)
                or any(not DIGEST.fullmatch(value) for value in (self.measurement_sha256, self.decision_rule_sha256))):
            raise ValueError("An explicit freeze, measurement protocol and decision rule are required")
        return {
            "freeze_commit": self.freeze_commit, "measurement_sha256": self.measurement_sha256,
            "decision_rule_sha256": self.decision_rule_sha256, "panel_commit": PANEL_COMMIT,
            "panel_prompts_sha256": EXPECTED_OUTPUTS["prompts.jsonl"]["sha256"],
            "panel_gold_sha256": EXPECTED_OUTPUTS["gold.jsonl"]["sha256"],
        }
