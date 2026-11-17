#!/usr/bin/env python3
"""[claude] Run one Claude Fable 5.2 judgment call through the Messages API.

The prompt and any attached local files go in one user message; Fable cannot read the repo
itself, so attach everything it needs. Credentials are read from ~/.config/anthropic/api_key
(chmod 600, never committed). Every call appends one line of usage and estimated cost to
icml/runs/fable-api/usage.jsonl.

Usage:
  /work/ChihChungHsu/cactus8603/icml/envs/anthropic-api/bin/python scripts/fable_api.py \
      --prompt brief.md --attach PREREG.md summary.json --out reply.md [--effort high]
"""
import argparse
import datetime
import json
import os
import sys
from pathlib import Path

import anthropic

MODEL = "claude-fable-5-2"
PRICE = {"input": 8.0, "output": 40.0, "cache_read": 0.20}  # USD per 1M tokens (Fable 5.2)
KEY_FILE = Path.home() / ".config/anthropic/api_key"
LOG = Path("/work/ChihChungHsu/cactus8603/icml/runs/fable-api/usage.jsonl")


def make_client():
    key = KEY_FILE.read_text().strip()
    return anthropic.Anthropic(api_key=key, timeout=3600.0, max_retries=3)


def build_content(prompt_text, attachments):
    parts = []
    for p in attachments:
        p = Path(p)
        parts.append(f'<file path="{p}">\n{p.read_text(errors="replace")}\n</file>')
    parts.append(prompt_text)
    return "\n\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True, help="file with the task / question")
    ap.add_argument("--attach", nargs="*", default=[], help="local text files to include verbatim")
    ap.add_argument("--system", help="optional file with a system prompt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--max-tokens", type=int, default=64000)
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    client = make_client()
    kw = dict(
        model=MODEL,
        max_tokens=a.max_tokens,
        output_config={"effort": a.effort},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        messages=[{"role": "user", "content": build_content(Path(a.prompt).read_text(), a.attach)}],
    )
    if a.system:
        kw["system"] = Path(a.system).read_text()
    try:
        with client.beta.messages.stream(**kw) as stream:
            msg = stream.get_final_message()
    except anthropic.APIStatusError as e:
        print(f"[claude] API error {e.status_code}: {e.message}", file=sys.stderr)
        sys.exit(2)

    text = "".join(b.text for b in msg.content if b.type == "text")
    Path(a.out).write_text(text)
    u = msg.usage
    cache_read = getattr(u, "cache_read_input_tokens", 0) or 0
    cost = (u.input_tokens * PRICE["input"] + u.output_tokens * PRICE["output"] + cache_read * PRICE["cache_read"]) / 1e6
    rec = {"time": datetime.datetime.now().isoformat(timespec="seconds"), "label": a.label, "model": msg.model,
           "stop_reason": msg.stop_reason, "input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
           "cache_read": cache_read, "usd_est": round(cost, 4), "out": str(a.out)}
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))
    if msg.stop_reason == "refusal":
        print("[claude] refusal (whole fallback chain declined)", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
