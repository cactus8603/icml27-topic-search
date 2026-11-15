"""Export the frozen Spanish judge grid or wrap its native Batch delivery offline."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from tempfile import NamedTemporaryFile


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lambda-author-import"))
import import_scores as importer


PACKET_SHA = "8873228267a59be7452ce2076bdc27b45fc63728b0e3cf05a647408bc94236c8"
IMPORTER_FREEZE = "bcbcc012d6f7f55b58e2bacafc9e1906c3741521"
BATCH_INPUT_SHA = "9c79776866bed3e35a6f5c863d4504b9c0af0bde785543d7fe2f61fd8a9f31ee"
ENDPOINT = "/v1/chat/completions"
WHITESPACE = re.compile(r"[ \t\r\n]*")


def reject_constant(_):
    raise ValueError("invalid JSON constant in Batch output")


DECODER = json.JSONDecoder(object_pairs_hook=importer.unique, parse_constant=reject_constant)


def export_grid(files):
    _, hashes = importer.request_grid(files)
    rows = importer.by_id(files["author_requests.jsonl"])
    bindings, payload = {}, []
    for key, (row, _) in rows.items():
        request_sha = hashes[key]
        custom_id = f"{key}-{request_sha[:32]}"
        bindings[custom_id] = key, request_sha
        payload.append(importer.encode({"custom_id": custom_id, "method": "POST", "url": ENDPOINT,
                                        "body": row["request"]}))
    return b"".join(payload), bindings


def frozen_export(packet):
    _, files = importer.review_packet(packet, PACKET_SHA)
    payload, bindings = export_grid(files)
    importer.need(importer.digest(payload) == BATCH_INPUT_SHA, "frozen Batch export fingerprint changed")
    return payload, bindings


def raw_member(document, wanted):
    """Return the exact value text of a member of an already validated object."""
    start = WHITESPACE.match(document).end()
    cursor = WHITESPACE.match(document, start + 1).end()
    while document[cursor] != "}":
        field, name_end = DECODER.raw_decode(document, cursor)
        colon = WHITESPACE.match(document, name_end).end()
        value_start = WHITESPACE.match(document, colon + 1).end()
        _, value_end = DECODER.raw_decode(document, value_start)
        if field == wanted:
            return document[value_start:value_end]
        next_character = WHITESPACE.match(document, value_end).end()
        if document[next_character] == "}":
            break
        cursor = WHITESPACE.match(document, next_character + 1).end()
    raise ValueError("missing raw Batch envelope member")


def record_id(value):
    return isinstance(value, str) and bool(value.strip())


def wrap_results(raw, bindings):
    native = {}
    seen_batch, seen_requests, seen_completions = set(), set(), set()
    for line in raw.splitlines():
        document = line.decode("utf-8")
        row = DECODER.decode(document)
        key, batch_id = row["custom_id"], row["id"]
        importer.need(isinstance(key, str) and key in bindings and key not in native,
                      "unexpected or duplicate Batch custom_id")
        response = row["response"]
        importer.need(row["error"] is None and isinstance(response, dict)
                      and type(response["status_code"]) is int and response["status_code"] == 200,
                      "Batch row did not complete successfully")
        request_id, body = response["request_id"], response["body"]
        importer.need(isinstance(body, dict) and body.get("object") == "chat.completion"
                      and body.get("model") == importer.JUDGE["model"]
                      and isinstance(body.get("id"), str) and body["id"].startswith("chatcmpl-")
                      and len(body["id"]) > len("chatcmpl-"),
                      "unexpected native response type, snapshot or completion identity")
        for identifier, seen in ((batch_id, seen_batch), (request_id, seen_requests),
                                 (body["id"], seen_completions)):
            importer.need(record_id(identifier) and identifier not in seen, "missing or reused response identity")
            seen.add(identifier)
        blind_id, request_sha = bindings[key]
        header = importer.encode({"id": blind_id, "request_sha256": request_sha})
        body_raw = raw_member(raw_member(document, "response"), "body").encode()
        native[key] = header[:-2] + b',"response":' + body_raw + b"}\n"
    importer.need(native.keys() == bindings.keys(), "Batch delivery does not cover the entire fixed grid")
    return b"".join(native[key] for key in bindings)


def checked_file(path, sha):
    data = importer.bounded(path)
    importer.need(importer.digest(data) == sha, "file differs from its checked SHA-256")
    return data


def private_write(path, payload):
    importer.need(len(payload) <= importer.LIMIT, "output exceeds the offline size limit")
    with NamedTemporaryFile(dir=path.parent) as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
        os.link(stream.name, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    commands = parser.add_subparsers(dest="mode", required=True)
    export = commands.add_parser("export")
    export.add_argument("--out", type=Path)
    wrap = commands.add_parser("wrap")
    wrap.add_argument("--batch-input", type=Path, required=True)
    wrap.add_argument("--batch-input-sha256", required=True)
    wrap.add_argument("--batch-output", type=Path, required=True)
    wrap.add_argument("--batch-output-sha256", required=True)
    wrap.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    importer.own_freeze(IMPORTER_FREEZE)
    exported, bindings = frozen_export(args.packet)
    summary = {"rows": len(bindings), "packet_sha256": PACKET_SHA,
               "batch_input_sha256": BATCH_INPUT_SHA}
    if args.mode == "export":
        if args.out is not None:
            private_write(args.out, exported)
    else:
        supplied = checked_file(args.batch_input, args.batch_input_sha256)
        importer.need(supplied == exported, "Batch input differs from the exact frozen export")
        raw = checked_file(args.batch_output, args.batch_output_sha256)
        native = wrap_results(raw, bindings)
        private_write(args.out, native)
        summary.update({"batch_output_sha256": args.batch_output_sha256,
                        "native_delivery_sha256": importer.digest(native)})
    print(importer.encode(summary).decode(), end="")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, TypeError, KeyError, AttributeError, IndexError, OSError,
            subprocess.CalledProcessError) as error:
        message = str(error) if type(error) is ValueError else "invalid or unavailable frozen inputs"
        sys.exit(f"offline Batch adapter refused: {message}")
