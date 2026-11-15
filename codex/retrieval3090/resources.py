"""[codex] Local shared-GPU queue and verified duplicate-checkpoint cleanup.

Only the standard library is used. GPU locks coordinate cooperating local jobs;
they do not reserve a card against other users or programs.
"""

from __future__ import annotations

import argparse
import ast
import csv
from datetime import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import stat
import struct
import subprocess
import time
from zipfile import ZipFile
from zoneinfo import ZoneInfo


REPO = Path(__file__).resolve().parents[2]
RUNS_ROOT = REPO / "dl/retrieval3090/runs"
EMBEDDINGS = tuple(f"{model}_{part}" for model in ("core_embed", "cs_clip")
                   for part in ("image", "full", "condition"))
CROSS_PARTS = ("cross_top50", "cross_rank51to200")
BLOCK = 1024 * 1024


def now():
    return datetime.now(ZoneInfo("Asia/Taipei")).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


def owned_regular(path, *, root=None):
    """Reject links, foreign files and traversal through symlink directories."""
    path = Path(path).absolute()
    if root is not None:
        root = Path(root).absolute()
        if not path.is_relative_to(root):
            raise ValueError("Path is outside the permitted cache directory")
        current = root
        for part in path.relative_to(root).parts[:-1]:
            current /= part
            info = current.lstat()
            if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
                raise ValueError(f"Unsafe or foreign cache directory: {current}")
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_nlink != 1:
        raise ValueError(f"Refusing a symlink, foreign file or multiply linked file: {path}")
    return info


def atomic_json(path, value):
    path = Path(path)
    if path.exists() or path.is_symlink():
        owned_regular(path)
    temporary = path.with_name(path.name + f".pending-{os.getpid()}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class FileLock:
    def __init__(self, path, *, create=True):
        self.path, self.create, self.fd = Path(path), create, None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_RDWR | os.O_NOFOLLOW | (os.O_CREAT if self.create else 0)
        self.fd = os.open(self.path, flags, 0o600)
        info = os.fstat(self.fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_nlink != 1:
            os.close(self.fd)
            self.fd = None
            raise ValueError("Unsafe advisory lock file")
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            os.close(self.fd)
            self.fd = None
            raise
        return self

    def __exit__(self, *_):
        if self.fd is not None:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            os.close(self.fd)
            self.fd = None


def parse_gpus(output):
    rows = []
    for row in csv.reader(output.splitlines()):
        if not row or all(not value.strip() for value in row):
            continue
        if len(row) != 4:
            raise ValueError("Unexpected nvidia-smi GPU columns")
        index, uuid, free, utilization = (value.strip() for value in row)
        record = {"index": int(index), "uuid": uuid, "free_mib": int(free), "utilization_percent": int(utilization)}
        if not re.fullmatch(r"GPU-[A-Za-z0-9-]+", uuid) or record["index"] < 0 or record["free_mib"] < 0:
            raise ValueError("Invalid GPU identity or memory report")
        if not 0 <= record["utilization_percent"] <= 100:
            raise ValueError("Invalid GPU utilization report")
        rows.append(record)
    if len({row["uuid"] for row in rows}) != len(rows) or len({row["index"] for row in rows}) != len(rows):
        raise ValueError("Duplicate GPU identities")
    return rows


def query_gpus():
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.free,utilization.gpu",
                             "--format=csv,noheader,nounits"],
                            check=True, capture_output=True, text=True, timeout=15)
    return parse_gpus(result.stdout)


def eligible_gpus(rows, *, gpus=None, min_free_mib=14336, max_utilization=20):
    return sorted((row for row in rows if (gpus is None or row["index"] in gpus)
                   and row["free_mib"] >= min_free_mib and row["utilization_percent"] <= max_utilization),
                  key=lambda row: (-row["free_mib"], row["utilization_percent"], row["index"]))


class QueueInterrupted(Exception):
    def __init__(self, signum):
        self.signum = signum


def stop_child(child):
    if child is None or child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()
    except ProcessLookupError:
        pass


def run_queued(command, queue_path, *, gpus=None, min_free_mib=14336, max_utilization=20,
               confirmation_seconds=5, retry_seconds=30, cwd=None, lock_dir=None,
               query=query_gpus, sleep=time.sleep, popen=subprocess.Popen, install_signals=True):
    if not command or any(not isinstance(arg, str) or "\0" in arg for arg in command):
        raise ValueError("A nonempty child argument vector is required")
    if min_free_mib <= 0 or not 0 <= max_utilization <= 100 or confirmation_seconds <= 0 or retry_seconds <= 0:
        raise ValueError("Invalid GPU queue thresholds or polling intervals")
    queue_path = Path(queue_path).absolute()
    if queue_path.is_symlink():
        raise ValueError("Queue state must not be a symlink")
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    lock_dir = Path(lock_dir) if lock_dir else RUNS_ROOT / ".gpu-locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    cwd = str(Path(cwd or REPO).resolve())
    state = {"schema": 1, "owner": "[codex]", "queue_pid": os.getpid(), "child_pid": None,
             "command": list(command), "cwd": cwd, "created_at": now(), "status": "waiting",
             "selected_gpu": None, "min_free_mib": min_free_mib, "max_utilization_percent": max_utilization,
             "confirmation_seconds": confirmation_seconds, "retry_seconds": retry_seconds,
             "allowed_gpu_indices": sorted(gpus) if gpus is not None else None,
             "lock_scope": "Advisory local locks among cooperating jobs; other GPU users remain outside this lock"}
    previous, child, gpu_lock = {}, None, None

    def update(**details):
        state.update(details, updated_at=now())
        atomic_json(queue_path, state)

    def interrupted(signum, _frame):
        raise QueueInterrupted(signum)

    with FileLock(queue_path.with_name(queue_path.name + ".lock")):
        if install_signals:
            for signum in (signal.SIGINT, signal.SIGTERM):
                previous[signum] = signal.signal(signum, interrupted)
        try:
            update()
            prior_available = set()
            while True:
                try:
                    snapshot = query()
                    available = eligible_gpus(snapshot, gpus=gpus, min_free_mib=min_free_mib,
                                              max_utilization=max_utilization)
                    update(status="waiting", gpu_snapshot=snapshot, last_query_error=None,
                           consecutive_candidate_uuids=sorted(prior_available))
                except (OSError, ValueError, subprocess.SubprocessError) as error:
                    prior_available = set()
                    update(status="waiting", last_query_error=f"{type(error).__name__}: {error}")
                    sleep(retry_seconds)
                    continue
                for gpu in available:
                    if gpu["uuid"] not in prior_available:
                        continue
                    candidate_lock = FileLock(lock_dir / f"{gpu['uuid']}.lock")
                    try:
                        candidate_lock.__enter__()
                    except BlockingIOError:
                        continue
                    # Check again after taking the advisory lock before launching.
                    try:
                        current = eligible_gpus(query(), gpus=gpus, min_free_mib=min_free_mib,
                                                max_utilization=max_utilization)
                    except (OSError, ValueError, subprocess.SubprocessError) as error:
                        candidate_lock.__exit__(None, None, None)
                        update(last_query_error=f"{type(error).__name__}: {error}")
                        available = []
                        prior_available = set()
                        break
                    except BaseException:
                        candidate_lock.__exit__(None, None, None)
                        raise
                    selected = next((row for row in current if row["uuid"] == gpu["uuid"]), None)
                    if selected is None:
                        candidate_lock.__exit__(None, None, None)
                        available = [row for row in available if row["uuid"] != gpu["uuid"]]
                        continue
                    gpu_lock = candidate_lock
                    environment = os.environ.copy()
                    environment["CUDA_VISIBLE_DEVICES"] = selected["uuid"]
                    environment["TZ"] = "Asia/Taipei"
                    child = popen(list(command), cwd=cwd, env=environment, shell=False, start_new_session=True)
                    update(status="running", child_pid=child.pid, selected_gpu=selected, started_at=now())
                    returncode = child.wait()
                    update(status="completed" if returncode == 0 else "failed", returncode=returncode, finished_at=now())
                    return returncode
                current_available = {row["uuid"] for row in available}
                # Once confirmed but every lock is busy, wait the normal no-card interval.
                delay = confirmation_seconds if current_available - prior_available else retry_seconds
                prior_available = current_available
                update(next_check_seconds=delay)
                sleep(delay)
        except (QueueInterrupted, KeyboardInterrupt) as error:
            stop_child(child)
            signum = error.signum if isinstance(error, QueueInterrupted) else signal.SIGINT
            update(status="cancelled", finished_at=now(), returncode=128 + signum)
            return 128 + signum
        except BaseException as error:
            stop_child(child)
            update(status="failed", finished_at=now(), error=f"{type(error).__name__}: {error}")
            raise
        finally:
            if gpu_lock is not None:
                gpu_lock.__exit__(None, None, None)
            for signum, handler in previous.items():
                signal.signal(signum, handler)


def read_exact(stream, size):
    data = stream.read(size)
    if len(data) != size:
        raise ValueError("Truncated NPY payload")
    return data


def npy_header(stream):
    if read_exact(stream, 6) != b"\x93NUMPY":
        raise ValueError("Invalid NPY signature")
    major, minor = read_exact(stream, 2)
    if (major, minor) == (1, 0):
        length = struct.unpack("<H", read_exact(stream, 2))[0]
    elif major in (2, 3) and minor == 0:
        length = struct.unpack("<I", read_exact(stream, 4))[0]
    else:
        raise ValueError("Unsupported NPY version")
    if length > 65536:
        raise ValueError("Unreasonably large NPY header")
    header = ast.literal_eval(read_exact(stream, length).decode("utf-8" if major == 3 else "latin1").strip())
    if not isinstance(header, dict) or set(header) != {"descr", "fortran_order", "shape"} or header["fortran_order"] is not False:
        raise ValueError("Only ordinary row-major numeric NPY arrays are accepted")
    shape, dtype = header["shape"], header["descr"]
    if not isinstance(shape, tuple) or not shape or any(type(n) is not int or n <= 0 for n in shape):
        raise ValueError("Invalid NPY array shape")
    match = re.fullmatch(r"[<>=|]([fiu])([1248])", dtype) if isinstance(dtype, str) else None
    if match is None or (match[1] == "f" and int(match[2]) not in (2, 4, 8)):
        raise ValueError("Unsupported or object NPY dtype")
    return {"shape": shape, "dtype": dtype, "itemsize": int(match[2]),
            "bytes": math.prod(shape) * int(match[2])}


def matching_payload(left, right, size):
    remaining = size
    while remaining:
        count = min(BLOCK, remaining)
        if read_exact(left, count) != read_exact(right, count):
            raise ValueError("Checkpoint differs from the final NPZ; no cleanup is allowed")
        remaining -= count


def consume(stream, size):
    remaining = size
    while remaining:
        remaining -= len(read_exact(stream, min(BLOCK, remaining)))


def finished_array(stream):
    if stream.read(1):
        raise ValueError("Trailing content after an NPY array")


def check_state(path, rows, root):
    owned_regular(path, root=root)
    state = json.loads(path.read_text())
    if state.get("completed") != rows or not re.fullmatch(r"[0-9a-f]{64}", str(state.get("fingerprint", ""))):
        raise ValueError("Checkpoint state is incomplete or lacks a provenance fingerprint")
    if not isinstance(state.get("active_seconds"), (int, float)) or state["active_seconds"] < 0:
        raise ValueError("Checkpoint timing state is malformed")


def cleanup_cache(cache_path, *, apply=False, receipt_path=None, runs_root=RUNS_ROOT):
    """Delete only byte-verified redundant arrays; runs_root injection is for tests."""
    root = Path(runs_root).absolute()
    cache = Path(cache_path).absolute()
    if root.is_symlink() or cache.is_symlink() or cache == root or not cache.is_relative_to(root):
        raise ValueError("Cleanup is restricted to new dl/retrieval3090/runs cache directories")
    if cache.resolve() != cache or root.resolve() != root:
        raise ValueError("Cache cleanup must not traverse symlinks")
    for directory in (root, cache):
        info = directory.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
            raise ValueError("Only owned cache directories may be cleaned")
    mandatory = ("vectors.npz", "cross.npz", "panel.json", "meta.json", "checkpoint-provenance.json", "writer.lock")
    for name in mandatory:
        owned_regular(cache / name, root=root)
    with FileLock(cache / "writer.lock", create=False):
        meta = json.loads((cache / "meta.json").read_text())
        if meta.get("replay", {}).get("state") != "PASS":
            raise ValueError("Only a complete baseline-PASS cache may be cleaned")
        panel = json.loads((cache / "panel.json").read_text())
        query_count, image_count = len(panel["queries"]), len(panel["ids"])
        condition_count = len(panel["condition_texts"])
        if min(query_count, image_count, condition_count) <= 0:
            raise ValueError("Empty cache panel")
        paths = []
        headers = {}
        for name in (*EMBEDDINGS, *CROSS_PARTS):
            path = cache / "checkpoints" / f"{name}.npy"
            if not path.exists() and not path.is_symlink():
                continue
            info = owned_regular(path, root=root)
            with path.open("rb") as stream:
                header = npy_header(stream)
                if stream.tell() + header["bytes"] != info.st_size:
                    raise ValueError("Checkpoint has incomplete or trailing array bytes")
            check_state(path.with_suffix(".state.json"), header["shape"][0], root)
            paths.append({"path": str(path), "name": name, "bytes": info.st_size, "sha256": sha256(path),
                          "inode": info.st_ino, "mtime_ns": info.st_mtime_ns, "status": "verified"})
            headers[name] = header
        with ZipFile(cache / "vectors.npz") as vectors, ZipFile(cache / "cross.npz") as cross:
            for archive, expected in ((vectors, {name + ".npy" for name in EMBEDDINGS}),
                                      (cross, {"core_order.npy", "cross.npy"})):
                names = archive.namelist()
                if len(names) != len(set(names)) or not expected.issubset(names):
                    raise ValueError("Final NPZ members are incomplete or duplicated")
                if sum(item.file_size for item in archive.infolist()) > 2 * 1024**3 or archive.testzip() is not None:
                    raise ValueError("Final NPZ archive failed completeness or CRC validation")
            for name in EMBEDDINGS:
                with vectors.open(name + ".npy") as final:
                    header = npy_header(final)
                    expected_rows = image_count if name.endswith("_image") else query_count if name.endswith("_full") else condition_count
                    if len(header["shape"]) != 2 or header["shape"][0] != expected_rows:
                        raise ValueError("Final embeddings do not align with the panel")
                    if name in headers:
                        if headers[name] != header:
                            raise ValueError("Checkpoint/final embedding dtype or shape differs")
                        with (cache / "checkpoints" / f"{name}.npy").open("rb") as checkpoint:
                            npy_header(checkpoint)
                            matching_payload(checkpoint, final, header["bytes"])
                            finished_array(checkpoint)
                    else:
                        consume(final, header["bytes"])
                    finished_array(final)
            with cross.open("core_order.npy") as order:
                header = npy_header(order)
                if header["shape"] != (query_count, image_count):
                    raise ValueError("CORE ordering does not align with the panel")
                consume(order, header["bytes"])
                finished_array(order)
            with cross.open("cross.npy") as final:
                header = npy_header(final)
                if header["shape"] != (query_count, 200):
                    raise ValueError("Only complete depth-200 development caches are eligible")
                streams = {}
                try:
                    for name, columns in zip(CROSS_PARTS, (50, 150), strict=True):
                        if name not in headers:
                            continue
                        item = headers[name]
                        if item["dtype"] != header["dtype"] or item["shape"] != (query_count * columns,):
                            raise ValueError("Cross checkpoint shape or dtype differs from the final array")
                        stream = (cache / "checkpoints" / f"{name}.npy").open("rb")
                        npy_header(stream)
                        streams[name] = stream
                    for _ in range(query_count):
                        for name, columns in zip(CROSS_PARTS, (50, 150), strict=True):
                            size = columns * header["itemsize"]
                            matching_payload(streams[name], final, size) if name in streams else consume(final, size)
                    for stream in streams.values():
                        finished_array(stream)
                    finished_array(final)
                finally:
                    for stream in streams.values():
                        stream.close()
        receipt_path = Path(receipt_path).absolute() if receipt_path else cache / f"cleanup-receipt-{time.time_ns()}-{os.getpid()}.json"
        if receipt_path.parent != cache or receipt_path.exists() or receipt_path.is_symlink():
            raise ValueError("Cleanup receipt must be a new JSON file directly inside this cache")
        receipt = {"schema": 1, "owner": "[codex]", "created_at": now(), "cache": str(cache),
                   "mode": "apply" if apply else "dry-run", "status": "verified",
                   "verification": "NPZ CRC, panel shapes, complete checkpoint states and exact numeric payload bytes; writer lock held",
                   "retained": {name: {"sha256": sha256(cache / name), "bytes": (cache / name).stat().st_size}
                                for name in mandatory if name != "writer.lock"},
                   "files": paths, "verified_duplicate_bytes": sum(row["bytes"] for row in paths), "removed_bytes": 0}
        atomic_json(receipt_path, receipt)
        if apply:
            try:
                for row in paths:
                    path = Path(row["path"])
                    info = owned_regular(path, root=root)
                    if (info.st_ino, info.st_size, info.st_mtime_ns) != (row["inode"], row["bytes"], row["mtime_ns"]) or sha256(path) != row["sha256"]:
                        raise ValueError("Checkpoint changed after verification; cleanup stopped")
                    path.unlink()
                    row["status"] = "removed"
                    receipt["removed_bytes"] += row["bytes"]
                    atomic_json(receipt_path, receipt)
            except BaseException as error:
                receipt.update(status="failed", error=f"{type(error).__name__}: {error}", finished_at=now())
                atomic_json(receipt_path, receipt)
                raise
        receipt.update(status="completed" if apply else "dry-run", finished_at=now())
        atomic_json(receipt_path, receipt)
        return receipt_path, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    queue = sub.add_parser("queue", help="Wait for a quiet, sufficiently free local GPU")
    queue.add_argument("--queue", type=Path, required=True)
    queue.add_argument("--gpus", help="Comma-separated physical GPU indices, e.g. 0,1,2,3")
    queue.add_argument("--min-free-mib", type=int, default=14336)
    queue.add_argument("--max-utilization", type=int, default=20)
    queue.add_argument("--confirmation-seconds", type=float, default=5)
    queue.add_argument("--retry-seconds", type=float, default=30)
    queue.add_argument("--cwd", type=Path)
    queue.add_argument("command", nargs=argparse.REMAINDER)
    cleanup = sub.add_parser("cleanup-cache", help="Verify redundant checkpoint copies; dry-run unless --apply")
    cleanup.add_argument("cache", type=Path)
    cleanup.add_argument("--apply", action="store_true")
    cleanup.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    if args.action == "queue":
        command = args.command[1:] if args.command[:1] == ["--"] else args.command
        try:
            gpus = {int(value) for value in args.gpus.split(",")} if args.gpus else None
            if gpus is not None and (not gpus or min(gpus) < 0):
                raise ValueError("GPU indices must be nonnegative")
        except ValueError as error:
            parser.error(str(error))
        raise SystemExit(run_queued(command, args.queue, gpus=gpus, min_free_mib=args.min_free_mib,
            max_utilization=args.max_utilization, confirmation_seconds=args.confirmation_seconds,
            retry_seconds=args.retry_seconds, cwd=args.cwd))
    path, receipt = cleanup_cache(args.cache, apply=args.apply, receipt_path=args.receipt)
    print(json.dumps({"receipt": str(path), "status": receipt["status"],
                      "verified_duplicate_bytes": receipt["verified_duplicate_bytes"], "removed_bytes": receipt["removed_bytes"]}))


if __name__ == "__main__":
    main()
