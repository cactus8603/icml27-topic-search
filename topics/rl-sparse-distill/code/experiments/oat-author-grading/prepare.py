"""Fetch bounded, hash-pinned published assets into the shared cluster cache."""

import argparse
import hashlib
import json
from pathlib import Path
import tarfile
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config.json"
LIMIT = 250_000
ARCHIVE = "pylatexenc-2.10.tar.gz"
LATEX_ROOT = "pylatexenc-2.10"
SHARED_TMP = Path("/work/ChihChungHsu/cactus8603/icml/tmp").resolve()


def acquire(root, relative, expected, url=None):
    path = root / relative
    if path.exists():
        with path.open("rb") as stream:
            contents = stream.read(LIMIT + 1)
    elif url:
        with urlopen(url, timeout=25) as stream:
            contents = stream.read(LIMIT + 1)
    else:
        raise FileNotFoundError(path)
    if len(contents) > LIMIT or hashlib.sha256(contents).hexdigest() != expected:
        raise ValueError(f"Published input size or digest mismatch: {relative}")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(contents)


def latex_files(root, install=False):
    prefix = f"{LATEX_ROOT}/pylatexenc/"
    with tarfile.open(root / ARCHIVE) as archive:
        names = set()
        for file in archive:
            if not file.isfile() or not file.name.startswith(prefix):
                continue
            path = root / file.name
            if ".." in Path(file.name).parts or not path.resolve().is_relative_to(root):
                raise ValueError("Unsafe path in the pinned pylatexenc archive")
            names.add(path)
            value = archive.extractfile(file).read()
            if not path.exists() and install:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("xb") as stream:
                    stream.write(value)
            if path.read_bytes() != value:
                raise ValueError("Cached pylatexenc source differs from the pinned release")
        if set(path for path in (root / prefix).rglob("*") if path.is_file()) != names or not names:
            raise ValueError("Cached pylatexenc module does not match the pinned release")


def assets(root, config, install=False):
    if not root.is_relative_to(SHARED_TMP) or root == SHARED_TMP:
        raise ValueError("Place the small author snapshot in a subdirectory of icml/tmp")
    origin = "https://raw.githubusercontent.com/sail-sg/understand-r1-zero/" + config["upstream_revision"]
    for file, expected in config["upstream_sha256"].items():
        acquire(root, file, expected, f"{origin}/{file}" if install else None)
    acquire(root, ARCHIVE, config["pylatexenc_sha256"], config["pylatexenc_url"] if install else None)
    latex_files(root, install)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assets", type=Path)
    args = parser.parse_args()
    assets(args.assets.resolve(), json.loads(CONFIG.read_text()), install=True)
