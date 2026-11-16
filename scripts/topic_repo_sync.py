#!/usr/bin/env python3
"""[claude] Mirror a topic's files from the hub into its topic repo, at one committed hub SHA.

Hub = icml27-topic-search; topic repo = icml27-<topic>. Rules: scripts/topic-repo-map.json.
Only committed files at <hub_sha> are copied (git archive), never uncommitted work.

  python3 scripts/topic_repo_sync.py <topic> <topic_repo_dir> [--sha <hub_sha>] [--mode mirror|cutover]

mirror  (default): the hub stays canonical; README gets a read-only mirror header.
cutover          : final sync; the README header says the topic repo is now canonical.
Mirrored paths are replaced on every run (files deleted in the hub disappear here too); README.md header,
LOCKS.md and .gitignore of the topic repo are kept as they are. Commit and push are left to the caller:
  cd <topic_repo_dir> && git add -A -f . && git commit -m "[who] Mirror/Cutover <topic> from icml27-topic-search@<sha>" && git push
(-f because the hub tracks a few files that the topic's own .gitignore files would otherwise hide.)
Markdown links: a target that is also mirrored becomes a relative link in the new layout; any other hub
target becomes https://github.com/cactus8603/icml27-topic-search/blob/<sha>/<path>.
"""
import argparse
import fnmatch
import io
import json
import os
import posixpath
import re
import shutil
import subprocess
import tarfile
from pathlib import Path

HUB_URL = "https://github.com/cactus8603/icml27-topic-search"
LINK = re.compile(r"(\]\()([^)\s]+)(\))")
MANAGED = {".git", "LOCKS.md", ".gitignore"}


def hub_files(hub, sha):
    out = subprocess.run(["git", "-C", str(hub), "ls-tree", "-r", "--name-only", sha], check=True,
                         capture_output=True, text=True).stdout.split("\n")
    return [p for p in out if p]


def mapper(rules):
    def dest(path):
        for pat, strip, add in rules:
            if fnmatch.fnmatch(path, pat) and path.startswith(strip):
                return add + path[len(strip):]
        return None
    return dest


def rewrite(md_hub_path, md_new_path, text, dest, sha, mirrored):
    def repl(m):
        target = m.group(2)
        if re.match(r"^[a-z][a-z0-9+.-]*:", target) or target.startswith("#"):
            return m.group(0)
        path, sep, anchor = target.partition("#")
        if not path:
            return m.group(0)
        hub_target = posixpath.normpath(posixpath.join(posixpath.dirname(md_hub_path), path))
        if hub_target.startswith(".."):
            return m.group(0)
        new = dest(hub_target) if hub_target in mirrored else None
        if new is None and any(p.startswith(hub_target + "/") for p in mirrored):
            probe = dest(hub_target + "/__dir__")            # directory link: map the directory itself
            new = posixpath.dirname(probe) if probe is not None else None
        if new is not None:
            rel = posixpath.relpath(new or ".", posixpath.dirname(md_new_path) or ".")
            return m.group(1) + rel + (sep + anchor if sep else "") + m.group(3)
        return m.group(1) + f"{HUB_URL}/blob/{sha}/{hub_target}" + (sep + anchor if sep else "") + m.group(3)
    return LINK.sub(repl, text)


def header(topic, sha, mode):
    rules = (f"> 本 repo 遵守中樞的[協作規則]({HUB_URL}/blob/main/COORDINATION.md)（Claude Code × Codex），規則適用於此 repo。"
             f"GPU 送件一律登記在中樞的 [`killtests/SUBMITTED.log`]({HUB_URL}/blob/main/killtests/SUBMITTED.log)。\n>\n")
    if mode == "mirror":
        state = (f"> **鏡像（唯讀）**：同步自中樞 `icml27-topic-search@{sha[:7]}`。這一輪結束前**正本仍在中樞**，請不要在這裡修改；"
                 f"結束後用中樞的 `scripts/topic_repo_sync.py --mode cutover` 做最後一次同步並切換正本。\n\n")
    else:
        state = (f"> **正本**：2026-11 自中樞 `icml27-topic-search@{sha[:7]}` 切換到本 repo。中樞原路徑為封存。"
                 f"程式內寫死的中樞路徑在切換時需改到本 repo。\n\n")
    return rules + state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("topic")
    ap.add_argument("repo_dir")
    ap.add_argument("--sha", default="origin/main")
    ap.add_argument("--mode", choices=("mirror", "cutover"), default="mirror")
    ap.add_argument("--hub", default=str(Path(__file__).resolve().parents[1]))
    a = ap.parse_args()
    hub, repo = Path(a.hub), Path(a.repo_dir)
    sha = subprocess.run(["git", "-C", str(hub), "rev-parse", a.sha], check=True, capture_output=True,
                         text=True).stdout.strip()
    rules = json.loads((hub / "scripts/topic-repo-map.json").read_text())[a.topic]
    dest = mapper(rules)
    files = [p for p in hub_files(hub, sha) if dest(p) is not None]
    if not files:
        raise SystemExit("no files matched")
    for item in repo.iterdir():            # replace everything except managed files
        if item.name in MANAGED:
            continue
        shutil.rmtree(item) if item.is_dir() else item.unlink()
    blob = subprocess.run(["git", "-C", str(hub), "archive", sha, "--", *files], check=True,
                          capture_output=True).stdout
    with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            data = tar.extractfile(member).read()
            new = dest(member.name)
            out = repo / new
            out.parent.mkdir(parents=True, exist_ok=True)
            if new.endswith(".md"):
                data = rewrite(member.name, new, data.decode("utf-8"), dest, sha, files).encode("utf-8")
            if new == "README.md":
                data = header(a.topic, sha, a.mode).encode("utf-8") + data
            out.write_bytes(data)
            os.chmod(out, member.mode & 0o777 or 0o644)
    if not (repo / "README.md").exists():
        (repo / "README.md").write_text(header(a.topic, sha, a.mode), encoding="utf-8")
    print(json.dumps({"topic": a.topic, "hub_sha": sha, "mode": a.mode, "files": len(files)}))


if __name__ == "__main__":
    main()
