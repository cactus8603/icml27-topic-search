"""下載資料與權重（login 節點執行即可，不需 GPU）。已存在則跳過／續傳。
- HF: kbora/Half-Truths（dataset, CC-BY-4.0）、kbora/CS-CLIP（model, MIT）
- COCO val2017 影像與 annotations_trainval2017.zip（官方 http://images.cocodataset.org，curl -C - 續傳）
- OpenAI CLIP ViT-B/32：open_clip 會從 HF hub（timm/vit_base_patch32_clip_224.openai）取得，也在此預先下載；
  openai-clip 套件（僅供 G1 交叉核對）會在首次使用時自 openaipublic 下載到 crosscheck 的 HOME 暫存。
用法：HF_HOME=... python scripts/download_assets.py
紀錄 revision 到 configs/assets_lock.json。
"""
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import COCO_DIR, EXP, HF_HOME, jdump

os.environ.setdefault("HF_HOME", HF_HOME)


def main():
    from huggingface_hub import HfApi, snapshot_download
    lock = {}
    for repo, kind in [("kbora/Half-Truths", "dataset"), ("kbora/CS-CLIP", "model"),
                       ("timm/vit_base_patch32_clip_224.openai", "model")]:
        path = snapshot_download(repo, repo_type=kind)
        lock[repo] = {"snapshot": path, "sha": HfApi().repo_info(repo, repo_type=kind).sha}
        print(repo, path)
    COCO_DIR.mkdir(parents=True, exist_ok=True)
    for url in ["http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
                "http://images.cocodataset.org/zips/val2017.zip"]:
        subprocess.run(["curl", "-sS", "-C", "-", "-O", url], cwd=COCO_DIR, check=True)
    if not (COCO_DIR / "val2017").exists():
        with zipfile.ZipFile(COCO_DIR / "val2017.zip") as z:
            z.extractall(COCO_DIR)
    lock["coco_val2017_jpgs"] = len(list((COCO_DIR / "val2017").glob("*.jpg")))
    jdump(lock, EXP / "configs" / "assets_lock.json")


if __name__ == "__main__":
    main()
