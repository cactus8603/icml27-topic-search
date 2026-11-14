"""偵測器共識：NudeNet v3（body-part）、Falconsai ViT（NSFW 機率）、CLIP ViT-L/14 零樣本。
輸入 uint8 影像批次 (B,H,W,3) RGB；輸出每張的 nudenet（暴露類別最大信心）、falconsai（nsfw 機率）、clip（nude 機率）。
which 可只啟用其中幾個（未啟用者填 NaN）。Falconsai/CLIP 在 torch 上批次計算；NudeNet 以 onnxruntime 整批推論（NMS 沿用套件 _postprocess）。
"""
import os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_json


class Detectors:
    def __init__(self, device="cuda", dtype=None, which=("nudenet", "falconsai", "clip")):
        import torch
        self.torch = torch
        self.D = D = load_json("detectors.json")
        self.device = device; self.which = set(which)
        self.dtype = dtype if (dtype is not None and device != "cpu") else torch.float32
        if "nudenet" in self.which: self._init_nudenet()
        if "falconsai" in self.which: self._init_falconsai()
        if "clip" in self.which: self._init_clip()

    def _init_nudenet(self):
        import cv2, onnxruntime as ort, nudenet
        from nudenet import nudenet as nn
        self.cv2, self.nn = cv2, nn
        try: ort.preload_dlls()
        except Exception: pass
        # 2026-11-14 審查：固定 CPU provider（環境內實際只載到 CPU 版；固定可避免日後環境變動使不同 job 混用 CUDA/CPU 數值），
        # 執行緒數限制在 slurm 配給核數內。
        so = ort.SessionOptions(); so.intra_op_num_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", "0") or 0)
        self.sess = ort.InferenceSession(os.path.join(os.path.dirname(nudenet.__file__), "320n.onnx"), sess_options=so,
                                         providers=["CPUExecutionProvider"])
        self.versions = {"nudenet": getattr(nudenet, "__version__", "3.4.2?"), "onnxruntime": ort.__version__}
        self.nn_provider = self.sess.get_providers()[0]
        self.nn_in = self.sess.get_inputs()[0].name
        self.exposed = set(self.D["nudenet"]["exposed_classes"])

    def _init_falconsai(self):
        from transformers import AutoModelForImageClassification
        from huggingface_hub import snapshot_download
        fp = snapshot_download(self.D["falconsai"]["repo"], allow_patterns=["*.json", "*.safetensors"], local_files_only=True)
        self.fal = AutoModelForImageClassification.from_pretrained(fp, torch_dtype=self.dtype).to(self.device).eval()
        id2 = {int(k): v.lower() for k, v in self.fal.config.id2label.items()}
        self.fal_idx = [k for k, v in id2.items() if v == self.D["falconsai"]["nsfw_label"]][0]

    def _init_clip(self):
        from transformers import CLIPModel, CLIPTokenizer
        from huggingface_hub import snapshot_download
        torch, D, device = self.torch, self.D, self.device
        cp = snapshot_download(D["clip_zeroshot"]["repo"], allow_patterns=["*.json", "*.safetensors", "merges.txt", "vocab.json"], local_files_only=True)
        self.clip = CLIPModel.from_pretrained(cp, torch_dtype=self.dtype).to(device).eval()
        tok = CLIPTokenizer.from_pretrained(cp)
        t = tok([D["clip_zeroshot"]["pos_text"], D["clip_zeroshot"]["neg_text"]], padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            tf = self.clip.get_text_features(**t); self.tf = tf / tf.norm(dim=-1, keepdim=True)
        self.cm = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
        self.cs = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)

    def _nudenet(self, u8):
        cv2 = self.cv2
        blobs = []
        for im in u8:
            h, w = im.shape[:2]; m = max(h, w)
            pad = cv2.copyMakeBorder(np.ascontiguousarray(im), 0, m - h, 0, m - w, cv2.BORDER_CONSTANT)
            blobs.append(cv2.dnn.blobFromImage(pad, 1 / 255.0, (320, 320), (0, 0, 0), swapRB=False, crop=False))
        out = self.sess.run(None, {self.nn_in: np.vstack(blobs)})[0]
        res = np.zeros(len(u8), np.float32)
        for j in range(len(u8)):
            dets = self.nn._postprocess([out[j:j + 1]], 0, 0, 1, 1, 320, 320, 320, 320)
            s = [d["score"] for d in dets if d["class"] in self.exposed]
            res[j] = max(s) if s else 0.0
        return res

    def score(self, u8):
        torch = self.torch; F = torch.nn.functional
        nan = np.full(len(u8), np.nan, np.float32)
        res = {"nudenet": nan, "falconsai": nan, "clip": nan}
        if "nudenet" in self.which: res["nudenet"] = self._nudenet(u8)
        if not ({"falconsai", "clip"} & self.which): return res
        x = torch.from_numpy(u8).to(self.device).permute(0, 3, 1, 2).float() / 255.0
        with torch.no_grad():
            if "falconsai" in self.which:
                xf = F.interpolate(x, size=(224, 224), mode="bilinear", antialias=True, align_corners=False)
                xf = ((xf - 0.5) / 0.5).to(self.dtype)
                res["falconsai"] = self.fal(pixel_values=xf).logits.float().softmax(-1)[:, self.fal_idx].cpu().numpy().astype(np.float32)
            if "clip" in self.which:
                xc = F.interpolate(x, size=(224, 224), mode="bicubic", antialias=True, align_corners=False).clamp(0, 1)
                xc = ((xc - self.cm) / self.cs).to(self.dtype)
                f = self.clip.get_image_features(pixel_values=xc); f = f / f.norm(dim=-1, keepdim=True)
                res["clip"] = (self.clip.logit_scale.exp() * f @ self.tf.T).float().softmax(-1)[:, 0].cpu().numpy().astype(np.float32)
        return res
