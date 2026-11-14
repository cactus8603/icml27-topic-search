"""共用函式：路徑、模型載入、COR／LLR 計算、bootstrap、細節字串抽取。

所有統計函式都是純 numpy，可在 CPU 上以已存的 embedding 重算，不需 GPU。
"""
import difflib
import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

EXP = Path(os.environ.get("EXP", Path(__file__).resolve().parents[1]))
ICML = Path("/work/ChihChungHsu/cactus8603/icml")
HF_HOME = os.environ.get("HF_HOME", str(ICML / "hf_home"))
EXTRA_SITE = ICML / "envs" / "t2irare_extra_cscomp"   # open_clip / pyarrow / torchvision 等補充套件
COCO_DIR = ICML / "data" / "coco2017"
SUBSETS = ["coco_qwen", "coco_mistral", "cc3m_qwen"]
PARAMS_PATH = EXP / "configs" / "params.json"

if EXTRA_SITE.exists() and str(EXTRA_SITE) not in sys.path:
    sys.path.insert(0, str(EXTRA_SITE))


def params():
    return json.loads(PARAMS_PATH.read_text())


# ---------------------------------------------------------------- 資料

def ht_parquet(subset):
    hits = glob.glob(f"{HF_HOME}/hub/datasets--kbora--Half-Truths/snapshots/*/data/{subset}.parquet")
    if not hits:
        raise FileNotFoundError(f"找不到 Half-Truths {subset}.parquet；先執行 scripts/download_assets.py")
    return Path(hits[0])


def load_ht(subset, limit=None):
    import pyarrow.parquet as pq
    t = pq.read_table(ht_parquet(subset))
    rows = t.to_pylist()
    return rows[:limit] if limit else rows


def cs_clip_ckpt():
    hits = glob.glob(f"{HF_HOME}/hub/models--kbora--CS-CLIP/snapshots/*/open_clip_pytorch_model.bin")
    if not hits:
        raise FileNotFoundError("找不到 CS-CLIP 權重；先執行 scripts/download_assets.py")
    return hits[0]


# ---------------------------------------------------------------- 模型

def load_model(name, device):
    """name: clip_b32（OpenAI CLIP ViT-B/32）或 cs_clip（CS-CLIP B/32）。
    CS-CLIP 的載入方式照 ext/CS-CLIP/scripts/evaluate_half_truth.py：
    open_clip 'ViT-B-32' 空模型 + strict load_state_dict。
    OpenAI CLIP 用 open_clip 的 'ViT-B-32-quickgelu' + pretrained='openai'（與 openai-clip 套件同權重，G1 另有交叉核對）。
    """
    import open_clip
    import torch
    if name == "cs_clip":
        model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", device=device)
        ck = torch.load(cs_clip_ckpt(), map_location="cpu", weights_only=True)
        state = ck.get("model_state_dict", ck.get("state_dict", ck))
        state = {k.replace("module.", "", 1) if k.startswith("module.") else k: v
                 for k, v in state.items() if not k.startswith("head.")}
        model.load_state_dict(state, strict=True)
    elif name == "clip_b32":
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32-quickgelu", pretrained="openai", device=device)
    else:
        raise ValueError(name)
    model.float().eval()
    tok = open_clip.get_tokenizer("ViT-B-32")
    return model, preprocess, tok


def encode_texts(model, tok, texts, device, bs=256):
    import torch
    import torch.nn.functional as F
    out = []
    with torch.inference_mode():
        for i in range(0, len(texts), bs):
            t = tok(texts[i:i + bs]).to(device)
            out.append(F.normalize(model.encode_text(t), dim=-1).float().cpu())
    return torch.cat(out).numpy()


def encode_images(model, preprocess, opener_list, device, bs=64):
    """opener_list: callables，各自回傳 PIL.Image（避免一次載入全部影像）。"""
    import torch
    import torch.nn.functional as F
    out = []
    with torch.inference_mode():
        for i in range(0, len(opener_list), bs):
            ims = [preprocess(o().convert("RGB")) for o in opener_list[i:i + bs]]
            x = torch.stack(ims).to(device)
            out.append(F.normalize(model.encode_image(x), dim=-1).float().cpu())
    return torch.cat(out).numpy()


# ---------------------------------------------------------------- 細節字串抽取（PREREG 規則 R-DIFF）

_tok_re = re.compile(r"[A-Za-z0-9']+|[^\sA-Za-z0-9']")


def _tokens(s):
    return _tok_re.findall(s.lower())


def extract_detail(anchor, extended):
    """以 token 層級 difflib 取得 extended 相對於 anchor「新增或替換」的片段。
    回傳 (detail_text, ok)。ok=True 表示 anchor 的 token 全數保留（只有插入／替換 extended 端）。
    detail_text 以原字串中對應 token 的小寫 token 以空白連接（標點略去）。
    規則固定、不看影像、不看模型分數。
    """
    a, e = _tokens(anchor), _tokens(extended)
    sm = difflib.SequenceMatcher(a=a, b=e, autojunk=False)
    pieces, ok = [], True
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op in ("insert", "replace"):
            pieces.extend(t for t in e[j1:j2] if re.match(r"[A-Za-z0-9]", t))
        if op in ("delete", "replace") and any(re.match(r"[A-Za-z0-9]", t) for t in a[i1:i2]):
            ok = False   # anchor 的字詞在 extended 中被刪／被換，不是純新增
    return " ".join(pieces), ok


# ---------------------------------------------------------------- 指標

def cor_flags(s_full, s_anchor, s_half):
    """逐列 COR 旗標：s_full > s_anchor > s_half（嚴格大於，平手算錯）。"""
    return (np.asarray(s_full) > np.asarray(s_anchor)) & (np.asarray(s_anchor) > np.asarray(s_half))


def ht_flags(s_anchor, s_half):
    return np.asarray(s_anchor) > np.asarray(s_half)


def logmeanexp(x, axis=0):
    m = x.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(x - m).mean(axis=axis, keepdims=True))).squeeze(axis)


def pmi(img_emb, txt_emb, pool_emb, tau):
    """逐單元對數概似比（PMI）：
        PMI_c(I) = tau*s(I,c) - log mean_{I' in pool} exp(tau*s(I',c))
    即 log p(I|c) - log p(I)，其中 p(I|c) 是文字 c 對池內影像的 softmax、p(I) 為池內均勻分布。
    img_emb: (N,d)；txt_emb: (N,d)，逐列對應（第 i 個影像配第 i 個文字）；pool_emb: (P,d)。
    """
    s = tau * np.einsum("nd,nd->n", img_emb, txt_emb)
    pool_s = tau * (txt_emb @ pool_emb.T)          # (N,P)
    return s - logmeanexp(pool_s, axis=1)


def bootstrap_ci(flags, n_boot, seed, alpha_two_sided=0.05):
    """對 0/1 旗標做列層級 bootstrap。回傳 dict(point, lo, hi, se, lo_1s, hi_1s)。
    lo/hi 為雙尾 (1-alpha) 百分位區間；lo_1s/hi_1s 為單尾（同 alpha/2 的位置，即 alpha/2 單尾顯著水準）。"""
    f = np.asarray(flags, dtype=float)
    n = len(f)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = f[idx].mean(axis=1) * 100
    lo, hi = np.percentile(means, [100 * alpha_two_sided / 2, 100 * (1 - alpha_two_sided / 2)])
    return {"n": int(n), "point": float(f.mean() * 100), "lo": float(lo), "hi": float(hi),
            "se": float(means.std(ddof=1)), "lo_1s": float(lo), "hi_1s": float(hi)}


def paired_bootstrap_diff(flags_a, flags_b, n_boot, seed, alpha_two_sided=0.05):
    a, b = np.asarray(flags_a, float), np.asarray(flags_b, float)
    d = a - b
    n = len(d)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = d[idx].mean(axis=1) * 100
    lo, hi = np.percentile(means, [100 * alpha_two_sided / 2, 100 * (1 - alpha_two_sided / 2)])
    return {"n": int(n), "point": float(d.mean() * 100), "lo": float(lo), "hi": float(hi)}


def jdump(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
