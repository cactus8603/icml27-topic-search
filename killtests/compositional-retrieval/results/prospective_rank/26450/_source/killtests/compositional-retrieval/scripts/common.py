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

def load_model(name, device, cs_clip_path=None):
    """name: clip_b32（OpenAI CLIP ViT-B/32）或 cs_clip（CS-CLIP B/32）。
    CS-CLIP 的載入方式照 ext/CS-CLIP/scripts/evaluate_half_truth.py：
    open_clip 'ViT-B-32' 空模型 + strict load_state_dict。
    OpenAI CLIP 用 open_clip 的 'ViT-B-32-quickgelu' + pretrained='openai'（與 openai-clip 套件同權重，G1 另有交叉核對）。
    """
    import open_clip
    import torch
    if name == "cs_clip":
        model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", device=device)
        ck = torch.load(cs_clip_path or cs_clip_ckpt(), map_location="cpu", weights_only=True)
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


def _one_sided(means, alpha_one_sided):
    """單尾 alpha：回傳 (下界, 上界) = (alpha, 1-alpha) 百分位。None -> 沿用 2.5/97.5。"""
    a = 0.025 if alpha_one_sided is None else alpha_one_sided
    lo, hi = np.percentile(means, [100 * a, 100 * (1 - a)])
    return float(lo), float(hi)


def bootstrap_ci(flags, n_boot, seed, alpha_two_sided=0.05, alpha_one_sided=None, groups=None):
    """對 0/1 旗標做 bootstrap。groups=None 時為列層級；給 groups（每列所屬影像等的群組標籤）時為群集 bootstrap
    （以群組為單位重抽，群組內的列一併帶入；統計量為被抽中列的平均）。
    回傳 dict(point, lo, hi, se, lo_1s, hi_1s)。lo/hi 為雙尾 (1-alpha_two_sided) 百分位區間（預設 2.5/97.5，G3 半寬用）；
    lo_1s/hi_1s 為單尾 alpha_one_sided 的 (alpha, 1-alpha) 百分位（判準用；P1 為 0.0125 即 1.25/98.75）。"""
    f = np.asarray(flags, dtype=float)
    means = _boot_means(f, n_boot, seed, groups) * 100
    lo, hi = np.percentile(means, [100 * alpha_two_sided / 2, 100 * (1 - alpha_two_sided / 2)])
    lo1, hi1 = _one_sided(means, alpha_one_sided)
    return {"n": int(len(f)), "point": float(f.mean() * 100), "lo": float(lo), "hi": float(hi),
            "se": float(means.std(ddof=1)), "lo_1s": lo1, "hi_1s": hi1}


def paired_bootstrap_diff(flags_a, flags_b, n_boot, seed, alpha_two_sided=0.05, alpha_one_sided=None, groups=None):
    a, b = np.asarray(flags_a, float), np.asarray(flags_b, float)
    d = a - b
    means = _boot_means(d, n_boot, seed, groups) * 100
    lo, hi = np.percentile(means, [100 * alpha_two_sided / 2, 100 * (1 - alpha_two_sided / 2)])
    lo1, hi1 = _one_sided(means, alpha_one_sided)
    return {"n": int(len(d)), "point": float(d.mean() * 100), "lo": float(lo), "hi": float(hi), "lo_1s": lo1, "hi_1s": hi1}


def _group_index(groups):
    """groups -> (唯一群組數 G, 每群組的列索引 list)。"""
    g = np.asarray(groups)
    uniq, inv = np.unique(g, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    bounds = np.r_[0, np.cumsum(np.bincount(inv, minlength=len(uniq)))]
    return len(uniq), [order[bounds[i]:bounds[i + 1]] for i in range(len(uniq))]


def _boot_means(f, n_boot, seed, groups=None):
    """回傳 n_boot 個 bootstrap 平均（0–1 尺度）。群集：以群組為單位有放回重抽 G 個群組，
    平均 = Σ群組內 f 之和 / Σ群組大小（等價於把被抽中的列全部合併取平均）。"""
    rng = np.random.default_rng(seed)
    if groups is None:
        idx = rng.integers(0, len(f), size=(n_boot, len(f)))
        return f[idx].mean(axis=1)
    G, members = _group_index(groups)
    gsum = np.array([f[m].sum() for m in members])
    gcnt = np.array([len(m) for m in members], float)
    idx = rng.integers(0, G, size=(n_boot, G))
    return gsum[idx].sum(axis=1) / gcnt[idx].sum(axis=1)


def jdump(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
