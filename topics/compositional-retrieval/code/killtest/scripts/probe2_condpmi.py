"""第二探針（停損後）：整句條件 PMI  s~(T) = tau*s(I,T) - log mean_{I' in pool} exp(tau*s(I',T))。
T 為整句（full / anchor / half），不拆單元。池 = HT 三子集 1379 張唯一影像（與 PREREG §3 相同）。
本檔含純函式（unit_tests_probe2.py 會測）與 CPU 分析 CLI（讀 results/ht/*.npz 與 results/probe2/ctrl_*.npz）。
主判定須等協調者 commit 凍結 PREREG_probe2.md 之後才可執行。
用法：python scripts/probe2_condpmi.py --model clip_b32|cs_clip [--out results/probe2/analysis_<model>.json]
"""
import argparse
import difflib
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (EXP, SUBSETS, _tokens, cor_flags, jdump, logmeanexp, paired_bootstrap_diff, bootstrap_ci)

P2_PARAMS = EXP / "configs" / "params_probe2.json"


def p2params():
    return json.loads(P2_PARAMS.read_text())


# ------------------------------------------------------------ 核心評分

def stilde(img, txt, pool, tau):
    """逐列 s~(T)：img (N,d)、txt (N,d)（第 i 個影像配第 i 個文字）、pool (P,d)。
    s~ = tau*<img,txt> - log mean_{p in pool} exp(tau*<p,txt>)。"""
    s = tau * np.einsum("nd,nd->n", img, txt)
    pool_s = tau * (txt @ pool.T)
    return s - logmeanexp(pool_s, axis=1)


def pairwise_acc(s_pos, s_neg):
    """P(s(A+) > s(A-))，嚴格大於（平手算錯），回傳百分比。"""
    return float((np.asarray(s_pos) > np.asarray(s_neg)).mean() * 100)


def cor_pct(s_full, s_anchor, s_half):
    return float(cor_flags(s_full, s_anchor, s_half).mean() * 100)


# ------------------------------------------------------------ 控制組

_PUNCT = re.compile(r"\s+([,.;:!?])")


def replace_detail(anchor, extended, new_detail):
    """把 extended 相對 anchor 的「新增／替換」字詞段換成 new_detail（第一段含字母數字的片段換成 new_detail，
    其餘含字母數字的新增段刪除；純標點段與 equal 段保留）。若沒有任何新增段，把 new_detail 接在句尾。
    回傳小寫、以空白連接的字串（標點前的空白去掉）。"""
    a, e = _tokens(anchor), _tokens(extended)
    sm = difflib.SequenceMatcher(a=a, b=e, autojunk=False)
    out, placed = [], False
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            out.extend(e[j1:j2])
        elif op in ("insert", "replace"):
            seg = e[j1:j2]
            if any(re.match(r"[A-Za-z0-9]", t) for t in seg):
                if not placed:
                    out.extend(new_detail.split())
                    placed = True
            else:
                out.extend(seg)
    if not placed:
        out.extend(new_detail.split())
    return _PUNCT.sub(r"\1", " ".join(out))


def pos_control_details(dplus_texts, seed, subset_index):
    """每列從「別列」抽一個 d+ 文字當替換細節；不得與本列 d+ 相同（字串比較），且不得為空。
    回傳 list[str]。無合格候選時 raise。"""
    rng = np.random.default_rng([seed, subset_index])
    dp = list(dplus_texts)
    n = len(dp)
    out = []
    for i in range(n):
        cand = [j for j in range(n) if j != i and dp[j].strip() and dp[j] != dp[i]]
        if not cand:
            raise ValueError(f"列 {i} 沒有合格的替換候選")
        out.append(dp[cand[int(rng.integers(0, len(cand)))]])
    return out


def neg_control_perm(n, seed, subset_index):
    """影像打亂的排列（無不動點）。"""
    rng = np.random.default_rng([seed, subset_index])
    perm = rng.permutation(n)
    for i in range(n):
        if perm[i] == i:
            j = (i + 1) % n
            perm[i], perm[j] = perm[j], perm[i]
    return perm


# ------------------------------------------------------------ 分析

def _pool_all(zs):
    seen, vecs = set(), []
    for sub in SUBSETS:
        if sub in zs:
            for sha, v in zip(zs[sub]["sha"], zs[sub]["img"]):
                if sha not in seen:
                    seen.add(sha)
                    vecs.append(v)
    return np.stack(vecs)


def analyze_subset(z, pool, ctrl, p, sub_idx, pool5000=None):
    I, tau = z["img"], float(z["tau"])
    pr = p["primary"]
    nb, seed, a1 = pr["n_boot"], pr["boot_seed"], pr["alpha_one_sided"]
    st = {k: stilde(I, z[k], pool, tau) for k in ("full", "anchor", "half")}
    ho = {k: np.einsum("nd,nd->n", I, z[k]) for k in ("full", "anchor", "half")}
    f_s = cor_flags(st["full"], st["anchor"], st["half"])
    f_h = cor_flags(ho["full"], ho["anchor"], ho["half"])
    out = {"n": int(len(I)),
           "cor_stilde": bootstrap_ci(f_s, nb, seed, alpha_one_sided=a1),
           "cor_holistic": bootstrap_ci(f_h, nb, seed, alpha_one_sided=a1),
           "paired_stilde_minus_holistic_pp": paired_bootstrap_diff(f_s, f_h, nb, seed + 2, alpha_one_sided=a1),
           "pairwise_acc_stilde": pairwise_acc(st["full"], st["half"]),
           "pairwise_acc_holistic": pairwise_acc(ho["full"], ho["half"]),
           "ht_acc_stilde": pairwise_acc(st["anchor"], st["half"]),
           "ht_acc_holistic": pairwise_acc(ho["anchor"], ho["half"])}
    # 構造控制
    out["pos_control_cor"] = None
    if ctrl is not None and "ctrl_pos" in ctrl:
        sc = stilde(I, ctrl["ctrl_pos"], pool, tau)
        out["pos_control_cor"] = cor_pct(st["full"], st["anchor"], sc)
    perm = neg_control_perm(len(I), p["seeds"]["neg_control"], sub_idx)
    Is = I[perm]
    sn = {k: stilde(Is, z[k], pool, tau) for k in ("full", "anchor", "half")}
    out["neg_control_cor"] = cor_pct(sn["full"], sn["anchor"], sn["half"])
    # 描述：COCO 5000 池
    if pool5000 is not None:
        s5 = {k: stilde(I, z[k], pool5000, tau) for k in ("full", "anchor", "half")}
        f5 = cor_flags(s5["full"], s5["anchor"], s5["half"])
        out["coco5000_pool"] = {"cor_stilde": bootstrap_ci(f5, nb, seed, alpha_one_sided=a1),
                                "paired_vs_holistic_pp": paired_bootstrap_diff(f5, f_h, nb, seed + 2, alpha_one_sided=a1),
                                "pairwise_acc": pairwise_acc(s5["full"], s5["half"])}
    return out


def analyze(model, ht_dir, p2_dir, p):
    zs = {s: dict(np.load(ht_dir / f"{model}_{s}.npz", allow_pickle=False)) for s in SUBSETS
          if (ht_dir / f"{model}_{s}.npz").exists()}
    pool = _pool_all(zs)
    p5 = p2_dir / f"pool_coco5000_{model}.npz"
    pool5000 = np.load(p5)["img"] if p5.exists() else None
    out = {"model": model, "pool_size": int(len(pool)),
           "pool_coco5000_size": None if pool5000 is None else int(len(pool5000)), "subsets": {}}
    for i, s in enumerate(SUBSETS):
        if s not in zs:
            continue
        cp = p2_dir / f"ctrl_{model}_{s}.npz"
        ctrl = dict(np.load(cp, allow_pickle=False)) if cp.exists() else None
        out["subsets"][s] = analyze_subset(zs[s], pool, ctrl, p, i, pool5000)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ht-dir", default=str(EXP / "results" / "ht"))
    ap.add_argument("--p2-dir", default=str(EXP / "results" / "probe2"))
    ap.add_argument("--out")
    a = ap.parse_args()
    p2 = Path(a.p2_dir)
    out = analyze(a.model, Path(a.ht_dir), p2, p2params())
    jdump(out, a.out or p2 / f"analysis_{a.model}.json")
    for s, v in out["subsets"].items():
        print(a.model, s, "COR s~ %.2f holistic %.2f diff %.2f pair %.1f pos %s neg %.1f" % (
            v["cor_stilde"]["point"], v["cor_holistic"]["point"], v["paired_stilde_minus_holistic_pp"]["point"],
            v["pairwise_acc_stilde"], v["pos_control_cor"], v["neg_control_cor"]))


if __name__ == "__main__":
    main()
