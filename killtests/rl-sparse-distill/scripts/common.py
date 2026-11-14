"""共用：設定、資料、prompt、判分、stop token。所有腳本 import 此檔。"""
import gzip, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pylib"))  # math-verify 等純 python 套件（pip --target 裝在此，不動共用 env）
WRB = "/work/ChihChungHsu/cactus8603/icml/ext/WhatRLBrings"
STOP_IDS = [151645, 151643]  # <|im_end|>, <|endoftext|>（Qwen2.5 系列）


def cfg(name):
    with open(os.path.join(ROOT, "configs", name)) as f:
        return json.load(f)


def pairs_cfg():
    return cfg("pairs.json")


def load_data(split=None, path=None):
    path = path or os.path.join(ROOT, "data", "eval.jsonl")
    rows = [json.loads(l) for l in open(path)]
    return [r for r in rows if split in (None, "all", r["split"])]


def build_prompt(question, pc=None):
    pc = pc or pairs_cfg()
    return pc["templates"][pc["template"]].replace("{question}", question)


def shard(rows, i, n):
    return [r for j, r in enumerate(rows) if j % n == i]


def open_gz_write(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return gzip.open(path, "wt", encoding="utf-8")


# ---- 判分（與 WhatRLBrings.wrb.grading 相同規則：最後一個 \boxed{} 經 Math-Verify 對照）----
def last_boxed(text):
    i = text.rfind("\\boxed")
    if i < 0:
        return None
    j = text.find("{", i)
    if j < 0:
        return None
    d = 0
    for k in range(j, len(text)):
        if text[k] == "{":
            d += 1
        elif text[k] == "}":
            d -= 1
            if d == 0:
                return text[j + 1:k]
    return None


_G = {}


def is_correct(text, ref):
    from math_verify import parse, verify
    b = last_boxed(text)
    if b is None:
        return False
    try:
        g = _G.get(ref)
        if g is None:
            g = _G[ref] = parse("\\boxed{%s}" % ref)
        return bool(verify(g, parse("\\boxed{%s}" % b)))
    except Exception:
        return False


def _grade_job(args):
    texts, ref = args
    return [int(is_correct(t, ref)) for t in texts]


def grade_raw(raw_paths, graded_path, workers=8):
    """raw jsonl.gz（每行一題）-> graded json：{id: {correct, finished, ntok, n_repl?}}。"""
    from multiprocessing import Pool
    data = {r["id"]: r for r in load_data("all")}
    recs = []
    for p in raw_paths:
        with gzip.open(p, "rt", encoding="utf-8") as f:
            recs += [json.loads(l) for l in f]
    with Pool(workers) as pool:
        res = pool.map(_grade_job, [(r["texts"], data[r["id"]]["answer"]) for r in recs], chunksize=4)
    out = {}
    for r, c in zip(recs, res):
        out[r["id"]] = {"correct": c, "finished": r["finished"], "ntok": r["ntok"]}
        if "n_repl" in r:
            out[r["id"]]["n_repl"] = r["n_repl"]
    with open(graded_path, "w") as f:
        json.dump(out, f)
    return out
