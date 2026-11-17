"""組出評測集 data/eval.jsonl（含 split 欄位 dev/test）。僅用標準庫 + pandas。

來源：MATH-500、Minerva、OlympiadBench（只留純文字、單一答案）、AMC23、AIME24、AIME25。
split：以固定 seed 隨機取 400 題為 dev（之後蒸餾調參／灰區加碼用），其餘為 test（判準 0／1 用）。
用法：python -I prepare_data.py --out data/eval.jsonl
"""
import argparse, glob, json, os, random
import pandas as pd
from huggingface_hub import hf_hub_download as dl

SPLIT_SEED = 20261114
DEV_N = 400


def jl(p):
    return pd.read_json(p, lines=True)


def rows():
    out = []
    df = jl(dl("HuggingFaceH4/MATH-500", "test.jsonl", repo_type="dataset"))
    for i, r in df.iterrows():
        out.append(("math500", f"math500-{i}", r["problem"], str(r["answer"])))
    df = jl(dl("math-ai/minervamath", "test.jsonl", repo_type="dataset"))
    for i, r in df.iterrows():
        out.append(("minerva", f"minerva-{i}", r["question"], str(r["answer"])))
    df = pd.read_parquet(dl("math-ai/olympiadbench", "test.parquet", repo_type="dataset"))
    for i, r in df.iterrows():
        fa = r["final_answer"]
        fa = list(fa) if fa is not None else []
        if r["modality"] != "Text-only" or len(fa) != 1 or str(r["is_multiple_answer"]).lower() == "true":
            continue
        a = str(fa[0]).strip().strip("$")
        out.append(("olympiad", f"olympiad-{r['id']}", r["question"], a))
    df = pd.read_parquet(dl("math-ai/amc23", "test-00000-of-00001.parquet", repo_type="dataset"))
    for i, r in df.iterrows():
        a = float(r["answer"])
        out.append(("amc23", f"amc23-{i}", r["question"], str(int(a)) if a == int(a) else str(a)))
    df = pd.read_parquet(dl("math-ai/aime24", "test-00000-of-00001.parquet", repo_type="dataset"))
    for i, r in df.iterrows():  # 答案在 solution 欄的 \\boxed{...}
        sol = str(r["solution"])
        j = sol.rfind("\\boxed{")
        a = sol[j + 7: sol.rfind("}")]
        out.append(("aime24", f"aime24-{i}", r["problem"], a))
    df = jl(dl("math-ai/aime25", "test.jsonl", repo_type="dataset"))
    qc = "problem" if "problem" in df.columns else "question"
    for i, r in df.iterrows():
        out.append(("aime25", f"aime25-{i}", r[qc], str(r["answer"])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/eval.jsonl")
    a = ap.parse_args()
    rs = rows()
    ids = [r[1] for r in rs]
    assert len(set(ids)) == len(ids)
    rng = random.Random(SPLIT_SEED)
    dev = set(rng.sample(sorted(ids), DEV_N))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    cnt = {}
    with open(a.out, "w") as f:
        for src, id_, q, ans in rs:
            sp = "dev" if id_ in dev else "test"
            cnt[(src, sp)] = cnt.get((src, sp), 0) + 1
            f.write(json.dumps({"id": id_, "source": src, "question": q, "answer": ans, "split": sp}, ensure_ascii=False) + "\n")
    print("total", len(rs), "test", sum(v for (s, p), v in cnt.items() if p == "test"))
    for k in sorted(cnt):
        print(k, cnt[k])


if __name__ == "__main__":
    main()
