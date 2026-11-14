import sys, json, numpy as np, pandas as pd
sys.path.insert(0, "/work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/neon-lm/scripts")
from common import *
from transformers import AutoTokenizer
n_scale = int(sys.argv[1]) if len(sys.argv) > 1 else 1  # unused
tok = AutoTokenizer.from_pretrained(f"{MODELS}/Qwen2.5-1.5B")
df = pd.read_parquet(f"{ROOT}/icml/data/openwebtext/plain_text/train-00000-of-00080.parquet")
print("docs", len(df))
rng = np.random.RandomState(0)
idx = rng.permutation(len(df))
need = 10000 + 2000 + 5000
rows = []
for i in idx:
    t = df.text.iloc[i]
    if len(t) < 1500: continue
    ids = tok(t[:6000], add_special_tokens=False).input_ids
    if len(ids) < 320: continue
    rows.append((int(i), ids[:320]))
    if len(rows) >= need: break
arr = np.array([r[1] for r in rows], dtype=np.int32)
docid = np.array([r[0] for r in rows])
assert len(set(docid.tolist())) == len(docid)
parts = {"selftrain": (0, 10000), "dev": (10000, 12000), "test": (12000, 17000)}
import os; os.makedirs(f"{RUNS}/data", exist_ok=True)
for k, (a, b) in parts.items():
    np.savez(f"{RUNS}/data/lm_{k}.npz", prompt=arr[a:b, :64], cont=arr[a:b, 64:], docid=docid[a:b])
    print(k, b - a)
# GSM8K
tr = pd.read_parquet(f"{ROOT}/icml/data/gsm8k-hf/main/train-00000-of-00001.parquet")
te = pd.read_parquet(f"{ROOT}/icml/data/gsm8k-hf/main/test-00000-of-00001.parquet")
rng = np.random.RandomState(0); p = rng.permutation(len(tr))
dev = tr.iloc[p[:200]]; st = tr.iloc[p[200:]]
def dump(d, name):
    jdump([dict(q=r.question, a=r.answer) for r in d.itertuples()], f"{RUNS}/data/gsm_{name}.json")
dump(st, "selftrain"); dump(dev, "dev"); dump(te.iloc[:500], "test")
print("gsm", len(st), len(dev), 500)
