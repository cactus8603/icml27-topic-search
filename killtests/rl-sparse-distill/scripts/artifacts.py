"""Sample, shard and grading evidence for formally evaluated conditions."""

import gzip
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
import re
import sys
import tempfile
from importlib.metadata import version

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, STOP_IDS, cfg, load_data, shard


REPO = Path(ROOT)
SAMPLE_SEED = 1234
HF_SEED_SCHEDULE = "seed + 1000 * shard_index + question_offset"


class ArtifactError(ValueError):
    """An incomplete or unverifiable result."""


def require(ok, message):
    if not ok:
        raise ArtifactError(message)


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def atomic_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
                                     encoding="utf-8", delete=False) as handle:
        tmp = Path(handle.name)
        try:
            json.dump(value, handle, ensure_ascii=False, allow_nan=False)
            handle.close()
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)


@contextmanager
def publish_raw(path):
    """Only one writer may publish a raw shard; partial files are private."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    tmp = Path(name)
    try:
        yield tmp
        os.link(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def dataset_hash(path=None):
    return sha256(path or REPO / "data/eval.jsonl")


def pairs_hash():
    return sha256(REPO / "configs/pairs.json")


def hf_batch_seed(seed, shard_index, question_offset):
    return seed + 1000 * shard_index + question_offset


def grader_receipt():
    math_verify = version("math-verify")
    require(math_verify == "0.9.0", "Math-Verify 版本與凍結設定不符")
    return {"common_sha256": sha256(REPO / "scripts/common.py"), "math_verify": math_verify}


def artifact_receipt(raw, pair, cond, engine, job_id, dtype, context, data=None):
    producer = "gen_vllm.py" if engine == "vllm" else "gen_replace.py"
    return {"raw_sha256": sha256(raw), "pair": pair, "condition": cond, "engine": engine,
            "run_id": os.environ.get("RSD_RUN_ID") or job_id, "job_id": job_id,
            "dtype": dtype, "max_model_len": context, "stop_ids": STOP_IDS,
            "eval_sha256": dataset_hash(data), "pairs_sha256": pairs_hash(),
            "producer_sha256": sha256(REPO / "scripts" / producer),
            "engine_version": version("vllm" if engine == "vllm" else "transformers")}


def parse_shard(value):
    match = re.fullmatch(r"([0-9]+)/([1-9][0-9]*)", value)
    require(match is not None, f"無效分片：{value}")
    i, n = map(int, match.groups())
    require(i < n, f"分片索引超界：{value}")
    return i, n


def model_matches(path, repo_id, snapshot=None):
    parts = Path(path).parts
    if path == repo_id and snapshot is None:
        return True
    return (len(parts) >= 3 and parts[-3] == "models--" + repo_id.replace("/", "--")
            and parts[-2] == "snapshots" and re.fullmatch(r"[0-9a-f]{40}", parts[-1]) is not None
            and (snapshot is None or parts[-1] == snapshot))


def historical_profile(pair):
    return cfg("legacy_artifacts.json").get(pair)


def check_gate_minus1(results, pair, pc):
    profile = historical_profile(pair)
    receipt_path = Path(results) / f"correctness_{pair}.json"
    historical_log = Path(results) / f"correctness_{pair}.log"
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        require(str(receipt.get("job_id", "")).isdigit(), "正確性憑證缺工作編號")
        log_path = Path(results) / f"correctness_{pair}_{receipt['job_id']}.log"
        log = log_path.read_text(encoding="utf-8")
        require_correctness_log(log)
        require(receipt["pair"] == pair and receipt["pairs_sha256"] == pairs_hash()
                and receipt["eval_sha256"] == dataset_hash() and receipt["log_sha256"] == sha256(log_path)
                and receipt["decode_sha256"] == sha256(REPO / "scripts/gen_replace.py")
                and receipt["tests_sha256"] == sha256(REPO / "tests/smoke_cpu.py")
                and receipt["device"] == "cuda" and receipt["dtype"] == "float32" and receipt["job_id"],
                "正確性憑證與目前模型／設定／測試不符")
        for side in ("base", "rl"):
            require(model_matches(receipt[side], pc["pairs"][pair][side]), f"正確性測試 {side} 模型不符")
        return {"job_id": receipt["job_id"], "source": "receipt", "base": receipt["base"], "rl": receipt["rl"],
                "hf_decode_checks": True,
                "geo_mix_checks": all(re.search(rf"^PASS {n}\b", log, re.M) for n in ("T6", "T7a", "T7b", "T8"))}
    require(profile is not None and profile["pairs_sha256"] == pairs_hash() and profile["eval_sha256"] == dataset_hash(),
            "缺正確性測試憑證")
    log = historical_log.read_text(encoding="utf-8")
    require_correctness_log(log)
    proof = profile["gate_minus1"]
    batch = REPO / f"slurm/logs/rsd-correct-{proof['job_id']}.out"
    require(sha256(historical_log) == proof["receipt_log_sha256"] and sha256(batch) == proof["job_log_sha256"],
            "歷史正確性測試 log 不符")
    require(log in batch.read_text(encoding="utf-8"), "歷史正確性測試與 Slurm log 不符")
    return {"job_id": proof["job_id"], "source": "historical_batch_log", "geo_mix_checks": False,
            "hf_decode_checks": False, "snapshots": {
        side: profile[f"{side}_snapshot"] for side in ("base", "rl")}}


def require_correctness_log(log):
    require(all(re.search(rf"^PASS T{n}\b", log, re.M) for n in range(1, 6)) and "ALL PASS" in log.splitlines(),
            "缺完整正確性測試 T1–T5 PASS")


def record_correctness(results, pair, base, rl, job_id):
    root = Path(results)
    out = root / f"correctness_{pair}.json"
    log = root / f"correctness_{pair}_{job_id}.log"
    require_correctness_log(log.read_text(encoding="utf-8"))
    atomic_json(out, {"pair": pair, "base": base, "rl": rl, "job_id": job_id,
                      "log_sha256": sha256(log), "pairs_sha256": pairs_hash(), "eval_sha256": dataset_hash(),
                      "decode_sha256": sha256(REPO / "scripts/gen_replace.py"),
                      "tests_sha256": sha256(REPO / "tests/smoke_cpu.py"), "device": "cuda", "dtype": "float32"})


def require_current_correctness(gate, cond):
    require(gate.get("source") == "receipt", "新取樣必須先有目前版本、真實模型的正確性憑證")
    if cond in ("rl_hf", "rl_replace", "geo_mix"):
        require(gate.get("hf_decode_checks"), "缺目前 HF 解碼的正確性憑證")
    if cond == "geo_mix":
        require(gate.get("geo_mix_checks"), "缺幾何混合 T6–T8 正確性憑證")


def check_run(directory, pair, cond, pc, ecfg, data, gate_receipt=None):
    directory = Path(directory)
    greedy = cond.endswith("_greedy")
    split = "all" if greedy else "test"
    n = 1 if greedy else ecfg["n_samples"]
    temp, top_p = (0, 1) if greedy else (pc["temperature"], pc["top_p"])
    hf = cond in ("rl_hf", "rl_replace", "geo_mix")
    if hf:
        require_current_correctness(gate_receipt or {}, cond)
    engine = "hf-lockstep" if hf else "vllm"
    raw_files = list(directory.glob("raw_*.jsonl.gz"))
    completion_files = list(directory.glob("meta_*.json"))
    require(raw_files and completion_files, "缺取樣分片或完成憑證")
    rows = [r for r in data if split == "all" or r["split"] == split]
    shards = {}
    historical = historical_profile(pair)
    for meta_file in completion_files:
        m = read_json(meta_file)
        index, count = parse_shard(m["shard"])
        require(meta_file.name == f"meta_{index}of{count}.json" and index not in shards,
                "分片憑證檔名或索引不符")
        raw = directory / f"raw_{index}of{count}.jsonl.gz"
        require(raw in raw_files, f"完成憑證缺 raw 分片 {index}/{count}")
        require((m["engine"], m["split"], m["n"], m["temperature"], m["top_p"], m["seed"],
                 m["max_new" if hf else "max_tokens"]) == (engine, split, n, temp, top_p, SAMPLE_SEED, pc["max_new_tokens"]),
                "引擎、樣本數、解碼、切分或 seed 與凍結設定不符")
        if hf:
            require(m["cond"] == ("rl" if cond == "rl_hf" else cond) and m["compact"] is True
                    and (cond != "geo_mix" or m["lam"] == 0.5), "HF 條件或設定不符")
            require(m.get("rows_per_batch") == 256 and m.get("seed_schedule") == HF_SEED_SCHEDULE,
                    "HF 批次大小或 seed 排程與凍結設定不符")
            models = {side: m[side] for side in ("base", "rl")}
        elif cond.startswith("wise_"):
            require(m["wise_lambda"] == float(cond.split("_", 1)[1]), "WiSE 權重不符")
            models = m["wise_models"]
            require(set(models) == {"base", "rl"}, "WiSE 缺少兩個來源模型")
        else:
            models = {cond.removesuffix("_greedy"): m["model"]}
        for side, model in models.items():
            pinned = gate_receipt.get("snapshots", {}).get(side) if gate_receipt else None
            require(model_matches(model, pc["pairs"][pair][side], pinned), f"{side} 模型或 snapshot 不符")
            if gate_receipt and side in gate_receipt:
                require(Path(model).name == Path(gate_receipt[side]).name, f"{side} snapshot 與正確性測試不符")
        if "provenance" in m:
            p = m["provenance"]
            producer = "gen_vllm.py" if engine == "vllm" else "gen_replace.py"
            require(p["pair"] == pair and p["condition"] == cond and p["engine"] == engine
                    and p["dtype"] == pc["dtype"] and p["max_model_len"] == pc["max_model_len"]
                    and p["stop_ids"] == STOP_IDS and p["eval_sha256"] == dataset_hash()
                    and p["pairs_sha256"] == pairs_hash() and p["run_id"] and p["job_id"]
                    and p["producer_sha256"] == sha256(REPO / "scripts" / producer) and p["engine_version"]
                    and p["raw_sha256"] == sha256(raw), "取樣完成憑證不相容")
            run_id = p["run_id"]
        else:
            require(historical is not None and cond in historical["runs"] and index == 0 and count == 1,
                    "取樣缺已登記的來源憑證")
            receipt = historical["runs"][cond]
            batch = REPO / f"slurm/logs/rsd-vllm-{receipt['job_id']}.out"
            require(historical["eval_sha256"] == dataset_hash() and historical["pairs_sha256"] == pairs_hash()
                    and sha256(batch) == receipt["log_sha256"], "歷史用資料／設定或 Slurm log 不符")
            log = batch.read_text(encoding="utf-8")
            require(f"done results/{pair}/{cond}/{raw.name}" in log and m["model"] in log
                    and f"dtype=torch.{pc['dtype']}" in log and f"max_seq_len={pc['max_model_len']}" in log,
                    "歷史 log 缺取樣完成或模型設定證據")
            run_id = f"historical-{receipt['job_id']}"
        shards[index] = (count, run_id, raw, meta_file)
    sizes = {s[0] for s in shards.values()}
    require(len(sizes) == 1, "分片總數不一致")
    size, = sizes
    require(set(shards) == set(range(size)) and len(raw_files) == size, "分片集合不完整或有多餘 raw 檔")
    require(len({s[1] for s in shards.values()}) == 1, "取樣分片屬於不同 run")
    return [(shards[i][2], shards[i][3], [r["id"] for r in shard(rows, i, size)]) for i in range(size)], n


def validate_records(records, ids, n, max_tokens, need_repl, raw=False):
    ids = list(ids)
    require(set(records) == set(ids) and len(records) == len(ids), "題目 ID 與凍結分片不一致")
    for i in ids:
        row = records[i]
        keys = ["finished", "ntok", "texts" if raw else "correct"]
        if need_repl or "n_repl" in row:
            keys.append("n_repl")
        require(all(isinstance(row.get(k), list) and len(row[k]) == n for k in keys),
                f"題目 {i} 的樣本數或欄位長度不符")
        require(all(type(x) is int and x in (0, 1) for x in row["finished"]), f"題目 {i} 結束狀態無效")
        require(all(type(x) is int and 0 <= x <= max_tokens for x in row["ntok"]), f"題目 {i} token 數無效")
        if "n_repl" in keys:
            require(all(type(x) is int and 0 <= x <= k for x, k in zip(row["n_repl"], row["ntok"], strict=True)),
                    f"題目 {i} 替換次數無效")
        if raw:
            require(all(isinstance(t, str) for t in row["texts"]), f"題目 {i} 生成資料格式無效")
        else:
            require(all(type(c) is int and 0 <= c <= f for c, f in zip(row["correct"], row["finished"], strict=True)),
                    f"題目 {i} 的對錯值無效或截斷樣本被判對")


def load_raw_bundle(bundle, n, max_tokens, cond):
    records = {}
    for raw, _, ids in bundle:
        # 檢查每個分片的 ID；不可把另一分片的題目挪來補缺。
        with gzip.open(raw, "rt", encoding="utf-8") as source:
            block = {}
            for line in source:
                r = json.loads(line)
                require(r["id"] not in block, f"重複題目 ID：{r['id']}")
                block[r["id"]] = r
        validate_records(block, ids, n, max_tokens, cond in ("rl_hf", "rl_replace", "geo_mix"), raw=True)
        records.update(block)
    return records


def input_receipts(bundle):
    return {path.name: sha256(path) for raw, meta, _ in bundle for path in (raw, meta)}


def load_verified(directory, pair, cond, pc, ecfg, data, gate_receipt):
    directory = Path(directory)
    bundle, n = check_run(directory, pair, cond, pc, ecfg, data, gate_receipt)
    receipt = read_json(directory / "graded_receipt.json")
    grade = directory / "graded_verified.json"
    require(receipt["grader"] == grader_receipt() and receipt["inputs"] == input_receipts(bundle)
            and receipt["grade_sha256"] == sha256(grade) and receipt["pair"] == pair and receipt["condition"] == cond,
            "缺相容判分憑證，或判分後輸入檔已變更")
    graded = read_json(grade)
    ids = [i for _, _, shard_ids in bundle for i in shard_ids]
    validate_records(graded, ids, n, pc["max_new_tokens"], cond in ("rl_hf", "rl_replace", "geo_mix"))
    return graded


def require_hf_pairing(root, pair, pc, ecfg, data, gate, *, reference="rl_hf", candidate="rl_replace"):
    """HF comparators must assign each question to the same seeded shard/batch."""
    def layout(cond):
        bundle, _ = check_run(Path(root) / pair / cond, pair, cond, pc, ecfg, data, gate)
        engine_versions = {read_json(meta)["provenance"]["engine_version"] for _, meta, _ in bundle}
        require(len(engine_versions) == 1, f"{cond} 的 HF 引擎環境不一致")
        return [ids for _, _, ids in bundle], engine_versions

    require(layout(reference) == layout(candidate), f"{reference} 與 {candidate} 的 HF 版本或分片配置不同")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="正式取樣前驗證正確性憑證")
    parser.add_argument("pair")
    parser.add_argument("condition")
    parser.add_argument("--results", type=Path, default=REPO / "results")
    args = parser.parse_args()
    try:
        gate = check_gate_minus1(args.results, args.pair, cfg("pairs.json"))
        require_current_correctness(gate, args.condition)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"缺相容正確性憑證：{exc}\n")
    print(f"正確性憑證通過：{args.pair} {args.condition} job {gate['job_id']}")


if __name__ == "__main__":
    main()
