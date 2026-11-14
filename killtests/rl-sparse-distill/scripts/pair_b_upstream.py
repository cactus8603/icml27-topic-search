"""Use the pinned SimpleRL-Zoo prompt and grading functions on CPU."""

import argparse
from concurrent.futures import TimeoutError
import gzip
from importlib.metadata import version
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
from artifacts import atomic_json, read_json, require, sha256
from common import cfg


def upstream_rows(cache, settings):
    for relative, digest in settings["upstream_files"].items():
        require(sha256(cache / relative) == digest, f"上游檔案指紋不符：{relative}")
    sys.path[:0] = [str(cache / "repro-deps"), str(cache), str(cache / "latex2sympy")]
    for package, expected in settings["upstream_deps"].items():
        require(version(package) == expected, f"上游判分環境版本不符：{package}")
    from parser import parse_ground_truth, parse_question
    from utils import construct_prompt

    spec = settings["reproduction"]
    args = SimpleNamespace(prompt_type=spec["template"], adapt_few_shot=False, num_shots=0)
    rows = []
    for name, count in spec["datasets"].items():
        path = cache / "data" / name / "test.jsonl"
        with path.open(encoding="utf-8") as source:
            examples = [json.loads(line) for line in source]
        if "idx" not in examples[0]:
            examples = [dict(idx=index, **row) for index, row in enumerate(examples)]
        examples.sort(key=lambda row: row["idx"])
        require(len(examples) == count and len({row["idx"] for row in examples}) == count,
                f"上游題數或 ID 不符：{name}")
        for row in examples:
            question = parse_question(row, name)
            gold_cot, gold = parse_ground_truth(row, name)
            require(question, f"上游題目為空：{name}/{row['idx']}")
            prompt = construct_prompt(dict(question=question, gt_ans=gold), name, args)
            # The author's evaluate() normalizes the already extracted answer again.
            _, scored_gold = parse_ground_truth({"gt_cot": gold_cot, "gt": gold}, name)
            rows.append((f"{name}:{row['idx']}", name, prompt, scored_gold))
    return rows


def score(rows, raw, stops):
    from grader import math_equal_process
    from parser import choice_answer_clean, run_execute
    from pebble import ProcessPool

    require(set(raw) == {row[0] for row in rows}, "原作重現輸出的題目集合不符")
    work = []
    for index, (key, name, prompt, gold) in enumerate(rows):
        # Reproduce the author's postprocessing. The abel branch only extracts text.
        text = (prompt + raw[key]["texts"][0]).split(prompt)[-1].strip()
        for delimiter in stops:
            text = text.split(delimiter, 1)[0].strip()
        prediction, _ = run_execute(None, text, "abel", name, execute=False)
        choices = set("ABCDE")
        if gold in choices and prediction not in choices:
            prediction = choice_answer_clean(text)
        elif isinstance(gold, str) and set(gold) <= choices and isinstance(prediction, str):
            prediction = "".join(character for character in prediction if character in choices)
        work.append((index, prediction, gold))
    with ProcessPool(max_workers=1) as pool:
        scores = pool.map(math_equal_process, work, timeout=3).result()
        output = []
        for key, name, *_ in rows:
            try:
                correct, timeout = bool(next(scores)), False
            except TimeoutError:
                correct, timeout = False, True
            output.append((key, name, int(correct), int(timeout)))
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "score", "check"])
    parser.add_argument("cache", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--prompts", type=Path)
    args = parser.parse_args()
    settings = cfg("pair_b.json")
    rows = upstream_rows(args.cache, settings)
    prompts = [{"id": key, "source": name, "prompt": prompt} for key, name, prompt, _ in rows]
    stops = settings["reproduction"]["stop"]
    if args.stage == "check":
        from grader import math_equal
        from parser import run_execute
        expected, _ = run_execute(None, "Therefore the answer is 42.", "abel", "math500")
        require(math_equal(expected, "42"), "原作答案抽取或計分無法使用")
        empty = sum(gold is None or not str(gold).strip() for *_, gold in rows)
        print(f"上游評測資料和計分檢查通過，共 {len(rows)} 題，其中 {empty} 題的作者抽取答案為空。")
    elif args.stage == "prepare":
        atomic_json(args.output, prompts)
    else:
        require(read_json(args.prompts) == prompts, "原作重現的 prompt 與上游凍結資料不符")
        with gzip.open(args.input, "rt", encoding="utf-8") as source:
            raw = {row["id"]: row for row in map(json.loads, source)}
        atomic_json(args.output, score(rows, raw, stops))


if __name__ == "__main__":
    main()
