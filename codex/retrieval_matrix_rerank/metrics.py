"""[codex] Strict published MCQ rules; ties fail; keep every original row."""
import math


def hit(task, values):
    values = [float(value) for value in values]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Non-finite official pair score")
    if task == "negbench_mcq" and len(values) == 4:
        return int(all(values[0] > value for value in values[1:]))
    if task == "scpp" and len(values) == 3:
        return int(values[0] > values[2] and values[1] > values[2])
    if task == "cola_text_choice" and len(values) == 4:
        return int(values[0] > values[1] and values[3] > values[2])
    if task == "cola_image_choice" and len(values) == 4:
        return int(values[0] > values[2] and values[3] > values[1])
    raise ValueError("Unknown official task or incorrect pair count")


def evaluate(examples, scores):
    lookup = {row["id"]: float(row["score"]) for row in scores}
    expected = [pair["id"] for example in examples for pair in example["pairs"]]
    if len(lookup) != len(scores) or len(set(expected)) != len(expected) or set(lookup) != set(expected):
        raise ValueError("Scores must cover every original pair exactly once")
    detail, subsets, benchmarks = [], {}, {}
    for row in examples:
        correct = hit(row["task"], [lookup[pair["id"]] for pair in row["pairs"]])
        detail.append({"id": row["id"], "subset": row["subset"], "task": row["task"], "hit": correct})
        subsets.setdefault(row["subset"], []).append(correct)
        benchmarks.setdefault(row["task"], set()).add(row["subset"])
    if not detail:
        raise ValueError("No official examples")
    means = {key: sum(values) / len(values) for key, values in subsets.items()}
    report = {"subsets": {key: {"accuracy": means[key], "rows": len(subsets[key])} for key in subsets},
              "benchmarks": {key: {"macro_accuracy": sum(means[s] for s in names) / len(names),
                                   "subsets": sorted(names), "rows": sum(len(subsets[s]) for s in names)}
                             for key, names in benchmarks.items()},
              "rows": len(detail),
              "semantics": "strict comparisons; macro only within each named benchmark; no cross-benchmark average or row deduplication"}
    if len(benchmarks) == 1:
        report["macro_accuracy"] = next(iter(report["benchmarks"].values()))["macro_accuracy"]
    return report, detail
