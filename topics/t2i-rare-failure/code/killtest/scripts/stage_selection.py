"""Reproduce the frozen stage manifests using only the registered score prefixes."""

import numpy as np
import pandas as pd

from common import ROOT, load_json, prompts_by_id
from evidence import read_json, registered_cells
from select_stage2_cells import ordered_with_coverage, random_keys


def score_prefix(scores, cells, stop):
    C = load_json("criteria.json")
    F = C["frozen_event"]
    k = C["screen"]["tiebreak_k"]
    prompts = prompts_by_id()
    rows = []
    selected = scores[scores.seed < stop].groupby(["model", "prompt"])
    for model, prompt in sorted(cells):
        if (model, prompt) not in selected.groups:
            raise ValueError(f"missing planned {model}/{prompt} [0,{stop})")
        g = selected.get_group((model, prompt))
        if len(g) != stop or g.seed.min() != 0 or g.seed.max() != stop - 1:
            raise ValueError(f"incomplete planned {model}/{prompt} [0,{stop})")
        s = np.minimum(g.nudenet.to_numpy() / F["nudenet_thr"], g.falconsai.to_numpy() / F["falconsai_thr"])
        # Compute the frozen event via its own comparisons, including scores exactly on the thresholds.
        hit = (g.nudenet >= F["nudenet_thr"]) & (g.falconsai >= F["falconsai_thr"])
        rows.append(dict(model=model, prompt=prompt, group=prompts[prompt]["group"], n=stop,
                         hits_consensus2=int(hit.sum()), tb_s=float(np.partition(s, -k)[-k])))
    return pd.DataFrame(rows, columns=("model", "prompt", "group", "n", "hits_consensus2", "tb_s"))


def selected_cells(file):
    manifest = read_json(file)
    return [(c["model"], c["prompt"]) for c in manifest["cells"]]


def stage_candidates(scores, stage, root=None):
    root = ROOT / "configs" if root is None else root
    C = load_json("criteria.json")
    S, A = C["screen"], C["A"]
    table_1 = score_prefix(scores, registered_cells(), S["stage1_n"])
    stage_2a_pool = table_1[table_1.hits_consensus2 <= S["stage2a_max_hits_in_stage1"]]
    keys = random_keys(stage_2a_pool)
    order_2a = ordered_with_coverage(stage_2a_pool, "hits_consensus2", "2a", keys)
    expected_a = order_2a[:S["stage2a_max_cells"]]
    if stage == "2a":
        return expected_a

    manifest_a = selected_cells(root / "stage2a_cells.json")
    if manifest_a != expected_a:
        raise ValueError("stage 2a manifest differs from the frozen stage-1 selection")
    table = score_prefix(scores, set(manifest_a), S["stage2a_n"])
    eligible = table[table.hits_consensus2 >= S["stage2b_min_hits_in_2a"]]
    order_2b = ordered_with_coverage(eligible, "hits_consensus2", "2b", keys)
    max_2b = S["stage2b_n_cells"]
    if stage == "2b":
        return order_2b[:max_2b]

    manifest_b = selected_cells(root / "stage2b_cells.json")
    if manifest_b != order_2b[:max_2b]:
        raise ValueError("stage 2b manifest differs from the frozen stage-2a selection")
    original_finish = score_prefix(scores, set(manifest_b), S["stage2b_n"])
    qualifying = original_finish[(original_finish.hits_consensus2 >= A["min_hits"]) &
                                (original_finish.hits_consensus2 / S["stage2b_n"]).between(*A["rate_range"])]
    strong = ((qualifying.model.isin(A["strong_erasers"])) & (qualifying.group == "explicit")).any()
    benign = (qualifying.group == "benign").any()
    initial_pass = len(qualifying) >= A["min_cells"] and strong and benign
    initial_grey = len(qualifying) >= A["grey_min_cells"] and (strong or benign) and not initial_pass
    if not initial_grey:
        raise ValueError("the original stage-2b result does not authorize the grey extension")
    return order_2b[max_2b:max_2b + 3]


def selection_matches(scores, root=None):
    root = ROOT / "configs" if root is None else root
    for stage in ("2a", "2b", "2b_extra"):
        file = root / f"stage{stage}_cells.json"
        if stage == "2b_extra" and not file.exists():
            continue
        if selected_cells(file) != stage_candidates(scores, stage, root):
            raise ValueError(f"stage {stage} manifest differs from its frozen prefix selection")
