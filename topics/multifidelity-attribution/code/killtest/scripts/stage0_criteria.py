"""Phenomenon, signal and headroom decision rules from report 60 §4."""

from math import fsum

import numpy as np
from scipy.stats import rankdata, spearmanr


def lds_q(scores, masks, losses):
    predicted = -(masks.astype(np.float32) @ scores)
    p_rank = rankdata(predicted, axis=0)
    l_rank = rankdata(losses, axis=0)
    p_rank -= p_rank.mean(axis=0)
    l_rank -= l_rank.mean(axis=0)
    denom = np.linalg.norm(p_rank, axis=0) * np.linalg.norm(l_rank, axis=0)
    return np.divide((p_rank * l_rank).sum(axis=0), denom, out=np.full(denom.shape, np.nan), where=denom > 0)


def descriptive_spearman(a, b):
    a, b = np.asarray(a), np.asarray(b)
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 2 or np.ptp(a[valid]) == 0 or np.ptp(b[valid]) == 0:
        return None
    return float(spearmanr(a[valid], b[valid]).statistic)


def column_cosine(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a_norm = np.linalg.norm(a, axis=0)
    b_norm = np.linalg.norm(b, axis=0)
    denom = a_norm * b_norm
    return np.divide(
        np.einsum("ij,ij->j", a, b), denom, out=np.full(denom.shape, np.nan), where=denom > 0
    )


def grade_interval(mean, ci, rule):
    boundary = rule["ci_boundary"]
    if mean >= rule["mean_pass_min"] and ci[0] > boundary:
        return "PASS"
    if ci[1] < boundary:
        return "FAIL"
    return "GREY"


def stable_mean(values):
    return fsum(values) / len(values)


def issue_stage0_decision(rho_q, perm_q, lds_exact, lds_spell, lds_cheap, perm_gain, criteria, bootstrap):
    queries = np.random.RandomState(bootstrap["seed"]).randint(
        0, len(lds_exact), size=(bootstrap["n"], len(lds_exact))
    )
    tail = (1 - bootstrap["ci"]) / 2

    def ci(values):
        return np.quantile(values, [tail, 1 - tail]).tolist()

    best_cheap = stable_mean(lds_cheap)
    phenomenon = "PASS" if best_cheap < criteria["phenomenon_best_cheap_lds_max_exclusive"] else "FAIL"
    gap_spell = lds_exact - lds_spell
    gap_cheap = lds_exact - lds_cheap
    headroom = min(stable_mean(gap_spell), stable_mean(gap_cheap))
    headroom_ci = ci(np.minimum(gap_spell[queries].mean(axis=1), gap_cheap[queries].mean(axis=1)))
    decision = {
        "phenomenon": {"status": phenomenon, "best_cheap_lds": best_cheap},
        "headroom": {"status": grade_interval(headroom, headroom_ci, criteria["headroom"]),
                     "H": headroom, "ci": headroom_ci},
    }
    control = criteria["negative_control"]
    finite = np.isfinite(rho_q).all() and np.isfinite(perm_q).all() and np.isfinite(perm_gain)
    if not finite:
        decision["signal"] = {"status": "UNTESTABLE", "undefined_queries": int((~np.isfinite(rho_q)).sum()),
                              "undefined_permuted_queries": int((~np.isfinite(perm_q)).sum())}
        return decision, "UNTESTABLE(補空間餘弦未定義)"
    perm_rho = stable_mean(perm_q)
    pass_control = abs(perm_rho) <= control["rho_abs_max"] and perm_gain <= control["gain_max"]
    rho = stable_mean(rho_q)
    rho_ci = ci(rho_q[queries].mean(axis=1))
    signal = grade_interval(rho, rho_ci, criteria["signal"]) if pass_control else "INVALID"
    decision["signal"] = {"status": signal, "rho_perp": rho, "ci": rho_ci,
                          "control": {"rho_perp_permuted": perm_rho, "hybrid_gain_permuted": float(perm_gain),
                                      "pass": bool(pass_control)}}
    if not pass_control:
        return decision, "INVALID(置換負控制未通過；未能測試)"
    states = [value["status"] for value in decision.values()]
    if "FAIL" in states:
        return decision, "FAIL(本設定內至少一項未通過)"
    if all(state == "PASS" for state in states):
        return decision, "NEED_CROSSSEED(三項通過，須補跨種子 H_x)"
    return decision, "GREY(現象／訊號／空間待判讀)"


def grade_crossseed(lds_exact, lds_cheap, minimum):
    exact = stable_mean(lds_exact)
    cheap = stable_mean(lds_cheap)
    headroom = stable_mean(lds_exact - lds_cheap)
    status = "PASS" if headroom >= minimum else "GREY"
    return {"status": status, "H_x": headroom, "exact_lds": exact, "best_cheap_lds": cheap}
