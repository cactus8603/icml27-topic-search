import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr


def mean_lds(loss_ranks, removal, scores):
    return lds_from_predictions(loss_ranks, (-removal) @ scores)


def lds_from_predictions(loss_ranks, predicted):
    pred_ranks = rankdata(predicted, axis=0)
    pred_ranks -= pred_ranks.mean(axis=0)
    denom = np.linalg.norm(pred_ranks, axis=0) * np.linalg.norm(loss_ranks, axis=0)
    numer = (pred_ranks * loss_ranks).sum(axis=0)
    return np.divide(numer, denom, out=np.full_like(numer, np.nan), where=denom > 0)


def column_cosine(a, b):
    denom = np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0)
    return np.divide((a * b).sum(axis=0), denom, out=np.full(denom.shape, np.nan), where=denom > 0)


def report60_audit(Y, C, U, C_perm, YP, full_exact, full_spell, full_cheap, m1_gain, removal, loss_ranks):
    yr, cr, pr = Y - YP, C - projection(C, U), C_perm - projection(C_perm, U)
    cos = column_cosine(yr, cr)
    perm_cos = column_cosine(yr, pr)
    pos_1, pos_2 = full_exact - full_spell, full_exact - full_cheap
    rng = np.random.RandomState(0)
    boot = rng.randint(0, len(cos), size=(10_000, len(cos)))
    H = float(min(pos_1.mean(), pos_2.mean()))
    H_ci = np.percentile(np.minimum(pos_1[boot].mean(axis=1), pos_2[boot].mean(axis=1)), [2.5, 97.5]).tolist()
    headroom = "PASS" if H >= 0.10 and H_ci[0] > 0.05 else "FAIL" if H_ci[1] < 0.05 else "GRAY"
    phenomenon = "PASS" if full_cheap.mean() < 0.8 else "FAIL"
    averages_finite = np.isfinite(cos).all() and np.isfinite(perm_cos).all()
    centered = lambda x: x - x.mean(axis=0)
    row_centered = lambda x: x - x.mean(axis=1, keepdims=True)
    rho = rho_perm = rho_ci = None
    signal = "UNTESTABLE_UNDEFINED_RESIDUAL_NORMS"
    if averages_finite:
        rho, rho_perm = float(cos.mean()), float(perm_cos.mean())
        rho_ci = np.percentile(cos[boot].mean(axis=1), [2.5, 97.5]).tolist()
        if abs(rho_perm) > 0.05 or m1_gain > 0.01:
            signal = "INVALID_CONTROL"
        else:
            signal = "PASS" if rho >= 0.20 and rho_ci[0] > 0.10 else "FAIL" if rho_ci[1] < 0.10 else "GRAY"
    return dict(phenomenon=phenomenon, signal=signal, headroom=headroom, candidate_gate_diagnostic_only=all(s == "PASS" for s in [phenomenon, signal, headroom]),
                rho_perp=rho, rho_perp_ci95=rho_ci, rho_perm=rho_perm, H=H, H_ci95=H_ci, baseline_lds=float(max(full_spell.mean(), full_cheap.mean())),
                diagnostic_only=dict(n_undefined=int(np.count_nonzero(~np.isfinite(cos))),
                                     n_perm_undefined=int(np.count_nonzero(~np.isfinite(perm_cos))),
                                     min_Y_residual_norm=float(np.linalg.norm(yr, axis=0).min()),
                                     min_C_residual_norm=float(np.linalg.norm(cr, axis=0).min()),
                                     centered_rho_perp=float(np.nanmean(column_cosine(centered(yr), centered(cr)))),
                                     column_centered_rho_perm=float(np.nanmean(column_cosine(centered(yr), centered(pr)))),
                                     row_centered_rho_perp=float(np.nanmean(column_cosine(row_centered(yr), row_centered(cr)))),
                                     row_centered_rho_perm=float(np.nanmean(column_cosine(row_centered(yr), row_centered(pr)))),
                                     removal_residual_lds_exact=float(mean_lds(loss_ranks, removal, yr).mean()),
                                     removal_residual_lds_cheap=float(mean_lds(loss_ranks, removal, cr).mean())))


def energy_fraction(numerator, denominator):
    return float(np.sum(numerator) / np.sum(denominator))


def equal_query_energy_fraction(numerator, denominator):
    centered_dims = denominator > 0
    return float(np.mean(numerator[centered_dims] / denominator[centered_dims]))


def leading_vectors(gram, budget):
    vals, vecs = np.linalg.eigh((gram + gram.T) / 2)
    return vecs[:, np.argsort(-vals)[:budget]]


def fit_alpha(Y, C, U):
    y_meas, c_meas = Y @ U, C @ U
    denominator = (c_meas * c_meas).sum()
    return float((y_meas * c_meas).sum() / denominator) if denominator else 0.0


def projection(X, U):
    return (X @ U) @ U.T


def lp_gain_summary(mixed, spell, cheap, threshold=0.04):
    bootstrap = np.random.default_rng(20261114).integers(0, len(mixed), size=(10_000, len(mixed)))
    delta_spell, delta_cheap = mixed - spell, mixed - cheap
    gains = np.minimum(delta_spell[bootstrap].mean(axis=1), delta_cheap[bootstrap].mean(axis=1))
    point = min(delta_spell.mean(), delta_cheap.mean())
    return dict(gain=float(point), ci95=np.percentile(gains, [2.5, 97.5]).tolist(),
                gap_to_practical_threshold=float(point - threshold), practical_threshold=threshold)


def crossfit_oracle(Y, C, G, budget):
    rng = np.random.default_rng(20261114)
    rows = rng.permutation(len(Y))
    fit, holdout = np.array_split(rows, 2)
    mixed, spell = np.empty_like(Y), np.empty_like(Y)
    captured, residual_total = np.zeros(Y.shape[1]), np.zeros(Y.shape[1])
    for train, test in [(fit, holdout), (holdout, fit)]:
        alpha = fit_alpha(Y[train], C[train], leading_vectors(G, budget))
        train_residual = Y[train] - alpha * C[train]
        U = leading_vectors(train_residual.T @ train_residual, budget)
        held_residual = Y[test] - alpha * C[test]
        held_projected = projection(held_residual, U)
        mixed[test] = alpha * C[test] + held_projected
        spell[test] = projection(Y[test], U)
        captured += (held_projected * held_projected).sum(axis=0)
        residual_total += (held_residual * held_residual).sum(axis=0)
    return mixed, spell, equal_query_energy_fraction(captured, residual_total)


def m2_synthetic(Y, U, noise, target, lds, spell_score, removals, loss_ranks):
    y_scale = Y.std(axis=0, keepdims=True)
    e = (noise * y_scale).astype(Y.dtype)
    lo, hi = 0.0, 1.0
    evaluated = []
    p_y = (-removals) @ Y
    p_e = (-removals) @ e

    def trial(sigma):
        score = float(lds_from_predictions(loss_ranks, p_y + sigma * p_e).mean())
        evaluated.append((sigma, score))
        return score

    score = trial(hi)
    while score > target + 0.005 and hi < 2**16:
        lo, hi = hi, hi * 2
        score = trial(hi)
    if abs(score - target) > 0.005:
        for _ in range(35):
            mid = (lo + hi) / 2
            score = trial(mid)
            if abs(score - target) <= 0.005:
                break
            if score > target:
                lo = mid
            else:
                hi = mid
    sigma, nearest_score = min(evaluated, key=lambda trial: abs(trial[1] - target))
    if abs(nearest_score - target) > 0.005:
        return dict(calibrated=False, nearest_lds=nearest_score, nearest_sigma=sigma, target_lds=target)
    C = Y + sigma * e
    nearest_score = lds(C)
    if abs(nearest_score - target) > 0.005:
        return dict(calibrated=False, nearest_lds=nearest_score, nearest_sigma=sigma, target_lds=target)
    alpha = fit_alpha(Y, C, U)
    H = projection(Y, U) + alpha * (C - projection(C, U))
    score = lds(H)
    return dict(calibrated=True, lds_c_syn=nearest_score, sigma=sigma, target_lds=target,
                hybrid_lds=score, gain=score - max(spell_score, nearest_score), noise_family="one fixed column-scaled iid Gaussian draw")


def load_all(run, cfg, criteria):
    n = cfg["n_train"] or 50_000
    k = cfg["k_stage0"]
    count = cfg["M"]
    losses = np.stack([np.load(run / "retrain" / f"loss_{i}.npy") for i in range(count)])
    assert losses.shape == (count, cfg["k_total"] + 1), losses.shape
    noise = np.stack([np.load(run / "retrain" / f"noise_{i}_task0.npy") for i in range(1, cfg["noise_repeats"] + 1)])
    assert noise.shape == (cfg["noise_repeats"], cfg["k_total"] + 1), noise.shape
    removals = np.stack([np.random.RandomState(cfg["mask_seed"] * 100003 + i).random_sample(n) < cfg["p_drop"] for i in range(count)]).astype(np.float32)
    chunks = []
    for s in range(0, k, cfg["qchunk"]):
        e = min(s + cfg["qchunk"], k)
        y = np.load(run / "exact" / f"Y_{s}_{e}.npy")
        assert y.shape == (n, e - s), y.shape
        chunks.append(y)
    exact = np.concatenate(chunks, axis=1)
    cheap = {}
    for method in criteria["cheap_variants"]:
        c = np.load(run / "cheap" / f"{method}.npy")
        assert c.shape == (n, cfg["k_total"]), (method, c.shape)
        cheap[method] = c[:, :k]
    gram = np.load(run / "cheap" / "gram_query_grads.npy")
    assert gram.shape == (cfg["k_total"], cfg["k_total"]), gram.shape
    assert all(np.isfinite(x).all() for x in [losses, noise, exact, gram, *cheap.values()])
    assert np.all((losses[:, -1] >= 0) & (losses[:, -1] <= 1))
    return losses[:, :k], noise[:, :k], removals, exact, cheap, gram[:k, :k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, type=Path)
    ap.add_argument("--criteria", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    cfg = json.loads(args.cfg.read_text())
    criteria = json.loads(args.criteria.read_text())
    losses, noise, removals, Y, cheap, G = load_all(Path(cfg["run_dir"]), cfg, criteria)
    B = max(1, round(Y.shape[1] * criteria["B_frac"]))
    threshold = criteria["stage0"]["descriptive_legacy"]["gain_threshold"]
    loss_ranks = rankdata(losses, axis=0)
    loss_ranks -= loss_ranks.mean(axis=0)
    lds = lambda scores: float(mean_lds(loss_ranks, removals, scores).mean())
    cheap_lds = {m: lds(c) for m, c in cheap.items()}
    best = max(cheap_lds, key=cheap_lds.get)
    C = cheap[best]
    U = leading_vectors(G, B)
    alpha = fit_alpha(Y, C, U)
    YP, CP = projection(Y, U), projection(C, U)
    ce_query_norms = np.linalg.norm(CP, axis=0) * np.linalg.norm(YP, axis=0)
    rho_hat = np.divide((CP * YP).sum(axis=0), ce_query_norms, out=np.zeros_like(ce_query_norms), where=ce_query_norms > 0)
    H = YP + alpha * (C - CP)
    R = Y - alpha * C
    RP = projection(R, U)
    y_sq, yp_sq = (Y * Y).sum(0), (YP * YP).sum(0)
    r_sq, rp_sq = (R * R).sum(0), (RP * RP).sum(0)
    kappa = energy_fraction(yp_sq, y_sq)
    kappa_res = energy_fraction(rp_sq, r_sq)
    pointwise_gain = lds(H) - max(lds(YP), cheap_lds[best])
    frozen_gain = lp_gain_summary(*[mean_lds(loss_ranks, removals, z) for z in (H, YP, C)], threshold=threshold)
    select_removals = removals[::2]
    select_losses = rankdata(losses[::2], axis=0)
    select_losses -= select_losses.mean(axis=0)
    select_lds = {m: mean_lds(select_losses, select_removals, c).mean() for m, c in cheap.items()}
    select_best = max(select_lds, key=select_lds.get)
    oracle_mixed, oracle_spell, oracle_capacity = crossfit_oracle(Y, cheap[select_best], G, B)
    eval_losses = rankdata(losses[1::2], axis=0)
    eval_losses -= eval_losses.mean(axis=0)
    oracle_l, oracle_s, oracle_c, frozen_s = [mean_lds(eval_losses, removals[1::2], z)
                                             for z in (oracle_mixed, oracle_spell, cheap[select_best], YP)]
    oracle_eval = dict(cheap_selected_on_even_retrain=select_best, evaluated_on_odd_retrain=int(len(eval_losses)),
                       equal_query_residual_energy=oracle_capacity, cheap_lds=float(oracle_c.mean()),
                       crossfit_oracle_spell_lds=float(oracle_s.mean()), crossfit_oracle_hybrid_lds=float(oracle_l.mean()),
                       frozen_spell_lds=float(frozen_s.mean()),
                       heldout_gain_vs_oracle_spell_and_cheap=lp_gain_summary(oracle_l, oracle_s, oracle_c, threshold),
                       heldout_gain_vs_frozen_spell_and_cheap=lp_gain_summary(oracle_l, frozen_s, oracle_c, threshold),
                       limitation="Exact residuals from other training rows inform U; the two row folds use two different subspaces, so this oracle is an empirical diagnostic, not an LDS bound or an implementable fixed-budget method.")
    even_q, odd_q = [mean_lds(ranks, removals[rows], C) for ranks, rows in [(select_losses, slice(None, None, 2)), (eval_losses, slice(1, None, 2))]]
    split_half = float(spearmanr(even_q, odd_q).statistic)
    full_exact = mean_lds(loss_ranks, removals, Y)
    full_cheap = mean_lds(loss_ranks, removals, C)
    full_spell = mean_lds(loss_ranks, removals, YP)
    query_loss = np.load(Path(cfg["run_dir"]) / "models" / f"model_s{cfg['train_seed']}.npz")["query_loss"][:len(full_exact)]
    reliability = dict(split_half_spearman=split_half, split_half_spearman_brown=float(2 * split_half / (1 + split_half)) if split_half > -1 else None,
                       full_replay_query_correlation=float(spearmanr(rho_hat, full_cheap).statistic),
                       baseline_query_loss_correlation=float(spearmanr(query_loss, full_cheap).statistic),
                       baseline_cheap_score_norm_correlation=float(spearmanr(np.linalg.norm(C, axis=0), full_cheap).statistic))
    rng_m1 = np.random.default_rng(20261114)
    C_perm = C[rng_m1.permutation(len(C))]
    YP_score, C_perm_score = lds(YP), lds(C_perm)
    alpha_perm = fit_alpha(Y, C_perm, U)
    C_perm_hybrid_score = lds(YP + alpha_perm * (C_perm - projection(C_perm, U)))
    m1_gain = C_perm_hybrid_score - max(YP_score, C_perm_score)
    m1 = dict(seed=20261114, gain=m1_gain, max_gain=0.01, pass_gain=(m1_gain <= 0.01),
              spell_lds=YP_score, cheap_permuted_lds=C_perm_score, hybrid_permuted_lds=C_perm_hybrid_score)
    m2 = None
    if m1["pass_gain"] and cheap_lds[best] < criteria["stage0"]["phenomenon_best_cheap_lds_max_exclusive"] and \
            kappa_res / kappa >= criteria["stage0"]["descriptive_legacy"]["kappa_ratio_threshold"] and pointwise_gain < threshold:
        syn = m2_synthetic(Y, U, np.random.default_rng(20261114).normal(size=Y.shape), cheap_lds[best], lds, YP_score, removals, loss_ranks)
        y_complement, c_complement = Y - YP, C - CP
        cd = (c_complement * c_complement).sum(axis=0)
        oracle_alpha = np.divide((y_complement * c_complement).sum(axis=0), cd, out=np.zeros_like(cd), where=cd > 0)
        orc_score = lds(YP + oracle_alpha[None, :] * c_complement)
        orc_gain = orc_score - max(YP_score, cheap_lds[best])
        if not syn["calibrated"]:
            classification = "synthetic calibration unavailable"
        elif syn["gain"] < threshold:
            classification = "below threshold for the specified synthetic control"
        elif orc_gain < threshold:
            classification = "weak complement for C and fixed G_V subspace under in-sample least squares"
        else:
            classification = "headroom with in-sample per-query alpha in fixed G_V subspace"
        m2 = dict(synthetic=syn, oracle_alpha_in_sample_gain=orc_gain, oracle_alpha_in_sample_lds=orc_score,
                  classification=classification)
    left_right = []
    for indices in [np.arange(Y.shape[1]), np.arange(Y.shape[1])[::2], np.arange(Y.shape[1])[1::2]]:
        a, b = Y[:, indices], C[:, indices]
        pB = max(1, round(len(indices) * criteria["B_frac"]))
        uc = leading_vectors(b.T @ b, pB)
        ap, bp = projection(a, uc), projection(b, uc)
        scalar = fit_alpha(a, b, uc)
        denom = (bp * bp).sum(axis=0)
        alpha_q = np.divide((ap * bp).sum(axis=0), denom, out=np.zeros_like(denom), where=denom > 0)
        h = ap + (0.5 * alpha_q + 0.5 * scalar)[None, :] * (b - bp)
        scores = lambda z: float(mean_lds(loss_ranks[:, indices], removals, z).mean())
        gv_p = projection(a, leading_vectors(G[np.ix_(indices, indices)], pB))
        cheap_score, spell_score, mixed_score, gv_score = map(scores, [b, ap, h, gv_p])
        left_right.append(dict(k=len(indices), cheap=cheap_score, cgram_projected_exact=spell_score, cgram_hybrid=mixed_score,
                               gv_projected_exact=gv_score, gain_vs_prespecified_baselines=mixed_score - max(cheap_score, spell_score, gv_score),
                               difference_vs_same_U_baselines=mixed_score - max(cheap_score, spell_score),
                               difference_vs_frozen_baselines=mixed_score - max(cheap_score, scores(YP[:, indices]))))
    m3_trigger = m1["pass_gain"] and cheap_lds[best] < criteria["stage0"]["phenomenon_best_cheap_lds_max_exclusive"] and \
                 (kappa_res/kappa < criteria["stage0"]["descriptive_legacy"]["kappa_ratio_threshold"] or pointwise_gain < threshold)
    m3_pass = m3_trigger and left_right[0]["gain_vs_prespecified_baselines"] >= threshold and \
              all(row["gain_vs_prespecified_baselines"] > 0 for row in left_right[1:])
    metrics = dict(exact_lds=lds(Y), cheap_lds=cheap_lds, best_cheap=best, best_cheap_stop_gap=cheap_lds[best] - criteria["stage0"]["phenomenon_best_cheap_lds_max_exclusive"],
                   simplified_spell_lds=lds(YP), hybrid_lds=lds(H), gain=pointwise_gain,
                   noise_over_signal=float(np.nanmean(noise.std(axis=0, ddof=1) / losses.std(axis=0, ddof=1))),
                   kappa=kappa, kappa_res=kappa_res, kappa_res_over_kappa=kappa_res/kappa, alpha=alpha,
                   report_59_audit=dict(m1=m1, m2_when_gain_stop=m2, m3_trigger=m3_trigger, m3_pass=m3_pass),
                   report_60_audit=report60_audit(Y, C, U, C_perm, YP, full_exact, full_spell, full_cheap, m1_gain, removals, loss_ranks),
                   descriptive_only=dict(equal_query_kappa=equal_query_energy_fraction(yp_sq, y_sq),
                                         equal_query_kappa_res=equal_query_energy_fraction(rp_sq, r_sq),
                                         equal_query_kappa_res_over_kappa=equal_query_energy_fraction(rp_sq, r_sq) / equal_query_energy_fraction(yp_sq, y_sq),
                                         frozen_gain_query_bootstrap=frozen_gain, crossfit_residual_oracle=oracle_eval,
                                         full_exact_reference_gain=lp_gain_summary(full_exact, full_spell, full_cheap, threshold),
                                         query_reliability=reliability,
                                         cgram_shrunk_reference_full_even_odd=left_right))
    args.out.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
