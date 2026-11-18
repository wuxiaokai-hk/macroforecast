"""Selection rules for the hybrid (theory + data) models.

core_plus_addons      Theory fixes the core (the literature's four channels); data may add up to `max_add` drivers
                      from the screen-first shortlist in themes the core does not cover, under the same rules as
                      the main search (|t|, sign, VIF, one per theme), ranked by out-of-sample error.
channels_data_measures Theory fixes the channels (themes) and their signs; data picks the best-measured series,
                      transform and lag in each channel; channels the data cannot support jointly
                      (|t| < prune_t) are dropped one at a time.
"""
import itertools

import numpy as np
import pandas as pd

from ..features.transforms import feature_matrix, parse_feature
from ..modelling.ardl import ols_np, vif
from ..modelling.search import backtest_origins, covid_mask, fast_backtest, refit_flags


def _arrays(ctx, names):
    """Numpy arrays used by the hybrid searches, cut at ctx.end.

    Args:
        ctx: Context. names: feature names.

    Returns:
        (features matrix, target vector, dummies matrix).
    """
    last = ctx.end + 1
    F = feature_matrix(ctx.hist, names, ctx.cfg["features"]["deflator"]).values[:last] if names else np.zeros((last, 0))
    return F, ctx.y.values[:last], ctx.dummies.values[:last]


def core_plus_addons(cfg, ctx, core, candidates, max_add=2):
    """core: list of features; candidates: DataFrame with feature, theme, expected_sign.

    Args:
        core: theory core features. candidates: DataFrame with feature, theme, expected_sign. max_add: data
        add-ons allowed.
    """
    fm = cfg["final_model"]
    meta = ctx.meta
    core_themes = {meta.loc[parse_feature(f)[0], "theme"] for f in core}
    core_codes = {parse_feature(f)[0] for f in core}
    cand = candidates[~candidates["theme"].isin(core_themes)
                      & ~candidates["feature"].map(lambda f: parse_feature(f)[0]).isin(core_codes)]
    feats = list(cand["feature"])
    themes, signs = cand["theme"].values, cand["expected_sign"].values
    Fc, y, D = _arrays(ctx, core)
    Fa, _, _ = _arrays(ctx, feats)
    rows = np.arange(ctx.est0, ctx.end + 1)
    Dr = D[rows][:, D[rows].std(axis=0) > 0] if D.shape[1] else np.zeros((len(rows), 0))
    base = np.column_stack([np.ones(len(rows)), y[rows - 1], Dr, Fc[rows]])
    n = len(rows)
    res = []
    for k in range(0, max_add + 1):
        for combo in itertools.combinations(range(len(feats)), k):
            if k and len(set(themes[list(combo)])) < k:
                continue
            Fk = Fa[rows][:, combo]
            X = np.column_stack([base, Fk])
            b, se, e = ols_np(X, y[rows])
            if k:
                bk, tk = b[-k:], b[-k:] / se[-k:]
                if np.any(np.abs(tk) < fm["t_min"]):
                    continue
                sg = signs[list(combo)]
                if np.any((sg != 0) & (np.sign(bk) != sg)):
                    continue
                if np.max(vif(np.column_stack([Fc[rows], Fk]))) > fm["vif_max"]:
                    continue
            res.append({"combo": combo, "k_added": k, "bic": n * np.log(e @ e / n) + X.shape[1] * np.log(n)})
    R = pd.DataFrame(res).sort_values("bic").head(fm["n_oos_candidates"]).reset_index(drop=True)
    H = cfg["sample"]["horizon"]
    origins = backtest_origins(ctx)
    flags = refit_flags(ctx, origins)
    M = covid_mask(ctx, origins, H)
    R["oos_mape_pct"] = [np.nanmean(np.abs(np.where(M, np.nan, fast_backtest(np.column_stack([Fc, Fa[:, list(c)]]), y, D,
                                                                             ctx.est0, origins, H, flags)))) * 100
                         for c in R["combo"]]
    best = R["oos_mape_pct"].min()
    pick = R[R["oos_mape_pct"] <= best * (1 + fm["selection_tolerance"])].sort_values(["k_added", "oos_mape_pct"]).index[0]
    added = [feats[i] for i in R.loc[pick, "combo"]]
    R["added"] = R["combo"].apply(lambda c: ", ".join(feats[i] for i in c) or "(none)")
    R["selected"] = R.index == pick
    return list(core) + added, R.drop(columns="combo")


def channels_data_measures(cfg, ctx, best, channels, prune_t=1.0):
    """best: screen results, one row per variable (feature, theme, coef, abs_t). channels: [{theme, sign, concept}].

    Args:
        best: screen results, one row per variable (feature, theme, coef, abs_t). channels: list of {theme,
        sign, concept}. prune_t: |t| below which a channel is dropped.
    """
    picks, rows_ = [], []
    for ch in channels:
        c = best[(best["theme"] == ch["theme"]) & (np.sign(best["coef"]) == ch["sign"])]
        if c.empty:
            rows_.append({"channel": ch.get("concept", ch["theme"]), "chosen measure": "(none with the right sign)"})
            continue
        r = c.sort_values("abs_t", ascending=False).iloc[0]
        picks.append(r["feature"])
        rows_.append({"channel": ch.get("concept", ch["theme"]), "chosen measure": r["feature"],
                      "univariate |t|": round(float(r["abs_t"]), 2)})
    kept = list(picks)
    rows = np.arange(ctx.est0, ctx.end + 1)
    while kept:
        F, y, D = _arrays(ctx, kept)
        Dr = D[rows][:, D[rows].std(axis=0) > 0] if D.shape[1] else np.zeros((len(rows), 0))
        X = np.column_stack([np.ones(len(rows)), y[rows - 1], Dr, F[rows]])
        b, se, _ = ols_np(X, y[rows])
        t = (b / se)[-len(kept):]
        if np.min(np.abs(t)) >= prune_t:
            break
        kept.pop(int(np.argmin(np.abs(t))))
    for r in rows_:
        r["kept after joint test"] = r.get("chosen measure") in kept
    return kept, pd.DataFrame(rows_)
