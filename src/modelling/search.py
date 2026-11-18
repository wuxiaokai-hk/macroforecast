"""Shared final step: try every combination of min_vars..max_vars candidates, keep those that satisfy the
economic and statistical rules, rank them on out-of-sample balance error, and pick the simplest model
within `selection_tolerance` of the best."""
import itertools

import numpy as np
import pandas as pd

from ..utils import target as tg
from .ardl import ols_np, vif


def backtest_origins(ctx):
    """Forecast origins: from sample.backtest_start to sample.backtest_end (default: the last quarter that
    still has an actual to compare with), every sample.backtest_step quarters."""
    s = ctx.cfg["sample"]
    start = pd.Period(s["backtest_start"], freq="Q")
    end = pd.Period(s["backtest_end"], freq="Q") if s.get("backtest_end") else None
    out = [i for i, p in enumerate(ctx.idx) if p >= start and i < ctx.end and (end is None or p <= end)]
    return out[::max(1, int(s.get("backtest_step", 1)))]


def refit_flags(ctx, origins, every=None):
    """True where the coefficients are re-estimated. With sample.reestimate_every = 4 and
    reestimate_quarter = 1 the model is estimated once a year, at the first origin in Q1-Q4 of each year
    counted from Q1; at the other origins the same coefficients are kept and only the forecast is redone
    from the latest actual data. With reestimate_every = 1 (default) every origin re-estimates.

    Args:
        every: quarters between re-estimations (None = sample.reestimate_every).
    """
    s = ctx.cfg["sample"]
    every = max(1, int(every or s.get("reestimate_every", 1)))
    anchor = pd.Period(f"2000Q{int(s.get('reestimate_quarter', 1))}", freq="Q").ordinal
    flags, last = [], None
    for o in origins:
        w = (ctx.idx[o].ordinal - anchor) // every
        flags.append(w != last)
        last = w
    return np.array(flags, dtype=bool)


def covid_mask(ctx, origins, H):
    """Mark backtest errors whose forecast window touches the COVID dummy quarters.

    Args:
        ctx: Context. origins: positions of the forecast origins. H: horizon.

    Returns:
        bool array (origins x H); True = left out of the error metrics.
    """
    dq = [pd.Period(q, freq="Q") for q in ctx.cfg["sample"]["dummy_quarters"]]
    M = np.zeros((len(origins), H), dtype=bool)
    if not dq:
        return M
    a, b = min(dq), max(dq)
    for i, o in enumerate(origins):
        for h in range(1, H + 1):
            if o + h < len(ctx.idx):
                win = ctx.idx[o + 1:o + h + 1]
                M[i, h - 1] = ((win >= a) & (win <= b)).any()
    return M


def fast_backtest(F, y, D, est0, origins, H, refit=None):
    """Level % errors of iterated ARDL forecasts. F: features (T x k), y: target, D: dummies (T x d).
    refit: boolean per origin (see refit_flags); False keeps the previous coefficients.

    Args:
        y: target vector. D: dummies matrix. est0: position of the first estimation quarter. refit: bool per
        origin (None = every origin).
    """
    T = len(y)
    E = np.full((len(origins), H), np.nan)
    for i, o in enumerate(origins):
        if refit is None or refit[i] or i == 0:
            rows = np.arange(est0, o + 1)
            Dk = D[rows][:, D[rows].std(axis=0) > 0] if D.shape[1] else np.zeros((len(rows), 0))
            X = np.column_stack([np.ones(len(rows)), y[rows - 1], Dk, F[rows]])
            b = np.linalg.lstsq(X, y[rows], rcond=None)[0]
            c, rho, beta = b[0], b[1], b[2 + Dk.shape[1]:]
        yprev, cp, ca = y[o], 0.0, 0.0
        for h in range(1, H + 1):
            t = o + h
            if t >= T:
                break
            yh = c + rho * yprev + F[t] @ beta
            cp += yh
            ca += y[t]
            E[i, h - 1] = tg.error(cp, ca)
            yprev = yh
    return E


def exhaustive_search(cfg, ctx, F, info, label):
    """F: candidate features on the history index; info: DataFrame indexed by feature with theme, expected_sign.

    Args:
        info: DataFrame indexed by feature with theme and expected_sign. label: name used in the log and the
        tables.
    """
    fm = cfg["final_model"]
    feats = list(F.columns)
    themes = info.loc[feats, "theme"].values
    signs = info.loc[feats, "expected_sign"].values
    last = ctx.end + 1                     # never use data after the estimation end (matters in the OOS test)
    y = ctx.y.values[:last]
    D = ctx.dummies.values[:last]
    rows = np.arange(ctx.est0, ctx.end + 1)
    Fv = F.values[:last]
    Dr = D[rows][:, D[rows].std(axis=0) > 0] if D.shape[1] else np.zeros((len(rows), 0))
    base = np.column_stack([np.ones(len(rows)), y[rows - 1], Dr])
    n = len(rows)
    res = []
    stats = dict.fromkeys(["combinations", "same theme twice", "required theme missing", f"a driver with |t| < {fm['t_min']}",
                           "a driver with the wrong sign", f"VIF above {fm['vif_max']}", "passed every rule"], 0)
    for k in range(fm["min_vars"], fm["max_vars"] + 1):
        for combo in itertools.combinations(range(len(feats)), k):
            stats["combinations"] += 1
            th = themes[list(combo)]
            if fm["one_per_theme"] and len(set(th)) < k:
                stats["same theme twice"] += 1
                continue
            if fm["required_themes"] and not set(fm["required_themes"]) <= set(th):
                stats["required theme missing"] += 1
                continue
            Fk = Fv[rows][:, combo]
            X = np.column_stack([base, Fk])
            b, se, e = ols_np(X, y[rows])
            bk, tk = b[-k:], b[-k:] / se[-k:]
            if np.any(np.abs(tk) < fm["t_min"]):
                stats[f"a driver with |t| < {fm['t_min']}"] += 1
                continue
            sg = signs[list(combo)]
            if np.any((sg != 0) & (np.sign(bk) != sg)):
                stats["a driver with the wrong sign"] += 1
                continue
            v = vif(Fk)
            if np.max(v) > fm["vif_max"]:
                stats[f"VIF above {fm['vif_max']}"] += 1
                continue
            stats["passed every rule"] += 1
            ssr = e @ e
            res.append({"combo": combo, "k": k, "bic": n * np.log(ssr / n) + X.shape[1] * np.log(n),
                        "min_abs_t": float(np.min(np.abs(tk))), "max_vif": float(np.max(v))})
    if not res:
        raise RuntimeError(f"[{label}] no combination passed the rules: relax t_min / vif_max / one_per_theme "
                           f"or give the search more candidates.")
    R = pd.DataFrame(res).sort_values("bic").head(fm["n_oos_candidates"]).reset_index(drop=True)
    H = cfg["sample"]["horizon"]
    origins = backtest_origins(ctx)
    flags = refit_flags(ctx, origins)
    M = covid_mask(ctx, origins, H)
    err = []
    for c in R["combo"]:
        E = fast_backtest(Fv[:, list(c)], y, D, ctx.est0, origins, H, flags)
        err.append(np.nanmean(np.abs(np.where(M, np.nan, E))) * 100)
    R["oos_mape_pct"] = err
    R["features"] = R["combo"].apply(lambda c: [feats[i] for i in c])
    R = R.sort_values("oos_mape_pct").reset_index(drop=True)
    best = R["oos_mape_pct"].iloc[0]
    pick = R[R["oos_mape_pct"] <= best * (1 + fm["selection_tolerance"])].sort_values(
        ["k", "min_abs_t"], ascending=[True, False]).index[0]
    R["selected"] = R.index == pick
    R = pd.concat([R.loc[[pick]], R.drop(index=pick)]).reset_index(drop=True)
    R["features"] = R["features"].apply(", ".join)
    R = R.drop(columns="combo")
    stats["backtested (best by BIC)"] = len(R)
    stats["within the tolerance of the best error"] = int((R["oos_mape_pct"] <= best * (1 + fm["selection_tolerance"])).sum())
    R.attrs["rules"] = stats                      # how many combinations each rule removed (for the data story)
    return R, R.loc[0, "features"].split(", ")
