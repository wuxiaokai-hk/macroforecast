"""Shared evaluation of a final model (any type): rolling backtest, diagnostics, driver table, scenario forecasts."""
import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox, breaks_cusumolsresid, het_breuschpagan
from statsmodels.stats.stattools import durbin_watson, jarque_bera

from ..features.transforms import TR_DESC, RATE_TRS, describe, is_rate_tr, parse_feature, tr_desc
from ..utils import target as tg
from .ardl import ARDL, vif
from .search import backtest_origins, covid_mask, refit_flags


def backtest(ctx, model, P, paths_out=None, models_out=None):
    """Re-estimate at every origin from backtest_start, forecast 1..H quarters with realised macro values,
    and compare the cumulated balance with the organic actual balance.

    Args:
        ctx: Context. model: any model. P: its panel.
        paths_out: optional list; the quarter-by-quarter forecast paths (a DataFrame) are appended to it.
        models_out: optional dict; the model as estimated at each re-estimation origin is stored in it
            (origin label -> pickle bytes), so every fitted version can be reloaded later.

    Returns:
        (errors by horizon, coefficients at each re-estimation).
    """
    H = ctx.cfg["sample"]["horizon"]
    start = pd.Period(ctx.cfg["sample"]["estimation_start"], freq="Q")
    origins = backtest_origins(ctx)
    E = np.full((len(origins), H), np.nan)
    coefs = []
    Ph = P.loc[:ctx.idx[ctx.end]]
    # re-estimate on the schedule in the config (sample.reestimate_every); slow models may ask for a longer
    # interval (refit_every, in quarters). Between re-estimations the coefficients are kept and only the
    # forecast is redone from the latest actual data. Every estimation uses data up to its origin only.
    every = max(int(ctx.cfg["sample"].get("reestimate_every", 1)), int(getattr(model, "refit_every", 1)))
    flags = refit_flags(ctx, origins, every)
    paths = []
    for i, o in enumerate(origins):
        lab = ctx.idx[o]
        if flags[i]:
            model.fit(P, start, lab)
            if models_out is not None:
                models_out[str(lab)] = freeze(ctx, model)
        fc = model.forecast(Ph, lab, H)
        act = ctx.y.values[o + 1:o + 1 + len(fc)]
        E[i, :len(fc)] = tg.error(np.cumsum(fc), np.cumsum(act))
        paths.append(test_rows(ctx, model, o, fc, act, E[i, :len(fc)], refit=bool(flags[i])).assign(_i=i))
        if getattr(model, "params", None) is not None:
            coefs.append(model.params.rename(str(lab)))
        elif getattr(model, "weights", None) is not None:
            coefs.append(pd.Series(model.weights, index=[m.name for m in model.members], name=str(lab)))
    M = covid_mask(ctx, origins, H)
    Ex = np.where(M, np.nan, E)
    T = pd.concat(paths, ignore_index=True) if paths else pd.DataFrame()
    if len(T):
        T["covid_window"] = M[T["_i"].values, T["horizon_q"].values - 1]
        T = T.drop(columns="_i")
    if paths_out is not None:
        paths_out.append(T)
    by_h = pd.DataFrame({"model": model.name, "family": model.family, "horizon_q": np.arange(1, H + 1),
                         "MAPE_all_pct": np.nanmean(np.abs(E), axis=0) * 100,
                         "MAPE_exCOVID_pct": np.nanmean(np.abs(Ex), axis=0) * 100,
                         "bias_exCOVID_pct": np.nanmean(Ex, axis=0) * 100,
                         "n_origins": (~np.isnan(Ex)).sum(axis=0), "re_estimations": int(flags.sum())})
    return by_h, pd.DataFrame(coefs)


def test_rows(ctx, model, o, fc, act, err, refit=True):
    """The forecast path from one origin, quarter by quarter, next to what happened.

    Args:
        ctx: Context. model: the model. o: position of the origin. fc: forecast log growth. act: actual organic
        log growth. err: balance error (fraction). refit: whether the model was re-estimated at this origin.

    Returns:
        DataFrame: model, family, origin, date, horizon_q, forecast and actual growth (% q/q), balance at the
        origin, forecast and actual balance, error (%), whether the model was re-estimated at this origin.
    """
    n = len(fc)
    lab = ctx.idx[o]
    L0 = float(ctx.level.loc[lab]) if lab in ctx.level.index else np.nan
    dates = [str(d) for d in ctx.idx[o + 1:o + 1 + n]]
    rep = [float(ctx.level.get(d, np.nan)) for d in ctx.idx[o + 1:o + 1 + n]]
    return pd.DataFrame({"model": model.name, "family": model.family, "origin": str(lab), "date": dates,
                         "horizon_q": np.arange(1, n + 1), "_i": np.full(n, -1),
                         "forecast_growth_pct": tg.growth_display(fc), "actual_growth_pct": tg.growth_display(act),
                         "balance_at_origin": L0, "forecast_balance": tg.path(L0, fc),
                         "actual_balance_organic": tg.path(L0, act), "actual_balance_reported": rep,
                         "error_pct": np.asarray(err) * 100, "re_estimated_at_origin": refit})


def horizon_summary(by_h, horizons, col="MAPE_exCOVID_pct", bench="AR(1) benchmark"):
    """One row per model: average error over 1..h and error at exactly h, for each h in `horizons`.

    Args:
        col: error column to summarise. bench: benchmark model for the improvement columns.
    """
    rows = []
    for (m, fam), d in by_h.groupby(["model", "family"], sort=False):
        d = d.set_index("horizon_q")[col]
        r = {"model": m, "family": fam}
        for h in horizons:
            r[f"avg_1_{h}_pct"] = d.loc[:h].mean() if h in d.index else np.nan
        for h in horizons:
            r[f"at_{h}q_pct"] = d.get(h, np.nan)
        rows.append(r)
    S = pd.DataFrame(rows)
    hmax = max(horizons)
    if bench in set(S["model"]):
        b = S.set_index("model").loc[bench]
        for h in horizons:
            S[f"vs_AR_1_{h}_pct"] = (1 - S[f"avg_1_{h}_pct"] / b[f"avg_1_{h}_pct"]) * 100
    S[f"rank_1_{hmax}"] = S[f"avg_1_{hmax}_pct"].rank(method="min").astype("Int64")
    return S


def diagnostics(ctx, model, P):
    """Residual tests of a model estimated on the full sample.

    Ljung-Box autocorrelation (4 and 8 lags), CUSUM stability, Breusch-Pagan, Jarque-Bera, maximum VIF and sign
    checks. Non-regression models get a row saying the tests do not apply.

    Args:
        ctx: Context. model: any model. P: the model's panel.

    Returns:
        dict, one row of comparison/diagnostics.
    """
    start = pd.Period(ctx.cfg["sample"]["estimation_start"], freq="Q")
    model.fit(P, start, ctx.idx[ctx.end])
    if not isinstance(model, ARDL) and getattr(model, "kind", "") != "LevelsARDL":
        return {"model": model.name, "family": model.family, "n_obs": np.nan,
                "note": "no regression residual tests for this model type"}
    r = model.res
    e = r.resid
    lb = acorr_ljungbox(e, lags=[4, 8], return_df=True)["lb_pvalue"].values
    try:
        cus = breaks_cusumolsresid(e.values, ddof=r.model.exog.shape[1])[1]
    except Exception:
        cus = np.nan
    exog = pd.DataFrame(r.model.exog, columns=r.model.exog_names)
    if getattr(model, "kind", "") == "LevelsARDL":
        feats = [c for c in exog.columns if c != "const" and not c.startswith(("D_", "ys_lag"))]
    else:
        feats = [f for f in model.features if f in exog]
    if hasattr(model, "bases"):                  # auto-ARDL: the sign of each driver's long-run effect (sum of its lags)
        wrong = [b for b, v in model.long_run().items()
                 if _expected_sign(ctx, model, f"{b}|L0") not in (0, np.sign(v))]
    elif getattr(model, "kind", "") == "LevelsARDL":
        wrong = []
    else:
        wrong = [f for f in feats if _expected_sign(ctx, model, f) not in (0, np.sign(r.params[f]))]
    out = {"model": model.name, "family": model.family, "n_obs": int(r.nobs), "adj_R2": round(r.rsquared_adj, 3),
           "BIC": round(r.bic, 1), "max p-value of drivers": round(max([r.pvalues[f] for f in feats] or [np.nan]), 4),
           "max VIF": round(float(np.max(vif(exog[feats].values))) if len(feats) > 1 else 1.0, 2),
           "Ljung-Box p (4)": round(lb[0], 3), "Ljung-Box p (8)": round(lb[1], 3),
           "Breusch-Pagan p": round(het_breuschpagan(e, r.model.exog)[1], 3),
           "Jarque-Bera p": round(jarque_bera(e)[1], 3), "Durbin-Watson": round(durbin_watson(e), 2),
           "CUSUM p": round(cus, 3), "wrong-sign drivers": ", ".join(wrong) or "none"}
    if hasattr(model, "eg_p"):
        out["note"] = (f"error-correction speed {r.params.get('ect_lag1', np.nan):.3f} "
                       f"(t {r.tvalues.get('ect_lag1', np.nan):.2f}); Engle-Granger ADF p = {model.eg_p:.3f}")
    if hasattr(model, "lag_summary"):
        out["lags"] = model.lag_summary()
    b = getattr(model, "bounds", None)
    if b:
        out.update({"bounds F": round(b["F"], 3), "bounds F 5% I(0)": round(b["F_I0_5"], 2),
                    "bounds F 5% I(1)": round(b["F_I1_5"], 2), "bounds t": round(b["t"], 3),
                    "bounds t 5% I(1)": b["t_I1_5"], "note": f"bounds test: {b['decision']}"})
    return out


def driver_sensitivity(ctx, model, P, H=8):
    """How much each driver's scenario path moves the forecast: baseline path against the driver held at its last
    actual value, all else equal. Works for every model type (regressions, trees, networks).

    Args:
        ctx: Context. model: a fitted model. P: its panel (history + base scenario). H: quarters ahead.

    Returns:
        dict feature -> (effect on growth in pp per year, effect on the balance after H quarters in %).
    """
    out = {}
    end = ctx.idx[ctx.end]
    pos = P.index.get_loc(end)
    H = min(H, len(P) - pos - 1)
    if H <= 0:
        return out
    try:
        base = np.asarray(model.forecast(P, end, H))
    except Exception:
        return out
    groups = {f: [f] for f in (getattr(model, "features", []) or []) if f in P.columns}
    if hasattr(model, "bases"):                    # auto-ARDL: freeze a driver in all its lags at once
        groups = {f"long run: {b}": [c for c in P.columns if c.startswith(b + "|L")] for b in model.bases}
    if getattr(model, "kind", "") == "LevelsARDL":
        groups = {f"long run: {n[3:]}": [n] for n in model.lr_names}
    for f, cols in groups.items():
        P2 = P.copy()
        for c in cols:
            lag = int(c.rsplit("|L", 1)[1]) if "|L" in c else 0
            base_col = c.rsplit("|L", 1)[0] + "|L0" if "|L" in c and (c.rsplit("|L", 1)[0] + "|L0") in P.columns else c
            z = P[base_col].copy()
            z.iloc[pos + 1:] = z.iloc[pos]                      # the driver stays at its last actual value
            P2[c] = z.shift(lag) if base_col != c else z
            P2.iloc[:pos + 1, P2.columns.get_loc(c)] = P[c].iloc[:pos + 1]
        try:
            alt = np.asarray(model.forecast(P2, end, H))
        except Exception:
            continue
        n = min(len(base), len(alt))
        out[f] = (float(tg.per_year(np.mean(base[:n] - alt[:n]))), float(tg.error(base[:n].sum(), alt[:n].sum()) * 100))
    return out


def driver_table(ctx, model, P=None):
    """How each driver enters the model, for the driver tables.

    ARDL: coefficient, Newey-West t-statistic, expected sign and long-run multiplier; ML: importance or
    standardised coefficient; combinations: member weights.

    Args:
        ctx: Context. model: an estimated model.

    Returns:
        DataFrame, one row per term.
    """
    rows = _driver_rows(ctx, model, P)
    D = pd.DataFrame(rows)
    if P is not None:                                            # scenario effect of each driver, for every model
        sens = driver_sensitivity(ctx, model, P)
        if sens:
            if "term" not in D:
                D["term"] = None
            for f in sens:
                if f not in set(D["term"]):
                    D = pd.concat([D, pd.DataFrame([{"model": model.name, "term": f, "reading": ""}])], ignore_index=True)
            D["growth_effect_pp_per_year"] = D["term"].map(lambda t: sens.get(t, (np.nan, np.nan))[0])
            D["balance_effect_pct_8q"] = D["term"].map(lambda t: sens.get(t, (np.nan, np.nan))[1])
            D["description"] = D.get("description", pd.Series(index=D.index, dtype=object))
            D["description"] = [d if isinstance(d, str) and d else _describe_term(t, ctx.meta)
                                for d, t in zip(D["description"], D["term"])]
    return D


def _expected_sign(ctx, model, f):
    """Expected sign of a driver: the model's own (literature workbook, sheet models) or the metadata's.

    Args:
        ctx: Context. model: the model. f: feature name.

    Returns:
        -1, 0 or 1.
    """
    s_ = (getattr(model, "signs", None) or {}).get(f)
    if s_ is not None:
        return int(s_)
    from ..features.transforms import feature_sign
    try:
        return int(feature_sign(ctx.meta, f))
    except ValueError:                      # not a feature name (e.g. a long-run term)
        return int(ctx.meta.loc[f, "expected_sign"]) if f in ctx.meta.index else 0


def _describe_term(t, meta):
    """Plain description of a driver-table term when it is a feature name; None for other terms (regime labels,
    member models, "long run: ...").

    Args:
        t: term. meta: variable metadata.

    Returns:
        str or None.
    """
    if not isinstance(t, str) or t.count("|") != 2 or " " in t:
        return None
    try:
        return describe(t, meta)
    except Exception:
        return None


def _driver_rows(ctx, model, P):
    """Rows of the driver table before the scenario-effect columns: coefficients, importances or weights.

    Args:
        ctx: Context. model: an estimated model. P: its panel (or None).

    Returns:
        list of dict rows.
    """
    rows = []
    if hasattr(model, "driver_rows"):                            # regime-switching models describe themselves
        return list(model.driver_rows())
    if hasattr(model, "detail_rows") and P is not None:          # ARIMAX, Theta, deep learning
        return list(model.detail_rows(ctx, P))
    if not isinstance(model, ARDL):
        if hasattr(model, "importance") and model.importance() is not None:
            imp = model.importance()
            for c, v in imp.items():
                if str(c).startswith("D_"):
                    continue
                rows.append({"model": model.name, "term": c, "coef": v,
                             "reading": ("importance in the trees" if model.kind != "ridge" else "standardised ridge coefficient")})
        elif getattr(model, "weights", None) is not None:
            for m, w in zip(model.members, model.weights):
                rows.append({"model": model.name, "term": m.name, "coef": w,
                             "reading": f"weight {w:.0%}; drivers {', '.join(m.drivers()) or '(none)'}"})
        else:
            rows.append({"model": model.name, "term": ", ".join(model.drivers()), "reading": model.note or ""})
        return rows
    r = model.res
    rho = sum(v for k, v in r.params.items() if str(k).startswith("y_lag"))
    for c in r.params.index:
        if c == "const" or c.startswith("D_"):
            continue
        b = r.params[c]
        row = {"model": model.name, "term": c, "coef": b, "se_hac": r.bse[c], "t": r.tvalues[c], "p": r.pvalues[c]}
        if c.count("|") == 2 and not hasattr(model, "bases"):   # auto-ARDL: judged on the long-run effect below
            es = _expected_sign(ctx, model, c)
            row["expected_sign"] = es
            row["sign_ok"] = "no rule" if es == 0 else ("yes" if np.sign(b) == es else "no")
        if c.startswith("y_lag"):
            k_ = int(c[5:])
            row.update(description=f"Target {'growth' if tg.is_log() else 'change'} {k_} quarter{'s' if k_ > 1 else ''} earlier (momentum)",
                       reading=f"{b:.2f} of the growth {k_} quarter{'s' if k_ > 1 else ''} earlier carries into this quarter")
        elif c == "ect_lag1":
            row.update(description="Gap between the target and its long-run level, last quarter",
                       reading=f"{abs(b):.1%} of the gap to the long-run level closes each quarter" if b < 0
                       else "no pull back to the long-run level (wrong sign)")
        else:
            code, tr, lag = parse_feature(c)
            per_unit = b * (1 if is_rate_tr(tr) else 0.01) * (100 if tg.is_log() else 1)
            unit = "1 pp" if is_rate_tr(tr) else "1%"
            when = "in the same quarter" if lag == 0 else f"{lag} quarter{'s' if lag > 1 else ''} later"
            lr = f"; long-run effect {abs(per_unit) / (1 - rho):.3f} {'pp' if tg.is_log() else 'units'}" if rho < 1 else ""
            row.update(description=describe(c, ctx.meta),
                       reading=(f"A {unit} rise in {ctx.meta.loc[code, 'description']} [{tr_desc(tr)}] "
                                f"{'raises' if b > 0 else 'lowers'} the q/q {'growth' if tg.is_log() else 'change'} "
                                f"of the target by {abs(per_unit):.3f} {'pp' if tg.is_log() else 'units'} {when}{lr}"))
        rows.append(row)
    if hasattr(model, "bases"):                                 # auto-ARDL: lag orders and long-run effects
        rows.append({"model": model.name, "term": "lag orders", "reading": model.lag_summary()
                     + f"; {model.selection['models_tried'] if model.selection else '?'} combinations compared"})
        for b_, v in model.long_run().items():
            code, tr = b_.split("|")
            es = _expected_sign(ctx, model, f"{b_}|L0")
            rows.append({"model": model.name, "term": f"long run: {b_}", "coef": v, "expected_sign": es,
                         "sign_ok": "no rule" if es == 0 else ("yes" if np.sign(v) == es else "no"),
                         "reading": "sum of its lag coefficients / (1 - sum of the growth lags): the total effect "
                                    "once the dynamics have played out"})
    if getattr(model, "beta", None) is not None and hasattr(model, "lr_names"):    # ECM: the long-run relation
        rows.append({"model": model.name, "term": "long run: constant", "coef": float(model.beta[0]),
                     "reading": "long-run relation of the real target level (Engle-Granger step 1)"})
        for n, b in zip(model.lr_names, model.beta[1:]):
            rows.append({"model": model.name, "term": f"long run: {n[3:]}", "coef": float(b),
                         "reading": f"a 1% higher {n[3:].split(':')[0]} goes with a {b:.2f}% higher real target level in the long run"
                         if n.endswith(("log", "log_real")) else "long-run coefficient"})
    return rows


def scenario_forecasts(ctx, model):
    """Forecast every scenario with the model estimated on all history.

    Args:
        ctx: Context (ctx.macro holds the scenarios). model: any model.

    Returns:
        long DataFrame: model, family, scenario, date, dlog_forecast, balance_forecast.
    """
    start = pd.Period(ctx.cfg["sample"]["estimation_start"], freq="Q")
    last = ctx.level.iloc[-1]
    out = []
    for name, mdf in ctx.macro.items():
        P = model.panel(ctx, mdf)
        model.fit(P, start, ctx.idx[ctx.end])
        fc = model.forecast(P, ctx.idx[ctx.end], len(P))
        dates = P.index[P.index.get_loc(ctx.idx[ctx.end]) + 1:][:len(fc)]
        lvl = tg.path(last, fc)
        out += [{"model": model.name, "family": model.family, "scenario": name, "date": str(d), "dlog_forecast": f,
                 "balance_forecast": l} for d, f, l in zip(dates, fc, lvl)]
    return pd.DataFrame(out)


def evaluate(ctx, model):
    """Everything the comparison needs for one model: backtest, diagnostics, drivers, scenarios.

    Args:
        ctx: Context. model: any model following the model interface.

    Returns:
        dict with by_horizon, coef_path, diagnostics, drivers and scenarios (DataFrames / dict).
    """
    save = ctx.cfg.get("save") or {}
    P = model.panel(ctx, ctx.base)
    keep, at_origin = [], ({} if save.get("origin_models", True) else None)
    by_h, coef_path = backtest(ctx, model, P, keep, at_origin)
    tests = keep[0]
    diag = diagnostics(ctx, model, P)                          # also estimates the model on all history
    fv = fitted_values(ctx, model, P) if save.get("fitted_values", True) else None
    coefs, stats_ = coefficient_table(model)
    drivers = driver_table(ctx, model, P)
    scen = scenario_forecasts(ctx, model)
    out = {"by_horizon": by_h, "coef_path": coef_path, "diagnostics": diag, "drivers": drivers, "scenarios": scen,
           "test_forecasts": tests, "fitted_values": fv, "coefficients": coefs, "fit_statistics": stats_,
           "origin_models": at_origin, "panel": P if save.get("panels", True) else None}
    if hasattr(model, "regime_path"):                         # regime-switching models: regime over history
        out["regimes"] = regime_history(ctx, model, P)
    out["fitted"] = freeze(ctx, model)                         # the model estimated on all history, for reuse
    return out


def fitted_values(ctx, model, P):
    """In-sample one-step-ahead fit of a model estimated on all history, for every quarter of the estimation window.

    Uses only the model interface (forecast one quarter from each quarter t-1 with the actual data), so it works
    for every model type; for a regression it equals X·b.

    Args:
        ctx: Context. model: model already estimated on all history. P: its panel.

    Returns:
        DataFrame: model, family, date, actual, fitted, residual (the target's change in the model's form) and
        the level implied by the fitted change from the previous actual level; None if the model cannot do it.
    """
    Ph = P.loc[:ctx.idx[ctx.end]]
    prm = getattr(model, "params", None)
    prm = prm if isinstance(prm, pd.Series) else pd.Series(dtype=float)
    rows, failed = [], 0
    for t in range(ctx.est0 + 1, ctx.end + 1):
        q = str(ctx.idx[t])
        try:                                         # a quarter the model cannot start from (e.g. too little
            f = float(np.asarray(model.forecast(Ph, ctx.idx[t - 1], 1))[0])   # history for a filter) is left empty
        except Exception:
            failed += 1
            f = np.nan
        d = float(prm.get(f"D_{q}", 0.0))               # forecasts set one-off dummies to 0: add their effect back
        a = float(ctx.y.iloc[t])
        prev = float(ctx.level.iloc[t - 1])
        rows.append({"model": model.name, "family": model.family, "date": q, "actual": a,
                     "fitted": f + d, "residual": a - f - d, "dummy_quarter": f"D_{q}" in ctx.dummies,
                     "level_actual": float(ctx.level.iloc[t]), "level_fitted": float(tg.level_after(prev, f + d))})
    if not rows or failed == len(rows):
        return None
    return pd.DataFrame(rows)


def coefficient_table(model):
    """Every estimated coefficient with its standard error, t and p value, and the fit statistics.

    Args:
        model: model estimated on all history.

    Returns:
        (DataFrame model, term, coef, std_err, t, p_value, or None; dict of fit statistics, or None).
    """
    try:
        return _coefficient_table(model)
    except Exception as ex:                     # never let the bookkeeping stop a run
        return None, {"model": model.name, "family": model.family, "note": f"coefficients not tabulated: {ex}"}


def _coefficient_table(model):
    """coefficient_table() without the safety net."""
    r = getattr(model, "res", None)
    params = getattr(model, "params", None)
    if r is not None and hasattr(r, "params"):
        rp = r.params
        if not isinstance(rp, pd.Series):          # numpy results: names from the model or the design
            names = (list(params.index) if isinstance(params, pd.Series) and len(params) == len(np.ravel(rp))
                     else list(getattr(getattr(r, "model", None), "exog_names", None) or range(len(np.ravel(rp)))))
            rp = pd.Series(np.ravel(np.asarray(rp, float)), index=names[:len(np.ravel(rp))])

        def get(a):
            try:
                v = getattr(r, a, None)
            except Exception:
                return None
            if v is None:
                return None
            v = np.ravel(np.asarray(v, float))
            return pd.Series(v, index=rp.index) if len(v) == len(rp) else None
        b, se, t, pv = rp, get("bse"), get("tvalues"), get("pvalues")
        T = pd.DataFrame({"model": model.name, "family": model.family, "term": b.index, "coef": b.values,
                          "std_err": se.values if se is not None else np.nan, "t": t.values if t is not None else np.nan,
                          "p_value": pv.values if pv is not None else np.nan})
        st = {"model": model.name, "family": model.family, "n_obs": float(getattr(r, "nobs", np.nan))}
        for k in ("rsquared", "rsquared_adj", "aic", "bic", "llf", "ssr"):
            v = getattr(r, k, None)
            if v is not None:
                try:
                    st[k] = float(v)
                except Exception:
                    pass
        st["cov_type"] = str(getattr(r, "cov_type", ""))
        st["residual_std"] = float(np.std(np.asarray(r.resid), ddof=1)) if hasattr(r, "resid") else np.nan
        if getattr(model, "beta", None) is not None and getattr(model, "long_run", None):   # ECM long run
            names = ["long run: constant"] + [f"long run: {i['mnemonic']}" for i in model.long_run]
            T = pd.concat([T, pd.DataFrame({"model": model.name, "family": model.family, "term": names,
                                            "coef": np.asarray(model.beta, float)[:len(names)]})], ignore_index=True)
        return T, st
    if isinstance(params, pd.Series) and len(params):
        return pd.DataFrame({"model": model.name, "family": model.family, "term": params.index,
                             "coef": params.values.astype(float)}), {"model": model.name, "family": model.family}
    w = getattr(model, "weights", None)
    if w is not None and getattr(model, "members", None):
        return pd.DataFrame({"model": model.name, "family": model.family,
                             "term": [f"weight: {m.name}" for m in model.members], "coef": np.asarray(w, float)}), \
            {"model": model.name, "family": model.family}
    return None, {"model": model.name, "family": model.family,
                  "note": "no coefficients (machine-learning, deep or pretrained model): see models.pkl"}


def freeze(ctx, model, max_mb=200):
    """The model as estimated on all history (call after scenario_forecasts), as pickle bytes; None if it cannot
    be pickled or is too large.

    Pretrained foundation models are not stored (they are downloaded again when needed).

    Args:
        ctx: Context. model: any model. max_mb: size limit.

    Returns:
        bytes or None.
    """
    import pickle
    if getattr(model, "no_pickle", False) or type(model).__name__ == "FoundationModel":
        return None
    try:                       # evaluate() has just estimated it on all history (scenario forecasts)
        b = pickle.dumps(model)
        return b if len(b) <= max_mb * 1e6 else None
    except Exception:
        return None


def regime_history(ctx, model, P):
    """The regime indicator of a regime-switching model over history, estimated on all data.

    Args:
        ctx: Context. model: a model with regime_path(). P: its panel on the history.

    Returns:
        DataFrame: model, family, date, value, meaning.
    """
    start = pd.Period(ctx.cfg["sample"]["estimation_start"], freq="Q")
    model.fit(P, start, ctx.idx[ctx.end])
    r = model.regime_path(P).loc[:ctx.idx[ctx.end]].dropna()
    return pd.DataFrame({"model": model.name, "family": model.family, "date": r.index.astype(str), "value": r.values,
                         "meaning": getattr(model, "regime_label", "")})
