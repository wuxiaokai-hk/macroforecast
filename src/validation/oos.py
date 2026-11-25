"""True out-of-sample tests for every model family.

The standard backtest in evaluate.py re-estimates the coefficients at every origin, but the drivers were
chosen once with ALL the data, so it is optimistic. Here, at each forecast origin, the whole pipeline is re-run
using only data up to that origin: stationarity tests, candidate features, both selection approaches, the
final search, the hybrid searches and the construction of every model in the zoo (models/zoo.py). Each model
is then estimated on data up to the origin and forecasts the next 1..H quarters.

Results:
  1. errors by horizon for every model, summarised at the report horizons (default 4, 8 and 12 quarters):
     average error over 1..h and error at h;
  2. the data-driven approaches with drivers fixed from the full sample (shows the look-ahead optimism);
  3. Diebold-Mariano tests (Harvey-Leybourne-Newbold small-sample correction) against the AR(1) benchmark;
  4. which drivers each data-driven and hybrid model chose at each origin, and how stable they were;
  5. a holdout test: everything selected and estimated on data up to hist_end - holdout_quarters.

Forecasts use the realised macro values (conditional forecasts), so they measure model error, not the
error of the macro scenario itself.
"""
import contextlib
import copy
import io

import numpy as np
import pandas as pd
from scipy import stats

from ..features.transforms import all_features, candidate_forms
from ..modelling.ardl import ARDL
from ..models import zoo
from ..utils import target as tg
from ..utils.cache import hash_code, hash_frame, hash_obj
from ..utils.io import log

CODE = ["features/transforms.py", "selection/screening.py", "selection/clustering.py", "selection/elastic_net.py",
        "modelling/search.py", "modelling/ardl.py", "pipelines/screen_first.py", "pipelines/en_first.py",
        "validation/oos.py", "models/zoo.py", "selection/double_selection.py", "pipelines/focus_first.py"]
PRETTY = zoo.APPROACH_NAMES


def inner_window(cfg, ctx, end_pos):
    """The backtest window used for selection at an origin. The configured window (sample.backtest_start to
    backtest_end) is kept if the origin is past it; otherwise a window of the same length ending at the origin
    is used (it must start at least 5 years after estimation_start), so early origins can still rank drivers."""
    s = cfg["sample"]
    start = pd.Period(s["backtest_start"], freq="Q")
    last = ctx.idx[end_pos - 1]
    end = pd.Period(s["backtest_end"], freq="Q") if s.get("backtest_end") else last
    if end <= last:
        return cfg
    length = max(end.ordinal - start.ordinal, 8)
    earliest = pd.Period(s["estimation_start"], freq="Q") + 20
    new_start = max(min(start, last - length), earliest)
    c = dict(cfg)
    c["sample"] = {**s, "backtest_start": str(new_start), "backtest_end": str(last)}
    return c


def truncated(ctx, end_pos, cfg):
    """A copy of the context that only 'knows' data up to end_pos (and, for a rolling window, from start)."""
    c = copy.copy(ctx)
    c.cfg = inner_window(cfg, ctx, end_pos)
    c.end = end_pos
    o = cfg["oos"]
    if o["window"] == "rolling":
        c.est0 = max(ctx.est0, end_pos - int(o["window_length"]) + 1)
    return c


def select_at(cfg, ctx, approach_mod, end_pos, cache, approach):
    """Run one approach using data up to end_pos only. Cached by the truncated data + settings + code.

    Args:
        approach_mod: the approach's pipeline module.
    """
    c = truncated(ctx, end_pos, cfg)
    comp = {"target": hash_frame(ctx.y.iloc[:end_pos + 1]), "macro": hash_frame(ctx.hist.iloc[:end_pos + 1]),
            "metadata": hash_frame(ctx.meta), "end": str(ctx.idx[end_pos]), "est0": str(ctx.idx[c.est0]),
            "config": hash_obj({**{k: cfg.get(k) for k in ("features", approach, "final_model", "sample")},
                                "impact": cfg.get("impact") if approach == "focus_first" else None,
                                "literature": cfg["literature"].get("models") if approach == "focus_first" else None,
                                "window": cfg["oos"]["window"], "window_length": cfg["oos"]["window_length"]}),
            "code": hash_code(*CODE)}

    def work():
        with contextlib.redirect_stdout(io.StringIO()):       # keep the per-origin logs quiet
            return _work()

    def _work():
        forms = candidate_forms(cfg, ctx.hist.iloc[:end_pos + 1], ctx.meta)      # ADF on data up to the origin
        F = all_features(cfg, c, forms)
        r = approach_mod.run(cfg, c, F, forms)
        return {**zoo.selection_info(approach, r), "est_start": str(ctx.idx[c.est0])}
    return cache.run(f"oos:{approach}:{ctx.idx[end_pos]}", comp, work, lambda *_: None)


def models_at(cfg, ctx, approaches, end_pos, cache, full_sample_drivers=None):
    """Every model, specified with data up to end_pos only (plus the look-ahead variants, if given).

    Args:
        full_sample_drivers: drivers chosen with all data, for the look-ahead variants (None = none).
    """
    sel = {ap: select_at(cfg, ctx, mod, end_pos, cache, ap) for ap, mod in approaches.items()}
    c = truncated(ctx, end_pos, cfg)
    with contextlib.redirect_stdout(io.StringIO()):
        models, notes, _ = zoo.build(cfg, c, {ap: {k: v for k, v in s.items() if k != "est_start"}
                                              for ap, s in sel.items()}, log=lambda *_: None)
    keep_check = any(m.family == "Data-driven" for m in models)     # the check compares against data-driven models
    present = {m.name for m in models}
    for ap, feats in ((full_sample_drivers or {}).items() if keep_check else []):
        if PRETTY[ap] in present:                  # only for the data-driven models that are in this run
            models.append(ARDL(f"{PRETTY[ap]} (full-sample drivers)", feats, family="Look-ahead check"))
    est = next(iter(sel.values()))["est_start"] if sel else str(ctx.idx[c.est0])
    return models, sel, est


def forecast_errors(ctx, model, origin_pos, H, est_start, detail=False):
    """Fit on est_start..origin and forecast H quarters; return level % errors (organic balance).

    Args:
        origin_pos: position of the forecast origin. est_start: first estimation quarter.
        detail: also return the forecast and actual log growth.

    Returns:
        errors (fractions), or (errors, forecast growth, actual growth) with detail=True.
    """
    P = model.panel(ctx, ctx.base).loc[:ctx.idx[ctx.end]]
    model.fit(P, pd.Period(est_start, freq="Q"), ctx.idx[origin_pos])
    fc = model.forecast(P, ctx.idx[origin_pos], H)
    act = ctx.y.values[origin_pos + 1:origin_pos + 1 + len(fc)]
    e = tg.error(np.cumsum(fc), np.cumsum(act))
    return (e, np.asarray(fc, dtype=float), np.asarray(act, dtype=float)) if detail else e


def dm_test(e_model, e_bench, h):
    """Diebold-Mariano on squared level errors, HAC variance with h-1 lags, HLN correction. Negative = model better.

    Args:
        e_model: model errors. e_bench: benchmark errors. h: horizon.
    """
    d = np.asarray(e_model) ** 2 - np.asarray(e_bench) ** 2
    d = d[~np.isnan(d)]
    n = len(d)
    corr = n + 1 - 2 * h + h * (h - 1) / n
    if n < h + 3 or corr <= 0:                 # too few origins for this horizon: no reliable test
        return np.nan, np.nan, n
    dbar = d.mean()
    gam = [np.mean((d[k:] - dbar) * (d[:n - k] - dbar)) for k in range(min(h, n - 1))]
    var = (gam[0] + 2 * sum(gam[1:])) / n
    if var <= 0:
        var = gam[0] / n
    dm = dbar / np.sqrt(var)
    stat = dm * np.sqrt(corr / n)
    return stat, 2 * stats.t.sf(abs(stat), df=n - 1), n


def run_oos(cfg, ctx, approaches, full_sample_drivers, cache, store=None):
    """Out-of-sample test: at each origin, re-run selection and re-estimate every model with data up to
    the origin only, then forecast 1..H quarters.

    Args:
        cfg: configuration (oos). ctx: Context. approaches: name -> pipeline module.
        full_sample_drivers: drivers chosen with all data (for the look-ahead check). cache: StepCache.
        store: optional dict, filled with {origin: {model name: pickle bytes}} (each model as estimated at that
            origin) and store["__coefficients__"]: list of coefficient rows by origin.

    Returns:
        (errors by model/origin/horizon, drivers picked at each origin, horizon).
    """
    o = cfg["oos"]
    H = int(o["horizon"] or cfg["sample"]["horizon"])
    first = pd.Period(o["first_origin"], freq="Q")
    origins = [i for i, p in enumerate(ctx.idx) if p >= first and i < ctx.end][::int(o["step"])]
    if not origins:
        raise RuntimeError("oos.first_origin is after the last usable quarter")
    dq = [pd.Period(q, freq="Q") for q in cfg["sample"]["dummy_quarters"]]
    a, b = (min(dq), max(dq)) if dq else (None, None)
    rows, picks, failed = [], [], {}
    log(f"Out-of-sample: {len(origins)} origins {ctx.idx[origins[0]]}..{ctx.idx[origins[-1]]}, step {o['step']}, "
        f"horizon {H}, {o['window']} window")
    for k, op in enumerate(origins):
        lab = ctx.idx[op]
        models, sel, est = models_at(cfg, ctx, approaches, op, cache, full_sample_drivers)
        est_full = cfg["sample"]["estimation_start"] if o["window"] == "expanding" else est
        for m in models:
            try:
                e, fc, act = forecast_errors(ctx, m, op, H, est_full, detail=True)
            except Exception as ex:                          # one model failing must not stop the test
                failed[m.name] = f"{type(ex).__name__}: {ex}"
                continue
            if store is not None:
                from ..modelling.evaluate import freeze
                store.setdefault(str(lab), {})[m.name] = freeze(ctx, m)
                prm = getattr(m, "params", None)
                if isinstance(prm, pd.Series):
                    store.setdefault("__coefficients__", []).extend(
                        {"origin": str(lab), "model": m.name, "family": m.family, "term": t_, "coef": float(v_)}
                        for t_, v_ in prm.items())
            if m.family in ("Data-driven", "Hybrid"):
                picks.append({"origin": str(lab), "model": m.name, "family": m.family, "drivers": ", ".join(m.drivers())})
            L0 = float(ctx.level.loc[lab]) if lab in ctx.level.index else np.nan
            cf, ca = np.cumsum(fc), np.cumsum(act)
            for h, v in enumerate(e, 1):
                win = ctx.idx[op + 1:op + h + 1]
                covid = bool(a is not None and ((win >= a) & (win <= b)).any())
                d = ctx.idx[op + h]
                rows.append({"origin": str(lab), "model": m.name, "family": m.family, "horizon_q": h, "error": v,
                             "covid_window": covid, "date": str(d), "drivers": ", ".join(m.drivers()),
                             "forecast_growth_pct": float(tg.growth_display(fc[h - 1])),
                             "actual_growth_pct": float(tg.growth_display(act[h - 1])),
                             "balance_at_origin": L0, "forecast_balance": tg.level_after(L0, cf[h - 1]),
                             "actual_balance_organic": tg.level_after(L0, ca[h - 1]),
                             "actual_balance_reported": float(ctx.level.get(d, np.nan)), "error_pct": v * 100})
        log(f"  origin {lab} ({k + 1}/{len(origins)}): " +
            "; ".join(f"{PRETTY[ap]}: {', '.join(s['features'])}" for ap, s in sel.items()))
    for m, why in failed.items():
        log(f"  {m}: left out of the test ({why[:200]})")
    return pd.DataFrame(rows), pd.DataFrame(picks), H


def summarise(E, picks, H, horizons):
    """Summarise out-of-sample errors: MAPE, RMSE, bias and Theil's U by horizon, and the averages
    over 1..h for each comparison horizon.

    Args:
        E: errors from run_oos. picks: drivers by origin. H: horizon. horizons: comparison horizons.

    Returns:
        (summary per model, errors by horizon).
    """
    Ex = E[~E["covid_window"]]
    g = Ex.groupby(["model", "family", "horizon_q"], sort=False)["error"]
    by_h = pd.DataFrame({"MAPE_pct": g.apply(lambda x: x.abs().mean() * 100),
                         "RMSE_pct": g.apply(lambda x: np.sqrt((x ** 2).mean()) * 100),
                         "bias_pct": g.mean() * 100, "n": g.size()}).reset_index()
    bench = by_h[by_h["model"] == "AR(1) benchmark"].set_index("horizon_q")["RMSE_pct"]
    by_h["theil_U_vs_AR"] = by_h.apply(lambda r: r["RMSE_pct"] / bench.get(r["horizon_q"], np.nan), axis=1)
    rows = []
    for (m, fam), d in by_h.groupby(["model", "family"], sort=False):
        d = d.set_index("horizon_q")
        r = {"model": m, "family": fam}
        for h in horizons:
            r[f"avg_1_{h}_pct"] = d["MAPE_pct"].loc[:h].mean()
        for h in horizons:
            r[f"at_{h}q_pct"] = d["MAPE_pct"].get(h, np.nan)
        r[f"theil_U_1_{H}"] = d["theil_U_vs_AR"].mean()
        r[f"bias_1_{H}_pct"] = d["bias_pct"].mean()
        r[f"origins_at_{max(horizons)}q"] = int(d["n"].get(max(horizons), 0))
        rows.append(r)
    summ = pd.DataFrame(rows)
    b = summ.set_index("model").loc["AR(1) benchmark"] if "AR(1) benchmark" in set(summ["model"]) else None
    if b is not None:
        for h in horizons:
            summ[f"vs_AR_1_{h}_pct"] = (1 - summ[f"avg_1_{h}_pct"] / b[f"avg_1_{h}_pct"]) * 100
    summ[f"rank_1_{max(horizons)}"] = summ[f"avg_1_{max(horizons)}_pct"].rank(method="min").astype("Int64")
    # Diebold-Mariano vs benchmark at the report horizons
    dm = []
    piv = Ex.pivot_table(index=["origin", "horizon_q"], columns="model", values="error")
    for m in summ["model"]:
        if m == "AR(1) benchmark" or "AR(1) benchmark" not in piv:
            continue
        for h in horizons:
            if h not in piv.index.get_level_values("horizon_q"):
                continue
            sub = piv.xs(h, level="horizon_q")[[m, "AR(1) benchmark"]].dropna()
            s, p, n = dm_test(sub[m], sub["AR(1) benchmark"], h)
            dm.append({"model": m, "horizon_q": h, "n_origins": n, "DM_stat": s, "p_value": p,
                       "better_than_benchmark_at_5pct": bool(s < 0 and p < 0.05) if not np.isnan(s) else False})
    dm = pd.DataFrame(dm)
    # driver stability
    stab = []
    for m, g in picks.groupby("model", sort=False):
        sets = [set(x.split(", ")) - {""} for x in g["drivers"]]
        jac = [len(s1 & s2) / max(len(s1 | s2), 1) for s1, s2 in zip(sets[:-1], sets[1:])]
        stab.append({"model": m, "family": g["family"].iloc[0], "distinct_driver_sets": len({frozenset(s) for s in sets}),
                     "avg_overlap_with_previous_origin": float(np.mean(jac)) if jac else np.nan,
                     "most_common_drivers": ", ".join(pd.Series([d for s in sets for d in s]).value_counts().head(4).index)})
    return by_h, summ, dm, pd.DataFrame(stab)


def holdout(cfg, ctx, approaches, cache):
    """Select and estimate every model on data up to hist_end - holdout_quarters; forecast the held-out quarters."""
    q = int(cfg["oos"]["holdout_quarters"])
    if q <= 0:
        return None
    op = ctx.end - q
    models, _, est = models_at(cfg, ctx, approaches, op, cache)
    est_full = cfg["sample"]["estimation_start"] if cfg["oos"]["window"] == "expanding" else est
    level0 = ctx.level.iloc[op]
    act = ctx.y.values[op + 1:op + 1 + q]
    out = []
    for m in models:
        try:
            P = m.panel(ctx, ctx.base).loc[:ctx.idx[ctx.end]]
            m.fit(P, pd.Period(est_full, freq="Q"), ctx.idx[op])
            fc = m.forecast(P, ctx.idx[op], q)
        except Exception as ex:
            log(f"  holdout: {m.name} left out ({type(ex).__name__}: {str(ex)[:150]})")
            continue
        for h in range(len(fc)):
            out.append({"model": m.name, "family": m.family, "drivers": ", ".join(m.drivers()) or "(none)",
                        "train_end": str(ctx.idx[op]), "date": str(ctx.idx[op + 1 + h]),
                        "forecast": tg.level_after(level0, np.cumsum(fc)[h]),
                        "actual_organic": tg.level_after(level0, np.cumsum(act)[h]),
                        "error_pct": float(tg.error(np.cumsum(fc)[h], np.cumsum(act)[h])) * 100})
    return pd.DataFrame(out)
