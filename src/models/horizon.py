"""Horizon combinations: a different model for each forecast horizon.

Some models forecast the next few quarters well (momentum, short-run drivers), others the medium term (long-run
relations). The horizon combination splices them: quarters 1-4 from the model best at 1-4 quarters ahead, quarters
5-8 from the best at 5-8, and so on (models.horizon_combination.buckets).

  best per horizon      in each bucket, the model with the smallest squared error
  weighted per horizon  in each bucket, the top_k models, weighted by 1 / their squared error

The error (models.horizon_combination.criterion) is the quarterly growth error in those quarters ("growth", the
default: it is what gets spliced) or the balance error at those horizons ("balance": as the ranking table reads).

No look-ahead: at each forecast origin, the choice uses only the errors of forecasts made at earlier origins whose
quarters had already happened by then (min_history origins at least; before that, the model best over all
horizons so far, or the AR(1) benchmark). It is evaluated on exactly the same origins and quarters as every other
model, in the rolling backtest and in the out-of-sample test. The scenario forecasts use the choice made with all
the test errors.

The combination works on the quarterly growth paths, so the balance is L(t) = L(t-1) * exp(spliced growth).
"""
import numpy as np
import pandas as pd

from ..utils import target as tg

FAMILY = "Horizon combination"
NAMES = {"best": "Horizon combination: best per horizon", "weighted": "Horizon combination: weighted per horizon"}
EXCLUDE = ("Look-ahead check", FAMILY)


def buckets(cfg, H):
    """The horizon buckets, clipped to the horizon and extended to cover every quarter up to it.

    Args:
        cfg: configuration (models.horizon_combination.buckets). H: forecast horizon.

    Returns:
        list of (first, last) quarter pairs.
    """
    raw = ((cfg.get("models") or {}).get("horizon_combination") or {}).get("buckets") or [[1, 4], [5, 8], [9, 12]]
    out = []
    for b in raw:
        a, z = (int(b[0]), int(b[-1])) if isinstance(b, (list, tuple)) else (int(b), int(b))
        a, z = max(1, a), min(H, z)
        if a <= z:
            out.append((a, z))
    out.sort()
    covered = set(h for a, z in out for h in range(a, z + 1))
    missing = [h for h in range(1, H + 1) if h not in covered]
    if missing:
        out.append((min(missing), max(missing)))
        out.sort()
    return out


def _bucket(h, B):
    """Index of the bucket that holds horizon h (the last bucket for horizons beyond it)."""
    for i, (a, z) in enumerate(B):
        if a <= h <= z:
            return i
    return len(B) - 1


def _sq_err(P, cfg):
    """Squared error used to rank models within a bucket (see the module notes).

    Args:
        P: test rows. cfg: configuration (models.horizon_combination.criterion: growth | balance).

    Returns:
        Series.
    """
    crit = str(((cfg.get("models") or {}).get("horizon_combination") or {}).get("criterion", "growth")).lower()
    if crit == "balance" and "error_pct" in P:
        return (P["error_pct"] / 100) ** 2
    return ((P["forecast_growth_pct"] - P["actual_growth_pct"]) / 100) ** 2


def _choose(hist, models, B, top_k, min_hist):
    """Best and weighted choice per bucket from realised forecast errors.

    Args:
        hist: realised test rows (model, origin, horizon_q, sq_err). models: candidates available now.
        B: buckets. top_k: models in the weighted version. min_hist: origins needed.

    Returns:
        list per bucket of dict(best=name, weights={name: w}, fallback=bool, mse=float).
    """
    hist = hist[hist["model"].isin(models)]
    overall = hist.groupby("model")["sq_err"].mean()
    out = []
    for a, z in B:
        hb = hist[(hist["horizon_q"] >= a) & (hist["horizon_q"] <= z)]
        st = hb.groupby("model").agg(mse=("sq_err", "mean"), n=("origin", "nunique"))
        ok = st[st["n"] >= min_hist]
        if len(ok):
            top = ok.sort_values("mse").head(top_k)
            w = (1 / top["mse"].clip(lower=1e-12))
            out.append({"best": top.index[0], "weights": (w / w.sum()).to_dict(), "fallback": False,
                        "mse": float(top["mse"].iloc[0])})
        else:
            if len(overall):
                b = overall.idxmin()
            else:
                b = "AR(1) benchmark" if "AR(1) benchmark" in models else sorted(models)[0]
            out.append({"best": b, "weights": {b: 1.0}, "fallback": True, "mse": np.nan})
    return out


def _splice(cur, choice, B, how):
    """One origin's combined growth path.

    Args:
        cur: test rows of every model at this origin. choice: _choose() result. B: buckets. how: best | weighted.

    Returns:
        DataFrame (one row per horizon) with the combined growth and the actual growth, or None.
    """
    rows = []
    for h in sorted(cur["horizon_q"].unique()):
        c = choice[_bucket(h, B)]
        w = {c["best"]: 1.0} if how == "best" else c["weights"]
        ch = cur[(cur["horizon_q"] == h) & cur["model"].isin(w)]
        if set(ch["model"]) != set(w):
            break                                 # a chosen model has no forecast this far ahead: stop the path
        g = float(sum(w[m] * v for m, v in zip(ch["model"], ch["forecast_growth_pct"])))
        r0 = ch.iloc[0]
        rows.append({"horizon_q": int(h), "date": r0["date"], "forecast_growth_pct": g,
                     "actual_growth_pct": float(r0["actual_growth_pct"]), "balance_at_origin": float(r0["balance_at_origin"]),
                     "actual_balance_organic": float(r0["actual_balance_organic"]),
                     "actual_balance_reported": float(r0.get("actual_balance_reported", np.nan)),
                     "covid_window": bool(r0.get("covid_window", False)), "chosen": ", ".join(
                         f"{m} ({w[m]:.0%})" if how == "weighted" else m for m in w)})
    if not rows:
        return None
    d = pd.DataFrame(rows)
    fc, act = tg.from_display(d["forecast_growth_pct"].values), tg.from_display(d["actual_growth_pct"].values)
    d["forecast_balance"] = tg.path(d["balance_at_origin"].values, fc)
    d["error"] = tg.error(np.cumsum(fc), np.cumsum(act))
    d["error_pct"] = d["error"] * 100
    return d


def combine(paths, cfg, H, wanted=lambda name: True):
    """Horizon combinations on a set of test forecast paths (backtest or out of sample).

    Args:
        paths: one row per model, origin and quarter ahead (columns model, family, origin, date, horizon_q,
        forecast_growth_pct, actual_growth_pct, balance_at_origin, actual_balance_organic, covid_window).
        cfg: configuration. H: horizon. wanted: filter on the combination names (models.only).

    Returns:
        (combined paths in the same format plus error, choices per origin and bucket); (None, None) when switched off.
    """
    hc = (cfg.get("models") or {}).get("horizon_combination") or {}
    kinds = [k for k in ("best", "weighted") if wanted(NAMES[k])]
    if not hc.get("enabled", True) or paths is None or paths.empty or not kinds:
        return None, None
    B = buckets(cfg, H)
    top_k, min_hist = int(hc.get("top_k", 3)), int(hc.get("min_history", 3))
    P = paths[~paths["family"].isin(EXCLUDE)].copy()
    P["sq_err"] = _sq_err(P, cfg)
    out, choices = [], []
    for o in sorted(P["origin"].unique()):
        cur = P[P["origin"] == o]
        hist = P[(P["origin"] < o) & (P["date"] <= o)]              # forecasts whose quarter had happened by o
        ch = _choose(hist, set(cur["model"]), B, top_k, min_hist)
        for k in kinds:
            d = _splice(cur, ch, B, k)
            if d is None:
                continue
            d.insert(0, "origin", o)
            d.insert(0, "family", FAMILY)
            d.insert(0, "model", NAMES[k])
            out.append(d)
        for (a, z), c in zip(B, ch):
            choices.append({"origin": o, "quarters_ahead": f"{a}-{z}", "best": c["best"],
                            "weighted": ", ".join(f"{m} ({w:.0%})" for m, w in c["weights"].items()),
                            "fallback (too little history)": c["fallback"]})
    if not out:
        return None, None
    return pd.concat(out, ignore_index=True), pd.DataFrame(choices)


def by_horizon(rows):
    """Errors by quarters ahead, in the format of the backtest table (balance error, COVID windows left out).

    Args:
        rows: combined paths from combine().

    Returns:
        DataFrame: model, family, horizon_q, MAPE_all_pct, MAPE_exCOVID_pct, bias_exCOVID_pct, n_origins, re_estimations.
    """
    out = []
    for (m, f), d in rows.groupby(["model", "family"], sort=False):
        for h, g in d.groupby("horizon_q"):
            x = g[~g["covid_window"]]
            out.append({"model": m, "family": f, "horizon_q": int(h), "MAPE_all_pct": g["error_pct"].abs().mean(),
                        "MAPE_exCOVID_pct": x["error_pct"].abs().mean(), "bias_exCOVID_pct": x["error_pct"].mean(),
                        "n_origins": int(len(x)), "re_estimations": np.nan})
    return pd.DataFrame(out)


def final_choice(paths, cfg, H):
    """The choice for the real forecast: every realised test error counts.

    Args:
        paths: test forecast paths. cfg: configuration. H: horizon.

    Returns:
        (buckets, choice per bucket) or (buckets, None).
    """
    hc = (cfg.get("models") or {}).get("horizon_combination") or {}
    B = buckets(cfg, H)
    P = paths[~paths["family"].isin(EXCLUDE)].copy()
    if P.empty:
        return B, None
    P["sq_err"] = _sq_err(P, cfg)
    return B, _choose(P, set(P["model"]), B, int(hc.get("top_k", 3)), 1)


def scenarios(scen, B, choice, wanted=lambda name: True):
    """Scenario forecasts of the horizon combinations, spliced from the chosen models' scenario paths.

    Args:
        scen: scenario forecasts of every model (model, family, scenario, date, dlog_forecast, balance_forecast).
        B: buckets. choice: final_choice() result. wanted: filter on the names.

    Returns:
        DataFrame in the same format (empty if nothing to splice).
    """
    out = []
    for k in ("best", "weighted"):
        if not wanted(NAMES[k]) or choice is None:
            continue
        for sc, d in scen.groupby("scenario", sort=False):
            piv = d.pivot_table(index="date", columns="model", values="dlog_forecast", sort=False)
            dates = list(dict.fromkeys(d["date"]))
            first = d.iloc[0]
            L0 = float(tg.level_after(first["balance_forecast"], -first["dlog_forecast"]))
            g = []
            for h, dt in enumerate(dates, 1):
                c = choice[_bucket(h, B)]
                w = {c["best"]: 1.0} if k == "best" else c["weights"]
                if not all(m in piv.columns for m in w):
                    break
                g.append(sum(wi * piv.loc[dt, m] for m, wi in w.items()))
            if not g:
                continue
            lvl = tg.path(L0, g)
            out += [{"model": NAMES[k], "family": FAMILY, "scenario": sc, "date": dt, "dlog_forecast": gi,
                     "balance_forecast": li} for dt, gi, li in zip(dates, g, lvl)]
    return pd.DataFrame(out)


def describe(B, choice, kind):
    """Model-list text and driver-table rows for a horizon combination.

    Args:
        B: buckets. choice: final_choice() result. kind: best | weighted.

    Returns:
        (drivers text, list of driver-table rows).
    """
    rows, parts = [], []
    for (a, z), c in zip(B, choice or []):
        w = {c["best"]: 1.0} if kind == "best" else c["weights"]
        txt = ", ".join(f"{m} ({v:.0%})" if kind == "weighted" else m for m, v in w.items())
        parts.append(f"{a}-{z}q: {txt}")
        for m, v in w.items():
            rows.append({"model": NAMES[kind], "term": f"quarters {a}-{z}: {m}", "coef": v,
                         "reading": f"weight {v:.0%} in quarters {a}-{z} ahead" + (
                             f" (squared error {c['mse']:.2e})" if c["mse"] == c["mse"] else "")})
    return "; ".join(parts), rows
