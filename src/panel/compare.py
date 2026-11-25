"""Comparing the panel models: any error metric, all banks or some, one bank at a time.

Everything is computed from the saved forecast errors (4_backtest_forecasts.csv, oos/4_forecasts.csv), so a
different metric or a different set of banks needs no re-estimation:

    python tools/panel_compare.py --run outputs/<run> --metric rmse --banks "Bank A" "Bank C" --bank "Bank A"

Metrics (each is computed for every bank, model and horizon over the forecast origins, then averaged over
horizons 1..h, so the naturally larger long-horizon errors do not dominate):

    mape      average absolute error, % of the actual level (for a rate or ratio target: in its units)
    rmse      root mean squared error: large misses weigh more
    p90       the error exceeded in 10% of the forecasts
    cvar      the average of the worst tail_share of errors: how bad the bad forecasts are
    worst     the single largest error at any horizon (not averaged)
    combined  mape + penalty x (cvar - mape): the average, plus a charge for the bad forecasts
    bias      average signed error, forecast / actual - 1 (positive = the model over-forecasts); shown, not ranked

Which banks: by default the ACTIVE ones (still reporting at the end of history; the ones you forecast), or all,
or a list. Across banks: the score of a model is the average of its bank scores (equal weights, or by size with
weighting: size), so every bank counts; 'pooled' also gives the metric over all banks' errors together.
"""
import numpy as np
import pandas as pd

METRICS = {"mape": "MAPE", "rmse": "RMSE", "p90": "90th percentile", "cvar": "Worst-tail average",
           "worst": "Largest error", "combined": "MAPE + tail penalty"}
LONG = {"mape": "average absolute error (MAPE)", "rmse": "root mean squared error",
        "p90": "90th percentile of the absolute errors", "cvar": "average of the worst {tail:.0%} of errors",
        "worst": "largest single error", "combined": "MAPE + {pen:g} x (worst-{tail:.0%} average - MAPE)"}
DEFAULTS = {"metric": "mape", "banks": "active", "horizon": None, "tail_share": 0.10, "penalty": 0.5,
            "weighting": "equal", "benchmark": None, "exclude_covid": True}


def settings(cfg):
    """panel.evaluation with the defaults filled in."""
    e = dict(DEFAULTS)
    e.update(((cfg.get("panel") or {}).get("evaluation") or {}))
    if e["metric"] not in METRICS:
        raise ValueError(f"panel.evaluation.metric must be one of {list(METRICS)}")
    return e


def describe(metric, tail_share=0.10, penalty=0.5):
    return LONG[metric].format(tail=tail_share, pen=penalty)


def _one(a, tail, pen):
    """Metrics of one set of signed errors."""
    a = np.asarray(a, float)
    a = a[~np.isnan(a)]
    if not len(a):
        return dict.fromkeys(list(METRICS) + ["bias"], np.nan)
    s = np.sort(np.abs(a))
    k = max(int(np.ceil(len(s) * tail)), 1)
    mape, cvar = s.mean(), s[-k:].mean()
    return {"mape": mape, "rmse": float(np.sqrt((s ** 2).mean())), "p90": float(np.quantile(s, 0.9)),
            "cvar": float(cvar), "worst": float(s[-1]), "combined": float(mape + pen * (cvar - mape)),
            "bias": float(a.mean())}


def load_errors(pdir, oos=False):
    """The saved forecast errors of a panel run (backtest, or the out-of-sample test)."""
    import os
    f = os.path.join(pdir, "oos", "4_forecasts.csv") if oos else os.path.join(pdir, "4_backtest_forecasts.csv")
    if not os.path.exists(f):
        raise FileNotFoundError(f"{f} not found: run the panel analysis{' with the out-of-sample test' if oos else ''}")
    return pd.read_csv(f)


def evaluate(E, metric="mape", banks=None, horizon=None, tail_share=0.10, penalty=0.5, weighting="equal",
             benchmark=None, exclude_covid=True, active=None, **_):
    """Compare the models on a metric, for some or all banks.

    Args:
        E: error rows (model, kind, bank, origin, horizon_q, error_pct, covid_window, level_at_origin).
        metric: one of METRICS (the ranking). banks: list of banks, 'active' (the banks still reporting at the end of
            history, given in `active`; all banks if not given), or None / 'all'. horizon: 1..h (default: all).
        tail_share, penalty: for cvar and combined. weighting: equal or size (latest level at origin).
        benchmark: model name for the 'vs' columns (default: the AR(1) benchmark).

    Returns:
        dict with overall (one row per model), by_bank (bank x model), by_horizon (model x horizon, averaged
        across banks), and the settings used.
    """
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {list(METRICS)}")
    E = E.copy()
    if isinstance(banks, str) and banks.lower() == "active":
        banks = [b for b in (active or []) if b in set(E["bank"].astype(str))] or None
    if exclude_covid and "covid_window" in E:
        E = E[~E["covid_window"].astype(bool)]
    if banks not in (None, "all", []):
        want = {str(b) for b in banks}
        missing = want - set(E["bank"].astype(str))
        if missing:
            raise ValueError(f"no errors for {sorted(missing)}; banks are {sorted(E['bank'].astype(str).unique())}")
        E = E[E["bank"].astype(str).isin(want)]
    H = int(horizon or E["horizon_q"].max())
    E = E[E["horizon_q"] <= H]
    if E.empty:
        raise ValueError("no forecast errors left for these banks and horizons")
    kinds = E.drop_duplicates("model").set_index("model")["kind"].to_dict() if "kind" in E else {}
    if benchmark is None:
        benchmark = next((m for m, k in kinds.items() if k == "ar1"), None)
    keys = list(METRICS) + ["bias"]

    # bank x model x horizon, then averaged over horizons (worst: the maximum)
    per = [{"bank": b, "model": m, "horizon_q": h, "n": len(g), **_one(g["error_pct"].values, tail_share, penalty)}
           for (b, m, h), g in E.groupby(["bank", "model", "horizon_q"], sort=False)]
    P = pd.DataFrame(per)
    BB = P.groupby(["bank", "model"], sort=False)[keys].mean()
    BB["worst"] = P.groupby(["bank", "model"], sort=False)["worst"].max()
    BB["forecasts"] = P.groupby(["bank", "model"], sort=False)["n"].sum()
    BB = BB.reset_index()
    BB.insert(2, "kind", BB["model"].map(kinds))
    BB.insert(3, "score", BB[metric])
    BB["rank_in_bank"] = BB.groupby("bank")["score"].rank(method="min")
    if benchmark is not None and benchmark in set(BB["model"]):
        ref = BB[BB["model"] == benchmark].set_index("bank")["score"]
        BB["vs_benchmark_pct"] = (1 - BB["score"] / BB["bank"].map(ref)) * 100

    # across banks: mean of bank scores (equal or size weights), median, pooled
    size = E.sort_values("origin").groupby("bank")["level_at_origin"].last() if "level_at_origin" in E else None
    rows = []
    for m, g in BB.groupby("model", sort=False):
        w = g["bank"].map(size).fillna(0).values + 1e-12 if (weighting == "size" and size is not None) \
            else np.ones(len(g))
        r = {"model": m, "kind": kinds.get(m), "score": float(np.average(g["score"], weights=w))}
        for k in keys:
            r[k] = float(np.average(g[k], weights=w))
        r["median_bank_score"] = float(g["score"].median())
        Em = E[E["model"] == m]
        pooled = [_one(d["error_pct"].values, tail_share, penalty)[metric] for _, d in Em.groupby("horizon_q")]
        r["pooled_score"] = float(np.max(pooled) if metric == "worst" else np.mean(pooled)) if pooled else np.nan
        r["banks_won"] = int((g["rank_in_bank"] == 1).sum())
        if "vs_benchmark_pct" in g:
            r["banks_better_than_benchmark"] = int((g["vs_benchmark_pct"] > 0).sum())
        r["banks"] = len(g)
        rows.append(r)
    O = pd.DataFrame(rows).sort_values("score").reset_index(drop=True)
    if benchmark is not None and benchmark in set(O["model"]):
        O["vs_benchmark_pct"] = (1 - O["score"] / float(O.loc[O["model"] == benchmark, "score"].iloc[0])) * 100
    O.insert(0, "rank", range(1, len(O) + 1))

    # by horizon, averaged across banks (the chart)
    HZ = P.groupby(["model", "horizon_q"], sort=False)[metric].mean().reset_index().rename(columns={metric: "score"})
    used = {"metric": metric, "metric_long": describe(metric, tail_share, penalty), "banks": sorted(E["bank"].unique()),
            "horizon": H, "tail_share": tail_share, "penalty": penalty, "weighting": weighting,
            "benchmark": benchmark, "exclude_covid": exclude_covid}
    return {"overall": O, "by_bank": BB.sort_values(["bank", "score"]).reset_index(drop=True), "by_horizon": HZ,
            "settings": used}


def one_bank(res, bank):
    """The comparison for one bank, ranked."""
    d = res["by_bank"]
    return d[d["bank"].astype(str) == str(bank)].sort_values("score").reset_index(drop=True)


def write(res, folder, prefix=""):
    """Write the evaluation tables next to the leaderboards."""
    import json
    import os
    os.makedirs(folder, exist_ok=True)
    res["overall"].round(4).to_csv(os.path.join(folder, f"{prefix}10_evaluation_overall.csv"), index=False)
    res["by_bank"].round(4).to_csv(os.path.join(folder, f"{prefix}11_evaluation_by_bank.csv"), index=False)
    res["by_horizon"].round(4).to_csv(os.path.join(folder, f"{prefix}12_evaluation_by_horizon.csv"), index=False)
    with open(os.path.join(folder, f"{prefix}10_evaluation_settings.json"), "w", encoding="utf-8") as f:
        json.dump(res["settings"], f, indent=1, default=str)
