"""Variable transformations, stationarity tests and the candidate feature list.

A feature is one variable in one transformation at one lag, named  'MNEMONIC|transform|L<lag>'.
All rolling windows are one-sided (past data only).
"""
import re

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

TR_DESC = {
    "dlog": "q/q % growth", "yoy": "y/y % growth", "dlog2": "2-quarter % growth",
    "acc": "acceleration (change in y/y growth vs a year ago)", "gap": "% gap from 3-year trailing average",
    "rdlog": "real q/q % growth", "ryoy": "real y/y % growth",
    "diff": "q/q change", "diff2": "2-quarter change", "diff4": "y/y change", "lvl": "level",
    "rgap": "gap from 3-year trailing average (pp)",
}
RATE_TRS = {"diff", "diff2", "diff4", "lvl", "rgap"}

# Moving-window transforms, any window k >= 2 (k quarters, past data only):
#   level series (log):  ma<k>   average quarterly log growth over the last k quarters  = ln(x_t / x_t-k) / k
#                        vol<k>  standard deviation of quarterly log growth over the last k quarters (volatility)
#   rate series:         dma<k>  average quarterly change over the last k quarters      = (x_t - x_t-k) / k
#                        dvol<k> standard deviation of the quarterly change over the last k quarters
# ma4 equals yoy / 4 and ma2 equals dlog2 / 2, so they add nothing new: use longer windows such as ma8 or ma12.
# Volatility has its own sign (features.volatility_sign, default -1: more uncertainty, less growth), because the
# variable's own sign (e.g. +1 for house prices) says nothing about the effect of its volatility.
WINDOW = re.compile(r"^(d?)(ma|vol)(\d+)$")
_SETTINGS = {"volatility_sign": -1}


def configure(cfg):
    """Read the transform settings of a run (features.volatility_sign)."""
    _SETTINGS["volatility_sign"] = int((cfg.get("features") or {}).get("volatility_sign", -1))


def window(tr):
    """(is_rate, kind 'ma'|'vol', k) for a moving-window transform, else None."""
    m = WINDOW.match(str(tr))
    return (m.group(1) == "d", m.group(2), int(m.group(3))) if m and int(m.group(3)) >= 2 else None


def is_rate_tr(tr):
    """True for transforms of rate-type series (units: percentage points)."""
    w = window(tr)
    return tr in RATE_TRS or (w is not None and w[0])


def is_volatility(tr):
    w = window(tr)
    return w is not None and w[1] == "vol"


def tr_desc(tr):
    """Plain-language description of any transform, including moving windows."""
    w = window(tr)
    if w:
        rate, kind, k = w
        what = "q/q change" if rate else "q/q % growth"
        return f"{k}-quarter average {what}" if kind == "ma" else f"volatility ({k}-quarter std. dev. of {what})"
    return TR_DESC.get(tr, tr)


def lookback(tr):
    """Quarters of history a transform needs before the current one."""
    w = window(tr)
    if w:
        return w[2] if w[1] == "ma" else w[2]
    return {"dlog": 1, "rdlog": 1, "dlog2": 2, "yoy": 4, "ryoy": 4, "acc": 8, "gap": 11, "rgap": 11,
            "diff": 1, "diff2": 2, "diff4": 4, "lvl": 0}.get(tr, 1)


def sign_of(meta, code, tr):
    """Expected sign of a feature: the variable's sign, or features.volatility_sign for volatility transforms."""
    base = int(meta.loc[code, "expected_sign"]) if code in meta.index else 0
    if is_volatility(tr) and base != 0:
        return _SETTINGS["volatility_sign"]
    return base


def feature_sign(meta, feature):
    """sign_of() for a feature name."""
    code, tr, _ = parse_feature(feature)
    return sign_of(meta, code, tr)


def transform(mdf, code, tr, deflator=None):
    """Apply one stationary transform to one macro series.

    Args:
        mdf: macro DataFrame (quarters x macro mnemonics).
        code: mnemonic of the series, e.g. "FHPI.ION".
        tr: transform code: diff, diff2, diff4, lvl, rgap, dlog, yoy, dlog2, acc, gap, rdlog (real growth).
        deflator: mnemonic of the price index, used by the real transforms.

    Returns:
        Series on the same quarterly index (NaN where the transform needs more history).
    """
    if str(code).startswith("FAC."):                  # theme factor: always rebuilt from its members
        from . import factors
        sp = factors.spec_for(mdf, code)
        x = factors.values(mdf, sp) if sp else mdf[code]
    else:
        x = mdf[code]
    w = window(tr)
    if w:
        rate, kind, k = w
        g = x.diff() if rate else np.log(x).diff()
        return g.rolling(k, min_periods=k).mean() if kind == "ma" else g.rolling(k, min_periods=k).std()
    if tr == "diff":
        return x.diff()
    if tr == "diff2":
        return x.diff(2)
    if tr == "diff4":
        return x.diff(4)
    if tr == "lvl":
        return x
    if tr == "rgap":
        return x - x.rolling(12, min_periods=12).mean()
    lx = np.log(x)
    if tr == "dlog":
        return lx.diff()
    if tr == "yoy":
        return lx.diff(4)
    if tr == "dlog2":
        return lx.diff(2)
    if tr == "acc":
        return lx.diff(4) - lx.diff(4).shift(4)
    if tr == "gap":
        return lx - lx.rolling(12, min_periods=12).mean()
    if tr in ("rdlog", "ryoy"):
        k = 1 if tr == "rdlog" else 4
        return lx.diff(k) - np.log(mdf[deflator]).diff(k)
    raise ValueError(f"unknown transform {tr}")


def feature_name(code, tr, lag):
    """Build a feature name "code|transform|Llag" (the format used everywhere in the project).

    Args:
        code: series mnemonic. tr: transform code. lag: lag in quarters.

    Returns:
        str, e.g. "FHPI.ION|dlog|L1".
    """
    return f"{code}|{tr}|L{lag}"


def parse_feature(name):
    """Split a feature name into its parts (inverse of feature_name).

    Args:
        name: "code|transform|Llag".

    Returns:
        (code, transform, lag) with lag as int.
    """
    code, tr, lag = name.rsplit("|", 2)
    return code, tr, int(lag[1:])


def feature_series(mdf, name, deflator=None):
    """Compute one feature: transform the series, then lag it.

    Args:
        mdf: macro DataFrame. name: feature name. deflator: price index mnemonic for real transforms.

    Returns:
        Series indexed by quarter.
    """
    code, tr, lag = parse_feature(name)
    return transform(mdf, code, tr, deflator).shift(lag)


def feature_matrix(mdf, names, deflator=None):
    """Compute several features side by side.

    Args:
        mdf: macro DataFrame (history or a scenario). names: list of feature names. deflator: see transform.

    Returns:
        DataFrame, one column per feature, on mdf's index.
    """
    return pd.DataFrame({n: feature_series(mdf, n, deflator) for n in names}, index=mdf.index)


def describe(name, meta):
    """Plain-language label of a feature for tables and charts.

    Args:
        name: feature name. meta: metadata table indexed by mnemonic (column "description").

    Returns:
        str, e.g. "House prices, Ontario (quarterly log change, lag 1Q)".
    """
    code, tr, lag = parse_feature(name)
    d = meta.loc[code, "description"] if code in meta.index else code
    if str(code).startswith("FAC.") and tr == "lvl":
        return f"{d} ({'same quarter' if lag == 0 else f'lag {lag}Q'})"
    return f"{d} ({tr_desc(tr)}, {'same quarter' if lag == 0 else f'lag {lag}Q'})"


def candidate_forms(cfg, hist, meta):
    """Relevance filter + every transformation of every variable + ADF test. One row per (variable, transform).

    Args:
        hist: macro history (DataFrame).
    """
    f = cfg["features"]
    configure(cfg)
    rows = []
    for code in hist.columns:
        if code not in meta.index:
            continue
        geo, theme = meta.loc[code, "geography"], meta.loc[code, "theme"]
        is_factor = meta.loc[code, "transform"] == "factor"       # members already filtered by geography
        if f["allowed_geographies"] and not is_factor and not any(str(geo).startswith(g) for g in f["allowed_geographies"]):
            continue
        if theme in (f["excluded_themes"] or []):
            continue
        if f["require_sign_prior"] and meta.loc[code, "expected_sign"] == 0:
            continue
        x = hist[code].dropna()
        is_level = meta.loc[code, "transform"] == "dlog" and len(x) and (x > 0).all()
        if meta.loc[code, "transform"] == "factor":          # theme factors are already growth/change indexes
            trs = ["lvl"]
        elif f.get("form_rule", "all") == "by_type":          # one form per variable: growth or change
            trs = ["dlog"] if is_level else ["diff"]
        else:
            trs = list(f["transforms_level"]) if is_level else list(f["transforms_rate"])
        # moving windows: features.moving_average / features.volatility = lists of windows, e.g. [8] and [4, 8]
        pre = "" if is_level else "d"
        trs += [f"{pre}ma{k}" for k in (f.get("moving_average") or [])]
        trs += [f"{pre}vol{k}" for k in (f.get("volatility") or [])]
        if meta.loc[code, "transform"] == "factor":
            trs = ["lvl"]
        elif is_level and theme in (f["real_themes"] or []) and f["deflator"] in hist.columns:
            trs += ["rdlog"] if f.get("form_rule", "all") == "by_type" else ["rdlog", "ryoy"]
        for tr in trs:
            s = transform(hist, code, tr, f["deflator"]).dropna()
            try:
                p = adfuller(s, autolag="AIC")[1] if (len(s) >= 30 and s.std() > 0) else 1.0
            except Exception:
                p = 1.0
            rows.append({"mnemonic": code, "transform": tr, "adf_p": p, "stationary": p < f["adf_p"],
                         "theme": theme, "geography": geo, "expected_sign": sign_of(meta, code, tr)})
    return pd.DataFrame(rows)


def all_features(cfg, ctx, forms):
    """All stationary (variable, transform) pairs at every lag, with no gaps in the estimation window."""
    lags = cfg["features"].get("lags") or range(cfg["features"]["max_lag"] + 1)    # features.lags: e.g. [0, 1, 4]
    names = [feature_name(r.mnemonic, r.transform, int(l)) for r in forms[forms["stationary"]].itertuples()
             for l in lags]
    F = feature_matrix(ctx.hist, names, cfg["features"]["deflator"])
    window = F.iloc[ctx.est0:ctx.end + 1]
    keep = window.columns[window.notna().all() & (window.std() > 0)]
    return F[keep]
