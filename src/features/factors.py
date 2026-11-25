"""Theme factors: one index per theme from all its series (targeted diffusion indexes).

With 200+ macro series, many measure the same thing (house prices in a dozen cities). Picking one of them is partly
luck, and the pick changes from quarter to quarter. A factor summarises a theme's series into one index:

1. Members: every series of the theme with full history in the estimation window, in one form by type: growth
   (log change) for levels, change for rates. Themes that are other outcomes of the economy (credit, deposits) are
   left out unless factors.exclude_outcomes is false.
2. Targeted (Bai & Ng 2008, "Forecasting economic time series using targeted predictors", Journal of
   Econometrics 146): only members that are related to the target are kept - |t| above factors.t_threshold when
   added (at their best lag, 0..max_lag) to  y_t = a + rho*y_{t-1} + b*x_{t-k} + dummies.
   If fewer than factors.min_members pass, the strongest min_members are used.
3. Principal component (Stock & Watson 2002, "Macroeconomic forecasting using diffusion indexes", Journal of
   Business & Economic Statistics 20): the members are standardised over the estimation window and the first
   principal component is the factor (factors.n_factors > 1 adds the next ones). Its sign is set so it rises when
   the member with the largest loading rises; its expected sign comes from the members' expected signs.

A factor is a series named FAC.<THEME> (FAC.<THEME>_2 for the second) added to every macro scenario, with a
metadata row, so every approach and model can use it like any other series. Its value is always recomputed from
its members with the fixed weights, means and standard deviations, so a shock to a member (what-if, scenarios)
moves the factor too. In the out-of-sample test the weights are re-estimated at every origin with data up to it.

Writes outputs/<run>/factors/: 1_factors.csv (one row per factor), 2_loadings.csv (one row per member).
"""
import re

import numpy as np
import pandas as pd

SPECS = {}          # active factor definitions: code -> spec (used by features.transforms.transform)
REGISTRY = {}       # key -> definitions; a frame carries only the key (pandas deep-copies attrs on every operation)


def register(specs):
    """Store a set of factor definitions; returns its short key."""
    import hashlib
    import json
    key = hashlib.sha1(json.dumps(specs, sort_keys=True, default=str).encode()).hexdigest()[:12]
    REGISTRY[key] = specs
    return key


def settings(cfg):
    d = {"enabled": True, "targeted": True, "t_threshold": 1.65, "min_members": 3, "n_factors": 1,
         "exclude_outcomes": True, "min_share": 0.0}
    d.update({k: v for k, v in (cfg.get("factors") or {}).items() if v is not None})
    return d


def slug(theme):
    return re.sub(r"[^A-Z0-9]+", "_", str(theme).upper()).strip("_")[:24] or "THEME"


def member_kind(meta, hist, code):
    """'dlog' for positive level series modelled in growth, else 'diff'."""
    x = hist[code].dropna()
    return "dlog" if (str(meta.loc[code, "transform"]) == "dlog" and len(x) and (x > 0).all()) else "diff"


def member_values(mdf, code, kind):
    x = mdf[code].astype(float)
    return np.log(x).diff() if kind == "dlog" else x.diff()


def spec_for(mdf, code):
    """The definition of factor `code` for this frame (its own, else the active ones), or None."""
    key = (getattr(mdf, "attrs", {}) or {}).get("factor_key")
    sp = REGISTRY.get(key) or {}
    return sp.get(code) or SPECS.get(code)


def values(mdf, spec):
    """The factor on mdf's quarters, from its members (NaN where a member is missing)."""
    z = None
    for m in spec["members"]:
        if m["code"] not in mdf.columns:
            return pd.Series(np.nan, index=mdf.index)
        v = (member_values(mdf, m["code"], m["kind"]) - m["mean"]) / m["sd"] * m["weight"]
        z = v if z is None else z + v
    return z if z is not None else pd.Series(np.nan, index=mdf.index)


def activate(specs):
    SPECS.clear()
    SPECS.update(specs or {})


def _t_best_lag(ctx, x, max_lag):
    """Largest |t| (and its signed value and lag) of x at lags 0..max_lag added to y ~ const + y_lag1 + dummies."""
    rows = slice(ctx.est0, ctx.end + 1)
    y = ctx.y.iloc[rows].values
    W = [np.ones(len(y)), ctx.y.shift(1).iloc[rows].values]
    for c in ctx.dummies.columns:
        d = ctx.dummies[c].iloc[rows].values
        if d.std() > 0:
            W.append(d)
    W = np.column_stack(W)
    best = (0.0, 0.0, 0)
    for k in range(max_lag + 1):
        v = x.shift(k).reindex(ctx.idx).iloc[rows].values
        ok = ~np.isnan(v) & ~np.isnan(W).any(axis=1) & ~np.isnan(y)
        if ok.sum() < 20 or np.std(v[ok]) == 0:
            continue
        X = np.column_stack([W[ok], v[ok]])
        b, *_ = np.linalg.lstsq(X, y[ok], rcond=None)
        e = y[ok] - X @ b
        s2 = e @ e / max(ok.sum() - X.shape[1], 1)
        se = np.sqrt(max(np.linalg.pinv(X.T @ X)[-1, -1] * s2, 1e-30))
        t = b[-1] / se
        if abs(t) > abs(best[0]):
            best = (abs(t), t, k)
    return best


def build(cfg, ctx, log=print):
    """Factor definitions from the data up to ctx.end.

    Returns:
        (specs dict, metadata rows DataFrame, loadings DataFrame).
    """
    st = settings(cfg)
    meta, hist = ctx.meta, ctx.base.reindex(ctx.idx)
    blocked = set()
    if st["exclude_outcomes"]:
        from ..selection.double_selection import outcome_themes
        blocked = outcome_themes(cfg, ctx)
    excluded = set((cfg.get("features") or {}).get("excluded_themes") or [])
    max_lag = int(cfg["features"]["max_lag"])
    win = slice(ctx.est0, ctx.end + 1)
    specs, mrows, lrows = {}, [], []
    geos = (cfg.get("features") or {}).get("allowed_geographies") or []
    codes = [c for c in hist.columns if c in meta.index and not str(c).startswith("FAC.")
             and (not geos or any(str(meta.loc[c, "geography"]).startswith(g) for g in geos))]
    for theme, grp in meta.loc[codes].groupby("theme"):
        if theme in blocked or theme in excluded or not str(theme).strip():
            continue
        cand = []
        for code in grp.index:
            kind = member_kind(meta, hist, code)
            v = member_values(hist, code, kind)
            w = v.iloc[win]
            if w.isna().any() or w.std() == 0 or len(w) < 20:
                continue
            at, t, k = _t_best_lag(ctx, v, max_lag) if st["targeted"] else (1.0, 1.0, 0)
            cand.append({"code": code, "kind": kind, "abs_t": at, "t": t, "lag": k, "mean": float(w.mean()),
                         "sd": float(w.std()), "expected_sign": int(np.sign(meta.loc[code, "expected_sign"] or 0))})
        if len(cand) < st["min_members"]:
            continue
        cand.sort(key=lambda r: -r["abs_t"])
        keep = [r for r in cand if r["abs_t"] >= st["t_threshold"]] if st["targeted"] else cand
        how = "targeted" if st["targeted"] else "all members"
        if len(keep) < st["min_members"]:
            keep, how = cand[:st["min_members"]], f"the {st['min_members']} strongest (too few above the threshold)"
        Z = np.column_stack([((member_values(hist, r["code"], r["kind"]).iloc[win] - r["mean"]) / r["sd"]).values
                             for r in keep])
        U, S, Vt = np.linalg.svd(Z - Z.mean(axis=0), full_matrices=False)
        share = S ** 2 / np.sum(S ** 2)
        for j in range(min(int(st["n_factors"]), len(keep) - 1 if len(keep) > 1 else 1)):
            if j > 0 and share[j] < float(st["min_share"] or 0):
                break
            w = Vt[j].copy()
            if w[np.argmax(np.abs(w))] < 0:                # rises when its main member rises
                w = -w
            code = f"FAC.{slug(theme)}" + ("" if j == 0 else f"_{j + 1}")
            es = sum(wi * r["expected_sign"] for wi, r in zip(w, keep))
            tot = sum(abs(wi) for wi, r in zip(w, keep) if r["expected_sign"])
            sign = int(np.sign(es)) if tot and abs(es) >= 0.5 * tot else 0
            members = [{"code": r["code"], "kind": r["kind"], "mean": r["mean"], "sd": r["sd"], "weight": float(wi)}
                       for wi, r in zip(w, keep)]
            specs[code] = {"theme": theme, "members": members, "share": float(share[j]), "how": how}
            top = sorted(zip(w, keep), key=lambda p: -abs(p[0]))[:3]
            lead = ", ".join(str(meta.loc[r["code"], "description"])[:40] if "description" in meta.columns
                             else r["code"] for _, r in top)
            mrows.append({"mnemonic": code, "description": f"{theme} index ({len(keep)} series"
                                                           + (")" if j == 0 else f", component {j + 1})"),
                          "main_series": lead,
                          "theme": theme, "geography": "All", "transform": "factor", "expected_sign": sign,
                          "members": len(keep), "of": len(cand), "variance_share": float(share[j]), "how": how})
            for wi, r in zip(w, keep):
                lrows.append({"factor": code, "theme": theme, "member": r["code"],
                              "description": meta.loc[r["code"], "description"] if "description" in meta.columns
                              else r["code"], "form": "growth" if r["kind"] == "dlog" else "change",
                              "loading": float(wi), "t_alone": r["t"], "best_lag": r["lag"],
                              "expected_sign": r["expected_sign"]})
    return specs, pd.DataFrame(mrows), pd.DataFrame(lrows)


def attach(cfg, ctx, log=print):
    """Build the factors on ctx's window and add them to ctx (scenarios, base, history, metadata); activate them.

    Returns:
        (metadata rows, loadings) for reporting.
    """
    if not settings(cfg)["enabled"]:
        activate({})
        ctx.factor_specs = {}
        return pd.DataFrame(), pd.DataFrame()
    specs, mrows, lrows = build(cfg, ctx, log)
    activate(specs)
    ctx.factor_specs = specs
    drop = [c for c in ctx.base.columns if str(c).startswith("FAC.")]
    key = register(specs)
    macro = {}
    for k, df in ctx.macro.items():
        df = df.drop(columns=drop, errors="ignore").copy()
        for code, sp in specs.items():
            df[code] = values(df, sp)
        df.attrs = {k: v for k, v in df.attrs.items() if k != "factor_specs"}
        df.attrs["factor_key"] = key            # travels with the frame (copies, slices, reindex)
        macro[k] = df
    ctx.macro = macro
    ctx.base = macro[cfg["data"]["macro"]["base_scenario"]]
    ctx.hist = ctx.base.reindex(ctx.idx)
    meta = ctx.meta.drop(index=[i for i in ctx.meta.index if str(i).startswith("FAC.")], errors="ignore")
    if len(mrows):
        add = mrows.set_index("mnemonic")[["description", "theme", "geography", "transform", "expected_sign"]]
        for c in meta.columns:
            if c not in add.columns:
                add[c] = "" if meta[c].dtype == object else np.nan
        meta = pd.concat([meta, add[meta.columns]])
    ctx.meta = meta
    return mrows, lrows
