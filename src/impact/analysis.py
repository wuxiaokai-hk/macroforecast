"""How the economy affects the target ("satellite model" questions), from the models of a finished run.

Shocks are applied to the macro data itself (the raw series, forecast quarters only), and every model rebuilds its
own drivers from them, so the same shock reaches regressions, error-correction, regime, machine-learning and
combination models alike. The response is the shocked forecast minus the baseline forecast, quarter by quarter.

Which models
  The explanatory model (focus drivers + double-selection controls, src/pipelines/focus_first.py) gives the main
  estimate, with a 90% band from its coefficient uncertainty (draws from the HAC covariance).
  The "tied" models are those not significantly worse than the most accurate model at impact.compare_horizon
  quarters ahead (Diebold-Mariano with the Harvey-Leybourne-Newbold correction, one-sided, impact.tied_alpha;
  at most impact.tied_max). Their spread shows how much the answer depends on the choice of model.

Concept shocks
  A model may measure a concept with another series (Toronto instead of Ontario unemployment). A shock to a focus
  series therefore also moves its look-alikes: series of the same theme and kind whose 4-quarter changes correlate
  above impact.lookalike_corr with it, each by its historical slope on the focus series. Without this, a model
  that uses a look-alike would show no effect at all.

Shock units
  Rate-type series (rates, ratios, unemployment): points (size 1 = +1 percentage point if the series is in %).
  Level-type series (prices, income, GDP): percent of the level (size 1 = the level 1% higher).
  "typical" shock = one standard deviation of the series' 4-quarter change over the history.
  Shocks build up linearly over `quarters` quarters and then stay (a permanent shift in the level of the series).

Outputs (outputs/<run>/impact/): 1_tied_models, 2_responses, 3_sensitivity, 4_reliability, 5_attribution,
6_what_if, summary.md and charts.
"""
import itertools
import os
import re

import numpy as np
import pandas as pd

from ..utils import target as tg

# ------------------------------------------------------------------------------------------------ helpers


def is_level(ctx, code):
    """Level-type series (positive, modelled in growth) vs rate-type (modelled in changes)."""
    x = ctx.base[code].dropna()
    t = str(ctx.meta.loc[code, "transform"]) if code in ctx.meta.index else ""
    return t == "dlog" and len(x) > 0 and bool((x > 0).all())


def _chg4(ctx, code):
    """4-quarter change over the history: log change x100 for levels, difference for rates."""
    x = ctx.hist[code].astype(float)
    return (np.log(x).diff(4) * 100) if is_level(ctx, code) else x.diff(4)


def typical_size(ctx, code):
    """One standard deviation of the 4-quarter change (the 'typical' shock)."""
    s = _chg4(ctx, code).dropna()
    return float(s.std()) if len(s) > 8 else 1.0


def unit_label(ctx, code, size):
    """'+1.00 pt' or '+1.00%'."""
    return f"{size:+.2f}{'%' if is_level(ctx, code) else ' pt'}"


def describe(ctx, code):
    """Series description from the metadata."""
    if code in ctx.meta.index and "description" in ctx.meta.columns:
        d = ctx.meta.loc[code, "description"]
        if isinstance(d, str) and d.strip():
            return d.strip()
    return code


def lookalikes(cfg, ctx, code):
    """{series: slope} of series that move with `code` (same theme and kind, correlated 4-quarter changes)."""
    thr = float((cfg.get("impact") or {}).get("lookalike_corr", 0.75))
    if code not in ctx.meta.index:
        return {}
    theme, lvl = ctx.meta.loc[code, "theme"], is_level(ctx, code)
    x = _chg4(ctx, code)
    out = {}
    for c in ctx.base.columns:
        if c == code or c not in ctx.meta.index or ctx.meta.loc[c, "theme"] != theme or is_level(ctx, c) != lvl:
            continue
        z = _chg4(ctx, c)
        ok = x.notna() & z.notna()
        if ok.sum() < 12 or z[ok].std() == 0:
            continue
        r = np.corrcoef(x[ok], z[ok])[0, 1]
        if r > thr:
            out[c] = float(np.polyfit(x[ok], z[ok], 1)[0])
    return out


def resolve_variable(ctx, v):
    """A concept id (literature table), a series name, or a feature name -> series name (or None)."""
    from ..models import literature as lit
    v = str(v).strip()
    if "|" in v:
        v = v.split("|")[0]
    if v in ctx.base.columns:
        return v
    if ctx.lit_table is not None:
        m = lit.resolve(v, ctx.lit_table)
        if m and m in ctx.base.columns:
            return m
    low = {c.lower(): c for c in ctx.base.columns}
    return low.get(v.lower())


def apply_shocks(cfg, ctx, mdf, shocks, with_lookalikes=True):
    """A copy of the macro frame with the shocks added in the forecast quarters.

    Args:
        mdf: macro frame (history + scenario). shocks: list of dicts {variable, size, quarters (build-up, default
            1), start (first forecast quarter hit, default 1)}.

    Returns:
        DataFrame.
    """
    out = mdf.copy()
    end = ctx.idx[ctx.end]
    pos = np.flatnonzero(out.index > end)
    for s in shocks:
        code = resolve_variable(ctx, s["variable"])
        if code is None:
            raise ValueError(f"what-if: unknown variable '{s['variable']}'")
        size, q, start = float(s["size"]), max(int(s.get("quarters", 1) or 1), 1), max(int(s.get("start", 1) or 1), 1)
        k = np.arange(1, len(pos) + 1)
        ramp = np.clip((k - start + 1) / q, 0, 1)
        targets = {code: 1.0}
        if with_lookalikes:
            targets.update(lookalikes(cfg, ctx, code))
        for c, slope in targets.items():
            col = out.columns.get_loc(c)
            vals = out.iloc[pos, col].astype(float).values
            if is_level(ctx, c):
                out.iloc[pos, col] = vals * np.exp(slope * size * ramp / 100)
            else:
                out.iloc[pos, col] = vals + slope * size * ramp
    return out


def forecast(ctx, model, mdf, H):
    """Quarterly changes of the target (model form) over H quarters from the end of history."""
    P = model.panel(ctx, mdf)
    return np.asarray(model.forecast(P, ctx.idx[ctx.end], H), dtype=float)[:H]


def effect(ctx, base_path, shocked_path):
    """Growth effect (pp per quarter or target units) and balance effect (% or target units) by quarter."""
    L0 = float(ctx.level.reindex([ctx.idx[ctx.end]]).iloc[0]) if ctx.idx[ctx.end] in ctx.level.index \
        else float(ctx.level.iloc[-1])
    b, s = tg.path(L0, base_path), tg.path(L0, shocked_path)
    if tg.is_log():
        return (shocked_path - base_path) * 100, (s / b - 1) * 100
    return shocked_path - base_path, s - b


# ------------------------------------------------------------------------------------------------ the models


def has_drivers(model):
    """False for models without macro drivers (benchmarks, own-history models)."""
    try:
        return len([d for d in model.drivers() if d and d != "(none)"]) > 0
    except Exception:
        return bool(getattr(model, "features", None))


def tied_models(cfg, tests, models, best_rank=None):
    """Models not significantly worse than the best at impact.compare_horizon (DM-HLN, one-sided).

    Args:
        tests: backtest forecasts (comparison/12_backtest_forecasts). models: {name: fitted model}.

    Returns:
        DataFrame: model, family, mse, ratio_to_best, dm_stat, p_worse, tied, in_set.
    """
    from ..validation.oos import dm_test
    imp = cfg.get("impact") or {}
    H = int(cfg["sample"]["horizon"])
    h = min(int(imp.get("compare_horizon", 8)), H)
    alpha, cap = float(imp.get("tied_alpha", 0.10)), int(imp.get("tied_max", 8))
    T = tests[(tests["horizon_q"] == h) & (~tests["covid_window"].astype(bool))]
    T = T[T["model"].isin([k for k, m in models.items() if has_drivers(m)])]
    if T.empty:
        return pd.DataFrame(columns=["model", "family", "mse", "ratio_to_best", "dm_stat", "p_worse", "tied", "in_set"])
    E = T.pivot_table(index="origin", columns="model", values="error_pct")
    mse = (E ** 2).mean().sort_values()
    best = mse.index[0]
    rows = []
    for m in mse.index:
        ok = E[m].notna() & E[best].notna()
        if m == best:
            st, p = 0.0, 1.0
        else:
            st, p2, _ = dm_test(E.loc[ok, m].values, E.loc[ok, best].values, h)
            p = (p2 / 2 if st > 0 else 1 - p2 / 2) if np.isfinite(st) else np.nan
        tied = m == best or not (np.isfinite(p) and p < alpha)
        rows.append({"model": m, "family": T.loc[T["model"] == m, "family"].iloc[0], "mse": mse[m],
                     "ratio_to_best": mse[m] / mse[best] if mse[best] > 0 else np.nan, "dm_stat": st,
                     "p_worse": p, "tied": tied})
    out = pd.DataFrame(rows)
    ratio = float(imp.get("tied_max_ratio", 2.0))
    out["close"] = out["ratio_to_best"] <= ratio
    out["in_set"] = False
    out["why"] = ""
    keep = out[out["tied"] & out["close"]].head(cap).index
    out.loc[keep, ["in_set", "why"]] = [True, f"not significantly worse and within {ratio:g}x the best error"]
    need = int(imp.get("tied_min", 3)) - int(out["in_set"].sum())
    if need > 0:                                     # too few: the next most accurate, so there is a range
        add = out[~out["in_set"]].head(need).index
        out.loc[add, ["in_set", "why"]] = [True, "added: among the most accurate (too few tied models)"]
    out.attrs["horizon"] = h
    return out


def explanatory(models):
    """The explanatory model (focus drivers), or None."""
    return next((m for m in models.values() if getattr(m, "family", "") == "Explanatory"), None)


def _draws(model, n, seed):
    """Coefficient draws from the model's covariance (regressions with HAC results); [] if not available."""
    res = getattr(model, "res", None)
    if res is None or getattr(model, "params", None) is None:
        return []
    try:
        V = np.asarray(res.cov_params(), float)
        b = np.asarray(model.params, float)
        rng = np.random.default_rng(seed)
        return [pd.Series(d, index=model.params.index) for d in rng.multivariate_normal(b, V, size=n, method="eigh")]
    except Exception:
        return []


def responses(cfg, ctx, models, shocks, names, label, H, draws=0):
    """Response of each named model to a set of shocks (and the explanatory model's band from coefficient draws).

    Returns:
        DataFrame: shock, model, kind (point/draw_lo/draw_hi), quarter, growth_effect, balance_effect.
    """
    shocked = apply_shocks(cfg, ctx, ctx.base, shocks)
    rows = []
    for nm in names:
        m = models.get(nm)
        if m is None:
            continue
        try:
            b, s = forecast(ctx, m, ctx.base, H), forecast(ctx, m, shocked, H)
        except Exception:
            continue
        g, lv = effect(ctx, b, s)
        rows += [{"shock": label, "model": nm, "kind": "point", "quarter": q + 1, "growth_effect": g[q],
                  "balance_effect": lv[q]} for q in range(len(g))]
        if draws and getattr(m, "family", "") == "Explanatory":
            D = _draws(m, draws, int((cfg.get("impact") or {}).get("seed", 42)))
            keep = m.params.copy()
            G, L = [], []
            Pb, Ps = m.panel(ctx, ctx.base), m.panel(ctx, shocked)      # built once; only the coefficients change
            o = ctx.idx[ctx.end]
            for d in D:
                m.params = d
                try:
                    bb = np.asarray(m.forecast(Pb, o, H), float)[:H]
                    ss = np.asarray(m.forecast(Ps, o, H), float)[:H]
                    g_, l_ = effect(ctx, bb, ss)
                    G.append(g_)
                    L.append(l_)
                except Exception:
                    pass
            m.params = keep
            if L:
                G, L = np.array(G), np.array(L)
                for kind, qq in (("band_lo", 5), ("band_hi", 95)):
                    gq, lq = np.percentile(G, qq, axis=0), np.percentile(L, qq, axis=0)
                    rows += [{"shock": label, "model": nm, "kind": kind, "quarter": q + 1, "growth_effect": gq[q],
                              "balance_effect": lq[q]} for q in range(len(gq))]
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ attribution


def used_codes(model, cols):
    """Macro series a model uses (found in its features, long-run terms and members)."""
    texts = []

    def collect(m, depth=0):
        if depth > 3 or m is None:
            return
        for a in ("features", "lr_names", "bases", "short_run", "items"):
            v = getattr(m, a, None)
            if callable(v):
                continue
            if isinstance(v, (list, tuple)):
                texts.extend(str(x) for x in v)
            elif isinstance(v, dict):
                texts.extend(str(x) for x in v)
        try:
            texts.extend(str(x) for x in m.drivers())
        except Exception:
            pass
        for a in ("members", "models"):
            v = getattr(m, a, None)
            if isinstance(v, (list, tuple)):
                for x in v:
                    collect(x[0] if isinstance(x, tuple) else x, depth + 1)
        lr = getattr(m, "long_run", None)
        for x in (lr if isinstance(lr, (list, tuple)) else []):
            if isinstance(x, dict) and "mnemonic" in x:
                texts.append(str(x["mnemonic"]))
    collect(model)
    blob = " ".join(texts)
    return [c for c in cols if re.search(r"(?<![A-Za-z0-9.])" + re.escape(c) + r"(?![A-Za-z0-9])", blob)]


def attribution(cfg, ctx, models, names, focus_groups, H, hz):
    """Each scenario minus the base scenario, split by concept (focus concepts) or theme, by Shapley value.

    Args:
        focus_groups: {series: concept label} for the focus series and their look-alikes.

    Returns:
        DataFrame: scenario, model, group, horizon, contribution, total.
    """
    base_name = cfg["data"]["macro"]["base_scenario"]
    rows = []
    end = ctx.idx[ctx.end]
    for sname, S in ctx.macro.items():
        if sname == base_name:
            continue
        for nm in names:
            m = models.get(nm)
            if m is None:
                continue
            codes = [c for c in used_codes(m, list(ctx.base.columns)) if c in S.columns]
            if cfg["features"].get("deflator") in S.columns and cfg["features"]["deflator"] not in codes:
                codes.append(cfg["features"]["deflator"])
            fut = S.index > end
            codes = [c for c in codes if not np.allclose(S.loc[fut, c].astype(float).values,
                                                         ctx.base.loc[fut, c].astype(float).values, equal_nan=True)]
            if not codes:
                continue
            groups = {}
            for c in codes:
                g = focus_groups.get(c) or (ctx.meta.loc[c, "theme"] if c in ctx.meta.index else c)
                groups.setdefault(g, []).append(c)
            names_g = list(groups)
            if len(names_g) > 6:                       # keep the 5 largest, the rest as one group
                single = {}
                for g in names_g:
                    M = ctx.base.copy()
                    M.loc[fut, groups[g]] = S.loc[fut, groups[g]].values
                    try:
                        single[g] = abs(effect(ctx, forecast(ctx, m, ctx.base, H), forecast(ctx, m, M, H))[1][-1])
                    except Exception:
                        single[g] = 0
                top = sorted(single, key=single.get, reverse=True)[:5]
                rest = [c for g in names_g if g not in top for c in groups[g]]
                groups = {g: groups[g] for g in top} | ({"Other drivers": rest} if rest else {})
                names_g = list(groups)
            try:
                base_fc = forecast(ctx, m, ctx.base, H)
            except Exception:
                continue
            cache = {}

            def value(sub):
                key = frozenset(sub)
                if key not in cache:
                    M = ctx.base.copy()
                    cols = [c for g in sub for c in groups[g]]
                    if cols:
                        M.loc[fut, cols] = S.loc[fut, cols].values
                    cache[key] = effect(ctx, base_fc, forecast(ctx, m, M, H))[1] if cols else np.zeros(H)
                return cache[key]
            try:
                total = value(names_g)
            except Exception:
                continue
            k = len(names_g)
            from math import factorial
            for g in names_g:
                others = [x for x in names_g if x != g]
                phi = np.zeros(H)
                for r in range(k):
                    w = factorial(r) * factorial(k - r - 1) / factorial(k)
                    for sub in itertools.combinations(others, r):
                        phi += w * (value(list(sub) + [g]) - value(list(sub)))
                for h in hz:
                    rows.append({"scenario": sname, "model": nm, "group": g, "horizon": h,
                                 "contribution": phi[h - 1], "total": total[h - 1],
                                 "series": ", ".join(groups[g])})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ reliability


def reliability(cfg, ctx, model, focus, run_dir, resp_typ, set_names, hz_main):
    """How far each focus effect can be trusted: significance, sign, stability over time, collinearity,
    inclusion probability and agreement between the tied models.

    Returns:
        DataFrame: one row per focus driver with the checks, a label and the reasons.
    """
    rows = []
    P = model.panel(ctx, ctx.base).iloc[ctx.est0:ctx.end + 1]
    feats = [f for f in model.features if f in P.columns]
    from ..modelling.ardl import vif
    Fm = P[feats].dropna()
    V = dict(zip(feats, vif(Fm.values))) if len(feats) > 1 and len(Fm) > len(feats) + 2 else {}
    cp_path = os.path.join(run_dir, "comparison", "7_coefficient_paths.csv")
    cp = pd.read_csv(cp_path) if os.path.exists(cp_path) else None
    pip_path = os.path.join(run_dir, "bayes_first", "C2_bma_inclusion.csv")
    pip = pd.read_csv(pip_path) if os.path.exists(pip_path) else None
    tv = getattr(getattr(model, "res", None), "tvalues", None)
    no_ctrl = {}
    try:                                     # the focus effects without the controls (robustness)
        foc = [f for f in focus["feature"] if f in P.columns]
        X = pd.DataFrame({"const": 1.0, "y_lag1": P["y"].shift(1)}, index=P.index)
        for c in [c for c in P.columns if c.startswith("D_")] + foc:
            X[c] = P[c]
        X = X.loc[:, (X.std() > 0) | (X.columns == "const")]
        ok = X.notna().all(axis=1) & P["y"].notna()
        bb = np.linalg.lstsq(X[ok].values, P.loc[ok, "y"].values, rcond=None)[0]
        no_ctrl = dict(zip(X.columns, bb))
    except Exception:
        pass
    for _, r in focus.iterrows():
        f, code = r["feature"], r["series"]
        es = int(r["expected_sign"])
        b = float(model.params.get(f, np.nan))
        t = float(tv.get(f, np.nan)) if tv is not None else np.nan
        stab = np.nan
        if cp is not None and f in cp.columns:
            s = cp.loc[cp["model"] == model.name, f].dropna()
            if len(s) and np.isfinite(b) and b != 0:
                stab = float((np.sign(s) == np.sign(b)).mean())
        p_inc = np.nan
        if pip is not None and "feature" in pip.columns:
            m_ = pip[pip["feature"].astype(str).str.split("|").str[0] == code]
            if len(m_):
                p_inc = float(m_["inclusion_probability"].max())
        R = resp_typ[(resp_typ["shock"] == r["concept"]) & (resp_typ["kind"] == "point")
                     & (resp_typ["quarter"] == hz_main) & (resp_typ["model"].isin(set_names))]
        vals = R["balance_effect"].values
        nz = vals[np.abs(vals) > 1e-9]
        agree = float((np.sign(nz) == np.sign(np.median(nz))).mean()) if len(nz) else np.nan
        reasons = []
        if es and np.isfinite(b) and np.sign(b) != es:
            reasons.append("sign opposite to the expected one")
        if not (np.isfinite(t) and abs(t) >= 1.96):
            reasons.append(f"not significant (t = {t:.1f})" if np.isfinite(t) else "no standard error")
        if np.isfinite(stab) and stab < 0.8:
            reasons.append(f"sign held in only {stab:.0%} of backtest re-estimations")
        if V.get(f, 1) > 5:
            reasons.append(f"collinear with other drivers (VIF {V[f]:.1f})")
        b0 = no_ctrl.get(f, np.nan)
        if np.isfinite(b0) and np.isfinite(b) and len(feats) > len(focus):
            if np.sign(b0) != np.sign(b) or abs(b - b0) > 0.5 * max(abs(b0), abs(b)):
                reasons.append(f"depends on the controls (without them: {b0:.3g})")
        if np.isfinite(agree) and agree < 0.75:
            reasons.append(f"tied models disagree on the direction ({agree:.0%} agree)")
        if len(vals) and len(nz) < len(vals) / 2:
            reasons.append(f"{len(vals) - len(nz)} of {len(vals)} tied models do not use it")
        label = "Reliable" if not reasons else "Use with care" if len(reasons) == 1 else "Weak"
        rows.append({"concept": r["concept"], "feature": f, "description": describe(ctx, code), "coefficient": b, "without_controls": b0,
                     "t": t, "expected_sign": es, "sign_ok": (np.sign(b) == es) if es and np.isfinite(b) else np.nan,
                     "sign_stability": stab, "vif": V.get(f, np.nan), "inclusion_probability": p_inc,
                     "tied_models_agree": agree, "label": label, "reasons": "; ".join(reasons) or "all checks passed"})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ episodes


def episode_macro(cfg, ctx, start, end):
    """The base scenario with each series' changes replaced by its changes during a historical episode
    (start..end), for as many quarters as the episode lasts; afterwards the base scenario's own changes resume."""
    s0, s1 = pd.Period(str(start), freq="Q"), pd.Period(str(end), freq="Q")
    H0 = ctx.base.index
    if s0 not in H0 or s1 not in H0 or s1 <= s0:
        raise ValueError(f"episode {start}-{end} is not inside the macro history")
    endq = ctx.idx[ctx.end]
    out = ctx.base.copy()
    n = int((s1 - s0).n)
    fut = list(out.index[out.index > endq])
    for c in out.columns:
        x = ctx.base[c].astype(float)
        if x.loc[s0:s1].isna().any() or pd.isna(x.get(endq)):
            continue
        lvl = is_level(ctx, c)
        cur = x[endq]
        for k, q in enumerate(fut):
            if k < n:
                a, b = x[s0 + k], x[s0 + k + 1]
                cur = cur * (b / a) if lvl else cur + (b - a)
            else:
                prev = x[q - 1]
                cur = cur * (x[q] / prev) if lvl else cur + (x[q] - prev)
            out.loc[q, c] = cur
    return out


# ------------------------------------------------------------------------------------------------ main


def focus_table(run_dir, ctx, model):
    """The focus drivers (focus_first/D1_focus_drivers.csv, or the model's own list)."""
    p = os.path.join(run_dir, "focus_first", "D1_focus_drivers.csv")
    if os.path.exists(p):
        return pd.read_csv(p)
    from ..features.transforms import feature_sign
    rows = [{"concept": f.split("|")[0], "series": f.split("|")[0], "feature": f,
             "expected_sign": feature_sign(ctx.meta, f)} for f in getattr(model, "focus", [])]
    return pd.DataFrame(rows)


def run(cfg, ctx, models, run_dir, log=print, what_if=None, episodes=None):
    """Everything in this module for a finished run; writes outputs/<run>/impact/.

    Args:
        models: {name: fitted model} (estimated on all history). run_dir: the run's output folder.
        what_if, episodes: override impact.what_if / impact.episodes.

    Returns:
        dict of tables.
    """
    imp = cfg.get("impact") or {}
    out = os.path.join(run_dir, "impact")
    os.makedirs(out, exist_ok=True)
    H = int(cfg["sample"]["horizon"])
    hz = sorted({h for h in (imp.get("horizons") or [1, 4, 8, 12]) if h <= H} | {H})
    hz_main = min(int(imp.get("compare_horizon", 8)), H)
    tests = pd.read_csv(os.path.join(run_dir, "comparison", "12_backtest_forecasts.csv"))
    ex = explanatory(models)
    if ex is None:
        log("  impact: no explanatory model in this run (impact.enabled: false?) - skipped")
        return {}
    tied = tied_models(cfg, tests, models)
    set_names = list(tied.loc[tied["in_set"], "model"])
    if ex.name not in set_names:
        set_names.append(ex.name)
    log(f"  impact: {len(set_names) - 1} model(s) tied with the best at {hz_main} quarters ahead + the explanatory model")
    focus = focus_table(run_dir, ctx, ex)
    focus_groups = {}
    shocks_unit, shocks_typ = [], []
    for _, r in focus.iterrows():
        code = r["series"]
        if code not in ctx.base.columns:
            continue
        focus_groups[code] = r["concept"]
        for c in lookalikes(cfg, ctx, code):
            focus_groups.setdefault(c, r["concept"])
        shocks_unit.append((r["concept"], code, 1.0))
        shocks_typ.append((r["concept"], code, typical_size(ctx, code)))
    draws = int(imp.get("draws", 300))
    R = []
    for (concept, code, size), kind in [(s, "unit") for s in shocks_unit] + [(s, "typical") for s in shocks_typ]:
        r_ = responses(cfg, ctx, models, [{"variable": code, "size": size, "quarters": 1}], set_names, concept, H,
                       draws if kind == "typical" else 0)
        R.append(r_.assign(shock_type=kind, series=code, size=size, shock_label=unit_label(ctx, code, size)))
    resp = pd.concat(R, ignore_index=True) if R else pd.DataFrame()
    # sensitivity summary
    sens = []
    for (concept, code, size) in shocks_typ:
        unit = next(s for c, cd, s in shocks_unit if c == concept)
        for h in hz:
            row = {"concept": concept, "series": code, "description": describe(ctx, code), "horizon": h,
                   "typical_shock": unit_label(ctx, code, size)}
            for kind, sz in (("typical", size), ("unit", unit)):
                d = resp[(resp["shock"] == concept) & (resp["shock_type"] == kind) & (resp["quarter"] == h)]
                pt = d[d["kind"] == "point"]
                e = pt.loc[pt["model"] == ex.name, "balance_effect"]
                allo = pt.loc[pt["model"] != ex.name, "balance_effect"]
                others = allo[allo.abs() > 1e-9]
                row[f"{kind}_explanatory"] = float(e.iloc[0]) if len(e) else np.nan
                if kind == "typical":
                    lo = d.loc[(d["model"] == ex.name) & (d["kind"] == "band_lo"), "balance_effect"]
                    hi = d.loc[(d["model"] == ex.name) & (d["kind"] == "band_hi"), "balance_effect"]
                    row["typical_band_lo"] = float(lo.iloc[0]) if len(lo) else np.nan
                    row["typical_band_hi"] = float(hi.iloc[0]) if len(hi) else np.nan
                    row["typical_tied_median"] = float(others.median()) if len(others) else np.nan
                    row["typical_tied_min"] = float(others.min()) if len(others) else np.nan
                    row["typical_tied_max"] = float(others.max()) if len(others) else np.nan
                    row["tied_models"] = int(len(allo))
                    row["tied_models_using_it"] = int(len(others))
            sens.append(row)
    sens = pd.DataFrame(sens)
    typ = resp[resp["shock_type"] == "typical"] if len(resp) else resp
    rel = reliability(cfg, ctx, ex, focus, run_dir, typ, [n for n in set_names if n != ex.name], hz_main)
    best = tied["model"].iloc[0] if len(tied) else None
    att_names = [ex.name] + ([best] if best and best != ex.name else [])
    if str(imp.get("attribution_models", "explanatory+best")) == "tied":
        att_names = set_names
    att = attribution(cfg, ctx, models, att_names, focus_groups, H, hz) if imp.get("attribution", True) \
        else pd.DataFrame()
    # what-if and episodes
    wi = []
    for w in (what_if if what_if is not None else imp.get("what_if") or []):
        try:
            r_ = responses(cfg, ctx, models, w["shocks"], set_names, w.get("name", "what-if"), H, draws)
            wi.append(r_.assign(kind_of="what_if"))
        except Exception as e_:
            log(f"  impact: what-if '{w.get('name')}' skipped ({e_})")
    for ep in (episodes if episodes is not None else imp.get("episodes") or []):
        try:
            M = episode_macro(cfg, ctx, ep["start"], ep["end"])
            name = ep.get("name") or f"{ep['start']}-{ep['end']}"
            for nm in set_names:
                m = models.get(nm)
                try:
                    g, lv = effect(ctx, forecast(ctx, m, ctx.base, H), forecast(ctx, m, M, H))
                except Exception:
                    continue
                wi.append(pd.DataFrame({"shock": name, "model": nm, "kind": "point", "quarter": np.arange(1, H + 1),
                                        "growth_effect": g, "balance_effect": lv, "kind_of": "episode"}))
        except Exception as e_:
            log(f"  impact: episode {ep} skipped ({e_})")
    wi = pd.concat(wi, ignore_index=True) if wi else pd.DataFrame()
    allr = all_model_responses(cfg, ctx, models, shocks_typ, hz, resp, set_names, H)
    tables = {"1_tied_models": tied, "2_responses": resp, "3_sensitivity": sens, "4_reliability": rel,
              "5_attribution": att, "6_what_if": wi, "7_all_model_responses": allr}
    for k, v in tables.items():
        f = os.path.join(out, f"{k}.csv")
        if isinstance(v, pd.DataFrame) and len(v):
            v.round(6).to_csv(f, index=False)
        elif os.path.exists(f):
            os.remove(f)    # an empty table this run: drop the old file so the app does not show a stale one
    try:
        from . import charts
        charts.all(out, cfg, ctx, tables, ex.name, hz_main)
    except Exception as e_:
        log(f"  impact: charts not drawn ({type(e_).__name__}: {e_})")
    write_summary(os.path.join(out, "summary.md"), cfg, ctx, tables, ex.name, hz_main, set_names)
    for _, r in rel.iterrows():
        s = sens[(sens["concept"] == r["concept"]) & (sens["horizon"] == hz_main)]
        if len(s):
            s = s.iloc[0]
            log(f"  {r['concept']}: {s['typical_shock']} -> {s['typical_explanatory']:+.2f}{_u()} after {hz_main}q "
                f"(90% band {s['typical_band_lo']:+.2f} to {s['typical_band_hi']:+.2f}); {r['label']}")
    log(f"  impact: tables and charts in {out}")
    return tables


def all_model_responses(cfg, ctx, models, shocks_typ, hz, resp, set_names, H):
    """Response of every model with drivers to a typical rise in each focus driver (for the model scorecard).

    Returns:
        DataFrame: model, shock, series, expected_sign, horizon, balance_effect.
    """
    from ..features.transforms import feature_sign
    rows = []
    sign_of = {}
    for concept, code, size in shocks_typ:
        sign_of[concept] = feature_sign(ctx.meta, f"{code}|lvl|L0") or (
            int(np.sign(ctx.meta.loc[code, "expected_sign"])) if code in ctx.meta.index else 0)
    have = resp[(resp.get("shock_type") == "typical") & (resp["kind"] == "point")] if len(resp) else resp
    shocked = {concept: apply_shocks(cfg, ctx, ctx.base, [{"variable": code, "size": size}])
               for concept, code, size in shocks_typ}
    for nm, m in models.items():
        if not has_drivers(m):
            continue
        if nm in set_names and len(have):
            d = have[(have["model"] == nm) & (have["quarter"].isin(hz))]
            rows += [{"model": nm, "shock": r["shock"], "series": r["series"], "expected_sign": sign_of.get(r["shock"], 0),
                      "horizon": int(r["quarter"]), "balance_effect": r["balance_effect"]} for _, r in d.iterrows()]
            continue
        try:
            b = forecast(ctx, m, ctx.base, H)
        except Exception:
            continue
        for concept, code, size in shocks_typ:
            try:
                _, lv = effect(ctx, b, forecast(ctx, m, shocked[concept], H))
            except Exception:
                continue
            rows += [{"model": nm, "shock": concept, "series": code, "expected_sign": sign_of.get(concept, 0),
                      "horizon": h, "balance_effect": lv[h - 1]} for h in hz]
    return pd.DataFrame(rows)


def _u():
    return "%" if tg.is_log() else ""


def sentence(ctx, row, h, unit_name="the balance"):
    """'A typical rise in X (+0.85 pt) changes the balance by -0.31% after 8 quarters (90% band ...)'."""
    u = _u()
    s = (f"A typical rise in {row['description']} ({row['typical_shock']}) changes {unit_name} by "
         f"{row['typical_explanatory']:+.2f}{u} after {h} quarters")
    if np.isfinite(row.get("typical_band_lo", np.nan)):
        s += f" (90% band {row['typical_band_lo']:+.2f}{u} to {row['typical_band_hi']:+.2f}{u})"
    n_all, n_use = int(row.get("tied_models", 0) or 0), int(row.get("tied_models_using_it", 0) or 0)
    if n_use:
        s += (f"; {n_use} of the {n_all} most accurate models use it and give {row['typical_tied_min']:+.2f}{u} to "
              f"{row['typical_tied_max']:+.2f}{u}")
    elif n_all:
        s += f"; none of the {n_all} most accurate models uses it"
    return s + "."


def write_summary(path, cfg, ctx, T, ex_name, h, set_names):
    """Plain-language summary of the impact analysis."""
    sens, rel, att, wi = T["3_sensitivity"], T["4_reliability"], T["5_attribution"], T["6_what_if"]
    target = cfg["data"]["bank"]["target_column"]
    L = [f"# How the economy affects {target}", "",
         f"Main estimate: the explanatory model ({ex_name}): the focus drivers always in, controls chosen by double "
         "selection. Range: the models statistically tied with the most accurate one "
         f"({len(set_names) - 1} models, Diebold-Mariano at {h} quarters ahead).", "",
         "Effects hold the other drivers at their baseline paths; a shock to one series also moves its look-alikes "
         "(same theme, correlated). They are associations estimated from history, not guarantees.", "",
         "## Sensitivity", ""]
    for _, r in sens[sens["horizon"] == h].iterrows():
        lab = rel.loc[rel["concept"] == r["concept"]]
        L.append(f"- {sentence(ctx, r, h)} **{lab['label'].iloc[0] if len(lab) else ''}**"
                 + (f": {lab['reasons'].iloc[0]}" if len(lab) and lab['label'].iloc[0] != 'Reliable' else ""))
    L += ["", "| driver | shock | " + " | ".join(f"{q}q" for q in sorted(sens["horizon"].unique())) + " |",
          "|---|---|" + "---|" * sens["horizon"].nunique()]
    for c, g in sens.groupby("concept", sort=False):
        L.append(f"| {g['description'].iloc[0]} | {g['typical_shock'].iloc[0]} | "
                 + " | ".join(f"{v:+.2f}" for v in g.sort_values("horizon")["typical_explanatory"]) + " |")
    if len(att):
        L += ["", "## Scenarios against the base scenario, by driver", ""]
        a = att[(att["model"] == ex_name) & (att["horizon"] == h)]
        for s, g in a.groupby("scenario", sort=False):
            parts = ", ".join(f"{r['group']} {r['contribution']:+.2f}" for _, r in
                              g.reindex(g["contribution"].abs().sort_values(ascending=False).index).iterrows())
            L.append(f"- {s}: {g['total'].iloc[0]:+.2f}{_u()} after {h} quarters ({parts})")
    if len(wi):
        L += ["", "## What-if", ""]
        for s, g in wi[(wi["kind"] == "point") & (wi["quarter"] == h)].groupby("shock", sort=False):
            e = g.loc[g["model"] == ex_name, "balance_effect"]
            o = g.loc[(g["model"] != ex_name) & (g["balance_effect"].abs() > 1e-9), "balance_effect"]
            rng = (f"; {len(o)} of the other models respond, {o.min():+.2f} to {o.max():+.2f}" if len(o)
                   else "; no other model uses these variables")
            L.append(f"- {s}: {float(e.iloc[0]) if len(e) else np.nan:+.2f}{_u()} after {h} quarters{rng}")
    L += ["", "Files: 1_tied_models, 2_responses (every model, every quarter), 3_sensitivity, 4_reliability, "
          "5_attribution, 6_what_if; charts sensitivity.png, responses.png, attribution.png."]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
