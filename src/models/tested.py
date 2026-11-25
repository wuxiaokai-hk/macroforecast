"""Tested twins: every model whose drivers are not already chosen by the shared search gets a twin whose drivers
pass the same tests as the data-driven approaches (screen first, Elastic Net first, Bayesian first).

General to specific, starting from the model's own drivers: they are tested with the rules of final_model - each
driver |t| >= t_min in  y_t = a + rho*y_{t-1} + b'x + dummies,  the expected sign, VIF <= vif_max, one driver per
theme (if one_per_theme). The weakest failing driver is first tried at its other lags (0..max_lag, same form); if
no lag makes it pass it is dropped; repeat until every driver passes. So the twin keeps every driver that passes
and only re-lags or drops the ones that fail. Twins identical to a model already in the run are not added. The twin keeps the original model's form (Auto-ARDL, regime
switching, machine learning, ARIMAX, ...) and is named "<model> (tested)". The original model stays.

Not twinned: models without drivers (benchmarks, own-history models), the levels ARDL (its bounds test is its own
test), combinations, same-growth rules, foundation models, and models the search already produced. A model whose
own drivers already pass gets no twin (it would be identical). An error-correction model is twinned on its
short-run drivers; its long-run relation stays as specified.

Settings: models.tested.enabled (true). The rules are final_model.t_min, vif_max and one_per_theme.
"""
import copy

import numpy as np
import pandas as pd

from ..features.transforms import feature_matrix, feature_sign, parse_feature

SKIP_CLASSES = {"Combination", "SameGrowth", "LevelsARDL", "FoundationModel", "Theta"}
SEARCHED = {"Screen first", "Elastic Net first", "Bayesian first"}


def _drivers(m):
    """The driver features the twin will re-choose, or [] if the model is not twinned."""
    cls = type(m).__name__
    if cls in SKIP_CLASSES or m.name in SEARCHED or m.name.endswith("(tested)"):
        return []
    if cls == "AutoARDL":
        return [f"{b}|L0" for b in getattr(m, "bases", [])]
    feats = list(getattr(m, "features", None) or [])
    return [f for f in feats if str(f).count("|") == 2]


def _rebuild(m, feats):
    """A copy of model m with the drivers feats (same form, settings and family)."""
    cls = type(m).__name__
    name = f"{m.name} (tested)"
    if cls == "AutoARDL":
        from .autoardl import AutoARDL
        bases = list(dict.fromkeys(f.rsplit("|", 1)[0] for f in feats))
        signs = {b: s for b, s in (getattr(m, "base_signs", {}) or {}).items() if b in bases}
        return AutoARDL(name, [f"{b}|L0" for b in bases], m.max_p, m.max_q, m.ic, family=m.family, note=m.note,
                        signs=signs)
    t = copy.deepcopy(m)
    t.name = name
    t.features = list(feats)
    for a in ("params", "res", "beta"):              # nothing estimated carries over
        if hasattr(t, a):
            setattr(t, a, None)
    if hasattr(t, "focus"):
        t.focus = [f for f in t.focus if f in feats]
    return t


def _check(ctx, F, feats, signs, themes, fm):
    """OLS y ~ 1 + y_lag1 + dummies + feats on the estimation window; (ok, per-feature problems, t-values)."""
    from ..modelling.ardl import ols_np, vif
    last = ctx.end + 1
    rows = np.arange(ctx.est0, ctx.end + 1)
    y = ctx.y.values[:last]
    D = ctx.dummies.values[:last]
    Dr = D[rows][:, D[rows].std(axis=0) > 0] if D.shape[1] else np.zeros((len(rows), 0))
    Fk = F[feats].values[rows]
    X = np.column_stack([np.ones(len(rows)), y[rows - 1], Dr, Fk])
    b, se, _ = ols_np(X, y[rows])
    k = len(feats)
    t = b[-k:] / se[-k:]
    v = vif(Fk) if k > 1 else np.ones(1)
    bad = {}
    for i, f in enumerate(feats):
        if signs[f] and np.sign(b[-k + i]) != signs[f]:
            bad[f] = ("wrong sign", 0.0)
        elif abs(t[i]) < fm["t_min"]:
            bad[f] = (f"|t| {abs(t[i]):.2f} < {fm['t_min']}", abs(t[i]))
        elif v[i] > fm["vif_max"]:
            bad[f] = (f"VIF {v[i]:.1f} > {fm['vif_max']}", abs(t[i]))
    if fm.get("one_per_theme"):
        by = {}
        for i, f in enumerate(feats):
            by.setdefault(themes[f], []).append((abs(t[i]), f))
        for th, lst in by.items():
            if th and len(lst) > 1:
                for tt, f in sorted(lst, reverse=True)[1:]:
                    bad.setdefault(f, ("second driver of the same theme", tt))
    return not bad, bad, dict(zip(feats, t))


def twin(cfg, ctx, m, log=print):
    """(twin model, how it was specified) or (None, reason): general-to-specific on the model's own drivers."""
    drv = _drivers(m)
    if not drv:
        return None, "not twinned"
    fm = cfg["final_model"]
    max_lag = int(cfg["features"]["max_lag"])
    pool = []
    for f in drv:
        code, tr, _ = parse_feature(f)
        pool += [f"{code}|{tr}|L{k}" for k in range(max_lag + 1)]
    pool = list(dict.fromkeys(pool))
    F = feature_matrix(ctx.hist, pool, cfg["features"]["deflator"]).reindex(ctx.idx)
    win = F.iloc[ctx.est0:ctx.end + 1]
    F = F.loc[:, win.notna().all() & (win.std() > 0)]
    signs = {f: feature_sign(ctx.meta, f) for f in F.columns}
    themes = {f: (ctx.meta.loc[parse_feature(f)[0], "theme"] if parse_feature(f)[0] in ctx.meta.index else "")
              for f in F.columns}
    cur = [f for f in dict.fromkeys(drv) if f in F.columns]
    if type(m).__name__ == "AutoARDL":                # its own lags are chosen in the fit: start from lag 1
        cur = [f.rsplit("|", 1)[0] + "|L1" if f.rsplit("|", 1)[0] + "|L1" in F.columns else f for f in cur]
    if not cur:
        return None, "no usable driver"
    steps = []
    for _ in range(4 * len(pool)):
        ok, bad, t = _check(ctx, F, cur, signs, themes, fm)
        if ok:
            break
        f, (why, _) = min(bad.items(), key=lambda kv: kv[1][1])   # the weakest failing driver first
        code, tr, lag = parse_feature(f)
        others = [g for g in cur if g != f]
        best = None
        for k in range(max_lag + 1):                  # another lag of the same variable?
            g = f"{code}|{tr}|L{k}"
            if k == lag or g not in F.columns:
                continue
            ok2, bad2, t2 = _check(ctx, F, others + [g], signs, themes, fm)
            if g not in bad2 and (best is None or abs(t2[g]) > best[1]):
                best = (g, abs(t2[g]))
        if best:
            cur = others + [best[0]]
            steps.append(f"{f}: {why} -> lag {parse_feature(best[0])[2]}")
        else:
            cur = others
            steps.append(f"{f}: {why} -> dropped")
        if not cur:
            return None, "no driver passes the tests"
    else:
        return None, "the drivers could not be made to pass the tests"
    if set(cur) == set(drv) or (type(m).__name__ == "AutoARDL" and
                                {f.rsplit("|", 1)[0] for f in cur} == {f.rsplit("|", 1)[0] for f in drv}):
        return None, "its drivers already pass the tests"
    n_vars = len({parse_feature(f)[0] for f in drv})
    kept = len({parse_feature(f)[0] for f in cur})
    how = (f"Twin of '{m.name}': its drivers re-checked until every one passes the tests of the data-driven "
           f"approaches (|t| >= {fm['t_min']}, expected sign, VIF <= {fm['vif_max']}"
           + (", one per theme" if fm.get("one_per_theme") else "") + f"); {kept} of its {n_vars} variables kept. "
           + "; ".join(steps))
    return _rebuild(m, cur), how


def twins(cfg, ctx, models, log=print):
    """Tested twins of a list of models: [(twin, how)] plus a table of what happened to each model."""
    st = (cfg["models"].get("tested") or {})
    if not st.get("enabled", True):
        return [], pd.DataFrame()
    out, rows = [], []

    def key(x):
        sp = dict(x.spec()) if hasattr(x, "spec") else {}
        sp.pop("name", None)
        sp.pop("note", None)
        sp = {k: (sorted(v) if isinstance(v, list) and all(isinstance(i, str) for i in v) else v) for k, v in sp.items()}
        return type(x).__name__, repr(sorted(sp.items(), key=lambda kv: kv[0]))
    seen = {key(m): m.name for m in models}
    for m in models:
        try:
            t, how = twin(cfg, ctx, m, log)
        except Exception as ex:                       # one model's twin never stops the run
            t, how = None, f"failed ({type(ex).__name__}: {ex})"
        if t is not None:
            k = key(t)
            if k in seen:
                t, how = None, f"tested version identical to '{seen[k]}'"
            else:
                seen[k] = t.name
        rows.append({"model": m.name, "family": m.family, "twin": t.name if t is not None else "",
                     "outcome": "twin added" if t is not None else how,
                     "original_drivers": ", ".join(_drivers(m)),
                     "tested_drivers": ", ".join(t.drivers()) if t is not None else ""})
        if t is not None:
            out.append((t, how))
    T = pd.DataFrame(rows)
    if len(T):
        log(f"  tested twins: {int((T['outcome'] == 'twin added').sum())} added; "
            f"{int((T['outcome'] == 'its drivers already pass the tests').sum())} already pass; "
            f"{int(T['outcome'].str.startswith('no combination').sum())} with no passing combination")
    return out, T
