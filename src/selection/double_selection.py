"""Focus drivers and their controls, chosen for a reliable effect rather than the best forecast.

1. Focus drivers. The variables whose effect the user wants to know (config impact.focus: literature concept ids,
   series names, or exact features "series|transform|Llag"). A concept or series without a form and lag gets the
   stationary form and lag (0..max_lag) with the largest t-statistic of the expected sign in
       y_t = a + rho*y_{t-1} + b*x_t + dummies.
   impact.focus empty: the literature core concepts; without a literature table, the three strongest variables of
   the expected sign from different themes.

2. Controls by double selection (Belloni, Chernozhukov and Hansen 2014, "Inference on treatment effects after
   selection among high-dimensional controls", Review of Economic Studies 81(2)). A control is kept if it
   predicts the target OR any focus driver. Selecting only on the target (as a forecasting search does) can drop a
   variable that moves with a focus driver and biases its effect; selecting on both guards against that.
   Controls whose |t| in the target equation (with the focus drivers) is below impact.control_t (1.0) are then
   dropped one at a time: their omitted-variable bias (coefficient x link) is small, while they add noise.
   Each selection is a rigorous lasso (BCH plug-in penalty, lambda = 2c*sigma*sqrt(n)*Phi^-1(1 - gamma/(2p)),
   c = 1.1, gamma = 0.1/log(n), sigma re-estimated from post-lasso residuals), after partialling out the constant,
   last quarter's target growth and the dummies (Frisch-Waugh-Lovell).
   Candidates are excluded as controls when they are the focus series themselves, belong to the theme of a focus
   driver (another city's house prices, the 3-year next to the 5-year mortgage rate: alternative measures of the
   same concept, which would split its effect; impact.controls_same_theme: true allows them), or correlate above
   impact.control_corr with a focus driver (near-duplicates make the focus effect unidentifiable). Themes that are
   other outcomes of the same economy (credit aggregates, deposits) are not controls either: they carry part of the
   macro effect, so holding them fixed would hide it (impact.excluded_control_themes, default "auto").
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

from ..features.transforms import feature_sign, parse_feature


def _window(ctx, F):
    """Target, base regressors (constant, lagged growth, dummies) and candidates on the estimation window."""
    rows = slice(ctx.est0, ctx.end + 1)
    y = ctx.y.iloc[rows]
    W = pd.DataFrame({"const": 1.0, "y_lag1": ctx.y.shift(1).iloc[rows]}, index=y.index)
    for c in ctx.dummies.columns:
        d = ctx.dummies[c].iloc[rows]
        if d.std() > 0:
            W[c] = d
    Fw = F.iloc[rows].replace([np.inf, -np.inf], np.nan)
    ok = np.isfinite(y) & np.isfinite(W).all(axis=1)
    return y[ok], W[ok], Fw[ok]


def _partial(M, W):
    """Residuals of every column of M on W (Frisch-Waugh-Lovell)."""
    Wv = W.values
    B = np.linalg.pinv(Wv) @ np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
    return M - Wv @ B


def single_t(ctx, F, features):
    """t-statistic of each feature added alone to y ~ const + y_lag1 + dummies (classical errors, fast)."""
    y, W, Fw = _window(ctx, F[features])
    ry = _partial(y.values[:, None], W)[:, 0]
    out = {}
    for f in features:
        x = Fw[f].values
        if not np.isfinite(x).all() or np.std(x) == 0:
            continue
        rx = _partial(x[:, None], W)[:, 0]
        sxx = rx @ rx
        if sxx <= 1e-14:
            continue
        b = rx @ ry / sxx
        e = ry - b * rx
        dof = max(len(ry) - W.shape[1] - 1, 1)
        out[f] = b / np.sqrt(e @ e / dof / sxx)
    return pd.Series(out, dtype=float)


def rlasso(X, y, c=1.1, gamma=None, iters=5):
    """Rigorous (plug-in penalty) lasso of Belloni-Chernozhukov-Hansen on standardised X; y and X already partialled.

    Args:
        X: n x p array. y: n vector. c, gamma: penalty constants (gamma default 0.1/log(n)). iters: sigma updates.

    Returns:
        indices of the selected columns.
    """
    from sklearn.linear_model import Lasso
    n, p = X.shape
    if p == 0:
        return np.array([], dtype=int)
    sd = X.std(axis=0)
    keep = sd > 1e-12
    Z = np.zeros_like(X)
    Z[:, keep] = (X[:, keep] - X[:, keep].mean(axis=0)) / sd[keep]
    gamma = gamma or 0.1 / np.log(max(n, 3))
    q = norm.ppf(1 - gamma / (2 * p))
    sigma = np.std(y)
    sel = np.array([], dtype=int)
    for _ in range(iters):
        # sklearn minimises (1/2n)||y - Zb||^2 + alpha*||b||_1  <=>  BCH ||y - Zb||^2 + lambda*||b||_1 with
        # lambda = 2*c*sigma*sqrt(n)*q   ->   alpha = lambda / (2n) = c*sigma*q / sqrt(n)
        alpha = c * sigma * q / np.sqrt(n)
        m = Lasso(alpha=alpha, fit_intercept=True, max_iter=20000).fit(Z, y)
        new = np.flatnonzero(np.abs(m.coef_) > 1e-10)
        if len(new):                                     # post-lasso residuals for the next sigma
            b = np.linalg.lstsq(np.column_stack([np.ones(n), Z[:, new]]), y, rcond=None)[0]
            e = y - np.column_stack([np.ones(n), Z[:, new]]) @ b
            sigma_new = np.sqrt(e @ e / max(n - len(new) - 1, 1))
        else:
            sigma_new = np.std(y)
        if set(new) == set(sel) and abs(sigma_new - sigma) < 1e-6 * max(sigma, 1e-12):
            break
        sel, sigma = new, sigma_new
    return sel


def resolve_focus(cfg, ctx, F, forms, log=print):
    """The focus drivers as features, with how each was chosen.

    Returns:
        DataFrame: concept, series, feature, expected_sign, t, how.
    """
    from ..models import literature as lit
    imp = cfg.get("impact") or {}
    table = ctx.lit_table
    specs = list(imp.get("focus") or [])
    how_default = "named in impact.focus"
    if not specs and table is not None:
        lit_models = (cfg.get("literature") or {}).get("models") or []
        core = next((m for m in lit_models if m.get("core")), lit_models[-1] if lit_models else None)
        if core and core.get("features"):
            specs = list(core["features"])
            how_default = "literature core model (impact.focus is empty)"
        else:
            specs = [c for c in table.index if table.loc[c, "core_flag"] and table.loc[c, "active"]]
            how_default = "literature core concepts (impact.focus is empty)"
    cols = list(F.columns)
    rows = []
    for s in specs:
        s = str(s).strip()
        concept = s.split("|")[0]
        name = lit.resolve_feature(s, table) if table is not None else s
        if not name:
            log(f"  [focus] '{s}' is switched off or not mapped in the literature table: skipped")
            continue
        if "|" in name and name.count("|") == 2:
            if name in cols:
                rows.append({"concept": concept, "series": name.split("|")[0], "feature": name, "how": how_default})
            else:
                log(f"  [focus] '{name}' is not among the stationary candidate features: choosing its form and lag")
                name = name.split("|")[0]
        if "|" not in name:
            cands = [f for f in cols if f.split("|")[0] == name]
            if not cands:
                log(f"  [focus] no stationary candidate feature for '{s}' ({name}): skipped")
                continue
            t = single_t(ctx, F, cands)
            sg = pd.Series({f: feature_sign(ctx.meta, f) for f in t.index})
            score = t * sg.replace(0, 1)
            best = score.idxmax()
            rows.append({"concept": concept, "series": name, "feature": best,
                         "how": f"{how_default}; form and lag with the largest t of the expected sign "
                                f"({len(cands)} tried)"})
    if not rows:                                   # nothing named and no literature: strongest variables
        t = single_t(ctx, F, cols)
        sg = pd.Series({f: feature_sign(ctx.meta, f) for f in t.index})
        score = (t * sg).dropna()
        score = score[sg.reindex(score.index) != 0].sort_values(ascending=False)
        seen_var, seen_theme = set(), set()
        for f in score.index:
            code = f.split("|")[0]
            th = ctx.meta.loc[code, "theme"] if code in ctx.meta.index else ""
            if code in seen_var or th in seen_theme:
                continue
            seen_var.add(code)
            seen_theme.add(th)
            rows.append({"concept": code, "series": code, "feature": f,
                         "how": "no focus named and no literature: among the strongest variables of the expected "
                                "sign, one per theme"})
            if len(rows) == 3:
                break
    out = pd.DataFrame(rows).drop_duplicates("feature")
    tt = single_t(ctx, F, list(out["feature"]))
    out["expected_sign"] = [feature_sign(ctx.meta, f) for f in out["feature"]]
    out["t_alone"] = out["feature"].map(tt).round(2)
    out["theme"] = [ctx.meta.loc[s, "theme"] if s in ctx.meta.index else "" for s in out["series"]]
    return out.reset_index(drop=True)


OUTCOME_WORDS = ("credit", "loan", "deposit", "lending", "mortgage balance", "bank")


def outcome_themes(cfg, ctx):
    """Themes not allowed as controls because they are other outcomes of the same economy (credit aggregates,
    deposits, ...): they carry part of the macro effect, so holding them fixed would hide it ("bad controls").
    impact.excluded_control_themes: "auto" (themes whose name mentions credit, loans, deposits, lending or banks),
    a list of theme names, or [] for none."""
    v = (cfg.get("impact") or {}).get("excluded_control_themes", "auto")
    themes = set(ctx.meta["theme"].dropna().astype(str))
    if v == "auto" or v is None:
        return {t for t in themes if any(w in t.lower() for w in OUTCOME_WORDS)}
    return {t for t in themes if t.lower() in {str(x).lower() for x in (v or [])}}


def double_selection(cfg, ctx, F, focus, log=print):
    """Controls chosen by double selection around the focus drivers.

    Returns:
        (controls list, table of every selected candidate with the equations that picked it, settings dict).
    """
    imp = cfg.get("impact") or {}
    max_c = int(imp.get("max_controls", 4))
    thr = float(imp.get("control_corr", 0.9))
    y, W, Fw = _window(ctx, F)
    foc = list(focus["feature"])
    focus_series = set(focus["series"])
    same_theme_ok = bool(imp.get("controls_same_theme", False))
    blocked = set() if same_theme_ok else set(focus["theme"]) - {""}
    blocked |= outcome_themes(cfg, ctx)
    theme_of = ctx.meta["theme"].to_dict()
    th = {c: theme_of.get(c.split("|")[0], "") for c in Fw.columns}
    usable = [c for c in Fw.columns if c not in foc and c.split("|")[0] not in focus_series
              and th[c] not in blocked and Fw[c].notna().all() and Fw[c].std() > 0]
    same_theme = sum(1 for c in Fw.columns if c.split("|")[0] not in focus_series and th[c] in blocked)
    X = Fw[usable]
    D = Fw[foc]
    corr = np.abs(np.corrcoef(np.column_stack([X.values, D.values]), rowvar=False)[:X.shape[1], X.shape[1]:])
    near = corr.max(axis=1) > thr
    excluded = int(near.sum())
    X = X.loc[:, ~near]
    Xr = _partial(X.values, W)
    picked = {}
    eqs = [("target", _partial(y.values[:, None], W)[:, 0])] + \
          [(f"focus: {f}", _partial(D[[f]].values, W)[:, 0]) for f in foc]
    for label, v in eqs:
        for j in rlasso(Xr, v):
            picked.setdefault(X.columns[j], []).append(label)
    tab = pd.DataFrame([{"feature": f, "selected_by": ", ".join(v), "n_equations": len(v),
                         "predicts_target": "target" in v} for f, v in picked.items()])
    controls = []
    if len(tab):
        # rank: picked by the target equation and by more equations first, then by |t| with the focus drivers in
        t = {}
        for f in tab["feature"]:
            Xc = np.column_stack([W.values, D.values, Fw[f].values])
            b, *_ = np.linalg.lstsq(Xc, y.values, rcond=None)
            e = y.values - Xc @ b
            s2 = e @ e / max(len(e) - Xc.shape[1], 1)
            se = np.sqrt(np.clip(np.linalg.pinv(Xc.T @ Xc)[-1, -1] * s2, 1e-30, None))
            t[f] = abs(b[-1] / se)
        tab["abs_t_with_focus"] = tab["feature"].map(t).round(2)
        tab = tab.sort_values(["predicts_target", "n_equations", "abs_t_with_focus"], ascending=False)
        # one control per variable, and no two controls correlated above the threshold
        for f in tab["feature"]:
            if len(controls) >= max_c:
                break
            if any(f.split("|")[0] == c.split("|")[0] or (th[f] and th[f] == th[c]) for c in controls):
                continue                                # one control per variable and per theme
            if any(abs(np.corrcoef(Fw[f], Fw[c])[0, 1]) > thr for c in controls):
                continue
            controls.append(f)
        # prune: a control whose coefficient in the target equation is insignificant cannot bias the focus effects
        # much (omitted-variable bias = its coefficient x its link with the focus driver), but adds noise
        t_min = float(imp.get("control_t", 1.0))
        dropped = []
        while controls:
            Xc = np.column_stack([W.values, D.values] + [Fw[c].values for c in controls])
            b, *_ = np.linalg.lstsq(Xc, y.values, rcond=None)
            e = y.values - Xc @ b
            s2 = e @ e / max(len(e) - Xc.shape[1], 1)
            se = np.sqrt(np.clip(np.diag(np.linalg.pinv(Xc.T @ Xc)) * s2, 1e-30, None))
            tc = np.abs(b / se)[-len(controls):]
            j = int(np.argmin(tc))
            if tc[j] >= t_min:
                break
            dropped.append(controls.pop(j))
        tab["used"] = tab["feature"].isin(controls)
        tab["pruned"] = tab["feature"].isin(dropped)
        tab = tab.reset_index(drop=True)
    settings = {"candidates": len(usable), "excluded_same_theme_or_outcome": same_theme, "outcome_themes": ", ".join(sorted(outcome_themes(cfg, ctx))), "excluded_near_duplicates": excluded, "equations": len(eqs),
                "selected_any": len(tab), "controls_used": len(controls), "max_controls": max_c, "control_t": float(imp.get("control_t", 1.0)),
                "control_corr": thr, "observations": len(y)}
    return controls, tab, settings
