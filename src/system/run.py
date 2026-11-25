"""System of targets: e.g. a mortgage rate and a deposit rate, mortgage and commercial loans, deposits.

How it works

1. Order. Targets are ordered so that every target comes after the targets it uses ("uses" in the config):
   rates first, then the loans that respond to them, then the deposits that respond to rates and loans.
2. Chain. Each target runs through the complete single-target project (selection, every model family,
   backtest, formula, calculator), with the earlier targets added to its candidate drivers as series
   "SYS.<NAME>" (history = actual values, scenarios = the earlier target's forecast), optionally as a spread
   over a market rate ("SYS.<NAME>_SPREAD"). Their expected signs come from the config. Selection decides
   whether they earn a place.
3. Joint backtest. A target's own backtest uses the ACTUAL path of the earlier targets. The joint backtest
   re-forecasts the whole chain from each origin: the earlier targets' FORECASTS feed the later ones. This is
   the honest accuracy of the system ("chain"), shown next to "chain, actual inputs" (what the single-target
   backtest measures).
4. Out of sample. At each origin, every target's drivers are re-selected with data up to the origin, and the
   chain forecasts itself (optional, system.oos).
5. BVARX challenger. All targets' changes in one VAR(1) with the macro drivers of the chain as exogenous
   inputs; cross-target effects shrunk towards 0 (Minnesota-type prior), own lags and macro effects not shrunk.
6. Reconciliation. For identities such as total loans = mortgage + commercial, the total is also forecast on
   its own and all forecasts are combined so the identity holds (WLS / MinT-diagonal, weights from backtest
   errors).
7. Ratios. E.g. loan-to-deposit under every scenario, against its historical range.

Outputs: outputs/<run>/system/ (tables, charts, summary.md, fitted/) and outputs/<run>/targets/<name>/ (each
target's complete single-target results).
"""
import copy
import json
import os
import pickle

import numpy as np
import pandas as pd
import yaml

from ..features.transforms import feature_matrix, parse_feature
from ..modelling.search import backtest_origins, refit_flags
from ..utils import target as tg
from ..utils.io import Context, add_constructed, read_table, to_quarters

DEFAULTS = {"presets": "fast", "hyperparameters": {}, "target_oos": False, "oos": True, "reconcile": True,
            "bvarx": {"enabled": True, "lambda": 0.2}, "sur": {"enabled": True, "simulations": 300},
            "vecm": {"enabled": True, "lambda": 0.2, "relations": []},
            "date_column": "date", "identities": [], "ratios": []}


def sysname(name):
    return "SYS." + str(name).upper()


def order(targets):
    """Targets sorted so every target comes after the ones it uses."""
    by = {t["name"]: t for t in targets}
    done, out = set(), []

    def visit(n, path=()):
        if n in done:
            return
        if n in path:
            raise ValueError(f"system targets use each other in a circle: {' -> '.join(path + (n,))}")
        for u in by[n].get("uses") or []:
            if u["target"] not in by:
                raise KeyError(f"target '{n}' uses '{u['target']}', which is not a system target")
            visit(u["target"], path + (n,))
        done.add(n)
        out.append(by[n])
    for t in targets:
        visit(t["name"])
    return out


def settings(scfg):
    s = copy.deepcopy(DEFAULTS)
    for k, v in scfg.items():
        s[k] = {**s[k], **v} if isinstance(v, dict) and isinstance(s.get(k), dict) else v
    return s


def tr_row(master, target):
    """The target_names row of a target, or None."""
    from ..utils import master as ms
    tr = ms.target_rows(master)
    hit = tr[tr["target"].map(lambda v: ms._match([v], target) is not None)]
    return next(hit.itertuples(), None)


def resolve_master(scfg, t, path_of, folder):
    """One master file for every target (system.master, or a target's metadata / literature pointing at one):
    replace it by this target's own workbook and metadata, extracted to folder (see src/utils/master.py).
    The target's rows are those of t['master_target'] (default: the target's name)."""
    from ..utils import master as ms
    m = path_of(scfg.get("master"))
    for k in ("metadata", "literature"):
        if ms.is_master(t.get(k)) or (m and not t.get(k)):
            t["_master"] = t.get(k) if ms.is_master(t.get(k)) else m
    if m:
        t["metadata"] = t.get("metadata") or m
        if "literature" not in t or t["literature"] is None:
            t["literature"] = m
    src = t.get("_master") or (t.get("metadata") if ms.is_master(t.get("metadata")) else None)
    if not t.get("master_target") and src:           # the master's target_names: your column -> its target
        tr = ms.target_rows(src)
        yours = {str(y).strip().lower(): tn for tn, y in zip(tr["target"], tr["your_name"]) if y}
        hit = yours.get(str(t.get("column", "")).strip().lower())
        if hit:
            t["master_target"] = hit
    name = t.get("master_target") or t["name"]
    if src and ms.is_master(src):
        x = ms.extract(src, name, folder)
        if x.get("generic"):                         # a target you added without 'like': free signs
            hp = t.setdefault("hyperparameters", {})
            if isinstance(hp, dict):
                hp["features.require_sign_prior"] = False
        if not t.get("form") and not t.get("transform"):
            row = tr_row(src, x["target"])
            if row is not None and row.form:
                t["transform"] = row.form
            if row is not None:
                t.setdefault("units", row.units or None)
                t.setdefault("label", row.label or None)
    if ms.is_master(t.get("metadata")):
        t["metadata"] = ms.resolve(t["metadata"], "metadata", name, folder)
    if ms.is_master(t.get("literature")):
        t["literature"] = ms.extract(t["literature"], name, folder)["workbook"]   # None: no workbook for it
    if not t.get("metadata"):
        raise ValueError(f"system target '{t['name']}' has no metadata (give metadata:, or system.master)")


def apply_names(scfg, base_cfg, path_of, folder):
    """Your macro names (the master's macro_names sheet, or system.names) applied to the system config (uses'
    spread_over, relations' against, ...) and the base config (deflator, constructed series), and passed on to
    every target's run. Returns the map."""
    from ..utils import master as ms
    src = path_of(scfg.get("master")) or next((t.get("_master") for t in scfg["targets"] if t.get("_master")), None)
    names = ms.name_map(src, path_of(scfg.get("names")))
    if not names:
        return names
    for k in ("targets", "vecm", "ratios", "identities"):
        if k in scfg:
            scfg[k] = ms.rename_tree(scfg[k], names)
    ms.rename_config(base_cfg, names)
    hp = scfg.get("hyperparameters") or {}
    if isinstance(hp, dict):
        hp["literature.names"] = ms.write_names(names, folder)
        scfg["hyperparameters"] = hp
    return names


# ------------------------------------------------------------------------------------------------- 1-2 chain
def _augment(base_cfg, t, done, outs, inputs):
    """Macro scenario files and metadata for one target: the base ones plus the earlier targets it uses.

    Returns:
        (macro {scenario: file}, metadata file, extra constructed entries)
    """
    from ..utils.config import load_config  # noqa: F401
    scen = base_cfg["data"]["macro"]["scenarios"]
    meta = read_table(t["metadata"])
    cons, rows = [], []
    for u in t.get("uses") or []:
        up = done[u["target"]]
        nm = sysname(up["name"])
        sp = u.get("spread_over")
        rows.append({"mnemonic": nm, "description": f"{up.get('label') or up['name']} (system: its own forecast)",
                     "geography": "Ontario", "theme": f"System: {up['name']}",
                     "transform": "diff" if up["form"] == "diff" else "dlog",
                     "expected_sign": 0 if sp else int(u.get("sign", 0))})
        if sp:
            cons.append({"name": f"{nm}_SPREAD", "expression": f"`{nm}` - `{sp}`",
                         "description": f"{up.get('label') or up['name']} minus {sp} (system spread, pp)",
                         "geography": "Ontario", "theme": f"System: {up['name']}", "transform": "diff",
                         "expected_sign": int(u.get("sign", 0))})
    if rows:
        meta = pd.concat([meta, pd.DataFrame(rows)], ignore_index=True)
    mf = os.path.join(inputs, f"metadata_{t['name']}.csv")
    meta.to_csv(mf, index=False)
    files = {}
    for s, f in scen.items():
        df = read_table(f)
        dcol = base_cfg["data"]["macro"].get("date_column", "date")
        q = to_quarters(df[dcol])
        for u in t.get("uses") or []:
            up = done[u["target"]]
            path = outs[up["name"]]["paths"][s] if s in outs[up["name"]]["paths"] else outs[up["name"]]["paths"]["__hist__"]
            df[sysname(up["name"])] = pd.Series(path.reindex(q).values, index=df.index)
        p = os.path.join(inputs, f"macro_{t['name']}_{s}.csv".replace(" ", "_"))
        df.to_csv(p, index=False)
        files[s] = p
    return files, mf, cons


def run_chain(scfg, base_cfg, data, out_dir, log=print):
    """Every target through the single-target project, in order. Returns {name: info}."""
    import macroforecast as mf
    from macroforecast.forecaster import PRESETS
    targets = order(scfg["targets"])
    inputs = os.path.join(out_dir, "system", "inputs")
    os.makedirs(inputs, exist_ok=True)
    done, outs = {}, {}
    for t in targets:
        col = t["column"]
        s = data[col].dropna()
        form = tg.resolve(t.get("transform", "auto"), s)
        t = {**t, "form": form}
        log(f"System: target {t['name']} ({form}){' using ' + ', '.join(u['target'] for u in t.get('uses') or []) if t.get('uses') else ''} ...")
        macro, meta_f, cons = _augment(base_cfg, t, done, outs, inputs)
        opts = copy.deepcopy(PRESETS[scfg["presets"]])
        opts = mf.api._merge(opts, mf.parse_options(scfg.get("hyperparameters")))
        opts = mf.api._merge(opts, mf.parse_options(t.get("hyperparameters")))
        opts.setdefault("target", {}).update(transform=form, label=t.get("label") or t["name"],
                                             units=t.get("units") or ("pp" if form == "diff" else None))
        if cons:
            opts.setdefault("data", {})["constructed"] = list(base_cfg["data"].get("constructed") or []) + cons
        series = s.rename(col)
        series.index = series.index.astype(str)
        res = mf.run(series.rename_axis("date"), meta_f, t.get("literature"), target_column=col, macro=macro,
                     events=t.get("events"), options=opts, run_name=t["name"],
                     output_dir=os.path.join(out_dir, "targets"), oos=bool(scfg.get("target_oos")))
        best = (res.best or {}).get("best_model")
        S = res.forecasts
        paths = {"__hist__": series_level(s)}
        for sc, g in S[S["model"] == best].groupby("scenario"):
            fc = pd.Series(g["balance_forecast"].values, index=pd.PeriodIndex(g["date"], freq="Q"))
            paths[sc] = pd.concat([series_level(s), fc])
        outs[t["name"]] = {"results": res, "best": best, "paths": paths, "form": form, "column": col}
        done[t["name"]] = t
        log(f"  {t['name']}: best model {best}")
    return targets, done, outs


def series_level(s):
    x = s.copy()
    x.index = pd.PeriodIndex(x.index.astype(str), freq="Q")
    return x


# ------------------------------------------------------------------------------------------------- 3 joint backtest
class Member:
    """One target inside the joint backtest: its Context, chosen model and the targets it uses."""

    def __init__(self, t, info):
        from ..utils.config import load_config
        from ..utils.saved import load_models
        self.t, self.name, self.form = t, t["name"], info["form"]
        self.cfg = load_config(info["results"].config_file)
        tg.set_form(self.form)
        self.ctx = Context(self.cfg)
        self.models = load_models(info["results"].out_dir)
        self.best = info["best"]
        self.model = self.models.get(self.best)
        if self.model is None:                                  # some model types cannot be stored
            dd = [m for m in self.models.values() if getattr(m, "family", "") in ("Data-driven", "Literature")]
            self.model = dd[0] if dd else next(iter(self.models.values()))
            self.best = self.model.name
        self.uses = [u["target"] for u in t.get("uses") or []]
        self.full = pickle.dumps(self.model)              # the model on all history (the backtest re-fits a copy)
        self.model = pickle.loads(self.full)
        self.P = self.model.panel(self.ctx, self.ctx.base)

    def panel_with(self, upstream):
        """The model's panel with the earlier targets replaced by their forecasts after the origin."""
        mdf = self.ctx.base.copy()
        for name, path in upstream.items():
            col = sysname(name)
            if col in mdf.columns:
                v = mdf[col].copy()
                common = path.index.intersection(v.index)
                v.loc[common] = path.loc[common].values
                mdf[col] = v
        scen, _ = add_constructed(self.cfg, {"x": mdf}, self.ctx.meta.copy())
        return self.model.panel(self.ctx, scen["x"])


def joint_forecast(members, origin, H, refit, start_of, models_at=None):
    """Forecast the chain from one origin. Returns {name: (forecast changes, actual changes, level at origin,
    forecast levels Series)} for the chain and for the chain with actual inputs."""
    out, oracle, paths = {}, {}, {}
    for m in members:
        tg.set_form(m.form)
        mod = models_at.get(m.name, m.model) if models_at else m.model
        P = mod.panel(m.ctx, m.ctx.base) if models_at else m.P
        if refit:
            mod.fit(P, start_of(m), origin)
        if origin not in m.ctx.idx:
            continue
        o = m.ctx.idx.get_loc(origin)
        L0 = float(m.ctx.level.iloc[o])
        Pm = (m.panel_with({u: paths[u] for u in m.uses}) if models_at is None else
              _panel_with(m, mod, {u: paths[u] for u in m.uses})) if m.uses else P
        h = min(H, len(m.ctx.idx) - o - 1)
        if h <= 0:
            continue
        fc = np.asarray(mod.forecast(Pm, origin, h), float)
        fo = np.asarray(mod.forecast(P, origin, h), float) if m.uses else fc
        act = m.ctx.y.iloc[o + 1:o + 1 + h].values.astype(float)
        dates = m.ctx.idx[o + 1:o + 1 + h]
        lv = pd.Series(tg.path(L0, fc), index=dates)
        hist = m.ctx.level.iloc[:o + 1]
        paths[m.name] = pd.concat([hist, lv])
        out[m.name] = (fc, act, L0, lv, m.form)
        oracle[m.name] = (fo, act, L0, pd.Series(tg.path(L0, fo), index=dates), m.form)
    return out, oracle


def _panel_with(m, mod, upstream):
    mdf = m.ctx.base.copy()
    for name, path in upstream.items():
        col = sysname(name)
        if col in mdf.columns:
            common = path.index.intersection(mdf.index)
            mdf.loc[common, col] = path.loc[common].values
    scen, _ = add_constructed(m.cfg, {"x": mdf}, m.ctx.meta.copy())
    return mod.panel(m.ctx, scen["x"])


def _rows(res, origin, method, test, covid):
    rows = []
    for name, (fc, act, L0, lv, form) in res.items():
        tg.set_form(form)
        cf, ca = np.cumsum(fc), np.cumsum(act)
        for h in range(1, len(fc) + 1):
            if np.isnan(ca[h - 1]):
                break
            rows.append({"target": name, "method": method, "test": test, "origin": str(origin), "horizon_q": h,
                         "date": str(lv.index[h - 1]), "forecast_level": float(lv.iloc[h - 1]),
                         "actual_level": float(tg.level_after(L0, ca[h - 1])),
                         "error_pct": float(tg.error(cf[h - 1], ca[h - 1])) * 100,
                         "units": "%" if form == "dlog" else "pp", "covid_window": covid(origin, h)})
    return rows


def covid_checker(cfg):
    dq = [pd.Period(q, freq="Q") for q in cfg["sample"]["dummy_quarters"]]

    def f(origin, h):
        if not dq:
            return False
        w = pd.period_range(origin + 1, origin + h, freq="Q")
        return bool(((w >= min(dq)) & (w <= max(dq))).any())
    return f


def joint_backtest(members, H, log=print, keep=None):
    """Rolling joint backtest at the first target's origins and re-estimation schedule."""
    c0 = members[0].ctx
    origins = [c0.idx[o] for o in backtest_origins(c0)]
    flags = refit_flags(c0, backtest_origins(c0))
    covid = covid_checker(members[0].cfg)
    start = {m.name: pd.Period(m.cfg["sample"]["estimation_start"], freq="Q") for m in members}
    rows = []
    for o, fl in zip(origins, flags):
        res, ora = joint_forecast(members, o, H, fl, lambda m: start[m.name])
        rows += _rows(res, o, "Chain", "backtest", covid)
        rows += _rows({k: v for k, v in ora.items() if members_by(members)[k].uses}, o, "Chain, actual inputs",
                      "backtest", covid)
        if keep is not None:
            keep[str(o)] = res
        if fl and keep is not None:
            keep.setdefault("__models__", {})[str(o)] = {m.name: pickle.dumps(m.model) for m in members}
    return pd.DataFrame(rows)


def members_by(members):
    return {m.name: m for m in members}


def joint_oos(members, scfg, H, log=print):
    """Out-of-sample: at each origin every target's drivers are re-selected with data up to the origin, then the
    chain forecasts itself."""
    from ..pipelines import bayes_first, en_first, factors_first, focus_first, screen_first, stable_first
    from ..utils.cache import StepCache
    from ..validation import oos
    APP = {"screen_first": screen_first, "en_first": en_first, "bayes_first": bayes_first, "focus_first": focus_first,
           "factors_first": factors_first, "stable_first": stable_first}
    c0 = members[0].cfg
    first = pd.Period(c0["oos"]["first_origin"], freq="Q")
    end = members[0].ctx.idx[members[0].ctx.end]
    origins = [q for q in members[0].ctx.idx if first <= q < end][::int(c0["oos"].get("step", 4))]
    covid = covid_checker(c0)
    rows, picks = [], []
    for o in origins:
        at = {}
        for m in members:
            tg.set_form(m.form)
            cache = StepCache(m.cfg["cache"]["dir"], enabled=m.cfg["cache"]["enabled"])
            aps = {n: APP[n] for n in m.cfg["approaches"] if n in APP}
            try:
                ms, sel, est = oos.models_at(m.cfg, m.ctx, aps, m.ctx.idx.get_loc(o), cache)
            except Exception as ex:
                log(f"  oos {o} {m.name}: {ex}")
                continue
            pick = next((x for x in ms if x.name == m.best), None) or \
                next((x for x in ms if x.family == "Data-driven"), ms[0])
            at[m.name] = pick
            picks.append({"origin": str(o), "target": m.name, "model": pick.name,
                          "drivers": ", ".join(getattr(pick, "features", []) or [])})
        start = {m.name: pd.Period(m.cfg["sample"]["estimation_start"], freq="Q") for m in members}
        res, ora = joint_forecast(members, o, H, True, lambda m: start[m.name], models_at=at)
        rows += _rows(res, o, "Chain", "oos", covid)
        rows += _rows({k: v for k, v in ora.items() if members_by(members)[k].uses}, o, "Chain, actual inputs",
                      "oos", covid)
        log(f"  system oos origin {o}")
    return pd.DataFrame(rows), pd.DataFrame(picks)


# ------------------------------------------------------------------------------------------------- 5 BVARX
class BVARX:
    """VAR(1) of every target's change with exogenous macro drivers; cross-target lags shrunk towards 0.

    Equation i: y_i,t = c_i + sum_j a_ij y_j,t-1 + b_i' x_t + e_i,t. Prior: a_ij ~ N(0, (lam * s_i / s_j)^2)
    for j != i (s = residual std. dev. of a univariate AR(1)); own lag, constant and macro effects flat.
    The posterior mean is a ridge regression with those penalties.
    """

    def __init__(self, names, forms, exog, lam=0.2):
        self.names, self.forms, self.exog, self.lam = list(names), dict(forms), list(exog), float(lam)
        self.name = "BVARX (all targets)"

    def fit(self, Y, X, rows):
        """Y: DataFrame of changes (columns = targets), X: DataFrame of exogenous features; rows: index labels."""
        n = len(self.names)
        Yv, Xv = Y.loc[rows].values, X.loc[rows].values
        Yl = Y.shift(1).loc[rows].values
        ok = ~np.isnan(Yv).any(1) & ~np.isnan(Yl).any(1) & ~np.isnan(Xv).any(1)
        Yv, Yl, Xv = Yv[ok], Yl[ok], Xv[ok]
        s = np.array([np.std(Yv[:, i] - np.polyval(np.polyfit(Yl[:, i], Yv[:, i], 1), Yl[:, i])) for i in range(n)])
        Z = np.column_stack([np.ones(len(Yv)), Yl, Xv])
        self.A, self.c, self.B = np.zeros((n, n)), np.zeros(n), np.zeros((n, Xv.shape[1]))
        for i in range(n):
            pen = np.zeros(Z.shape[1])
            for j in range(n):
                if j != i:
                    pen[1 + j] = (s[i] ** 2) / max((self.lam * s[i] / max(s[j], 1e-12)) ** 2, 1e-18)
            coef = np.linalg.solve(Z.T @ Z + np.diag(pen), Z.T @ Yv[:, i])
            self.c[i], self.A[i], self.B[i] = coef[0], coef[1:1 + n], coef[1 + n:]
        self.n_obs = len(Yv)
        return self

    def forecast(self, y0, Xf):
        """y0: last actual changes (n), Xf: exogenous values for the forecast quarters (H x k)."""
        out, y = [], np.asarray(y0, float)
        for x in Xf:
            y = self.c + self.A @ y + self.B @ x
            out.append(y.copy())
        return np.array(out)

    def table(self):
        rows = []
        for i, nm in enumerate(self.names):
            rows.append({"equation": nm, "term": "constant", "coef": self.c[i]})
            rows += [{"equation": nm, "term": f"{self.names[j]} (lag 1)", "coef": self.A[i, j],
                      "cross": j != i} for j in range(len(self.names))]
            rows += [{"equation": nm, "term": f, "coef": self.B[i, k]} for k, f in enumerate(self.exog)]
        return pd.DataFrame(rows)


def bvarx_inputs(members):
    """Changes of every target and the exogenous macro drivers (non-system) of the chain's models."""
    Y = pd.DataFrame({m.name: m.ctx.y for m in members})
    exog = []
    for m in members:
        for f in getattr(m.model, "features", []) or []:
            try:
                code = parse_feature(f)[0]
            except ValueError:
                continue
            if not code.startswith("SYS.") and f not in exog:
                exog.append(f)
    c0 = members[0]
    X = feature_matrix(c0.ctx.base, exog, c0.cfg["features"].get("deflator")) if exog else \
        pd.DataFrame(index=c0.ctx.base.index)
    return Y, X, exog


def bvarx_backtest(members, lam, H, keep=None):
    Y, X, exog = bvarx_inputs(members)
    forms = {m.name: m.form for m in members}
    c0 = members[0].ctx
    covid = covid_checker(members[0].cfg)
    start = pd.Period(members[0].cfg["sample"]["estimation_start"], freq="Q")
    origins = [c0.idx[o] for o in backtest_origins(c0)]
    flags = refit_flags(c0, backtest_origins(c0))
    bv = BVARX(Y.columns, forms, exog, lam)
    rows = []
    for o, fl in zip(origins, flags):
        if fl:
            bv.fit(Y, X, Y.loc[start:o].index)
            if keep is not None:
                keep[str(o)] = pickle.dumps(bv)
        rows += _bvarx_rows(bv, members, Y, X, o, H, covid, "backtest")
    bv.fit(Y, X, Y.loc[start:c0.idx[c0.end]].index)
    return pd.DataFrame(rows), bv, Y, X


def _bvarx_rows(bv, members, Y, X, o, H, covid, test):
    idx = Y.index
    p = idx.get_loc(o)
    h = min(H, len(idx) - p - 1)
    if h <= 0:
        return []
    Xf = X.reindex(idx[p + 1:p + 1 + h]).values
    if np.isnan(Xf).any() or np.isnan(Y.iloc[p].values).any():
        return []
    F = bv.forecast(Y.iloc[p].values, Xf)
    res = {}
    for i, m in enumerate(members):
        L0 = float(m.ctx.level.loc[o])
        act = Y[m.name].iloc[p + 1:p + 1 + h].values
        tg.set_form(m.form)
        res[m.name] = (F[:, i], act, L0, pd.Series(tg.path(L0, F[:, i]), index=idx[p + 1:p + 1 + h]), m.form)
    return _rows(res, o, "BVARX", test, covid)


# ------------------------------------------------------------------------------------------------- 6 reconciliation
def reconcile_levels(total, parts, var_total, var_parts):
    """WLS (MinT with a diagonal error covariance) for total = sum(parts) at one horizon.

    Args:
        total: base forecast of the total. parts: base forecasts of the parts (array).
        var_total, var_parts: error variances (levels) from the backtest.

    Returns:
        reconciled parts (array); the reconciled total is their sum.
    """
    k = len(parts)
    S = np.vstack([np.ones(k), np.eye(k)])
    W = np.diag(np.r_[var_total, var_parts])
    Wi = np.linalg.inv(W)
    yhat = np.r_[total, parts]
    b = np.linalg.solve(S.T @ Wi @ S, S.T @ Wi @ yhat)
    return b


# ------------------------------------------------------------------------------------------------- summaries
def leaderboard(E, hz):
    if E.empty:
        return pd.DataFrame(), pd.DataFrame()
    Ex = E[~E["covid_window"]]
    by_h = (Ex.assign(a=Ex["error_pct"].abs()).groupby(["target", "method", "test", "horizon_q"], sort=False)
            .agg(MAPE_exCOVID=("a", "mean"), bias_exCOVID=("error_pct", "mean"), n=("a", "size"), units=("units", "first"))
            .reset_index())
    rows = []
    for (t, m, te), d in by_h.groupby(["target", "method", "test"], sort=False):
        d = d.set_index("horizon_q")
        r = {"target": t, "method": m, "test": te, "units": d["units"].iloc[0]}
        for h in hz:
            r[f"avg_1_{h}"] = d["MAPE_exCOVID"].loc[:h].mean() if h in d.index else np.nan
        rows.append(r)
    LB = pd.DataFrame(rows)
    have = [h for h in hz if LB[f"avg_1_{h}"].notna().any()]
    col = f"avg_1_{max(have)}" if have else f"avg_1_{max(hz)}"
    LB = LB.sort_values(["test", "target", col]).reset_index(drop=True)
    LB["ranked_on"] = col
    return by_h, LB


def scenario_paths(members, outs, bv, Y, X, scfg):
    """Every target under every scenario: the chain (each target's best model with the earlier targets'
    scenario forecasts) and the BVARX."""
    rows = []
    for m in members:
        tg.set_form(m.form)
        S = outs[m.name]["results"].forecasts
        d = S[S["model"] == outs[m.name]["best"]]
        rows += [{"target": m.name, "method": "Chain", "scenario": r.scenario, "date": r.date,
                  "level": r.balance_forecast} for r in d.itertuples()]
    if bv is not None:
        c0 = members[0]
        hist_end = c0.ctx.idx[c0.ctx.end]
        for sc, mdf in c0.ctx.macro.items():
            Xs = feature_matrix(mdf, bv.exog, c0.cfg["features"].get("deflator")) if bv.exog else \
                pd.DataFrame(index=mdf.index)
            fut = mdf.index[mdf.index > hist_end]
            if not len(fut):
                continue
            F = bv.forecast(Y.loc[hist_end].values, Xs.reindex(fut).values if bv.exog else np.zeros((len(fut), 0)))
            for i, m in enumerate(members):
                tg.set_form(m.form)
                lv = tg.path(float(m.ctx.level.loc[hist_end]), F[:, i])
                rows += [{"target": m.name, "method": "BVARX", "scenario": sc, "date": str(q), "level": float(v)}
                         for q, v in zip(fut, lv)]
    return pd.DataFrame(rows)


def ratio_table(paths, members, ratios, method="Chain"):
    """Ratios such as loan-to-deposit, history and every scenario, with the historical range."""
    rows = []
    hist = {m.name: m.ctx.level for m in members}
    for r in ratios:
        num = sum(hist[n] for n in r["numerator"])
        den = sum(hist[n] for n in r["denominator"])
        h = (num / den).dropna()
        lo, hi = float(h.min()), float(h.max())
        rows += [{"ratio": r["name"], "scenario": "history", "date": str(q), "value": float(v), "hist_min": lo,
                  "hist_max": hi, "outside_history": False} for q, v in h.items()]
        for meth in [x for x in ("Chain", "SUR chain", "VECM", "BVARX") if x in set(paths["method"])]:
            P = paths[paths["method"] == meth]
            for sc, g in P.groupby("scenario"):
                w = g.pivot_table(index="date", columns="target", values="level")
                if not set(r["numerator"] + r["denominator"]) <= set(w.columns):
                    continue
                v = w[r["numerator"]].sum(axis=1) / w[r["denominator"]].sum(axis=1)
                rows += [{"ratio": r["name"], "method": meth, "scenario": sc, "date": q, "value": float(x),
                          "hist_min": lo, "hist_max": hi, "outside_history": not (lo <= x <= hi)} for q, x in v.items()]
    return pd.DataFrame(rows)


def _fan_chart(out, fan, members):
    """SUR fan charts: 5-95% and 25-75% bands of every target, baseline and the most severe scenario."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    scs = list(dict.fromkeys(fan["scenario"]))
    show = [scs[0]] + ([scs[-1]] if len(scs) > 1 else [])
    n = len(members)
    fig, axs = plt.subplots(n, 1, figsize=(7.5, 2.3 * n), squeeze=False)
    colors = ["#3b5ba5", "#c0504d"]
    for ax, m in zip(axs.ravel(), members):
        h = m.ctx.level.iloc[-20:]
        ax.plot(h.index.to_timestamp(), h.values, color="#222", lw=1.5)
        for col, sc in zip(colors, show):
            d = fan[(fan["target"] == m.name) & (fan["scenario"] == sc)]
            t = pd.PeriodIndex(d["date"], freq="Q").to_timestamp()
            ax.fill_between(t, d["p5"], d["p95"], color=col, alpha=0.12, lw=0)
            ax.fill_between(t, d["p25"], d["p75"], color=col, alpha=0.25, lw=0)
            ax.plot(t, d["p50"], color=col, lw=1.2, label=sc)
        ax.set_title(f"{m.name} (SUR chain: 50%, 90% bands)", fontsize=8, loc="left")
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    axs.ravel()[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "system_fan.png"), dpi=150)
    plt.close(fig)


def links_table(members):
    """How each target responds to the earlier targets in its chosen model (from its driver table)."""
    rows = []
    empty = pd.DataFrame(columns=["target", "model", "feature", "coef", "t", "p_value", "reading"])
    for m in members:
        f = os.path.join(m.cfg["out_dir"], "comparison", "3_drivers.csv")
        if not os.path.exists(f):
            continue
        d = pd.read_csv(f)
        d = d[(d["model"] == m.best) & d["term"].astype(str).str.contains("SYS.", regex=False)]
        for r in d.itertuples():
            rows.append({"target": m.name, "model": m.best, "feature": r.term, "coef": r.coef, "t": r.t,
                         "p_value": r.p, "reading": r.reading})
    return pd.DataFrame(rows) if rows else empty


def _chart(out, paths, members):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(members)
    cols = min(3, n)
    rows_ = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rows_, cols, figsize=(4.4 * cols, 2.9 * rows_), squeeze=False)
    for ax, m in zip(axs.ravel(), members):
        h = m.ctx.level.iloc[-24:]
        ax.plot(h.index.to_timestamp(), h.values, color="#222", lw=1.6)
        d = paths[(paths["target"] == m.name) & (paths["method"] == "Chain")]
        for sc, g in d.groupby("scenario", sort=False):
            ax.plot(pd.PeriodIndex(g["date"], freq="Q").to_timestamp(), g["level"].values, lw=1.1, label=sc)
        ax.set_title(f"{m.name}: {m.best}", fontsize=8, loc="left")
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axs.ravel()[n:]:
        ax.axis("off")
    axs.ravel()[0].legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "system_scenarios.png"), dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------- run
def run(config_path, out_root=None, log=print):
    """The whole system analysis from a system config (see config/system_example.yaml)."""
    from macroforecast.api import DEFAULT_BASE, PROJECT_ROOT, _abs
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if (raw.get("system") or {}).get("panel_column"):          # several banks: the panel system
        from . import panel as psys
        return psys.run(config_path, out_root, log)
    scfg = settings(raw["system"])
    root = os.path.dirname(os.path.abspath(config_path))
    proj = PROJECT_ROOT

    def path_of(p):
        if p is None:
            return None
        for b in (os.getcwd(), proj, root):
            q = os.path.join(b, p)
            if os.path.exists(q):
                return os.path.abspath(q)
        return _abs(p)
    for t in scfg["targets"]:
        for k in ("metadata", "literature", "events"):
            t[k] = path_of(t.get(k))
        resolve_master(scfg, t, path_of, os.path.join(_abs(out_root) if out_root else os.path.join(proj, "outputs"),
                                                      raw.get("run_name", "system"), "inputs", "master"))
    with open(DEFAULT_BASE, encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)
    if raw.get("macro"):
        base_cfg["data"]["macro"]["scenarios"] = {k: path_of(v) for k, v in raw["macro"].items()}
    else:
        base_cfg["data"]["macro"]["scenarios"] = {k: os.path.join(proj, v)
                                                   for k, v in base_cfg["data"]["macro"]["scenarios"].items()}
    run_name = raw.get("run_name", "system")
    out_dir = os.path.join(_abs(out_root) if out_root else os.path.join(proj, "outputs"), run_name)
    apply_names(scfg, base_cfg, path_of, os.path.join(out_dir, "inputs", "master"))
    sd = os.path.join(out_dir, "system")
    os.makedirs(sd, exist_ok=True)
    df = read_table(path_of(scfg["data"]))
    df.index = to_quarters(df.pop(scfg.get("date_column", "date")))
    data = df.apply(pd.to_numeric, errors="coerce")

    targets, done, outs = run_chain(scfg, base_cfg, data, out_dir, log)
    log("System: joint backtest (earlier targets' forecasts feed the later ones) ...")
    members = [Member(done[t["name"]], outs[t["name"]]) for t in targets]
    H = int(members[0].cfg["sample"]["horizon"])
    hz = members[0].cfg["sample"]["report_horizons"]
    keep = {}
    E = joint_backtest(members, H, log, keep)
    bv, Y, X, bkeep = None, None, None, {}
    if (scfg.get("bvarx") or {}).get("enabled", True):
        log("System: BVARX challenger ...")
        Eb, bv, Y, X = bvarx_backtest(members, scfg["bvarx"].get("lambda", 0.2), H, bkeep)
        E = pd.concat([E, Eb], ignore_index=True)
    from . import joint
    covid = covid_checker(members[0].cfg)
    start0 = pd.Period(members[0].cfg["sample"]["estimation_start"], freq="Q")
    end0 = members[0].ctx.idx[members[0].ctx.end]
    sur, skeep, fan, sur_pts = None, {}, pd.DataFrame(), pd.DataFrame()
    if (scfg.get("sur") or {}).get("enabled", True):
        log("System: SUR estimation of the chain (equations estimated jointly) ...")
        Es, sur = joint.sur_backtest(members, H, covid, _rows, skeep)
        E = pd.concat([E, Es], ignore_index=True)
        sur_pts, fan = joint.sur_scenarios(members, sur, int(scfg["sur"].get("simulations", 300)))
    vm, vkeep, coint, vec_pts = None, {}, pd.DataFrame(), pd.DataFrame()
    rels = (scfg.get("vecm") or {}).get("relations") or []
    if (scfg.get("vecm") or {}).get("enabled", True) and rels:
        log("System: VECM challenger (long-run relations between targets) ...")
        if Y is None:
            Y, X, _ = bvarx_inputs(members)
        exog = list(X.columns)
        Ev, vm = joint.vecm_backtest(members, rels, scfg["vecm"].get("lambda", 0.2), H, covid, _rows, Y, X, exog, vkeep)
        E = pd.concat([E, Ev], ignore_index=True)
        coint = joint.cointegration_tests(vm, members[0].ctx.base, start0, end0)
        vec_pts = joint.vecm_scenarios(members, vm, Y)
    picks = pd.DataFrame()
    if scfg.get("oos"):
        log("System: out-of-sample test (re-selection at every origin, chained) ...")
        Eo, picks = joint_oos(members, scfg, H, log)
        E = pd.concat([E, Eo], ignore_index=True)
    rec = pd.DataFrame()
    paths = scenario_paths(members, outs, bv, Y, X, scfg)
    paths = pd.concat([paths, sur_pts, vec_pts], ignore_index=True)
    if scfg.get("reconcile") and scfg.get("identities"):
        log("System: reconciliation to identities ...")
        E, rec, paths = reconcile_all(scfg, base_cfg, data, members, outs, E, paths, out_dir, H, log)
    by_h, LB = leaderboard(E, hz)
    ratios = ratio_table(paths, members, scfg.get("ratios") or [])
    links = links_table(members)
    tables = {"1_system_leaderboard": LB.round(4), "2_by_horizon": by_h.round(4), "3_forecast_paths": E.round(6),
              "4_scenario_paths": paths, "5_ratios": ratios, "6_links": links,
              "7_bvarx_coefficients": bv.table() if bv is not None else pd.DataFrame(),
              "8_reconciliation": rec, "9_oos_choices": picks,
              "4b_sur_fan": fan.round(4) if len(fan) else fan,
              "7b_sur_coefficients": sur.table() if sur is not None else pd.DataFrame(),
              "7c_sur_residual_correlation": sur.correlation().round(3).reset_index().rename(columns={"index": "target"})
              if sur is not None else pd.DataFrame(),
              "7d_vecm_coefficients": vm.table() if vm is not None else pd.DataFrame(),
              "7e_cointegration": coint,
              "0_chain": pd.DataFrame([{"order": i + 1, "target": t["name"], "form": outs[t["name"]]["form"],
                                        "uses": ", ".join(u["target"] + (f" (spread over {u['spread_over']})"
                                                                         if u.get("spread_over") else "")
                                                          for u in t.get("uses") or []),
                                        "best_model": members[i].best,
                                        "drivers": ", ".join(getattr(members[i].model, "features", []) or [])}
                                       for i, t in enumerate(targets)])}
    for k, v in tables.items():
        if isinstance(v, pd.DataFrame):
            (v if len(v.columns) else pd.DataFrame(columns=["note"])).to_csv(os.path.join(sd, f"{k}.csv"), index=False)
    fd = os.path.join(sd, "fitted")
    os.makedirs(fd, exist_ok=True)
    with open(os.path.join(fd, "chain_models.pkl"), "wb") as f:
        pickle.dump({m.name: m.full for m in members}, f)
    with open(os.path.join(fd, "backtest_models.pkl"), "wb") as f:
        pickle.dump(keep.get("__models__", {}), f)
    if bv is not None:
        with open(os.path.join(fd, "bvarx.pkl"), "wb") as f:
            pickle.dump({"final": pickle.dumps(bv), "by_origin": bkeep}, f)
    if sur is not None:
        with open(os.path.join(fd, "sur.pkl"), "wb") as f:
            pickle.dump({"final": pickle.dumps(sur), "by_origin": skeep}, f)
    if vm is not None:
        with open(os.path.join(fd, "vecm.pkl"), "wb") as f:
            pickle.dump({"final": pickle.dumps(vm), "by_origin": vkeep}, f)
    _chart(sd, paths, members)
    if len(fan):
        _fan_chart(sd, fan, members)
    _summary(sd, run_name, tables, LB)
    with open(os.path.join(fd, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"order": [t["name"] for t in targets], "best": {m.name: m.best for m in members},
                   "forms": {m.name: m.form for m in members}}, f, indent=1)
    log(f"System results in {sd}")
    return {"leaderboard": LB, "folder": sd, "chain": tables["0_chain"], "links": links, "ratios": ratios}


def reconcile_all(scfg, base_cfg, data, members, outs, E, paths, out_dir, H, log):
    """For each identity (total = sum of parts): forecast the total on its own (single-target project, macro
    drivers only), then reconcile the chain's part forecasts with it, horizon by horizon, in the backtest and
    in every scenario."""
    import macroforecast as mf
    from macroforecast.forecaster import PRESETS
    rec_rows, newE, newP = [], [E], [paths]
    for idn in scfg["identities"]:
        parts = idn["sum"]
        tot = sum(data[members_by(members)[p].t["column"]] for p in parts).dropna()
        opts = copy.deepcopy(PRESETS[scfg["presets"]])
        opts = mf.api._merge(opts, mf.parse_options(scfg.get("hyperparameters")))
        opts.setdefault("target", {}).update(transform="dlog", label=idn["name"])
        s = tot.rename(idn["name"])
        s.index = s.index.astype(str)
        res = mf.run(s.rename_axis("date"), idn.get("metadata") and os.path.abspath(idn["metadata"]) or
                     members_by(members)[parts[0]].t["metadata"], idn.get("literature"), target_column=idn["name"],
                     macro=base_cfg["data"]["macro"]["scenarios"], options=opts, run_name=idn["name"],
                     output_dir=os.path.join(out_dir, "targets"), oos=False)
        best = (res.best or {}).get("best_model")
        bt = pd.read_csv(os.path.join(res.out_dir, "comparison", "12_backtest_forecasts.csv")) \
            if os.path.exists(os.path.join(res.out_dir, "comparison", "12_backtest_forecasts.csv")) else None
        ch = E[(E["method"] == "Chain") & (E["test"] == "backtest") & (E["target"].isin(parts))]
        if bt is None or ch.empty:
            log(f"  {idn['name']}: no backtest paths to reconcile with")
            continue
        bt = bt[bt["model"] == best]
        tcol = "forecast_balance"
        T = bt.rename(columns={tcol: "total_fc"})[["origin", "horizon_q", "total_fc", "actual_balance_organic"]]
        P = ch.pivot_table(index=["origin", "horizon_q"], columns="target", values="forecast_level")
        A = ch.pivot_table(index=["origin", "horizon_q"], columns="target", values="actual_level")
        J = P.join(T.set_index(["origin", "horizon_q"]), how="inner")
        if J.empty:
            continue
        # weights from mean squared errors (bias counts too), dummy windows left out
        cov = ch.groupby(["origin", "horizon_q"])["covid_window"].any()
        var = {}
        for h in sorted(set(J.index.get_level_values(1))):
            Jh = J.xs(h, level=1)
            Ah = A.xs(h, level=1).reindex(Jh.index)
            ok = ~cov.xs(h, level=1).reindex(Jh.index).fillna(False).astype(bool).values
            if ok.sum() < 3:
                ok[:] = True
            var[h] = (np.nanmean(((Jh["total_fc"] - Ah[parts].sum(axis=1)).values[ok]) ** 2) + 1e-9,
                      np.array([np.nanmean(((Jh[p] - Ah[p]).values[ok]) ** 2) + 1e-9 for p in parts]))
        rows = []
        for (o, h), r in J.iterrows():
            vt, vp = var[h]
            b = reconcile_levels(r["total_fc"], r[parts].values.astype(float), vt, vp)
            for p, v in zip(parts, b):
                a = A.loc[(o, h), p]
                rows.append({"target": p, "method": "Chain, reconciled", "test": "backtest", "origin": o,
                             "horizon_q": h, "forecast_level": v, "actual_level": a,
                             "error_pct": (v / a - 1) * 100, "units": "%",
                             "covid_window": bool(E[(E["origin"] == o) & (E["horizon_q"] == h)]["covid_window"].any())})
        newE.append(pd.DataFrame(rows))
        for h, (vt, vp) in var.items():
            w = 1 / np.r_[vt, vp]
            rec_rows.append({"identity": idn["name"], "horizon_q": h, "total_model": best,
                             **{f"weight_{n}": x / w.sum() for n, x in zip([idn["name"]] + parts, w)}})
        S = res.forecasts
        S = S[S["model"] == best]
        ch_s = paths[paths["method"] == "Chain"]
        for sc, g in S.groupby("scenario"):
            w = ch_s[ch_s["scenario"] == sc].pivot_table(index="date", columns="target", values="level")
            if not set(parts) <= set(w.columns):
                continue
            hzn = 0
            out = []
            for d, tv in zip(g["date"], g["balance_forecast"]):
                hzn += 1
                if d not in w.index:
                    continue
                vt, vp = var.get(min(hzn, max(var)), next(iter(var.values())))
                b = reconcile_levels(tv, w.loc[d, parts].values.astype(float), vt, vp)
                out += [{"target": p, "method": "Chain, reconciled", "scenario": sc, "date": d, "level": float(v)}
                        for p, v in zip(parts, b)]
                out.append({"target": idn["name"], "method": "Total model", "scenario": sc, "date": d, "level": float(tv)})
            newP.append(pd.DataFrame(out))
    return pd.concat(newE, ignore_index=True), pd.DataFrame(rec_rows), pd.concat(newP, ignore_index=True)


def _summary(sd, run_name, tables, LB):
    L = [f"# System run: {run_name}", "", "## Chain (in order)", "",
         "| # | Target | Form | Uses | Best model | Drivers |", "|---|---|---|---|---|---|"]
    for r in tables["0_chain"].itertuples():
        L.append(f"| {r.order} | {r.target} | {r.form} | {r.uses or '-'} | {r.best_model} | {r.drivers} |")
    if len(LB):
        col = LB["ranked_on"].iloc[0]
        L += ["", f"## Accuracy ({col}: average error over 1..h quarters; % for balances, pp for rates)", "",
              "| Test | Target | Method | Error |", "|---|---|---|---|"]
        for r in LB.itertuples():
            L.append(f"| {r.test} | {r.target} | {r.method} | {getattr(r, col):.3f} {r.units} |")
    if len(tables["6_links"]):
        L += ["", "## Links between targets (chosen models)", ""]
        L += [f"- {r.target}: {r.reading}" for r in tables["6_links"].itertuples()]
    C = tables.get("7c_sur_residual_correlation")
    if isinstance(C, pd.DataFrame) and len(C):
        L += ["", "## SUR: correlation of the equations' surprises", "", "| | " + " | ".join(C.columns[1:]) + " |",
              "|---" * len(C.columns) + "|"]
        L += ["| " + " | ".join([str(r[0])] + [f"{v:.2f}" for v in r[1:]]) + " |" for r in C.itertuples(index=False)]
    K = tables.get("7e_cointegration")
    if isinstance(K, pd.DataFrame) and len(K):
        L += ["", "## VECM: long-run relations", ""]
        for r in K.itertuples():
            if r.test == "ADF on the relation":
                L.append(f"- {r.relation}: ADF {r.statistic:.2f} (5% critical {r.crit_5pct:.2f}) -> "
                         f"{'stationary: a long-run anchor' if r.stationary_5pct else 'not stationary at 5%'}")
            elif "estimated number" in str(r.relation):
                L.append(f"- Johansen trace: {int(r.statistic)} long-run relation(s) among the targets' levels")
    R = tables["5_ratios"]
    if len(R):
        out = R[R["outside_history"]]
        L += ["", "## Ratios", "", f"{len(out)} scenario quarters outside the historical range"
              + (": " + ", ".join(sorted(set(out["scenario"] + " " + out["ratio"])))[:400] if len(out) else ".")]
    open(os.path.join(sd, "summary.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
