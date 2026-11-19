"""System with panel: every bank has the same chain of targets (own rates -> loans -> deposits), and every
target's equation is estimated across all banks with the panel models.

How it works

1. Order. As in the system: each target comes after the targets it uses.
2. Each target is a panel. Candidate drivers: the macro features (the same for every bank) plus the earlier
   targets of the SAME bank ("SYS.<NAME>", or a spread over a market rate), with the signs from the config.
   Drivers are chosen on all banks together (pooled Driscoll-Kraay screen and search, as in run_panel.py).
   Competing panel models: separate (each bank alone), pooled fixed effects, partial pooling (shrinkage) and
   the AR(1) benchmark, with the panel drivers and with the literature core. The most accurate across banks
   (backtest, mean of the banks' errors) is the target's model in the chain.
3. Joint backtest. From each origin, every bank's chain forecasts itself: the bank's own forecast rates feed
   its loan equations, its forecast loans and rates feed its deposit equation ("Panel chain"); next to it the
   same models with the actual values of the earlier targets ("Panel chain, actual inputs").
4. Scenarios: every bank, every target, every scenario, chained. Ratios (e.g. loan-to-deposit) per bank
   against the bank's own historical range.
5. Links: the pooled coefficient of each earlier target in each equation (Driscoll-Kraay t) and every bank's
   own (partial pooling) coefficient, next to each other.

Outputs: outputs/<run>/system_panel/ (tables, charts, summary.md, fitted/) and one folder per target with its
panel selection, leaderboards, coefficients and shrinkage.
"""
import copy
import json
import os
import pickle

import numpy as np
import pandas as pd
import yaml

from ..features.transforms import all_features, candidate_forms, feature_matrix
from ..modelling.search import backtest_origins, refit_flags
from ..panel.data import Panel
from ..panel.estimate import KINDS, PanelModel
from ..panel.run import summarise
from ..panel.select import BankFeatures, select
from ..utils import target as tg
from .run import apply_names, order, resolve_master, settings, sysname

SYS_TRS = {"diff": ["diff", "diff2", "diff4"], "dlog": ["dlog", "yoy", "dlog2"]}


def _target_cfg(base_cfg, scfg, t, data_file, pc, out_t):
    """The full config of one target (written next to its results and loaded with the project defaults)."""
    import macroforecast as mf
    from ..utils.config import load_config
    c = copy.deepcopy(base_cfg)
    c = mf.api._merge(c, mf.parse_options(scfg.get("hyperparameters")))
    c = mf.api._merge(c, mf.parse_options(t.get("hyperparameters")))
    c["run_name"] = t["name"]
    c["output_dir"] = os.path.dirname(out_t)
    c["data"]["bank"].update(file=data_file, target_column=t["column"], date_column=scfg.get("date_column", "date"),
                             sheet=None, frequency="Q", panel_column=pc)
    ev = t.get("events") or scfg.get("events")
    c["data"]["events"].update(file=ev, bank_column=pc if ev else None)
    c["data"]["metadata"]["file"] = t["metadata"]
    c.setdefault("literature", {})["workbook"] = t.get("literature")
    c.setdefault("target", {}).update(transform=t.get("transform", "auto"), label=t.get("label") or t["name"])
    c.setdefault("panel", {})
    c["panel"]["models"] = [k for k in (c["panel"].get("models") or ["separate", "pooled", "partial"]) if k != "topdown"]
    c["panel"]["driver_sets"] = ["panel", "core"] if t.get("literature") else ["panel"]
    given = (mf.parse_options(scfg.get("hyperparameters")).get("sample") or {})
    for k in ("hist_end", "estimation_start", "backtest_start"):
        if k not in given:
            c["sample"].pop(k, None)
    if "backtest_end" not in given:
        c["sample"]["backtest_end"] = None
    if "first_origin" not in (mf.parse_options(scfg.get("hyperparameters")).get("oos") or {}):
        c.setdefault("oos", {})["first_origin"] = None
    miss = [k for k in ("hist_end", "estimation_start", "backtest_start") if k not in c["sample"]]
    if miss:
        g = mf.infer_dates(c)
        for k in miss:
            c["sample"][k] = g[k]
    os.makedirs(out_t, exist_ok=True)
    f = os.path.join(out_t, "target_config.yaml")
    with open(f, "w", encoding="utf-8") as fh:
        yaml.safe_dump(c, fh, sort_keys=False, allow_unicode=True)
    return load_config(f)


def sys_names(t, up, max_lag):
    """Candidate feature names for an earlier target `up` used by target t."""
    out = []
    for u in t.get("uses") or []:
        if u["target"] != up["name"]:
            continue
        if u.get("spread_over"):
            base, trs = sysname(up["name"]) + "_SPREAD", SYS_TRS["diff"]
        else:
            base, trs = sysname(up["name"]), SYS_TRS[up["form"]]
        out += [f"{base}|{tr}|L{l}" for tr in trs for l in range(0, max_lag + 1)]
    return out


def bank_sys_frame(index, macro, levels, t, done, deflator):
    """One bank's system series (earlier targets' levels, and spreads) on `index`, ready for feature_matrix."""
    df = pd.DataFrame(index=index)
    for u in t.get("uses") or []:
        up = done[u["target"]]
        lv = levels.get(up["name"])
        s = lv.reindex(index) if lv is not None else pd.Series(np.nan, index=index)
        df[sysname(up["name"])] = s.values
        if u.get("spread_over"):
            df[sysname(up["name"]) + "_SPREAD"] = s.values - macro[u["spread_over"]].reindex(index).values
    return df


def sys_cube(pn, t, done, uplevels, names, deflator, index=None, macro=None):
    """Bank-specific system features: (banks, dates, k) on `index` (default: the panel's dates)."""
    index = pn.dates if index is None else index
    macro = pn.base.base if macro is None else macro
    cube = np.full((len(pn.banks), len(index), len(names)), np.nan)
    for i, b in enumerate(pn.banks):
        df = bank_sys_frame(index, macro, {k: v.get(b) for k, v in uplevels.items()}, t, done, deflator)
        if len(df.columns):
            cube[i] = feature_matrix(df, names, deflator).reindex(index).values
    return cube


def sys_meta(t, done):
    rows = []
    for u in t.get("uses") or []:
        up = done[u["target"]]
        nm = sysname(up["name"]) + ("_SPREAD" if u.get("spread_over") else "")
        rows.append({"mnemonic": nm, "description": f"{up.get('label') or up['name']} of the same bank"
                     + (f" minus {u['spread_over']}" if u.get("spread_over") else "") + " (system)",
                     "geography": "Ontario", "theme": f"System: {up['name']}", "expected_sign": int(u.get("sign", 0)),
                     "transform": "diff" if (u.get("spread_over") or up["form"] == "diff") else "dlog"})
    return pd.DataFrame(rows)


class Node:
    """One target of the panel system: its panel, features, chosen drivers, competing models and the winner."""

    def __init__(self, t, cfg, pn, F, FS, sel, core, models, LB, O, best):
        self.t, self.name, self.cfg, self.pn = t, t["name"], cfg, pn
        self.F, self.FS, self.sel, self.core, self.models, self.LB, self.O = F, FS, sel, core, models, LB, O
        self.best = best
        self.uses = [u["target"] for u in t.get("uses") or []]
        self.form = tg.form()

    def bank_X(self, bank, feats, index=None, macro=None, uplevels=None, done=None):
        """One bank's (dates, k) feature values: macro features from `macro` (default: history), system
        features from `uplevels` ({target: {bank: level Series}}; default: actual history)."""
        index = self.pn.dates if index is None else index
        macro = self.pn.base.base if macro is None else macro
        defl = self.cfg["features"].get("deflator")
        common = [f for f in feats if not f.startswith("SYS.")]
        sysf = [f for f in feats if f.startswith("SYS.")]
        out = pd.DataFrame(index=index)
        if common:
            out = feature_matrix(macro, common, defl).reindex(index)
        if sysf:
            lv = uplevels if uplevels is not None else {}
            df = bank_sys_frame(index, macro, {k: v.get(bank) for k, v in lv.items()}, self.t, done, defl)
            out = pd.concat([out, feature_matrix(df, sysf, defl).reindex(index)], axis=1)
        return out[feats].values if feats else np.zeros((len(index), 0))


def build_node(t, base_cfg, scfg, data_file, pc, out_dir, done, nodes, log):
    out_t = os.path.join(out_dir, "targets", t["name"])
    cfg = _target_cfg(base_cfg, scfg, t, data_file, pc, out_t)
    pdir = os.path.join(out_t, "panel")
    os.makedirs(pdir, exist_ok=True)
    pn = Panel(cfg, pdir, log)
    t = {**t, "form": tg.form()}
    forms = candidate_forms(cfg, pn.base.hist, pn.base.meta)
    F = all_features(cfg, pn.base, forms)
    names = []
    for u in t.get("uses") or []:
        names += sys_names(t, done[u["target"]], int(cfg["features"].get("max_lag", 4)))
    names = list(dict.fromkeys(names))
    uplevels = {u: nodes[u].pn.levels for u in [x["target"] for x in t.get("uses") or []]}
    defl = cfg["features"].get("deflator")
    cube = sys_cube(pn, t, done, uplevels, names, defl) if names else None
    FS = BankFeatures(F, cube, names, sys_meta(t, done) if names else None)
    start, end = cfg["sample"]["estimation_start"], cfg["sample"]["hist_end"]
    sel = select(cfg, pn, FS, start, end, log)
    core = []
    if t.get("literature"):
        from ..models import zoo
        spec = next((m for m in zoo.literature_models(cfg, pn.base, lambda *_: None) if m.get("core")), None)
        core = list(spec["features"]) if spec else []
    missing = [f for f in core if f not in F.columns]
    if missing:
        F2 = pd.concat([F, feature_matrix(pn.base.hist, missing, defl)], axis=1)
        FS = BankFeatures(F2, cube, names, FS.bank_meta)
    p = cfg["panel"]
    models = [PanelModel("ar1", [], "AR(1) benchmark (each bank)", p["min_obs"], p["dk_lags"])]
    for label, feats in (("panel drivers", sel["features"]), ("literature core", core)):
        if feats:
            for kind in p["models"]:
                models.append(PanelModel(kind, feats, f"{KINDS[kind]}: {label}", p["min_obs"], p["dk_lags"]))
    # backtest of every model with the earlier targets at their actual values
    from ..panel.run import backtest as pbacktest
    E, coef_paths, at = pbacktest(cfg, pn, models, FS, log)
    weights = {b: float(pn.levels[b].iloc[-1]) for b in pn.banks}
    by_h, LB, O, col = summarise(E, int(cfg["sample"]["horizon"]), cfg["sample"]["report_horizons"], weights)
    best_name = O.iloc[0]["model"]
    for m in models:
        X = FS.array(m.features) if m.features else np.zeros((len(pn.dates), 0))
        m.fit(pn, X, start, end)
    best = next(m for m in models if m.name == best_name)
    for k, v in dict(sel["tables"], **{"1_leaderboard_overall": O.round(4), "2_leaderboard_by_bank": LB.round(4),
                                       "3_backtest_by_horizon": by_h.round(4), "4_backtest_forecasts": E.round(6),
                                       "5_coefficients": pd.DataFrame([r for m in models for r in m.coefficient_rows()]),
                                       "6_shrinkage": pd.concat([m.weights.assign(model=m.name) for m in models
                                                                 if m.weights is not None], ignore_index=True)
                                       if any(m.weights is not None for m in models) else pd.DataFrame(),
                                       "9_coefficients_by_origin": coef_paths}).items():
        if isinstance(v, pd.DataFrame):
            v.to_csv(os.path.join(pdir, f"{k}.csv"), index=False)
    with open(os.path.join(pdir, "models.pkl"), "wb") as f:
        pickle.dump({m.name: pickle.dumps(m) for m in models}, f)
    log(f"  {t['name']}: panel drivers {sel['features'] or '(none)'}; best across banks: {best_name}")
    return t, Node(t, cfg, pn, F, FS, sel, core, models, LB, O, best)


def chain_forecast(nodes, order_, origin_lab, H, refit, actual=False, scen=None):
    """Every bank's chain from one origin. Returns {(target, bank): (changes, level Series)}.

    scen: scenario name (forecast past the history with that scenario's macro path), or None (backtest).
    actual: earlier targets at their actual values (backtest only).
    """
    out, paths = {}, {}
    for name in order_:
        nd = nodes[name]
        tg.set_form(nd.form)
        pn, m = nd.pn, nd.best
        start = nd.cfg["sample"]["estimation_start"]
        if refit:
            X = nd.FS.array(m.features) if m.features else np.zeros((len(pn.dates), 0))
            m.fit(pn, X, start, origin_lab)
        macro = pn.base.base if scen is None else pn.base.macro[scen]
        index = pn.dates if scen is None else macro.index
        if origin_lab not in index:
            continue
        o = index.get_loc(origin_lab)
        paths[name] = {}
        for b in pn.active(origin_lab):
            if b not in m.coef:
                continue
            ylast = pn.ys[b].get(origin_lab, np.nan)
            if pd.isna(ylast):
                continue
            if nd.uses and not actual:
                uplv = {u: {b: paths[u][b]} for u in nd.uses if u in paths and b in paths[u]}
                uplv.update({u: {b: nodes[u].pn.levels[b]} for u in nd.uses if u not in uplv
                             and b in nodes[u].pn.levels})
            else:
                uplv = {u: {b: nodes[u].pn.levels[b]} for u in nd.uses if b in nodes[u].pn.levels}
            X = nd.bank_X(b, m.features, index, macro, uplv, {nm: nodes[nm].t for nm in nodes})
            h = min(H, len(index) - o - 1)
            if scen is None:
                h = min(h, pn.pos(pn.dates[-1]) - o)
            if h <= 0:
                continue
            ytot = float(pn.base.y.loc[origin_lab]) if origin_lab in pn.base.y.index else np.nan
            g = m.forecast(pn, X, b, o, h, {"bank": float(ylast), "total": ytot})
            L0 = float(pn.levels[b].loc[origin_lab])
            lv = pd.Series(tg.path(L0, g), index=index[o + 1:o + 1 + len(g)])
            hist = pn.levels[b].loc[:origin_lab]
            paths[name][b] = pd.concat([hist, lv])
            out[(name, b)] = (g, lv)
    return out


def _err_rows(nodes, res, origin, method, covid):
    rows = []
    for (name, b), (g, lv) in res.items():
        nd = nodes[name]
        tg.set_form(nd.form)
        y = nd.pn.ys[b]
        act = np.array([y.get(q, np.nan) for q in lv.index], float)
        cf, ca = np.cumsum(g), np.cumsum(act)
        L0 = float(nd.pn.levels[b].loc[origin])
        for hh in range(1, len(g) + 1):
            if np.isnan(ca[hh - 1]):
                break
            rows.append({"target": name, "bank": b, "method": method, "origin": str(origin), "horizon_q": hh,
                         "date": str(lv.index[hh - 1]), "forecast_level": float(lv.iloc[hh - 1]),
                         "actual_level": float(tg.level_after(L0, ca[hh - 1])),
                         "error_pct": float(tg.error(cf[hh - 1], ca[hh - 1])) * 100,
                         "units": "%" if nd.form == "dlog" else "pp", "covid_window": covid(origin, hh)})
    return rows


def links(nodes):
    """Coefficients of the earlier targets in each target's equations: the chosen model first, then every other
    panel model with the same driver (pooled value with its Driscoll-Kraay t; each bank's own value; for
    partial pooling, each bank's weight on its own estimate)."""
    rows = []
    for name, nd in nodes.items():
        sysf = [f for f in nd.best.features if f.startswith("SYS.")]
        if not sysf:
            continue
        ms = [nd.best] + [m for m in nd.models if m is not nd.best and m.kind in ("pooled", "partial", "separate")
                          and any(f in m.features for f in sysf)]
        for m in ms:
            pt = m.pooled_table.set_index("term") if m.pooled_table is not None else None
            for j, f in enumerate(m.features):
                if f not in sysf:
                    continue
                r = {"target": name, "model": m.name, "chosen": m is nd.best, "feature": f}
                if pt is not None and f in pt.index:
                    r.update(pooled=float(pt.loc[f, "coef"]), pooled_t_dk=float(pt.loc[f, "t_dk"]))
                for b, c in m.coef.items():
                    if "beta" in c:
                        r[f"bank: {b}"] = float(c["beta"][j])
                if m.weights is not None:
                    w = m.weights[m.weights["term"] == f].set_index("bank")["weight_on_own"]
                    for b, v in w.items():
                        r[f"weight on own: {b}"] = float(v)
                rows.append(r)
    return pd.DataFrame(rows)


def run(config_path, out_root=None, log=print):
    """The panel system from a config with system.panel_column (see config/panel_system_example.yaml)."""
    from macroforecast.api import DEFAULT_BASE, PROJECT_ROOT, _abs
    from .run import covid_checker
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    scfg = settings(raw["system"])
    pc = scfg.get("panel_column")
    if not pc:
        raise ValueError("system.panel_column is not set: name the column that holds the bank")
    root = os.path.dirname(os.path.abspath(config_path))

    def path_of(p):
        if p is None:
            return None
        for b_ in (os.getcwd(), PROJECT_ROOT, root):
            q = os.path.join(b_, p)
            if os.path.exists(q):
                return os.path.abspath(q)
        return _abs(p)
    for t in scfg["targets"]:
        for k in ("metadata", "literature", "events"):
            t[k] = path_of(t.get(k))
        resolve_master(scfg, t, path_of, os.path.join(_abs(out_root) if out_root else os.path.join(PROJECT_ROOT, "outputs"),
                                                      raw.get("run_name", "system_panel"), "inputs", "master"))
    with open(DEFAULT_BASE, encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)
    if raw.get("macro"):
        base_cfg["data"]["macro"]["scenarios"] = {k: path_of(v) for k, v in raw["macro"].items()}
    else:
        base_cfg["data"]["macro"]["scenarios"] = {k: os.path.join(PROJECT_ROOT, v)
                                                   for k, v in base_cfg["data"]["macro"]["scenarios"].items()}
    run_name = raw.get("run_name", "panel_system")
    out_dir = os.path.join(_abs(out_root) if out_root else os.path.join(PROJECT_ROOT, "outputs"), run_name)
    apply_names(scfg, base_cfg, path_of, os.path.join(out_dir, "inputs", "master"))
    sd = os.path.join(out_dir, "system_panel")
    os.makedirs(sd, exist_ok=True)
    data_file = path_of(scfg["data"])

    targets = order(scfg["targets"])
    done, nodes = {}, {}
    for t in targets:
        log(f"Panel system: target {t['name']}{' using ' + ', '.join(u['target'] for u in t.get('uses') or []) if t.get('uses') else ''} ...")
        t2, nd = build_node(t, base_cfg, scfg, data_file, pc, out_dir, done, nodes, log)
        done[t["name"]] = t2
        nodes[t["name"]] = nd
    order_ = [t["name"] for t in targets]
    for nd in nodes.values():
        nd.t = done[nd.name]
    # joint backtest: every bank's chain forecasts itself
    log("Panel system: joint backtest (each bank's forecast rates and loans feed its later targets) ...")
    n0 = nodes[order_[0]]
    c0 = n0.pn.base
    pos = backtest_origins(c0)
    origins, flags = [c0.idx[o] for o in pos], refit_flags(c0, pos)
    H = int(n0.cfg["sample"]["horizon"])
    covid = covid_checker(n0.cfg)
    rows, keep = [], {}
    for o, fl in zip(origins, flags):
        res = chain_forecast(nodes, order_, o, H, fl)
        rows += _err_rows(nodes, res, o, "Panel chain", covid)
        rows += _err_rows(nodes, {k: v for k, v in chain_forecast(nodes, order_, o, H, False, actual=True).items()
                                  if nodes[k[0]].uses}, o, "Panel chain, actual inputs", covid)
        if fl:
            keep[str(o)] = {nm: pickle.dumps(nodes[nm].best) for nm in order_}
    E = pd.DataFrame(rows)
    # final estimates on all history, scenarios
    for nd in nodes.values():
        tg.set_form(nd.form)
        m = nd.best
        X = nd.FS.array(m.features) if m.features else np.zeros((len(nd.pn.dates), 0))
        m.fit(nd.pn, X, nd.cfg["sample"]["estimation_start"], nd.cfg["sample"]["hist_end"])
    hist_end = pd.Period(n0.cfg["sample"]["hist_end"], freq="Q")
    log("Panel system: scenarios ...")
    paths = []
    for sc in n0.pn.base.macro:
        res = chain_forecast(nodes, order_, hist_end, 10 ** 6, False, scen=sc)
        for (name, b), (g, lv) in res.items():
            paths += [{"target": name, "bank": b, "scenario": sc, "date": str(q), "level": float(v)}
                      for q, v in lv.items()]
    P = pd.DataFrame(paths)
    # accuracy tables
    Ex = E[~E["covid_window"]] if len(E) else E
    hz = n0.cfg["sample"]["report_horizons"]
    by = (Ex.assign(a=Ex["error_pct"].abs()).groupby(["target", "bank", "method", "horizon_q"])
          .agg(MAPE_exCOVID=("a", "mean"), n=("a", "size"), units=("units", "first")).reset_index())
    rows = []
    for (t_, b_, m_), d in by.groupby(["target", "bank", "method"]):
        d = d.set_index("horizon_q")
        r = {"target": t_, "bank": b_, "method": m_, "units": d["units"].iloc[0]}
        for h in hz:
            r[f"avg_1_{h}"] = d["MAPE_exCOVID"].loc[:h].mean() if h in d.index else np.nan
        rows.append(r)
    LBb = pd.DataFrame(rows)
    have = [h for h in hz if LBb[f"avg_1_{h}"].notna().any()]
    col = f"avg_1_{max(have)}"
    LB = LBb.groupby(["target", "method", "units"])[[f"avg_1_{h}" for h in hz]].mean().reset_index()
    LB["ranked_on"] = col
    # ratios per bank
    ratios = []
    for r in scfg.get("ratios") or []:
        for b in n0.pn.banks:
            try:
                num = sum(nodes[n].pn.levels[b] for n in r["numerator"])
                den = sum(nodes[n].pn.levels[b] for n in r["denominator"])
            except KeyError:
                continue
            hh = (num / den).dropna()
            lo, hi = float(hh.min()), float(hh.max())
            w = P[(P["bank"] == b)].pivot_table(index=["scenario", "date"], columns="target", values="level")
            if not set(r["numerator"] + r["denominator"]) <= set(w.columns):
                continue
            v = w[r["numerator"]].sum(axis=1) / w[r["denominator"]].sum(axis=1)
            ratios += [{"ratio": r["name"], "bank": b, "scenario": sc, "date": d, "value": float(x), "hist_min": lo,
                        "hist_max": hi, "last_actual": float(hh.iloc[-1]), "outside_history": not (lo <= x <= hi)}
                       for (sc, d), x in v.items()]
    R = pd.DataFrame(ratios)
    L = links(nodes)
    chain = pd.DataFrame([{"order": i + 1, "target": n, "form": nodes[n].form,
                           "uses": ", ".join(u["target"] + (f" (spread over {u['spread_over']})" if u.get("spread_over")
                                                            else "") for u in nodes[n].t.get("uses") or []),
                           "panel_drivers": ", ".join(nodes[n].sel["features"]),
                           "best_model": nodes[n].best.name, "drivers": ", ".join(nodes[n].best.features),
                           "banks": len(nodes[n].pn.banks)} for i, n in enumerate(order_)])
    tables = {"0_chain": chain, "1_system_leaderboard": LB.round(4), "2_by_bank": LBb.round(4),
              "3_forecast_paths": E.round(6), "4_scenario_paths": P, "5_ratios": R, "6_links": L,
              "7_target_leaderboards": pd.concat([nodes[n].O.assign(target=n) for n in order_], ignore_index=True)}
    for k, v in tables.items():
        v.to_csv(os.path.join(sd, f"{k}.csv"), index=False)
    fd = os.path.join(sd, "fitted")
    os.makedirs(fd, exist_ok=True)
    with open(os.path.join(fd, "chain_models.pkl"), "wb") as f:
        pickle.dump({n: pickle.dumps(nodes[n].best) for n in order_}, f)
    with open(os.path.join(fd, "backtest_models.pkl"), "wb") as f:
        pickle.dump(keep, f)
    with open(os.path.join(fd, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"order": order_, "banks": n0.pn.banks, "best": {n: nodes[n].best.name for n in order_},
                   "forms": {n: nodes[n].form for n in order_}}, f, indent=1)
    _chart(sd, P, nodes, order_)
    _summary(sd, run_name, chain, LB, L, R, col)
    log(f"Panel system results in {sd}")
    return {"folder": sd, "chain": chain, "leaderboard": LB, "links": L, "ratios": R}


def _chart(sd, P, nodes, order_):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    banks = nodes[order_[0]].pn.banks
    sc0 = P["scenario"].iloc[0] if len(P) else None
    fig, axs = plt.subplots(len(order_), 1, figsize=(7.5, 2.2 * len(order_)), squeeze=False)
    for ax, n in zip(axs.ravel(), order_):
        nd = nodes[n]
        for b in banks:
            if b not in nd.pn.levels:
                continue
            h = nd.pn.levels[b].iloc[-16:]
            line, = ax.plot(h.index.to_timestamp(), h.values, lw=1.3)
            d = P[(P["target"] == n) & (P["bank"] == b) & (P["scenario"] == sc0)]
            ax.plot(pd.PeriodIndex(d["date"], freq="Q").to_timestamp(), d["level"].values, lw=1.1, ls="--",
                    color=line.get_color(), label=b)
        ax.set_title(f"{n} ({nd.best.name}; {sc0}, dashed)", fontsize=8, loc="left")
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    axs.ravel()[0].legend(fontsize=6, frameon=False, ncol=min(5, len(banks)))
    fig.tight_layout()
    fig.savefig(os.path.join(sd, "system_panel_scenarios.png"), dpi=150)
    plt.close(fig)


def _summary(sd, run_name, chain, LB, L, R, col):
    lines = [f"# Panel system: {run_name}", "", "## Chain", "",
             "| # | Target | Form | Uses | Chosen on all banks | Best across banks |", "|---|---|---|---|---|---|"]
    for r in chain.itertuples():
        lines.append(f"| {r.order} | {r.target} | {r.form} | {r.uses or '-'} | {r.panel_drivers or '-'} | {r.best_model} |")
    lines += ["", f"## Accuracy ({col}, mean across banks; % for balances, pp for rates)", "",
              "| Target | Method | Error |", "|---|---|---|"]
    for r in LB.itertuples():
        lines.append(f"| {r.target} | {r.method} | {getattr(r, col):.3f} {r.units} |")
    if len(L):
        lines += ["", "## Links between targets (pooled coefficient, Driscoll-Kraay t)", ""]
        for r in L[L["chosen"]].itertuples():
            lines.append(f"- {r.target} <- {r.feature}: {getattr(r, 'pooled', float('nan')):.4f} "
                         f"(t {getattr(r, 'pooled_t_dk', float('nan')):.1f}); bank by bank in 6_links.csv")
    if len(R):
        out = R[R["outside_history"]]
        lines += ["", "## Ratios", "", f"{len(out)} of {len(R)} bank-scenario quarters outside the bank's own "
                  "historical range."]
    open(os.path.join(sd, "summary.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
