"""The panel run: drivers chosen on all banks, the panel models compared bank by bank, everything saved.

Outputs (outputs/<run>/panel/):
    P1_panel_screen.csv ... P3_search.csv   driver selection on the pooled panel
    1_leaderboard_overall.csv               every model: average error across banks, wins, size-weighted error
    2_leaderboard_by_bank.csv               every bank x model: average error over 1..h quarters
    3_backtest_by_horizon.csv               bank x model x horizon
    4_backtest_forecasts.csv                every forecast path, quarter by quarter, next to what happened
    5_coefficients.csv                      every bank's equation in every model (Driscoll-Kraay se for pooled)
    6_shrinkage.csv                         partial pooling: own, pooled, tau^2, weight on own, shrunk
    7_scenario_forecasts.csv                bank x model x scenario forecasts (change and level)
    8_fitted_values.csv                     in-sample one-step fit of every bank in every model
    10_evaluation_overall.csv               every model on the chosen metric (panel.evaluation), across the chosen
                                            banks; 11_evaluation_by_bank.csv, 12_evaluation_by_horizon.csv (also in oos/)
    9_coefficients_by_origin.csv            the coefficients at every backtest re-estimation
    oos/                                    out-of-sample test with re-selection at each origin (same tables)
    fitted/models.pkl                       every model estimated on all history; backtest_models.pkl,
                                            oos_models.pkl: the models at every origin
    panel_leaderboard.png, panel_forecasts.png, summary.md
"""
import copy
import json
import os
import pickle

import numpy as np
import pandas as pd

from ..features.transforms import all_features, candidate_forms, feature_matrix
from ..modelling.search import backtest_origins, refit_flags
from ..utils import target as tg
from .data import Panel
from .estimate import KINDS, PanelModel
from .select import select

PANEL_DEFAULTS = {"min_obs": 20, "dk_lags": 4, "n_search_candidates": 20,
                  "models": ["separate", "pooled", "partial", "topdown"], "driver_sets": ["panel", "core"],
                  "banks": [], "per_bank_full": False, "drop_exited": False, "exited_after_missing": 2,
                  "sample": "auto", "sample_tolerance": 0.02, "comparable_pct": 25, "workers": "auto"}


def settings(cfg):
    """panel: block with defaults."""
    p = copy.deepcopy(PANEL_DEFAULTS)
    p.update(cfg.get("panel") or {})
    cfg["panel"] = p
    return p


def _arr(FF, feats, pn):
    """Feature values for a model: (dates, k) for common drivers, (banks, dates, k) with bank-specific ones."""
    if not feats:
        return np.zeros((len(pn.dates), 0))
    from .select import BankFeatures
    return FF.array(feats) if isinstance(FF, BankFeatures) else FF[feats].values


def build_models(cfg, pn, sets):
    """The competing panel models for each driver set, plus the AR(1) benchmark."""
    p = cfg["panel"]
    out = [PanelModel("ar1", [], "AR(1) benchmark (each bank)", p["min_obs"], p["dk_lags"])]
    for label, feats in sets.items():
        if not feats:
            continue
        for kind in p["models"]:
            if kind == "topdown" and not tg.is_log():
                continue                          # shares of a total only make sense for balances
            out.append(PanelModel(kind, feats, f"{KINDS[kind]}: {label}", p["min_obs"], p["dk_lags"]))
    return out


def _hist(pn, b, o):
    """Last actual change of a bank and of the total at position o."""
    q = pn.dates[o]
    return {"bank": float(pn.ys[b].get(q, np.nan)), "total": float(pn.base.y.iloc[o])}


def _covid(cfg, dates, o, h):
    dq = [pd.Period(q, freq="Q") for q in cfg["sample"]["dummy_quarters"]]
    if not dq:
        return False
    w = dates[o + 1:o + h + 1]
    return bool(((w >= min(dq)) & (w <= max(dq))).any())


def forecast_rows(cfg, pn, m, X, o, H, store_rows, label="backtest"):
    """Forecast every active bank from position o and append error rows."""
    q0 = pn.dates[o]
    for b in pn.active(q0):
        if b not in m.coef or (m.kind == "topdown" and b not in m.share):
            continue
        yh = _hist(pn, b, o)
        if np.isnan(yh["bank"]):
            continue
        last = min(H, len(pn.dates) - o - 1)
        fc = m.forecast(pn, X, b, o, last, yh)
        y = pn.ys[b]
        L0 = float(pn.levels[b].loc[q0])
        cf = ca = 0.0
        for h in range(1, len(fc) + 1):
            q = pn.dates[o + h]
            a = y.get(q, np.nan)
            if pd.isna(a):
                break
            cf += fc[h - 1]
            ca += a
            e = float(tg.error(cf, ca))
            store_rows.append({"model": m.name, "kind": m.kind, "bank": b, "origin": str(q0), "date": str(q),
                               "horizon_q": h, "forecast_change": float(tg.growth_display(fc[h - 1])),
                               "actual_change": float(tg.growth_display(a)), "level_at_origin": L0,
                               "forecast_level": float(tg.level_after(L0, cf)),
                               "actual_level_organic": float(tg.level_after(L0, ca)),
                               "error_pct": e * 100, "covid_window": _covid(cfg, pn.dates, o, h), "test": label})


def summarise(E, H, hz, weights):
    """Errors -> by-horizon table, leaderboard by bank, overall leaderboard."""
    if E.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), f"avg_1_{max(hz)}_pct"
    Ex = E[~E["covid_window"]]
    by_h = (Ex.assign(abs_err=Ex["error_pct"].abs()).groupby(["bank", "model", "kind", "horizon_q"], sort=False)
            .agg(MAPE_exCOVID_pct=("abs_err", "mean"), bias_exCOVID_pct=("error_pct", "mean"),
                 n_origins=("error_pct", "size")).reset_index())
    rows = []
    for (b, m, k), d in by_h.groupby(["bank", "model", "kind"], sort=False):
        d = d.set_index("horizon_q")["MAPE_exCOVID_pct"]
        r = {"bank": b, "model": m, "kind": k}
        for h in hz:
            r[f"avg_1_{h}_pct"] = d.loc[:h].mean() if h in d.index else np.nan
        rows.append(r)
    LB = pd.DataFrame(rows)
    # rank on the longest horizon that has errors (few origins, or dummy windows, can leave the longest empty)
    have = [h for h in hz if LB[f"avg_1_{h}_pct"].notna().any()]
    col = f"avg_1_{max(have) if have else max(hz)}_pct"
    LB["rank_in_bank"] = LB.groupby("bank")[col].rank(method="min")
    ar = LB[LB["kind"] == "ar1"].set_index("bank")[col]
    LB["vs_AR_pct"] = (1 - LB[col] / LB["bank"].map(ar)) * 100
    g = LB.groupby(["model", "kind"], sort=False)
    O = g[[f"avg_1_{h}_pct" for h in hz]].mean()
    O["median_" + col] = g[col].median()
    O["size_weighted_" + col] = g.apply(lambda d: np.average(d[col], weights=d["bank"].map(weights).fillna(0) + 1e-12))
    O["banks_won"] = g["rank_in_bank"].apply(lambda r: int((r == 1).sum()))
    O["banks"] = g.size()
    O["banks_better_than_AR"] = g["vs_AR_pct"].apply(lambda v: int((v > 0).sum()))
    O = O.reset_index().sort_values(col).reset_index(drop=True)
    O.insert(0, "rank", range(1, len(O) + 1))
    O["ranked_on"] = col
    return by_h, LB.sort_values(["bank", col]).reset_index(drop=True), O, col


def backtest(cfg, pn, models, FF, log=print):
    """Rolling backtest of every panel model (same origins and re-estimation schedule as the single-series run)."""
    H = int(cfg["sample"]["horizon"])
    origins = backtest_origins(pn.base)
    flags = refit_flags(pn.base, origins)
    rows, coefs, at = [], [], {}
    save = (cfg.get("save") or {}).get("origin_models", True)
    for m in models:
        X = _arr(FF, m.features, pn)
        for o, fl in zip(origins, flags):
            if fl:
                try:
                    m.fit(pn, X, cfg["sample"]["estimation_start"], pn.dates[o])
                except Exception as ex:
                    log(f"  {m.name}: not estimated at {pn.dates[o]} ({ex})")
                    continue
                coefs += [{**r, "origin": str(pn.dates[o])} for r in m.coefficient_rows()]
                if save:
                    at.setdefault(str(pn.dates[o]), {})[m.name] = pickle.dumps(m)
            forecast_rows(cfg, pn, m, X, o, H, rows)
    return pd.DataFrame(rows), pd.DataFrame(coefs), at


def scenarios(cfg, pn, models, FF):
    """Forecast every scenario with every model estimated on all history."""
    out = []
    hist_end = pd.Period(cfg["sample"]["hist_end"], freq="Q")
    defl = cfg["features"].get("deflator")
    for name, mdf in pn.base.macro.items():
        if hist_end not in mdf.index:
            continue
        o = mdf.index.get_loc(hist_end)
        for m in models:
            feats = m.features
            X = feature_matrix(mdf, feats, defl).values if feats else np.zeros((len(mdf), 0))
            for b in pn.active(hist_end):
                if b not in m.coef or (m.kind == "topdown" and b not in m.share):
                    continue
                yh = {"bank": float(pn.ys[b].loc[hist_end]), "total": float(pn.base.y.loc[hist_end])}
                fc = m.forecast(pn, X, b, o, len(mdf) - o - 1, yh)
                lv = tg.path(float(pn.levels[b].loc[hist_end]), fc)
                out += [{"model": m.name, "kind": m.kind, "bank": b, "scenario": name, "date": str(mdf.index[o + h]),
                         "change_forecast": float(f), "level_forecast": float(l)}
                        for h, (f, l) in enumerate(zip(fc, lv), 1)]
    return pd.DataFrame(out)


def fitted_values(cfg, pn, models, FF):
    """In-sample one-step fit of every bank in every model estimated on all history."""
    rows = []
    a = pn.pos(cfg["sample"]["estimation_start"])
    z = pn.pos(cfg["sample"]["hist_end"])
    for m in models:
        X = _arr(FF, m.features, pn)
        for b in m.coef:
            y = pn.ys[b]
            for t in range(max(a, 1), z + 1):
                q = pn.dates[t]
                if q not in y.index or pd.isna(y.loc[q]) or pd.isna(y.get(q - 1, np.nan)):
                    continue
                f = m.fitted(pn, X, b, t)
                rows.append({"model": m.name, "bank": b, "date": str(q), "actual": float(y.loc[q]), "fitted": f,
                             "residual": float(y.loc[q]) - f if pd.notna(f) else np.nan})
    return pd.DataFrame(rows)


def run_oos(cfg, pn, F, core, log=print):
    """Out-of-sample test: at every origin the drivers are re-selected on all banks with data up to the origin,
    every model is re-estimated, then forecasts 1..H quarters for every bank."""
    o_ = cfg["oos"]
    H = int(o_.get("horizon") or cfg["sample"]["horizon"])
    first = pd.Period(o_["first_origin"], freq="Q")
    end = pn.pos(cfg["sample"]["hist_end"])
    origins = [i for i, q in enumerate(pn.dates) if q >= first and i < end][::int(o_.get("step", 4))]
    workers = _workers(cfg, len(origins))
    if workers > 1:
        from joblib import Parallel, delayed
        log(f"  {len(origins)} origins on {workers} processes ...")
        parts = Parallel(n_jobs=workers, backend="loky")(delayed(_oos_origin)(cfg, pn, F, core, o, H, None, tg.form())
                                                         for o in origins)
    else:
        parts = [_oos_origin(cfg, pn, F, core, o, H, log) for o in origins]
    rows, picks, at = [], [], {}
    for r, pick, a, msgs in parts:                   # in origin order, whatever order they finished in
        rows += r
        picks.append(pick)
        at.update(a)
        if workers > 1:
            for m in msgs:
                log(m)
    return pd.DataFrame(rows), pd.DataFrame(picks), at


def _workers(cfg, n):
    """panel.workers: 'auto' (the CPU cores, at most one per origin), or a number (1 = no parallel processes)."""
    w = (cfg.get("panel") or {}).get("workers", "auto")
    if str(w).lower() == "auto":
        w = max(1, min(os.cpu_count() or 1, n, 8))
    return max(1, int(w))


def _oos_origin(cfg, pn, F, core, o, H, log=None, form=None):
    """One origin of the out-of-sample test: re-select, re-estimate, forecast. Returns (rows, pick, models, log).
    In a separate process the target form (a module setting) is set again from `form`."""
    if form is not None:
        tg.set_form(form)
    msgs = []
    say = log or msgs.append
    lab = pn.dates[o]
    defl = cfg["features"].get("deflator")
    sel = select(cfg, pn, F, cfg["sample"]["estimation_start"], lab, lambda *_: None)
    sets = {"panel drivers": sel["features"], "literature core": core}
    models = build_models(cfg, pn, sets)
    pick = {"origin": str(lab), "panel drivers": ", ".join(sel["features"]) or "(none)"}
    cols = list(dict.fromkeys(sel["features"] + core))
    FF = F.reindex(columns=[c for c in cols if c in F.columns])
    if any(c not in F.columns for c in cols):
        FF = pd.concat([FF, feature_matrix(pn.base.hist, [c for c in cols if c not in F.columns], defl)], axis=1)
    rows, at = [], {}
    for m in models:
        X = _arr(FF, m.features, pn)
        try:
            m.fit(pn, X, cfg["sample"]["estimation_start"], lab)
        except Exception as ex:
            say(f"  oos {lab}: {m.name} not estimated ({ex})")
            continue
        at.setdefault(str(lab), {})[m.name] = pickle.dumps(m)
        forecast_rows(cfg, pn, m, X, o, H, rows, "oos")
    say(f"  oos origin {lab}: panel drivers {sel['features']}")
    return rows, pick, at, msgs


def _charts(out, O, scen, pn, best_by_bank, col):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(O) + 1.2))
    d = O.iloc[::-1]
    colors = {"ar1": "#9a9a9a", "separate": "#3b5ba5", "pooled": "#0f8a8a", "partial": "#c0504d", "topdown": "#8a6d3b"}
    ax.barh(d["model"], d[col], color=[colors.get(k, "#555") for k in d["kind"]])
    ax.set_xlabel(f"Average error over 1-{col.split('_')[2]} quarters, mean across banks ({tg.units()})")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "panel_leaderboard.png"), dpi=150)
    plt.close(fig)
    if scen.empty:
        return
    banks = list(best_by_bank)
    n = len(banks)
    cols_ = min(3, n)
    rows_ = int(np.ceil(n / cols_))
    fig, axs = plt.subplots(rows_, cols_, figsize=(4.2 * cols_, 2.8 * rows_), squeeze=False)
    for ax, b in zip(axs.ravel(), banks):
        h = pn.levels[b].iloc[-24:]
        ax.plot(h.index.to_timestamp(), h.values, color="#222", lw=1.6)
        d = scen[(scen["bank"] == b) & (scen["model"] == best_by_bank[b])]
        for sc, g in d.groupby("scenario", sort=False):
            ax.plot(pd.PeriodIndex(g["date"], freq="Q").to_timestamp(), g["level_forecast"].values, lw=1.1, label=sc)
        ax.set_title(f"{b}: {best_by_bank[b]}", fontsize=8, loc="left")
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axs.ravel()[n:]:
        ax.axis("off")
    axs.ravel()[0].legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "panel_forecasts.png"), dpi=150)
    plt.close(fig)


def run(cfg, out_dir, oos=True, log=print, force=False):
    """The whole panel analysis.

    Args:
        cfg: configuration with data.bank.panel_column. out_dir: outputs/<run>. oos: also the out-of-sample test.
        force: re-estimate even when nothing changed since the last run.

    Returns:
        dict with the leaderboards and the panel folder (and reused=True when the earlier results were kept).
    """
    from . import reuse
    from .data import bank_events, load_long
    p = settings(cfg)
    out = os.path.join(out_dir, "panel")
    os.makedirs(out, exist_ok=True)
    lv0 = load_long(copy.deepcopy(cfg))
    comps = reuse.panel_components(copy.deepcopy(cfg), lv0, {b: bank_events(cfg, b) for b in lv0})
    fp_path = reuse.panel_path(out_dir)
    ok, why = reuse.compare(comps, fp_path, need_oos=oos)
    if ok and not force and os.path.exists(os.path.join(out, "1_leaderboard_overall.csv")):
        log(f"Panel: nothing changed ({why}) - reusing the earlier results in {out} "
            f"(use --force to re-estimate)")
        res = _reused(out, oos)
        _re_evaluate(cfg, out, res, log)
        return res
    log(f"Panel: estimating ({'forced' if force else why}) ...")
    log("Panel: loading the banks ...")
    pn = Panel(cfg, out, log)
    H = int(cfg["sample"]["horizon"])
    hz = cfg["sample"]["report_horizons"]
    col = f"avg_1_{max(hz)}_pct"
    log("Panel: candidate features ...")
    forms = candidate_forms(cfg, pn.base.hist, pn.base.meta)
    F = all_features(cfg, pn.base, forms)
    log(f"  {F.shape[1]} candidate features")
    log("Panel: selecting drivers on all banks together ...")
    sel = select(cfg, pn, F, cfg["sample"]["estimation_start"], cfg["sample"]["hist_end"], log)
    log(f"  panel drivers: {sel['features'] or '(none passed the rules)'}")
    # ---- which credit unions to estimate on (panel.sample: auto compares all / weighted / comparable / active)
    from . import sample as smp
    s_name, s_banks, s_w, s_tab, s_info = smp.choose(cfg, pn, F, sel["features"], out, log)
    s_tab.round(4).to_csv(os.path.join(out, "0_sample_choice.csv"), index=False)
    s_info.round(6).to_csv(os.path.join(out, "0_banks.csv"), index=False)
    active_banks = list(s_info.loc[s_info["active"], "bank"])
    log(f"Panel: estimation sample: {smp.LABELS[s_name]} ({len(s_banks)} of {len(pn.banks)} banks; "
        f"{s_tab.loc[s_tab['chosen'], 'reason'].iloc[0]})")
    if s_name != "all":
        if set(s_banks) != set(pn.banks):
            pn = Panel(smp.restrict(cfg, s_banks), out, log)
        pn.wrow = smp.weights(pn, s_info) if s_w else None
        sel = select(cfg, pn, F, cfg["sample"]["estimation_start"], cfg["sample"]["hist_end"], log)
        log(f"  panel drivers on the chosen sample: {sel['features'] or '(none passed the rules)'}")
    core = []
    if "core" in p["driver_sets"]:
        from ..models import zoo
        lit = zoo.literature_models(cfg, pn.base, lambda *_: None)
        spec = next((m for m in lit if m.get("core")), None)
        core = list(spec["features"]) if spec else []
        log(f"  literature core: {core or '(no literature workbook / no core model)'}")
    sets = {}
    if "panel" in p["driver_sets"]:
        sets["panel drivers"] = sel["features"]
    sets["literature core"] = core
    cols = list(dict.fromkeys([f for v in sets.values() for f in v]))
    defl = cfg["features"].get("deflator")
    FF = F.reindex(columns=[c for c in cols if c in F.columns])
    missing = [c for c in cols if c not in F.columns]
    if missing:
        FF = pd.concat([FF, feature_matrix(pn.base.hist, missing, defl)], axis=1)
    models = build_models(cfg, pn, sets)
    log(f"Panel: backtest of {len(models)} models on {len(pn.banks)} banks ...")
    E, coef_paths, at = backtest(cfg, pn, models, FF, log)
    weights = {b: float(pn.levels[b].iloc[-1]) for b in pn.banks}
    by_h, LB, O, col = summarise(E, H, hz, weights)
    log("Panel: estimating on all history, scenarios, fitted values ...")
    fitted = {}
    for m in models:
        X = _arr(FF, m.features, pn)
        m.fit(pn, X, cfg["sample"]["estimation_start"], cfg["sample"]["hist_end"])
        fitted[m.name] = m
    coefs = pd.DataFrame([r for m in models for r in m.coefficient_rows()])
    shrink = pd.concat([m.weights.assign(model=m.name) for m in models if m.weights is not None], ignore_index=True) \
        if any(m.weights is not None for m in models) else pd.DataFrame()
    scen = scenarios(cfg, pn, models, FF)
    fv = fitted_values(cfg, pn, models, FF)
    tables = dict(sel["tables"])
    tables.update({"1_leaderboard_overall": O.round(4), "2_leaderboard_by_bank": LB.round(4),
                   "3_backtest_by_horizon": by_h.round(4), "4_backtest_forecasts": E.round(6),
                   "5_coefficients": coefs, "6_shrinkage": shrink, "7_scenario_forecasts": scen,
                   "8_fitted_values": fv, "9_coefficients_by_origin": coef_paths})
    for k, v in tables.items():
        if isinstance(v, pd.DataFrame):
            v.to_csv(os.path.join(out, f"{k}.csv"), index=False)
    fdir = os.path.join(out, "fitted")
    os.makedirs(fdir, exist_ok=True)
    with open(os.path.join(fdir, "models.pkl"), "wb") as f:
        pickle.dump({k: pickle.dumps(m) for k, m in fitted.items()}, f)
    if at:
        with open(os.path.join(fdir, "backtest_models.pkl"), "wb") as f:
            pickle.dump(at, f)
    from . import compare
    ev_cfg = compare.settings(cfg)
    ev = compare.evaluate(E, **ev_cfg, active=active_banks) if len(E) else None
    if ev is not None:
        compare.write(ev, out)
    best_by_bank = (ev["by_bank"].groupby("bank").head(1).set_index("bank")["model"].to_dict() if ev is not None
                    else LB.sort_values(col).groupby("bank").head(1).set_index("bank")["model"].to_dict() if len(LB) else {})
    res = {"overall": O, "by_bank": LB, "best_by_bank": best_by_bank, "panel_drivers": sel["features"],
           "core": core, "folder": out}
    if oos:
        log("Panel: out-of-sample test (drivers re-selected at every origin) ...")
        Eo, picks, at_o = run_oos(cfg, pn, F, core, log)
        bo, LBo, Oo, col_o = summarise(Eo, H, hz, weights)
        od = os.path.join(out, "oos")
        os.makedirs(od, exist_ok=True)
        for k, v in {"1_leaderboard_overall": Oo.round(4), "2_leaderboard_by_bank": LBo.round(4),
                     "3_by_horizon": bo.round(4), "4_forecasts": Eo.round(6), "5_drivers_by_origin": picks}.items():
            v.to_csv(os.path.join(od, f"{k}.csv"), index=False)
        with open(os.path.join(fdir, "oos_models.pkl"), "wb") as f:
            pickle.dump(at_o, f)
        res.update(oos_overall=Oo, oos_by_bank=LBo)
        if len(Eo):
            evo = compare.evaluate(Eo, **ev_cfg, active=active_banks)
            compare.write(evo, od)
            res["oos_evaluation"] = evo["overall"]
            res["best_by_bank"] = evo["by_bank"].groupby("bank").head(1).set_index("bank")["model"].to_dict()
            col = col_o
    if ev is not None:
        res["evaluation"] = ev["overall"]
    _charts(out, res.get("oos_overall", O) if len(res.get("oos_overall", [])) else O, scen, pn, res["best_by_bank"], col)
    res["sample"] = {"name": s_name, "table": s_tab, "active": active_banks}
    _summary(out, cfg, pn, res, col)
    with open(os.path.join(fdir, "manifest.json"), "w", encoding="utf-8") as f:
        from .data import DROPPED
        json.dump({"banks": pn.banks, "active_banks": active_banks, "sample": s_name, "sample_weighted": bool(s_w),
                   "left_out_exited": dict(DROPPED), "target_form": tg.form(),
                   "panel_drivers": sel["features"], "core": core,
                   "models": [m.name for m in models], "best_by_bank": res["best_by_bank"],
                   "backtest_origins": sorted(at), "oos": bool(oos)}, f, indent=1)
    reuse.save(fp_path, comps, oos, {"banks": pn.banks})
    log(f"Panel results in {out}")
    return res


def _re_evaluate(cfg, out, res, log=print):
    """The evaluation is cheap and depends only on the saved errors: redo it with the current settings."""
    from . import compare
    ev_cfg = compare.settings(cfg)
    act = None
    try:
        act = json.load(open(os.path.join(out, "fitted", "manifest.json"), encoding="utf-8")).get("active_banks")
    except Exception:
        pass
    for sub, key in (("", "evaluation"), ("oos", "oos_evaluation")):
        try:
            E = compare.load_errors(out, oos=bool(sub))
        except FileNotFoundError:
            continue
        ev = compare.evaluate(E, **ev_cfg, active=act)
        compare.write(ev, os.path.join(out, sub) if sub else out)
        res[key] = ev["overall"]
        res["best_by_bank"] = ev["by_bank"].groupby("bank").head(1).set_index("bank")["model"].to_dict()
    sm = os.path.join(out, "summary.md")
    if os.path.exists(sm) and ("evaluation" in res or "oos_evaluation" in res):   # refresh its ranking block
        txt = open(sm, encoding="utf-8").read().split("\n")
        i = next((k for k, x in enumerate(txt) if x.startswith("## Ranked on")), None)
        j = next((k for k, x in enumerate(txt) if x.startswith("## Best model for each bank")), len(txt))
        basis = "out of sample" if "oos_evaluation" in res else "backtest"
        block = _eval_lines(cfg, res.get("oos_evaluation", res.get("evaluation")), basis)[1:] + [""]
        best = ["## Best model for each bank", ""] + [f"- {b}: {m}" for b, m in res["best_by_bank"].items()]
        head = txt[:i - 1] if i is not None else txt[:j]
        open(sm, "w", encoding="utf-8").write("\n".join(head + [""] + block + best) + "\n")
    log(f"Panel: evaluation redone on the saved errors ({compare.describe(ev_cfg['metric'], ev_cfg['tail_share'], ev_cfg['penalty'])})")


def _reused(out, oos):
    """The results of an earlier, still valid, panel run."""
    res = {"folder": out, "reused": True}
    for key, f in (("overall", "1_leaderboard_overall.csv"), ("by_bank", "2_leaderboard_by_bank.csv"),
                   ("oos_overall", os.path.join("oos", "1_leaderboard_overall.csv")),
                   ("oos_by_bank", os.path.join("oos", "2_leaderboard_by_bank.csv"))):
        fp = os.path.join(out, f)
        if os.path.exists(fp):
            res[key] = pd.read_csv(fp)
    mf = os.path.join(out, "fitted", "manifest.json")
    if os.path.exists(mf):
        with open(mf, encoding="utf-8") as f:
            m = json.load(f)
        res.update(best_by_bank=m.get("best_by_bank", {}), panel_drivers=m.get("panel_drivers", []),
                   core=m.get("core", []))
    return res


def _sample_lines(smp):
    """The estimation-sample block of summary.md."""
    if not smp:
        return []
    from .sample import LABELS
    T = smp["table"]
    L = [f"Estimation sample: **{LABELS[smp['name']]}**. {T.loc[T['chosen'], 'reason'].iloc[0]}.", ""]
    if len(T) > 1:
        L += ["| Sample | Banks | Exited included | Weighted | Error for the active banks |", "|---|---|---|---|---|"]
        for r in T.itertuples():
            L.append(f"| {r.label}{' (chosen)' if r.chosen else ''} | {r.banks} | {getattr(r, 'exited_included', '')} | "
                     f"{'yes' if getattr(r, 'weighted', False) else 'no'} | {getattr(r, 'score_active', float('nan')):.3f} |")
        L.append("")
    L += [f"Active banks (reporting at the end of history): {len(smp['active'])}.", ""]
    return L


def _eval_lines(cfg, ev, basis):
    """The 'Ranked on ...' block of summary.md."""
    if ev is None or not len(ev):
        return []
    from . import compare
    e = compare.settings(cfg)
    L = ["", f"## Ranked on the {compare.describe(e['metric'], e['tail_share'], e['penalty'])} ({basis}; "
         f"{'every bank' if e['banks'] in (None, 'all', []) else ', '.join(map(str, e['banks']))}, "
         f"{e['weighting']} weights)", "",
         "| Rank | Model | Score | MAPE | RMSE | Worst-tail avg | Largest | Banks won |", "|---|---|---|---|---|---|---|---|"]
    for r in ev.itertuples():
        L.append(f"| {r.rank} | {r.model} | {r.score:.3f} | {r.mape:.3f} | {r.rmse:.3f} | {r.cvar:.3f} | "
                 f"{r.worst:.3f} | {r.banks_won}/{r.banks} |")
    return L + ["", "Other metrics, other banks or one bank: `python tools/panel_compare.py --run <outputs/run> "
                    "--metric rmse --banks ... --bank ...` (no re-estimation), or ⑤ in the app."]


def _summary(out, cfg, pn, res, col):
    basis = "out of sample" if "oos_overall" in res else "backtest"
    O = res.get("oos_overall", res["overall"])
    L = [f"# Panel run: {cfg['run_name']}", "",
         f"{len(pn.banks)} banks: {', '.join(pn.banks)}. Target form: {tg.describe()}.", "",
         *(["Left out because they stopped reporting (panel.drop_exited): "
            + ", ".join(f"{k} (last {v})" for k, v in __import__('src.panel.data', fromlist=['DROPPED']).DROPPED.items()), ""]
           if (cfg.get("panel") or {}).get("drop_exited") else []),
         *_sample_lines(res.get("sample")),
         f"Panel drivers (chosen on the estimation sample): {', '.join(res['panel_drivers']) or 'none'}",
         f"Literature core: {', '.join(res['core']) or 'none'}", "",
         f"## Models ({basis}; average error over 1-{col.split('_')[2]} quarters, mean across banks)", "",
         "| Rank | Model | Mean | Banks won | Better than AR(1) |", "|---|---|---|---|---|"]
    for r in O.itertuples():
        L.append(f"| {r.rank} | {r.model} | {getattr(r, col):.3f} | {r.banks_won}/{r.banks} | "
                 f"{r.banks_better_than_AR}/{r.banks} |")
    L += _eval_lines(cfg, res.get("oos_evaluation", res.get("evaluation")), basis)
    L += ["", "## Best model for each bank", ""] + [f"- {b}: {m}" for b, m in res["best_by_bank"].items()]
    open(os.path.join(out, "summary.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
