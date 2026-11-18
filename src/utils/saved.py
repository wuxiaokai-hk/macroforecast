"""Keep the results of every run so they can be opened later without running anything again.

outputs/<run>/fitted/models.pkl   every model as estimated on all history (load_models() reads it back)
outputs/<run>/fitted/coefficients.csv, fit_statistics.csv, fitted_values.csv, panels/, data/target.csv
                                  every coefficient with its standard error, the in-sample fit and the data used
outputs/<run>/fitted/backtest_models/<model>.pkl
                                  the model at every backtest re-estimation (load_origin_models())
outputs/<run>/oos/fitted_models/<origin>.pkl
                                  the re-selected models at every out-of-sample origin
outputs/<run>/fitted/formula_models.json
                                  the regression and same-growth models as plain numbers (coefficients, drivers,
                                  long-run relation): readable on any computer and with any package versions
outputs/<run>/saved_runs/<date_time>_<script>/
                                  a complete copy of the results of one run: tables, charts, formula, fitted
                                  models, the config and the literature workbook used. The app lists them in
                                  ⑤ Results ("Results to show"); delete a folder to forget a run.

    from src.utils.saved import load_models
    models = load_models("outputs/ontario_loans_example")        # {model name: fitted model}
    m = models["Screen first"]; m.params                          # coefficients, ready to forecast
"""
import datetime as _dt
import json
import os
import pickle
import shutil

KEEP = ["common", "screen_first", "en_first", "comparison", "oos", "fitted", "formula", "summary.md",
        "run_manifest.json"]


def save_fitted(out_dir, fitted, log=print):
    """Write the fitted models of a run to fitted/models.pkl.

    Args:
        out_dir: the run's output folder. fitted: dict model name -> pickle bytes (None = not stored).
        log: message function.

    Returns:
        path of the file, or None if nothing could be stored.
    """
    keep = {k: v for k, v in fitted.items() if v is not None}
    if not keep:
        return None
    d = os.path.join(out_dir, "fitted")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "models.pkl")
    with open(path, "wb") as f:
        pickle.dump(keep, f)
    plain = {}
    for k, b in keep.items():
        try:
            m = pickle.loads(b)
        except Exception:
            continue
        pm = portable(m)
        if pm:
            plain[k] = pm
    with open(os.path.join(d, "formula_models.json"), "w", encoding="utf-8") as f:
        json.dump(plain, f, indent=1)
    skipped = sorted(k for k, v in fitted.items() if v is None)
    with open(os.path.join(d, "README.txt"), "w", encoding="utf-8") as f:
        f.write("models.pkl            every model as estimated on all history, keyed by model name\n"
                "formula_models.json   the regressions as plain numbers (any computer, any package version)\n"
                "coefficients.csv      every coefficient with standard error, t and p value\n"
                "fit_statistics.csv    observations, R2, AIC, BIC, residual std. dev. per model\n"
                "fitted_values.csv     in-sample one-step fit of every model: actual, fitted, residual, levels\n"
                "panels/               the data each model was estimated on\n"
                "backtest_models/      each model as re-estimated at every backtest origin {origin: model}\n"
                "data/target.csv       the target level and its organic change\n"
                "manifest.json         what was saved\n\n"
                "Read them with:\n"
                "  from src.utils.saved import load_models, load_origin_models\n"
                "  models = load_models('<run folder>')                        # {name: model}\n"
                "  by_origin = load_origin_models('<run folder>', '<model name>') # {origin: model}\n"
                "Each model has the usual interface: panel(ctx, mdf), forecast(P, origin, H), params (regressions).\n"
                "Out-of-sample models are in oos/fitted_models/<origin>.pkl (load_origin_models(file)).\n")
        if skipped:
            f.write("\nNot stored (cannot be pickled, or pretrained and downloaded again when needed):\n  "
                    + "\n  ".join(skipped) + "\n")
    log(f"  fitted models: {len(keep)} saved to {os.path.relpath(path)}"
        + (f" ({len(skipped)} not stored: {', '.join(skipped)})" if skipped else ""))
    return path


def _slug(name):
    """A model name as a safe file name."""
    import re
    return re.sub(r"[^A-Za-z0-9._+-]+", "_", str(name)).strip("_")[:80] or "model"


def save_all(out_dir, ev, ctx, log=print):
    """Everything estimated in a run, next to models.pkl (fitted/):

        models.pkl                    every model estimated on all history (load_models())
        formula_models.json           regressions as plain numbers (any computer, any package version)
        coefficients.csv              every coefficient: estimate, standard error, t, p value (regressions),
                                      or weights / parameters for the other model types
        fit_statistics.csv            observations, R², adjusted R², AIC, BIC, residual std. dev. per model
        fitted_values.csv             in-sample one-step fit of every model: actual, fitted, residual, levels
        panels/<model>.csv            the data each model was estimated on (history and baseline scenario)
        backtest_models/<model>.pkl   the model as re-estimated at every backtest origin {origin: model}
                                      (load_origin_models())
        data/target.csv               the target: level, organic change in the model's form, cumulated level
        manifest.json                 what was saved, the target form and the file of each model

    Args:
        out_dir: the run's output folder. ev: model name -> evaluate() output. ctx: Context. log: message function.

    Returns:
        the fitted/ folder.
    """
    import pandas as pd
    from . import target as tg
    d = os.path.join(out_dir, "fitted")
    os.makedirs(d, exist_ok=True)
    save_fitted(out_dir, {k: v.get("fitted") for k, v in ev.items()}, log)
    man = {"target_form": tg.form(), "target_units": tg.units(), "models": {}}

    def cat(key):
        parts = [v.get(key) for v in ev.values() if isinstance(v.get(key), pd.DataFrame) and len(v.get(key))]
        return pd.concat(parts, ignore_index=True) if parts else None
    for key, fname in (("coefficients", "coefficients.csv"), ("fitted_values", "fitted_values.csv")):
        T = cat(key)
        if T is not None:
            T.to_csv(os.path.join(d, fname), index=False)
    st = [v.get("fit_statistics") for v in ev.values() if v.get("fit_statistics")]
    if st:
        pd.DataFrame(st).to_csv(os.path.join(d, "fit_statistics.csv"), index=False)
    pdir, bdir = os.path.join(d, "panels"), os.path.join(d, "backtest_models")
    n_orig = 0
    for k, v in ev.items():
        info = {"in_models_pkl": v.get("fitted") is not None}
        P = v.get("panel")
        if isinstance(P, pd.DataFrame):
            os.makedirs(pdir, exist_ok=True)
            f = os.path.join(pdir, _slug(k) + ".csv")
            P.rename_axis("date").reset_index().assign(date=lambda x: x["date"].astype(str)).to_csv(f, index=False)
            info["panel"] = os.path.relpath(f, d)
        om = {o: b for o, b in (v.get("origin_models") or {}).items() if b is not None}
        if om:
            os.makedirs(bdir, exist_ok=True)
            f = os.path.join(bdir, _slug(k) + ".pkl")
            with open(f, "wb") as fh:
                pickle.dump(om, fh)
            info["backtest_models"] = os.path.relpath(f, d)
            info["backtest_origins"] = sorted(om)
            n_orig += len(om)
        man["models"][k] = info
    dd = os.path.join(d, "data")
    os.makedirs(dd, exist_ok=True)
    pd.DataFrame({"date": [str(q) for q in ctx.level.index], "level": ctx.level.values,
                  "organic_change": ctx.y.reindex(ctx.level.index).values,
                  "organic_state": ctx.logL.reindex(ctx.level.index).values}).to_csv(os.path.join(dd, "target.csv"),
                                                                                     index=False)
    with open(os.path.join(d, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(man, f, indent=1)
    log(f"  fitted results: coefficients, fit statistics, fitted values, {len(man['models'])} panels and "
        f"{n_orig} backtest re-estimations saved to {os.path.relpath(d) if not os.path.relpath(d).startswith('..') else d}")
    return d


def save_origin_models(folder, store, log=print):
    """Out-of-sample models: {origin: {model name: pickle bytes}} -> folder/<origin>.pkl, one file per origin."""
    if not store:
        return
    os.makedirs(folder, exist_ok=True)
    n = 0
    for origin, mods in store.items():
        keep = {k: b for k, b in mods.items() if b is not None}
        if keep:
            with open(os.path.join(folder, f"{origin}.pkl"), "wb") as f:
                pickle.dump(keep, f)
            n += len(keep)
    log(f"  out-of-sample fitted models: {n} (model, origin) pairs saved to {folder}")


def load_origin_models(path_or_run, model=None):
    """Models re-estimated at each origin.

    Args:
        path_or_run: a backtest_models/<model>.pkl file, an oos/fitted_models/<origin>.pkl file, or a run folder
            (then `model` names the model, and its backtest file is read).

    Returns:
        dict origin (or model name) -> fitted model.
    """
    p = path_or_run
    if os.path.isdir(p):
        p = os.path.join(p, "fitted", "backtest_models", _slug(model) + ".pkl")
    with open(p, "rb") as f:
        raw = pickle.load(f)
    out = {}
    for k, b in raw.items():
        try:
            out[k] = pickle.loads(b)
        except Exception:
            continue
    return out


def portable(model):
    """A regression, error-correction or same-growth model as plain numbers (None for other model types).

    Args:
        model: fitted model.

    Returns:
        dict (JSON-ready) or None.
    """
    kind = type(model).__name__
    if kind not in ("ARDL", "ECM", "SameGrowth", "AutoARDL", "LevelsARDL") or getattr(model, "params", None) is None:
        return None
    if kind == "LevelsARDL":
        return {"class": kind, "name": model.name, "family": model.family, "long_run": model.long_run,
                "deflator": model.deflator, "p": int(model.p), "q": {k: int(v) for k, v in model.q.items()},
                "max_p": model.max_p, "max_q": model.max_q, "ic": model.ic,
                "params": {k: float(v) for k, v in model.params.items()},
                "bounds": {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in (model.bounds or {}).items()}}
    d = {"class": kind, "name": model.name, "family": model.family, "features": list(model.features),
         "params": {k: float(v) for k, v in model.params.items()}, "ar": bool(getattr(model, "ar", True)),
         "signs": dict(getattr(model, "signs", {}) or {})}
    if kind == "ECM":
        d.update(long_run=model.long_run, deflator=model.deflator,
                 beta=[float(b) for b in (model.beta if model.beta is not None else [])])
    if kind == "SameGrowth":
        d["gap"] = bool(model.gap)
    if kind == "AutoARDL":
        d.update(bases=model.bases, p=int(model.p), q={k: int(v) for k, v in model.q.items()}, max_p=model.max_p,
                 max_q=model.max_q, ic=model.ic, base_signs=model.base_signs)
    return d


def rebuild(d):
    """The model back from portable(): ready to forecast and to write its formula (no standard errors).

    Args:
        d: dict from portable().

    Returns:
        model.
    """
    import numpy as np
    import pandas as pd
    from ..modelling.ardl import ARDL
    from ..models.ecm import ECM
    from ..models.simple import SameGrowth
    from ..models.autoardl import AutoARDL, LevelsARDL
    if d["class"] == "LevelsARDL":
        m = LevelsARDL(d["name"], d["long_run"], d["deflator"], d["max_p"], d["max_q"], d["ic"], family=d["family"])
        m.p, m.q, m.params, m.bounds, m.frozen = d["p"], d["q"], pd.Series(d["params"]), d.get("bounds"), True
        return m
    if d["class"] == "AutoARDL":
        m = AutoARDL(d["name"], [f"{b}|L0" for b in d["bases"]], d["max_p"], d["max_q"], d["ic"], family=d["family"],
                     signs=d.get("base_signs"))
        m.p, m.q, m.features, m.frozen = d["p"], d["q"], d["features"], True
        m.params = pd.Series(d["params"])
        m.signs = d.get("signs") or {}
        return m
    if d["class"] == "SameGrowth":
        m = SameGrowth(d["name"], d["features"][0], gap=d.get("gap", False), family=d["family"])
    elif d["class"] == "ECM":
        m = ECM(d["name"], d["features"], d["long_run"], d["deflator"], ar=d.get("ar", True), family=d["family"])
        m.beta = np.array(d["beta"])
    else:
        m = ARDL(d["name"], d["features"], ar=d.get("ar", True), family=d["family"])
    m.params = pd.Series(d["params"])
    m.signs = d.get("signs") or {}
    return m


def load_models(run_dir):
    """The fitted models of a run (or of a saved run folder).

    Models that cannot be unpickled on this computer (other package versions) are rebuilt from
    fitted/formula_models.json when they are regressions, error-correction or same-growth models.

    Args:
        run_dir: outputs/<run> or one of its saved_runs/<...> folders.

    Returns:
        dict model name -> fitted model ({} if the run has no saved models).
    """
    out = {}
    path = os.path.join(run_dir, "fitted", "models.pkl")
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                raw = pickle.load(f)
        except Exception:
            raw = {}
        for k, b in raw.items():
            try:
                out[k] = pickle.loads(b)
            except Exception:                  # e.g. other pandas version, or a deep-learning package missing
                continue
    jp = os.path.join(run_dir, "fitted", "formula_models.json")
    if os.path.exists(jp):
        for k, d in json.load(open(jp, encoding="utf-8")).items():
            if k not in out:
                try:
                    out[k] = rebuild(d)
                except Exception:
                    continue
    return out


def snapshot(cfg, script, log=print):
    """Copy the run's current results into saved_runs/<date_time>_<script>/ and keep the newest save.keep_runs.

    Args:
        cfg: configuration (out_dir, config_file, literature.workbook, save.keep_runs). script: "pipeline" or "oos".
        log: message function.

    Returns:
        the snapshot folder, or None when saving is switched off.
    """
    keep_n = int((cfg.get("save") or {}).get("keep_runs", 20) or 0)
    if keep_n <= 0:
        return None
    out = cfg["out_dir"]
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = os.path.join(out, "saved_runs", f"{stamp}_{script}")
    os.makedirs(dst, exist_ok=True)
    for k in KEEP:
        src = os.path.join(out, k)
        if os.path.isdir(src):
            shutil.copytree(src, os.path.join(dst, k), dirs_exist_ok=True)
        elif os.path.isfile(src):
            shutil.copy2(src, dst)
    inputs = os.path.join(dst, "inputs")
    os.makedirs(inputs, exist_ok=True)
    for f in (cfg.get("config_file"), (cfg.get("literature") or {}).get("workbook")):
        if f and os.path.exists(f):
            shutil.copy2(f, inputs)
    info = {"saved": stamp, "script": script, "run_name": cfg["run_name"],
            "config_file": os.path.basename(cfg.get("config_file") or "")}
    best = _best(dst)
    if best:
        info.update(best)
    with open(os.path.join(dst, "saved_run.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=1)
    runs = list_snapshots(out)
    for old in runs[keep_n:]:
        shutil.rmtree(old["path"], ignore_errors=True)
    log(f"Saved a copy of these results: {os.path.relpath(dst)} (open it later in the app, ⑤ Results)")
    return dst


def _best(folder):
    """The most accurate model of a saved run (out of sample if available), for the run list.

    Args:
        folder: a saved run folder.

    Returns:
        dict with best_model, best_error_pct and basis, or {}.
    """
    import pandas as pd
    for sub, f, basis in (("oos", "1_oos_summary.csv", "out of sample"), ("comparison", "1_model_comparison.csv", "backtest")):
        p = os.path.join(folder, sub, f)
        if os.path.exists(p):
            d = pd.read_csv(p)
            d = d[d["family"] != "Look-ahead check"]
            col = sorted([c for c in d.columns if c.startswith("avg_1_") and c.endswith("_pct")],
                         key=lambda c: int(c.split("_")[2]))[-1:]
            if col and len(d):
                r = d.sort_values(col[0]).iloc[0]
                return {"best_model": r["model"], "best_error_pct": round(float(r[col[0]]), 3), "basis": basis,
                        "horizon": col[0]}
    return {}


def list_snapshots(out_dir):
    """Saved runs of one run name, newest first.

    Args:
        out_dir: outputs/<run>.

    Returns:
        list of dicts: path, name, saved, script, best_model, best_error_pct, basis.
    """
    root = os.path.join(out_dir, "saved_runs")
    if not os.path.isdir(root):
        return []
    rows = []
    for n in sorted(os.listdir(root), reverse=True):
        p = os.path.join(root, n)
        if not os.path.isdir(p):
            continue
        info = {}
        try:
            info = json.load(open(os.path.join(p, "saved_run.json"), encoding="utf-8"))
        except Exception:
            pass
        rows.append({"path": p, "name": n, **info})
    return rows
