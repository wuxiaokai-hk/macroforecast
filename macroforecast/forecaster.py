"""AutoGluon-style interface: create a Forecaster, fit it, then read the leaderboard, forecasts and fitted models.

    from macroforecast import Forecaster

    fc = Forecaster(target="loan_balance", path="outputs/mortgages")           # or any column: deposits, a rate ...
    fc.fit(data="data/bank_data.csv",
           metadata="data/macro_metadata.csv",
           literature="config/literature_workbook.xlsx",
           presets="medium",                                                  # fast | medium | best | full
           hyperparameters={"en_first.target_n": 8, "sample": {"horizon": 8}})  # any config setting

    fc.leaderboard()                     # every model ranked (backtest, and out of sample when it was run)
    fc.predict()                         # scenario forecasts of the best model
    fc.predict(macro="new_scenario.csv") # forecast a new scenario with the fitted model, no re-estimation
    fc.fitted_values(), fc.coefficients(), fc.model("Screen first"), fc.backtest_models("Screen first")

    fc = Forecaster.load("outputs/mortgages")      # later, without running anything

For an interest rate (or anything that can be zero or negative) use target_form="diff": the model works on the
quarterly change and errors are in the target's units (percentage points), not percent.

`hyperparameters` (alias `settings`) takes the same keys as the YAML config, as a dict (nested or dotted keys),
a list of "key=value" strings or a YAML file, so nothing has to be edited in the YAML itself. It is applied on top
of the preset.
"""
import copy
import json
import os

import pandas as pd

from .api import PROJECT_ROOT, Results, _abs, _merge, parse_options, run

# Presets: how many model families compete (all use the same selection, backtest and out-of-sample test).
_OFF = {"deep": {"enabled": False}, "foundation": {"enabled": False}}
PRESETS = {
    "fast": {"models": {"machine_learning": [], "statistical": [], "regime": {"models": []}, **copy.deepcopy(_OFF),
                        "auto_ardl": {"levels": False}},
             "en_first": {"stability": {"n_subsamples": 30}}, "oos": {"step": 8}},
    "medium": {"models": {"machine_learning": ["ridge", "random_forest", "gradient_boosting"],
                          "statistical": ["arimax", "theta"], "regime": {"models": ["markov", "threshold"]},
                          **copy.deepcopy(_OFF)}},
    "best": {"models": {"deep": {"enabled": "auto"}, "foundation": {"enabled": "auto"}},
             "features": {"moving_average": [8], "volatility": [4, 8]}},
    "full": {},                                        # exactly the example config (every family it lists)
}


class Forecaster:
    """Explainable macro-driver forecaster for one quarterly target series.

    Args:
        target: column of the data file to forecast (e.g. "loan_balance", "deposits", "mortgage_rate").
        path: folder for everything the run produces (default outputs/<target>).
        target_form: "auto" (log growth if every value is positive), "dlog" (log growth: balances, volumes,
            prices) or "diff" (quarterly change: interest rates, spreads, ratios that can be negative).
        label: name used in titles and the deck, e.g. "Ontario deposit balance". units: e.g. "$ millions", "%".
        date_column: date column of the data file (default "date", else the first column).
        panel: column naming the bank (or region, portfolio) when the file holds several: then drivers are
            chosen on all banks together and the panel models (separate, pooled, partial pooling, top-down)
            are compared bank by bank. leaderboard(bank=...), predict(bank=...), coefficients(bank=...),
            shrinkage() give the results.
    """

    def __init__(self, target=None, path=None, target_form="auto", label=None, units=None, date_column=None,
                 panel=None):
        self.panel = panel
        self.target = target
        self.path = _abs(path) if path else None
        self.target_form = target_form
        self.label, self.units, self.date_column = label, units, date_column
        self.results = None

    # ------------------------------------------------------------------------------------------- fit
    def fit(self, data, metadata, literature=None, *, macro=None, base_scenario=None, events=None,
            presets="medium", hyperparameters=None, settings=None, oos=True, deck=False, approach=None,
            base_config=None, force=False, cache=True, trace=False, check=True):
        """Select drivers, estimate every model, backtest, forecast the scenarios and (by default) run the
        out-of-sample test. Everything estimated is saved under `path`.

        Args:
            data: file (csv/xlsx) with a date column and the target, or a pandas Series / DataFrame.
            metadata: macro metadata file (mnemonic, description, geography, theme, transform, expected_sign).
            literature: literature workbook, or None (literature, hybrid and same-growth models switched off).
            macro: scenario files {"Baseline": f1, "S4 Severe": f2, ...} or one file. Default: the example files.
            base_scenario: scenario whose history is used for estimation (default: the first).
            events: inorganic events file (date, series, amount), or None.
            presets: "fast", "medium" (default), "best" (adds deep learning, pretrained models, moving-average
                and volatility transforms) or "full" (the example config as it is).
            hyperparameters / settings: any config setting, applied on top of the preset (see module note).
            oos: also run the out-of-sample test. deck: also build the PowerPoint deck (needs Quarto).
            approach: "screen_first", "en_first" or None (both). base_config: YAML to start from.
            force / cache / trace / check: recompute everything / reuse unchanged steps / write the detailed
                trace / stop on pre-flight errors.

        Returns:
            self (so fc.fit(...).leaderboard() works).
        """
        if presets not in PRESETS:
            raise ValueError(f"presets must be one of {list(PRESETS)}")
        opts = copy.deepcopy(PRESETS[presets])
        for extra in (hyperparameters, settings):
            opts = _merge(opts, parse_options(extra))
        t = opts.setdefault("target", {})
        t.setdefault("transform", self.target_form or "auto")
        if self.label:
            t.setdefault("label", self.label)
        if self.units:
            t.setdefault("units", self.units)
        if (self.units or t.get("transform") == "diff") and "display_scale" not in t:
            t["display_scale"], t["display_units"] = None, None   # not the example's "$ billions"
        run_name, out_root = None, None
        if self.path:
            out_root, run_name = os.path.dirname(self.path), os.path.basename(self.path.rstrip(os.sep))
        self.results = run(data, metadata, literature, target_column=self.target, date_column=self.date_column,
                           macro=macro, base_scenario=base_scenario, events=events, options=opts,
                           base_config=base_config, run_name=run_name, output_dir=out_root, oos=oos, deck=deck,
                           approach=approach, force=force, cache=cache, trace=trace, check=check,
                           panel=self.panel)
        self.path = self.results.out_dir
        with open(os.path.join(self.path, "forecaster.json"), "w", encoding="utf-8") as f:
            json.dump({"target": self.target, "target_form": self.target_form, "label": self.label,
                       "units": self.units, "date_column": self.date_column, "presets": presets,
                       "panel": self.panel}, f, indent=1)
        return self

    @classmethod
    def load(cls, path):
        """A Forecaster from a finished run folder (nothing is re-estimated)."""
        path = _abs(path)
        info = {}
        p = os.path.join(path, "forecaster.json")
        if os.path.exists(p):
            info = json.load(open(p, encoding="utf-8"))
        fc = cls(info.get("target"), path, info.get("target_form", "auto"), info.get("label"), info.get("units"),
                 info.get("date_column"), info.get("panel"))
        fc.results = Results(path, os.path.join(path, "run_config.yaml"))
        return fc

    # ------------------------------------------------------------------------------------------- results
    def _need(self):
        if self.results is None:
            raise RuntimeError("call fit() first, or Forecaster.load(path)")
        return self.results

    def leaderboard(self, basis="auto", bank=None):
        """Every model ranked by average error over 1..h quarters.

        Args:
            basis: "backtest", "oos" (out of sample) or "auto" (out of sample when it was run, else backtest).
            bank: panel runs: one bank's ranking ("all" = every bank); default the overall ranking
                (mean across banks, banks won, size-weighted error).

        Returns:
            DataFrame, best first. For a diff target the *_pct columns are in the target's units.
        """
        r = self._need()
        if self.panel:
            return self._panel_table("1_leaderboard_overall" if bank is None else "2_leaderboard_by_bank", basis,
                                     bank)
        use_oos = basis == "oos" or (basis == "auto" and r.oos is not None)
        T = r.oos if use_oos else r.comparison
        if T is None:
            raise RuntimeError(f"no {'out-of-sample' if use_oos else 'backtest'} results in {r.out_dir}")
        T = T[T["family"] != "Look-ahead check"] if "family" in T else T
        cols = sorted([c for c in T.columns if c.startswith("avg_1_") and c.endswith("_pct")],
                      key=lambda c: int(c.split("_")[2]))
        T = T.sort_values(cols[-1]).reset_index(drop=True) if cols else T
        T.insert(0, "rank", range(1, len(T) + 1))
        T.attrs["basis"] = "out of sample" if use_oos else "backtest"
        return T

    @property
    def best_model(self):
        """Name of the most accurate model (out of sample when available); for a panel, the best overall."""
        if self.panel:
            return self.leaderboard().iloc[0]["model"]
        return (self._need().best or {}).get("best_model")

    # ------------------------------------------------------------------------------------------- panel
    def _pdir(self):
        return os.path.join(self._need().out_dir, "panel")

    def _panel_table(self, name, basis="auto", bank=None):
        d = self._pdir()
        oosf = os.path.join(d, "oos", f"{name}.csv")
        use_oos = basis == "oos" or (basis == "auto" and os.path.exists(oosf))
        f = oosf if use_oos else os.path.join(d, f"{name}.csv")
        T = pd.read_csv(f)
        if bank not in (None, "all") and "bank" in T:
            T = T[T["bank"] == bank].reset_index(drop=True)
            cols = sorted([c for c in T.columns if c.startswith("avg_1_") and T[c].notna().any()],
                          key=lambda c: int(c.split("_")[2]))
            if cols:
                T = T.sort_values(cols[-1]).reset_index(drop=True)
        T.attrs["basis"] = "out of sample" if use_oos else "backtest"
        return T

    @property
    def banks(self):
        """Banks in a panel run."""
        m = json.load(open(os.path.join(self._pdir(), "fitted", "manifest.json"), encoding="utf-8"))
        return m["banks"]

    def best_by_bank(self):
        """{bank: its most accurate model} (out of sample when available)."""
        m = json.load(open(os.path.join(self._pdir(), "fitted", "manifest.json"), encoding="utf-8"))
        return m["best_by_bank"]

    def shrinkage(self, bank=None):
        """Partial pooling: each bank's own coefficient, the pooled one, tau^2 and the weight on its own."""
        T = pd.read_csv(os.path.join(self._pdir(), "6_shrinkage.csv"))
        return T[T["bank"] == bank].reset_index(drop=True) if bank else T

    def predict(self, model=None, scenario=None, macro=None, horizon=None, bank=None):
        """Forecasts of the target level and its quarterly change.

        Args:
            model: model name (default: the best model). scenario: one scenario name (default: all).
            macro: a NEW scenario (file or DataFrame with a date column and the macro series, history included):
                forecast it with the fitted model, without re-estimating anything.
            horizon: quarters to return (default: all available).

        Returns:
            DataFrame: model, scenario, date, change_forecast (model form: log growth or change), level_forecast.
        """
        r = self._need()
        if self.panel:
            if macro is not None:
                raise NotImplementedError("panel runs forecast the scenarios of the run; add the scenario to macro=")
            S = pd.read_csv(os.path.join(self._pdir(), "7_scenario_forecasts.csv"))
            best = self.best_by_bank()
            if bank is not None:
                S = S[S["bank"] == bank]
            S = S[S["model"] == model] if model else S[S["model"] == S["bank"].map(best)]
            if scenario is not None:
                S = S[S["scenario"] == scenario]
            if horizon:
                S = S.groupby(["bank", "scenario"], sort=False).head(int(horizon))
            return S.reset_index(drop=True)
        name = model or self.best_model
        if macro is not None:
            return self._predict_new(name, macro, horizon, scenario or "new scenario")
        S = r.forecasts
        if S is None:
            raise RuntimeError("no scenario forecasts in the run folder")
        S = S[S["model"] == name]
        if S.empty:
            raise KeyError(f"no forecasts for model '{name}'; see leaderboard()['model']")
        if scenario is not None:
            S = S[S["scenario"] == scenario]
        S = S.rename(columns={"dlog_forecast": "change_forecast", "balance_forecast": "level_forecast"})
        if horizon:
            S = S.groupby("scenario", sort=False).head(int(horizon))
        return S.reset_index(drop=True)

    def _predict_new(self, name, macro, horizon, label):
        """Forecast a new macro scenario with a model estimated on all history."""
        import numpy as np
        from src.utils.config import load_config
        from src.utils.io import Context, add_constructed, read_table, to_quarters
        from src.utils import target as tg
        r = self._need()
        cfg = load_config(r.config_file)
        ctx = Context(cfg)
        if isinstance(macro, pd.DataFrame):
            df = macro.copy()
        else:
            df = read_table(_abs(macro), cfg["data"]["macro"].get("sheet"))
        dcol = cfg["data"]["macro"].get("date_column", "date")
        if dcol in df.columns:
            df.index = to_quarters(df.pop(dcol))
        elif not isinstance(df.index, pd.PeriodIndex):
            df.index = to_quarters(df.index)
        scen, _ = add_constructed(cfg, {label: df.apply(pd.to_numeric, errors="coerce")}, ctx.meta.copy())
        mdf = scen[label]
        m = r.models().get(name)
        if m is None:
            raise KeyError(f"model '{name}' is not in fitted/models.pkl (some model types cannot be stored)")
        P = m.panel(ctx, mdf)
        origin = ctx.idx[ctx.end]
        if origin not in P.index:
            raise ValueError(f"the new scenario must include history up to {origin}")
        H = len(P) - P.index.get_loc(origin) - 1
        H = min(H, int(horizon)) if horizon else H
        fc = np.asarray(m.forecast(P, origin, H), float)
        dates = P.index[P.index.get_loc(origin) + 1:][:len(fc)]
        lvl = tg.path(float(ctx.level.iloc[-1]), fc)
        return pd.DataFrame({"model": name, "scenario": label, "date": [str(d) for d in dates],
                             "change_forecast": fc, "level_forecast": lvl})

    # ------------------------------------------------------------------------------------------- fitted results
    def models(self):
        """Every model estimated on all history: {name: model}."""
        if self.panel:
            import pickle
            raw = pickle.load(open(os.path.join(self._pdir(), "fitted", "models.pkl"), "rb"))
            return {k: pickle.loads(b) for k, b in raw.items()}
        return self._need().models()

    def model(self, name=None):
        """One fitted model (default: the best)."""
        name = name or self.best_model
        m = self.models().get(name)
        if m is None:
            raise KeyError(f"'{name}' not stored; stored: {sorted(self.models())}")
        return m

    def _fitted_csv(self, fname, model):
        p = os.path.join(self._need().out_dir, "fitted", fname)
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        T = pd.read_csv(p)
        return T[T["model"] == model].reset_index(drop=True) if model else T

    def _panel_csv(self, name, model, bank):
        T = pd.read_csv(os.path.join(self._pdir(), name))
        if model:
            T = T[T["model"] == model]
        if bank:
            T = T[T["bank"] == bank]
        return T.reset_index(drop=True)

    def fitted_values(self, model=None, bank=None):
        """In-sample one-step fit: actual, fitted, residual and levels per quarter (all models, or one)."""
        if self.panel:
            return self._panel_csv("8_fitted_values.csv", model, bank)
        return self._fitted_csv("fitted_values.csv", model)

    def coefficients(self, model=None, bank=None):
        """Every coefficient with standard error, t and p value (all models, or one).
        Panel runs: every bank's equation; bank "(common)" holds the pooled coefficients with
        Driscoll-Kraay standard errors."""
        if self.panel:
            return self._panel_csv("5_coefficients.csv", model, bank)
        return self._fitted_csv("coefficients.csv", model)

    def fit_statistics(self):
        """Observations, R², AIC, BIC and residual standard deviation per model."""
        return self._fitted_csv("fit_statistics.csv", None)

    def backtest_models(self, name=None):
        """The model as re-estimated at every backtest origin: {origin: model}."""
        if self.panel:
            return self._panel_origins("backtest_models.pkl", name or self.best_model)
        from src.utils.saved import load_origin_models
        return load_origin_models(self._need().out_dir, name or self.best_model)

    def _panel_origins(self, fname, name=None):
        import pickle
        raw = pickle.load(open(os.path.join(self._pdir(), "fitted", fname), "rb"))
        if name:
            return {o: pickle.loads(d[name]) for o, d in raw.items() if name in d}
        return {o: {k: pickle.loads(b) for k, b in d.items()} for o, d in raw.items()}

    def oos_models(self, origin=None):
        """Out-of-sample models: {origin: {model name: model}} (or one origin's dict)."""
        if self.panel:
            allm = self._panel_origins("oos_models.pkl")
            return allm[str(origin)] if origin is not None else allm
        from src.utils.saved import load_origin_models
        d = os.path.join(self._need().out_dir, "oos", "fitted_models")
        if not os.path.isdir(d):
            raise FileNotFoundError("no out-of-sample models saved (run fit with oos=True)")
        files = sorted(f[:-4] for f in os.listdir(d) if f.endswith(".pkl"))
        if origin is not None:
            return load_origin_models(os.path.join(d, f"{origin}.pkl"))
        return {o: load_origin_models(os.path.join(d, f"{o}.pkl")) for o in files}

    def table(self, name):
        """Any output table by (part of) its file name."""
        return self._need().table(name)

    def files(self, pattern="*"):
        return self._need().files(pattern)

    def process_story(self, fmt="both"):
        """What happened at every step of the run, for an audience: an HTML page and a PowerPoint deck
        (see tools/process_story.py). fmt: html, pptx or both. Returns the paths written."""
        import importlib.util
        r = self._need()
        folder = getattr(r, "out_dir", None) or self.path
        cfg = getattr(r, "config_file", None) or os.path.join(folder, "run_config.yaml")
        spec = importlib.util.spec_from_file_location("process_story_tool",
                                                      os.path.join(PROJECT_ROOT, "tools", "process_story.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.make(cfg, folder, fmt)

    def __repr__(self):
        if self.results is None:
            return f"Forecaster(target={self.target!r}, form={self.target_form!r}, not fitted)"
        if self.panel:
            lb = self.leaderboard()
            return (f"Forecaster(target={self.target!r}, panel={self.panel!r}, path={self.path!r})\n"
                    f"  {len(self.banks)} banks; best overall ({lb.attrs['basis']}): {lb.iloc[0]['model']}")
        b = self.results.best or {}
        return (f"Forecaster(target={self.target!r}, path={self.path!r})\n  best: {b.get('best_model')} "
                f"({b.get('best_error_pct')} {b.get('horizon', '')}, {b.get('basis', '')})")
