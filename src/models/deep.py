"""Deep-learning and classic statistical time-series challengers.

Deep learning (needs `pip install "darts[torch]"`: darts itself no longer installs PyTorch; skipped with a
message, listed in comparison/10_models_not_run.csv, if missing)
  TiDE   (Das et al. 2023, Google)       MLP encoder-decoder; uses the drivers' future (scenario) values
  N-HiTS (Challu et al. 2023)            hierarchical interpolation; drivers enter as past covariates (their
                                         scenario path is used only beyond the first 4 quarters)
  TFT    (Lim et al. 2021)               temporal fusion transformer (attention + LSTM); uses future covariates
  N-BEATS (Oreshkin et al. 2020)         optional; target growth history only
  TSMixer (Chen et al. 2023, Google)     optional; uses future covariates
All are trained on our ~100 quarters, so they are kept SMALL (a few thousand parameters), trained for a fixed
number of epochs, and refitted on the same yearly calendar as the other models. Training one takes seconds on a
CPU. With so little data they are noisy: `n_models` averages several random seeds.

Statistical
  ARIMAX  ARMA(1,1) errors with the literature drivers and the dummies as regressors (statsmodels SARIMAX);
          coefficients kept between re-estimations, the state is updated with the latest data
  Theta   the M3-competition winner (Assimakopoulos & Nikolopoulos 2000) on target growth alone
"""
import hashlib

import numpy as np
import pandas as pd

from ..modelling.ardl import make_panel

DEEP_LABELS = {"tide": "TiDE", "nhits": "N-HiTS", "tft": "TFT (transformer)", "nbeats": "N-BEATS", "tsmixer": "TSMixer"}
FUTURE_COV = {"tide", "tft", "tsmixer"}            # models that take known future covariates (scenarios)
PAST_COV = {"nhits"}                               # drivers as past covariates
NO_COV = {"nbeats"}
PER_MODEL = {"tft": {"n_models": 2, "epochs": 50}}   # TFT is ~8x slower per epoch: fewer seeds and epochs
DEEP_DEFAULTS = {"epochs": 80, "input_chunk": 8, "output_chunk": 4, "n_models": 3, "batch_size": 16, "learning_rate": 3e-3}


INSTALL = 'pip install "darts[torch]"   (darts itself no longer installs PyTorch)'


def deep_available(kinds=("tide", "nhits", "tft")):
    """Check that darts, PyTorch, Lightning and each requested network class can be loaded.

    Recent darts releases install without PyTorch: `import darts` then works but every network (TFTModel, TiDEModel,
    ...) is a placeholder that fails when used. So the classes themselves are checked.

    Args:
        kinds: the deep models requested (tide, nhits, tft, nbeats, tsmixer).

    Returns:
        (ok, reason).
    """
    try:
        import darts  # noqa: F401
    except Exception as e:
        return False, f"darts is not installed ({type(e).__name__}: {e}). Install with: {INSTALL}"
    try:
        import torch  # noqa: F401
    except Exception as e:
        return False, f"PyTorch is not installed ({type(e).__name__}: {e}). Install with: {INSTALL}"
    try:
        import pytorch_lightning  # noqa: F401
    except Exception:
        try:
            import lightning  # noqa: F401
        except Exception as e:
            return False, f"PyTorch Lightning is not installed ({type(e).__name__}: {e}). Install with: {INSTALL}"
    import darts.models as dm
    cls = {"tide": "TiDEModel", "nhits": "NHiTSModel", "tft": "TFTModel", "nbeats": "NBEATSModel",
           "tsmixer": "TSMixerModel"}
    bad = []
    for k in kinds:
        c = getattr(dm, cls.get(k, ""), None)
        if c is None or not isinstance(c, type):          # darts' NotImportedModule placeholder is not a class
            bad.append(cls.get(k, k))
    if bad:
        return False, (f"darts cannot load {', '.join(bad)} (its PyTorch part is missing or broken). "
                       f"Install with: {INSTALL}")
    return True, ""


def _dummies(P):
    """Names of the COVID dummy columns in a panel.

    Args:
        P: panel.

    Returns:
        list of str.
    """
    return [c for c in P.columns if c.startswith("D_")]


class DeepModel:
    """Small neural forecaster from darts (TiDE, N-HiTS, TFT, N-BEATS or TSMixer) trained on this data.

    The target and drivers are standardised with training-window statistics; several random seeds are averaged;
    the 12-quarter path is built from 4-quarter blocks. A fit is skipped when the training data are unchanged.
    """
    family = "Deep learning"

    def __init__(self, name, kind, features, settings=None, note=""):
        """Create a deep-learning model.

        Args:
            name: display name. kind: tide, nhits, tft, nbeats or tsmixer. features: driver features (covariates).
            settings: epochs, input_chunk, output_chunk, n_models, batch_size, learning_rate, refit_every, and
            per-model overrides such as {"tft": {"epochs": 50}}. note: free text.
        """
        self.name, self.kind, self.features = name, kind, list(features)
        self.family, self.note = "Deep learning", note
        settings = dict(settings or {})
        own = {k: settings.pop(k) for k in list(settings) if isinstance(settings[k], dict)}   # per-model overrides
        self.s = {**DEEP_DEFAULTS, **(PER_MODEL.get(kind, {})), **settings, **own.get(kind, {})}
        self.refit_every = int(self.s.pop("refit_every", 4))
        self.params, self.res = None, None
        self._fit_key, self._models = None, []

    def spec(self):
        """Part of the model interface (see src/models/zoo.py): kind, features and settings.

        Returns:
            dict.
        """
        return {"class": "Deep", "kind": self.kind, "features": self.features, "settings": self.s,
                "refit_every": self.refit_every}

    def drivers(self):
        """Part of the model interface (see src/models/zoo.py): covariates, or "(own history only)" for N-BEATS.

        Returns:
            list of str.
        """
        return list(self.features) if self.kind not in NO_COV else ["(own history only)"]

    def panel(self, ctx, mdf):
        """Part of the model interface (see src/models/zoo.py): panel with target, drivers and dummies.

        Args:
            ctx: Context. mdf: macro DataFrame.

        Returns:
            DataFrame.
        """
        return make_panel(ctx, mdf, self.features)

    # ---- data
    def _cov_cols(self, P):
        """Covariate columns used: drivers plus dummies (none for N-BEATS).

        Args:
            P: panel.

        Returns:
            list of str.
        """
        return [] if self.kind in NO_COV else self.features + _dummies(P)

    def _first_valid(self, P):
        """Position of the first quarter where the target and all covariates are known.

        Args:
            P: panel.

        Returns:
            int.
        """
        cols = ["y"] + self._cov_cols(P)
        ok = P[cols].notna().all(axis=1).values
        return int(np.argmax(ok))

    def _series(self, values, start):
        """Convert an array to a darts TimeSeries on an integer time index.

        Args:
            values: 1-D or 2-D array. start: first index.

        Returns:
            darts.TimeSeries (float32).
        """
        from darts import TimeSeries
        v = np.asarray(values, dtype=np.float32)
        if v.ndim == 1:
            v = v[:, None]
        return TimeSeries.from_times_and_values(pd.RangeIndex(start, start + len(v)), v)

    def _build(self, seed):
        """Create one untrained darts network with the configured size.

        Args:
            seed: random seed.

        Returns:
            darts model.
        """
        import torch
        from darts.models import NBEATSModel, NHiTSModel, TFTModel, TiDEModel, TSMixerModel
        s = self.s
        common = dict(input_chunk_length=int(s["input_chunk"]), output_chunk_length=int(s["output_chunk"]),
                      n_epochs=int(s["epochs"]), batch_size=int(s["batch_size"]), random_state=seed,
                      optimizer_kwargs={"lr": float(s["learning_rate"])},
                      pl_trainer_kwargs={"accelerator": "cpu", "enable_progress_bar": False,
                                         "enable_model_summary": False, "logger": False, "enable_checkpointing": False})
        if self.kind == "tide":
            return TiDEModel(hidden_size=32, num_encoder_layers=1, num_decoder_layers=1, decoder_output_dim=8,
                             temporal_decoder_hidden=16, dropout=0.1, **common)
        if self.kind == "nhits":
            return NHiTSModel(num_stacks=2, num_blocks=1, num_layers=2, layer_widths=32, dropout=0.1, **common)
        if self.kind == "tft":
            return TFTModel(hidden_size=16, lstm_layers=1, num_attention_heads=2, hidden_continuous_size=8,
                            dropout=0.1, likelihood=None, loss_fn=torch.nn.MSELoss(), **common)
        if self.kind == "nbeats":
            return NBEATSModel(generic_architecture=True, num_stacks=4, num_blocks=1, num_layers=2, layer_widths=32,
                               **common)
        if self.kind == "tsmixer":
            return TSMixerModel(hidden_size=16, ff_size=32, num_blocks=2, dropout=0.1, **common)
        raise ValueError(f"unknown deep model {self.kind}")

    # ---- fit / forecast
    def fit(self, P, start, end):
        """Part of the model interface (see src/models/zoo.py): train n_models networks on start..end (skipped if the data are unchanged).

        Args:
            P: panel. start, end: training window.

        Returns:
            self.
        """
        import logging
        logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
        logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
        f0 = max(self._first_valid(P), P.index.get_loc(start))
        e = P.index.get_loc(end)
        cols = self._cov_cols(P)
        Y = P["y"].values[f0:e + 1]
        X = P[cols].values[f0:e + 1] if cols else None
        key = hashlib.sha1(np.r_[Y, X.ravel() if X is not None else []].astype(np.float64).tobytes()).hexdigest()
        if key == self._fit_key:                      # same training data (e.g. another scenario): reuse
            return self
        self.mu, self.sd = np.nanmean(Y), np.nanstd(Y) or 1.0
        if X is not None:
            self.xmu, self.xsd = X.mean(axis=0), np.where(X.std(axis=0) > 0, X.std(axis=0), 1.0)
        ys = self._series((Y - self.mu) / self.sd, f0)
        kw = {}
        if X is not None:
            xs = self._series((X - self.xmu) / self.xsd, f0)
            kw = {"future_covariates": xs} if self.kind in FUTURE_COV else {"past_covariates": xs}
        self._models = []
        for seed in range(int(self.s["n_models"])):
            m = self._build(seed)
            m.fit(ys, verbose=False, **kw)
            self._models.append(m)
        self._fit_key, self._f0, self._fit_end = key, f0, e
        return self

    def detail_rows(self, ctx, P):
        """Driver table rows: for TFT, the importance its variable-selection networks give each input (darts
        TFTExplainer, average over the seeds); other networks have no built-in importance, so their table relies on
        the scenario-sensitivity columns added for every model.

        Args:
            ctx: Context. P: the panel the model was fitted on.

        Returns:
            list of dict rows.
        """
        if self.kind != "tft" or not self._models:
            return [{"model": self.name, "term": f, "reading": "see the scenario effect: networks have no coefficients"}
                    for f in self.features] or [{"model": self.name, "term": "(own history only)", "reading": self.note}]
        try:
            from darts.explainability import TFTExplainer
            cols = self._cov_cols(P)
            e, f0 = self._fit_end, self._f0
            ys = self._series((P["y"].values[f0:e + 1] - self.mu) / self.sd, f0)
            X = P[cols].values[f0:e + 1 + 16].astype(float)
            X = np.vstack([X, np.repeat(X[-1:], 16, axis=0)])
            xs = self._series((X - self.xmu) / self.xsd, f0)
            enc, dec = [], []
            for m in self._models:
                r = TFTExplainer(m, background_series=ys, background_future_covariates=xs).explain()
                enc.append(r.get_encoder_importance().iloc[0])
                dec.append(r.get_decoder_importance().iloc[0])
            enc, dec = pd.concat(enc, axis=1).mean(axis=1), pd.concat(dec, axis=1).mean(axis=1)
            name = lambda k: "target growth (own history)" if k.endswith("_target") else cols[int(k.split("_")[0])]
            rows = []
            for k in sorted(set(enc.index) | set(dec.index), key=lambda k: -(enc.get(k, 0) + dec.get(k, 0))):
                t = name(k)
                if t.startswith("D_"):
                    continue
                rows.append({"model": self.name, "term": t, "coef": float(enc.get(k, np.nan)),
                             "importance_decoder_pct": float(dec.get(k, np.nan)),
                             "reading": f"TFT variable selection: {enc.get(k, 0):.1f}% of the weight looking back (encoder), "
                                        f"{dec.get(k, 0):.1f}% looking ahead (decoder)"})
            return rows
        except Exception as ex:                                  # explainability is optional
            return [{"model": self.name, "term": f, "reading": f"TFT importance not available ({type(ex).__name__})"}
                    for f in self.features]

    def forecast(self, P, origin, H):
        """Part of the model interface (see src/models/zoo.py): average of the networks' forecasts, back in growth units.

        Args:
            P: panel (future covariates come from it). origin: last known quarter. H: quarters ahead.

        Returns:
            array of forecasts.
        """
        pos = P.index.get_loc(origin)
        H = min(H, len(P) - pos - 1)
        if H <= 0:
            return np.zeros(0)
        f0 = self._first_valid(P)
        ys = self._series((P["y"].values[f0:pos + 1] - self.mu) / self.sd, f0)
        cols = self._cov_cols(P)
        kw = {}
        if cols:
            X = P[cols].values[f0:pos + 1 + H].astype(float)
            pad = int(self.s["output_chunk"]) * 2 + H          # darts may ask for covariates beyond n: repeat the last row
            X = np.vstack([X, np.repeat(X[-1:], pad, axis=0)])
            X[pos + 1 - f0:, [i for i, c in enumerate(cols) if c.startswith("D_")]] = 0.0
            xs = self._series((X - self.xmu) / self.xsd, f0)
            kw = {"future_covariates": xs} if self.kind in FUTURE_COV else {"past_covariates": xs}
        preds = [m.predict(n=H, series=ys, verbose=False, **kw).values()[:, 0] for m in self._models]
        return np.mean(preds, axis=0) * self.sd + self.mu


class ARIMAX:
    """Regression on the literature drivers and dummies with ARMA errors (statsmodels SARIMAX)."""
    family = "Statistical time series"

    def __init__(self, name, features, order=(1, 0, 1), note=""):
        """Create an ARIMAX.

        Args:
            name: display name. features: driver features. order: (p, d, q) of the errors. note: free text.
        """
        self.name, self.features, self.order = name, list(features), tuple(order)
        self.family, self.note = "Statistical time series", note
        self.params, self.res = None, None

    def spec(self):
        """Part of the model interface (see src/models/zoo.py): features and order.

        Returns:
            dict.
        """
        return {"class": "ARIMAX", "features": self.features, "order": self.order}

    def drivers(self):
        """Part of the model interface (see src/models/zoo.py): driver features.

        Returns:
            list of str.
        """
        return list(self.features)

    def panel(self, ctx, mdf):
        """Part of the model interface (see src/models/zoo.py): panel with target, drivers and dummies.

        Args:
            ctx: Context. mdf: macro DataFrame.

        Returns:
            DataFrame.
        """
        return make_panel(ctx, mdf, self.features)

    def _exog(self, P):
        """Exogenous regressors: drivers and dummies.

        Args:
            P: panel.

        Returns:
            DataFrame.
        """
        return P[self.features + _dummies(P)]

    def fit(self, P, start, end):
        """Part of the model interface (see src/models/zoo.py): maximum-likelihood estimate on start..end.

        Args:
            P: panel. start, end: estimation window.

        Returns:
            self.
        """
        import warnings
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        W = P.loc[start:end]
        X = self._exog(W)
        X = X.loc[:, X.std() > 0]
        self.cols = list(X.columns)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.res = SARIMAX(W["y"].values, exog=X.values, order=self.order, trend="c").fit(disp=False)
        self.sm_params, self.start = self.res.params, start
        return self

    def detail_rows(self, ctx, P):
        """Coefficients of the regression part and of the ARMA errors, with maximum-likelihood standard errors.

        Args:
            ctx: Context. P: panel.

        Returns:
            list of dict rows.
        """
        r = self.res
        names = ["const"] + self.cols + [n for n in r.model.param_names if n.startswith(("ar.", "ma.", "sigma2"))]
        rows = []
        for i, n in enumerate(r.model.param_names):
            label = names[i] if i < len(names) else n
            if label.startswith("D_") or label == "sigma2":
                continue
            what = {"const": "constant", "ar.L1": "AR(1) term of the errors", "ma.L1": "MA(1) term of the errors"}.get(label, "")
            rows.append({"model": self.name, "term": label, "coef": float(r.params[i]), "se_hac": float(r.bse[i]),
                         "t": float(r.tvalues[i]), "p": float(r.pvalues[i]),
                         "reading": what or "effect on q/q target growth (ML estimate; errors follow ARMA(1,1))"})
        return rows

    def forecast(self, P, origin, H):
        """Part of the model interface (see src/models/zoo.py): run the Kalman filter with the stored coefficients up to the origin, then forecast with the
        macro path (dummies zero).

        Args:
            P: panel. origin: last known quarter. H: quarters ahead.

        Returns:
            array of forecasts.
        """
        import warnings
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        pos = P.index.get_loc(origin)
        H = min(H, len(P) - pos - 1)
        if H <= 0:
            return np.zeros(0)
        W = P.loc[self.start:origin]
        Xf = P[self.cols].iloc[pos + 1:pos + 1 + H].copy()
        Xf[[c for c in self.cols if c.startswith("D_")]] = 0.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = SARIMAX(W["y"].values, exog=W[self.cols].values, order=self.order, trend="c").filter(self.sm_params)
        return np.asarray(r.forecast(H, exog=Xf.values))


class Theta:
    """Theta method on target growth alone (COVID quarters replaced by the average)."""
    family = "Statistical time series"

    def __init__(self, name, note=""):
        """Create a Theta model.

        Args:
            name: display name. note: free text.
        """
        self.name, self.features, self.note = name, [], note
        self.family = "Statistical time series"
        self.params, self.res = None, None

    def spec(self):
        """Part of the model interface (see src/models/zoo.py): no settings.

        Returns:
            dict.
        """
        return {"class": "Theta"}

    def drivers(self):
        """Part of the model interface (see src/models/zoo.py): "(own history only)".

        Returns:
            list of str.
        """
        return ["(own history only)"]

    def panel(self, ctx, mdf):
        """Part of the model interface (see src/models/zoo.py): panel with target and dummies.

        Args:
            ctx: Context. mdf: macro DataFrame.

        Returns:
            DataFrame.
        """
        return make_panel(ctx, mdf, [])

    def detail_rows(self, ctx, P):
        """Theta parameters estimated on the full sample: theta and the smoothing weight alpha.

        Args:
            ctx: Context. P: panel.

        Returns:
            list of dict rows.
        """
        from statsmodels.tsa.forecasting.theta import ThetaModel
        W = P.loc[self.start:ctx.idx[ctx.end]]
        y = W["y"].astype(float).copy()
        dmy = _dummies(W)
        if dmy:
            busy = W[dmy].sum(axis=1) > 0
            y[busy] = y[~busy].mean()
        y.index = W.index.to_timestamp()
        r = ThetaModel(y, period=4, deseasonalize=False).fit()
        return [{"model": self.name, "term": k, "coef": float(v),
                 "reading": {"b0": "drift (trend slope of the theta line)", "alpha": "weight on the latest quarter in the smoothing"}.get(k, k)}
                for k, v in r.params.items()]

    def fit(self, P, start, end):
        """Part of the model interface (see src/models/zoo.py): remembers the start; the method is fitted at forecast time.

        Args:
            P: panel. start, end: window.

        Returns:
            self.
        """
        self.start = start                 # Theta has no regression coefficients; it is fitted at forecast time
        return self

    def forecast(self, P, origin, H):
        """Part of the model interface (see src/models/zoo.py): fit Theta on start..origin and forecast H quarters.

        Args:
            P: panel. origin: last known quarter. H: quarters ahead.

        Returns:
            array of forecasts.
        """
        from statsmodels.tsa.forecasting.theta import ThetaModel
        pos = P.index.get_loc(origin)
        H = min(H, len(P) - pos - 1)
        if H <= 0:
            return np.zeros(0)
        W = P.loc[self.start:origin]
        y = W["y"].astype(float).copy()
        dmy = _dummies(W)
        if dmy:
            busy = W[dmy].sum(axis=1) > 0
            y[busy] = y[~busy].mean()                  # COVID quarters replaced by the average
        y.index = W.index.to_timestamp()
        return np.asarray(ThetaModel(y, period=4, deseasonalize=False).fit().forecast(H))
