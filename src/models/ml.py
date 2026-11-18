"""Machine-learning challengers: ridge, random forest, gradient boosting and a small neural network.

Same information set as the regression models: last quarter's target growth, the dummies and a pool of macro
features (the literature core plus the data-chosen drivers). Forecasts are iterated quarter by quarter with
the macro path given (realised values in backtests, the scenario in scenario forecasts), so the models are
directly comparable with the ARDLs and respond to scenarios.

Hyperparameters are fixed, sensible values for about 100 quarterly observations (no tuning loop), which keeps
the run time low; they can be changed in the config under models.machine_learning_settings. In the rolling
backtest these models are re-estimated every `refit_every` (default 4) origins instead of every origin, again
only with data up to that origin; set refit_every: 1 to refit at every origin (about four times slower).
"""
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..modelling.ardl import make_panel

LABELS = {"ridge": "Ridge regression", "random_forest": "Random forest", "gradient_boosting": "Gradient boosting",
          "neural_net": "Neural network (MLP)"}
DEFAULTS = {
    "ridge": {},
    "random_forest": {"n_estimators": 200, "min_samples_leaf": 3, "max_features": 0.5},
    "gradient_boosting": {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 2, "subsample": 0.8},
    "neural_net": {"hidden": 8, "alpha": 3.0, "n_models": 5},
}


def make_estimator(kind, params):
    """Build the scikit-learn estimator for a machine-learning model.

    Args:
        kind: ridge, random_forest, gradient_boosting or neural_net. params: overrides of DEFAULTS[kind].

    Returns:
        unfitted estimator (the neural net is an averaged ensemble of small MLPs).
    """
    p = {**DEFAULTS[kind], **(params or {})}
    if kind == "ridge":
        return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 25)))
    if kind == "random_forest":
        return RandomForestRegressor(random_state=0, n_jobs=-1, **p)
    if kind == "gradient_boosting":
        return GradientBoostingRegressor(random_state=0, **p)
    if kind == "neural_net":                 # small averaged ensemble: one net on 100 points is too seed-dependent
        nets = [(f"m{i}", MLPRegressor(hidden_layer_sizes=(int(p["hidden"]),), alpha=float(p["alpha"]), solver="lbfgs",
                                       max_iter=1000, random_state=i)) for i in range(int(p["n_models"]))]
        return TransformedTargetRegressor(regressor=make_pipeline(StandardScaler(), VotingRegressor(nets)),
                                          transformer=StandardScaler())
    raise ValueError(f"unknown machine-learning model {kind}")


class MLModel:
    """Machine-learning model on the same inputs as the ARDL (lagged growth, drivers, dummies);
    forecasts are iterated with the macro path like the regressions.
    """
    family = "Machine learning"

    def __init__(self, name, features, kind, params=None, note="", refit_every=4):
        """Create a machine-learning model.

        Args:
            name: display name. features: driver features. kind: see make_estimator. params: estimator settings
            (may contain refit_every). note: free text. refit_every: quarters between re-estimations in the backtest.
        """
        params = dict(params or {})
        self.refit_every = int(params.pop("refit_every", refit_every))
        self.name, self.features, self.kind, self.params_cfg, self.note = name, list(features), kind, params, note
        self.family = "Machine learning"
        self.params = None
        self.res = None

    def spec(self):
        """Part of the model interface (see src/models/zoo.py): kind, features and settings.

        Returns:
            dict.
        """
        return {"class": "ML", "kind": self.kind, "features": self.features, "params": self.params_cfg,
                "refit_every": self.refit_every}

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

    def _X(self, P, yprev=None):
        """Input matrix: lagged growth, drivers and dummies.

        Args:
            P: panel. yprev: not used (kept for compatibility).

        Returns:
            DataFrame.
        """
        X = pd.DataFrame({"y_lag1": P["y"].shift(1)}, index=P.index)
        for f in self.features:
            X[f] = P[f]
        for c in [c for c in P.columns if c.startswith("D_")]:
            X[c] = P[c]
        return X

    def fit(self, P, start, end):
        """Part of the model interface (see src/models/zoo.py): train the estimator on complete rows of start..end.

        Args:
            P: panel. start, end: training window.

        Returns:
            self.
        """
        X = self._X(P).loc[start:end]
        y = P["y"].loc[start:end]
        ok = X.notna().all(axis=1) & y.notna()
        self.cols = list(X.columns)
        self.est = make_estimator(self.kind, self.params_cfg).fit(X[ok].values, y[ok].values)
        if self.kind == "random_forest":
            self.est.set_params(n_jobs=1)        # trained in parallel; one-row predictions are faster on one core
        return self

    def forecast(self, P, origin, H):
        """Part of the model interface (see src/models/zoo.py): predict one quarter at a time, feeding each prediction back as lagged growth.

        Args:
            P: panel. origin: last known quarter. H: quarters ahead.

        Returns:
            array of forecasts.
        """
        pos = P.index.get_loc(origin)
        H = min(H, len(P) - pos - 1)
        X = self._X(P)
        yprev = P["y"].iloc[pos]
        out = np.zeros(H)
        for h in range(1, H + 1):
            row = X.iloc[pos + h].copy()
            row["y_lag1"] = yprev
            row[[c for c in self.cols if c.startswith("D_")]] = 0.0
            v = float(self.est.predict(row[self.cols].values.reshape(1, -1))[0])
            out[h - 1] = v
            yprev = v
        return out

    def importance(self):
        """Feature importance (trees) or standardised coefficients (ridge), for the driver table."""
        est = self.est
        if self.kind in ("random_forest", "gradient_boosting"):
            return pd.Series(est.feature_importances_, index=self.cols)
        if self.kind == "ridge":
            return pd.Series(est[-1].coef_, index=self.cols)
        return None
