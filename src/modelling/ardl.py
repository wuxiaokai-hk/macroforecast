"""ARDL model used as the final model by both approaches:

    y_t = a + rho*y_{t-1} + sum_j b_j*x_{j,t-l_j} + sum_q g_q*D_{q,t} + e_t

Estimated by OLS with Newey-West standard errors; forecasts are iterated, with the lagged target set to
its own forecast and the dummies set to zero.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from ..features.transforms import feature_matrix


def ols_np(X, y):
    """Plain OLS with numpy (fast; used inside the exhaustive search).

    Args:
        X: regressor matrix (n x k), including the constant. y: target vector (n).

    Returns:
        (coefficients, standard errors, residuals). Standard errors are classical (not Newey-West).
    """
    X, y = np.asarray(X, float), np.asarray(y, float)
    ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
    if not ok.all():
        # quarters with a missing or infinite value (e.g. the log of a zero) are left out of the fit: they get zero
        # weight and a zero residual, instead of stopping the whole run with "SVD did not converge"
        X, y = np.where(ok[:, None], X, 0.0), np.where(ok, y, 0.0)
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    s2 = e @ e / max(int(ok.sum()) - X.shape[1], 1)
    return b, np.sqrt(np.clip(np.diag(XtX_inv) * s2, 1e-30, None)), e


def vif(F):
    """Variance inflation factors of the columns of F (multicollinearity check).

    Args:
        F: matrix of drivers (n x k).

    Returns:
        array of k VIFs (1 when k < 2, inf if the correlation matrix is singular).
    """
    if F.shape[1] < 2:
        return np.ones(F.shape[1])
    try:
        return np.diag(np.linalg.inv(np.corrcoef(F, rowvar=False)))
    except np.linalg.LinAlgError:
        return np.full(F.shape[1], np.inf)


def make_panel(ctx, mdf, features):
    """Quarterly panel for one scenario: target, dummies and features (history + scenario path)."""
    idx = mdf.index[mdf.index >= ctx.idx[0]]
    P = pd.DataFrame(index=idx)
    P["y"] = ctx.y.reindex(idx)
    for c in ctx.dummies.columns:
        P[c] = ctx.dummies[c].reindex(idx).fillna(0.0)
    F = feature_matrix(mdf, features, ctx.cfg["features"]["deflator"]).reindex(idx)
    return pd.concat([P, F], axis=1).replace([np.inf, -np.inf], np.nan)


class ARDL:
    """Every model type (ARDL, ECM, machine learning, combinations, foundation models) has the same interface:
    panel(ctx, mdf) builds its data, fit(P, start, end) estimates on [start, end], forecast(P, origin, H)
    returns H quarterly log changes, and spec() describes it for the cache."""
    kind = "ARDL"

    def __init__(self, name, features, ar=True, family="", note=""):
        """Create an ARDL specification (nothing is estimated yet).

        Args:
            name: display name. features: driver feature names. ar: include lagged target growth.
            family: model family for the comparison. note: free text shown in the model list.
        """
        self.name, self.features, self.ar = name, list(features), ar
        self.family, self.note = family, note
        self.params = None

    def panel(self, ctx, mdf):
        """Part of the model interface (see src/models/zoo.py): build the model's data panel (target, drivers, dummies) from a macro scenario.

        Args:
            ctx: Context. mdf: macro DataFrame (history or a scenario).

        Returns:
            DataFrame with columns y, the features and D_ dummies.
        """
        return make_panel(ctx, mdf, self.features)

    def spec(self):
        """Part of the model interface (see src/models/zoo.py): the settings that define the model (used in cache keys).

        Returns:
            dict.
        """
        out = {"class": type(self).__name__, "features": self.features, "ar": self.ar}
        if getattr(self, "signs", None):
            out["signs"] = self.signs
        return out

    def drivers(self):
        """Part of the model interface (see src/models/zoo.py): the model's drivers for tables.

        Returns:
            list of feature names.
        """
        return list(self.features)

    def _X(self, P):
        """Regressor matrix: constant, lagged growth, drivers and dummies.

        Args:
            P: panel from panel().

        Returns:
            DataFrame of regressors on P's index.
        """
        X = pd.DataFrame({"const": 1.0}, index=P.index)
        if self.ar:
            X["y_lag1"] = P["y"].shift(1)
        for f in self.features:
            X[f] = P[f]
        for c in [c for c in P.columns if c.startswith("D_")]:
            X[c] = P[c]
        return X

    def fit(self, P, start, end, fast=False):
        """fast=True: plain least squares, no standard errors (used inside combinations' weight estimation).

        Args:
            fast: plain least squares without standard errors.
        """
        X = self._X(P).loc[start:end]
        yv = P["y"].loc[start:end]
        X = X.loc[:, (X.std() > 0) | (X.columns == "const")]
        ok = X.notna().all(axis=1) & yv.notna()
        if fast:
            b = np.linalg.lstsq(X[ok].values, yv[ok].values, rcond=None)[0]
            self.params, self.res = pd.Series(b, index=X.columns), None
            return self
        self.res = sm.OLS(yv[ok], X[ok]).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
        self.params = self.res.params
        return self

    def forecast(self, P, origin, H):
        """Part of the model interface (see src/models/zoo.py): iterate the estimated equation forward from an origin.

        Lagged growth is replaced by the model's own forecast after the first quarter; drivers come from P
        (actual values in the backtest, the scenario path in scenario forecasts); dummies are zero.

        Args:
            P: panel. origin: last known quarter (Period). H: quarters ahead.

        Returns:
            array of up to H forecasts of quarterly log growth.
        """
        pos = P.index.get_loc(origin)
        H = min(H, len(P) - pos - 1)
        yprev = P["y"].iloc[pos]
        out = np.zeros(H)
        for h in range(1, H + 1):
            row = P.iloc[pos + h]
            v = self.params.get("const", 0.0) + self.params.get("y_lag1", 0.0) * yprev
            v += sum(self.params[f] * row[f] for f in self.features if f in self.params)
            out[h - 1] = v
            yprev = v
        return out
