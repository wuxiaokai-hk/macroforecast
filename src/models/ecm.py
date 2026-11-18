"""Error-correction model (Engle-Granger two-step), in the spirit of Hofmann (2001) and Calza, Gartner & Sousa (2001).

Long run (levels):   ln(real target)_t = a + sum_k c_k * z_k,t + u_t          (u_t = equilibrium error)
Short run:           y_t = a + rho*y_{t-1} + lambda*u_{t-1} + sum_j b_j*x_j,t + dummies + e_t

lambda < 0 means the target moves back towards their long-run level: a share |lambda| of the gap closes each quarter.
The long-run regressors z are built from the macro file (log, log real, level or real rate), so they follow
each scenario. Forecasts iterate the target level so the equilibrium error is updated every quarter.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from ..modelling.ardl import ARDL, make_panel
from ..utils import target as tg


def long_run_column(mdf, item, deflator):
    """One long-run regressor. form: log | log_real (deflated by CPI) | level | real_rate (rate minus y/y inflation).

    Args:
        item: {mnemonic, form} with form log, log_real, level or real_rate.
    """
    x = mdf[item["mnemonic"]]
    form = item.get("form", "log")
    if form == "log":
        return np.log(x)
    if form == "log_real":
        return np.log(x / mdf[deflator])
    if form == "level":
        return x
    if form == "real_rate":
        return x - 100 * np.log(mdf[deflator]).diff(4)
    raise ValueError(f"unknown long-run form {form}")


class ECM(ARDL):
    """Error-correction model (Engle-Granger two-step).

    Step 1 estimates the long-run relation of the real target level with its long-run drivers; step 2 is an ARDL
    for target growth that adds last quarter's deviation from that relation (the error-correction term).
    """
    kind = "ECM"

    def __init__(self, name, short_run, long_run, deflator, ar=True, family="", note=""):
        """Create an ECM.

        Args:
            name: display name. short_run: short-run driver features. long_run: list of {mnemonic, form} items.
            deflator: price index mnemonic used to make the target level real. ar, family, note: as for ARDL.
        """
        super().__init__(name, short_run, ar=ar, family=family, note=note)
        self.long_run, self.deflator = list(long_run), deflator
        self.lr_names = [f"lr:{i['mnemonic']}:{i.get('form', 'log')}" for i in self.long_run]
        self.beta = None

    def spec(self):
        """Part of the model interface (see src/models/zoo.py): ARDL settings plus the long-run items and deflator.

        Returns:
            dict.
        """
        return {**super().spec(), "long_run": self.long_run, "deflator": self.deflator}

    def drivers(self):
        """Part of the model interface (see src/models/zoo.py): short-run drivers and the long-run variables.

        Returns:
            list of str.
        """
        return list(self.features) + [f"long run: {n[3:]}" for n in self.lr_names]

    def panel(self, ctx, mdf):
        """Part of the model interface (see src/models/zoo.py): ARDL panel plus the log level of the target, log deflator and long-run columns.

        Args:
            ctx: Context. mdf: macro DataFrame.

        Returns:
            DataFrame.
        """
        P = make_panel(ctx, mdf, self.features)
        P["__logL"] = ctx.logL.reindex(P.index)
        P["__lcpi"] = tg.target_deflator(mdf, self.deflator, P.index)
        for item, n in zip(self.long_run, self.lr_names):
            P[n] = long_run_column(mdf, item, self.deflator).reindex(P.index)
        return P

    def _lr(self, P):
        """Long-run regressors as a numpy array.

        Args:
            P: panel.

        Returns:
            array (quarters x long-run items).
        """
        return P[self.lr_names].values

    def _X(self, P):
        """ARDL regressors plus the lagged error-correction term.

        Args:
            P: panel with the __ect column.

        Returns:
            DataFrame.
        """
        X = super()._X(P)
        if "__ect" in P:
            X["ect_lag1"] = P["__ect"].shift(1)
        return X

    def fit(self, P, start, end, fast=False):
        """Part of the model interface (see src/models/zoo.py): both Engle-Granger steps on the window start..end.

        Args:
            P: panel. start, end: estimation window. fast: skip the cointegration test (inside combinations).

        Returns:
            self (beta = long-run coefficients, params = short-run coefficients).
        """
        W = P.loc[start:end]
        lrL = (W["__logL"] - W["__lcpi"]).values
        Z = np.column_stack([np.ones(len(W)), W[self.lr_names].values])
        ok = ~np.isnan(lrL) & ~np.isnan(Z).any(axis=1)
        self.beta = np.linalg.lstsq(Z[ok], lrL[ok], rcond=None)[0]
        P = P.copy()
        Zall = np.column_stack([np.ones(len(P)), self._lr(P)])
        P["__ect"] = (P["__logL"] - P["__lcpi"]).values - Zall @ self.beta
        self.ect = P["__ect"]
        super().fit(P, start, end, fast=fast)
        if fast:
            return self
        try:                                                  # Engle-Granger: is the long-run error stationary?
            from statsmodels.tsa.stattools import adfuller
            self.eg_p = float(adfuller(self.ect.loc[start:end].dropna(), autolag="AIC")[1])
        except Exception:
            self.eg_p = np.nan
        return self

    def forecast(self, P, origin, H):
        """Part of the model interface (see src/models/zoo.py): iterate growth forward, updating the target level and the error-correction term each quarter.

        Args:
            P: panel. origin: last known quarter. H: quarters ahead.

        Returns:
            array of forecasts.
        """
        pos = P.index.get_loc(origin)
        H = min(H, len(P) - pos - 1)
        p = self.params
        yprev = P["y"].iloc[pos]
        logL = P["__logL"].iloc[pos]
        z = np.r_[1.0, P[self.lr_names].iloc[pos].values]
        ect = (logL - P["__lcpi"].iloc[pos]) - z @ self.beta
        out = np.zeros(H)
        for h in range(1, H + 1):
            row = P.iloc[pos + h]
            v = p.get("const", 0.0) + p.get("y_lag1", 0.0) * yprev + p.get("ect_lag1", 0.0) * ect
            v += sum(p[f] * row[f] for f in self.features if f in p)
            out[h - 1] = v
            yprev = v
            logL += v
            z = np.r_[1.0, row[self.lr_names].values]
            ect = (logL - row["__lcpi"]) - z @ self.beta
        return out
