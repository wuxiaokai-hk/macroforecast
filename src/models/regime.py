"""Regime-switching models: target growth whose relation with the drivers changes with the state of the economy.

All are ARDL(1) models on the literature core drivers (the same inputs as the four-channel core, so the
comparison isolates the effect of allowing regimes), with the COVID dummies as fixed controls.

  Markov switching (Hamilton 1989)    two unobserved regimes (e.g. credit boom / slowdown) with their own
                                      mean growth and volatility; the regime is inferred from the data and its
                                      probability is carried forward with the estimated transition matrix
  Threshold ARDL (Tong; Hansen 2000)  two regimes set by an observed variable (default: house price growth)
                                      crossing an estimated threshold; every coefficient may differ by regime;
                                      the scenario path decides the regime in each future quarter
  Smooth transition (LSTAR; Terasvirta 1994)
                                      as the threshold model, but the switch is gradual (logistic weight)
  Time-varying parameters (Kalman filter; Cogley & Sargent 2005 style)
                                      every coefficient drifts slowly as a random walk: gradual regime change
  Break-adaptive ARDL (Andrews 1993; Pesaran & Timmermann 2007)
                                      tests for a structural break at every re-estimation (sup-F) and, if one
                                      is found, estimates only on the data after the latest break

Forecasts are iterated like the ARDL: lagged growth is replaced by the model's own forecast; drivers (and the
threshold variable) come from the panel, i.e. realised values in the backtest and the Moody's path in scenarios.
"""
import warnings

import numpy as np
import pandas as pd

from ..modelling.ardl import make_panel

FAMILY = "Regime switching"
# 5% critical values of the sup-F (Quandt-Andrews) break test, 15% trimming, by number of coefficients tested
# (Andrews 1993, Table 1, corrected in Andrews 2003)
SUP_F_5PCT = {1: 8.85, 2: 11.79, 3: 14.15, 4: 16.45, 5: 18.62, 6: 20.44, 7: 22.40, 8: 24.25, 9: 26.06, 10: 27.82,
              11: 29.53, 12: 31.22}


def _dummies(P):
    """Names of the COVID dummy columns in a panel.

    Args:
        P: panel.

    Returns:
        list of str.
    """
    return [c for c in P.columns if c.startswith("D_")]


class _Regime:
    """Shared plumbing: data panel, design matrices, iterated forecasts and the model-interface methods."""
    family = FAMILY
    refit_every = 1

    def __init__(self, name, features, note="", threshold=None):
        """Create the model.

        Args:
            name: display name. features: driver features. note: free text.
            threshold: feature that sets the regime (threshold and smooth-transition models).
        """
        self.name, self.features, self.note = name, list(features), note
        self.threshold = threshold
        self.family = FAMILY
        self.params, self.res = None, None

    def spec(self):
        """Part of the model interface (see src/models/zoo.py): class, drivers and threshold variable.

        Returns:
            dict.
        """
        return {"class": type(self).__name__, "features": self.features, "threshold": self.threshold,
                **getattr(self, "_extra_spec", {})}

    def drivers(self):
        """Part of the model interface (see src/models/zoo.py): drivers, plus the regime variable.

        Returns:
            list of str.
        """
        return list(self.features) + ([f"regime: {self.threshold}"] if self.threshold else [])

    def panel(self, ctx, mdf):
        """Part of the model interface (see src/models/zoo.py): target, drivers, regime variable and dummies.

        Args:
            ctx: Context. mdf: macro DataFrame.

        Returns:
            DataFrame.
        """
        feats = list(dict.fromkeys(self.features + ([self.threshold] if self.threshold else [])))
        return make_panel(ctx, mdf, feats)

    def _window(self, P, start, end):
        """Complete rows of start..end: switching regressors S (const, lagged growth, drivers), fixed regressors
        D (dummies that vary in the window), target y and regime variable z.

        Args:
            P: panel. start, end: estimation window.

        Returns:
            (S, D, y, z, index) as numpy arrays and the quarters used.
        """
        W = P.loc[start:end]
        S = pd.DataFrame({"const": 1.0, "y_lag1": P["y"].shift(1).loc[start:end]}, index=W.index)
        for f in self.features:
            S[f] = W[f]
        D = W[_dummies(W)]
        D = D.loc[:, D.std() > 0]
        z = W[self.threshold] if self.threshold else pd.Series(0.0, index=W.index)
        ok = S.notna().all(axis=1) & W["y"].notna() & z.notna()
        self.s_cols, self.d_cols = list(S.columns), list(D.columns)
        return S[ok].values, D[ok].values, W["y"][ok].values, z[ok].values, W.index[ok]

    def _row(self, P, pos, yprev):
        """Switching regressors for quarter pos with lagged growth set to yprev (dummies are zero ahead).

        Args:
            P: panel. pos: position of the quarter. yprev: last (forecast) growth.

        Returns:
            array.
        """
        r = P.iloc[pos]
        return np.r_[1.0, yprev, [r[f] for f in self.features]]

    def forecast(self, P, origin, H):
        """Part of the model interface (see src/models/zoo.py): iterate growth forward, one quarter at a time.

        Args:
            P: panel. origin: last known quarter. H: quarters ahead.

        Returns:
            array of forecasts.
        """
        pos = P.index.get_loc(origin)
        H = min(H, len(P) - pos - 1)
        yprev = P["y"].iloc[pos]
        out = np.zeros(H)
        state = self._start_state(P, origin)
        for h in range(1, H + 1):
            x = self._row(P, pos + h, yprev)
            v, state = self._step(x, P, pos + h, state, h)
            out[h - 1] = yprev = v
        return out

    def _start_state(self, P, origin):
        """Regime information at the origin (overridden by the Markov model).

        Args:
            P: panel. origin: last known quarter.
        """
        return None

    def _named(self, b, prefix=""):
        """Coefficients as a Series named after the switching regressors.

        Args:
            b: coefficient vector for the switching regressors. prefix: label added in front.

        Returns:
            Series.
        """
        return pd.Series(b, index=[f"{prefix}{c}" for c in self.s_cols])


def _ols(X, y):
    """Least squares with the residual sum of squares.

    Args:
        X: regressors. y: target.

    Returns:
        (coefficients, SSR).
    """
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    return b, float(e @ e)


# ------------------------------------------------------------------------------------------------- threshold
class ThresholdARDL(_Regime):
    """Two-regime threshold ARDL: all coefficients differ below and above an estimated threshold of z."""
    regime_label = "1 = high regime (regime variable above the threshold)"


    def __init__(self, name, features, threshold, note="", trim=0.15, grid=40):
        """Create a threshold model.

        Args:
            name: display name. features: driver features. threshold: feature that sets the regime.
            note: free text. trim: smallest share of observations in each regime. grid: thresholds tried.
        """
        super().__init__(name, features, note, threshold)
        self.trim, self.grid = trim, grid
        self._extra_spec = {"trim": trim, "grid": grid}

    def fit(self, P, start, end):
        """Part of the model interface (see src/models/zoo.py): grid search of the threshold (least squares,
        Hansen 2000), then OLS in each regime; a sup-F statistic measures the evidence for two regimes.

        Args:
            P: panel. start, end: estimation window.

        Returns:
            self.
        """
        S, D, y, z, _ = self._window(P, start, end)
        n, k = S.shape
        lin, ssr0 = _ols(np.column_stack([S, D]), y)
        nmin = max(int(self.trim * n), k + 3)
        cands = np.unique(np.quantile(z, np.linspace(self.trim, 1 - self.trim, self.grid)))
        best = (np.inf, None, None)
        for c in cands:
            lo = z <= c
            if lo.sum() < nmin or (~lo).sum() < nmin:
                continue
            X = np.column_stack([S * lo[:, None], S * (~lo)[:, None], D])
            b, ssr = _ols(X, y)
            if ssr < best[0]:
                best = (ssr, c, b)
        if best[1] is None:                                       # too few observations: one regime
            self.c, self.b_lo = np.inf, lin[:k]
            self.b_hi, self.supF = lin[:k], 0.0
        else:
            ssr, self.c, b = best
            self.b_lo, self.b_hi = b[:k], b[k:2 * k]
            self.supF = (ssr0 - ssr) / k / (ssr / max(n - 2 * k - D.shape[1], 1))
        self.share_low = float(np.mean(z <= self.c))
        self.params = pd.concat([self._named(self.b_lo, "low:"), self._named(self.b_hi, "high:"),
                                 pd.Series({"threshold": self.c})])
        return self

    def _step(self, x, P, pos, state, h):
        """One quarter ahead: the regime is set by the regime variable in that quarter.

        Args:
            x: regressors. P: panel. pos: quarter position. state: unused. h: step.

        Returns:
            (forecast, state).
        """
        zt = P[self.threshold].iloc[pos]
        b = self.b_lo if (np.isnan(zt) or zt <= self.c) else self.b_hi
        return float(x @ b), state

    def driver_rows(self):
        """Coefficients in each regime, for the driver table.

        Returns:
            list of dict rows.
        """
        rows = [{"model": self.name, "term": f"{self.threshold} threshold", "coef": self.c,
                 "reading": f"{self.share_low:.0%} of quarters in the low regime; sup-F {self.supF:.1f} "
                            f"(5% critical value about {SUP_F_5PCT.get(len(self.b_lo), np.nan):.1f}, Hansen's test needs a bootstrap)"}]
        for c, lo, hi in zip(self.s_cols, self.b_lo, self.b_hi):
            if c != "const":
                rows.append({"model": self.name, "term": c, "coef": hi, "reading": f"low regime {lo:.4f}, high regime {hi:.4f}"})
        return rows

    def regime_path(self, P):
        """Regime in each quarter of the panel: 1 = high regime, 0 = low.

        Args:
            P: panel.

        Returns:
            Series.
        """
        z = P[self.threshold]
        return (z > self.c).astype(float).where(z.notna())


# ------------------------------------------------------------------------------------------------- LSTAR
class SmoothTransitionARDL(_Regime):
    """Logistic smooth-transition ARDL: coefficients move gradually from one regime to the other as z rises."""
    regime_label = "weight of the high regime (0 to 1)"


    GAMMAS = (1.0, 2.5, 5.0, 10.0, 25.0)

    def __init__(self, name, features, threshold, note="", trim=0.15, grid=15):
        """Create a smooth-transition model.

        Args:
            name: display name. features: driver features. threshold: transition variable.
            note: free text. trim: quantile range for the location. grid: locations tried.
        """
        super().__init__(name, features, note, threshold)
        self.trim, self.grid = trim, grid
        self._extra_spec = {"trim": trim, "grid": grid}

    def _G(self, z):
        """Logistic transition weight G in [0, 1] (0 = low regime, 1 = high regime).

        Args:
            z: transition variable.

        Returns:
            array.
        """
        return 1.0 / (1.0 + np.exp(-np.clip(self.gamma * (np.asarray(z, float) - self.c) / self.zsd, -50, 50)))

    def fit(self, P, start, end):
        """Part of the model interface (see src/models/zoo.py): grid search of slope gamma and location c
        (concentrated least squares), then OLS of y on S, G*S and the dummies.

        Args:
            P: panel. start, end: estimation window.

        Returns:
            self.
        """
        S, D, y, z, _ = self._window(P, start, end)
        k = S.shape[1]
        self.zsd = float(np.std(z)) or 1.0
        best = (np.inf, None)
        for g in self.GAMMAS:
            for c in np.quantile(z, np.linspace(self.trim, 1 - self.trim, self.grid)):
                self.gamma, self.c = g, c
                G = self._G(z)
                b, ssr = _ols(np.column_stack([S, S * G[:, None], D]), y)
                if ssr < best[0]:
                    best = (ssr, (g, c, b))
        self.gamma, self.c, b = best[1]
        self.b0, self.b1 = b[:k], b[k:2 * k]
        self.params = pd.concat([self._named(self.b0, "low:"), self._named(self.b0 + self.b1, "high:"),
                                 pd.Series({"gamma": self.gamma, "location": self.c})])
        return self

    def _step(self, x, P, pos, state, h):
        """One quarter ahead with the transition weight of that quarter.

        Args:
            x: regressors. P: panel. pos: quarter position. state: unused. h: step.

        Returns:
            (forecast, state).
        """
        zt = P[self.threshold].iloc[pos]
        G = 0.0 if np.isnan(zt) else float(self._G([zt])[0])
        return float(x @ (self.b0 + G * self.b1)), state

    def driver_rows(self):
        """Coefficients at the two ends of the transition, for the driver table.

        Returns:
            list of dict rows.
        """
        rows = [{"model": self.name, "term": f"{self.threshold} transition", "coef": self.c,
                 "reading": f"centre {self.c:.4f}, slope {self.gamma:g} (per s.d.)"}]
        for c, lo, d in zip(self.s_cols, self.b0, self.b1):
            if c != "const":
                rows.append({"model": self.name, "term": c, "coef": lo + d, "reading": f"low end {lo:.4f}, high end {lo + d:.4f}"})
        return rows

    def regime_path(self, P):
        """Weight of the high regime in each quarter (0 to 1).

        Args:
            P: panel.

        Returns:
            Series.
        """
        z = P[self.threshold]
        return pd.Series(self._G(z.fillna(self.c).values), index=P.index).where(z.notna())


# ------------------------------------------------------------------------------------------------- Markov
class MarkovSwitchingARDL(_Regime):
    """Hamilton two-regime Markov-switching ARDL: the intercept (mean growth) switches between a low- and a
    high-growth regime; the drivers' effects are common (lagged growth may switch too).

    Estimated in percent with standardised regressors, which the likelihood search needs with ~100 quarters
    of small growth rates; coefficients are reported back in the original units.
    """
    regime_label = "probability of the high-growth regime"


    def __init__(self, name, features, note="", switching_ar=False, switching_variance=False, search_reps=20):
        """Create a Markov-switching model.

        Args:
            name: display name. features: driver features. note: free text.
            switching_ar: let the coefficient on lagged growth switch too.
            switching_variance: give each regime its own volatility (with ~100 quarters this often lets one
            regime collapse onto a few outliers, so it is off by default). search_reps: random starts in the fit.
        """
        super().__init__(name, features, note)
        self.switching_ar, self.switching_variance, self.search_reps = switching_ar, switching_variance, search_reps
        self._extra_spec = {"switching_ar": switching_ar, "switching_variance": switching_variance,
                            "search_reps": search_reps}

    def _exog(self, P, start, end):
        """Target and exogenous regressors (lagged growth, drivers, dummies) on complete rows.

        Args:
            P: panel. start, end: window.

        Returns:
            (y Series, exog DataFrame).
        """
        W = P.loc[start:end]
        X = pd.DataFrame({"y_lag1": P["y"].shift(1).loc[start:end]}, index=W.index)
        for f in self.features:
            X[f] = W[f]
        for c in self.d_cols:
            X[c] = W[c]
        ok = X.notna().all(axis=1) & W["y"].notna()
        return W["y"][ok], X[ok]

    def _std(self, X):
        """Regressors in estimation units: lagged growth in percent, then standardised with the fit's statistics.

        Args:
            X: DataFrame of regressors (original units).

        Returns:
            array.
        """
        X = X.reindex(columns=self.x_cols, fill_value=0.0).astype(float).copy()
        X["y_lag1"] = X["y_lag1"] * 100
        return ((X - self.mu) / self.sd).values

    def _model(self, y, Xs):
        """statsmodels MarkovRegression with a switching intercept.

        Args:
            y: target in percent. Xs: standardised regressors.

        Returns:
            MarkovRegression.
        """
        from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
        sw = [c == "y_lag1" and self.switching_ar for c in self.x_cols]
        return MarkovRegression(np.asarray(y, float), k_regimes=2, exog=Xs, trend="c", switching_exog=sw,
                                switching_variance=self.switching_variance)

    def fit(self, P, start, end):
        """Part of the model interface (see src/models/zoo.py): maximum likelihood with several random starts;
        falls back to one regime (OLS) if the likelihood search fails or gives a degenerate regime.

        Args:
            P: panel. start, end: estimation window.

        Returns:
            self.
        """
        W = P.loc[start:end]
        D = W[_dummies(W)]
        self.d_cols = list(D.columns[D.std() > 0])
        y, X = self._exog(P, start, end)
        self.x_cols, self.start, self.fit_index = list(X.columns), start, y.index
        X100 = X.copy()
        X100["y_lag1"] *= 100
        self.mu, self.sd = X100.mean(), X100.std().replace(0, 1.0)
        self.failed, self.res = False, None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rng_state = np.random.get_state()
                np.random.seed(0)
                try:
                    self.res = self._model(y * 100, self._std(X)).fit(search_reps=self.search_reps, disp=False)
                finally:
                    np.random.set_state(rng_state)
            self.mp = np.asarray(self.res.params)
            if not np.all(np.isfinite(self.mp)):
                raise ValueError("non-finite parameters")
            c = [v for k_, v in zip(self.res.model.param_names, self.mp) if k_.startswith("const")]
            if max(abs(v - 100 * np.mean(y.values)) for v in c) > 600 * np.std(y.values):
                raise ValueError("degenerate regime (intercept far outside the data)")
        except Exception as e:                                   # one regime instead
            self.failed, self.fail_reason = True, f"{type(e).__name__}: {e}"
            Z = np.column_stack([np.ones(len(y)), self._std(X)])
            self.ols, _ = _ols(Z, y.values * 100)
        self._unpack()
        return self

    def _unpack(self):
        """Intercepts, slopes (estimation units) and transition matrix, regime 1 = high growth; plus params in
        original units for the coefficient path."""
        if self.failed:
            b = self.ols
            self.order = np.array([0, 1])
            self.const = np.array([b[0], b[0]])
            self.beta = {c: np.array([v, v]) for c, v in zip(self.x_cols, b[1:])}
            self.T = np.eye(2)
            self.sigma = np.array([np.nan, np.nan])
        else:
            p = dict(zip(self.res.model.param_names, self.mp))
            const = np.array([p["const[0]"], p["const[1]"]])
            order = np.argsort(const)                           # regime 0 = low growth, 1 = high growth
            self.order, self.const = order, const[order]
            self.beta = {}
            for j, c in enumerate(self.x_cols):
                a, b_ = p.get(f"x{j + 1}[0]"), p.get(f"x{j + 1}[1]")
                if a is not None and b_ is not None:              # switching coefficient
                    self.beta[c] = np.array([a, b_])[order]
                else:                                             # common coefficient (named x[1] or x by statsmodels)
                    v = next(p[k] for k in (f"x{j + 1}", f"x{j + 1}[1]", f"x{j + 1}[0]") if k in p)
                    self.beta[c] = np.array([v, v])
            T = np.asarray(self.res.regime_transition)[:, :, 0]   # T[i, j] = P(s_t = i | s_t-1 = j)
            self.T = T[np.ix_(order, order)]
            s2 = [p.get("sigma2[0]", p.get("sigma2")), p.get("sigma2[1]", p.get("sigma2"))]
            self.sigma = np.sqrt(np.array(s2, dtype=float))[order] / 100
        # original units: y = c0 + sum b_c x_c with b_c = beta/sd (x100 for drivers' effect on y in %)
        self.b_orig = {c: self.beta[c] / self.sd[c] * (1.0 if c == "y_lag1" else 0.01) for c in self.x_cols}
        shift = sum(self.beta[c] * self.mu[c] / self.sd[c] for c in self.x_cols)
        self.c_orig = (self.const - shift) / 100
        out = {"const[low]": self.c_orig[0], "const[high]": self.c_orig[1]}
        for c in self.x_cols:
            b = self.b_orig[c]
            if b[0] == b[1]:
                out[c] = b[0]
            else:
                out[f"{c}[low]"], out[f"{c}[high]"] = b[0], b[1]
        out.update({"p(stay low)": self.T[0, 0], "p(stay high)": self.T[1, 1]})
        self.params = pd.Series(out)

    def _probs(self, P, origin):
        """Filtered regime probabilities at every quarter up to origin, with the stored parameters.

        Args:
            P: panel. origin: last known quarter.

        Returns:
            (probabilities array (quarters x 2, low/high), quarters).
        """
        y, X = self._exog(P, self.start, origin)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = self._model(y * 100, self._std(X)).filter(self.mp)
        pr = np.asarray(r.filtered_marginal_probabilities)
        pr = pr if pr.shape[1] == 2 else pr.T
        return pr[:, self.order], y.index

    def _start_state(self, P, origin):
        """Probability of each regime at the origin.

        Args:
            P: panel. origin: last known quarter.

        Returns:
            array (low, high).
        """
        p = np.array([0.5, 0.5])
        if not self.failed:
            try:
                p = self._probs(P, origin)[0][-1]
            except Exception:
                p = np.array([np.nan, np.nan])
        if not np.all(np.isfinite(p)):                            # fall back to the long-run regime shares
            a, b = 1 - self.T[0, 0], 1 - self.T[1, 1]
            p = np.array([b, a]) / (a + b) if a + b > 0 else np.array([0.5, 0.5])
        return p

    def _row(self, P, pos, yprev):
        """Exogenous regressors for quarter pos (dummies zero ahead).

        Args:
            P: panel. pos: position. yprev: last (forecast) growth.

        Returns:
            dict column -> value.
        """
        r = P.iloc[pos]
        return {c: (yprev if c == "y_lag1" else (0.0 if c.startswith("D_") else r[c])) for c in self.x_cols}

    def _step(self, x, P, pos, state, h):
        """One quarter ahead: carry the regime probabilities forward and average the regimes' forecasts.

        Args:
            x: regressors (dict). P: panel. pos: quarter position. state: regime probabilities. h: step.

        Returns:
            (forecast, new probabilities).
        """
        p = self.T @ state
        mean = self.c_orig + sum(self.b_orig[c] * x[c] for c in self.x_cols)
        return float(p @ mean), p

    def driver_rows(self):
        """Regime means, persistence and slopes, for the driver table.

        Returns:
            list of dict rows.
        """
        if self.failed:
            return [{"model": self.name, "term": "(one regime)", "reading": f"Markov fit failed: {self.fail_reason}"}]
        dur = 1 / np.clip(1 - np.diag(self.T), 1e-6, None)
        rows = [{"model": self.name, "term": "regimes", "coef": self.c_orig[1] - self.c_orig[0],
                 "reading": f"intercept low {self.c_orig[0]:.4f} / high {self.c_orig[1]:.4f}; expected duration "
                            f"{dur[0]:.0f} and {dur[1]:.0f} quarters; residual s.d. {self.sigma[0]:.4f}"}]
        for c in self.x_cols:
            if not c.startswith("D_"):
                b = self.b_orig[c]
                rows.append({"model": self.name, "term": c, "coef": b[1],
                             "reading": "same in both regimes" if b[0] == b[1] else f"low {b[0]:.4f}, high {b[1]:.4f}"})
        return rows

    def regime_path(self, P):
        """Smoothed probability of the high-growth regime over the estimation sample.

        Args:
            P: panel.

        Returns:
            Series.
        """
        if self.failed or self.res is None:
            return pd.Series(np.nan, index=P.index)
        sm_ = np.asarray(self.res.smoothed_marginal_probabilities)
        sm_ = sm_ if sm_.ndim == 2 and sm_.shape[1] == 2 else sm_.T
        return pd.Series(sm_[:, self.order[1]], index=self.fit_index[-len(sm_):]).reindex(P.index)


# ------------------------------------------------------------------------------------------------- TVP
class TimeVaryingARDL(_Regime):
    """ARDL whose coefficients follow random walks, estimated with the Kalman filter.

    The signal-to-noise ratio q (how fast the coefficients may drift) is chosen by maximum likelihood on a grid;
    the dummies' coefficients are held fixed. Forecasts use the latest filtered coefficients.
    """
    regime_label = "coefficient on the first driver"


    QS = (0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)

    def __init__(self, name, features, note=""):
        """Create a time-varying-parameter model.

        Args:
            name: display name. features: driver features. note: free text.
        """
        super().__init__(name, features, note)

    def _kalman(self, X, y, q, keep_path=False):
        """Kalman filter for y_t = X_t b_t + e_t, b_t = b_t-1 + w_t, with Var(w) = q * s2 * diag(scale).

        Args:
            X: regressors (standardised). y: target. q: drift ratio. keep_path: return the filtered coefficients.

        Returns:
            (concentrated log-likelihood, last coefficients, path or None, s2).
        """
        n, k = X.shape
        b = np.zeros(k)
        Pm = np.eye(k) * 1e4                                      # diffuse start
        Q = np.diag(self.drift) * q
        v2, ll, path = [], 0.0, []
        for t in range(n):
            Pm = Pm + Q
            x = X[t]
            f = x @ Pm @ x + 1.0
            v = y[t] - x @ b
            K = Pm @ x / f
            b = b + K * v
            Pm = Pm - np.outer(K, x @ Pm)
            if t >= k:                                            # skip the diffuse start
                v2.append(v * v / f)
                ll -= 0.5 * np.log(f)
            if keep_path:
                path.append(b.copy())
        s2 = float(np.mean(v2)) if v2 else 1.0
        ll -= 0.5 * len(v2) * np.log(s2)
        return ll, b, (np.array(path) if keep_path else None), s2

    def fit(self, P, start, end):
        """Part of the model interface (see src/models/zoo.py): choose q by likelihood, then filter.

        Args:
            P: panel. start, end: estimation window.

        Returns:
            self.
        """
        S, D, y, _, idx = self._window(P, start, end)
        X = np.column_stack([S, D])
        self.sc = np.where(X.std(axis=0) > 0, X.std(axis=0), 1.0)
        self.sc[0] = 1.0
        Xs = X / self.sc
        k_s = S.shape[1]
        self.drift = np.r_[np.ones(k_s), np.zeros(D.shape[1])]
        lls = {q: self._kalman(Xs, y, q)[0] for q in self.QS}
        self.q = max(lls, key=lls.get)
        _, b, path, _ = self._kalman(Xs, y, self.q, keep_path=True)
        self.b = (b / self.sc)[:k_s]
        self.path = pd.DataFrame(path[:, :k_s] / self.sc[:k_s], index=idx, columns=self.s_cols)
        self.params = pd.concat([self._named(self.b), pd.Series({"q": self.q})])
        return self

    def _step(self, x, P, pos, state, h):
        """One quarter ahead with the latest coefficients.

        Args:
            x: regressors. P: panel. pos: quarter position. state: unused. h: step.

        Returns:
            (forecast, state).
        """
        return float(x @ self.b), state

    def driver_rows(self):
        """Each coefficient at the start of the sample and now, for the driver table.

        Returns:
            list of dict rows.
        """
        first = self.path.iloc[min(len(self.path) - 1, 3 * len(self.s_cols))]
        rows = [{"model": self.name, "term": "drift ratio q", "coef": self.q,
                 "reading": "0 = constant coefficients (no drift found)" if self.q == 0 else "coefficients drift over time"}]
        for c in self.s_cols:
            if c != "const":
                rows.append({"model": self.name, "term": c, "coef": self.b[self.s_cols.index(c)],
                             "reading": f"{first.name}: {first[c]:.4f}, now: {self.path[c].iloc[-1]:.4f}"})
        return rows

    def regime_path(self, P):
        """Filtered coefficient of the first driver (how the main relation has drifted).

        Args:
            P: panel.

        Returns:
            Series.
        """
        c = self.features[0] if self.features else "y_lag1"
        return self.path[c].reindex(P.index)


# ------------------------------------------------------------------------------------------------- breaks
class BreakAdaptiveARDL(_Regime):
    """ARDL estimated only on the data after the latest significant structural break (sup-F test)."""
    regime_label = "1 = after the detected break"


    def __init__(self, name, features, note="", trim=0.15, min_obs=24):
        """Create a break-adaptive model.

        Args:
            name: display name. features: driver features. note: free text.
            trim: share of the sample excluded at each end when searching for the break.
            min_obs: fewest quarters to estimate on after a break.
        """
        super().__init__(name, features, note)
        self.trim, self.min_obs = trim, min_obs
        self._extra_spec = {"trim": trim, "min_obs": min_obs}

    def fit(self, P, start, end):
        """Part of the model interface (see src/models/zoo.py): sup-F (Quandt-Andrews) test for one break in
        all coefficients; if significant at 5% and enough data follow it, re-estimate after the break.

        Args:
            P: panel. start, end: estimation window.

        Returns:
            self.
        """
        S, D, y, _, idx = self._window(P, start, end)
        n, k = S.shape
        X = np.column_stack([S, D])
        b_all, ssr0 = _ols(X, y)
        lo, hi = max(int(self.trim * n), k + 2), min(int((1 - self.trim) * n), n - k - 2)
        F = np.full(n, np.nan)
        for t in range(lo, hi):
            pre = np.arange(n) < t
            Xb = np.column_stack([S * pre[:, None], S * (~pre)[:, None], D])
            _, ssr = _ols(Xb, y)
            F[t] = (ssr0 - ssr) / k / (ssr / max(n - 2 * k - D.shape[1], 1))
        self.supF = float(np.nanmax(F)) if np.isfinite(F).any() else 0.0
        self.cv = SUP_F_5PCT.get(k, 1.6 * k + 7)
        self.break_at = None
        b = b_all
        if self.supF > self.cv:
            t = int(np.nanargmax(F))
            if n - t >= max(self.min_obs, 3 * k):
                self.break_at = idx[t]
                post = np.arange(n) >= t
                Dp = D[post][:, D[post].std(axis=0) > 0] if D.shape[1] else D[post]
                b, _ = _ols(np.column_stack([S[post], Dp]), y[post])
        self.b = b[:k]
        self.params = self._named(self.b)
        return self

    def _step(self, x, P, pos, state, h):
        """One quarter ahead.

        Args:
            x: regressors. P: panel. pos: quarter position. state: unused. h: step.

        Returns:
            (forecast, state).
        """
        return float(x @ self.b), state

    def driver_rows(self):
        """Break test result and coefficients, for the driver table.

        Returns:
            list of dict rows.
        """
        rows = [{"model": self.name, "term": "structural break", "coef": self.supF,
                 "reading": (f"break in {self.break_at}: estimated on the data after it" if self.break_at is not None
                             else "no significant break (or too little data after it): full sample")
                            + f"; sup-F {self.supF:.1f} vs 5% critical value {self.cv:.1f}"}]
        rows += [{"model": self.name, "term": c, "coef": v, "reading": ""} for c, v in zip(self.s_cols, self.b) if c != "const"]
        return rows

    def regime_path(self, P):
        """1 after the detected break, 0 before (all 0 if none).

        Args:
            P: panel.

        Returns:
            Series.
        """
        s = pd.Series(0.0, index=P.index)
        if self.break_at is not None:
            s[s.index >= self.break_at] = 1.0
        return s


LABELS = {"markov": "Markov switching (Hamilton)", "threshold": "Threshold ARDL", "smooth_transition": "Smooth transition (LSTAR)",
          "tvp": "Time-varying parameters (Kalman)", "break_adaptive": "Break-adaptive ARDL"}


def build(kind, features, threshold, settings=None):
    """Create one regime-switching model.

    Args:
        kind: markov, threshold, smooth_transition, tvp or break_adaptive. features: driver features.
        threshold: regime variable for threshold and smooth_transition. settings: extra keyword arguments.

    Returns:
        model.
    """
    s = dict(settings or {})
    name = LABELS[kind] + (f" on {threshold.split('|')[0]}" if kind in ("threshold", "smooth_transition") else "")
    if kind == "markov":
        return MarkovSwitchingARDL(name, features, **s)
    if kind == "threshold":
        return ThresholdARDL(name, features, threshold, **s)
    if kind == "smooth_transition":
        return SmoothTransitionARDL(name, features, threshold, **s)
    if kind == "tvp":
        return TimeVaryingARDL(name, features, **s)
    if kind == "break_adaptive":
        return BreakAdaptiveARDL(name, features, **s)
    raise ValueError(f"unknown regime model {kind}")
