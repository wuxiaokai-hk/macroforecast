"""Auto-ARDL models, as in EViews' ARDL estimator: the lag lengths are chosen by an information criterion.

AutoARDL (growth form)
    g_t = c + sum_{i=1..p} rho_i g_{t-i} + sum_j sum_{k=0..q_j} b_jk x_{j,t-k} + dummies + e_t
    g = quarterly log growth of the organic target level; x_j = each driver in its stationary form (the literature
    core, or the drivers chosen by screen first). p in 1..max_p and every q_j in 0..max_q are chosen together by AIC
    or BIC on one common sample (the same rule as EViews), and chosen again at every re-estimation, so the backtest
    and the out-of-sample test only use lags that could have been chosen at the time.

LevelsARDL (levels form, with the bounds test)
    y*_t = c + sum_{i=1..p} a_i y*_{t-i} + sum_j sum_{k=0..q_j} b_jk z_{j,t-k} + dummies + e_t
    y* = ln(real organic target) = ln(L*/P); z_j = the long-run variables of the literature error-correction model
    (log house prices, log income, real mortgage rate, ...). Written as an error-correction model it gives the
    long-run coefficients theta_j = sum_k b_jk / (1 - sum_i a_i) and the adjustment speed -(1 - sum_i a_i).
    The Pesaran, Shin & Smith (2001) bounds test checks whether a long-run relation exists at all: F (all lagged
    levels zero) and t (lagged y* zero), case III (unrestricted constant, no trend), against the I(0) and I(1)
    critical bounds. F above the I(1) bound: a long-run relation; below the I(0) bound: none; in between: inconclusive.
    Forecasts: y* is iterated forward and target growth is g_t = y*_t - y*_{t-1} + ln(P_t / P_{t-1}).
"""
import itertools

import numpy as np
import pandas as pd
import statsmodels.api as sm

from ..features.transforms import parse_feature
from ..modelling.ardl import ARDL, make_panel
from .ecm import long_run_column
from ..utils import target as tg

FAMILY = "Auto ARDL"


def _ic(ssr, n, k, ic):
    """Information criterion of an OLS fit (EViews' per-observation form; only the ranking matters).

    Args:
        ssr: sum of squared residuals. n: observations. k: estimated coefficients. ic: "aic", "bic" or "hq".
    """
    ll = -n / 2 * (1 + np.log(2 * np.pi) + np.log(ssr / n))
    pen = {"aic": 2 * k, "bic": k * np.log(n), "hq": 2 * k * np.log(np.log(n))}[ic]
    return (-2 * ll + pen) / n


def select_orders(y, X, fixed, max_p, max_q, ic="bic", min_p=1, max_models=20000):
    """Choose the lag of y and of every regressor together by an information criterion, EViews style.

    All candidate models are estimated on the same sample (the one the longest lags allow), so their criteria are
    comparable.

    Args:
        y: Series, the dependent variable. X: DataFrame of regressors (lag 0). fixed: DataFrame always included
        (dummies) or None. max_p: most lags of y. max_q: most lags of each regressor (0 = same quarter only).
        ic: "aic", "bic" or "hq". min_p: fewest lags of y. max_models: cap on the grid (max_q is lowered to fit).

    Returns:
        dict with p, q (dict regressor -> lag), ic value, models tried.
    """
    cols = list(X.columns)
    while max_q > 0 and (max_p - min_p + 1) * (max_q + 1) ** len(cols) > max_models:
        max_q -= 1
    L = max(max_p, max_q)
    D = pd.DataFrame({"y": y})
    for i in range(1, max_p + 1):
        D[f"y.L{i}"] = y.shift(i)
    for c in cols:
        for k in range(0, max_q + 1):
            D[f"{c}.L{k}"] = X[c].shift(k)
    if fixed is not None and fixed.shape[1]:
        D = D.join(fixed)
    D = D.iloc[L:].dropna()
    n = len(D)
    yv = D["y"].values
    base = [np.ones(n)] + ([D[c].values for c in fixed.columns if D[c].std() > 0] if fixed is not None else [])
    best = None
    tried = 0
    for p in range(min_p, max_p + 1):
        for qs in itertools.product(range(0, max_q + 1), repeat=len(cols)):
            M = np.column_stack(base + [D[f"y.L{i}"].values for i in range(1, p + 1)]
                                + [D[f"{c}.L{k}"].values for c, q in zip(cols, qs) for k in range(0, q + 1)])
            b, *_ = np.linalg.lstsq(M, yv, rcond=None)
            e = yv - M @ b
            v = _ic(float(e @ e), n, M.shape[1], ic)
            tried += 1
            if best is None or v < best[0]:
                best = (v, p, dict(zip(cols, qs)))
    return {"p": best[1], "q": best[2], "ic": best[0], "models_tried": tried, "n_obs": n}


class AutoARDL(ARDL):
    """ARDL in growth form whose lags are chosen by AIC/BIC at every estimation (EViews' automatic selection)."""
    kind = "AutoARDL"

    def __init__(self, name, base, max_p=4, max_q=4, ic="bic", family=FAMILY, note="", signs=None):
        """Create the model (nothing is estimated yet).

        Args:
            name: display name. base: driver features; their lag is ignored (lags 0..max_q are candidates).
            max_p: most lags of target growth. max_q: most lags of each driver. ic: "aic", "bic" or "hq".
            family, note: as for ARDL. signs: expected sign per base variable "code|transform" (optional).
        """
        self.bases = list(dict.fromkeys(f.rsplit("|", 1)[0] for f in base))
        self.max_p, self.max_q, self.ic = int(max_p), int(max_q), str(ic).lower()
        self.cands = [f"{b}|L{k}" for b in self.bases for k in range(0, self.max_q + 1)]
        super().__init__(name, [f"{b}|L0" for b in self.bases], ar=True, family=family, note=note)
        self.p, self.q, self.selection = 1, {b: 0 for b in self.bases}, None
        self.base_signs = dict(signs or {})
        self.frozen = False

    def panel(self, ctx, mdf):
        """Part of the model interface (see src/models/zoo.py): target, dummies and every candidate lag."""
        return make_panel(ctx, mdf, self.cands)

    def spec(self):
        """Part of the model interface (see src/models/zoo.py): settings that define the model (for the cache)."""
        return {"class": "AutoARDL", "bases": self.bases, "max_p": self.max_p, "max_q": self.max_q, "ic": self.ic}

    def drivers(self):
        """Part of the model interface (see src/models/zoo.py): the drivers with their chosen lags."""
        return list(self.features)

    def _X(self, P):
        """Constant, the chosen lags of target growth, the chosen driver lags and the dummies."""
        X = pd.DataFrame({"const": 1.0}, index=P.index)
        for i in range(1, self.p + 1):
            X[f"y_lag{i}"] = P["y"].shift(i)
        for f in self.features:
            X[f] = P[f]
        for c in [c for c in P.columns if c.startswith("D_")]:
            X[c] = P[c]
        return X

    def fit(self, P, start, end, fast=False):
        """Choose the lags on start..end (unless frozen), then estimate by OLS with HAC standard errors.

        Args:
            P: panel. start, end: estimation window. fast: plain least squares (inside combinations).

        Returns:
            self.
        """
        if not self.frozen:
            W = P.loc[start:end]
            X0 = pd.DataFrame({b: W[f"{b}|L0"] for b in self.bases})
            D = W[[c for c in W.columns if c.startswith("D_")]]
            sel = select_orders(W["y"], X0, D, self.max_p, self.max_q, self.ic)
            self.p, self.q, self.selection = sel["p"], sel["q"], sel
            self.features = [f"{b}|L{k}" for b in self.bases for k in range(0, self.q[b] + 1)]
            self.signs = {f: self.base_signs[f.rsplit("|", 1)[0]] for f in self.features
                          if f.rsplit("|", 1)[0] in self.base_signs}
        return super().fit(P, start, end, fast=fast)

    def forecast(self, P, origin, H):
        """Part of the model interface (see src/models/zoo.py): iterate the equation, own forecasts replacing
        lagged growth.

        Returns:
            array of up to H forecasts of quarterly log growth.
        """
        pos = P.index.get_loc(origin)
        H = min(H, len(P) - pos - 1)
        past = list(P["y"].iloc[max(0, pos - self.p + 1):pos + 1].values)    # g(origin-p+1) .. g(origin)
        out = np.zeros(H)
        for h in range(1, H + 1):
            row = P.iloc[pos + h]
            v = self.params.get("const", 0.0)
            for i in range(1, self.p + 1):
                v += self.params.get(f"y_lag{i}", 0.0) * past[-i]
            v += sum(self.params[f] * row[f] for f in self.features if f in self.params)
            out[h - 1] = v
            past.append(v)
        return out

    def long_run(self):
        """Long-run effect of each driver: sum of its lag coefficients / (1 - sum of the growth lags).

        Returns:
            dict base variable -> long-run effect (nan if the growth lags sum to 1 or more).
        """
        rho = sum(self.params.get(f"y_lag{i}", 0.0) for i in range(1, self.p + 1))
        out = {}
        for b in self.bases:
            s = sum(self.params.get(f"{b}|L{k}", 0.0) for k in range(0, self.q.get(b, 0) + 1))
            out[b] = s / (1 - rho) if rho < 1 else np.nan
        return out

    def lag_summary(self):
        """The chosen lags in EViews notation, e.g. ARDL(2, 1, 0, 3).

        Returns:
            str.
        """
        return f"ARDL({self.p}, " + ", ".join(str(self.q[b]) for b in self.bases) + f"), chosen by {self.ic.upper()}"


class LevelsARDL:
    """ARDL in levels of the real organic target and the long-run variables, with lags chosen by AIC/BIC and the
    Pesaran-Shin-Smith bounds test. Same interface as the other models."""
    kind = "LevelsARDL"

    def __init__(self, name, long_run, deflator, max_p=4, max_q=4, ic="bic", family=FAMILY, note=""):
        """Create the model.

        Args:
            name: display name. long_run: list of {mnemonic, form} (forms as in the error-correction model: log,
            log_real, level, real_rate). deflator: price index mnemonic. max_p, max_q, ic: lag search.
        """
        self.name, self.family, self.note = name, family, note
        self.long_run, self.deflator = list(long_run), deflator
        self.lr_names = [f"lr:{i['mnemonic']}:{i.get('form', 'log')}" for i in self.long_run]
        self.max_p, self.max_q, self.ic = int(max_p), int(max_q), str(ic).lower()
        self.p, self.q, self.params, self.bounds, self.res = 1, {n: 0 for n in self.lr_names}, None, None, None
        self.frozen = False
        self.features = []

    def panel(self, ctx, mdf):
        """Part of the model interface (see src/models/zoo.py): target, dummies, y* = ln(L*/P), the long-run levels."""
        P = make_panel(ctx, mdf, [])
        P["__logL"] = ctx.logL.reindex(P.index)
        P["__lcpi"] = tg.target_deflator(mdf, self.deflator, P.index)
        P["__ys"] = P["__logL"] - P["__lcpi"]
        for item, n in zip(self.long_run, self.lr_names):
            P[n] = long_run_column(mdf, item, self.deflator).reindex(P.index)
        return P.replace([np.inf, -np.inf], np.nan)             # log of a zero or negative value: missing

    def spec(self):
        """Part of the model interface (see src/models/zoo.py)."""
        return {"class": "LevelsARDL", "long_run": self.long_run, "deflator": self.deflator, "max_p": self.max_p,
                "max_q": self.max_q, "ic": self.ic}

    def drivers(self):
        """Part of the model interface (see src/models/zoo.py): the long-run variables with their chosen lags."""
        return [f"{n[3:]} (lags 0-{self.q.get(n, 0)})" for n in self.lr_names]

    def _design(self, P):
        """Regressors: constant, lags of y*, lags of each long-run variable, dummies."""
        X = pd.DataFrame({"const": 1.0}, index=P.index)
        for i in range(1, self.p + 1):
            X[f"ys_lag{i}"] = P["__ys"].shift(i)
        for n in self.lr_names:
            for k in range(0, self.q[n] + 1):
                X[f"{n}|L{k}"] = P[n].shift(k)
        for c in [c for c in P.columns if c.startswith("D_")]:
            X[c] = P[c]
        return X

    def fit(self, P, start, end, fast=False):
        """Choose the lags (unless frozen), estimate by OLS (HAC), and run the bounds test.

        Args:
            P: panel. start, end: estimation window. fast: skip the bounds test.

        Returns:
            self.
        """
        W = P.loc[start:end]
        if not self.frozen:
            X0 = W[self.lr_names]
            D = W[[c for c in W.columns if c.startswith("D_")]]
            sel = select_orders(W["__ys"], X0, D, self.max_p, self.max_q, self.ic)
            self.p, self.q, self.selection = sel["p"], sel["q"], sel
        X = self._design(P).loc[start:end]
        yv = W["__ys"]
        X = X.loc[:, (X.std() > 0) | (X.columns == "const")]
        ok = X.notna().all(axis=1) & yv.notna()
        self.res = sm.OLS(yv[ok], X[ok]).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
        self.params = self.res.params
        if not fast:
            self.bounds = bounds_test(W, self.lr_names, self.p, self.q)
        return self

    def forecast(self, P, origin, H):
        """Part of the model interface (see src/models/zoo.py): iterate y* forward; growth = change in y* plus
        inflation.

        Returns:
            array of up to H forecasts of quarterly log growth of the organic balance.
        """
        pos = P.index.get_loc(origin)
        H = min(H, len(P) - pos - 1)
        p = self.params
        ys = list(P["__ys"].iloc[max(0, pos - self.p + 1):pos + 1].values)
        out = np.zeros(H)
        for h in range(1, H + 1):
            t = pos + h
            v = p.get("const", 0.0) + sum(p.get(f"ys_lag{i}", 0.0) * ys[-i] for i in range(1, self.p + 1))
            for n in self.lr_names:
                for k in range(0, self.q[n] + 1):
                    v += p.get(f"{n}|L{k}", 0.0) * P[n].iloc[t - k]
            out[h - 1] = v - ys[-1] + (P["__lcpi"].iloc[t] - P["__lcpi"].iloc[t - 1])
            ys.append(v)
        return out

    def long_run_coefs(self):
        """Long-run coefficients theta_j = sum_k b_jk / (1 - sum_i a_i), and the adjustment speed.

        Returns:
            (dict long-run name -> theta, adjustment speed = -(1 - sum a_i), long-run constant).
        """
        p = self.params
        a = sum(p.get(f"ys_lag{i}", 0.0) for i in range(1, self.p + 1))
        th = {n: sum(p.get(f"{n}|L{k}", 0.0) for k in range(0, self.q[n] + 1)) / (1 - a) for n in self.lr_names}
        return th, -(1 - a), p.get("const", 0.0) / (1 - a)

    def lag_summary(self):
        """EViews notation of the chosen lags, e.g. ARDL(2, 0, 1, 1)."""
        return f"ARDL({self.p}, " + ", ".join(str(self.q[n]) for n in self.lr_names) + f"), chosen by {self.ic.upper()}"

    def detail_rows(self, ctx, P):
        """Rows for the model detail page: coefficients, long-run relation, adjustment speed and the bounds test."""
        rows = []
        r = self.res
        for t in r.params.index:
            if t.startswith("D_"):
                continue
            rows.append({"model": self.name, "term": t, "coef": float(r.params[t]), "se_hac": float(r.bse[t]),
                         "t": float(r.tvalues[t]), "p": float(r.pvalues[t]),
                         "reading": ("constant" if t == "const" else
                                     f"ln(real organic target), {t[-1]} quarter(s) earlier" if t.startswith("ys_lag") else
                                     f"{t.split(':')[1]} ({t.split(':')[2].split('|')[0]}), lag {t.rsplit('L', 1)[-1]}")})
        th, speed, c0 = self.long_run_coefs()
        rows.append({"model": self.name, "term": "lag orders", "reading": self.lag_summary()})
        rows.append({"model": self.name, "term": "adjustment speed", "coef": speed,
                     "reading": f"{abs(speed):.1%} of a gap from the long-run level closes each quarter"
                     if speed < 0 else "no adjustment towards a long-run level (speed is not negative)"})
        for n, v in th.items():
            rows.append({"model": self.name, "term": f"long run: {n[3:]}", "coef": float(v),
                         "reading": f"in the long run, 1 unit higher {n.split(':')[1]} ({n.split(':')[2]}) goes with "
                                    f"{v:+.3f} in ln(real target)"})
        b = self.bounds or {}
        if b:
            rows.append({"model": self.name, "term": "bounds test: F", "coef": b["F"],
                         "reading": f"5% bounds I(0) {b['F_I0_5']:.2f}, I(1) {b['F_I1_5']:.2f}; p-value "
                                    f"{b['F_p_I0']:.3f} (I(0)) / {b['F_p_I1']:.3f} (I(1)): {b['decision']}"})
            rows.append({"model": self.name, "term": "bounds test: t", "coef": b["t"],
                         "reading": f"5% bounds I(0) {b['t_I0_5']:.2f}, I(1) {b['t_I1_5']:.2f} (more negative than the "
                                    "I(1) bound: long-run relation)"})
        return rows


# Pesaran, Shin & Smith (2001), Table CII(iii): t-statistic bounds, case III, by number of regressors k
PSS_T = {1: (-2.57, -2.91, -2.86, -3.22, -3.43, -3.82), 2: (-2.57, -3.21, -2.86, -3.53, -3.43, -4.10),
         3: (-2.57, -3.46, -2.86, -3.78, -3.43, -4.37), 4: (-2.57, -3.66, -2.86, -3.99, -3.43, -4.60),
         5: (-2.57, -3.86, -2.86, -4.19, -3.43, -4.79), 6: (-2.57, -4.04, -2.86, -4.38, -3.43, -4.99),
         7: (-2.57, -4.23, -2.86, -4.57, -3.43, -5.19), 8: (-2.57, -4.40, -2.86, -4.72, -3.43, -5.37),
         9: (-2.57, -4.56, -2.86, -4.88, -3.43, -5.54), 10: (-2.57, -4.69, -2.86, -5.03, -3.43, -5.68)}


def uecm_design(W, lr_names, p, q):
    """The conditional error-correction form of ARDL(p, q): the same model as the levels ARDL, rewritten.

    dy*_t = c + pi_y y*_{t-1} + sum_j pi_j z_{j,t-1} + sum_{i=1..p-1} phi_i dy*_{t-i}
            + sum_j sum_{k=0..max(q_j-1,0)} theta_jk dz_{j,t-k} + dummies

    Args:
        W: panel window (with __ys, the long-run columns and the dummies). lr_names: long-run columns.
        p, q: lag orders of the levels ARDL.

    Returns:
        (dependent series, regressor DataFrame) with complete rows only.
    """
    ys = W["__ys"]
    X = pd.DataFrame({"const": 1.0, "ys.L1": ys.shift(1)}, index=W.index)
    for n in lr_names:
        X[f"{n}.L1"] = W[n].shift(1)
    for i in range(1, p):
        X[f"D.ys.L{i}"] = ys.diff().shift(i)
    for n in lr_names:
        for k in range(0, max(q.get(n, 0) - 1, 0) + 1):
            X[f"D.{n}.L{k}"] = W[n].diff().shift(k)
    for c in [c for c in W.columns if c.startswith("D_") and W[c].std() > 0]:
        X[c] = W[c]
    d = pd.concat([ys.diff().rename("dy"), X], axis=1).dropna()
    return d["dy"], d.drop(columns="dy")


def bounds_test(W, lr_names, p, q):
    """Pesaran, Shin & Smith (2001) bounds test for a long-run relation, case III (unrestricted constant, no trend),
    as in EViews: F-test that all lagged levels are zero, and t-test on lagged y*, in the conditional
    error-correction form of the chosen ARDL (dummies included).

    Critical values: F from PSS Table CI(iii) (statsmodels' copy, by number of regressors k), with asymptotic
    p-values; t from PSS Table CII(iii).

    Args:
        W: estimation window of the panel. lr_names: long-run columns. p, q: chosen lags.

    Returns:
        dict: F, t, the 10/5/1% I(0) and I(1) bounds of each, F p-values, decision, k, n_obs.
    """
    from statsmodels.tsa.ardl import pss_critical_values as pss
    k = len(lr_names)
    try:
        y, X = uecm_design(W, lr_names, max(int(p), 1), q)
        r = sm.OLS(y, X).fit()
        lv = ["ys.L1"] + [f"{n}.L1" for n in lr_names]
        R = np.zeros((len(lv), X.shape[1]))
        for i, c in enumerate(lv):
            R[i, list(X.columns).index(c)] = 1
        F = float(np.squeeze(r.f_test(R).fvalue))
        t = float(r.tvalues["ys.L1"])
        out = {"F": F, "t": t, "k": k, "n_obs": int(r.nobs)}
        lo, up = pss.crit_vals[(k, 3, False)], pss.crit_vals[(k, 3, True)]
        for j, lev in enumerate((10, 5, 1)):
            out[f"F_I0_{lev}"], out[f"F_I1_{lev}"] = float(lo[j]), float(up[j])
            tt = PSS_T.get(k)
            out[f"t_I0_{lev}"], out[f"t_I1_{lev}"] = (tt[2 * j], tt[2 * j + 1]) if tt else (np.nan, np.nan)
        try:
            from statsmodels.tsa.ardl.model import _pss_pvalue
            out["F_p_I0"], out["F_p_I1"] = float(_pss_pvalue(F, k, 3, False)), float(_pss_pvalue(F, k, 3, True))
        except Exception:
            out["F_p_I0"] = out["F_p_I1"] = np.nan
        out["decision"] = ("long-run relation (F above the 5% I(1) bound)" if F > out["F_I1_5"] else
                           "no long-run relation (F below the 5% I(0) bound)" if F < out["F_I0_5"] else
                           "inconclusive (F between the 5% bounds)")
        return out
    except Exception as ex:
        return {"F": np.nan, "t": np.nan, "k": k, "decision": f"bounds test failed: {type(ex).__name__}: {ex}",
                "F_p_I0": np.nan, "F_p_I1": np.nan, **{f"{s_}_{b_}_{lev}": np.nan for s_ in ("F", "t")
                                                       for b_ in ("I0", "I1") for lev in (10, 5, 1)}}


def base_of(feature):
    """'FHPI.ION|dlog|L1' -> 'FHPI.ION|dlog'."""
    code, tr, _ = parse_feature(feature)
    return f"{code}|{tr}"
