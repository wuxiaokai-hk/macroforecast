"""Panel estimators. All of them end as one equation per bank, so forecasting is the same for every model:

    y_i,t = a_i + rho_i * y_i,t-1 + beta_i' x_t + delta_i' d_t           (d = one-off dummies, 0 in forecasts)

    separate        a_i, rho_i, beta_i, delta_i estimated on bank i alone (no pooling)
    pooled          one rho and beta for all banks, a bank intercept a_i (fixed effects), common dummies;
                    standard errors Driscoll-Kraay (robust to the same shock hitting every bank in a quarter)
    partial         each bank's rho and beta pulled towards the pooled value by w = tau^2 / (tau^2 + se_i^2):
                    a bank with a short or noisy history borrows from the others, one with a clear history keeps
                    its own (empirical Bayes; tau^2 = how much banks genuinely differ, estimated from the data)
    topdown         the total of all banks from the drivers, times each bank's share:
                    y_i,t = y_total,t + s_i,t,  s_i,t = c_i + phi_i * s_i,t-1  (s = change in the bank's share)
    ar1             benchmark: a_i + rho_i * y_i,t-1 per bank, no drivers
"""
import numpy as np
import pandas as pd

KINDS = {"separate": "Separate (each bank alone)", "pooled": "Pooled FE", "partial": "Partial pooling",
         "topdown": "Top-down (total x share)", "ar1": "AR(1) benchmark"}


def ols(X, y):
    """Least squares: (coefficients, residuals, (X'X)^-1 or None when singular)."""
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    try:
        XtXi = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        XtXi = np.linalg.pinv(X.T @ X)
    return b, e, XtXi


def driscoll_kraay(X, e, T, XtXi, lags=4):
    """Driscoll-Kraay covariance: scores summed over banks in each quarter, then Newey-West (Bartlett) over
    quarters. Robust to heteroskedasticity, autocorrelation and correlation across banks in the same quarter.
    With one bank it is the Newey-West covariance.

    Args:
        X: stacked regressors. e: residuals. T: quarter of each row (int). XtXi: (X'X)^-1. lags: Bartlett lags.

    Returns:
        covariance matrix of the coefficients.
    """
    ut = np.unique(T)
    G = np.zeros((len(ut), X.shape[1]))
    idx = np.searchsorted(ut, T)
    np.add.at(G, idx, X * e[:, None])
    # quarters may have gaps (unbalanced): lags are by quarter position
    S = G.T @ G
    where = {u: k for k, u in enumerate(ut)}
    for l in range(1, lags + 1):
        j = np.array([where.get(u - l, -1) for u in ut])
        k = j >= 0
        if k.any():
            A = G[k].T @ G[j[k]]
            S += (1 - l / (lags + 1)) * (A + A.T)
    n, p = X.shape
    G_ = len(ut)                                  # small-sample correction as statsmodels (hac-groupsum)
    return XtXi @ S @ XtXi * ((n - 1) / max(n - p, 1)) * (G_ / max(G_ - 1, 1))


def stacked(pn, X):
    """Feature values for the stacked bank-quarter rows. X: (dates, k) when every bank sees the same drivers
    (macro data), or (banks, dates, k) when some drivers are bank-specific (e.g. a bank's own rate in a system)."""
    if X.ndim == 3:
        return X[pn.B, pn.T]
    return X[pn.T] if X.shape[1] else np.zeros((len(pn.T), 0))


def of_bank(pn, X, bank):
    """One bank's (dates, k) feature values."""
    return X[pn.code[bank]] if X.ndim == 3 else X


class PanelModel:
    """One panel model: kind, features, and after fit() the per-bank equations.

    Attributes after fit: coef[bank] = dict(a, rho, beta (array), delta (array)); se[bank] (separate/partial);
    pooled_table (pooled/partial/topdown: coefficient, Driscoll-Kraay se, t); weights (partial: shrinkage
    weight per bank and coefficient); share (topdown: c_i, phi_i); total (topdown: the total's equation).
    """

    def __init__(self, kind, features, name=None, min_obs=20, dk_lags=4):
        self.kind, self.features = kind, list(features)
        self.name = name or KINDS[kind]
        self.min_obs, self.dk_lags = int(min_obs), int(dk_lags)
        self.coef, self.se, self.weights, self.pooled_table, self.share, self.total = {}, {}, None, None, {}, None
        self.n_obs = {}

    # --------------------------------------------------------------------------------------------- estimation
    def fit(self, pn, X, start, end):
        """Estimate on the quarters start..end (labels) of panel pn. X: feature values on pn.dates (array)."""
        Xr = stacked(pn, X)
        m = pn.window(start, end, Xr)
        k = 0 if self.kind == "ar1" else X.shape[-1]
        self.coef, self.se, self.n_obs = {}, {}, {}
        if self.kind in ("separate", "ar1"):
            for b in pn.banks:
                r = m & (pn.B == pn.code[b])
                if r.sum() < max(self.min_obs, k + 4):
                    continue
                self.coef[b], self.se[b] = self._own(pn, r, Xr if k else Xr[:, :0])
                self.n_obs[b] = int(r.sum())
        elif self.kind == "pooled":
            self._pooled(pn, m, Xr)
        elif self.kind == "partial":
            self._pooled(pn, m, Xr)
            self._shrink(pn, m, Xr)
        elif self.kind == "topdown":
            if X.ndim == 3:
                raise ValueError("top-down needs drivers common to all banks")
            self._topdown(pn, X, m, start, end)
        return self

    def _own(self, pn, r, Xr):
        """One bank's own regression: [1, y_lag, x, dummies that vary in its window]."""
        D = pn.Dm[pn.T[r]]
        dk = D.std(axis=0) > 0 if D.shape[1] else np.zeros(0, bool)
        Z = np.column_stack([np.ones(r.sum()), pn.YL[r], Xr[r], D[:, dk]])
        b, e, XtXi = ols(Z, pn.Y[r])
        V = driscoll_kraay(Z, e, pn.T[r], XtXi, self.dk_lags)
        k = Xr.shape[1]
        delta = np.zeros(pn.Dm.shape[1])
        delta[dk] = b[2 + k:]
        se = np.sqrt(np.clip(np.diag(V), 0, None))
        return ({"a": b[0], "rho": b[1], "beta": b[2:2 + k], "delta": delta},
                {"a": se[0], "rho": se[1], "beta": se[2:2 + k]})

    def _pooled(self, pn, m, Xr):
        """Fixed effects: a bank intercept for every bank with enough rows, common rho, beta, dummies."""
        ok = [b for b in pn.banks if (m & (pn.B == pn.code[b])).sum() >= max(4, min(self.min_obs, 8))]
        if not ok:
            raise ValueError("no bank has enough quarters in the estimation window")
        m = m & np.isin(pn.B, [pn.code[b] for b in ok])
        Bm = pn.B[m]
        F = np.column_stack([(Bm == pn.code[b]).astype(float) for b in ok])
        D = pn.Dm[pn.T[m]]
        dk = D.std(axis=0) > 0 if D.shape[1] else np.zeros(0, bool)
        k = Xr.shape[1]
        Z = np.column_stack([F, pn.YL[m], Xr[m], D[:, dk]])
        b, e, XtXi = ols(Z, pn.Y[m])
        V = driscoll_kraay(Z, e, pn.T[m], XtXi, self.dk_lags)
        nb = len(ok)
        rho, beta = b[nb], b[nb + 1:nb + 1 + k]
        delta = np.zeros(pn.Dm.shape[1])
        delta[dk] = b[nb + 1 + k:]
        se = np.sqrt(np.clip(np.diag(V), 0, None))
        names = ["rho (lagged change)"] + self.features + [d for d, f in zip(pn.dummy_names, dk) if f]
        self.pooled_table = pd.DataFrame({"term": names, "coef": b[nb:], "std_err_dk": se[nb:],
                                          "t_dk": b[nb:] / np.where(se[nb:] > 0, se[nb:], np.nan)})
        self.pooled = {"rho": rho, "beta": beta, "delta": delta, "se_rho": se[nb], "se_beta": se[nb + 1:nb + 1 + k]}
        self.n_pooled = int(m.sum())
        for j, bk in enumerate(ok):
            self.coef[bk] = {"a": b[j], "rho": rho, "beta": beta.copy(), "delta": delta.copy()}
            self.n_obs[bk] = int((m & (pn.B == pn.code[bk])).sum())

    def _shrink(self, pn, m, Xr):
        """Empirical-Bayes partial pooling of rho and beta towards the pooled values (method of moments).

        For each coefficient: tau^2 = max(0, mean_i (b_i - b_pooled)^2 - mean_i se_i^2); bank weight
        w_i = tau^2 / (tau^2 + se_i^2). The intercept and dummies are then re-estimated for each bank with the
        shrunk slopes held fixed, so the bank's average level of growth stays its own.
        """
        own = {}
        for b in self.coef:
            r = m & (pn.B == pn.code[b])
            if r.sum() >= max(self.min_obs, Xr.shape[1] + 4):
                own[b] = self._own(pn, r, Xr)
        center = np.r_[self.pooled["rho"], self.pooled["beta"]]
        names = ["rho"] + self.features
        if len(own) >= 2:
            Bv = np.array([np.r_[c["rho"], c["beta"]] for c, _ in own.values()])
            Sv = np.array([np.r_[s["rho"], s["beta"]] for _, s in own.values()])
            tau2 = np.clip(np.mean((Bv - center) ** 2, axis=0) - np.mean(Sv ** 2, axis=0), 0, None)
        else:
            tau2 = np.zeros(len(center))
        rows = []
        for b in list(self.coef):
            if b in own:
                c, s = own[b]
                bi, si = np.r_[c["rho"], c["beta"]], np.r_[s["rho"], s["beta"]]
                w = np.where(tau2 + si ** 2 > 0, tau2 / (tau2 + si ** 2), 0.0)
            else:                                     # too few quarters for its own regression: fully pooled
                bi, si, w = center.copy(), np.full(len(center), np.nan), np.zeros(len(center))
            sh = w * bi + (1 - w) * center
            r = m & (pn.B == pn.code[b])
            D = pn.Dm[pn.T[r]]
            dk = D.std(axis=0) > 0 if D.shape[1] else np.zeros(0, bool)
            resid = pn.Y[r] - sh[0] * pn.YL[r] - (Xr[r] @ sh[1:] if Xr.shape[1] else 0)
            Z = np.column_stack([np.ones(r.sum()), D[:, dk]])
            g, *_ = np.linalg.lstsq(Z, resid, rcond=None)
            delta = np.zeros(pn.Dm.shape[1])
            delta[dk] = g[1:]
            self.coef[b] = {"a": g[0], "rho": sh[0], "beta": sh[1:], "delta": delta}
            self.se[b] = {"rho": si[0], "beta": si[1:]}
            for j, nm in enumerate(names):
                rows.append({"bank": b, "term": nm, "own": bi[j], "own_se": si[j], "pooled": center[j],
                             "tau2": tau2[j], "weight_on_own": w[j], "shrunk": sh[j]})
        self.weights = pd.DataFrame(rows)

    def _topdown(self, pn, X, m, start, end):
        """The total from the drivers; each bank's share change as AR(1) around its own average."""
        base = pn.base
        a, z = pn.pos(start), pn.pos(end)
        y = base.y.values
        rows = [t for t in range(max(a, 1), z + 1) if not np.isnan(y[t]) and not np.isnan(y[t - 1])
                and (X.shape[1] == 0 or not np.isnan(X[t]).any())]
        rows = np.array(rows, int)
        D = pn.Dm[rows]
        dk = D.std(axis=0) > 0 if D.shape[1] else np.zeros(0, bool)
        Z = np.column_stack([np.ones(len(rows)), y[rows - 1], X[rows], D[:, dk]])
        b, e, XtXi = ols(Z, y[rows])
        V = driscoll_kraay(Z, e, rows, XtXi, self.dk_lags)
        k = X.shape[1]
        delta = np.zeros(pn.Dm.shape[1])
        delta[dk] = b[2 + k:]
        se = np.sqrt(np.clip(np.diag(V), 0, None))
        self.total = {"a": b[0], "rho": b[1], "beta": b[2:2 + k], "delta": delta}
        names = ["constant", "rho (lagged change)"] + self.features + [d for d, f in zip(pn.dummy_names, dk) if f]
        self.pooled_table = pd.DataFrame({"term": names, "coef": b, "std_err_dk": se,
                                          "t_dk": b / np.where(se > 0, se, np.nan)})
        self.share = {}
        for bk in pn.banks:
            r = m & (pn.B == pn.code[bk])
            if r.sum() < max(self.min_obs, 6):
                continue
            s = pn.Y[r] - y[pn.T[r]]                     # change in the bank's share of the total
            sl = pn.YL[r] - y[pn.T[r] - 1]
            ok = ~np.isnan(s) & ~np.isnan(sl)
            Dd = pn.Dm[pn.T[r]][ok]
            dk2 = Dd.std(axis=0) > 0 if Dd.shape[1] else np.zeros(0, bool)
            Z2 = np.column_stack([np.ones(ok.sum()), sl[ok], Dd[:, dk2]])
            g, *_ = np.linalg.lstsq(Z2, s[ok], rcond=None)
            phi = float(np.clip(g[1], -0.95, 0.95))
            self.share[bk] = {"c": float(g[0]), "phi": phi}
            self.coef[bk] = {"share": True}
            self.n_obs[bk] = int(ok.sum())

    # --------------------------------------------------------------------------------------------- forecasting
    def forecast(self, pn, X, bank, origin, H, y_hist=None):
        """Changes for quarters origin+1 .. origin+H for one bank (dummies 0).

        Args:
            pn: Panel. X: feature values on the dates of the macro path used (array, rows = pn.dates for history,
            or a scenario's quarters). bank: bank name. origin: position of the origin in X's rows.
            H: quarters. y_hist: dict with the bank's and the total's last actual change at the origin.

        Returns:
            array of H changes (shorter if X ends).
        """
        X = of_bank(pn, X, bank)
        H = min(H, X.shape[0] - origin - 1)
        if H <= 0 or bank not in self.coef:
            return np.zeros(0)
        out = np.zeros(H)
        if self.kind == "topdown":
            c, sh = self.total, self.share[bank]
            yt, s = y_hist["total"], y_hist["bank"] - y_hist["total"]
            for h in range(1, H + 1):
                xt = X[origin + h]
                yt = c["a"] + c["rho"] * yt + (xt @ c["beta"] if len(c["beta"]) else 0.0)
                s = sh["c"] + sh["phi"] * s
                out[h - 1] = yt + s
            return out
        c = self.coef[bank]
        yp = y_hist["bank"]
        for h in range(1, H + 1):
            xt = X[origin + h] if self.kind != "ar1" else np.zeros(0)
            yp = c["a"] + c["rho"] * yp + (xt @ c["beta"] if len(c["beta"]) else 0.0)
            out[h - 1] = yp
        return out

    def fitted(self, pn, X, bank, t):
        """In-sample one-step fit at position t of pn.dates (actual lag and dummies)."""
        X = of_bank(pn, X, bank)
        yb = pn.ys[bank]
        q = pn.dates[t]
        prev = yb.get(q - 1, np.nan)
        if self.kind == "topdown":
            c, sh = self.total, self.share.get(bank)
            if sh is None:
                return np.nan
            ytot = pn.base.y.values
            yt = c["a"] + c["rho"] * ytot[t - 1] + (X[t] @ c["beta"] if len(c["beta"]) else 0) + pn.Dm[t] @ c["delta"]
            s = sh["c"] + sh["phi"] * (prev - ytot[t - 1])
            return float(yt + s)
        c = self.coef.get(bank)
        if c is None:
            return np.nan
        xt = X[t] if self.kind != "ar1" else np.zeros(0)
        return float(c["a"] + c["rho"] * prev + (xt @ c["beta"] if len(c["beta"]) else 0) + pn.Dm[t] @ c["delta"])

    def coefficient_rows(self):
        """Every bank's equation as rows: model, bank, term, coef (and std_err where estimated)."""
        rows = []
        if self.kind == "topdown":
            for r in (self.pooled_table if self.pooled_table is not None else pd.DataFrame()).itertuples():
                rows.append({"model": self.name, "bank": "(total)", "term": r.term, "coef": r.coef,
                             "std_err": r.std_err_dk, "t": r.t_dk})
            for b, sh in self.share.items():
                rows += [{"model": self.name, "bank": b, "term": "share: constant", "coef": sh["c"]},
                         {"model": self.name, "bank": b, "term": "share: lagged share change", "coef": sh["phi"]}]
            return rows
        for b, c in self.coef.items():
            se = self.se.get(b, {})
            rows.append({"model": self.name, "bank": b, "term": "constant", "coef": float(c["a"]),
                         "std_err": float(se["a"]) if "a" in se else np.nan})
            rows.append({"model": self.name, "bank": b, "term": "rho (lagged change)", "coef": float(c["rho"]),
                         "std_err": float(se["rho"]) if "rho" in se else np.nan})
            for j, f in enumerate(self.features if self.kind != "ar1" else []):
                rows.append({"model": self.name, "bank": b, "term": f, "coef": float(c["beta"][j]),
                             "std_err": float(se["beta"][j]) if "beta" in se and len(se["beta"]) > j else np.nan})
        if self.kind in ("pooled",) and self.pooled_table is not None:
            for r in self.pooled_table.itertuples():
                rows.append({"model": self.name, "bank": "(common)", "term": r.term, "coef": r.coef,
                             "std_err": r.std_err_dk, "t": r.t_dk})
        return rows
