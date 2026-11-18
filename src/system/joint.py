"""Joint estimators for the system: SUR estimation of the chain, and a VECM challenger.

SUR chain
    The same equations as the chain (each target's chosen drivers, its own last-quarter change, constant and
    dummies), estimated jointly by feasible GLS (Zellner): OLS for each equation, then the covariance of their
    residuals, then GLS on all equations stacked. Uses the correlation of the targets' surprises (the same
    shocks hit loans, deposits and rates in the same quarter): more precise coefficients, and joint uncertainty.
    Forecasts run the chain in order (earlier targets' forecasts feed the later ones). Fan charts: shocks drawn
    from the estimated residual covariance, so the targets move together as they did in history.

VECM challenger
    Every target's change, with error-correction terms for long-run relations given in the config:
        ratio   ln(sum of numerator levels) - ln(sum of denominator levels), e.g. loan-to-deposit
        against a target regressed on macro series in levels (Engle-Granger), e.g. own mortgage rate on the 5Y yield
    dy_i,t = c_i + sum_k alpha_ik ect_k,t-1 + sum_j a_ij dy_j,t-1 + b_i' x_t + e_i,t
    Cross-target lags are shrunk towards 0 as in the BVARX; the error-correction and macro terms are not.
    Each relation is tested for stationarity (ADF; Engle-Granger critical values for estimated relations), and
    the targets' levels get a Johansen trace test of the number of long-run relations.
"""
import numpy as np
import pandas as pd

from ..features.transforms import feature_matrix, parse_feature
from ..modelling.search import backtest_origins, refit_flags
from ..utils.io import add_constructed


def _step(form, L, g):
    return L * np.exp(g) if form == "dlog" else L + g


def _path(form, L0, g):
    c = np.cumsum(g)
    return L0 * np.exp(c) if form == "dlog" else L0 + c


def mdf_with(m, base, upstream, origin):
    """A member's macro data with the earlier targets replaced by `upstream` level paths after the origin."""
    mdf = base.copy()
    for name, path in upstream.items():
        col = "SYS." + str(name).upper()
        if col in mdf.columns:
            p = path[path.index > origin]
            common = p.index.intersection(mdf.index)
            mdf.loc[common, col] = p.loc[common].values
    scen, _ = add_constructed(m.cfg, {"x": mdf}, m.ctx.meta.copy())
    return scen["x"]


def drivers_of(m):
    """The drivers of a member's chosen model; for a model without drivers, those of 'Screen first'."""
    f = list(getattr(m.model, "features", []) or [])
    if not f:
        sf = m.models.get("Screen first")
        f = list(getattr(sf, "features", []) or []) if sf is not None else []
    out = []
    for x in f:
        try:
            parse_feature(x)
            out.append(x)
        except ValueError:
            continue
    return out


# ================================================================================================ SUR
class SURChain:
    name = "SUR chain"

    def __init__(self, members):
        self.members = members
        self.feats = {m.name: drivers_of(m) for m in members}
        self.F = {m.name: feature_matrix(m.ctx.base, self.feats[m.name], m.cfg["features"].get("deflator"))
                  if self.feats[m.name] else pd.DataFrame(index=m.ctx.base.index) for m in members}

    def _design(self, m, rows):
        y = m.ctx.y.reindex(rows).values
        yl = m.ctx.y.shift(1).reindex(rows).values
        X = self.F[m.name].reindex(rows).values
        D = m.ctx.dummies.reindex(rows).fillna(0).values
        return y, np.column_stack([np.ones(len(rows)), yl, X, D]), list(m.ctx.dummies.columns)

    def fit(self, start, end):
        """Feasible GLS on the quarters start..end where every equation has data."""
        idx = self.members[0].ctx.y.index
        rows = idx[(idx >= start) & (idx <= end)]
        des = {m.name: self._design(m, rows) for m in self.members}
        ok = np.ones(len(rows), bool)
        for y, X, _ in des.values():
            ok &= ~np.isnan(y) & ~np.isnan(X).any(1)
        rows = rows[ok]
        self.rows = rows
        Ys, Xs, names, keep = [], [], {}, {}
        for m in self.members:
            y, X, dn = des[m.name]
            y, X = y[ok], X[ok]
            k = (X.std(0) > 0)
            k[0] = True                                    # constant
            keep[m.name] = k
            Xs.append(X[:, k])
            Ys.append(y)
            names[m.name] = (["constant", "lag of own change"] + self.feats[m.name] + dn)
            names[m.name] = [n for n, f in zip(names[m.name], k) if f]
        T, n = len(rows), len(self.members)
        E = np.column_stack([y - X @ np.linalg.lstsq(X, y, rcond=None)[0] for y, X in zip(Ys, Xs)])
        S = E.T @ E / T
        Si = np.linalg.inv(S)
        sizes = [X.shape[1] for X in Xs]
        off = np.r_[0, np.cumsum(sizes)]
        K = off[-1]
        XtOX, XtOy = np.zeros((K, K)), np.zeros(K)
        for i in range(n):
            for j in range(n):
                XtOX[off[i]:off[i + 1], off[j]:off[j + 1]] = Si[i, j] * Xs[i].T @ Xs[j]
            XtOy[off[i]:off[i + 1]] = sum(Si[i, j] * Xs[i].T @ Ys[j] for j in range(n))
        V = np.linalg.inv(XtOX)
        b = V @ XtOy
        self.coef, self.se, self.keep, self.names = {}, {}, keep, names
        res = []
        for i, m in enumerate(self.members):
            bi = b[off[i]:off[i + 1]]
            full = np.zeros(len(keep[m.name]))
            full[keep[m.name]] = bi
            self.coef[m.name] = full
            self.se[m.name] = np.sqrt(np.clip(np.diag(V)[off[i]:off[i + 1]], 0, None))
            res.append(Ys[i] - Xs[i] @ bi)
        R = np.column_stack(res)
        self.Sigma = R.T @ R / T
        self.ols_Sigma = S
        return self

    def forecast(self, origin, H, base_of=None, shocks=None):
        """The chain from `origin`: {name: (changes, level path Series)}.

        Args:
            origin: quarter. H: quarters. base_of: {name: macro DataFrame} (default: each member's history).
            shocks: optional array (H, n) added to the changes (fan charts).
        """
        out, paths = {}, {}
        for i, m in enumerate(self.members):
            base = base_of[m.name] if base_of else m.ctx.base
            if origin not in base.index or origin not in m.ctx.level.index:
                continue
            if m.uses:
                mdf = mdf_with(m, base, {u: paths[u] for u in m.uses if u in paths}, origin)
                F = feature_matrix(mdf, self.feats[m.name], m.cfg["features"].get("deflator")) \
                    if self.feats[m.name] else pd.DataFrame(index=mdf.index)
            else:
                F = self.F[m.name] if base_of is None else \
                    (feature_matrix(base, self.feats[m.name], m.cfg["features"].get("deflator"))
                     if self.feats[m.name] else pd.DataFrame(index=base.index))
            p = F.index.get_loc(origin)
            h = min(H, len(F) - p - 1)
            if h <= 0:
                continue
            c = self.coef[m.name]
            k = len(self.feats[m.name])
            yl = float(m.ctx.y.loc[origin])
            g = np.zeros(h)
            for s in range(h):
                x = F.iloc[p + 1 + s].values if k else np.zeros(0)
                yl = c[0] + c[1] * yl + (x @ c[2:2 + k] if k else 0.0)
                if shocks is not None:
                    yl += shocks[s, i]
                g[s] = yl
            dates = F.index[p + 1:p + 1 + h]
            lv = pd.Series(_path(m.form, float(m.ctx.level.loc[origin]), g), index=dates)
            paths[m.name] = pd.concat([m.ctx.level.loc[:origin], lv])
            out[m.name] = (g, lv)
        return out

    def table(self):
        rows = []
        for m in self.members:
            k = self.keep[m.name]
            for nm, b, se in zip(self.names[m.name], self.coef[m.name][k], self.se[m.name]):
                rows.append({"equation": m.name, "term": nm, "coef": b, "std_err": se,
                             "t": b / se if se > 0 else np.nan})
        return pd.DataFrame(rows)

    def correlation(self):
        d = np.sqrt(np.diag(self.Sigma))
        C = self.Sigma / np.outer(d, d)
        nm = [m.name for m in self.members]
        return pd.DataFrame(C, index=nm, columns=nm)


def _res_rows(members, out, origin, method, test, covid, rows_fn):
    res = {}
    for m in members:
        if m.name not in out:
            continue
        g, lv = out[m.name]
        o = m.ctx.y.index.get_loc(origin)
        act = m.ctx.y.iloc[o + 1:o + 1 + len(g)].values.astype(float)
        n = len(act)                           # a backtest path stops where the history ends
        if n == 0:
            continue
        res[m.name] = (g[:n], act, float(m.ctx.level.loc[origin]), lv.iloc[:n], m.form)
    return rows_fn(res, origin, method, test, covid)


def sur_backtest(members, H, covid, rows_fn, keep=None):
    c0 = members[0].ctx
    pos = backtest_origins(c0)
    origins, flags = [c0.idx[o] for o in pos], refit_flags(c0, pos)
    start = pd.Period(members[0].cfg["sample"]["estimation_start"], freq="Q")
    sur = SURChain(members)
    rows = []
    for o, fl in zip(origins, flags):
        if fl:
            sur.fit(start, o)
            if keep is not None:
                keep[str(o)] = {"coef": {k: v.copy() for k, v in sur.coef.items()}, "Sigma": sur.Sigma.copy()}
        rows += _res_rows(members, sur.forecast(o, H), o, "SUR chain", "backtest", covid, rows_fn)
    sur.fit(start, c0.idx[c0.end])
    return pd.DataFrame(rows), sur


def sur_scenarios(members, sur, n_sims=300, seed=7):
    """Point paths and fan-chart percentiles of every target under every scenario."""
    c0 = members[0]
    hist_end = c0.ctx.idx[c0.ctx.end]
    pts, fan = [], []
    rng = np.random.default_rng(seed)
    Lc = np.linalg.cholesky(sur.Sigma + 1e-12 * np.eye(len(members)))
    for sc in c0.ctx.macro:
        base_of = {m.name: m.ctx.macro[sc] for m in members}
        out = sur.forecast(hist_end, 10 ** 6, base_of)
        for nm, (g, lv) in out.items():
            pts += [{"target": nm, "method": "SUR chain", "scenario": sc, "date": str(q), "level": float(v)}
                    for q, v in lv.items()]
        H = max(len(v[0]) for v in out.values()) if out else 0
        if not H or n_sims <= 0:
            continue
        sims = {m.name: [] for m in members}
        for _ in range(int(n_sims)):
            sh = rng.standard_normal((H, len(members))) @ Lc.T
            o2 = sur.forecast(hist_end, H, base_of, sh)
            for nm, (g, lv) in o2.items():
                sims[nm].append(lv.values)
        for nm, arr in sims.items():
            if not arr:
                continue
            A = np.array(arr)
            dates = out[nm][1].index
            q = np.percentile(A, [5, 25, 50, 75, 95], axis=0)
            fan += [{"target": nm, "scenario": sc, "date": str(d), "p5": q[0, t], "p25": q[1, t], "p50": q[2, t],
                     "p75": q[3, t], "p95": q[4, t]} for t, d in enumerate(dates)]
    return pd.DataFrame(pts), pd.DataFrame(fan)


# ================================================================================================ VECM
def _against(rel):
    """Macro series a target is regressed on in a relation ('against'; YAML reads a bare 'on:' key as True)."""
    v = rel.get("against", rel.get("on", rel.get(True)))
    return v if isinstance(v, list) else [v]


class VECM:
    name = "VECM (all targets)"

    def __init__(self, members, relations, exog, lam=0.2):
        relations = [{**r, "against": _against(r)} if not r.get("numerator") else r for r in relations]
        self.members, self.relations, self.exog, self.lam = members, list(relations), list(exog), float(lam)
        self.names = [m.name for m in members]
        self.forms = {m.name: m.form for m in members}

    def _levels(self, base=None):
        return pd.DataFrame({m.name: m.ctx.level for m in self.members})

    def _ect(self, L, macro, rel, beta=None):
        """Value of one relation over the quarters of L (levels) and macro (levels)."""
        if rel.get("numerator"):
            return np.log(L[rel["numerator"]].sum(axis=1)) - np.log(L[rel["denominator"]].sum(axis=1)), None
        on = _against(rel)
        y = L[rel["target"]]
        Z = macro[on].reindex(L.index)
        if beta is None:
            ok = y.notna() & Z.notna().all(axis=1)
            A = np.column_stack([np.ones(ok.sum()), Z[ok].values])
            beta = np.linalg.lstsq(A, y[ok].values, rcond=None)[0]
        return y - (beta[0] + Z.values @ beta[1:]), beta

    def fit(self, Y, X, macro, start, end):
        L = self._levels()
        rows = Y.loc[start:end].index
        self.beta, ects = {}, {}
        for r in self.relations:
            e, b = self._ect(L.loc[:end], macro.loc[:end], r)
            self.beta[r["name"]] = b
            ects[r["name"]] = e - e.loc[start:end].mean()
        self.ect_mean = {r["name"]: float(self._ect(L.loc[:end], macro.loc[:end], r, self.beta[r["name"]])[0]
                                          .loc[start:end].mean()) for r in self.relations}
        E = pd.DataFrame(ects).reindex(Y.index)
        n, k = len(self.names), len(self.relations)
        Yv = Y.loc[rows].values
        Yl = Y.shift(1).loc[rows].values
        El = E.shift(1).loc[rows].values
        Xv = X.reindex(rows).values if len(self.exog) else np.zeros((len(rows), 0))
        ok = ~np.isnan(Yv).any(1) & ~np.isnan(Yl).any(1) & ~np.isnan(El).any(1) & ~np.isnan(Xv).any(1)
        Yv, Yl, El, Xv = Yv[ok], Yl[ok], El[ok], Xv[ok]
        s = np.array([np.std(Yv[:, i] - np.polyval(np.polyfit(Yl[:, i], Yv[:, i], 1), Yl[:, i])) for i in range(n)])
        Z = np.column_stack([np.ones(len(Yv)), El, Yl, Xv])
        self.c, self.alpha = np.zeros(n), np.zeros((n, k))
        self.A, self.B = np.zeros((n, n)), np.zeros((n, Xv.shape[1]))
        for i in range(n):
            pen = np.zeros(Z.shape[1])
            for j in range(n):
                if j != i:
                    pen[1 + k + j] = (s[i] ** 2) / max((self.lam * s[i] / max(s[j], 1e-12)) ** 2, 1e-18)
            co = np.linalg.solve(Z.T @ Z + np.diag(pen) + 1e-10 * np.eye(Z.shape[1]), Z.T @ Yv[:, i])
            self.c[i], self.alpha[i] = co[0], co[1:1 + k]
            self.A[i], self.B[i] = co[1 + k:1 + k + n], co[1 + k + n:]
        self.n_obs = len(Yv)
        return self

    def forecast(self, Y, X, macro, origin, H):
        """{name: (changes, level path Series)} from `origin` with exogenous X and macro levels (scenario)."""
        L = self._levels()
        idx = X.index if len(self.exog) else macro.index
        if origin not in idx:
            return {}
        p = idx.get_loc(origin)
        h = min(H, len(idx) - p - 1)
        if h <= 0:
            return {}
        dates = idx[p + 1:p + 1 + h]
        lev = {nm: float(L[nm].loc[origin]) for nm in self.names}
        y = Y.loc[origin].values.astype(float)
        out = {nm: [] for nm in self.names}
        for s, q in enumerate(dates):
            ect = []
            prev_q = idx[p + s]
            for r in self.relations:
                if r.get("numerator"):
                    v = np.log(sum(lev[a] for a in r["numerator"])) - np.log(sum(lev[a] for a in r["denominator"]))
                else:
                    on = _against(r)
                    b = self.beta[r["name"]]
                    v = lev[r["target"]] - (b[0] + macro.loc[prev_q, on].values @ b[1:])
                ect.append(v - self.ect_mean[r["name"]])
            x = X.loc[q].values if len(self.exog) else np.zeros(0)
            y = self.c + self.alpha @ np.array(ect) + self.A @ y + self.B @ x
            for i, nm in enumerate(self.names):
                lev[nm] = _step(self.forms[nm], lev[nm], y[i])
                out[nm].append(y[i])
        res = {}
        for nm in self.names:
            g = np.array(out[nm])
            res[nm] = (g, pd.Series(_path(self.forms[nm], float(L[nm].loc[origin]), g), index=dates))
        return res

    def table(self):
        rows = []
        for i, nm in enumerate(self.names):
            rows.append({"equation": nm, "term": "constant", "coef": self.c[i]})
            rows += [{"equation": nm, "term": f"error correction: {r['name']}", "coef": self.alpha[i, k]}
                     for k, r in enumerate(self.relations)]
            rows += [{"equation": nm, "term": f"{self.names[j]} (lag 1)", "coef": self.A[i, j], "cross": j != i}
                     for j in range(len(self.names))]
            rows += [{"equation": nm, "term": f, "coef": self.B[i, k]} for k, f in enumerate(self.exog)]
        for r in self.relations:
            b = self.beta.get(r["name"])
            if b is not None:
                on = _against(r)
                rows.append({"equation": f"long run: {r['name']}", "term": "constant", "coef": b[0]})
                rows += [{"equation": f"long run: {r['name']}", "term": o, "coef": v} for o, v in zip(on, b[1:])]
        return pd.DataFrame(rows)


def cointegration_tests(vecm, macro, start, end):
    """ADF on each relation (Engle-Granger critical values when the relation is estimated) and a Johansen trace
    test on the targets' levels (log for balances, level for rates)."""
    from statsmodels.tsa.stattools import adfuller
    rows = []
    L = vecm._levels()
    for r in vecm.relations:
        e, _ = vecm._ect(L.loc[:end], macro.loc[:end], r, vecm.beta.get(r["name"]))
        e = e.loc[start:end].dropna()
        stat, p, *_ = adfuller(e.values, regression="c", autolag="AIC")
        n_on = 0 if r.get("numerator") else len(_against(r))
        crit = {0: -2.86, 1: -3.34, 2: -3.74, 3: -4.10}.get(n_on, -4.4)     # 5%, Engle-Granger (MacKinnon)
        rows.append({"test": "ADF on the relation", "relation": r["name"], "statistic": stat,
                     "p_value": p if n_on == 0 else np.nan, "crit_5pct": crit,
                     "stationary_5pct": bool(stat < crit),
                     "note": "fixed vector" if n_on == 0 else "estimated vector: Engle-Granger critical value"})
    try:
        from statsmodels.tsa.vector_ar.vecm import coint_johansen
        S = pd.DataFrame({nm: (np.log(L[nm]) if vecm.forms[nm] == "dlog" else L[nm]) for nm in vecm.names})
        S = S.loc[start:end].dropna()
        j = coint_johansen(S.values, det_order=0, k_ar_diff=1)
        rank = 0
        for r_ in range(len(vecm.names)):
            ok = j.lr1[r_] > j.cvt[r_, 1]
            rows.append({"test": "Johansen trace", "relation": f"rank <= {r_}", "statistic": j.lr1[r_],
                         "crit_5pct": j.cvt[r_, 1], "stationary_5pct": None,
                         "note": "reject: more relations than this" if ok else "not rejected"})
            if ok:
                rank = r_ + 1
            else:
                break
        rows.append({"test": "Johansen trace", "relation": "estimated number of long-run relations",
                     "statistic": rank, "note": "among the targets' levels (5%)"})
    except Exception as ex:
        rows.append({"test": "Johansen trace", "relation": "not run", "note": str(ex)})
    return pd.DataFrame(rows)


def vecm_backtest(members, relations, lam, H, covid, rows_fn, Y, X, exog, keep=None):
    c0 = members[0].ctx
    macro = c0.base
    pos = backtest_origins(c0)
    origins, flags = [c0.idx[o] for o in pos], refit_flags(c0, pos)
    start = pd.Period(members[0].cfg["sample"]["estimation_start"], freq="Q")
    vm = VECM(members, relations, exog, lam)
    rows = []
    for o, fl in zip(origins, flags):
        if fl:
            vm.fit(Y, X, macro, start, o)
            if keep is not None:
                import pickle
                keep[str(o)] = pickle.dumps(vm)
        rows += _res_rows(members, vm.forecast(Y, X, macro, o, H), o, "VECM", "backtest", covid, rows_fn)
    vm.fit(Y, X, macro, start, c0.idx[c0.end])
    return pd.DataFrame(rows), vm


def vecm_scenarios(members, vm, Y):
    c0 = members[0]
    hist_end = c0.ctx.idx[c0.ctx.end]
    out = []
    for sc, mdf in c0.ctx.macro.items():
        Xs = feature_matrix(mdf, vm.exog, c0.cfg["features"].get("deflator")) if vm.exog else pd.DataFrame(index=mdf.index)
        res = vm.forecast(Y, Xs, mdf, hist_end, 10 ** 6)
        for nm, (g, lv) in res.items():
            out += [{"target": nm, "method": "VECM", "scenario": sc, "date": str(q), "level": float(v)}
                    for q, v in lv.items()]
    return pd.DataFrame(out)
