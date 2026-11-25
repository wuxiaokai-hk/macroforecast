"""Driver selection on the pooled panel: the same steps as "screen first", with every regression run on all banks.

1. Screen: each candidate alone in a fixed-effects regression with every bank's lagged change and the dummies
   (Frisch-Waugh: the controls are partialled out once). Driscoll-Kraay t statistics; the sign must match.
2. Best form and lag of each variable; correlated variables grouped (same clustering as screen first); one per
   group, up to shortlist_max.
3. Search: every combination of min_vars..max_vars shortlisted drivers that passes the final_model rules
   (|t_DK| >= t_min, expected signs, VIF <= vif_max, at most one per theme). The candidates with the lowest
   pooled BIC are backtested on all banks; the simplest model within selection_tolerance of the best error wins.
"""
import itertools

import numpy as np
import pandas as pd
from scipy import stats

from ..features.transforms import feature_sign, parse_feature
from ..modelling.ardl import vif
from ..modelling.search import backtest_origins, refit_flags
from ..selection.clustering import cluster, representatives
from ..utils import target as tg
from .estimate import PanelModel, driscoll_kraay, ols, sqrt_w


class BankFeatures:
    """Candidate features, some of which may differ by bank.

    common: DataFrame on pn.dates (macro features, the same for every bank).
    bank: optional array (banks, dates, k) of bank-specific features named bank_names (e.g. each bank's own
    mortgage rate change in a system), with their metadata (mnemonic, description, geography, theme,
    expected_sign) in bank_meta.
    Correlations and VIFs of bank-specific features use their average across banks.
    """

    def __init__(self, common, bank=None, bank_names=(), bank_meta=None):
        self.common = common
        self.bank = bank
        self.bank_names = list(bank_names)
        self.bank_meta = bank_meta
        self.columns = list(common.columns) + self.bank_names
        self._bi = {n: i for i, n in enumerate(self.bank_names)}

    def array(self, cols, nb=None):
        """(dates, k) if every column is common, else (banks, dates, k)."""
        cols = list(cols)
        if not any(c in self._bi for c in cols):
            return self.common[cols].values if cols else np.zeros((len(self.common), 0))
        nb = self.bank.shape[0]
        out = np.empty((nb, len(self.common), len(cols)))
        for j, c in enumerate(cols):
            out[:, :, j] = self.bank[:, :, self._bi[c]] if c in self._bi else self.common[c].values[None, :]
        return out

    def stacked(self, pn, cols):
        from .estimate import stacked as _st
        return _st(pn, self.array(cols))

    def frame(self, cols):
        """Dates x cols (bank-specific columns averaged across banks)."""
        d = {}
        for c in cols:
            d[c] = self.common[c] if c in self.common.columns else \
                pd.Series(np.nanmean(self.bank[:, :, self._bi[c]], axis=0), index=self.common.index)
        return pd.DataFrame(d, index=self.common.index)

    def meta(self, base):
        if self.bank_meta is None or not len(self.bank_meta):
            return base
        bm = self.bank_meta.set_index("mnemonic")
        return pd.concat([base, bm[[c for c in base.columns if c in bm.columns]]])


def _fs(F):
    return F if isinstance(F, BankFeatures) else BankFeatures(F)


def _controls(pn, m):
    """Bank intercepts, every bank's lagged change and the dummies, for the rows in mask m."""
    Bm = pn.B[m]
    ok = sorted(set(Bm))
    F = np.column_stack([(Bm == c).astype(float) for c in ok])
    D = pn.Dm[pn.T[m]]
    dk = D.std(axis=0) > 0 if D.shape[1] else np.zeros(0, bool)
    return np.column_stack([F, pn.YL[m], D[:, dk]])


def screen(pn, F, start, end, lags=4):
    """Univariate pooled screen of every candidate feature.

    Args:
        pn: Panel. F: candidate features on pn.dates (DataFrame). start, end: window labels.

    Returns:
        DataFrame: feature, mnemonic, transform, lag, coef, t_dk, p_dk, expected_sign, sign_ok, theme, geography.
    """
    F = _fs(F)
    Xall = F.stacked(pn, F.columns)
    m = pn.window(start, end)
    rows = []
    C = _controls(pn, m)
    y = pn.Y[m]
    X = Xall[m]
    sw = sqrt_w(pn, m)
    if sw is not None:                                # size-weighted rows
        C, y, X = C * sw[:, None], y * sw, X * sw[:, None]
    yr = y - C @ np.linalg.lstsq(C, y, rcond=None)[0]
    okc = ~np.isnan(X).any(axis=0) & (np.nanstd(X, axis=0) > 0)
    cols = np.where(okc)[0]
    Xr = X[:, cols] - C @ np.linalg.lstsq(C, X[:, cols], rcond=None)[0]
    xx = (Xr ** 2).sum(axis=0)
    b = (Xr * yr[:, None]).sum(axis=0) / np.where(xx > 0, xx, np.nan)
    E = yr[:, None] - Xr * b
    T = pn.T[m]
    ut = np.unique(T)
    idx = np.searchsorted(ut, T)
    G = np.zeros((len(ut), len(cols)))
    np.add.at(G, idx, Xr * E)
    S = (G ** 2).sum(axis=0)
    pos = {u: k for k, u in enumerate(ut)}
    for l in range(1, lags + 1):
        j = np.array([pos.get(u - l, -1) for u in ut])
        k = j >= 0
        S += 2 * (1 - l / (lags + 1)) * (G[k] * G[j[k]]).sum(axis=0)
    n, p = Xr.shape[0], C.shape[1] + 1
    se = np.sqrt(np.clip(S, 0, None) / xx ** 2 * (n - 1) / max(n - p, 1) * len(ut) / max(len(ut) - 1, 1))
    t = b / se
    pv = 2 * stats.t.sf(np.abs(t), df=max(len(ut) - 1, 1))
    meta = F.meta(pn.base.meta)
    for jj, c in enumerate(cols):
        f = F.columns[c]
        code, tr, lag = parse_feature(f)
        es = int(feature_sign(meta, f)) if code in meta.index else 0
        rows.append({"feature": f, "mnemonic": code, "transform": tr, "lag": lag, "coef": b[jj], "t_dk": t[jj],
                     "p_dk": pv[jj], "expected_sign": es,
                     "sign_ok": es == 0 or np.sign(b[jj]) == es,
                     "theme": meta.loc[code, "theme"] if code in meta.index else "",
                     "geography": meta.loc[code, "geography"] if code in meta.index else "",
                     "description": meta.loc[code, "description"] if code in meta.index else code})
    sc = pd.DataFrame(rows)
    sc["abs_t"] = sc["t_dk"].abs()
    return sc


def shortlist(cfg, pn, F, sc, start, end):
    """Best form of each variable among those that pass, grouped by correlation, one per group."""
    s = cfg["screen_first"]
    F = _fs(F)
    passed = sc[sc["sign_ok"] & (sc["p_dk"] < s["screen_p"])]
    if passed.empty:
        return passed.assign(cluster=[], is_representative=[])
    best = passed.sort_values("abs_t", ascending=False).groupby("mnemonic").head(1).set_index("feature", drop=False)
    W = F.frame(best.index).loc[pd.Period(start, freq="Q"):pd.Period(end, freq="Q")]
    best["cluster"] = cluster(W, s["cluster_corr"]).reindex(best.index).values
    rep = representatives(best, "abs_t", s.get("primary_geography"), s.get("rep_tolerance", 0.75))
    return rep[rep["is_representative"]].sort_values("abs_t", ascending=False).head(s["shortlist_max"])


def pooled_fit(pn, F, feats, start, end, lags=4):
    """Pooled FE regression of the combination: (beta, Driscoll-Kraay t, BIC, n)."""
    X = _fs(F).stacked(pn, feats)
    m = pn.window(start, end, X)
    C = _controls(pn, m)
    Z = np.column_stack([C, X[m]])
    yv = pn.Y[m]
    sw = sqrt_w(pn, m)
    if sw is not None:
        Z, yv = Z * sw[:, None], yv * sw
    b, e, XtXi = ols(Z, yv)
    V = driscoll_kraay(Z, e, pn.T[m], XtXi, lags)
    k = len(feats)
    se = np.sqrt(np.clip(np.diag(V)[-k:], 0, None))
    n = int(m.sum())
    bic = n * np.log(max(e @ e, 1e-300) / n) + Z.shape[1] * np.log(n)
    return b[-k:], b[-k:] / np.where(se > 0, se, np.nan), bic, n


class _FastPooled:
    """Pooled FE regressions of many driver combinations on the same window, fast and with identical results.

    The stacked data are built once. The controls (bank intercepts, lagged change, dummies) are partialled out of
    the target and of every driver once per estimation sample (Frisch-Waugh), so each combination is a small
    regression of residuals on residuals: the same slopes, residuals, BIC and Driscoll-Kraay covariance as the full
    regression (the small-sample correction is given the full number of regressors).
    """

    def __init__(self, pn, F, feats, start, end):
        self.pn, self.feats = pn, list(feats)
        self.col = {f: j for j, f in enumerate(self.feats)}
        self.X = F.stacked(pn, self.feats)
        self.base = pn.window(start, end)
        self.valid = ~np.isnan(self.X)
        self.cache = {}

    def _sample(self, idx):
        m = self.base & self.valid[:, idx].all(axis=1)
        key = np.packbits(m).tobytes()
        if key not in self.cache:
            ok = self.valid[m].all(axis=0)              # drivers complete on this sample
            C = _controls(self.pn, m)
            Xm = np.where(ok[None, :], self.X[m], 0.0)
            YX = np.column_stack([self.pn.Y[m], Xm])
            sw = sqrt_w(self.pn, m)
            if sw is not None:                        # size-weighted rows
                C, YX = C * sw[:, None], YX * sw[:, None]
            B = np.linalg.lstsq(C, YX, rcond=None)[0]
            R = YX - C @ B
            self.cache[key] = (R[:, 0], R[:, 1:], self.pn.T[m], C.shape[1], int(m.sum()))
        return self.cache[key]

    def slopes(self, combo):
        idx = [self.col[f] for f in combo]
        ry, RX, T, pc, n = self._sample(idx)
        Xt = RX[:, idx]
        if n <= pc + len(idx):
            return None
        b, *_ = np.linalg.lstsq(Xt, ry, rcond=None)
        return b, Xt, ry - Xt @ b, T, pc, n

    def tstats(self, combo, got, lags):
        b, Xt, e, T, pc, n = got
        try:
            XtXi = np.linalg.inv(Xt.T @ Xt)
        except np.linalg.LinAlgError:
            XtXi = np.linalg.pinv(Xt.T @ Xt)
        V = driscoll_kraay(Xt, e, T, XtXi, lags, p=pc + len(combo))
        se = np.sqrt(np.clip(np.diag(V), 0, None))
        bic = n * np.log(max(e @ e, 1e-300) / n) + (pc + len(combo)) * np.log(n)
        return b / np.where(se > 0, se, np.nan), bic, n


def search(cfg, pn, F, short, start, end, log=print):
    """Every valid combination of shortlisted drivers, ranked by pooled BIC, then the best few backtested.

    Returns:
        (chosen features, table of valid combinations with their BIC and backtest error).
    """
    fm = cfg["final_model"]
    pc = cfg.get("panel") or {}
    feats = list(short["feature"])
    if not feats:
        return [], pd.DataFrame()
    themes = short.set_index("feature")["theme"].to_dict()
    signs = short.set_index("feature")["expected_sign"].to_dict()
    F = _fs(F)
    W = F.frame(feats).loc[pd.Period(start, freq="Q"):pd.Period(end, freq="Q")]
    lags = int(pc.get("dk_lags", 4))
    fit = _FastPooled(pn, F, feats, start, end)
    t_min, vif_max = float(fm.get("t_min", 1.96)), float(fm.get("vif_max", 5))
    rows = []
    for k in range(int(fm.get("min_vars", 1)), int(fm.get("max_vars", 3)) + 1):
        for combo in itertools.combinations(feats, k):
            if fm.get("one_per_theme", True) and len({themes[f] for f in combo}) < k:
                continue
            if fm.get("required_themes") and not set(fm["required_themes"]) <= {themes[f] for f in combo}:
                continue
            # cheapest checks first: the slopes and their signs (plain least squares), then the VIF, then the
            # Driscoll-Kraay t statistics only for the combinations still in (the same rules, in a cheaper order)
            got = fit.slopes(combo)
            if got is None:
                continue
            b = got[0]
            if any(signs[f] != 0 and np.sign(bv) != signs[f] for f, bv in zip(combo, b)):
                continue
            if k > 1 and np.nanmax(vif(W[list(combo)])) > vif_max:
                continue
            t, bic, n = fit.tstats(combo, got, lags)
            if np.any(np.abs(t) < t_min) or np.any(~np.isfinite(t)):
                continue
            rows.append({"features": list(combo), "k": k, "bic": bic, "min_abs_t": float(np.min(np.abs(t))), "n": n})
    if not rows:
        return [], pd.DataFrame()
    R = pd.DataFrame(rows).sort_values("bic").reset_index(drop=True)
    top = R.head(int(pc.get("n_search_candidates", 20))).copy()
    errs = []
    for f in top["features"]:
        try:
            errs.append(quick_error(cfg, pn, F, f, end))
        except Exception:
            errs.append(np.nan)
    top["backtest_error_pct"] = np.array(errs) * 100
    top = top.dropna(subset=["backtest_error_pct"])
    if top.empty:
        return list(R.iloc[0]["features"]), R
    best = top["backtest_error_pct"].min()
    tol = float(fm.get("selection_tolerance", 0.05))
    ok = top[top["backtest_error_pct"] <= best * (1 + tol)].sort_values(["k", "backtest_error_pct"])
    chosen = list(ok.iloc[0]["features"])
    R = R.merge(top[["bic", "backtest_error_pct"]], on="bic", how="left")
    R["chosen"] = R["features"].apply(lambda f: list(f) == chosen)
    R["features"] = R["features"].apply(", ".join)
    return chosen, R


def quick_error(cfg, pn, F, feats, end, banks=None, metric=None):
    """Average absolute level error over 1..H quarters, all banks and backtest origins up to `end`, of the
    pooled model with these drivers (dummy windows left out). banks: only these banks' forecasts are scored;
    metric: a panel.evaluation metric instead of the average (src.panel.compare)."""
    H = int(cfg["sample"]["horizon"])
    X = _fs(F).array(feats)
    origins = [o for o in backtest_origins(pn.base) if o + 1 < len(pn.dates) and pn.dates[o] <= pd.Period(end, freq="Q")]
    if not origins:
        return np.nan
    flags = refit_flags(pn.base, origins)
    m = PanelModel("pooled", feats, min_obs=int((cfg.get("panel") or {}).get("min_obs", 20)))
    errs = []
    start = cfg["sample"]["estimation_start"]
    endp = pn.pos(end)
    dq = [pd.Period(q, freq="Q") for q in cfg["sample"]["dummy_quarters"]]
    lo, hi = (min(dq), max(dq)) if dq else (None, None)
    for o, fl in zip(origins, flags):
        if fl:
            m.fit(pn, X, start, pn.dates[o])
        for b in pn.active(pn.dates[o]):
            if banks is not None and b not in banks:
                continue
            y = pn.ys[b]
            q0 = pn.dates[o]
            if b not in m.coef or pd.isna(y.get(q0, np.nan)):
                continue
            hmax = min(H, endp - o)
            fc = m.forecast(pn, X, b, o, hmax, {"bank": float(y.loc[q0])})
            act = np.array([y.get(pn.dates[o + h], np.nan) for h in range(1, len(fc) + 1)])
            for h in range(1, len(fc) + 1):
                win = pn.dates[o + 1:o + h + 1]
                if lo is not None and ((win >= lo) & (win <= hi)).any():
                    continue
                if np.isnan(act[:h]).any():
                    break
                errs.append(float(tg.error(fc[:h].sum(), act[:h].sum())))
    if not errs:
        return np.nan
    if metric:
        from .compare import _one
        return _one(np.array(errs), 0.10, 0.5)[metric]
    return float(np.mean(np.abs(errs)))


def select(cfg, pn, F, start, end, log=print):
    """Screen, shortlist and search on the window start..end.

    Returns:
        dict: features (chosen), tables (P1_panel_screen, P2_shortlist, P3_search).
    """
    lags = int((cfg.get("panel") or {}).get("dk_lags", 4))
    sc = screen(pn, F, start, end, lags)
    short = shortlist(cfg, pn, F, sc, start, end)
    chosen, R = search(cfg, pn, F, short, start, end, log)
    return {"features": chosen, "tables": {"P1_panel_screen": sc.sort_values("abs_t", ascending=False),
                                           "P2_shortlist": short.reset_index(drop=True), "P3_search": R}}
