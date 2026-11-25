"""A small, exact model search shared by 'Factors first' and 'Stable first'.

1. For each candidate variable (or factor), the form and lag with the largest t of the expected sign when added to
       y_t = a + rho*y_{t-1} + b*x + dummies        (estimation window)
   Candidates whose best |t| is below t_min are dropped.
2. Every subset of up to max_vars of them is estimated; subsets with a coefficient of the wrong sign (expected sign
   known) are rejected; the subset with the lowest BIC wins.
"""
import itertools

import numpy as np
import pandas as pd

from ..features.transforms import feature_sign
from .double_selection import _window, single_t


def best_forms(ctx, F, variables, t_min=1.0):
    """{variable: (feature, t)} for the best form and lag of each variable."""
    out = {}
    for v in variables:
        cols = [c for c in F.columns if c.split("|")[0] == v]
        if not cols:
            continue
        t = single_t(ctx, F, cols)
        if t.empty:
            continue
        sg = pd.Series({c: feature_sign(ctx.meta, c) for c in t.index})
        score = t * sg.replace(0, 1)
        score[sg == 0] = t[sg == 0].abs()
        f = score.idxmax()
        if score[f] >= t_min:
            out[v] = (f, float(t[f]))
    return out


def subset_search(ctx, F, feats, max_vars=4, top=20):
    """Lowest-BIC subset (signs respected) of feats; returns (best list, table of the best subsets)."""
    y, W, Fw = _window(ctx, F[feats]) if feats else (None, None, None)
    if not feats:
        return [], pd.DataFrame()
    n = len(y)
    sign = {f: feature_sign(ctx.meta, f) for f in feats}
    rows = []
    for k in range(1, min(max_vars, len(feats)) + 1):
        for S in itertools.combinations(feats, k):
            X = np.column_stack([W.values] + [Fw[f].values for f in S])
            if not np.isfinite(X).all():
                continue
            try:
                b, *_ = np.linalg.lstsq(X, y.values, rcond=None)
            except np.linalg.LinAlgError:
                continue
            e = y.values - X @ b
            rss = float(e @ e)
            bic = n * np.log(rss / n) + X.shape[1] * np.log(n)
            coefs = b[-k:]
            ok = all(sign[f] == 0 or np.sign(c) == sign[f] for f, c in zip(S, coefs))
            rows.append({"features": ", ".join(S), "n": k, "bic": bic, "signs_ok": ok})
    if not rows:
        return [], pd.DataFrame()
    T = pd.DataFrame(rows).sort_values("bic")
    good = T[T["signs_ok"]]
    best = good.iloc[0]["features"].split(", ") if len(good) else T[T["n"] == 1].iloc[0]["features"].split(", ")
    return best, T.head(top).reset_index(drop=True)
