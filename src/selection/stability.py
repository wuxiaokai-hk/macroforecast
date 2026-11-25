"""Stability selection (Meinshausen & Buehlmann 2010, "Stability selection", JRSS B 72(4); with the bound of
Shah & Samworth 2013): keep only the variables a lasso picks again and again on many sub-samples of the history.

1. The target and every candidate feature are partialled on the constant, last quarter's target growth and the
   dummies, and standardised (estimation window).
2. B times (stability.draws, default 100): draw a sub-sample of half the quarters, in blocks of
   stability.block quarters (4) so seasonal and serial patterns stay together; run the lasso path and record the
   first q variables to enter (stability.q, default 8). Variables, not features: a variable is picked when any of
   its forms or lags is. Closely related variables (correlation of their best features above stability.cluster_corr, 0.75)
   are counted together as one group, so they do not split their votes.
3. Selection probability = share of draws in which a variable (group) was picked. Kept: probability >=
   stability.threshold (0.6). Expected number of falsely kept variables <= q^2 / ((2*threshold - 1) * p)
   (Meinshausen & Buehlmann's bound; p = number of variables), reported with the results.
"""
import numpy as np
import pandas as pd

from .double_selection import _partial, _window


def settings(cfg):
    d = {"enabled": True, "draws": 100, "q": 8, "threshold": 0.6, "block": 4, "cluster_corr": 0.75, "seed": 7,
         "max_vars": 4, "t_min": 1.0}
    d.update({k: v for k, v in (cfg.get("stability") or {}).items() if v is not None})
    return d


def _groups(ctx, F, variables, thr):
    """Variables grouped by near-identical best features: {variable: group label}."""
    from .double_selection import single_t
    best = {}
    for v in variables:
        cols = [c for c in F.columns if c.split("|")[0] == v]
        t = single_t(ctx, F, cols)
        if len(t):
            best[v] = t.abs().idxmax()
    vs = list(best)
    if not vs:
        return {}
    win = F[[best[v] for v in vs]].iloc[ctx.est0:ctx.end + 1]
    C = np.abs(np.nan_to_num(np.corrcoef(win.values, rowvar=False)))
    lab = {}
    for i, v in enumerate(vs):
        if v in lab:
            continue
        lab[v] = v
        for j in range(i + 1, len(vs)):
            if vs[j] not in lab and C[i, j] > thr:
                lab[vs[j]] = v
    return lab


def select(cfg, ctx, F, log=print):
    """Selection probabilities of every variable (group) and the stable ones.

    Returns:
        (table of probabilities, list of stable variables (the group's most-picked member), settings dict).
    """
    from sklearn.linear_model import lars_path
    st = settings(cfg)
    y, W, Fw = _window(ctx, F)
    cols = [c for c in Fw.columns if Fw[c].notna().all() and Fw[c].std() > 0]
    Fw = Fw[cols]
    X = _partial(Fw.values, W)
    yv = _partial(y.values[:, None], W)[:, 0]
    sd = X.std(axis=0)
    X = (X - X.mean(axis=0)) / np.where(sd > 0, sd, 1)
    var_of = np.array([c.split("|")[0] for c in cols])
    variables = sorted(set(var_of))
    group = _groups(ctx, F[cols], variables, float(st["cluster_corr"]))
    n = len(yv)
    rng = np.random.default_rng(int(st["seed"]))
    blk = max(int(st["block"]), 1)
    starts = np.arange(0, n, blk)
    counts, member_counts = {}, {}
    q = int(st["q"])
    for _ in range(int(st["draws"])):
        pick = rng.choice(len(starts), size=max(len(starts) // 2, 1), replace=False)
        rows = np.concatenate([np.arange(s, min(s + blk, n)) for s in starts[pick]])
        Xs, ys = X[rows], yv[rows] - yv[rows].mean()
        try:
            _, _, coefs = lars_path(Xs - Xs.mean(axis=0), ys, method="lasso", max_iter=4 * q)
        except Exception:
            continue
        order = []
        for k in range(coefs.shape[1]):              # order of entry along the path
            for j in np.flatnonzero(coefs[:, k]):
                g = group.get(var_of[j], var_of[j])
                if g not in [o[0] for o in order]:
                    order.append((g, var_of[j]))
            if len(order) >= q:
                break
        for g, v in order[:q]:
            counts[g] = counts.get(g, 0) + 1
            member_counts[(g, v)] = member_counts.get((g, v), 0) + 1
    B = int(st["draws"])
    rows = []
    for g, c in sorted(counts.items(), key=lambda kv: -kv[1]):
        mem = [v for (gg, v) in member_counts if gg == g]
        top = max(mem, key=lambda v: member_counts[(g, v)])
        rows.append({"group": g, "representative": top, "probability": c / B,
                     "members": len([v for v in variables if group.get(v, v) == g]),
                     "theme": ctx.meta.loc[top, "theme"] if top in ctx.meta.index else "",
                     "description": ctx.meta.loc[top, "description"] if top in ctx.meta.index and
                     "description" in ctx.meta.columns else top})
    T = pd.DataFrame(rows)
    p = max(len(set(group.values()) | (set(variables) - set(group))), 1)
    thr = float(st["threshold"])
    bound = q * q / ((2 * thr - 1) * p) if thr > 0.5 else np.nan
    stable = list(T.loc[T["probability"] >= thr, "representative"]) if len(T) else []
    if len(T):
        T["stable"] = T["probability"] >= thr
    info = {"draws": B, "q": q, "threshold": thr, "variables": len(variables), "groups": p,
            "expected_false_kept_at_most": bound, "stable": len(stable), "observations": n}
    return T, stable, info
