"""Bayesian model averaging over the candidate drivers (the 'Bayesian first' approach).

    y_t = a + rho*y_{t-1} + dummies + sum_{j in M} b_j x_j,t + e_t       for every model M (a set of drivers)

- The controls (constant, lagged target growth, dummies) are in every model; they are partialled out first.
- One feature per variable: the form and lag most correlated with the target (with the expected sign). Then
  near-duplicates (|correlation| above `candidate_corr`, e.g. the same series for several cities) are grouped and
  one representative is kept, preferring the primary geography; then the `max_candidates` strongest. Without this,
  near-identical series split the probability between them (the 'dilution' problem, George 2010) and none looks
  important.
- Prior on the coefficients: Zellner's g-prior with the benchmark g = max(n, K^2) of Fernandez, Ley & Steel (2001)
  ("bric"), or the unit-information prior g = n ("uip"). The marginal likelihood is then closed-form:
      log p(y | M) = (n - 1 - k)/2 log(1 + g) - (n - 1)/2 log(1 + g (1 - R2_M))            (up to a constant)
- Prior on the models: beta-binomial on the number of drivers with prior mean `model_size` (Ley & Steel 2009), so
  the result is not driven by an arbitrary fixed inclusion probability; models above `max_vars` drivers get none.
- Sign restrictions: a model in which a driver with an expected sign gets the wrong sign has prior 0 (as in the
  IMF's sign-constrained BMA), so only economically sensible models carry weight.
- Model space search: MC3 (Madigan & York 1995) - a Metropolis walk that adds, drops or swaps one driver. Posterior
  probabilities are computed exactly over the models visited (renormalised marginal likelihood x prior).
- Inclusion probability of a driver = total posterior probability of the models that contain it. The drivers with
  probability >= `threshold` form the median probability model (Barbieri & Berger 2004), the optimal single model
  for prediction under squared loss.
"""
import math

import numpy as np
import pandas as pd

from ..features.transforms import feature_sign, parse_feature
from .screening import controls


def prepare(cfg, ctx, F):
    """Partial the controls out of the target and of every feature; keep one feature per variable.

    Returns:
        (y residual, X residual of the kept features, table of candidates with correlation and sign).
    """
    b = cfg["bayes_first"]
    rows = np.arange(ctx.est0, ctx.end + 1)
    y = ctx.y.values[rows]
    X = F.values[rows]
    C = controls(ctx).values[rows]
    C = C[:, (C.std(axis=0) > 0) | (np.arange(C.shape[1]) == 0)]
    ok = ~np.isnan(X).any(axis=0)
    X, names = X[:, ok], [n for n, k in zip(F.columns, ok) if k]
    beta = np.linalg.lstsq(C, np.column_stack([y, X]), rcond=None)[0]
    R = np.column_stack([y, X]) - C @ beta
    y, X = R[:, 0], R[:, 1:]
    sd = X.std(axis=0)
    keep = sd > 1e-12
    X, names = X[:, keep], [n for n, k in zip(names, keep) if k]
    corr = (X - X.mean(0)).T @ (y - y.mean()) / (len(y) * X.std(0) * y.std())
    tab = pd.DataFrame({"feature": names, "corr": corr})
    tab["mnemonic"] = tab["feature"].map(lambda f: parse_feature(f)[0])
    tab["expected_sign"] = [int(feature_sign(ctx.meta, f)) for f in tab["feature"]]
    tab["score"] = np.where(tab["expected_sign"] != 0, tab["corr"] * tab["expected_sign"], tab["corr"].abs())
    tab = tab[tab["score"] > 0]                      # right sign (or no sign prior)
    tab = tab.sort_values("score", ascending=False).drop_duplicates("mnemonic")
    tab = tab.head(4 * b["max_candidates"]).reset_index(drop=True)
    if b.get("candidate_corr") and len(tab) > 1:     # one representative per group of near-duplicates
        from .clustering import cluster, representatives
        tab["geography"] = tab["mnemonic"].map(ctx.meta["geography"])
        Fc = F[tab["feature"]].iloc[ctx.est0:ctx.end + 1]
        tab["cluster"] = cluster(Fc, b["candidate_corr"]).reindex(tab["feature"]).values
        rep = representatives(tab.set_index("feature", drop=False), "score", b.get("primary_geography"),
                              b.get("rep_tolerance", 0.75))
        tab = rep[rep["is_representative"]].reset_index(drop=True).sort_values("score", ascending=False)
    tab = tab.head(b["max_candidates"]).reset_index(drop=True)
    idx = [names.index(f) for f in tab["feature"]]
    return y, X[:, idx], tab, len(rows) - C.shape[1]


class _Space:
    """Marginal likelihood x prior of models (sets of candidate indices), cached."""

    def __init__(self, y, X, signs, cfg, n_eff):
        b = cfg["bayes_first"]
        self.y, self.X, self.signs = y - y.mean(), X - X.mean(0), np.asarray(signs)
        self.n, self.K = len(y), X.shape[1]
        self.n_eff = max(n_eff, self.K + 2)
        self.g = max(self.n, self.K ** 2) if str(b.get("g", "bric")).lower() == "bric" else float(self.n)
        self.max_vars = int(b["max_vars"])
        m = min(float(b["model_size"]), self.K - 0.5)
        self.a, self.bb = 1.0, (self.K - m) / m          # beta-binomial with prior mean model size m
        self.tss = float(self.y @ self.y)
        self.cache = {}

    def logpost(self, S):
        """log marginal likelihood + log prior of model S (a frozenset); -inf if it breaks a sign or the size cap."""
        if S in self.cache:
            return self.cache[S]
        k = len(S)
        if k > self.max_vars:
            v = -np.inf
        elif k == 0:
            v = 0.0
        else:
            cols = sorted(S)
            Z = self.X[:, cols]
            coef, *_ = np.linalg.lstsq(Z, self.y, rcond=None)
            sg = self.signs[cols]
            if np.any((sg != 0) & (np.sign(coef) != sg)):
                v = -np.inf
            else:
                r = self.y - Z @ coef
                r2 = min(max(1 - float(r @ r) / self.tss, 0.0), 1 - 1e-12)
                n1 = self.n_eff - 1
                v = 0.5 * (n1 - k) * math.log1p(self.g) - 0.5 * n1 * math.log1p(self.g * (1 - r2))
        if np.isfinite(v):
            v += (math.lgamma(self.a + k) + math.lgamma(self.bb + self.K - k))   # beta-binomial prior (up to const)
        self.cache[S] = v
        return v


def mc3(y, X, signs, cfg, n_eff):
    """Search the model space with MC3; return the visited models with their posterior probabilities.

    Returns:
        (models DataFrame: drivers (tuple of indices), log_post, prob; acceptance rate).
    """
    b = cfg["bayes_first"]
    sp = _Space(y, X, signs, cfg, n_eff)
    rng = np.random.default_rng(b["seed"])
    K = sp.K
    cur = frozenset()
    lp = sp.logpost(cur)
    acc = 0
    for it in range(int(b["iterations"]) + int(b["burn_in"])):
        j = int(rng.integers(K))
        if j in cur:
            if cur and rng.random() < 0.5 and len(cur) < K:          # swap j for a driver not in the model
                k_ = int(rng.choice([i for i in range(K) if i not in cur]))
                prop = (cur - {j}) | {k_}
            else:
                prop = cur - {j}
        else:
            prop = cur | {j}
        lq = sp.logpost(prop)
        if np.isfinite(lq) and math.log(rng.random() + 1e-300) < lq - lp:
            cur, lp = prop, lq
            acc += 1
    visited = {S: v for S, v in sp.cache.items() if np.isfinite(v)}
    S_ = list(visited)
    v = np.array([visited[s] for s in S_])
    p = np.exp(v - v.max())
    p /= p.sum()
    out = pd.DataFrame({"drivers": [tuple(sorted(s)) for s in S_], "log_post": v, "prob": p})
    return out.sort_values("prob", ascending=False).reset_index(drop=True), acc / max(1, int(b["iterations"]) + int(b["burn_in"]))


def bma_select(cfg, ctx, F):
    """The full BMA: candidates, MC3, inclusion probabilities and the top models.

    Returns:
        dict with tables: candidates (C1), inclusion (C2), top_models (C3); and 'models' as
        [(features, weight)] for the averaged forecast.
    """
    y, X, cand, n_eff = prepare(cfg, ctx, F)
    if cand.empty:
        raise RuntimeError("[bayes_first] no candidate with the expected sign")
    models, acc = mc3(y, X, cand["expected_sign"].values, cfg, n_eff)
    names = list(cand["feature"])
    K = len(names)
    pip = np.zeros(K)
    post_mean = np.zeros(K)
    Xc, yc = X - X.mean(0), y - y.mean()
    sd_y = yc.std()
    for drv, p in zip(models["drivers"], models["prob"]):
        if not drv:
            continue
        pip[list(drv)] += p
        Z = Xc[:, list(drv)]
        coef, *_ = np.linalg.lstsq(Z, yc, rcond=None)
        post_mean[list(drv)] += p * coef * Xc[:, list(drv)].std(0) / sd_y
    inc = cand[["feature", "mnemonic", "expected_sign", "corr"]].copy()
    inc["inclusion_probability"] = pip
    inc["posterior_mean_std_coef"] = post_mean
    inc["description"] = inc["mnemonic"].map(ctx.meta["description"])
    inc["theme"] = inc["mnemonic"].map(ctx.meta["theme"])
    inc["geography"] = inc["mnemonic"].map(ctx.meta["geography"])
    inc = inc.sort_values("inclusion_probability", ascending=False).reset_index(drop=True)
    top = models.head(25).copy()
    top["drivers"] = top["drivers"].map(lambda d: ", ".join(names[i] for i in d) or "(no macro driver)")
    top["cumulative_prob"] = top["prob"].cumsum()
    avg = cfg["bayes_first"]["average"]
    chosen, mass = [], 0.0
    for drv, p in zip(models["drivers"], models["prob"]):
        chosen.append(([names[i] for i in drv], float(p)))
        mass += p
        if mass >= avg["mass"] or len(chosen) >= avg["max_models"]:
            break
    tot = sum(w for _, w in chosen)
    chosen = [(f, w / tot) for f, w in chosen]
    meta = pd.DataFrame([{"candidates": K, "models_visited": len(models), "acceptance_rate": round(acc, 3),
                          "g": "bric" if str(cfg["bayes_first"].get("g", "bric")).lower() == "bric" else "uip",
                          "prior_model_size": cfg["bayes_first"]["model_size"],
                          "posterior_mean_model_size": float(sum(len(d) * p for d, p in
                                                                 zip(models["drivers"], models["prob"]))),
                          "averaged_models": len(chosen), "averaged_mass": round(min(mass, 1.0), 3)}])
    return {"candidates": cand, "inclusion": inc, "top_models": top, "meta": meta, "models": chosen}
