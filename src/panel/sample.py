"""Which credit unions the panel is estimated on, chosen by the data (panel.sample: auto).

A panel of credit unions is usually very unbalanced: a few dozen still report, hundreds stopped (most merged
into others). Pooling them all lets hundreds of small, exited institutions decide the coefficients used to
forecast the active ones. Four estimation samples are compared:

    all          every credit union, each bank-quarter counts the same
    weighted     every credit union, bank-quarters weighted by the institution's size (its average share of
                 the total balance over the quarters it reported), so large ones count more
    comparable   the active credit unions, plus the exited ones of comparable size (relative size at least the
                 comparable_pct percentile of the active ones)
    active       only the credit unions still reporting at the end of history

For each, the drivers are chosen on that sample (the same panel selection), and the pooled fixed-effects model
with them is backtested at every backtest origin; only the ACTIVE credit unions' forecasts are scored (they are the ones to forecast), on the
panel.evaluation metric over 1..h quarters. The best sample is used for everything that follows (driver
selection, every model, the out-of-sample test), unless 'all' is within sample_tolerance of it: then all is kept,
since it uses the most information. The comparison is written to 0_sample_choice.csv and the summary.

Relative size of a bank: the median over its quarters of its balance divided by the median balance of all banks
reporting in that quarter (so a bank from 2005 and one from 2024 are compared with their own peers).
"""
import copy

import numpy as np
import pandas as pd

CHOICES = ("auto", "all", "weighted", "comparable", "active")
LABELS = {"all": "All credit unions, equal weights", "weighted": "All credit unions, weighted by size",
          "comparable": "Active + exited of comparable size", "active": "Active credit unions only"}


def describe(levels, end):
    """Per bank: active (reports at `end`), first and last quarter, relative size, size share (weight)."""
    df = pd.DataFrame(levels)
    med = df.median(axis=1)
    rel = df.div(med, axis=0).median(axis=0)
    share = df.div(df.sum(axis=1), axis=0).mean(axis=0)
    endq = pd.Period(end, freq="Q")
    rows = [{"bank": b, "active": bool(endq in s.index), "first": str(s.index[0]), "last": str(s.index[-1]),
             "quarters": len(s), "relative_size": float(rel[b]), "size_share": float(share[b])}
            for b, s in levels.items()]
    return pd.DataFrame(rows).sort_values("relative_size", ascending=False).reset_index(drop=True)


def candidates(info, pc):
    """The candidate samples that differ from each other: {name: (banks, weighted)}."""
    allb = list(info["bank"])
    active = list(info.loc[info["active"], "bank"])
    out = {"all": (allb, False)}
    if info["size_share"].nunique() > 1:
        out["weighted"] = (allb, True)
    if 0 < len(active) < len(allb):
        cut = np.percentile(info.loc[info["active"], "relative_size"], float(pc.get("comparable_pct", 25)))
        comp = list(info.loc[info["active"] | (info["relative_size"] >= cut), "bank"])
        if len(active) < len(comp) < len(allb):
            out["comparable"] = (comp, False)
        if len(active) >= 2:
            out["active"] = (active, False)
    return out


def weights(pn, info):
    """Row weights of a panel: each bank-quarter gets its bank's size share, scaled to mean 1 over the rows."""
    sh = info.set_index("bank")["size_share"]
    w = np.array([sh[pn.banks[c]] for c in pn.B], float)
    return w / w.mean()


def restrict(cfg, banks):
    """A config whose panel holds only these banks."""
    c = copy.deepcopy(cfg)
    c.setdefault("panel", {})["banks"] = list(banks)
    c["panel"]["drop_exited"] = False
    return c


def choose(cfg, pn, F, feats, out, log=print):
    """Compare the samples and return (name, banks, weighted, table).

    pn: the panel of all credit unions; feats: the drivers chosen on it.
    """
    from .compare import settings as ev_settings
    from .data import Panel
    from .select import quick_error, select
    pc = cfg.get("panel") or {}
    mode = str(pc.get("sample", "auto")).lower()
    if mode not in CHOICES:
        raise ValueError(f"panel.sample must be one of {CHOICES}")
    info = describe(pn.levels, cfg["sample"]["hist_end"])
    cands = candidates(info, pc)
    active = set(info.loc[info["active"], "bank"])
    if mode != "auto":
        if mode not in cands:
            log(f"  panel.sample: {mode} is the same as all here (no exited banks or equal sizes) - using all")
            mode = "all"
        b, w = cands[mode]
        return mode, b, w, pd.DataFrame([{"sample": mode, "label": LABELS[mode], "banks": len(b), "chosen": True,
                                          "reason": "set in the config (panel.sample)"}]), info
    if len(cands) == 1:
        return "all", cands["all"][0], False, pd.DataFrame([{"sample": "all", "label": LABELS["all"],
                                                             "banks": len(cands["all"][0]), "chosen": True,
                                                             "reason": "only one sample possible"}]), info
    metric = ev_settings(cfg)["metric"]
    rows = []
    for name, (banks, wtd) in cands.items():
        p_ = pn if (name in ("all", "weighted")) else Panel(restrict(cfg, banks), out, lambda *_: None)
        p_.wrow = weights(p_, info) if wtd else None
        try:
            f_ = feats if name == "all" else select(cfg, p_, F, cfg["sample"]["estimation_start"],
                                                    cfg["sample"]["hist_end"], lambda *_: None)["features"]
            err = quick_error(cfg, p_, F, f_, cfg["sample"]["hist_end"], banks=active, metric=metric)
        except Exception as ex:                       # e.g. too few banks for the drivers
            err, f_ = np.nan, []
            log(f"  sample {name}: not scored ({ex})")
        p_.wrow = None
        log(f"  sample {name}: {len(banks)} banks, drivers {f_}, error for the active banks {err * 100:.3f}")
        rows.append({"sample": name, "label": LABELS[name], "banks": len(banks),
                     "exited_included": len([b for b in banks if b not in active]), "weighted": wtd,
                     "drivers": ", ".join(f_), "score_active": err * 100, "_drivers": f_})
    T = pd.DataFrame(rows)
    tol = float(pc.get("sample_tolerance", 0.02))
    ok = T.dropna(subset=["score_active"])
    if ok.empty:
        best = "all"
        reason = "no sample could be scored - all kept"
    else:
        best = ok.sort_values("score_active").iloc[0]["sample"]
        s_all = T.loc[T["sample"] == "all", "score_active"].iloc[0]
        s_best = ok["score_active"].min()
        if best != "all" and np.isfinite(s_all) and s_all <= s_best * (1 + tol):
            reason = f"all is within {tol:.0%} of the best ({best}): all kept, as it uses the most information"
            best = "all"
        else:
            reason = (f"lowest error for the active credit unions" +
                      (f" ({(1 - s_best / s_all) * 100:.1f}% below all)" if best != "all" and np.isfinite(s_all) else ""))
    T["chosen"] = T["sample"] == best
    T["reason"] = np.where(T["chosen"], reason, "")
    b, w = cands[best]
    T.attrs["drivers"] = T.loc[T["chosen"], "_drivers"].iloc[0]
    return best, b, w, T.drop(columns="_drivers"), info
