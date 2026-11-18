"""How the target series is modelled: one place for the transform, the rebuilt level and the error measure.

Two forms (config: target.transform):

    dlog   y_t = ln(L_t) - ln(L_{t-1})                 balances, deposits, investment, prices: anything > 0
           level after h quarters: L_0 * exp(sum of y)  error: forecast level / actual level - 1   (a fraction)
    diff   y_t = L_t - L_{t-1}                         interest rates, spreads, rates of change, anything that
           level after h quarters: L_0 + sum of y         can be zero or negative
           error: (forecast level - actual level) / 100 (so that "x 100" gives the error in the target's units,
           e.g. percentage points for a rate; the *_pct columns then hold units, not percent)
    auto   dlog when every value is positive, else diff  (set diff yourself for a rate, which is positive too)

The pipeline uses one target per process, so the form is held here once Context has read the data.
Every function works on numpy arrays or pandas objects.
"""
import numpy as np

FORMS = ("dlog", "diff")
_state = {"form": "dlog"}


def set_form(form):
    """Set the target form for this run ('dlog' or 'diff')."""
    if form not in FORMS:
        raise ValueError(f"target.transform must be one of {FORMS} or auto, not {form!r}")
    _state["form"] = form


def form():
    """The current target form."""
    return _state["form"]


def is_log():
    """True for dlog targets (levels in logs, errors in percent)."""
    return _state["form"] == "dlog"


def resolve(requested, level):
    """The form to use: 'dlog', 'diff', or auto (dlog if every value is positive).

    Args:
        requested: config value (dlog, diff, auto or None). level: the target series.
    """
    r = (requested or "auto").lower()
    if r == "auto":
        return "dlog" if float(np.nanmin(np.asarray(level, float))) > 0 else "diff"
    if r == "dlog" and float(np.nanmin(np.asarray(level, float))) <= 0:
        raise ValueError("target.transform is dlog but the target has values <= 0: use diff")
    return r


def growth(level):
    """Quarterly change of the target in the model's form (pandas Series in, Series out)."""
    return np.log(level).diff() if is_log() else level.diff()


def change(new, old):
    """One quarter's change from old to new (used to remove inorganic events)."""
    return np.log(new / old) if is_log() else new - old


def state(level):
    """The level in the space where changes add up: ln(L) for dlog, L for diff."""
    return np.log(level) if is_log() else level


def from_state(s):
    """Inverse of state()."""
    return np.exp(s) if is_log() else s


def path(L0, changes):
    """Level after each quarter, starting from L0 and adding up the changes."""
    c = np.cumsum(np.asarray(changes, float))
    return L0 * np.exp(c) if is_log() else L0 + c


def level_after(L0, total_change):
    """Level after a total change (scalar or array)."""
    return L0 * np.exp(total_change) if is_log() else L0 + total_change


def error(cum_forecast, cum_actual):
    """Level error from cumulated changes: a fraction for dlog, units / 100 for diff (see the module note)."""
    d = np.asarray(cum_forecast, float) - np.asarray(cum_actual, float)
    return np.exp(d) - 1 if is_log() else d / 100.0


def pct(g):
    """A quarter's change for display: % for dlog, units for diff."""
    return (np.exp(g) - 1) * 100 if is_log() else g


def annualised(g):
    """Four quarters of the same change: % a year for dlog, units a year for diff."""
    return (np.exp(4 * g) - 1) * 100 if is_log() else 4 * g


def growth_display(g):
    """A change in the model's form as shown in tables: log points x 100 for dlog, units for diff."""
    return np.asarray(g, float) * 100 if is_log() else np.asarray(g, float)


def units():
    """What the *_pct error columns and growth columns hold."""
    return "%" if is_log() else "target units"


def describe():
    """One line for logs and reports."""
    return ("log growth: ln(L_t) - ln(L_t-1); errors in % of the level" if is_log() else
            "change: L_t - L_t-1; errors in the target's units (e.g. percentage points)")


def from_display(v):
    """Inverse of growth_display()."""
    return np.asarray(v, float) / 100 if is_log() else np.asarray(v, float)


def per_year(mean_change):
    """Average quarterly change -> per year: percentage points of growth for dlog, units for diff."""
    return mean_change * 400 if is_log() else mean_change * 4


def target_deflator(mdf, deflator, index):
    """ln(price index) used to put the target in real terms in long-run relations (ECM, levels ARDL).

    A diff target (a rate) is not deflated: the column is 0, so the long run is in the target's own units.
    """
    import pandas as pd
    if not is_log() or not deflator or deflator not in mdf.columns:
        return pd.Series(0.0, index=index)
    return np.log(mdf[deflator]).reindex(index)
