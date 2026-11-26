"""
Dynamic stability of a fitted model's own-lag part.

A model with lags of its own dependent variable is stable when every root of its autoregressive polynomial lies
inside the unit circle; with one lag, when |a1| < 1. For a model of the LEVEL (levels ARDL), a sum of the lag
coefficients of 1 or more means there is no error correction: a gap from the long-run relation never closes (or
grows), the long-run coefficients b / (1 - sum a) are meaningless and the forecast drifts or explodes over long
horizons, even if it looks fine over a short backtest. For an error-correction model, the same condition is an
adjustment coefficient that is not negative.

    s = check(model)        # {"ar_sum", "max_root", "stable", "reason"} or None for models without own lags
"""
import numpy as np


def _coefs(model):
    """The own-lag coefficients a_1..a_p of a fitted model, and what they are lags of ('level' or 'growth')."""
    p = getattr(model, "params", None)
    if p is None or not hasattr(p, "get"):
        return None, None
    if getattr(model, "kind", "") == "LevelsARDL":
        a = [float(p.get(f"ys_lag{i}", 0.0)) for i in range(1, int(getattr(model, "p", 0)) + 1)]
        return a, "level"
    a = [float(p[k]) for k in sorted((k for k in p.index if str(k).startswith("y_lag")),
                                     key=lambda k: int(str(k)[5:]) if str(k)[5:].isdigit() else 0)]
    return (a, "growth") if a else (None, None)


def check(model, tol=1e-6):
    """Stability of the model's own-lag dynamics.

    Args:
        model: a fitted model. tol: numerical tolerance.

    Returns:
        dict ar_sum, max_root (largest modulus of the AR polynomial's roots), stable (bool), reason (text when
        unstable, else ""), or None when the model has no own lags.
    """
    out = None
    a, of = _coefs(model)
    if a:
        roots = np.roots([1.0] + [-x for x in a])          # z^p - a1 z^(p-1) - ... - ap
        mr = float(np.max(np.abs(roots))) if len(roots) else 0.0
        s = float(np.sum(a))
        ok = mr < 1 - tol
        why = ""
        if not ok:
            why = (f"no error correction: the lagged {of} coefficient{'s sum' if len(a) > 1 else ''} "
                   f"{'to ' if len(a) > 1 else 'is '}{s:.3f} (largest root {mr:.3f} >= 1), so the forecast drifts or "
                   "explodes over long horizons and the long-run coefficients are meaningless" if of == "level" else
                   f"the lagged-growth dynamics are not stable (largest root {mr:.3f} >= 1)")
        out = {"ar_sum": s, "max_root": mr, "stable": ok, "reason": why}
    p = getattr(model, "params", None)
    if p is not None and hasattr(p, "get") and "ect_lag1" in getattr(p, "index", []):    # error-correction model
        sp = float(p.get("ect_lag1"))
        if sp >= -tol:
            r = f"no error correction: the adjustment coefficient is {sp:+.3f} (it must be negative)"
            out = {**(out or {"ar_sum": np.nan, "max_root": np.nan}), "stable": False,
                   "reason": "; ".join(x for x in [(out or {}).get("reason", ""), r] if x)}
    return out
